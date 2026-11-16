//! WAL record framing: `[len:u32][checksum:u32][entry:rkyv]`.

use super::{Lsn, WalEntry};
use crate::error::{LatticeError, LatticeResult};

/// Frame one entry as a WAL record and append it to `out`
pub(super) fn encode_record(entry: &WalEntry, out: &mut Vec<u8>) -> LatticeResult<()> {
    // Serialize entry with rkyv
    let entry_bytes =
        rkyv::to_bytes::<rkyv::rancor::Error>(entry).map_err(|e| LatticeError::Serialization {
            message: format!("Failed to serialize WAL entry: {}", e),
        })?;

    // Calculate xxHash32 checksum (faster than CRC32)
    let checksum = twox_hash::xxhash32::Hasher::oneshot(0, &entry_bytes);

    // Build WAL record: [len:u32][checksum:u32][entry]
    let len = entry_bytes.len() as u32;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&checksum.to_le_bytes());
    out.extend_from_slice(&entry_bytes);
    Ok(())
}

/// Count entries in a byte buffer by scanning WAL record headers
pub(super) fn count_entries_in_bytes(data: &[u8]) -> u64 {
    let mut count = 0;
    let mut offset = 0;
    while offset + 8 <= data.len() {
        let len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        let record_size = 8 + len;
        if offset + record_size > data.len() {
            break;
        }
        offset += record_size;
        count += 1;
    }
    count
}

/// Parse entries from a single page
pub(super) fn parse_page_entries(
    page: &[u8],
    start_lsn: Lsn,
    current_lsn: &mut Lsn,
) -> LatticeResult<Vec<(Lsn, WalEntry)>> {
    let mut entries = Vec::new();
    let mut offset = 0;

    while offset + 8 <= page.len() {
        // Read length
        let len = u32::from_le_bytes([
            page[offset],
            page[offset + 1],
            page[offset + 2],
            page[offset + 3],
        ]) as usize;
        offset += 4;

        // Read checksum
        let expected_checksum = u32::from_le_bytes([
            page[offset],
            page[offset + 1],
            page[offset + 2],
            page[offset + 3],
        ]);
        offset += 4;

        // Validate we have enough data
        if offset + len > page.len() {
            // Truncated entry - stop reading this page
            break;
        }

        // Read and validate entry
        let entry_bytes = &page[offset..offset + len];
        let actual_checksum = twox_hash::xxhash32::Hasher::oneshot(0, entry_bytes);

        if actual_checksum != expected_checksum {
            #[cfg(not(target_arch = "wasm32"))]
            tracing::error!(
                lsn = *current_lsn,
                expected = expected_checksum,
                actual = actual_checksum,
                "Corrupted WAL entry, skipping"
            );
            offset += len;
            *current_lsn += 1;
            continue;
        }

        // Deserialize entry
        let entry = match rkyv::from_bytes::<WalEntry, rkyv::rancor::Error>(entry_bytes) {
            Ok(entry) => entry,
            Err(_e) => {
                #[cfg(not(target_arch = "wasm32"))]
                tracing::error!(
                    lsn = *current_lsn,
                    error = %_e,
                    "Failed to deserialize WAL entry, skipping"
                );
                offset += len;
                *current_lsn += 1;
                continue;
            }
        };

        if *current_lsn >= start_lsn {
            entries.push((*current_lsn, entry));
        }

        offset += len;
        *current_lsn += 1;
    }

    Ok(entries)
}
