//! WAL header: the durable record of where the log stands.

use super::{Lsn, WAL_PAGE_START};
use crate::error::{LatticeError, LatticeResult};

/// WAL header stored in metadata
///
/// Tracks current LSN, checkpoint information, and page boundaries.
#[derive(Debug, Clone)]
pub struct WalHeader {
    /// Next LSN to assign
    pub next_lsn: Lsn,
    /// LSN of last checkpoint (WAL before this can be truncated)
    pub last_checkpoint_lsn: Lsn,
    /// Number of entries since last checkpoint
    pub entries_since_checkpoint: u64,
    /// Current active page for appends
    pub current_page_id: u64,
    /// Total bytes written to current page
    pub current_page_bytes: u64,
    /// First LSN on each page: (page_id, first_lsn) for truncation
    pub page_first_lsn: Vec<(u64, Lsn)>,
}

impl Default for WalHeader {
    fn default() -> Self {
        Self {
            next_lsn: 0,
            last_checkpoint_lsn: 0,
            entries_since_checkpoint: 0,
            current_page_id: WAL_PAGE_START,
            current_page_bytes: 0,
            page_first_lsn: vec![(WAL_PAGE_START, 0)],
        }
    }
}

/// Serialize WAL header as raw little-endian bytes (fast, no rkyv overhead)
pub(super) fn serialize_header(header: &WalHeader) -> Vec<u8> {
    let vec_len = header.page_first_lsn.len();
    let total = 5 * 8 + 8 + vec_len * 16;
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(&header.next_lsn.to_le_bytes());
    buf.extend_from_slice(&header.last_checkpoint_lsn.to_le_bytes());
    buf.extend_from_slice(&header.entries_since_checkpoint.to_le_bytes());
    buf.extend_from_slice(&header.current_page_id.to_le_bytes());
    buf.extend_from_slice(&header.current_page_bytes.to_le_bytes());
    buf.extend_from_slice(&(vec_len as u64).to_le_bytes());
    for (page_id, lsn) in &header.page_first_lsn {
        buf.extend_from_slice(&page_id.to_le_bytes());
        buf.extend_from_slice(&lsn.to_le_bytes());
    }
    buf
}

/// Deserialize WAL header from raw little-endian bytes
pub(super) fn deserialize_header(bytes: &[u8]) -> LatticeResult<WalHeader> {
    const FIXED_SIZE: usize = 6 * 8; // 5 fields + vec_len
    if bytes.len() < FIXED_SIZE {
        return Err(LatticeError::Serialization {
            message: format!(
                "WAL header too short: {} bytes, need at least {}",
                bytes.len(),
                FIXED_SIZE
            ),
        });
    }

    let read_u64 = |offset: usize| -> u64 {
        u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
    };

    let next_lsn = read_u64(0);
    let last_checkpoint_lsn = read_u64(8);
    let entries_since_checkpoint = read_u64(16);
    let current_page_id = read_u64(24);
    let current_page_bytes = read_u64(32);
    let vec_len = read_u64(40) as usize;

    let expected_size = FIXED_SIZE + vec_len * 16;
    if bytes.len() < expected_size {
        return Err(LatticeError::Serialization {
            message: format!(
                "WAL header truncated: {} bytes, need {} for {} page entries",
                bytes.len(),
                expected_size,
                vec_len
            ),
        });
    }

    let mut page_first_lsn = Vec::with_capacity(vec_len);
    for i in 0..vec_len {
        let base = FIXED_SIZE + i * 16;
        page_first_lsn.push((read_u64(base), read_u64(base + 8)));
    }

    Ok(WalHeader {
        next_lsn,
        last_checkpoint_lsn,
        entries_since_checkpoint,
        current_page_id,
        current_page_bytes,
        page_first_lsn,
    })
}
