//! Checkpoint snapshots in data storage.
//!
//! A snapshot is the serialized collection as of an LSN: every entry before
//! that LSN is in it, so recovery loads it and replays the WAL from there.
//!
//! Snapshots are split into chunks of [`SNAPSHOT_CHUNK_BYTES`] at pages from
//! `PAGE_POINTS_START`, above the WAL's page range, in one of two slots. A
//! new snapshot is written into the slot the manifest does not name, then the
//! manifest (one metadata key) is switched to it. A failure anywhere before
//! that switch leaves the previous snapshot whole and still current.

use super::Lsn;
use crate::engine::collection::PAGE_POINTS_START;
use crate::error::{LatticeError, LatticeResult};
use crate::storage::{LatticeStorage, StorageError};

/// Metadata key holding the current snapshot's manifest
pub const SNAPSHOT_MANIFEST_KEY: &str = "checkpoint_manifest";

/// Snapshot chunk size, kept well under per-value limits of key-value stores
pub const SNAPSHOT_CHUNK_BYTES: usize = 64 * 1024;

/// Page-id distance between the two snapshot slots
const SLOT_SPAN: u64 = 1 << 32;

const MANIFEST_VERSION: u64 = 1;
const MANIFEST_BYTES: usize = 7 * 8;

/// A loaded snapshot
pub struct Snapshot {
    /// Every WAL entry before this LSN is reflected in `bytes`
    pub lsn: Lsn,
    /// The serialized collection
    pub bytes: Vec<u8>,
}

struct Manifest {
    slot: u64,
    lsn: Lsn,
    len: u64,
    chunks: u64,
    checksum: u64,
    /// Chunks left in the other slot by the snapshot before this one
    other_slot_chunks: u64,
}

impl Manifest {
    fn encode(&self) -> Vec<u8> {
        let fields = [
            MANIFEST_VERSION,
            self.slot,
            self.lsn,
            self.len,
            self.chunks,
            self.checksum,
            self.other_slot_chunks,
        ];
        fields.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    fn decode(bytes: &[u8]) -> LatticeResult<Self> {
        if bytes.len() != MANIFEST_BYTES {
            return Err(corrupt(format!("manifest is {} bytes", bytes.len())));
        }
        let field = |i: usize| {
            let mut word = [0u8; 8];
            word.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
            u64::from_le_bytes(word)
        };
        if field(0) != MANIFEST_VERSION || field(1) > 1 {
            return Err(corrupt(format!(
                "manifest version {} slot {} not recognised",
                field(0),
                field(1)
            )));
        }
        Ok(Self {
            slot: field(1),
            lsn: field(2),
            len: field(3),
            chunks: field(4),
            checksum: field(5),
            other_slot_chunks: field(6),
        })
    }
}

fn corrupt(context: String) -> LatticeError {
    LatticeError::DataCorruption {
        code: 60020,
        context: format!("checkpoint snapshot: {}", context),
    }
}

fn chunk_page(slot: u64, index: u64) -> u64 {
    PAGE_POINTS_START + slot * SLOT_SPAN + index
}

fn checksum(bytes: &[u8]) -> u64 {
    u64::from(twox_hash::xxhash32::Hasher::oneshot(0, bytes))
}

async fn read_manifest<S: LatticeStorage>(storage: &S) -> LatticeResult<Option<Manifest>> {
    match storage.get_meta(SNAPSHOT_MANIFEST_KEY).await? {
        Some(bytes) => Manifest::decode(&bytes).map(Some),
        None => Ok(None),
    }
}

/// Durably store `bytes` as the snapshot as of `lsn`
///
/// Returns once the manifest naming it is synced. On error the previous
/// snapshot, if any, is still the current one.
pub async fn write_snapshot<S: LatticeStorage>(
    storage: &S,
    lsn: Lsn,
    bytes: &[u8],
) -> LatticeResult<()> {
    let current = read_manifest(storage).await?;
    let (slot, stale_chunks, other_slot_chunks) = match &current {
        Some(m) => (1 - m.slot, m.other_slot_chunks, m.chunks),
        None => (0, 0, 0),
    };
    let chunks = bytes.chunks(SNAPSHOT_CHUNK_BYTES).count() as u64;

    // The target slot is not referenced; clear what an older, larger
    // snapshot left beyond the chunks about to be written.
    for index in chunks..stale_chunks {
        match storage.delete_page(chunk_page(slot, index)).await {
            Ok(()) | Err(StorageError::PageNotFound { .. }) => {}
            Err(e) => return Err(e.into()),
        }
    }
    for (index, chunk) in bytes.chunks(SNAPSHOT_CHUNK_BYTES).enumerate() {
        storage
            .write_page(chunk_page(slot, index as u64), chunk)
            .await?;
    }
    storage.sync().await?;

    let manifest = Manifest {
        slot,
        lsn,
        len: bytes.len() as u64,
        chunks,
        checksum: checksum(bytes),
        other_slot_chunks,
    };
    storage
        .set_meta(SNAPSHOT_MANIFEST_KEY, &manifest.encode())
        .await?;
    storage.sync().await?;
    Ok(())
}

/// Load the current snapshot, or `None` if no checkpoint was ever taken
///
/// A manifest whose chunks are missing, short or fail the checksum is an
/// error, never an empty collection.
pub async fn read_snapshot<S: LatticeStorage>(storage: &S) -> LatticeResult<Option<Snapshot>> {
    let Some(manifest) = read_manifest(storage).await? else {
        return Ok(None);
    };

    let mut bytes = Vec::with_capacity(manifest.len as usize);
    for index in 0..manifest.chunks {
        let page = chunk_page(manifest.slot, index);
        match storage.read_page(page).await {
            Ok(chunk) => bytes.extend_from_slice(&chunk),
            Err(StorageError::PageNotFound { .. }) => {
                return Err(corrupt(format!("chunk page {} is missing", page)))
            }
            Err(e) => return Err(e.into()),
        }
    }

    if bytes.len() as u64 != manifest.len || checksum(&bytes) != manifest.checksum {
        return Err(corrupt(format!(
            "{} bytes read, manifest records {} with checksum {:#x}",
            bytes.len(),
            manifest.len,
            manifest.checksum
        )));
    }
    Ok(Some(Snapshot {
        lsn: manifest.lsn,
        bytes,
    }))
}
