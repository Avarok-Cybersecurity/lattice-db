//! Write-Ahead Log (WAL) for ACID durability
//!
//! The WAL provides crash recovery by logging all mutations before applying them.
//! On recovery, the WAL is replayed to restore the in-memory state.
//!
//! # Format
//!
//! Each WAL entry is stored as:
//! ```text
//! [len:u32][checksum:u32][entry:rkyv]
//! ```
//!
//! - `len`: Entry size in bytes (excluding this field)
//! - `checksum`: xxHash32 of the serialized entry
//! - `entry`: rkyv-serialized WalEntry
//!
//! # SBIO Compliance
//!
//! The WAL uses the `LatticeStorage` trait for I/O operations,
//! enabling platform-independent durability (disk, OPFS, etc).

#[cfg(not(target_arch = "wasm32"))]
mod group_commit;

#[cfg(not(target_arch = "wasm32"))]
mod checkpoint_manager;

mod compaction;
mod entry;
mod header;
mod read;
mod record;
pub mod relations;
pub mod replay;
pub mod snapshot;

#[cfg(test)]
mod tests;

#[cfg(not(target_arch = "wasm32"))]
pub use group_commit::GroupCommitHandle;

#[cfg(not(target_arch = "wasm32"))]
pub use checkpoint_manager::{
    CheckpointCallback, CheckpointConfig, CheckpointManager, CHECKPOINT_PAGE_START,
};

pub use entry::*;
pub use header::WalHeader;

use crate::error::LatticeResult;
use crate::storage::LatticeStorage;
use read::{committed_prefix, read_page_or_empty};

/// Write-Ahead Log
///
/// Provides durable logging of mutations for crash recovery.
///
/// A `sync` is all-or-nothing: if any storage call in it fails, every entry
/// appended since the last successful sync is discarded, so a failed commit
/// can never be persisted later by the next one.
pub struct WriteAheadLog<S: LatticeStorage> {
    storage: S,
    header: WalHeader,
    /// Header as last persisted; reads are bounded by it and a failed sync
    /// restores it
    synced_header: WalHeader,
    /// Buffer for batching writes before sync
    write_buffer: Vec<u8>,
    /// Cached current page content to avoid reading on every sync
    page_cache: Vec<u8>,
}

/// WAL page start ID (reserved range)
const WAL_PAGE_START: u64 = 0;

/// Metadata key holding the WAL header: a commit is durable once this one
/// key is written, so storage must write a single key all-or-nothing
pub const WAL_HEADER_KEY: &str = "wal_header";

/// Maximum entries before automatic checkpoint suggestion
const CHECKPOINT_THRESHOLD: u64 = 10000;

/// Maximum WAL page size before rotation (256KB)
///
/// Smaller pages reduce copy overhead on sync but increase page rotation frequency.
/// 256KB balances copy overhead vs rotation overhead for typical workloads.
///
/// A page rotates before a commit would take it past this size, so pages stay
/// within it, except that a single commit larger than this is written as one
/// page of its own size. Storage must accept pages of at least this size.
pub const WAL_PAGE_MAX_BYTES: u64 = 256 * 1024;

impl<S: LatticeStorage> WriteAheadLog<S> {
    /// Create or open a WAL with the given storage backend
    pub async fn open(storage: S) -> LatticeResult<Self> {
        // Load header from metadata if exists
        let header = match storage.get_meta(WAL_HEADER_KEY).await? {
            Some(bytes) => header::deserialize_header(&bytes)?,
            None => WalHeader::default(),
        };

        // Load current page into cache
        let page = read_page_or_empty(&storage, header.current_page_id).await?;
        let page_cache = committed_prefix(page, header.current_page_id, header.current_page_bytes)?;

        Ok(Self {
            storage,
            synced_header: header.clone(),
            header,
            write_buffer: Vec::with_capacity(4096),
            page_cache,
        })
    }

    /// Append an entry to the WAL
    ///
    /// Returns the LSN assigned to this entry.
    pub async fn append(&mut self, entry: &WalEntry) -> LatticeResult<Lsn> {
        let lsn = self.header.next_lsn;
        record::encode_record(entry, &mut self.write_buffer)?;

        // Update header
        self.header.next_lsn += 1;
        self.header.entries_since_checkpoint += 1;

        Ok(lsn)
    }

    /// Sync WAL to durable storage
    ///
    /// This must be called after append() to guarantee durability.
    /// Batching multiple appends before sync() improves throughput.
    /// Pages are rotated when they exceed WAL_PAGE_MAX_BYTES (256KB).
    ///
    /// On error, the entries appended since the last successful sync are
    /// discarded and the log is back at its last durable state.
    pub async fn sync(&mut self) -> LatticeResult<()> {
        if self.write_buffer.is_empty() {
            return Ok(());
        }

        let cache_len = self.page_cache.len();
        let rotated_from = self.stage_buffer();

        match self.persist_current_page().await {
            Ok(()) => {
                self.write_buffer.clear();
                self.synced_header = self.header.clone();
                Ok(())
            }
            Err(e) => {
                match rotated_from {
                    Some(previous_page) => self.page_cache = previous_page,
                    None => self.page_cache.truncate(cache_len),
                }
                self.discard_unsynced();
                Err(e)
            }
        }
    }

    /// Drop every entry appended since the last successful sync
    pub(crate) fn discard_unsynced(&mut self) {
        self.header = self.synced_header.clone();
        self.write_buffer.clear();
    }

    /// Move the write buffer onto the cached page, rotating first if it would
    /// overflow. Returns the previous page's cache when it rotated.
    fn stage_buffer(&mut self) -> Option<Vec<u8>> {
        let new_bytes = self.write_buffer.len() as u64;
        let mut rotated_from = None;

        // Check if rotation needed before writing
        if self.header.current_page_bytes + new_bytes > WAL_PAGE_MAX_BYTES {
            // Count entries in buffer to compute first LSN on new page
            let buffer_entries = record::count_entries_in_bytes(&self.write_buffer);

            // Rotate to new page
            self.header.current_page_id += 1;
            self.header.current_page_bytes = 0;
            rotated_from = Some(std::mem::take(&mut self.page_cache));

            // Track first LSN on new page
            let first_lsn_on_new_page = self.header.next_lsn - buffer_entries;
            self.header
                .page_first_lsn
                .push((self.header.current_page_id, first_lsn_on_new_page));

            #[cfg(not(target_arch = "wasm32"))]
            tracing::info!(
                page_id = self.header.current_page_id,
                first_lsn = first_lsn_on_new_page,
                "WAL page rotation"
            );
        }

        // Append new data to cached page
        self.page_cache.extend_from_slice(&self.write_buffer);
        self.header.current_page_bytes += new_bytes;
        rotated_from
    }

    async fn persist_current_page(&self) -> LatticeResult<()> {
        // Crash safety argument for the write ordering below:
        //
        // 1. write_page(): new page data lands on storage (may not be durable yet)
        // 2. set_meta(): header updated with new page_id/offset/LSN
        // 3. sync(): both page data and header flushed to durable storage
        //
        // If crash occurs between (1) and (2): header still points to the previous
        // valid state. The bytes past the header's committed length are ignored on
        // open and read, and overwritten by the next sync, so no data loss.
        //
        // If crash occurs between (2) and (3): neither page nor header is durable.
        // Recovery sees the last successfully synced state. No corruption possible
        // because the storage layer treats un-synced writes as uncommitted.
        self.storage
            .write_page(self.header.current_page_id, &self.page_cache)
            .await?;

        let header_bytes = header::serialize_header(&self.header);
        self.storage.set_meta(WAL_HEADER_KEY, &header_bytes).await?;

        self.storage.sync().await?;
        Ok(())
    }

    /// Get the next LSN that will be assigned
    pub fn next_lsn(&self) -> Lsn {
        self.header.next_lsn
    }

    /// Get the last checkpoint LSN
    pub fn last_checkpoint_lsn(&self) -> Lsn {
        self.header.last_checkpoint_lsn
    }

    /// Check if a checkpoint is recommended
    ///
    /// Returns true if entries since last checkpoint exceeds threshold.
    pub fn should_checkpoint(&self) -> bool {
        self.header.entries_since_checkpoint >= CHECKPOINT_THRESHOLD
    }
}

// Note: StorageError -> LatticeError conversion is handled by #[from] in error.rs
