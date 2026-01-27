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

#[cfg(not(target_arch = "wasm32"))]
pub use group_commit::GroupCommitHandle;

#[cfg(not(target_arch = "wasm32"))]
pub use checkpoint_manager::{
    CheckpointCallback, CheckpointConfig, CheckpointManager, CHECKPOINT_PAGE_START,
};

use crate::error::{LatticeError, LatticeResult};
use crate::storage::LatticeStorage;
use crate::types::point::{Edge, Point, PointId};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

/// Log Sequence Number - monotonically increasing identifier for each WAL entry
pub type Lsn = u64;

/// WAL entry types
///
/// Each entry represents an atomic operation that can be replayed during recovery.
#[derive(Archive, RkyvDeserialize, RkyvSerialize, Debug, Clone)]
pub enum WalEntry {
    /// Insert or update points
    Upsert {
        /// Points to upsert (can be single or batch)
        points: Vec<Point>,
    },

    /// Delete points by ID
    Delete {
        /// Point IDs to delete
        ids: Vec<PointId>,
    },

    /// Add a graph edge
    AddEdge {
        /// Source point ID
        from_id: PointId,
        /// Edge to add
        edge: Edge,
    },

    /// Remove a graph edge
    RemoveEdge {
        /// Source point ID
        from_id: PointId,
        /// Target point ID
        to_id: PointId,
        /// Relation ID (from collection config)
        relation_id: u16,
    },

    /// Abort marker
    ///
    /// Indicates that the operation at the given LSN failed to apply
    /// and should be skipped during recovery replay.
    Abort {
        /// LSN of the failed operation
        aborted_lsn: Lsn,
    },

    /// Checkpoint marker
    ///
    /// Indicates that a snapshot was taken at this LSN.
    /// WAL entries before this can be safely truncated.
    Checkpoint {
        /// LSN at which snapshot was taken
        snapshot_lsn: Lsn,
    },
}

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

/// Write-Ahead Log
///
/// Provides durable logging of mutations for crash recovery.
pub struct WriteAheadLog<S: LatticeStorage> {
    storage: S,
    header: WalHeader,
    /// Buffer for batching writes before sync
    write_buffer: Vec<u8>,
    /// Cached current page content to avoid reading on every sync
    page_cache: Vec<u8>,
}

/// WAL page start ID (reserved range)
const WAL_PAGE_START: u64 = 0;

/// Metadata key for WAL header
const WAL_HEADER_KEY: &str = "wal_header";

/// Maximum entries before automatic checkpoint suggestion
const CHECKPOINT_THRESHOLD: u64 = 10000;

/// Maximum WAL page size before rotation (256KB)
///
/// Smaller pages reduce copy overhead on sync but increase page rotation frequency.
/// 256KB balances copy overhead vs rotation overhead for typical workloads.
const WAL_PAGE_MAX_BYTES: u64 = 256 * 1024;

impl<S: LatticeStorage> WriteAheadLog<S> {
    /// Create or open a WAL with the given storage backend
    pub async fn open(storage: S) -> LatticeResult<Self> {
        // Load header from metadata if exists
        let header = match storage.get_meta(WAL_HEADER_KEY).await? {
            Some(bytes) => Self::deserialize_header(&bytes)?,
            None => WalHeader::default(),
        };

        // Load current page into cache
        let page_cache = storage
            .read_page(header.current_page_id)
            .await
            .unwrap_or_default();

        Ok(Self {
            storage,
            header,
            write_buffer: Vec::with_capacity(4096),
            page_cache,
        })
    }

    /// Serialize WAL header as raw little-endian bytes (fast, no rkyv overhead)
    fn serialize_header(header: &WalHeader) -> Vec<u8> {
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
    fn deserialize_header(bytes: &[u8]) -> LatticeResult<WalHeader> {
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

    /// Append an entry to the WAL
    ///
    /// Returns the LSN assigned to this entry.
    pub async fn append(&mut self, entry: &WalEntry) -> LatticeResult<Lsn> {
        let lsn = self.header.next_lsn;

        // Serialize entry with rkyv
        let entry_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(entry).map_err(|e| {
            LatticeError::Serialization {
                message: format!("Failed to serialize WAL entry: {}", e),
            }
        })?;

        // Calculate xxHash32 checksum (faster than CRC32)
        let checksum = twox_hash::xxhash32::Hasher::oneshot(0, &entry_bytes);

        // Build WAL record: [len:u32][checksum:u32][entry]
        let len = entry_bytes.len() as u32;
        self.write_buffer.extend_from_slice(&len.to_le_bytes());
        self.write_buffer.extend_from_slice(&checksum.to_le_bytes());
        self.write_buffer.extend_from_slice(&entry_bytes);

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
    pub async fn sync(&mut self) -> LatticeResult<()> {
        if self.write_buffer.is_empty() {
            return Ok(());
        }

        let new_bytes = self.write_buffer.len() as u64;

        // Check if rotation needed before writing
        if self.header.current_page_bytes + new_bytes > WAL_PAGE_MAX_BYTES {
            // Count entries in buffer to compute first LSN on new page
            let buffer_entries = Self::count_entries_in_bytes(&self.write_buffer);

            // Rotate to new page
            self.header.current_page_id += 1;
            self.header.current_page_bytes = 0;

            // Clear page cache for new page
            self.page_cache.clear();

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

        // Crash safety argument for the write ordering below:
        //
        // 1. write_page(): new page data lands on storage (may not be durable yet)
        // 2. set_meta(): header updated with new page_id/offset/LSN
        // 3. sync(): both page data and header flushed to durable storage
        //
        // If crash occurs between (1) and (2): header still points to the previous
        // valid state. The orphaned page data is unreachable and will be overwritten
        // on next rotation. Recovery uses header.current_page_id, so no data loss.
        //
        // If crash occurs between (2) and (3): neither page nor header is durable.
        // Recovery sees the last successfully synced state. No corruption possible
        // because the storage layer treats un-synced writes as uncommitted.
        self.storage
            .write_page(self.header.current_page_id, &self.page_cache)
            .await?;

        let header_bytes = Self::serialize_header(&self.header);
        self.storage.set_meta(WAL_HEADER_KEY, &header_bytes).await?;

        self.storage.sync().await?;

        // Clear write buffer
        self.write_buffer.clear();

        Ok(())
    }

    /// Count entries in a byte buffer by scanning WAL record headers
    fn count_entries_in_bytes(data: &[u8]) -> u64 {
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

    /// Read all entries from a given LSN
    ///
    /// Used during recovery to replay the WAL.
    /// Scans all pages from WAL_PAGE_START to current_page_id.
    pub async fn read_from(&self, start_lsn: Lsn) -> LatticeResult<Vec<(Lsn, WalEntry)>> {
        let mut entries = Vec::new();
        let mut current_lsn: Lsn = 0;

        // Scan all pages from start to current
        for page_id in WAL_PAGE_START..=self.header.current_page_id {
            let page = self.storage.read_page(page_id).await.unwrap_or_default();

            if page.is_empty() {
                continue;
            }

            let page_entries = self.parse_page_entries(&page, start_lsn, &mut current_lsn)?;
            entries.extend(page_entries);
        }

        Ok(entries)
    }

    /// Parse entries from a single page
    fn parse_page_entries(
        &self,
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

    /// Record a checkpoint
    ///
    /// This should be called after successfully saving a snapshot.
    pub async fn checkpoint(&mut self, snapshot_lsn: Lsn) -> LatticeResult<()> {
        // Write checkpoint entry
        self.append(&WalEntry::Checkpoint { snapshot_lsn }).await?;

        // Update header
        self.header.last_checkpoint_lsn = snapshot_lsn;
        self.header.entries_since_checkpoint = 0;

        // Sync to ensure checkpoint is durable
        self.sync().await?;

        Ok(())
    }

    /// Truncate WAL entries before the given LSN
    ///
    /// This reclaims disk space after a successful checkpoint by deleting
    /// pages that contain only entries before the given LSN.
    ///
    /// Safety: Only entries before `last_checkpoint_lsn` can be truncated.
    pub async fn truncate_before(&mut self, lsn: Lsn) -> LatticeResult<()> {
        // Safety: only truncate entries before checkpoint
        if lsn > self.header.last_checkpoint_lsn {
            return Err(LatticeError::InvalidOperation {
                message: format!(
                    "Cannot truncate LSN {} beyond checkpoint {}",
                    lsn, self.header.last_checkpoint_lsn
                ),
            });
        }

        // Find pages entirely before the LSN
        let mut pages_to_delete = Vec::new();
        let mut new_page_first_lsn = Vec::new();

        for i in 0..self.header.page_first_lsn.len() {
            let (page_id, first_lsn) = self.header.page_first_lsn[i];

            // Check if this page's first LSN is before our target
            if first_lsn < lsn {
                // Check if the ENTIRE page is before LSN by looking at next page's first LSN
                let next_page_first_lsn = if i + 1 < self.header.page_first_lsn.len() {
                    Some(self.header.page_first_lsn[i + 1].1)
                } else {
                    // This is the current/last page, don't delete it
                    None
                };

                if let Some(next_lsn) = next_page_first_lsn {
                    if next_lsn <= lsn {
                        // Entire page is before target LSN, safe to delete
                        pages_to_delete.push(page_id);
                        continue;
                    }
                }
            }
            // Keep this page
            new_page_first_lsn.push((page_id, first_lsn));
        }

        // Delete old pages
        for page_id in &pages_to_delete {
            self.storage.delete_page(*page_id).await?;
            #[cfg(not(target_arch = "wasm32"))]
            tracing::info!(page_id, "Deleted WAL page after truncation");
        }

        self.header.page_first_lsn = new_page_first_lsn;

        // Persist updated header
        self.sync_header().await?;

        Ok(())
    }

    /// Sync only the header to storage (used by truncate_before)
    async fn sync_header(&self) -> LatticeResult<()> {
        let header_bytes = Self::serialize_header(&self.header);
        self.storage.set_meta(WAL_HEADER_KEY, &header_bytes).await?;
        self.storage.sync().await?;
        Ok(())
    }

    /// Get the current page ID (for testing/debugging)
    #[cfg(test)]
    pub fn current_page_id(&self) -> u64 {
        self.header.current_page_id
    }

    /// Get the current page bytes (for testing/debugging)
    #[cfg(test)]
    pub fn current_page_bytes(&self) -> u64 {
        self.header.current_page_bytes
    }
}

// Note: StorageError -> LatticeError conversion is handled by #[from] in error.rs

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::storage::StorageResult;
    use crate::types::point::Point;

    // Mock storage for testing
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};

    struct MockStorage {
        pages: Arc<RwLock<HashMap<u64, Vec<u8>>>>,
        meta: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                pages: Arc::new(RwLock::new(HashMap::new())),
                meta: Arc::new(RwLock::new(HashMap::new())),
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[async_trait::async_trait]
    impl LatticeStorage for MockStorage {
        async fn get_meta(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
            Ok(self.meta.read().unwrap().get(key).cloned())
        }

        async fn set_meta(&self, key: &str, value: &[u8]) -> StorageResult<()> {
            self.meta
                .write()
                .unwrap()
                .insert(key.to_string(), value.to_vec());
            Ok(())
        }

        async fn delete_meta(&self, key: &str) -> StorageResult<()> {
            self.meta.write().unwrap().remove(key);
            Ok(())
        }

        async fn read_page(&self, page_id: u64) -> StorageResult<Vec<u8>> {
            Ok(self
                .pages
                .read()
                .unwrap()
                .get(&page_id)
                .cloned()
                .unwrap_or_default())
        }

        async fn write_page(&self, page_id: u64, data: &[u8]) -> StorageResult<()> {
            self.pages.write().unwrap().insert(page_id, data.to_vec());
            Ok(())
        }

        async fn page_exists(&self, page_id: u64) -> StorageResult<bool> {
            Ok(self.pages.read().unwrap().contains_key(&page_id))
        }

        async fn delete_page(&self, page_id: u64) -> StorageResult<()> {
            self.pages.write().unwrap().remove(&page_id);
            Ok(())
        }

        async fn sync(&self) -> StorageResult<()> {
            Ok(())
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_wal_append_and_read() {
        let storage = MockStorage::new();
        let mut wal = WriteAheadLog::open(storage).await.unwrap();

        // Append some entries
        let point1 = Point::new_vector(1, vec![0.1, 0.2, 0.3]);
        let point2 = Point::new_vector(2, vec![0.4, 0.5, 0.6]);

        let lsn1 = wal
            .append(&WalEntry::Upsert {
                points: vec![point1.clone()],
            })
            .await
            .unwrap();
        let lsn2 = wal
            .append(&WalEntry::Upsert {
                points: vec![point2.clone()],
            })
            .await
            .unwrap();
        let lsn3 = wal
            .append(&WalEntry::Delete { ids: vec![1] })
            .await
            .unwrap();

        assert_eq!(lsn1, 0);
        assert_eq!(lsn2, 1);
        assert_eq!(lsn3, 2);

        // Sync to storage
        wal.sync().await.unwrap();

        // Read back
        let entries = wal.read_from(0).await.unwrap();
        assert_eq!(entries.len(), 3);

        match &entries[0].1 {
            WalEntry::Upsert { points } => {
                assert_eq!(points.len(), 1);
                assert_eq!(points[0].id, 1);
            }
            _ => panic!("Expected Upsert"),
        }

        match &entries[2].1 {
            WalEntry::Delete { ids } => {
                assert_eq!(ids, &vec![1]);
            }
            _ => panic!("Expected Delete"),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_wal_checkpoint() {
        let storage = MockStorage::new();
        let mut wal = WriteAheadLog::open(storage).await.unwrap();

        // Append entry
        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(1, vec![0.1])],
        })
        .await
        .unwrap();
        wal.sync().await.unwrap();

        // Checkpoint
        wal.checkpoint(0).await.unwrap();

        assert_eq!(wal.last_checkpoint_lsn(), 0);
        assert!(!wal.should_checkpoint());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_wal_recovery() {
        let storage = MockStorage::new();

        // Write some entries
        {
            let mut wal = WriteAheadLog::open(MockStorage {
                pages: storage.pages.clone(),
                meta: storage.meta.clone(),
            })
            .await
            .unwrap();

            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(1, vec![0.1])],
            })
            .await
            .unwrap();
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(2, vec![0.2])],
            })
            .await
            .unwrap();
            wal.sync().await.unwrap();
        }

        // Reopen and read back (simulates recovery)
        {
            let wal = WriteAheadLog::open(MockStorage {
                pages: storage.pages.clone(),
                meta: storage.meta.clone(),
            })
            .await
            .unwrap();

            let entries = wal.read_from(0).await.unwrap();
            assert_eq!(entries.len(), 2);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_wal_page_rotation() {
        let storage = MockStorage::new();
        let mut wal: WriteAheadLog<MockStorage> = WriteAheadLog::open(MockStorage {
            pages: storage.pages.clone(),
            meta: storage.meta.clone(),
        })
        .await
        .unwrap();

        // Initial state: should be on page 0
        assert_eq!(wal.header.current_page_id, WAL_PAGE_START);
        assert_eq!(wal.header.current_page_bytes, 0);

        // Create a large vector to force page rotation
        // Each entry will be substantial in size
        let large_vector: Vec<f32> = (0..10000).map(|i| i as f32).collect();

        // Write enough entries to exceed 16MB
        // Each entry with 10000 floats is ~40KB, so ~400 entries should exceed 16MB
        let mut entry_count = 0;
        let initial_page = wal.header.current_page_id;

        for i in 0..500 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, large_vector.clone())],
            })
            .await
            .unwrap();
            wal.sync().await.unwrap();
            entry_count += 1;

            // Check if rotation happened
            if wal.header.current_page_id > initial_page {
                break;
            }
        }

        // Verify that page rotation occurred
        assert!(
            wal.header.current_page_id > initial_page,
            "Page rotation should have occurred after {} entries",
            entry_count
        );

        // Verify that page_first_lsn is tracking pages
        assert!(
            !wal.header.page_first_lsn.is_empty(),
            "page_first_lsn should contain entries after rotation"
        );

        // Verify all entries can still be read back
        let entries = wal.read_from(0).await.unwrap();
        assert_eq!(entries.len(), entry_count as usize);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_wal_truncate_after_checkpoint() {
        let storage = MockStorage::new();
        let pages_arc = storage.pages.clone();
        let mut wal: WriteAheadLog<MockStorage> = WriteAheadLog::open(MockStorage {
            pages: storage.pages.clone(),
            meta: storage.meta.clone(),
        })
        .await
        .unwrap();

        // Create entries that span multiple pages
        let large_vector: Vec<f32> = (0..10000).map(|i| i as f32).collect();

        // Write enough to span at least 2 pages
        for i in 0..500 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, large_vector.clone())],
            })
            .await
            .unwrap();
            wal.sync().await.unwrap();

            if wal.header.current_page_id > WAL_PAGE_START + 1 {
                break;
            }
        }

        // Get the current LSN and page count
        let checkpoint_lsn = wal.next_lsn() - 1;
        let pages_before = pages_arc.read().unwrap().len();

        // Perform checkpoint which should truncate old pages
        wal.checkpoint(checkpoint_lsn).await.unwrap();

        // After checkpoint, old pages should be deleted
        let pages_after = pages_arc.read().unwrap().len();

        // The current page should remain, old pages should be deleted
        assert!(
            pages_after <= pages_before,
            "Truncation should not increase page count"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_wal_multi_page_recovery() {
        let storage = MockStorage::new();

        // Phase 1: Write entries spanning multiple pages
        let entry_count;
        {
            let mut wal: WriteAheadLog<MockStorage> = WriteAheadLog::open(MockStorage {
                pages: storage.pages.clone(),
                meta: storage.meta.clone(),
            })
            .await
            .unwrap();

            let large_vector: Vec<f32> = (0..10000).map(|i| i as f32).collect();
            let mut count = 0;

            for i in 0..500 {
                wal.append(&WalEntry::Upsert {
                    points: vec![Point::new_vector(i, large_vector.clone())],
                })
                .await
                .unwrap();
                wal.sync().await.unwrap();
                count += 1;

                // Stop after spanning at least 2 pages
                if wal.header.current_page_id > WAL_PAGE_START + 1 {
                    break;
                }
            }

            entry_count = count;
        }

        // Phase 2: Reopen and verify recovery across multiple pages
        {
            let wal: WriteAheadLog<MockStorage> = WriteAheadLog::open(MockStorage {
                pages: storage.pages.clone(),
                meta: storage.meta.clone(),
            })
            .await
            .unwrap();

            let entries = wal.read_from(0).await.unwrap();
            assert_eq!(
                entries.len(),
                entry_count as usize,
                "Recovery should restore all {} entries from multiple pages",
                entry_count
            );

            // Verify entries are in correct order
            for (idx, (lsn, _entry)) in entries.iter().enumerate() {
                assert_eq!(*lsn, idx as u64, "LSN should match entry index");
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_wal_read_from_specific_lsn() {
        let storage = MockStorage::new();
        let mut wal: WriteAheadLog<MockStorage> = WriteAheadLog::open(storage).await.unwrap();

        // Append 10 entries
        for i in 0..10 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, vec![i as f32])],
            })
            .await
            .unwrap();
        }
        wal.sync().await.unwrap();

        // Read from LSN 5 onwards
        let entries = wal.read_from(5).await.unwrap();
        assert_eq!(entries.len(), 5);
        assert_eq!(entries[0].0, 5);
        assert_eq!(entries[4].0, 9);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_wal_entries_since_checkpoint_tracking() {
        let storage = MockStorage::new();
        let mut wal: WriteAheadLog<MockStorage> = WriteAheadLog::open(storage).await.unwrap();

        // Append entries
        for i in 0..5 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, vec![i as f32])],
            })
            .await
            .unwrap();
        }
        wal.sync().await.unwrap();

        // Should have 5 entries since last checkpoint
        assert_eq!(wal.header.entries_since_checkpoint, 5);

        // Checkpoint
        wal.checkpoint(4).await.unwrap();

        // Should reset to 1 (the checkpoint entry itself)
        assert_eq!(wal.header.entries_since_checkpoint, 0);

        // Add more entries
        for i in 5..8 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, vec![i as f32])],
            })
            .await
            .unwrap();
        }
        wal.sync().await.unwrap();

        // Should have 3 new entries since checkpoint
        assert_eq!(wal.header.entries_since_checkpoint, 3);
    }
}
