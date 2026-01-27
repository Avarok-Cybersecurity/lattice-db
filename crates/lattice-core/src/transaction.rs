//! Transaction Manager for ACID operations
//!
//! Provides auto-commit transactions with WAL durability.
//! Each mutation (upsert, delete, add_edge, remove_edge) is logged
//! before being applied, ensuring crash recovery.
//!
//! # Transaction Model
//!
//! Phase 1 uses single-operation auto-commit transactions:
//! - Each operation is its own transaction
//! - Log-then-apply: write to WAL, sync, then apply to memory
//! - Recovery: replay WAL from last checkpoint
//!
//! # Future Enhancements
//!
//! See ACID-FUTURE.md for multi-operation transactions (BEGIN/COMMIT/ROLLBACK).

use crate::error::LatticeResult;
use crate::storage::LatticeStorage;
use crate::types::point::{Edge, Point, PointId};
use crate::wal::{Lsn, WalEntry, WriteAheadLog};

/// Transaction Manager
///
/// Coordinates WAL logging with in-memory state mutations.
pub struct TransactionManager<S: LatticeStorage> {
    wal: WriteAheadLog<S>,
}

impl<S: LatticeStorage> TransactionManager<S> {
    /// Create a new transaction manager with the given storage
    pub async fn open(storage: S) -> LatticeResult<Self> {
        let wal = WriteAheadLog::open(storage).await?;
        Ok(Self { wal })
    }

    /// Log an upsert operation and return the LSN
    ///
    /// After calling this, the caller should apply the mutation to memory.
    /// The sync ensures durability before returning.
    pub async fn log_upsert(&mut self, points: Vec<Point>) -> LatticeResult<Lsn> {
        let lsn = self.wal.append(&WalEntry::Upsert { points }).await?;
        self.wal.sync().await?;
        Ok(lsn)
    }

    /// Log an upsert operation WITHOUT syncing (for batch operations)
    ///
    /// Call `sync()` after all operations to ensure durability.
    pub async fn log_upsert_nosync(&mut self, points: Vec<Point>) -> LatticeResult<Lsn> {
        self.wal.append(&WalEntry::Upsert { points }).await
    }

    /// Sync WAL to durable storage
    ///
    /// Call this after batching multiple log_*_nosync operations.
    pub async fn sync(&mut self) -> LatticeResult<()> {
        self.wal.sync().await
    }

    /// Log a delete operation and return the LSN
    pub async fn log_delete(&mut self, ids: Vec<PointId>) -> LatticeResult<Lsn> {
        let lsn = self.wal.append(&WalEntry::Delete { ids }).await?;
        self.wal.sync().await?;
        Ok(lsn)
    }

    /// Log an add_edge operation and return the LSN
    pub async fn log_add_edge(&mut self, from_id: PointId, edge: Edge) -> LatticeResult<Lsn> {
        let lsn = self
            .wal
            .append(&WalEntry::AddEdge { from_id, edge })
            .await?;
        self.wal.sync().await?;
        Ok(lsn)
    }

    /// Log an abort for a previously logged operation
    ///
    /// Used when in-memory apply fails after WAL commit.
    /// Recovery will skip the aborted LSN during replay.
    pub async fn log_abort(&mut self, aborted_lsn: Lsn) -> LatticeResult<Lsn> {
        let lsn = self.wal.append(&WalEntry::Abort { aborted_lsn }).await?;
        self.wal.sync().await?;
        Ok(lsn)
    }

    /// Log a remove_edge operation and return the LSN
    pub async fn log_remove_edge(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation_id: u16,
    ) -> LatticeResult<Lsn> {
        let lsn = self
            .wal
            .append(&WalEntry::RemoveEdge {
                from_id,
                to_id,
                relation_id,
            })
            .await?;
        self.wal.sync().await?;
        Ok(lsn)
    }

    /// Read all WAL entries from the given LSN for recovery
    pub async fn read_entries_from(&self, start_lsn: Lsn) -> LatticeResult<Vec<(Lsn, WalEntry)>> {
        self.wal.read_from(start_lsn).await
    }

    /// Get the LSN of the last checkpoint
    pub fn last_checkpoint_lsn(&self) -> Lsn {
        self.wal.last_checkpoint_lsn()
    }

    /// Get the next LSN that will be assigned
    pub fn next_lsn(&self) -> Lsn {
        self.wal.next_lsn()
    }

    /// Check if a checkpoint is recommended
    pub fn should_checkpoint(&self) -> bool {
        self.wal.should_checkpoint()
    }

    /// Record a checkpoint after saving a snapshot
    pub async fn checkpoint(&mut self, snapshot_lsn: Lsn) -> LatticeResult<()> {
        self.wal.checkpoint(snapshot_lsn).await
    }

    /// Truncate WAL entries before the given LSN
    pub async fn truncate_before(&mut self, lsn: Lsn) -> LatticeResult<()> {
        self.wal.truncate_before(lsn).await
    }
}

/// Recovery helper functions
pub mod recovery {
    use super::*;
    use crate::engine::collection::CollectionEngine;
    use std::collections::HashSet;

    /// Replay WAL entries to restore engine state
    ///
    /// This is called during engine startup when durability mode is enabled.
    /// Aborted operations are collected first, then skipped during replay.
    pub async fn replay_wal<S: LatticeStorage>(
        engine: &mut CollectionEngine,
        txn_manager: &TransactionManager<S>,
    ) -> LatticeResult<usize> {
        let start_lsn = txn_manager.last_checkpoint_lsn();
        let entries = txn_manager.read_entries_from(start_lsn).await?;

        // First pass: collect aborted LSNs
        let aborted_lsns: HashSet<Lsn> = entries
            .iter()
            .filter_map(|(_, entry)| match entry {
                WalEntry::Abort { aborted_lsn } => Some(*aborted_lsn),
                _ => None,
            })
            .collect();

        // Second pass: replay non-aborted entries
        let mut replayed = 0;

        for (lsn, entry) in entries {
            if aborted_lsns.contains(&lsn) {
                continue;
            }

            match entry {
                WalEntry::Upsert { points } => {
                    engine.upsert_points(points)?;
                    replayed += 1;
                }
                WalEntry::Delete { ids } => {
                    engine.delete_points(&ids)?;
                    replayed += 1;
                }
                WalEntry::AddEdge { from_id, edge } => {
                    let relation_name = engine
                        .config()
                        .relation_name(edge.relation_id)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("relation_{}", edge.relation_id));

                    engine.add_edge(from_id, edge.target_id, &relation_name, edge.weight)?;
                    replayed += 1;
                }
                WalEntry::RemoveEdge {
                    from_id,
                    to_id,
                    relation_id,
                } => {
                    let relation_name = engine
                        .config()
                        .relation_name(relation_id)
                        .map(|s| s.to_string());

                    engine.remove_edge(from_id, to_id, relation_name.as_deref())?;
                    replayed += 1;
                }
                WalEntry::Checkpoint { .. } | WalEntry::Abort { .. } => {
                    // Skip control entries during replay
                }
            }
        }

        Ok(replayed)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::storage::StorageResult;
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};

    // Mock storage for testing
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

    #[tokio::test]
    async fn test_transaction_manager_log_upsert() {
        let storage = MockStorage::new();
        let mut txn = TransactionManager::open(storage).await.unwrap();

        let point = Point::new_vector(1, vec![0.1, 0.2, 0.3]);
        let lsn = txn.log_upsert(vec![point.clone()]).await.unwrap();

        assert_eq!(lsn, 0);
        assert_eq!(txn.next_lsn(), 1);
    }

    #[tokio::test]
    async fn test_transaction_manager_log_delete() {
        let storage = MockStorage::new();
        let mut txn = TransactionManager::open(storage).await.unwrap();

        let lsn = txn.log_delete(vec![1, 2, 3]).await.unwrap();

        assert_eq!(lsn, 0);
    }

    #[tokio::test]
    async fn test_transaction_manager_multiple_operations() {
        let storage = MockStorage::new();
        let mut txn = TransactionManager::open(storage).await.unwrap();

        // Multiple operations
        let lsn1 = txn
            .log_upsert(vec![Point::new_vector(1, vec![0.1])])
            .await
            .unwrap();
        let lsn2 = txn.log_delete(vec![1]).await.unwrap();
        let lsn3 = txn.log_add_edge(1, Edge::new(2, 1.0, 0)).await.unwrap();

        assert_eq!(lsn1, 0);
        assert_eq!(lsn2, 1);
        assert_eq!(lsn3, 2);

        // Read back
        let entries = txn.read_entries_from(0).await.unwrap();
        assert_eq!(entries.len(), 3);
    }
}
