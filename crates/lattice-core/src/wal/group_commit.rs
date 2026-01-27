//! Group Commit for batched WAL syncs
//!
//! Coordinates multiple transactions to share a single fsync,
//! reducing I/O overhead for high-throughput workloads.
//!
//! # How It Works
//!
//! Instead of syncing after every append, callers request a sync which
//! gets batched with other pending sync requests. A background thread
//! periodically performs a single sync for all waiting callers.

use crate::error::{LatticeError, LatticeResult};
use crate::storage::LatticeStorage;
use crate::wal::WriteAheadLog;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tokio::sync::Mutex as TokioMutex;
use std::sync::Arc;

/// Message types for group commit coordination
enum GroupCommitMsg {
    /// Request a sync (returns completion via oneshot channel)
    RequestSync(tokio::sync::oneshot::Sender<Result<(), String>>),
    /// Shutdown the group commit thread
    Shutdown,
}

/// Handle for coordinating group commits
///
/// Spawns a background thread that batches sync requests together.
pub struct GroupCommitHandle {
    sync_tx: mpsc::Sender<GroupCommitMsg>,
    thread: Option<JoinHandle<()>>,
}

impl GroupCommitHandle {
    /// Spawn a group commit coordinator thread
    ///
    /// # Arguments
    /// - `wal`: Shared WAL instance (wrapped in TokioMutex)
    /// - `sync_interval`: Maximum time to wait before syncing (even if only one request)
    pub fn spawn<S: LatticeStorage + Send + Sync + 'static>(
        wal: Arc<TokioMutex<WriteAheadLog<S>>>,
        sync_interval: Duration,
    ) -> Self {
        let (tx, rx) = mpsc::channel();

        let thread = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime for group commit");

            rt.block_on(async {
                let mut pending_syncs: Vec<tokio::sync::oneshot::Sender<Result<(), String>>> =
                    Vec::new();

                loop {
                    // Wait for sync request or timeout
                    match rx.recv_timeout(sync_interval) {
                        Ok(GroupCommitMsg::RequestSync(responder)) => {
                            pending_syncs.push(responder);

                            // Batch: collect more requests with short timeout (100µs)
                            while let Ok(msg) = rx.recv_timeout(Duration::from_micros(100)) {
                                match msg {
                                    GroupCommitMsg::RequestSync(r) => pending_syncs.push(r),
                                    GroupCommitMsg::Shutdown => {
                                        // Drain pending before shutdown
                                        Self::sync_and_notify(&wal, &mut pending_syncs).await;
                                        return;
                                    }
                                }
                            }
                        }
                        Ok(GroupCommitMsg::Shutdown) => {
                            // Drain pending before shutdown
                            Self::sync_and_notify(&wal, &mut pending_syncs).await;
                            return;
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            // Timer expired, sync if pending
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            return;
                        }
                    }

                    if !pending_syncs.is_empty() {
                        Self::sync_and_notify(&wal, &mut pending_syncs).await;
                    }
                }
            });
        });

        Self {
            sync_tx: tx,
            thread: Some(thread),
        }
    }

    /// Perform sync and notify all waiting callers
    async fn sync_and_notify<S: LatticeStorage>(
        wal: &Arc<TokioMutex<WriteAheadLog<S>>>,
        pending_syncs: &mut Vec<tokio::sync::oneshot::Sender<Result<(), String>>>,
    ) {
        // Perform single sync for all pending
        let result = wal.lock().await.sync().await;

        // Convert result to string for sending
        let send_result = match &result {
            Ok(()) => Ok(()),
            Err(e) => Err(e.to_string()),
        };

        // Notify all waiters
        for responder in pending_syncs.drain(..) {
            let _ = responder.send(send_result.clone());
        }
    }

    /// Request sync and wait for completion
    ///
    /// This will batch with other pending sync requests for efficiency.
    pub async fn sync(&self) -> LatticeResult<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        self.sync_tx
            .send(GroupCommitMsg::RequestSync(tx))
            .map_err(|_| LatticeError::Internal {
                code: 50010,
                message: "Group commit thread died".into(),
            })?;

        rx.await
            .map_err(|_| LatticeError::Internal {
                code: 50011,
                message: "Group commit sync channel closed".into(),
            })?
            .map_err(|e| LatticeError::Internal {
                code: 50012,
                message: format!("Group commit sync failed: {}", e),
            })
    }

    /// Shutdown the group commit thread gracefully
    pub fn shutdown(&mut self) {
        let _ = self.sync_tx.send(GroupCommitMsg::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for GroupCommitHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::StorageResult;
    use crate::wal::WalEntry;
    use crate::types::point::Point;
    use std::collections::HashMap;
    use std::sync::RwLock;

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
            self.pages
                .write()
                .unwrap()
                .insert(page_id, data.to_vec());
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
    async fn test_group_commit_single_sync() {
        let storage = MockStorage::new();
        let wal = Arc::new(TokioMutex::new(WriteAheadLog::open(storage).await.unwrap()));

        // Append an entry
        {
            let mut wal_guard = wal.lock().await;
            wal_guard
                .append(&WalEntry::Upsert {
                    points: vec![Point::new_vector(1, vec![0.1, 0.2, 0.3])],
                })
                .await
                .unwrap();
        }

        let group_commit = GroupCommitHandle::spawn(Arc::clone(&wal), Duration::from_millis(10));

        // Request sync
        group_commit.sync().await.unwrap();

        // Verify entry was synced
        let entries = wal.lock().await.read_from(0).await.unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[tokio::test]
    async fn test_group_commit_batching() {
        let storage = MockStorage::new();
        let wal = Arc::new(TokioMutex::new(WriteAheadLog::open(storage).await.unwrap()));

        let group_commit =
            Arc::new(GroupCommitHandle::spawn(Arc::clone(&wal), Duration::from_millis(50)));

        // Spawn multiple tasks that request sync concurrently
        let mut handles = Vec::new();
        for i in 0..5 {
            let wal_clone = Arc::clone(&wal);
            let gc_clone = Arc::clone(&group_commit);
            handles.push(tokio::spawn(async move {
                {
                    let mut wal_guard = wal_clone.lock().await;
                    wal_guard
                        .append(&WalEntry::Upsert {
                            points: vec![Point::new_vector(i, vec![i as f32])],
                        })
                        .await
                        .unwrap();
                }
                gc_clone.sync().await.unwrap();
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // Verify all entries were synced
        let entries = wal.lock().await.read_from(0).await.unwrap();
        assert_eq!(entries.len(), 5);
    }
}
