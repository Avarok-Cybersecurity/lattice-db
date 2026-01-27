//! Background Checkpoint Manager
//!
//! Periodically creates checkpoints to:
//! 1. Reduce WAL replay time on recovery
//! 2. Allow truncation of old WAL entries
//! 3. Ensure data durability in the face of crashes

use crate::error::LatticeResult;
use crate::storage::LatticeStorage;
use crate::wal::{Lsn, WriteAheadLog};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tokio::sync::Mutex as TokioMutex;
use std::sync::Arc;

/// Reserved page range for checkpoints in data storage
pub const CHECKPOINT_PAGE_START: u64 = 1_000_000;

/// Configuration for the checkpoint manager
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Interval between checkpoint checks
    pub interval: Duration,
    /// Entry threshold to trigger checkpoint
    pub entry_threshold: u64,
    /// Whether checkpointing is enabled
    pub enabled: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(60),
            entry_threshold: 10_000,
            enabled: true,
        }
    }
}

/// Callback type for performing a checkpoint
///
/// This is called by the checkpoint manager when it's time to checkpoint.
/// The callback should:
/// 1. Serialize the current engine state to bytes
/// 2. Write the snapshot to data storage
/// 3. Return the LSN at which the checkpoint was taken
pub type CheckpointCallback = Box<dyn Fn() -> LatticeResult<(Vec<u8>, Lsn)> + Send + Sync>;

/// Handle for the background checkpoint manager
pub struct CheckpointManager<S: LatticeStorage> {
    shutdown_tx: mpsc::Sender<()>,
    thread: Option<JoinHandle<()>>,
    wal: Arc<TokioMutex<WriteAheadLog<S>>>,
}

impl<S: LatticeStorage + Send + Sync + 'static> CheckpointManager<S> {
    /// Spawn a background checkpoint manager
    ///
    /// # Arguments
    /// - `wal`: Shared WAL instance
    /// - `data_storage`: Storage for writing checkpoints
    /// - `config`: Checkpoint configuration
    /// - `checkpoint_fn`: Callback to create checkpoint data
    pub fn spawn<D: LatticeStorage + Send + Sync + 'static>(
        wal: Arc<TokioMutex<WriteAheadLog<S>>>,
        data_storage: Arc<D>,
        config: CheckpointConfig,
        checkpoint_fn: CheckpointCallback,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = mpsc::channel();
        let wal_clone = Arc::clone(&wal);

        let thread = if config.enabled {
            Some(thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create checkpoint runtime");

                rt.block_on(async {
                    Self::checkpoint_loop(
                        wal_clone,
                        data_storage,
                        config,
                        checkpoint_fn,
                        shutdown_rx,
                    )
                    .await
                });
            }))
        } else {
            None
        };

        Self {
            shutdown_tx,
            thread,
            wal,
        }
    }

    /// Main checkpoint loop
    async fn checkpoint_loop<D: LatticeStorage>(
        wal: Arc<TokioMutex<WriteAheadLog<S>>>,
        data_storage: Arc<D>,
        config: CheckpointConfig,
        checkpoint_fn: CheckpointCallback,
        shutdown_rx: mpsc::Receiver<()>,
    ) {
        loop {
            // Check for shutdown or wait interval
            match shutdown_rx.recv_timeout(config.interval) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }

            // Check if checkpoint needed
            let should_checkpoint = {
                let wal_guard = wal.lock().await;
                wal_guard.should_checkpoint()
            };

            if should_checkpoint {
                if let Err(e) = Self::perform_checkpoint(
                    &wal,
                    &data_storage,
                    &checkpoint_fn,
                )
                .await
                {
                    tracing::error!(?e, "Background checkpoint failed");
                }
            }
        }
    }

    /// Perform a checkpoint
    async fn perform_checkpoint<D: LatticeStorage>(
        wal: &Arc<TokioMutex<WriteAheadLog<S>>>,
        data_storage: &Arc<D>,
        checkpoint_fn: &CheckpointCallback,
    ) -> LatticeResult<()> {
        tracing::info!("Starting background checkpoint");

        // 1. Create checkpoint data via callback
        let (snapshot, lsn) = checkpoint_fn()?;

        // 2. Write snapshot to data storage
        data_storage
            .write_page(CHECKPOINT_PAGE_START, &snapshot)
            .await?;
        data_storage.sync().await?;

        // 3. Record checkpoint in WAL and truncate old entries
        {
            let mut wal_guard = wal.lock().await;
            wal_guard.checkpoint(lsn).await?;
        }

        tracing::info!(lsn, bytes = snapshot.len(), "Background checkpoint complete");
        Ok(())
    }

    /// Manually trigger a checkpoint
    pub async fn trigger_checkpoint<D: LatticeStorage>(
        &self,
        data_storage: &Arc<D>,
        checkpoint_fn: &CheckpointCallback,
    ) -> LatticeResult<()> {
        Self::perform_checkpoint(&self.wal, data_storage, checkpoint_fn).await
    }

    /// Shutdown the checkpoint manager gracefully
    pub fn shutdown(&mut self) {
        let _ = self.shutdown_tx.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl<S: LatticeStorage> Drop for CheckpointManager<S> {
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
    use std::sync::atomic::{AtomicU64, Ordering};

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

    impl Clone for MockStorage {
        fn clone(&self) -> Self {
            Self {
                pages: Arc::clone(&self.pages),
                meta: Arc::clone(&self.meta),
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
    async fn test_checkpoint_manager_manual_trigger() {
        let wal_storage = MockStorage::new();
        let data_storage = Arc::new(MockStorage::new());
        let wal = Arc::new(TokioMutex::new(
            WriteAheadLog::open(wal_storage).await.unwrap(),
        ));

        // Add entries to WAL
        {
            let mut wal_guard = wal.lock().await;
            for i in 0..100 {
                wal_guard
                    .append(&WalEntry::Upsert {
                        points: vec![Point::new_vector(i, vec![i as f32])],
                    })
                    .await
                    .unwrap();
            }
            wal_guard.sync().await.unwrap();
        }

        let checkpoint_counter = Arc::new(AtomicU64::new(0));
        let counter_clone1 = Arc::clone(&checkpoint_counter);
        let counter_clone2 = Arc::clone(&checkpoint_counter);

        // Create two separate callbacks that share the counter
        let checkpoint_fn_for_spawn: CheckpointCallback = Box::new(move || {
            let count = counter_clone1.fetch_add(1, Ordering::SeqCst);
            Ok((format!("checkpoint_{}", count).into_bytes(), 50))
        });

        let checkpoint_fn_for_trigger: CheckpointCallback = Box::new(move || {
            let count = counter_clone2.fetch_add(1, Ordering::SeqCst);
            Ok((format!("checkpoint_{}", count).into_bytes(), 50))
        });

        let config = CheckpointConfig {
            enabled: false, // Disable background thread for manual test
            ..Default::default()
        };

        let manager = CheckpointManager::spawn(
            Arc::clone(&wal),
            Arc::clone(&data_storage),
            config,
            checkpoint_fn_for_spawn,
        );

        // Trigger checkpoint manually using the manager
        manager
            .trigger_checkpoint(&data_storage, &checkpoint_fn_for_trigger)
            .await
            .unwrap();

        // Verify checkpoint was written
        let checkpoint_data = data_storage.read_page(CHECKPOINT_PAGE_START).await.unwrap();
        assert!(!checkpoint_data.is_empty());
        assert_eq!(checkpoint_data, b"checkpoint_0");

        // Verify checkpoint counter was incremented
        assert_eq!(checkpoint_counter.load(Ordering::SeqCst), 1);

        drop(manager);
    }
}
