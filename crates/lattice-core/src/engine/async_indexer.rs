//! Async Indexer - Background worker for HNSW index updates
//!
//! This module provides asynchronous indexing for improved insert latency.
//! Points are stored immediately and indexed in the background.
//!
//! Native-only: Uses std::thread (not tokio) to maintain lattice-core's I/O-free design.

use crate::index::hnsw::HnswIndex;
use crate::types::point::{Point, PointId};
use rustc_hash::FxHashSet;
use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};

/// Task sent to the background indexer
#[derive(Debug)]
pub enum IndexTask {
    /// Insert a point into the index
    Insert(PointId),
    /// Delete a point from the index
    Delete(PointId),
    /// Shutdown the indexer
    Shutdown,
}

/// Handle to control the background indexer
pub struct AsyncIndexerHandle {
    /// Channel to send tasks to the background worker
    tx: Sender<IndexTask>,
    /// Handle to the background thread
    thread_handle: Option<JoinHandle<()>>,
}

impl AsyncIndexerHandle {
    /// Create a new async indexer and spawn the background worker
    pub fn spawn(
        index: Arc<RwLock<HnswIndex>>,
        points: Arc<RwLock<BTreeMap<PointId, Point>>>,
        pending: Arc<RwLock<FxHashSet<PointId>>>,
    ) -> Self {
        let (tx, rx) = mpsc::channel();

        let thread_handle = thread::spawn(move || {
            AsyncIndexer::new(rx, index, points, pending).run();
        });

        Self {
            tx,
            thread_handle: Some(thread_handle),
        }
    }

    /// Queue a point for indexing
    pub fn queue_insert(&self, id: PointId) {
        // Ignore send errors (worker may have shut down)
        let _ = self.tx.send(IndexTask::Insert(id));
    }

    /// Queue a point for deletion from index
    pub fn queue_delete(&self, id: PointId) {
        let _ = self.tx.send(IndexTask::Delete(id));
    }

    /// Shutdown the background worker and wait for completion
    pub fn shutdown(&mut self) {
        let _ = self.tx.send(IndexTask::Shutdown);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }

    /// Check if the indexer is still running
    pub fn is_running(&self) -> bool {
        self.thread_handle.as_ref().map_or(false, |h| !h.is_finished())
    }
}

impl Drop for AsyncIndexerHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Background indexer worker
struct AsyncIndexer {
    /// Channel to receive tasks
    rx: Receiver<IndexTask>,
    /// Shared HNSW index
    index: Arc<RwLock<HnswIndex>>,
    /// Shared point storage
    points: Arc<RwLock<BTreeMap<PointId, Point>>>,
    /// Set of points pending indexing (FxHashSet for faster integer hashing)
    pending: Arc<RwLock<FxHashSet<PointId>>>,
}

impl AsyncIndexer {
    fn new(
        rx: Receiver<IndexTask>,
        index: Arc<RwLock<HnswIndex>>,
        points: Arc<RwLock<BTreeMap<PointId, Point>>>,
        pending: Arc<RwLock<FxHashSet<PointId>>>,
    ) -> Self {
        Self {
            rx,
            index,
            points,
            pending,
        }
    }

    /// Run the indexer loop
    fn run(self) {
        loop {
            // Process all available tasks before blocking
            loop {
                match self.rx.try_recv() {
                    Ok(task) => {
                        if !self.process_task(task) {
                            return; // Shutdown requested
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return,
                }
            }

            // Block waiting for next task
            match self.rx.recv() {
                Ok(task) => {
                    if !self.process_task(task) {
                        return; // Shutdown requested
                    }
                }
                Err(_) => return, // Channel disconnected
            }
        }
    }

    /// Process a single task. Returns false if shutdown requested.
    fn process_task(&self, task: IndexTask) -> bool {
        match task {
            IndexTask::Insert(id) => {
                self.index_point(id);
                true
            }
            IndexTask::Delete(id) => {
                self.delete_from_index(id);
                true
            }
            IndexTask::Shutdown => false,
        }
    }

    /// Index a single point
    fn index_point(&self, id: PointId) {
        // Get point data (need read lock on points)
        let point = {
            let points = self.points.read().unwrap();
            points.get(&id).cloned()
        };

        if let Some(point) = point {
            // Insert into index (need write lock on index)
            {
                let mut index = self.index.write().unwrap();
                index.insert(&point);
            }

            // Remove from pending set
            {
                let mut pending = self.pending.write().unwrap();
                pending.remove(&id);
            }
        }
    }

    /// Delete a point from the index
    fn delete_from_index(&self, id: PointId) {
        let mut index = self.index.write().unwrap();
        index.delete(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::collection::Distance as DistanceMetric;
    use crate::types::collection::HnswConfig;
    use std::time::Duration;

    fn test_index() -> HnswIndex {
        HnswIndex::new(
            HnswConfig {
                m: 16,
                m0: 32,
                ml: HnswConfig::recommended_ml(16),
                ef: 100,
                ef_construction: 200,
            },
            DistanceMetric::Cosine,
        )
    }

    #[test]
    fn test_async_indexer_insert() {
        let index = Arc::new(RwLock::new(test_index()));
        let points = Arc::new(RwLock::new(BTreeMap::new()));
        let pending = Arc::new(RwLock::new(FxHashSet::default()));

        // Add point to storage and pending
        {
            let mut pts = points.write().unwrap();
            pts.insert(1, Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4]));
            let mut pend = pending.write().unwrap();
            pend.insert(1);
        }

        // Start indexer
        let mut handle =
            AsyncIndexerHandle::spawn(Arc::clone(&index), Arc::clone(&points), Arc::clone(&pending));

        // Queue insert
        handle.queue_insert(1);

        // Wait for indexing
        thread::sleep(Duration::from_millis(100));

        // Verify point is indexed and removed from pending
        {
            let pend = pending.read().unwrap();
            assert!(pend.is_empty(), "Pending should be empty after indexing");
        }

        // Shutdown
        handle.shutdown();
    }

    #[test]
    fn test_async_indexer_shutdown() {
        let index = Arc::new(RwLock::new(test_index()));
        let points = Arc::new(RwLock::new(BTreeMap::new()));
        let pending = Arc::new(RwLock::new(FxHashSet::default()));

        let mut handle =
            AsyncIndexerHandle::spawn(Arc::clone(&index), Arc::clone(&points), Arc::clone(&pending));

        assert!(handle.is_running());

        handle.shutdown();

        // Give thread time to fully terminate
        thread::sleep(Duration::from_millis(50));
        assert!(!handle.is_running());
    }
}
