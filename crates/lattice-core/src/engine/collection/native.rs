//! Native (non-WASM) CollectionEngine implementation with async indexing
//!
//! On native builds, the engine uses async indexing for improved insert latency:
//! - Points are stored immediately in `points` storage
//! - Index updates happen in background via `AsyncIndexer`
//! - Search merges indexed results with brute-force on pending points
//!
//! ## Durability
//!
//! The engine supports two modes via generic type parameters:
//! - **Ephemeral** (`CollectionEngine<NoStorage, NoStorage>`): Fast, in-memory only
//! - **Durable** (`CollectionEngine<W, D>`): WAL-backed with crash recovery

use crate::engine::async_indexer::{AsyncIndexerHandle, BackpressureSignal};
use crate::error::{ConfigError, LatticeError, LatticeResult};
use crate::index::hnsw::HnswIndex;
use crate::storage::{LatticeStorage, StorageError, StorageResult};
use crate::sync::{LockExt, MutexExt};
use crate::transaction::TransactionManager;
use crate::types::collection::{CollectionConfig, DurabilityMode};
use crate::types::point::{Edge, Point, PointId};
use crate::types::query::{ScrollPoint, ScrollQuery, ScrollResult, SearchQuery, SearchResult};
use crate::wal::{Lsn, WalEntry};
use rayon::prelude::*;
use rkyv::rancor::Error as RkyvError;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::marker::PhantomData;
use std::ops::Bound;
use std::sync::{Arc, Condvar, Mutex, RwLock};
use tokio::sync::Mutex as TokioMutex;

use super::types::{EdgeInfo, TraversalPath, TraversalResult, UpsertResult};

/// Marker type for no storage (ephemeral mode)
///
/// Implements `LatticeStorage` to satisfy trait bounds. All methods return
/// errors because ephemeral engines never use the transaction manager
/// (it's always `None`). If a code path reaches these methods, it indicates
/// a bug in engine construction — returning an error is safer than panicking.
#[derive(Debug, Clone, Copy)]
pub struct NoStorage;

const NO_STORAGE_MSG: &str = "NoStorage: ephemeral mode has no storage backend";

#[async_trait::async_trait]
impl LatticeStorage for NoStorage {
    async fn get_meta(&self, _key: &str) -> StorageResult<Option<Vec<u8>>> {
        Err(StorageError::Io {
            message: NO_STORAGE_MSG.into(),
        })
    }

    async fn set_meta(&self, _key: &str, _value: &[u8]) -> StorageResult<()> {
        Err(StorageError::Io {
            message: NO_STORAGE_MSG.into(),
        })
    }

    async fn delete_meta(&self, _key: &str) -> StorageResult<()> {
        Err(StorageError::Io {
            message: NO_STORAGE_MSG.into(),
        })
    }

    async fn read_page(&self, _page_id: u64) -> StorageResult<Vec<u8>> {
        Err(StorageError::Io {
            message: NO_STORAGE_MSG.into(),
        })
    }

    async fn write_page(&self, _page_id: u64, _data: &[u8]) -> StorageResult<()> {
        Err(StorageError::Io {
            message: NO_STORAGE_MSG.into(),
        })
    }

    async fn page_exists(&self, _page_id: u64) -> StorageResult<bool> {
        Err(StorageError::Io {
            message: NO_STORAGE_MSG.into(),
        })
    }

    async fn delete_page(&self, _page_id: u64) -> StorageResult<()> {
        Err(StorageError::Io {
            message: NO_STORAGE_MSG.into(),
        })
    }

    async fn sync(&self) -> StorageResult<()> {
        Err(StorageError::Io {
            message: NO_STORAGE_MSG.into(),
        })
    }
}

use super::PAGE_POINTS_START;

/// Max pending points before blocking upsert to let indexer catch up
/// Higher = faster upserts, slower search on pending buffer
/// Lower = slower upserts (blocks more), faster search
/// Set high enough that typical benchmark batches don't trigger blocking
const PENDING_THRESHOLD: usize = 10_000;

/// Initial capacity for the label index HashMap. 64 covers typical collections
/// with moderate label variety; the map resizes automatically if exceeded.
const INITIAL_LABEL_CAPACITY: usize = 64;

/// Maximum iterations for backpressure wait loop before proceeding anyway.
/// At BACKPRESSURE_POLL_MS per iteration, this is a 5-second timeout.
const BACKPRESSURE_MAX_ITERATIONS: u32 = 500;

/// Polling interval (ms) for backpressure condvar wait.
const BACKPRESSURE_POLL_MS: u64 = 10;

/// Collection engine - manages a single collection (Native implementation)
///
/// Thread-safe with Arc<RwLock<>> for concurrent reads.
/// Uses async indexing for improved insert latency.
///
/// # Type Parameters
///
/// - `W`: WAL storage backend. Use `NoStorage` for ephemeral mode.
/// - `D`: Data/checkpoint storage backend. Defaults to same as WAL.
///
/// # Examples
///
/// ```rust,ignore
/// // Ephemeral engine (fast, no persistence)
/// let engine = CollectionEngine::new(config)?;
///
/// // Durable engine (WAL-backed with crash recovery)
/// let engine = CollectionEngine::open_durable(config, wal_storage, data_storage).await?;
/// ```
pub struct CollectionEngine<W: LatticeStorage = NoStorage, D: LatticeStorage = W> {
    /// Collection configuration
    config: CollectionConfig,
    /// HNSW index for vector search
    index: Arc<RwLock<HnswIndex>>,
    /// Point storage: id -> Point
    points: Arc<RwLock<BTreeMap<PointId, Point>>>,
    /// Points pending indexing (not yet in HNSW)
    pending_index: Arc<RwLock<FxHashSet<PointId>>>,
    /// Background indexer handle
    indexer: AsyncIndexerHandle,
    /// Label index: label -> set of point IDs (for O(1) label lookups)
    label_index: Arc<RwLock<HashMap<String, FxHashSet<PointId>>>>,
    /// Signal for backpressure coordination with async indexer
    backpressure_signal: BackpressureSignal,
    /// Next available page ID for storage (reserved for Phase 2: point-level persistence)
    _next_page_id: u64,

    // === Durability fields ===
    /// Transaction manager for WAL (None in ephemeral mode)
    txn_manager: Option<Arc<TokioMutex<TransactionManager<W>>>>,
    /// Data storage for checkpoints (None in ephemeral mode, reserved for Phase 2)
    _data_storage: Option<D>,
    /// Phantom data for type parameters
    _phantom: PhantomData<(W, D)>,
}

/// Type aliases for common engine configurations
pub type EphemeralEngine = CollectionEngine<NoStorage, NoStorage>;

/// Durable engine with unified storage for WAL and data
pub type DurableEngine<S> = CollectionEngine<S, S>;

impl CollectionEngine<NoStorage, NoStorage> {
    /// Create a new ephemeral collection engine (no durability)
    ///
    /// This is the fast path with no WAL overhead. Data is lost on restart.
    pub fn new(mut config: CollectionConfig) -> LatticeResult<Self> {
        // Force ephemeral mode for this constructor
        config.durability = DurabilityMode::Ephemeral;
        config.validate()?;

        let index = Arc::new(RwLock::new(HnswIndex::new(
            config.hnsw.clone(),
            config.vectors.distance,
        )));
        let points = Arc::new(RwLock::new(BTreeMap::new()));
        let pending_index = Arc::new(RwLock::new(FxHashSet::default()));
        // Pre-allocate for typical number of unique labels in a graph
        let label_index = Arc::new(RwLock::new(HashMap::with_capacity(INITIAL_LABEL_CAPACITY)));
        let backpressure_signal = Arc::new((Mutex::new(()), Condvar::new()));

        // Spawn background indexer
        let indexer = AsyncIndexerHandle::spawn(
            Arc::clone(&index),
            Arc::clone(&points),
            Arc::clone(&pending_index),
            Arc::clone(&backpressure_signal),
        );

        Ok(Self {
            config,
            index,
            points,
            pending_index,
            indexer,
            label_index,
            backpressure_signal,
            _next_page_id: PAGE_POINTS_START,
            txn_manager: None,
            _data_storage: None,
            _phantom: PhantomData,
        })
    }

    // --- Public Mutation Methods (Sync, Ephemeral Only) ---

    /// Upsert points into the collection (sync, ephemeral mode)
    ///
    /// Points are stored immediately and indexed asynchronously in background.
    /// If pending buffer exceeds threshold, blocks until indexer catches up.
    pub fn upsert_points(&mut self, points: Vec<Point>) -> LatticeResult<UpsertResult> {
        self.apply_upsert_internal(points)
    }

    /// Delete points from the collection (sync, ephemeral mode)
    ///
    /// Returns the number of points actually deleted.
    pub fn delete_points(&mut self, ids: &[PointId]) -> LatticeResult<usize> {
        self.apply_delete_internal(ids)
    }

    /// Add a directed edge between two points (sync, ephemeral mode)
    ///
    /// Creates a relation if it doesn't exist. Edges are stored in the source point.
    pub fn add_edge(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: &str,
        weight: f32,
    ) -> LatticeResult<()> {
        let relation_id = self.get_or_create_relation_id(relation);
        let edge = Edge::new(to_id, weight, relation_id);
        self.apply_add_edge_internal(from_id, edge)
    }

    /// Remove an edge between two points (sync, ephemeral mode)
    ///
    /// If `relation` is None, removes edges with any relation (first match only).
    /// Returns true if an edge was removed.
    pub fn remove_edge(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: Option<&str>,
    ) -> LatticeResult<bool> {
        if let Some(rel) = relation {
            if let Some(relation_id) = self.config.relation_id(rel) {
                return self.apply_remove_edge_internal(from_id, to_id, relation_id);
            }
            Ok(false) // Relation doesn't exist
        } else {
            // Remove first matching edge regardless of relation
            let mut pts = self.points.write_safe()?;
            let point = pts
                .get_mut(&from_id)
                .ok_or(LatticeError::PointNotFound { id: from_id })?;

            if let Some(edges) = &mut point.outgoing_edges {
                let original_len = edges.len();
                edges.retain(|e| e.target_id != to_id);
                Ok(edges.len() < original_len)
            } else {
                Ok(false)
            }
        }
    }

    /// Deserialize a collection from bytes (creates ephemeral engine)
    ///
    /// This always creates an ephemeral engine. For durable engines, use
    /// `open_durable()` with the appropriate storage backends.
    pub fn from_bytes(bytes: &[u8]) -> LatticeResult<Self> {
        if bytes.len() < 8 {
            return Err(LatticeError::Serialization {
                message: "Invalid data: too short".to_string(),
            });
        }

        // Read config length with overflow check
        let config_len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let config_end =
            4usize
                .checked_add(config_len)
                .ok_or_else(|| LatticeError::Serialization {
                    message: "Invalid data: config length overflow".to_string(),
                })?;

        let config_end_plus_4 =
            config_end
                .checked_add(4)
                .ok_or_else(|| LatticeError::Serialization {
                    message: "Invalid data: offset overflow".to_string(),
                })?;

        if bytes.len() < config_end_plus_4 {
            return Err(LatticeError::Serialization {
                message: "Invalid data: config truncated".to_string(),
            });
        }

        // Deserialize config (JSON)
        let config: CollectionConfig =
            serde_json::from_slice(&bytes[4..config_end]).map_err(|e| {
                LatticeError::Serialization {
                    message: e.to_string(),
                }
            })?;

        // Read points length
        let points_len = u32::from_le_bytes([
            bytes[config_end],
            bytes[config_end + 1],
            bytes[config_end + 2],
            bytes[config_end + 3],
        ]) as usize;

        // Calculate padding (same formula as serialization) with overflow checks
        let header_size = 4usize
            .checked_add(config_len)
            .and_then(|v| v.checked_add(4))
            .ok_or_else(|| LatticeError::Serialization {
                message: "Invalid data: header size overflow".to_string(),
            })?;
        let padding = (16 - (header_size % 16)) % 16;
        let points_start = config_end
            .checked_add(4)
            .and_then(|v| v.checked_add(padding))
            .ok_or_else(|| LatticeError::Serialization {
                message: "Invalid data: points start offset overflow".to_string(),
            })?;

        let required_len =
            points_start
                .checked_add(points_len)
                .ok_or_else(|| LatticeError::Serialization {
                    message: "Invalid data: total size overflow".to_string(),
                })?;

        if bytes.len() < required_len {
            return Err(LatticeError::Serialization {
                message: "Invalid data: points truncated".to_string(),
            });
        }

        // Copy to aligned buffer for rkyv deserialization
        let mut aligned_bytes = rkyv::util::AlignedVec::<16>::new();
        aligned_bytes.extend_from_slice(&bytes[points_start..points_start + points_len]);

        // Deserialize points with rkyv
        let points_vec: Vec<Point> = rkyv::from_bytes::<Vec<Point>, RkyvError>(&aligned_bytes)
            .map_err(|e| LatticeError::Serialization {
                message: e.to_string(),
            })?;

        // Build ephemeral engine
        let mut engine = Self::new(config)?;
        engine.upsert_points(points_vec)?;

        Ok(engine)
    }
}

impl<W, D> CollectionEngine<W, D>
where
    W: LatticeStorage + Send + Sync + 'static,
    D: LatticeStorage + Send + Sync + 'static,
{
    /// Create a durable collection engine with separate WAL and data storage
    ///
    /// # Arguments
    ///
    /// - `config`: Collection configuration (durability should be `Durable`)
    /// - `wal_storage`: Storage backend for WAL entries (fast, append-heavy)
    /// - `data_storage`: Storage backend for checkpoints/snapshots
    ///
    /// # Recovery
    ///
    /// On open, the engine replays the WAL from the last checkpoint to restore state.
    pub async fn open_durable(
        config: CollectionConfig,
        wal_storage: W,
        data_storage: D,
    ) -> LatticeResult<Self> {
        if config.durability != DurabilityMode::Durable {
            return Err(LatticeError::Config(ConfigError::InvalidParameter {
                name: "durability",
                message: "Expected DurabilityMode::Durable for open_durable".to_string(),
            }));
        }

        config.validate()?;

        // Open transaction manager with WAL storage
        let txn_manager = TransactionManager::open(wal_storage).await?;

        // Create base engine
        let index = Arc::new(RwLock::new(HnswIndex::new(
            config.hnsw.clone(),
            config.vectors.distance,
        )));
        let points = Arc::new(RwLock::new(BTreeMap::new()));
        let pending_index = Arc::new(RwLock::new(FxHashSet::default()));
        let label_index = Arc::new(RwLock::new(HashMap::with_capacity(INITIAL_LABEL_CAPACITY)));
        let backpressure_signal = Arc::new((Mutex::new(()), Condvar::new()));

        let indexer = AsyncIndexerHandle::spawn(
            Arc::clone(&index),
            Arc::clone(&points),
            Arc::clone(&pending_index),
            Arc::clone(&backpressure_signal),
        );

        let txn_arc = Arc::new(TokioMutex::new(txn_manager));

        let mut engine = Self {
            config,
            index,
            points,
            pending_index,
            indexer,
            label_index,
            backpressure_signal,
            _next_page_id: PAGE_POINTS_START,
            txn_manager: Some(txn_arc),
            _data_storage: Some(data_storage),
            _phantom: PhantomData,
        };

        // Replay WAL to recover state
        let replayed = engine.recover_from_wal().await?;
        tracing::info!(replayed, "WAL recovery complete");

        Ok(engine)
    }

    /// Replay WAL entries to restore state after crash
    async fn recover_from_wal(&mut self) -> LatticeResult<usize> {
        let txn = self
            .txn_manager
            .as_ref()
            .expect("durable mode requires txn_manager");
        let txn_guard = txn.lock().await;

        let start_lsn = txn_guard.last_checkpoint_lsn();
        let entries = txn_guard.read_entries_from(start_lsn).await?;
        drop(txn_guard); // Release lock before mutating

        // Collect aborted LSNs first
        let aborted_lsns: std::collections::HashSet<Lsn> = entries
            .iter()
            .filter_map(|(_, entry)| match entry {
                WalEntry::Abort { aborted_lsn } => Some(*aborted_lsn),
                _ => None,
            })
            .collect();

        let mut replayed = 0;
        for (lsn, entry) in entries {
            if aborted_lsns.contains(&lsn) {
                continue;
            }

            match entry {
                WalEntry::Upsert { points } => {
                    self.apply_upsert_internal(points)?;
                    replayed += 1;
                }
                WalEntry::Delete { ids } => {
                    self.apply_delete_internal(&ids)?;
                    replayed += 1;
                }
                WalEntry::AddEdge { from_id, edge } => {
                    self.apply_add_edge_internal(from_id, edge)?;
                    replayed += 1;
                }
                WalEntry::RemoveEdge {
                    from_id,
                    to_id,
                    relation_id,
                } => {
                    self.apply_remove_edge_internal(from_id, to_id, relation_id)?;
                    replayed += 1;
                }
                WalEntry::Checkpoint { .. } | WalEntry::Abort { .. } => { /* skip */ }
            }
        }

        Ok(replayed)
    }

    // --- Public Mutation Methods (Async, Durable Mode) ---

    /// Upsert points with WAL durability (async, durable mode)
    ///
    /// Logs to WAL first (for crash recovery), then applies to memory.
    /// Lock is held through both phases for atomicity.
    pub async fn upsert_points_async(&mut self, points: Vec<Point>) -> LatticeResult<UpsertResult> {
        // Validate all points first
        for point in &points {
            if point.vector.len() != self.config.vectors.size {
                return Err(LatticeError::DimensionMismatch {
                    expected: self.config.vectors.size,
                    actual: point.vector.len(),
                });
            }
        }

        if let Some(txn_arc) = self.txn_manager.clone() {
            let mut txn = txn_arc.lock().await;
            let lsn = txn.log_upsert(points.clone()).await?;
            // Hold lock through apply for atomicity
            match self.apply_upsert_internal(points) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    let _ = txn.log_abort(lsn).await;
                    return Err(e);
                }
            }
        }

        self.apply_upsert_internal(points)
    }

    /// Batch upsert multiple individual points with WAL durability (async, durable mode)
    ///
    /// More efficient than calling `upsert_points_async` multiple times because
    /// it batches WAL entries and syncs once at the end.
    ///
    /// Each point is logged as a separate WAL entry for individual recovery.
    pub async fn upsert_batch_async(
        &mut self,
        points_batch: Vec<Vec<Point>>,
    ) -> LatticeResult<Vec<UpsertResult>> {
        // Validate all points first
        for points in &points_batch {
            for point in points {
                if point.vector.len() != self.config.vectors.size {
                    return Err(LatticeError::DimensionMismatch {
                        expected: self.config.vectors.size,
                        actual: point.vector.len(),
                    });
                }
            }
        }

        if let Some(txn_arc) = self.txn_manager.clone() {
            let mut txn = txn_arc.lock().await;
            // Log all operations to WAL without syncing
            let mut lsns = Vec::with_capacity(points_batch.len());
            for points in &points_batch {
                lsns.push(txn.log_upsert_nosync(points.clone()).await?);
            }
            // Single sync at the end
            txn.sync().await?;

            // Apply all to in-memory state (lock held through apply)
            let mut results = Vec::with_capacity(points_batch.len());
            for points in points_batch {
                results.push(self.apply_upsert_internal(points)?);
            }
            return Ok(results);
        }

        let mut results = Vec::with_capacity(points_batch.len());
        for points in points_batch {
            results.push(self.apply_upsert_internal(points)?);
        }
        Ok(results)
    }

    /// Delete points with WAL durability (async, durable mode)
    ///
    /// Returns the number of points actually deleted.
    /// Lock is held through both phases for atomicity.
    pub async fn delete_points_async(&mut self, ids: &[PointId]) -> LatticeResult<usize> {
        if let Some(txn_arc) = self.txn_manager.clone() {
            let mut txn = txn_arc.lock().await;
            let lsn = txn.log_delete(ids.to_vec()).await?;
            match self.apply_delete_internal(ids) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    let _ = txn.log_abort(lsn).await;
                    return Err(e);
                }
            }
        }

        self.apply_delete_internal(ids)
    }

    /// Add a directed edge with WAL durability (async, durable mode)
    ///
    /// Lock is held through both phases for atomicity.
    pub async fn add_edge_async(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: &str,
        weight: f32,
    ) -> LatticeResult<()> {
        let relation_id = self.get_or_create_relation_id(relation);
        let edge = Edge::new(to_id, weight, relation_id);

        if let Some(txn_arc) = self.txn_manager.clone() {
            let mut txn = txn_arc.lock().await;
            let lsn = txn.log_add_edge(from_id, edge.clone()).await?;
            match self.apply_add_edge_internal(from_id, edge) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    let _ = txn.log_abort(lsn).await;
                    return Err(e);
                }
            }
        }

        self.apply_add_edge_internal(from_id, edge)
    }

    /// Remove an edge with WAL durability (async, durable mode)
    ///
    /// Lock is held through both phases for atomicity.
    pub async fn remove_edge_async(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: Option<&str>,
    ) -> LatticeResult<bool> {
        if let Some(rel) = relation {
            if let Some(relation_id) = self.config.relation_id(rel) {
                if let Some(txn_arc) = self.txn_manager.clone() {
                    let mut txn = txn_arc.lock().await;
                    let lsn = txn.log_remove_edge(from_id, to_id, relation_id).await?;
                    match self.apply_remove_edge_internal(from_id, to_id, relation_id) {
                        Ok(result) => return Ok(result),
                        Err(e) => {
                            let _ = txn.log_abort(lsn).await;
                            return Err(e);
                        }
                    }
                }

                return self.apply_remove_edge_internal(from_id, to_id, relation_id);
            }
            Ok(false) // Relation doesn't exist
        } else {
            // For edges without specific relation, find and log the first match
            let removed_relation_id = {
                let mut pts = self.points.write_safe()?;
                let point = pts
                    .get_mut(&from_id)
                    .ok_or(LatticeError::PointNotFound { id: from_id })?;

                if let Some(edges) = &mut point.outgoing_edges {
                    if let Some(idx) = edges.iter().position(|e| e.target_id == to_id) {
                        let edge = edges.remove(idx);
                        Some(edge.relation_id)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }; // pts dropped here before await

            if let Some(rel_id) = removed_relation_id {
                if let Some(txn_arc) = self.txn_manager.clone() {
                    let mut txn = txn_arc.lock().await;
                    txn.log_remove_edge(from_id, to_id, rel_id).await?;
                }
                return Ok(true);
            }
            Ok(false)
        }
    }
}

impl<S> CollectionEngine<S, S>
where
    S: LatticeStorage + Clone + Send + Sync + 'static,
{
    /// Create a durable engine with unified storage for both WAL and data
    ///
    /// This is a convenience method when using the same storage backend for everything.
    pub async fn open_durable_unified(config: CollectionConfig, storage: S) -> LatticeResult<Self> {
        Self::open_durable(config, storage.clone(), storage).await
    }
}

impl<W: LatticeStorage, D: LatticeStorage> CollectionEngine<W, D> {
    /// Get collection configuration
    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    /// Get collection name
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get number of points in the collection
    pub fn point_count(&self) -> usize {
        self.points.read().map(|g| g.len()).unwrap_or(0)
    }

    /// Check if collection is empty
    pub fn is_empty(&self) -> bool {
        self.points.read().map(|g| g.is_empty()).unwrap_or(true)
    }

    /// Get vector dimension
    pub fn vector_dim(&self) -> usize {
        self.config.vectors.size
    }

    /// Get number of points pending indexing
    pub fn pending_count(&self) -> usize {
        self.pending_index.read().map(|g| g.len()).unwrap_or(0)
    }

    // --- Point Operations (Internal) ---

    /// Internal upsert without WAL logging (used for recovery and ephemeral mode)
    ///
    /// Points are stored immediately and indexed asynchronously in background.
    /// If pending buffer exceeds threshold, blocks until indexer catches up.
    fn apply_upsert_internal(&mut self, points: Vec<Point>) -> LatticeResult<UpsertResult> {
        let mut inserted = 0;
        let mut updated = 0;

        // Validate all points first
        for point in &points {
            if point.vector.len() != self.config.vectors.size {
                return Err(LatticeError::DimensionMismatch {
                    expected: self.config.vectors.size,
                    actual: point.vector.len(),
                });
            }
        }

        // Check pending threshold - block if too many pending.
        // Note: there is a benign TOCTOU race between the initial read and the
        // while-loop re-check. The pending count can change after we drop the
        // read lock, which may cause an unnecessary wait or skip. This is harmless:
        // the condvar loop will re-verify the condition, and the timeout prevents hangs.
        {
            let pending = self.pending_index.read_safe()?;
            if pending.len() > PENDING_THRESHOLD {
                drop(pending); // Release read lock

                // Wait for indexer to catch up using condvar (efficient, no CPU spin)
                let (lock, cv) = &*self.backpressure_signal;
                let mut guard = lock.lock_safe()?;
                let mut wait_iterations = 0;
                const MAX_WAIT_ITERATIONS: u32 = BACKPRESSURE_MAX_ITERATIONS;

                while self.pending_index.read_safe()?.len() > PENDING_THRESHOLD / 2 {
                    // Check if indexer has crashed before waiting
                    if !self.indexer.is_running() {
                        // Indexer is dead - proceed anyway since points are still stored
                        // They'll be indexed on restart when index is rebuilt
                        break;
                    }

                    // Prevent infinite wait if indexer is somehow stuck
                    wait_iterations += 1;
                    if wait_iterations >= MAX_WAIT_ITERATIONS {
                        break; // Proceed with upsert anyway
                    }

                    // Wait with timeout to handle edge cases
                    let result = cv
                        .wait_timeout(
                            guard,
                            std::time::Duration::from_millis(BACKPRESSURE_POLL_MS),
                        )
                        .map_err(|e| LatticeError::Internal {
                            code: 50004,
                            message: format!("Condvar wait failed: {}", e),
                        })?;
                    guard = result.0;
                }
            }
        }

        // Pre-parse labels BEFORE acquiring locks (JSON parsing is expensive)
        let points_with_labels: Vec<(Point, Option<Vec<String>>)> = points
            .into_iter()
            .map(|point| {
                let labels = point
                    .payload
                    .get("_labels")
                    .and_then(|bytes| serde_json::from_slice::<Vec<String>>(bytes).ok());
                (point, labels)
            })
            .collect();

        // Store points immediately and queue for background indexing
        {
            let mut pts = self.points.write_safe()?;
            let mut pending = self.pending_index.write_safe()?;
            let mut label_idx = self.label_index.write_safe()?;

            for (point, labels) in points_with_labels {
                let is_update = pts.contains_key(&point.id);

                if is_update {
                    // For updates, remove old labels from label index
                    if let Some(old_point) = pts.get(&point.id) {
                        Self::remove_point_labels_from_index(old_point, &mut label_idx);
                    }
                    // Delete from HNSW index first (sync to ensure consistency)
                    self.indexer.queue_delete(point.id);
                    updated += 1;
                } else {
                    inserted += 1;
                }

                // Update label index with pre-parsed labels (no JSON parsing here)
                if let Some(labels) = labels {
                    for label in labels {
                        label_idx
                            .entry(label)
                            .or_insert_with(FxHashSet::default)
                            .insert(point.id);
                    }
                }

                // Store point (move, no clone — id is Copy)
                let point_id = point.id;
                pts.insert(point_id, point);

                // Add to pending and queue for indexing
                pending.insert(point_id);
                self.indexer.queue_insert(point_id);
            }
        }

        Ok(UpsertResult { inserted, updated })
    }

    /// Remove a point's labels from the label index
    fn remove_point_labels_from_index(
        point: &Point,
        label_idx: &mut HashMap<String, FxHashSet<PointId>>,
    ) {
        if let Some(labels_bytes) = point.payload.get("_labels") {
            if let Ok(labels) = serde_json::from_slice::<Vec<String>>(labels_bytes) {
                for label in labels {
                    if let Some(ids) = label_idx.get_mut(&label) {
                        ids.remove(&point.id);
                    }
                }
            }
        }
    }

    /// Get points by IDs (returns cloned points for thread safety)
    pub fn get_points(&self, ids: &[PointId]) -> LatticeResult<Vec<Option<Point>>> {
        let pts = self.points.read_safe()?;
        Ok(ids.iter().map(|id| pts.get(id).cloned()).collect())
    }

    /// Get a single point by ID (returns cloned point for thread safety)
    pub fn get_point(&self, id: PointId) -> LatticeResult<Option<Point>> {
        Ok(self.points.read_safe()?.get(&id).cloned())
    }

    /// Batch extract specific properties from points without cloning entire Points.
    ///
    /// This is optimized for ORDER BY queries where we only need sort key properties.
    /// Returns `Vec<Vec<Option<Vec<u8>>>>` - outer: per point, inner: per property.
    /// The values are the raw JSON bytes from the payload (caller deserializes).
    ///
    /// Returns None if point doesn't exist, Some(None) if property doesn't exist.
    pub fn batch_extract_properties(
        &self,
        ids: &[PointId],
        properties: &[&str],
    ) -> LatticeResult<Vec<Vec<Option<Vec<u8>>>>> {
        let pts = self.points.read_safe()?;
        Ok(ids
            .iter()
            .map(|id| {
                if let Some(point) = pts.get(id) {
                    properties
                        .iter()
                        .map(|prop| point.payload.get(*prop).cloned())
                        .collect()
                } else {
                    // Point not found - return None for all properties
                    vec![None; properties.len()]
                }
            })
            .collect())
    }

    /// Batch extract a single numeric property as i64, optimized for ORDER BY on integer fields.
    ///
    /// This is highly optimized:
    /// - No byte cloning (parses in-place)
    /// - No CypherValue allocation
    /// - Single lock acquisition
    /// - Fast integer parsing without serde overhead
    ///
    /// Returns i64::MIN for missing points/properties (sorts to bottom for DESC).
    pub fn batch_extract_i64_property(
        &self,
        ids: &[PointId],
        property: &str,
    ) -> LatticeResult<Vec<i64>> {
        // For large N, pre-extract ALL values in BTreeMap order (sequential access),
        // then use O(1) Vec lookup (IDs are typically sequential starting from 0)
        const LARGE_THRESHOLD: usize = 5000;
        // Cap allocation to prevent OOM on malicious/edge-case inputs
        const MAX_DENSE_ALLOCATION: usize = 10_000_000;

        if ids.len() >= LARGE_THRESHOLD && !ids.is_empty() {
            // Check if IDs are roughly sequential (common case after AllNodesScan)
            // Safe: ids is non-empty so min/max will succeed
            let min_id = *ids.iter().min().unwrap_or(&0);
            let max_id = *ids.iter().max().unwrap_or(&0);

            // Use checked arithmetic to prevent overflow when max_id is near u64::MAX
            let id_range = match max_id.checked_sub(min_id).and_then(|r| r.checked_add(1)) {
                Some(r) if r <= MAX_DENSE_ALLOCATION as u64 => r,
                _ => {
                    // Overflow or range too large - fall back to direct lookups
                    return self.batch_extract_i64_fallback(ids, property);
                }
            };

            // Only use dense Vec if IDs are ~dense (not sparse)
            if id_range <= ids.len() as u64 * 2 {
                // Pre-extract all properties in BTreeMap order (sequential, cache-friendly)
                let pts = self.points.read_safe()?;
                let mut values: Vec<i64> = vec![i64::MIN; id_range as usize];

                // Single sequential iteration over BTreeMap
                for (&id, point) in pts.range(min_id..=max_id) {
                    let idx = (id - min_id) as usize;
                    if let Some(bytes) = point.payload.get(property) {
                        if let Some(val) = Self::fast_parse_i64(bytes) {
                            values[idx] = val;
                        }
                    }
                }

                // O(1) lookup for each id
                return Ok(ids
                    .iter()
                    .map(|id| values[(id - min_id) as usize])
                    .collect());
            }
        }

        // Fallback: direct BTreeMap lookups
        self.batch_extract_i64_fallback(ids, property)
    }

    /// Fallback path for batch_extract_i64_property using direct lookups.
    /// Used when dense vector optimization is not applicable.
    fn batch_extract_i64_fallback(
        &self,
        ids: &[PointId],
        property: &str,
    ) -> LatticeResult<Vec<i64>> {
        let pts = self.points.read_safe()?;
        Ok(ids
            .iter()
            .map(|id| {
                pts.get(id)
                    .and_then(|point| point.payload.get(property))
                    .and_then(|bytes| Self::fast_parse_i64(bytes))
                    .unwrap_or(i64::MIN)
            })
            .collect())
    }

    /// Fast parse i64 from JSON bytes without allocation.
    /// Returns None for non-integer values (floats, strings, etc).
    #[inline]
    fn fast_parse_i64(bytes: &[u8]) -> Option<i64> {
        if bytes.is_empty() {
            return None;
        }

        let (start, negative) = if bytes[0] == b'-' {
            if bytes.len() < 2 {
                return None;
            }
            (1, true)
        } else {
            (0, false)
        };

        let mut result: i64 = 0;
        for &b in &bytes[start..] {
            if b.is_ascii_digit() {
                result = result.checked_mul(10)?;
                let digit = (b - b'0') as i64;
                result = if negative {
                    result.checked_sub(digit)?
                } else {
                    result.checked_add(digit)?
                };
            } else {
                // Not a simple integer (float, string, etc.)
                return None;
            }
        }
        Some(result)
    }

    /// Internal delete without WAL logging (used for recovery and ephemeral mode)
    ///
    /// Returns the number of points actually deleted.
    fn apply_delete_internal(&mut self, ids: &[PointId]) -> LatticeResult<usize> {
        let mut deleted = 0;

        let mut pts = self.points.write_safe()?;
        let mut pending = self.pending_index.write_safe()?;
        let mut label_idx = self.label_index.write_safe()?;

        for &id in ids {
            if let Some(point) = pts.remove(&id) {
                // Remove from label index
                Self::remove_point_labels_from_index(&point, &mut label_idx);
                // Queue deletion from HNSW index (background)
                self.indexer.queue_delete(id);
                // Remove from pending if it was there
                pending.remove(&id);
                deleted += 1;
            }
        }

        Ok(deleted)
    }

    /// Check if a point exists
    pub fn point_exists(&self, id: PointId) -> bool {
        self.points
            .read()
            .map(|g| g.contains_key(&id))
            .unwrap_or(false)
    }

    /// Get all point IDs
    pub fn point_ids(&self) -> LatticeResult<Vec<PointId>> {
        Ok(self.points.read_safe()?.keys().copied().collect())
    }

    /// Get point IDs that have a specific label (O(1) lookup via label index)
    pub fn point_ids_by_label(&self, label: &str) -> LatticeResult<Vec<PointId>> {
        let label_idx = self.label_index.read_safe()?;
        Ok(label_idx
            .get(label)
            .map(|ids| ids.iter().copied().collect())
            .unwrap_or_default())
    }

    // --- Search Operations ---

    /// Search for nearest neighbors
    ///
    /// Searches both the HNSW index and pending (unindexed) points.
    /// Pending points are searched with parallel brute-force and merged with HNSW results.
    pub fn search(&self, query: SearchQuery) -> LatticeResult<Vec<SearchResult>> {
        use crate::index::distance::DistanceCalculator;

        // Validate query vector dimension
        if query.vector.len() != self.config.vectors.size {
            return Err(LatticeError::DimensionMismatch {
                expected: self.config.vectors.size,
                actual: query.vector.len(),
            });
        }

        // Use query ef or default from config
        let ef = query.ef.unwrap_or(self.config.hnsw.ef);

        // Search HNSW index
        let mut results = {
            let index = self.index.read_safe()?;
            index.search(&query.vector, query.limit, ef)
        };

        // Search pending points with parallel brute-force
        // Acquire both locks atomically to prevent TOCTOU race where points could be
        // deleted or replaced between reading pending_index and accessing points
        let pending_results: Vec<SearchResult> = {
            let pending = self.pending_index.read_safe()?;
            let pending_ids: Vec<PointId> = pending.iter().copied().collect();

            if pending_ids.is_empty() {
                Vec::new()
            } else {
                let pts = self.points.read_safe()?;
                let distance_calc = DistanceCalculator::new(self.config.vectors.distance);

                // Parallel brute-force on pending points
                pending_ids
                    .par_iter()
                    .filter_map(|&id| {
                        pts.get(&id).map(|point| {
                            let score = distance_calc.calculate(&query.vector, &point.vector);
                            SearchResult {
                                id,
                                score,
                                vector: None,
                                payload: None,
                            }
                        })
                    })
                    .collect()
            }
        };

        if !pending_results.is_empty() {
            // Merge results
            results.extend(pending_results);
            results.sort_by(|a, b| crate::sync::cmp_f32(a.score, b.score));
            results.truncate(query.limit);
        }

        // Apply score threshold if specified
        if let Some(threshold) = query.score_threshold {
            results.retain(|r| r.score <= threshold);
        }

        // Enrich results with vector/payload if requested
        {
            let pts = self.points.read_safe()?;
            for result in &mut results {
                if let Some(point) = pts.get(&result.id) {
                    if query.with_vector {
                        result.vector = Some(point.vector.clone());
                    }
                    if query.with_payload {
                        result.payload = Some(self.payload_to_json(&point.payload));
                    }
                }
            }
        }

        Ok(results)
    }

    /// Batch search for multiple queries (parallel processing)
    ///
    /// More efficient than calling `search` multiple times for many queries.
    /// Uses rayon for parallel query processing on native platforms.
    pub fn search_batch(&self, queries: Vec<SearchQuery>) -> LatticeResult<Vec<Vec<SearchResult>>> {
        if queries.is_empty() {
            return Ok(vec![]);
        }

        // Validate all query dimensions upfront
        for query in &queries {
            if query.vector.len() != self.config.vectors.size {
                return Err(LatticeError::DimensionMismatch {
                    expected: self.config.vectors.size,
                    actual: query.vector.len(),
                });
            }
        }

        // Extract vector references for batch search (avoids cloning 4MB+ for 1000 queries × 512-dim)
        let query_vectors: Vec<&[f32]> = queries.iter().map(|q| q.vector.as_slice()).collect();
        let first_query = &queries[0];
        let ef = first_query.ef.unwrap_or(self.config.hnsw.ef);
        let limit = first_query.limit;

        // Batch search on HNSW index
        let batch_results = {
            let index = self.index.read_safe()?;
            index.search_batch(&query_vectors, limit, ef)
        };

        // Handle pending points for each query (parallel)
        // Acquire both locks atomically to prevent TOCTOU race
        let all_results: Vec<Vec<SearchResult>> = {
            let pending = self.pending_index.read_safe()?;
            let pending_ids: Vec<PointId> = pending.iter().copied().collect();

            if pending_ids.is_empty() {
                batch_results
            } else {
                use crate::index::distance::DistanceCalculator;
                let pts = self.points.read_safe()?;
                let distance_calc = DistanceCalculator::new(self.config.vectors.distance);

                // Process each query with pending points
                let mut results: Vec<Vec<SearchResult>> = batch_results;
                for (i, query) in queries.iter().enumerate() {
                    let pending_results: Vec<SearchResult> = pending_ids
                        .par_iter()
                        .filter_map(|&id| {
                            pts.get(&id).map(|point| {
                                let score = distance_calc.calculate(&query.vector, &point.vector);
                                SearchResult {
                                    id,
                                    score,
                                    vector: None,
                                    payload: None,
                                }
                            })
                        })
                        .collect();

                    results[i].extend(pending_results);
                    results[i].sort_by(|a, b| crate::sync::cmp_f32(a.score, b.score));
                    results[i].truncate(query.limit);
                }
                results
            }
        };

        Ok(all_results)
    }

    /// Scroll through points (paginated retrieval)
    pub fn scroll(&self, query: ScrollQuery) -> LatticeResult<ScrollResult> {
        let pts = self.points.read_safe()?;

        // Use BTreeMap's range for efficient iteration from offset
        let iter: Box<dyn Iterator<Item = (&PointId, &Point)>> = match query.offset {
            Some(offset) => Box::new(pts.range((Bound::Excluded(offset), Bound::Unbounded))),
            None => Box::new(pts.iter()),
        };

        // Take limit + 1 to check if there are more results
        let page: Vec<_> = iter.take(query.limit + 1).collect();
        let has_more = page.len() > query.limit;

        // Build scroll points (only up to limit)
        let result_count = if has_more { query.limit } else { page.len() };
        let points: Vec<ScrollPoint> = page[..result_count]
            .iter()
            .map(|(&id, point)| ScrollPoint {
                id,
                vector: if query.with_vector {
                    Some(point.vector.clone())
                } else {
                    None
                },
                payload: if query.with_payload {
                    Some(self.payload_to_json(&point.payload))
                } else {
                    None
                },
            })
            .collect();

        let next_offset = if has_more {
            points.last().map(|p| p.id)
        } else {
            None
        };

        Ok(ScrollResult {
            points,
            next_offset,
        })
    }

    // --- Graph Operations (Internal) ---

    /// Internal add_edge without WAL logging (takes pre-created Edge)
    fn apply_add_edge_internal(&mut self, from_id: PointId, edge: Edge) -> LatticeResult<()> {
        let mut pts = self.points.write_safe()?;

        // Validate both points exist
        if !pts.contains_key(&from_id) {
            return Err(LatticeError::PointNotFound { id: from_id });
        }
        if !pts.contains_key(&edge.target_id) {
            return Err(LatticeError::PointNotFound { id: edge.target_id });
        }

        // Add edge to source point
        let point = pts
            .get_mut(&from_id)
            .ok_or(LatticeError::PointNotFound { id: from_id })?;

        match &mut point.outgoing_edges {
            Some(edges) => {
                // Check for duplicate
                if !edges
                    .iter()
                    .any(|e| e.target_id == edge.target_id && e.relation_id == edge.relation_id)
                {
                    edges.push(edge);
                }
            }
            None => {
                point.outgoing_edges = Some(SmallVec::from_vec(vec![edge]));
            }
        }

        Ok(())
    }

    /// Internal remove_edge without WAL logging (by relation_id)
    fn apply_remove_edge_internal(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation_id: u16,
    ) -> LatticeResult<bool> {
        let mut pts = self.points.write_safe()?;

        let point = pts
            .get_mut(&from_id)
            .ok_or(LatticeError::PointNotFound { id: from_id })?;

        let removed = if let Some(edges) = &mut point.outgoing_edges {
            let original_len = edges.len();
            edges.retain(|e| !(e.target_id == to_id && e.relation_id == relation_id));
            edges.len() < original_len
        } else {
            false
        };

        Ok(removed)
    }

    /// Get outgoing edges from a point
    pub fn get_edges(&self, point_id: PointId) -> LatticeResult<Vec<EdgeInfo>> {
        let pts = self.points.read_safe()?;

        let point = pts
            .get(&point_id)
            .ok_or(LatticeError::PointNotFound { id: point_id })?;

        let edges = point
            .outgoing_edges
            .as_ref()
            .map(|edges| {
                edges
                    .iter()
                    .map(|e| EdgeInfo {
                        target_id: e.target_id,
                        weight: e.weight,
                        relation: self
                            .config
                            .relation_name(e.relation_id)
                            .unwrap_or("unknown")
                            .to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(edges)
    }

    /// Traverse the graph from a starting point
    pub fn traverse(
        &self,
        start_id: PointId,
        max_depth: usize,
        relations: Option<&[&str]>,
    ) -> LatticeResult<TraversalResult> {
        let pts = self.points.read_safe()?;

        if !pts.contains_key(&start_id) {
            return Err(LatticeError::PointNotFound { id: start_id });
        }

        // Convert relation names to IDs
        let relation_ids: Option<Vec<u16>> = relations.map(|rels| {
            rels.iter()
                .filter_map(|r| self.config.relation_id(r))
                .collect()
        });

        let mut visited: HashMap<PointId, usize> = HashMap::new(); // id -> depth
        let mut paths: Vec<TraversalPath> = Vec::new();
        // Use VecDeque for proper BFS traversal (pop_front instead of pop)
        let mut queue: VecDeque<(PointId, usize, Vec<PointId>)> =
            VecDeque::from([(start_id, 0, vec![start_id])]);

        visited.insert(start_id, 0);

        while let Some((current_id, depth, path)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            if let Some(point) = pts.get(&current_id) {
                if let Some(edges) = &point.outgoing_edges {
                    for edge in edges.iter() {
                        // Filter by relation if specified
                        if let Some(ref rel_ids) = relation_ids {
                            if !rel_ids.contains(&edge.relation_id) {
                                continue;
                            }
                        }

                        let target_id = edge.target_id;
                        let new_depth = depth + 1;

                        // Single lookup: only visit if not visited or found at shorter depth
                        let should_visit = visited
                            .get(&target_id)
                            .map_or(true, |&prev_depth| prev_depth > new_depth);

                        if should_visit {
                            visited.insert(target_id, new_depth);

                            // Build path: one clone for results, one for queue
                            // Optimization: extend in place rather than clone+push
                            let mut result_path = path.clone();
                            result_path.push(target_id);

                            // Clone for queue only - moves are free, clones are O(depth)
                            let queue_path = result_path.clone();

                            paths.push(TraversalPath {
                                target_id,
                                depth: new_depth,
                                path: result_path, // Move, don't clone
                                weight: edge.weight,
                                relation_id: edge.relation_id,
                            });

                            queue.push_back((target_id, new_depth, queue_path));
                        }
                    }
                }
            }
        }

        Ok(TraversalResult {
            start_id,
            max_depth,
            nodes_visited: visited.len(),
            paths,
        })
    }

    // --- Helper Methods ---

    /// Convert payload HashMap to JSON Value
    fn payload_to_json(&self, payload: &HashMap<String, Vec<u8>>) -> serde_json::Value {
        let map: serde_json::Map<String, serde_json::Value> = payload
            .iter()
            .filter_map(|(k, v)| serde_json::from_slice(v).ok().map(|val| (k.clone(), val)))
            .collect();
        serde_json::Value::Object(map)
    }

    /// Get or create a relation ID
    fn get_or_create_relation_id(&mut self, relation: &str) -> u16 {
        if let Some(id) = self.config.relation_id(relation) {
            id
        } else {
            let new_id = self.config.relations.len() as u16;
            self.config.relations.insert(relation.to_string(), new_id);
            new_id
        }
    }

    /// Serialize the collection to bytes (for persistence)
    ///
    /// Format: `[config_len:u32][config:JSON][padding][points:rkyv]`
    /// Padding ensures rkyv data starts at 16-byte alignment.
    pub fn to_bytes(&self) -> LatticeResult<Vec<u8>> {
        // Serialize config as JSON (small, schema-flexible)
        let config_bytes =
            serde_json::to_vec(&self.config).map_err(|e| LatticeError::Serialization {
                message: e.to_string(),
            })?;

        // Serialize points with rkyv (zero-copy, fast)
        let pts = self.points.read_safe()?;
        let points_vec: Vec<Point> = pts.values().cloned().collect();
        drop(pts); // Release lock before serialization

        let points_bytes =
            rkyv::to_bytes::<RkyvError>(&points_vec).map_err(|e| LatticeError::Serialization {
                message: e.to_string(),
            })?;

        // Calculate padding to align rkyv data to 16 bytes
        let header_size = 4 + config_bytes.len() + 4; // config_len + config + points_len
        let padding = (16 - (header_size % 16)) % 16;

        // Build result: [config_len][config][points_len][padding][points]
        let mut result = Vec::with_capacity(header_size + padding + points_bytes.len());
        result.extend_from_slice(&(config_bytes.len() as u32).to_le_bytes());
        result.extend_from_slice(&config_bytes);
        result.extend_from_slice(&(points_bytes.len() as u32).to_le_bytes());
        result.extend(std::iter::repeat(0u8).take(padding)); // Alignment padding
        result.extend_from_slice(&points_bytes);

        Ok(result)
    }

    /// Wait for all pending indexing to complete
    ///
    /// This is useful for tests or when you need consistent search results.
    /// Uses condvar notification to avoid CPU-wasting spin-wait.
    pub fn flush_pending(&self) -> LatticeResult<()> {
        let (lock, cv) = &*self.backpressure_signal;

        // Wait on condvar until pending is empty
        // The async indexer notifies after each batch, so we'll wake up periodically
        let mut guard = lock
            .lock()
            .map_err(|_| crate::error::LatticeError::Internal {
                code: 50001,
                message: "backpressure_signal mutex poisoned".to_string(),
            })?;

        while !self.pending_index.read_safe()?.is_empty() {
            // Wait with timeout to handle edge cases (indexer shutdown, etc.)
            let result = cv.wait_timeout(guard, std::time::Duration::from_millis(100));
            guard = result
                .map_err(|_| crate::error::LatticeError::Internal {
                    code: 50001,
                    message: "backpressure_signal mutex poisoned".to_string(),
                })?
                .0;
        }
        Ok(())
    }
}

impl<W: LatticeStorage, D: LatticeStorage> crate::engine::EngineOps for CollectionEngine<W, D> {
    fn get_point(&self, id: PointId) -> LatticeResult<Option<Point>> {
        self.get_point(id)
    }

    fn point_ids(&self) -> LatticeResult<Vec<PointId>> {
        self.point_ids()
    }

    fn point_ids_by_label(&self, label: &str) -> LatticeResult<Vec<PointId>> {
        self.point_ids_by_label(label)
    }

    fn get_points(&self, ids: &[PointId]) -> LatticeResult<Vec<Option<Point>>> {
        self.get_points(ids)
    }

    fn vector_dim(&self) -> usize {
        self.vector_dim()
    }

    fn get_edges(&self, point_id: PointId) -> LatticeResult<Vec<EdgeInfo>> {
        self.get_edges(point_id)
    }

    fn delete_points(&mut self, ids: &[PointId]) -> LatticeResult<usize> {
        self.apply_delete_internal(ids)
    }

    fn batch_extract_properties(
        &self,
        ids: &[PointId],
        properties: &[&str],
    ) -> LatticeResult<Vec<Vec<Option<Vec<u8>>>>> {
        self.batch_extract_properties(ids, properties)
    }

    fn batch_extract_i64_property(
        &self,
        ids: &[PointId],
        property: &str,
    ) -> LatticeResult<Vec<i64>> {
        self.batch_extract_i64_property(ids, property)
    }

    fn traverse(
        &self,
        start_id: PointId,
        max_depth: usize,
        relations: Option<&[&str]>,
    ) -> LatticeResult<TraversalResult> {
        self.traverse(start_id, max_depth, relations)
    }

    fn upsert_points(&mut self, points: Vec<Point>) -> LatticeResult<UpsertResult> {
        self.apply_upsert_internal(points)
    }

    fn add_edge(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: &str,
        weight: f32,
    ) -> LatticeResult<()> {
        let relation_id = self.get_or_create_relation_id(relation);
        let edge = Edge::new(to_id, weight, relation_id);
        self.apply_add_edge_internal(from_id, edge)
    }

    fn remove_edge(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: Option<&str>,
    ) -> LatticeResult<bool> {
        if let Some(rel) = relation {
            if let Some(relation_id) = self.config.relation_id(rel) {
                return self.apply_remove_edge_internal(from_id, to_id, relation_id);
            }
            Ok(false)
        } else {
            let mut pts = self.points.write_safe()?;
            let point = pts
                .get_mut(&from_id)
                .ok_or(LatticeError::PointNotFound { id: from_id })?;

            if let Some(edges) = &mut point.outgoing_edges {
                let original_len = edges.len();
                edges.retain(|e| e.target_id != to_id);
                Ok(edges.len() < original_len)
            } else {
                Ok(false)
            }
        }
    }
}
