//! Type-erased engine wrapper for mixed ephemeral/durable collections
//!
//! Wraps both `CollectionEngine<NoStorage, NoStorage>` (ephemeral) and
//! `CollectionEngine<DiskStorage, DiskStorage>` (durable) behind a single
//! enum so the server can store heterogeneous collections in one map.

use lattice_core::engine::collection::{
    CollectionEngineBuilder, EdgeInfo, NoStorage, TraversalResult, UpsertResult,
};
use lattice_core::{
    CollectionConfig, CollectionEngine, EngineOps, LatticeResult, Point, PointId, ScrollQuery,
    ScrollResult, SearchQuery, SearchResult,
};
use lattice_storage::DiskStorage;

/// Engine wrapper supporting both ephemeral and durable storage backends
pub enum AnyEngine {
    /// In-memory only, no persistence (fast, data lost on restart)
    Ephemeral(CollectionEngine<NoStorage, NoStorage>),
    /// Disk-backed with WAL for ACID guarantees
    Durable(CollectionEngine<DiskStorage, DiskStorage>),
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl AnyEngine {
    /// Create an ephemeral (in-memory) engine
    pub fn new_ephemeral(config: CollectionConfig) -> LatticeResult<Self> {
        Ok(Self::Ephemeral(CollectionEngine::new(config)?))
    }

    /// Open a durable engine backed by DiskStorage
    pub async fn open_durable(
        config: CollectionConfig,
        wal_storage: DiskStorage,
        data_storage: DiskStorage,
    ) -> LatticeResult<Self> {
        let engine = CollectionEngineBuilder::new(config)
            .with_wal(wal_storage)
            .with_data(data_storage)
            .open()
            .await?;
        Ok(Self::Durable(engine))
    }

    /// Wrap an already-built ephemeral engine (used by import)
    pub fn from_ephemeral(engine: CollectionEngine<NoStorage, NoStorage>) -> Self {
        Self::Ephemeral(engine)
    }
}

// ---------------------------------------------------------------------------
// Sync read delegations (used by handlers that only need &self)
// ---------------------------------------------------------------------------

impl AnyEngine {
    pub fn config(&self) -> &CollectionConfig {
        match self {
            Self::Ephemeral(e) => e.config(),
            Self::Durable(e) => e.config(),
        }
    }

    pub fn point_count(&self) -> usize {
        match self {
            Self::Ephemeral(e) => e.point_count(),
            Self::Durable(e) => e.point_count(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Ephemeral(e) => e.is_empty(),
            Self::Durable(e) => e.is_empty(),
        }
    }

    pub fn vector_dim(&self) -> usize {
        match self {
            Self::Ephemeral(e) => e.vector_dim(),
            Self::Durable(e) => e.vector_dim(),
        }
    }

    pub fn pending_count(&self) -> usize {
        match self {
            Self::Ephemeral(e) => e.pending_count(),
            Self::Durable(e) => e.pending_count(),
        }
    }

    pub fn search(&self, query: SearchQuery) -> LatticeResult<Vec<SearchResult>> {
        match self {
            Self::Ephemeral(e) => e.search(query),
            Self::Durable(e) => e.search(query),
        }
    }

    pub fn search_batch(
        &self,
        queries: Vec<SearchQuery>,
    ) -> LatticeResult<Vec<Vec<SearchResult>>> {
        match self {
            Self::Ephemeral(e) => e.search_batch(queries),
            Self::Durable(e) => e.search_batch(queries),
        }
    }

    pub fn scroll(&self, query: ScrollQuery) -> LatticeResult<ScrollResult> {
        match self {
            Self::Ephemeral(e) => e.scroll(query),
            Self::Durable(e) => e.scroll(query),
        }
    }

    pub fn traverse(
        &self,
        start_id: PointId,
        max_depth: usize,
        relations: Option<&[&str]>,
    ) -> LatticeResult<TraversalResult> {
        match self {
            Self::Ephemeral(e) => e.traverse(start_id, max_depth, relations),
            Self::Durable(e) => e.traverse(start_id, max_depth, relations),
        }
    }

    pub fn get_points(&self, ids: &[PointId]) -> LatticeResult<Vec<Option<Point>>> {
        match self {
            Self::Ephemeral(e) => e.get_points(ids),
            Self::Durable(e) => e.get_points(ids),
        }
    }

    pub fn get_point(&self, id: PointId) -> LatticeResult<Option<Point>> {
        match self {
            Self::Ephemeral(e) => e.get_point(id),
            Self::Durable(e) => e.get_point(id),
        }
    }

    pub fn point_ids(&self) -> LatticeResult<Vec<PointId>> {
        match self {
            Self::Ephemeral(e) => e.point_ids(),
            Self::Durable(e) => e.point_ids(),
        }
    }

    pub fn to_bytes(&self) -> LatticeResult<Vec<u8>> {
        match self {
            Self::Ephemeral(e) => e.to_bytes(),
            Self::Durable(e) => e.to_bytes(),
        }
    }

    pub fn get_edges(&self, point_id: PointId) -> LatticeResult<Vec<EdgeInfo>> {
        match self {
            Self::Ephemeral(e) => e.get_edges(point_id),
            Self::Durable(e) => e.get_edges(point_id),
        }
    }

    pub fn point_exists(&self, id: PointId) -> bool {
        match self {
            Self::Ephemeral(e) => e.point_exists(id),
            Self::Durable(e) => e.point_exists(id),
        }
    }
}

// ---------------------------------------------------------------------------
// Async write delegations
// Ephemeral uses sync methods; Durable uses *_async variants.
// ---------------------------------------------------------------------------

impl AnyEngine {
    pub async fn upsert_points_async(
        &mut self,
        points: Vec<Point>,
    ) -> LatticeResult<UpsertResult> {
        match self {
            Self::Ephemeral(e) => e.upsert_points(points),
            Self::Durable(e) => e.upsert_points_async(points).await,
        }
    }

    pub async fn delete_points_async(
        &mut self,
        ids: &[PointId],
    ) -> LatticeResult<usize> {
        match self {
            Self::Ephemeral(e) => e.delete_points(ids),
            Self::Durable(e) => e.delete_points_async(ids).await,
        }
    }

    pub async fn add_edge_async(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: &str,
        weight: f32,
    ) -> LatticeResult<()> {
        match self {
            Self::Ephemeral(e) => e.add_edge(from_id, to_id, relation, weight),
            Self::Durable(e) => e.add_edge_async(from_id, to_id, relation, weight).await,
        }
    }

    pub async fn remove_edge_async(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: Option<&str>,
    ) -> LatticeResult<bool> {
        match self {
            Self::Ephemeral(e) => e.remove_edge(from_id, to_id, relation),
            Self::Durable(e) => e.remove_edge_async(from_id, to_id, relation).await,
        }
    }
}

// ---------------------------------------------------------------------------
// EngineOps impl (sync trait used by Cypher executor via &mut dyn EngineOps)
// ---------------------------------------------------------------------------

impl EngineOps for AnyEngine {
    fn get_point(&self, id: PointId) -> LatticeResult<Option<Point>> {
        match self {
            Self::Ephemeral(e) => EngineOps::get_point(e, id),
            Self::Durable(e) => EngineOps::get_point(e, id),
        }
    }

    fn point_ids(&self) -> LatticeResult<Vec<PointId>> {
        match self {
            Self::Ephemeral(e) => EngineOps::point_ids(e),
            Self::Durable(e) => EngineOps::point_ids(e),
        }
    }

    fn point_ids_by_label(&self, label: &str) -> LatticeResult<Vec<PointId>> {
        match self {
            Self::Ephemeral(e) => EngineOps::point_ids_by_label(e, label),
            Self::Durable(e) => EngineOps::point_ids_by_label(e, label),
        }
    }

    fn get_points(&self, ids: &[PointId]) -> LatticeResult<Vec<Option<Point>>> {
        match self {
            Self::Ephemeral(e) => EngineOps::get_points(e, ids),
            Self::Durable(e) => EngineOps::get_points(e, ids),
        }
    }

    fn vector_dim(&self) -> usize {
        match self {
            Self::Ephemeral(e) => EngineOps::vector_dim(e),
            Self::Durable(e) => EngineOps::vector_dim(e),
        }
    }

    fn get_edges(&self, point_id: PointId) -> LatticeResult<Vec<EdgeInfo>> {
        match self {
            Self::Ephemeral(e) => EngineOps::get_edges(e, point_id),
            Self::Durable(e) => EngineOps::get_edges(e, point_id),
        }
    }

    fn delete_points(&mut self, ids: &[PointId]) -> LatticeResult<usize> {
        match self {
            Self::Ephemeral(e) => EngineOps::delete_points(e, ids),
            Self::Durable(e) => EngineOps::delete_points(e, ids),
        }
    }

    fn batch_extract_properties(
        &self,
        ids: &[PointId],
        properties: &[&str],
    ) -> LatticeResult<Vec<Vec<Option<Vec<u8>>>>> {
        match self {
            Self::Ephemeral(e) => EngineOps::batch_extract_properties(e, ids, properties),
            Self::Durable(e) => EngineOps::batch_extract_properties(e, ids, properties),
        }
    }

    fn batch_extract_i64_property(
        &self,
        ids: &[PointId],
        property: &str,
    ) -> LatticeResult<Vec<i64>> {
        match self {
            Self::Ephemeral(e) => EngineOps::batch_extract_i64_property(e, ids, property),
            Self::Durable(e) => EngineOps::batch_extract_i64_property(e, ids, property),
        }
    }

    fn traverse(
        &self,
        start_id: PointId,
        max_depth: usize,
        relations: Option<&[&str]>,
    ) -> LatticeResult<TraversalResult> {
        match self {
            Self::Ephemeral(e) => EngineOps::traverse(e, start_id, max_depth, relations),
            Self::Durable(e) => EngineOps::traverse(e, start_id, max_depth, relations),
        }
    }

    fn upsert_points(&mut self, points: Vec<Point>) -> LatticeResult<UpsertResult> {
        match self {
            Self::Ephemeral(e) => EngineOps::upsert_points(e, points),
            Self::Durable(e) => EngineOps::upsert_points(e, points),
        }
    }

    fn add_edge(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: &str,
        weight: f32,
    ) -> LatticeResult<()> {
        match self {
            Self::Ephemeral(e) => EngineOps::add_edge(e, from_id, to_id, relation, weight),
            Self::Durable(e) => EngineOps::add_edge(e, from_id, to_id, relation, weight),
        }
    }

    fn remove_edge(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: Option<&str>,
    ) -> LatticeResult<bool> {
        match self {
            Self::Ephemeral(e) => EngineOps::remove_edge(e, from_id, to_id, relation),
            Self::Durable(e) => EngineOps::remove_edge(e, from_id, to_id, relation),
        }
    }
}
