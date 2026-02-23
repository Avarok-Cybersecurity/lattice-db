//! Engine module - high-level APIs for collection and point management
//!
//! This module provides the main entry points for LatticeDB operations:
//! - `CollectionEngine`: Manages a single collection (points, search, graph)
//! - `LatticeEngine`: Manages multiple collections
//! - `EngineOps`: Trait for engine operations used by the Cypher executor

pub mod collection;

// Native-only: Async indexer for background HNSW updates
#[cfg(not(target_arch = "wasm32"))]
pub mod async_indexer;

pub use collection::{CollectionEngine, EdgeInfo, TraversalPath, TraversalResult, UpsertResult};

use crate::error::LatticeResult;
use crate::types::point::{Point, PointId};

/// Trait for engine operations used by the Cypher executor.
///
/// Enables the Cypher system to work with any storage backend
/// via dynamic dispatch (`&mut dyn EngineOps`), avoiding the need
/// to propagate storage type parameters through the entire query pipeline.
pub trait EngineOps {
    /// Get a single point by ID
    fn get_point(&self, id: PointId) -> LatticeResult<Option<Point>>;

    /// Get all point IDs in the collection
    fn point_ids(&self) -> LatticeResult<Vec<PointId>>;

    /// Get point IDs filtered by label
    fn point_ids_by_label(&self, label: &str) -> LatticeResult<Vec<PointId>>;

    /// Get multiple points by IDs
    fn get_points(&self, ids: &[PointId]) -> LatticeResult<Vec<Option<Point>>>;

    /// Get the vector dimensionality for this collection
    fn vector_dim(&self) -> usize;

    /// Get outgoing edges from a point
    fn get_edges(&self, point_id: PointId) -> LatticeResult<Vec<EdgeInfo>>;

    /// Delete points by IDs, returning count of actually deleted points
    fn delete_points(&mut self, ids: &[PointId]) -> LatticeResult<usize>;

    /// Batch extract raw property bytes from points (for ORDER BY)
    fn batch_extract_properties(
        &self,
        ids: &[PointId],
        properties: &[&str],
    ) -> LatticeResult<Vec<Vec<Option<Vec<u8>>>>>;

    /// Batch extract a single i64 property (optimized for integer ORDER BY)
    fn batch_extract_i64_property(
        &self,
        ids: &[PointId],
        property: &str,
    ) -> LatticeResult<Vec<i64>>;

    /// Traverse the graph from a starting point (BFS)
    fn traverse(
        &self,
        start_id: PointId,
        max_depth: usize,
        relations: Option<&[&str]>,
    ) -> LatticeResult<TraversalResult>;

    /// Upsert (insert or update) points into the collection
    fn upsert_points(&mut self, points: Vec<Point>) -> LatticeResult<UpsertResult>;

    /// Add a directed edge between two points
    fn add_edge(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: &str,
        weight: f32,
    ) -> LatticeResult<()>;

    /// Remove an edge between two points
    fn remove_edge(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: Option<&str>,
    ) -> LatticeResult<bool>;
}
