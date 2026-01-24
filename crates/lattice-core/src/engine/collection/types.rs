//! Collection types for point operations, graph traversal, and edge information

use crate::types::point::PointId;

/// Result of upsert operation
#[derive(Debug, Clone)]
pub struct UpsertResult {
    pub inserted: usize,
    pub updated: usize,
}

/// Edge information for API responses
#[derive(Debug, Clone)]
pub struct EdgeInfo {
    pub target_id: PointId,
    pub weight: f32,
    pub relation: String,
}

/// Single path in traversal result
#[derive(Debug, Clone)]
pub struct TraversalPath {
    pub target_id: PointId,
    pub depth: usize,
    pub path: Vec<PointId>,
    pub weight: f32,
    /// The relation type ID of the edge leading to this node
    pub relation_id: u16,
}

/// Result of graph traversal
#[derive(Debug, Clone)]
pub struct TraversalResult {
    pub start_id: PointId,
    pub max_depth: usize,
    pub nodes_visited: usize,
    pub paths: Vec<TraversalPath>,
}
