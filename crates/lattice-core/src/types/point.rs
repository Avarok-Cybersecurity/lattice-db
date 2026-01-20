//! Point and Edge data structures with zero-copy serialization
//!
//! The `Point` struct is the fundamental storage unit in LatticeDB.
//! It combines vector data, payload metadata, and optional graph connectivity.

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;

/// Unique identifier for a point
///
/// UUIDs are hashed to u64 internally for performance.
/// This provides sufficient uniqueness for most use cases while
/// enabling efficient storage and comparison.
pub type PointId = u64;

/// High-dimensional vector data
///
/// Stored as f32 for balance between precision and memory efficiency.
/// Future: Int8 quantization for 4x memory reduction.
pub type Vector = Vec<f32>;

/// Edge in the graph - connects points with weighted relations
///
/// # Zero-Copy
///
/// Uses `#[repr(C)]` for consistent memory layout across WASM boundaries.
/// The `rkyv` derives enable zero-copy deserialization.
#[derive(Archive, RkyvDeserialize, RkyvSerialize, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[rkyv(compare(PartialEq))]
#[repr(C)]
pub struct Edge {
    /// Target point ID
    pub target_id: PointId,
    /// Edge weight (e.g., similarity score)
    pub weight: f32,
    /// Relation type ID - maps to string in collection metadata
    ///
    /// Using u16 instead of String for:
    /// - Memory efficiency (2 bytes vs 24+ bytes)
    /// - Cache-friendly iteration
    /// - Zero-copy compatibility
    pub relation_id: u16,
}

impl Edge {
    /// Create a new edge
    pub fn new(target_id: PointId, weight: f32, relation_id: u16) -> Self {
        Self {
            target_id,
            weight,
            relation_id,
        }
    }
}

/// The fundamental storage unit - combines vector, payload, and graph edges
///
/// # SSOT
///
/// This is THE authoritative representation of a point.
/// All other representations (API DTOs, archived forms) derive from this.
///
/// # Zero-Copy
///
/// Uses `#[repr(C)]` for consistent memory layout.
/// The `rkyv` derives enable guaranteed zero-copy deserialization.
///
/// # Graph Connectivity
///
/// The `outgoing_edges` field is optional:
/// - `None`: Pure vector point (VectorDB mode)
/// - `Some`: Graph-connected point (GraphDB mode)
///
/// `SmallVec<[Edge; 4]>` optimization:
/// - First 4 edges stored inline (no heap allocation)
/// - Avoids pointer chasing for typical cases
/// - Seamlessly grows to heap for highly connected nodes
#[derive(Archive, RkyvDeserialize, RkyvSerialize, Serialize, Deserialize, Debug, Clone)]
#[repr(C)]
pub struct Point {
    /// Unique identifier
    pub id: PointId,

    /// High-dimensional vector data
    pub vector: Vector,

    /// JSON-like payload for filtering
    ///
    /// Stored as raw bytes (serialized JSON) to delay parsing until needed.
    /// Keys are field names, values are JSON-encoded bytes.
    pub payload: HashMap<String, Vec<u8>>,

    /// Optional graph connectivity
    ///
    /// If `None`, this is a pure VectorDB node.
    /// If `Some`, contains outgoing edges to other points.
    pub outgoing_edges: Option<SmallVec<[Edge; 4]>>,
}

impl Point {
    /// Create a new vector-only point (no graph edges)
    pub fn new_vector(id: PointId, vector: Vector) -> Self {
        Self {
            id,
            vector,
            payload: HashMap::new(),
            outgoing_edges: None,
        }
    }

    /// Create a point with graph connectivity
    pub fn new_graph(id: PointId, vector: Vector, edges: SmallVec<[Edge; 4]>) -> Self {
        Self {
            id,
            vector,
            payload: HashMap::new(),
            outgoing_edges: Some(edges),
        }
    }

    /// Create a point with payload
    pub fn with_payload(mut self, payload: HashMap<String, Vec<u8>>) -> Self {
        self.payload = payload;
        self
    }

    /// Add a payload field
    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<Vec<u8>>) -> Self {
        self.payload.insert(key.into(), value.into());
        self
    }

    /// Get vector dimensionality
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Check if point has graph connectivity
    pub fn has_edges(&self) -> bool {
        self.outgoing_edges.is_some()
    }

    /// Get outgoing edge count
    pub fn edge_count(&self) -> usize {
        self.outgoing_edges.as_ref().map_or(0, |e| e.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkyv::rancor::Error;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn test_edge_creation() {
        let edge = Edge::new(42, 0.95, 1);
        assert_eq!(edge.target_id, 42);
        assert_eq!(edge.weight, 0.95);
        assert_eq!(edge.relation_id, 1);
    }

    #[test]
    fn test_point_new_vector() {
        let point = Point::new_vector(1, vec![0.1, 0.2, 0.3]);
        assert_eq!(point.id, 1);
        assert_eq!(point.vector, vec![0.1, 0.2, 0.3]);
        assert!(point.payload.is_empty());
        assert!(point.outgoing_edges.is_none());
        assert!(!point.has_edges());
        assert_eq!(point.edge_count(), 0);
        assert_eq!(point.dim(), 3);
    }

    #[test]
    fn test_point_new_graph() {
        let edges = SmallVec::from_vec(vec![
            Edge::new(2, 0.9, 0),
            Edge::new(3, 0.8, 1),
        ]);
        let point = Point::new_graph(1, vec![0.1, 0.2], edges);
        assert!(point.has_edges());
        assert_eq!(point.edge_count(), 2);
    }

    #[test]
    fn test_point_with_payload() {
        let point = Point::new_vector(1, vec![0.1])
            .with_field("name", b"test".to_vec());
        assert_eq!(point.payload.get("name"), Some(&b"test".to_vec()));
    }

    #[test]
    fn test_edge_rkyv_roundtrip() {
        let edge = Edge::new(123, 0.75, 5);

        // Serialize
        let bytes = rkyv::to_bytes::<Error>(&edge).expect("serialize edge");

        // Deserialize (zero-copy access) - use unsafe for test since we control serialization
        let archived = unsafe { rkyv::access_unchecked::<ArchivedEdge>(&bytes) };

        assert_eq!(archived.target_id, 123);
        assert_eq!(archived.weight, 0.75);
        assert_eq!(archived.relation_id, 5);
    }

    #[test]
    fn test_point_rkyv_roundtrip() {
        let mut payload = HashMap::new();
        payload.insert("category".to_string(), b"test".to_vec());

        let edges = SmallVec::from_vec(vec![
            Edge::new(10, 0.9, 0),
            Edge::new(20, 0.8, 1),
        ]);

        let point = Point {
            id: 42,
            vector: vec![0.1, 0.2, 0.3, 0.4],
            payload,
            outgoing_edges: Some(edges),
        };

        // Serialize
        let bytes = rkyv::to_bytes::<Error>(&point).expect("serialize point");

        // Deserialize (zero-copy access) - use unsafe for test since we control serialization
        let archived = unsafe { rkyv::access_unchecked::<ArchivedPoint>(&bytes) };

        assert_eq!(archived.id, 42);
        assert_eq!(archived.vector.len(), 4);
        assert_eq!(archived.vector[0], 0.1);
        assert!(archived.payload.contains_key("category"));
        assert!(archived.outgoing_edges.is_some());

        let edges = archived.outgoing_edges.as_ref().unwrap();
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].target_id, 10);
    }

    #[test]
    fn test_edge_rkyv_from_bytes() {
        // Test full deserialization (not zero-copy but validates roundtrip)
        let edge = Edge::new(123, 0.75, 5);

        // Serialize
        let bytes = rkyv::to_bytes::<Error>(&edge).expect("serialize edge");

        // Deserialize back to owned type
        let deserialized: Edge = rkyv::from_bytes::<Edge, Error>(&bytes)
            .expect("deserialize edge");

        assert_eq!(deserialized, edge);
    }

    #[test]
    fn test_smallvec_inline_storage() {
        // 4 or fewer edges should be stored inline (no heap allocation)
        let edges: SmallVec<[Edge; 4]> = SmallVec::from_vec(vec![
            Edge::new(1, 0.9, 0),
            Edge::new(2, 0.8, 0),
            Edge::new(3, 0.7, 0),
            Edge::new(4, 0.6, 0),
        ]);

        // SmallVec stores inline when len <= capacity
        assert!(!edges.spilled());

        // Adding a 5th edge causes spill to heap
        let mut edges_spilled = edges.clone();
        edges_spilled.push(Edge::new(5, 0.5, 0));
        assert!(edges_spilled.spilled());
    }
}
