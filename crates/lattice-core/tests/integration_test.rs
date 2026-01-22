//! Integration tests for LatticeDB full workflow
//!
//! Tests the complete pipeline:
//! 1. Create collection with explicit config (PCND)
//! 2. Upsert points with vectors and payloads
//! 3. Add graph edges
//! 4. Vector similarity search
//! 5. Graph traversal
//! 6. Serialization roundtrip
//! 7. Data integrity verification
//!
//! These tests run on both native and WASM targets.

// On WASM, use wasm_bindgen_test as the test macro
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

// Run WASM tests in browser (Node.js has issues with wasm-bindgen-test)
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use lattice_core::engine::collection::CollectionEngine;
use lattice_core::graph::{BfsIterator, DfsIterator};
use lattice_core::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
use lattice_core::types::point::{Edge, Point, PointId};
use lattice_core::types::query::SearchQuery;
use smallvec::SmallVec;
use std::collections::HashSet;

/// Create a test collection config with explicit PCND-compliant parameters
fn test_config() -> CollectionConfig {
    CollectionConfig::new(
        "integration_test",
        VectorConfig::new(4, Distance::Cosine),
        HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 100,
            ef_construction: 200,
        },
    )
}

/// Create test points with vectors forming a known similarity pattern
fn create_test_points() -> Vec<Point> {
    vec![
        // Cluster A: similar vectors
        Point::new_vector(1, vec![1.0, 0.0, 0.0, 0.0]),
        Point::new_vector(2, vec![0.9, 0.1, 0.0, 0.0]),
        Point::new_vector(3, vec![0.8, 0.2, 0.0, 0.0]),
        // Cluster B: different direction
        Point::new_vector(4, vec![0.0, 1.0, 0.0, 0.0]),
        Point::new_vector(5, vec![0.0, 0.9, 0.1, 0.0]),
        // Cluster C: third direction
        Point::new_vector(6, vec![0.0, 0.0, 1.0, 0.0]),
        Point::new_vector(7, vec![0.0, 0.0, 0.9, 0.1]),
    ]
}

#[test]
fn test_full_workflow_vector_operations() {
    // Step 1: Create collection with explicit config (PCND)
    let config = test_config();
    let mut engine = CollectionEngine::new(config.clone()).unwrap();

    // Verify config was stored correctly
    assert_eq!(engine.config().name, "integration_test");
    assert_eq!(engine.config().vectors.size, 4);
    assert_eq!(engine.config().hnsw.m, 16);

    // Step 2: Upsert points
    let points = create_test_points();
    let result = engine.upsert_points(points.clone());
    assert!(result.is_ok());
    assert_eq!(result.unwrap().inserted, 7);
    assert_eq!(engine.point_count(), 7);

    // Step 3: Verify points can be retrieved
    let retrieved = engine.get_points(&[1, 4, 7]);
    assert_eq!(retrieved.len(), 3);
    assert!(retrieved[0].is_some());
    assert!(retrieved[1].is_some());
    assert!(retrieved[2].is_some());
    assert_eq!(retrieved[0].as_ref().unwrap().id, 1);
    assert_eq!(retrieved[1].as_ref().unwrap().id, 4);
    assert_eq!(retrieved[2].as_ref().unwrap().id, 7);

    // Step 4: Search for similar vectors
    // Query close to cluster A
    let query = SearchQuery::new(vec![0.95, 0.05, 0.0, 0.0], 3).with_ef(50);
    let results = engine.search(query);
    assert!(results.is_ok());
    let results = results.unwrap();

    // Should find cluster A points (1, 2, 3) as most similar
    assert_eq!(results.len(), 3);
    let found_ids: HashSet<PointId> = results.iter().map(|r| r.id).collect();
    assert!(found_ids.contains(&1));
    assert!(found_ids.contains(&2));
    assert!(found_ids.contains(&3));

    // Step 5: Verify search ranking (point 1 should be closest)
    assert_eq!(results[0].id, 1, "Point 1 should be the closest match");
}

#[test]
fn test_full_workflow_graph_operations() {
    let config = test_config();
    let mut engine = CollectionEngine::new(config).unwrap();

    // Insert points
    let points = create_test_points();
    engine.upsert_points(points).unwrap();

    // Step 1: Add graph edges
    // Create a graph: 1 -> 2 -> 3, 1 -> 4 -> 5
    // add_edge signature: (from, to, relation: &str, weight: f32)
    engine.add_edge(1, 2, "similar", 0.9).unwrap();
    engine.add_edge(2, 3, "similar", 0.8).unwrap();
    engine.add_edge(1, 4, "related", 0.7).unwrap();
    engine.add_edge(4, 5, "related", 0.6).unwrap();
    engine.add_edge(3, 6, "similar", 0.5).unwrap();

    // Step 2: Verify edges were added
    let point1 = engine.get_point(1);
    assert!(point1.is_some());
    let point1 = point1.unwrap();
    assert!(point1.has_edges());
    assert_eq!(point1.edge_count(), 2); // edges to 2 and 4

    // Step 3: Traverse from point 1 (all relations)
    let traversal = engine.traverse(1, 3, None);
    assert!(traversal.is_ok());
    let traversal = traversal.unwrap();

    // Should find paths to: 2, 4, 3, 5, 6
    assert_eq!(traversal.start_id, 1);
    assert!(traversal.nodes_visited >= 5); // 1 + at least 4 others

    // Verify paths exist
    let path_targets: HashSet<PointId> = traversal.paths.iter().map(|p| p.target_id).collect();
    assert!(path_targets.contains(&2));
    assert!(path_targets.contains(&4));
    assert!(path_targets.contains(&3));
    assert!(path_targets.contains(&5));

    // Step 4: Filter by relation type
    let traversal_similar = engine.traverse(1, 3, Some(&["similar"]));
    assert!(traversal_similar.is_ok());
    let traversal_similar = traversal_similar.unwrap();

    // With "similar" relation only: 1 -> 2 -> 3 -> 6
    let similar_targets: HashSet<PointId> = traversal_similar
        .paths
        .iter()
        .map(|p| p.target_id)
        .collect();
    assert!(similar_targets.contains(&2));
    assert!(similar_targets.contains(&3));
    assert!(similar_targets.contains(&6));
    // Point 4 and 5 use "related" relation, should not be visited
    assert!(!similar_targets.contains(&4));
    assert!(!similar_targets.contains(&5));
}

#[test]
fn test_full_workflow_serialization_roundtrip() {
    let config = test_config();
    let mut engine = CollectionEngine::new(config).unwrap();

    // Build complete state: points + edges
    let points = create_test_points();
    engine.upsert_points(points).unwrap();

    // Add edges
    engine.add_edge(1, 2, "similar", 0.9).unwrap();
    engine.add_edge(2, 3, "similar", 0.8).unwrap();
    engine.add_edge(1, 4, "related", 0.7).unwrap();

    // Serialize
    let bytes = engine.to_bytes();
    assert!(bytes.is_ok(), "Serialization failed: {:?}", bytes.err());
    let bytes = bytes.unwrap();
    assert!(!bytes.is_empty());

    // Deserialize
    let restored = CollectionEngine::from_bytes(&bytes);
    assert!(
        restored.is_ok(),
        "Deserialization failed: {:?}",
        restored.err()
    );
    let restored = restored.unwrap();

    // Verify config preserved
    assert_eq!(restored.config().name, "integration_test");
    assert_eq!(restored.config().vectors.size, 4);
    assert_eq!(restored.config().hnsw.m, 16);
    assert_eq!(restored.config().hnsw.ef_construction, 200);

    // Verify point count
    assert_eq!(restored.point_count(), 7);

    // Verify specific points
    let point1 = restored.get_point(1);
    assert!(point1.is_some());
    let point1 = point1.unwrap();
    assert_eq!(point1.vector, vec![1.0, 0.0, 0.0, 0.0]);
    assert!(point1.has_edges());
    assert_eq!(point1.edge_count(), 2);

    // Verify search still works
    let query = SearchQuery::new(vec![0.95, 0.05, 0.0, 0.0], 3).with_ef(50);
    let results = restored.search(query).unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].id, 1);
}

#[test]
fn test_bfs_dfs_iterators_standalone() {
    // Test the standalone graph iterators
    // Graph: 1 -> [2, 3], 2 -> [4], 3 -> [4, 5], 4 -> [6], 5 -> []
    let get_neighbors = |node: PointId| -> Vec<PointId> {
        match node {
            1 => vec![2, 3],
            2 => vec![4],
            3 => vec![4, 5],
            4 => vec![6],
            _ => vec![],
        }
    };

    // BFS: level order traversal
    let bfs: Vec<(PointId, usize)> = BfsIterator::new(1, 10, get_neighbors).collect();

    // Should visit in level order
    assert_eq!(bfs[0], (1, 0)); // depth 0
                                // depth 1: 2, 3
    assert!(bfs[1..3].contains(&(2, 1)));
    assert!(bfs[1..3].contains(&(3, 1)));
    // depth 2: 4, 5 (4 discovered via 2, 5 via 3)
    assert!(bfs[3..5].contains(&(4, 2)));
    assert!(bfs[3..5].contains(&(5, 2)));
    // depth 3: 6
    assert_eq!(bfs[5], (6, 3));

    // DFS: depth-first order
    let dfs: Vec<(PointId, usize)> = DfsIterator::new(1, 10, get_neighbors).collect();
    assert_eq!(dfs[0], (1, 0)); // starts at root
    assert_eq!(dfs.len(), 6); // visits all 6 nodes

    // Verify all nodes visited
    let visited: HashSet<PointId> = dfs.iter().map(|(n, _)| *n).collect();
    assert!(visited.contains(&1));
    assert!(visited.contains(&2));
    assert!(visited.contains(&3));
    assert!(visited.contains(&4));
    assert!(visited.contains(&5));
    assert!(visited.contains(&6));
}

#[test]
fn test_point_with_payload() {
    let config = test_config();
    let mut engine = CollectionEngine::new(config).unwrap();

    // Create point with payload (payload values must be valid JSON bytes)
    let mut point = Point::new_vector(1, vec![1.0, 0.0, 0.0, 0.0]);
    point
        .payload
        .insert("category".to_string(), br#""test""#.to_vec());
    point.payload.insert("score".to_string(), b"42".to_vec());

    engine.upsert_points(vec![point]).unwrap();

    // Retrieve and verify payload
    let retrieved = engine.get_point(1);
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(
        retrieved.payload.get("category"),
        Some(&br#""test""#.to_vec())
    );
    assert_eq!(retrieved.payload.get("score"), Some(&b"42".to_vec()));

    // Serialize/deserialize preserves payload
    let bytes = engine.to_bytes().unwrap();
    let restored = CollectionEngine::from_bytes(&bytes).unwrap();

    let retrieved = restored.get_point(1).unwrap();
    assert_eq!(
        retrieved.payload.get("category"),
        Some(&br#""test""#.to_vec())
    );
}

#[test]
fn test_delete_points() {
    let config = test_config();
    let mut engine = CollectionEngine::new(config).unwrap();

    let points = create_test_points();
    engine.upsert_points(points).unwrap();
    assert_eq!(engine.point_count(), 7);

    // Delete some points
    let deleted = engine.delete_points(&[1, 3, 5]);
    assert_eq!(deleted, 3);
    assert_eq!(engine.point_count(), 4);

    // Verify deleted points are gone
    assert!(engine.get_point(1).is_none());
    assert!(engine.get_point(2).is_some());
    assert!(engine.get_point(3).is_none());
    assert!(engine.get_point(4).is_some());
    assert!(engine.get_point(5).is_none());

    // Search should not return deleted points
    let query = SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0], 5).with_ef(50);
    let results = engine.search(query).unwrap();
    let result_ids: HashSet<PointId> = results.iter().map(|r| r.id).collect();
    assert!(!result_ids.contains(&1));
    assert!(!result_ids.contains(&3));
    assert!(!result_ids.contains(&5));
}

#[test]
fn test_upsert_updates_existing() {
    let config = test_config();
    let mut engine = CollectionEngine::new(config).unwrap();

    // Insert initial point
    let point1 = Point::new_vector(1, vec![1.0, 0.0, 0.0, 0.0]);
    engine.upsert_points(vec![point1]).unwrap();

    // Verify initial state
    let retrieved = engine.get_point(1).unwrap();
    assert_eq!(retrieved.vector, vec![1.0, 0.0, 0.0, 0.0]);

    // Upsert with new vector (same ID)
    let point1_updated = Point::new_vector(1, vec![0.0, 1.0, 0.0, 0.0]);
    let result = engine.upsert_points(vec![point1_updated]).unwrap();
    assert_eq!(result.inserted, 0);
    assert_eq!(result.updated, 1);

    // Verify update
    let retrieved = engine.get_point(1).unwrap();
    assert_eq!(retrieved.vector, vec![0.0, 1.0, 0.0, 0.0]);

    // Point count should still be 1
    assert_eq!(engine.point_count(), 1);
}

#[test]
fn test_edge_with_smallvec_inline() {
    // Verify SmallVec optimization: ≤4 edges should be inline
    let edges: SmallVec<[Edge; 4]> = SmallVec::from_vec(vec![
        Edge::new(1, 0.9, 0),
        Edge::new(2, 0.8, 0),
        Edge::new(3, 0.7, 0),
        Edge::new(4, 0.6, 0),
    ]);
    assert!(!edges.spilled(), "4 edges should be inline");

    // 5th edge causes spill
    let mut edges_spilled = edges.clone();
    edges_spilled.push(Edge::new(5, 0.5, 0));
    assert!(edges_spilled.spilled(), "5 edges should spill to heap");
}

#[test]
fn test_distance_metrics() {
    let config = CollectionConfig::new(
        "cosine_test",
        VectorConfig::new(3, Distance::Cosine),
        HnswConfig {
            m: 8,
            m0: 16,
            ml: HnswConfig::recommended_ml(8),
            ef: 50,
            ef_construction: 100,
        },
    );

    let mut engine = CollectionEngine::new(config).unwrap();

    // Orthogonal vectors
    engine
        .upsert_points(vec![
            Point::new_vector(1, vec![1.0, 0.0, 0.0]),
            Point::new_vector(2, vec![0.0, 1.0, 0.0]),
            Point::new_vector(3, vec![0.0, 0.0, 1.0]),
        ])
        .unwrap();

    // Search with [1, 0, 0] should find point 1 as closest
    let query = SearchQuery::new(vec![1.0, 0.0, 0.0], 3).with_ef(50);
    let results = engine.search(query).unwrap();
    assert_eq!(results[0].id, 1);
    // For cosine, identical vectors have distance 0
    assert!(
        results[0].score < 0.01,
        "Identical vectors should have ~0 distance"
    );
}

#[test]
fn test_combined_vector_and_graph_search() {
    // Test scenario: Find similar vectors, then traverse their graph connections
    let config = test_config();
    let mut engine = CollectionEngine::new(config).unwrap();

    // Insert points
    engine.upsert_points(create_test_points()).unwrap();

    // Create graph connections representing semantic relationships
    // Cluster A has internal connections
    engine.add_edge(1, 2, "similar", 0.95).unwrap();
    engine.add_edge(2, 3, "similar", 0.90).unwrap();
    // Cross-cluster connection (rare but important)
    engine.add_edge(3, 4, "related", 0.5).unwrap();

    // Step 1: Vector search to find entry points
    let query = SearchQuery::new(vec![0.85, 0.15, 0.0, 0.0], 2).with_ef(50);
    let search_results = engine.search(query).unwrap();

    // Should find points 1, 2, or 3 (cluster A)
    let entry_point = search_results[0].id;
    assert!(
        entry_point == 1 || entry_point == 2 || entry_point == 3,
        "Entry point should be in cluster A"
    );

    // Step 2: Graph traversal from search result
    let traversal = engine.traverse(entry_point, 3, None).unwrap();

    // Should discover related content through graph
    assert!(
        traversal.nodes_visited >= 2,
        "Should discover connected nodes"
    );
}

#[test]
fn test_scroll_pagination() {
    use lattice_core::types::query::ScrollQuery;

    let config = test_config();
    let mut engine = CollectionEngine::new(config).unwrap();

    // Insert 25 points
    for i in 0..25 {
        let vector = vec![i as f32 * 0.1, 0.0, 0.0, 0.0];
        engine
            .upsert_points(vec![Point::new_vector(i, vector)])
            .unwrap();
    }

    // First page
    let result = engine.scroll(ScrollQuery::new(10));
    assert_eq!(result.points.len(), 10);
    assert!(result.next_offset.is_some());

    // Second page
    let result = engine.scroll(ScrollQuery::new(10).with_offset(result.next_offset.unwrap()));
    assert_eq!(result.points.len(), 10);
    assert!(result.next_offset.is_some());

    // Third page (partial)
    let result = engine.scroll(ScrollQuery::new(10).with_offset(result.next_offset.unwrap()));
    assert_eq!(result.points.len(), 5);
    assert!(result.next_offset.is_none());
}

#[test]
fn test_score_threshold() {
    let config = test_config();
    let mut engine = CollectionEngine::new(config).unwrap();

    // Insert points at varying distances from query
    engine
        .upsert_points(vec![
            Point::new_vector(1, vec![1.0, 0.0, 0.0, 0.0]), // Very close
            Point::new_vector(2, vec![0.7, 0.7, 0.0, 0.0]), // Medium distance
            Point::new_vector(3, vec![0.0, 1.0, 0.0, 0.0]), // Far
        ])
        .unwrap();

    // Search with score threshold to filter results
    let query = SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0], 10)
        .with_ef(50)
        .with_score_threshold(0.5); // Only return results with score <= 0.5

    let results = engine.search(query).unwrap();

    // Point 1 should be returned (score ~0), point 3 should be filtered (score ~1.0)
    assert!(
        results.iter().any(|r| r.id == 1),
        "Point 1 should pass threshold"
    );
    assert!(
        !results.iter().any(|r| r.id == 3),
        "Point 3 should be filtered by threshold"
    );
}
