//! Collection Engine - manages a single collection
//!
//! Provides high-level APIs for point operations, search, and graph traversal.
//!
//! ## Async Indexing (Native)
//!
//! On native builds, the engine uses async indexing for improved insert latency:
//! - Points are stored immediately in `points` storage
//! - Index updates happen in background via `AsyncIndexer`
//! - Search merges indexed results with brute-force on pending points
//!
//! On WASM builds, indexing is synchronous (existing behavior).

mod types;

#[cfg(not(target_arch = "wasm32"))]
mod native;

#[cfg(target_arch = "wasm32")]
mod wasm;

// Re-export types
pub use types::{EdgeInfo, TraversalPath, TraversalResult, UpsertResult};

// Re-export platform-specific CollectionEngine
#[cfg(not(target_arch = "wasm32"))]
pub use native::CollectionEngine;

#[cfg(target_arch = "wasm32")]
pub use wasm::CollectionEngine;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
    use crate::types::point::{Point, Vector};
    use crate::types::query::{ScrollQuery, SearchQuery};

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn test_config() -> CollectionConfig {
        CollectionConfig::new(
            "test_collection",
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

    fn random_vector(dim: usize, seed: u64) -> Vector {
        let mut rng = seed;
        (0..dim)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng as f64 / u64::MAX as f64) as f32
            })
            .collect()
    }

    #[test]
    fn test_new_collection() {
        let engine = CollectionEngine::new(test_config()).unwrap();
        assert_eq!(engine.name(), "test_collection");
        assert_eq!(engine.point_count(), 0);
        assert!(engine.is_empty());
        assert_eq!(engine.vector_dim(), 4);
    }

    #[test]
    fn test_upsert_points() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();

        let points = vec![
            Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4]),
            Point::new_vector(2, vec![0.5, 0.6, 0.7, 0.8]),
        ];

        let result = engine.upsert_points(points).unwrap();
        assert_eq!(result.inserted, 2);
        assert_eq!(result.updated, 0);
        assert_eq!(engine.point_count(), 2);
    }

    #[test]
    fn test_upsert_update() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();

        // Insert
        engine
            .upsert_points(vec![Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4])])
            .unwrap();

        // Update
        let result = engine
            .upsert_points(vec![Point::new_vector(1, vec![0.9, 0.8, 0.7, 0.6])])
            .unwrap();

        assert_eq!(result.inserted, 0);
        assert_eq!(result.updated, 1);
        assert_eq!(engine.point_count(), 1);

        // Verify updated
        let point = engine.get_point(1).unwrap().unwrap();
        assert_eq!(point.vector[0], 0.9);
    }

    #[test]
    fn test_dimension_mismatch() {
        use crate::error::LatticeError;

        let mut engine = CollectionEngine::new(test_config()).unwrap();

        let result = engine.upsert_points(vec![Point::new_vector(1, vec![0.1, 0.2])]);

        assert!(matches!(
            result,
            Err(LatticeError::DimensionMismatch {
                expected: 4,
                actual: 2
            })
        ));
    }

    #[test]
    fn test_get_points() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();

        engine
            .upsert_points(vec![
                Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4]),
                Point::new_vector(2, vec![0.5, 0.6, 0.7, 0.8]),
            ])
            .unwrap();

        let results = engine.get_points(&[1, 2, 99]).unwrap();
        assert!(results[0].is_some());
        assert!(results[1].is_some());
        assert!(results[2].is_none());
    }

    #[test]
    fn test_delete_points() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();

        engine
            .upsert_points(vec![
                Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4]),
                Point::new_vector(2, vec![0.5, 0.6, 0.7, 0.8]),
            ])
            .unwrap();

        let deleted = engine.delete_points(&[1, 99]).unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(engine.point_count(), 1);
        assert!(!engine.point_exists(1));
        assert!(engine.point_exists(2));
    }

    #[test]
    fn test_search() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();

        // Insert points
        for i in 0..50 {
            let point = Point::new_vector(i, random_vector(4, i));
            engine.upsert_points(vec![point]).unwrap();
        }

        let query = SearchQuery::new(random_vector(4, 999), 10);
        let results = engine.search(query).unwrap();

        assert_eq!(results.len(), 10);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].score <= results[i].score);
        }
    }

    #[test]
    fn test_search_with_payload() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();

        let point = Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4])
            .with_field("name", br#""test""#.to_vec());
        engine.upsert_points(vec![point]).unwrap();

        let query = SearchQuery::new(vec![0.1, 0.2, 0.3, 0.4], 1).include_payload();
        let results = engine.search(query).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].payload.is_some());
    }

    #[test]
    fn test_scroll() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();

        // Insert 25 points
        for i in 0..25 {
            engine
                .upsert_points(vec![Point::new_vector(i, random_vector(4, i))])
                .unwrap();
        }

        // First page
        let result = engine.scroll(ScrollQuery::new(10)).unwrap();
        assert_eq!(result.points.len(), 10);
        assert!(result.next_offset.is_some());

        // Second page
        let result = engine
            .scroll(ScrollQuery::new(10).with_offset(result.next_offset.unwrap()))
            .unwrap();
        assert_eq!(result.points.len(), 10);
        assert!(result.next_offset.is_some());

        // Third page (partial)
        let result = engine
            .scroll(ScrollQuery::new(10).with_offset(result.next_offset.unwrap()))
            .unwrap();
        assert_eq!(result.points.len(), 5);
        assert!(result.next_offset.is_none());
    }

    #[test]
    fn test_add_edge() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();

        engine
            .upsert_points(vec![
                Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4]),
                Point::new_vector(2, vec![0.5, 0.6, 0.7, 0.8]),
            ])
            .unwrap();

        engine.add_edge(1, 2, "similar_to", 0.9).unwrap();

        let edges = engine.get_edges(1).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target_id, 2);
        assert_eq!(edges[0].weight, 0.9);
        assert_eq!(edges[0].relation, "similar_to");
    }

    #[test]
    fn test_remove_edge() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();

        engine
            .upsert_points(vec![
                Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4]),
                Point::new_vector(2, vec![0.5, 0.6, 0.7, 0.8]),
            ])
            .unwrap();

        engine.add_edge(1, 2, "similar_to", 0.9).unwrap();
        let removed = engine.remove_edge(1, 2, Some("similar_to")).unwrap();

        assert!(removed);
        assert!(engine.get_edges(1).unwrap().is_empty());
    }

    #[test]
    fn test_traverse() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();

        // Create a small graph: 1 -> 2 -> 3
        engine
            .upsert_points(vec![
                Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4]),
                Point::new_vector(2, vec![0.2, 0.3, 0.4, 0.5]),
                Point::new_vector(3, vec![0.3, 0.4, 0.5, 0.6]),
            ])
            .unwrap();

        engine.add_edge(1, 2, "next", 1.0).unwrap();
        engine.add_edge(2, 3, "next", 1.0).unwrap();

        let result = engine.traverse(1, 3, None).unwrap();

        assert_eq!(result.start_id, 1);
        assert_eq!(result.nodes_visited, 3); // 1, 2, 3
        assert_eq!(result.paths.len(), 2); // paths to 2 and 3
    }

    #[test]
    fn test_traverse_with_relation_filter() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();

        engine
            .upsert_points(vec![
                Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4]),
                Point::new_vector(2, vec![0.2, 0.3, 0.4, 0.5]),
                Point::new_vector(3, vec![0.3, 0.4, 0.5, 0.6]),
            ])
            .unwrap();

        engine.add_edge(1, 2, "friend", 1.0).unwrap();
        engine.add_edge(1, 3, "enemy", 1.0).unwrap();

        // Only traverse "friend" edges
        let result = engine.traverse(1, 2, Some(&["friend"])).unwrap();

        assert_eq!(result.paths.len(), 1);
        assert_eq!(result.paths[0].target_id, 2);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();

        engine
            .upsert_points(vec![
                Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4]),
                Point::new_vector(2, vec![0.5, 0.6, 0.7, 0.8]),
            ])
            .unwrap();

        engine.add_edge(1, 2, "similar", 0.9).unwrap();

        // Serialize
        let bytes = engine.to_bytes().unwrap();

        // Deserialize
        let restored = CollectionEngine::from_bytes(&bytes).unwrap();

        assert_eq!(restored.name(), "test_collection");
        assert_eq!(restored.point_count(), 2);
        assert!(restored.point_exists(1));
        assert!(restored.point_exists(2));
    }
}
