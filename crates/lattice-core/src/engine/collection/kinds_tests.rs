//! Index kinds, vector storage and snapshot restore, through the engine
//! (runs on native and wasm32 against each platform's `CollectionEngine`).

use super::CollectionEngine;
use crate::engine::snapshot;
use crate::text::{TextIndexConfig, TextSearchQuery};
use crate::types::collection::{
    CollectionConfig, Distance, HnswConfig, VectorConfig, VectorIndexKind, VectorStorage,
};
use crate::types::point::{Point, PointId};
use crate::types::query::SearchQuery;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

const DIM: usize = 8;

fn hnsw() -> HnswConfig {
    HnswConfig {
        m: 8,
        m0: 16,
        ml: HnswConfig::recommended_ml(8),
        ef: 64,
        ef_construction: 64,
    }
}

fn vector(seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(1);
    (0..DIM)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn doc(id: PointId, vector: Vec<f32>) -> Point {
    let words = ["alpha", "beta", "gamma", "delta", "epsilon"];
    let text = format!(
        "\"{} {}\"",
        words[id as usize % 5],
        words[(id as usize / 5) % 5]
    );
    Point::new_vector(id, vector).with_field("body", text.into_bytes())
}

fn engine(vectors: Option<VectorConfig>, n: u64) -> CollectionEngine {
    let config = CollectionConfig::with_optional_vectors("c", vectors.clone(), hnsw())
        .with_text_index(TextIndexConfig::bm25("body"));
    let mut engine = CollectionEngine::new(config).unwrap();
    let points = (0..n)
        .map(|i| {
            doc(
                i,
                if vectors.is_some() {
                    vector(i)
                } else {
                    Vec::new()
                },
            )
        })
        .collect();
    engine.upsert_points(points).unwrap();
    engine
}

fn ids(engine: &CollectionEngine, q: &[f32], k: usize) -> Vec<PointId> {
    let results = engine.search(SearchQuery::new(q.to_vec(), k)).unwrap();
    results.iter().map(|r| r.id).collect()
}

fn text_ids(engine: &CollectionEngine, text: &str) -> Vec<PointId> {
    let query = TextSearchQuery {
        text: text.to_string(),
        limit: 10,
        with_payload: false,
        with_vector: false,
    };
    engine
        .search_text(&query)
        .unwrap()
        .iter()
        .map(|r| r.id)
        .collect()
}

#[test]
fn flat_search_is_exact_and_survives_a_snapshot() {
    let flat = VectorConfig::new(DIM, Distance::Cosine).with_index(VectorIndexKind::Flat);
    let engine = engine(Some(flat), 120);
    let query = vector(999);
    let mut expected: Vec<(f32, PointId)> = (0..120)
        .map(|i| {
            (
                crate::index::distance::cosine_distance(&query, &vector(i)),
                i,
            )
        })
        .collect();
    expected.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let top: Vec<PointId> = expected.iter().take(10).map(|e| e.1).collect();
    assert_eq!(ids(&engine, &query, 10), top);

    let restored = CollectionEngine::from_bytes(&engine.to_bytes().unwrap()).unwrap();
    assert_eq!(ids(&restored, &query, 10), top);
    assert_eq!(text_ids(&restored, "gamma"), text_ids(&engine, "gamma"));
}

#[test]
fn hnsw_graph_and_postings_are_restored_not_rebuilt() {
    let engine = engine(Some(VectorConfig::new(DIM, Distance::Cosine)), 200);
    let bytes = engine.to_bytes().unwrap();
    let decoded = snapshot::decode(&bytes).unwrap();
    assert!(decoded.hnsw.is_some(), "graph section written");
    assert!(decoded.text.is_some(), "postings section written");
    let restored = CollectionEngine::from_bytes(&bytes).unwrap();
    for seed in [7, 77, 777] {
        assert_eq!(
            ids(&restored, &vector(seed), 5),
            ids(&engine, &vector(seed), 5)
        );
    }
    assert_eq!(
        text_ids(&restored, "beta alpha"),
        text_ids(&engine, "beta alpha")
    );
}

#[test]
fn compact_storage_rounds_on_write_and_round_trips_exactly() {
    for storage in [VectorStorage::F16, VectorStorage::Int8] {
        let config = VectorConfig::new(DIM, Distance::Dot)
            .with_index(VectorIndexKind::Flat)
            .with_storage(storage);
        let engine = engine(Some(config), 30);
        let stored = engine.get_point(3).unwrap().unwrap().vector;
        let mut expected = vector(3);
        storage.round_in_place(&mut expected);
        assert_eq!(stored, expected, "{storage:?}");
        let restored = CollectionEngine::from_bytes(&engine.to_bytes().unwrap()).unwrap();
        assert_eq!(restored.get_point(3).unwrap().unwrap().vector, stored);
        let q = vector(5);
        let a = engine.search(SearchQuery::new(q.clone(), 5)).unwrap();
        let b = restored.search(SearchQuery::new(q, 5)).unwrap();
        let scores = |r: &[crate::types::query::SearchResult]| {
            r.iter()
                .map(|x| (x.id, x.score.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(scores(&a), scores(&b), "{storage:?}");
    }
}

#[test]
fn collections_without_vectors_are_lexical_only() {
    let mut engine = engine(None, 25);
    assert_eq!(engine.vector_dim(), 0);
    assert_eq!(text_ids(&engine, "delta").len(), 9);
    let err = engine
        .search(SearchQuery::new(vec![1.0], 3))
        .unwrap_err()
        .to_string();
    assert!(err.contains("no dense vectors"), "{err}");
    let err = engine
        .upsert_points(vec![Point::new_vector(99, vec![1.0])])
        .unwrap_err()
        .to_string();
    assert!(err.contains("no dense vectors"), "{err}");
    let restored = CollectionEngine::from_bytes(&engine.to_bytes().unwrap()).unwrap();
    assert!(restored.config().vectors.is_none());
    assert_eq!(text_ids(&restored, "delta"), text_ids(&engine, "delta"));
}

#[test]
fn legacy_snapshots_rebuild_their_indexes() {
    let old_config = br#"{"name":"old","vectors":{"size":8,"distance":"cosine"},"hnsw":{"m":8,"m0":16,"ml":0.48,"ef":64,"ef_construction":64},"text_index":{"field":"body"}}"#;
    let points: Vec<Point> = (0..50).map(|i| doc(i, vector(i))).collect();
    let bytes = snapshot::legacy_for_tests(old_config, &points);
    let engine = CollectionEngine::from_bytes(&bytes).unwrap();
    assert_eq!(engine.point_count(), 50);
    assert_eq!(ids(&engine, &vector(4), 1), vec![4]);
    assert_eq!(text_ids(&engine, "epsilon").len(), 10);
}
