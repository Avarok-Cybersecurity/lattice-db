#![cfg(target_arch = "wasm32")]
//! Durable collections on wasm32: acknowledged writes survive a restart.
//!
//! The storage double is the harness's `SharedState`, an in-memory page and
//! metadata store held behind `Arc`s. It stands in for a Durable Object's
//! storage: dropping the engine is the object being evicted or restarted, and
//! opening a new engine over the same state is the object coming back to the
//! storage it left. `FailingStorage` wraps the same state to fail a chosen
//! write, the way a storage operation can reject.

use lattice_core::engine::collection::{CollectionEngine, DurableEngine};
use lattice_core::types::collection::{
    CollectionConfig, Distance, DurabilityMode, HnswConfig, VectorConfig,
};
use lattice_core::types::point::Point;
use lattice_core::types::query::SearchQuery;
use lattice_test_harness::{FailingStorage, MockStorage, SharedState};
use std::sync::atomic::Ordering;
use wasm_bindgen_test::wasm_bindgen_test;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

type Engine = DurableEngine<MockStorage, MockStorage>;

fn config() -> CollectionConfig {
    CollectionConfig::new(
        "do_collection",
        VectorConfig::new(4, Distance::Cosine),
        HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 50,
            ef_construction: 100,
        },
    )
    .with_durability(DurabilityMode::Durable)
}

fn point(id: u64) -> Point {
    let base = id as f32;
    Point::new_vector(id, vec![base, base + 1.0, base * 0.5, 1.0])
        .with_field("name", format!("\"point-{}\"", id).into_bytes())
}

async fn open(state: &SharedState) -> Engine {
    DurableEngine::open_durable(config(), state.mock(), state.mock())
        .await
        .unwrap()
}

fn edges(engine: &CollectionEngine, id: u64) -> Vec<(u64, String, f32)> {
    let mut edges: Vec<_> = engine
        .get_edges(id)
        .unwrap()
        .into_iter()
        .map(|e| (e.target_id, e.relation, e.weight))
        .collect();
    edges.sort_by_key(|e| e.0);
    edges
}

#[wasm_bindgen_test]
async fn points_and_edges_survive_reopen() {
    let state = SharedState::new();
    {
        let mut engine = open(&state).await;
        engine
            .upsert_points_async((1..=4).map(point).collect())
            .await
            .unwrap();
        engine.add_edge_async(1, 2, "knows", 0.9).await.unwrap();
        engine.add_edge_async(1, 3, "likes", 0.4).await.unwrap();
        engine.add_edge_async(2, 3, "knows", 0.7).await.unwrap();
        assert!(engine.remove_edge_async(2, 3, Some("knows")).await.unwrap());
        assert_eq!(engine.delete_points_async(&[4]).await.unwrap(), 1);
    }

    let engine = open(&state).await;
    assert_eq!(engine.point_ids().unwrap(), vec![1, 2, 3]);
    let restored = engine.get_point(2).unwrap().unwrap();
    assert_eq!(restored.vector, point(2).vector);
    assert_eq!(restored.payload, point(2).payload);
    assert_eq!(
        edges(&engine, 1),
        vec![(2, "knows".into(), 0.9), (3, "likes".into(), 0.4)]
    );
    assert!(edges(&engine, 2).is_empty());
    let hits = engine.search(SearchQuery::new(point(3).vector, 1)).unwrap();
    assert_eq!(hits[0].id, 3, "the index is rebuilt on reopen");
}

#[wasm_bindgen_test]
async fn writes_after_the_last_checkpoint_are_replayed() {
    let state = SharedState::new();
    {
        let mut engine = open(&state).await;
        engine
            .upsert_points_async(vec![point(1), point(2)])
            .await
            .unwrap();
        engine.add_edge_async(1, 2, "knows", 0.5).await.unwrap();
        engine.checkpoint().await.unwrap();

        // Appended to the WAL, never checkpointed: the restart interrupts here.
        engine.upsert_points_async(vec![point(3)]).await.unwrap();
        engine.add_edge_async(3, 1, "follows", 0.2).await.unwrap();
        assert_eq!(engine.delete_points_async(&[2]).await.unwrap(), 1);
    }

    let engine = open(&state).await;
    assert_eq!(engine.point_ids().unwrap(), vec![1, 3]);
    assert_eq!(edges(&engine, 3), vec![(1, "follows".into(), 0.2)]);
    assert_eq!(edges(&engine, 1), vec![(2, "knows".into(), 0.5)]);
}

#[wasm_bindgen_test]
async fn checkpoint_interrupted_before_its_wal_marker_loses_nothing() {
    let state = SharedState::new();
    {
        let mut engine = open(&state).await;
        engine
            .upsert_points_async(vec![point(1), point(2)])
            .await
            .unwrap();
        engine.add_edge_async(1, 2, "knows", 0.5).await.unwrap();
    }
    {
        // The snapshot lands in data storage; the WAL write recording the
        // checkpoint is rejected.
        let wal = FailingStorage::new(&state);
        wal.set_meta_fail_at.store(1, Ordering::SeqCst);
        let mut engine = DurableEngine::open_durable(config(), wal, state.mock())
            .await
            .unwrap();
        assert!(engine.checkpoint().await.is_err());
    }

    let mut engine = open(&state).await;
    assert_eq!(engine.point_ids().unwrap(), vec![1, 2]);
    assert_eq!(edges(&engine, 1), vec![(2, "knows".into(), 0.5)]);
    engine.upsert_points_async(vec![point(3)]).await.unwrap();
    engine.checkpoint().await.unwrap();
    drop(engine);
    assert_eq!(open(&state).await.point_ids().unwrap(), vec![1, 2, 3]);
}

#[wasm_bindgen_test]
async fn failed_storage_write_is_reported_and_never_visible() {
    let state = SharedState::new();
    open(&state)
        .await
        .upsert_points_async(vec![point(1), point(2)])
        .await
        .unwrap();
    {
        let wal = FailingStorage::new(&state);
        let trigger = wal.write_page_fail_at.clone();
        trigger.store(1, Ordering::SeqCst);
        let mut engine = DurableEngine::open_durable(config(), wal, state.mock())
            .await
            .unwrap();

        assert!(engine.upsert_points_async(vec![point(9)]).await.is_err());
        assert!(engine.add_edge_async(1, 2, "knows", 1.0).await.is_err());
        assert!(!engine.point_exists(9), "a failed write is not applied");
        assert!(edges(&engine, 1).is_empty());
        assert_eq!(engine.config().relation_id("knows"), None);

        // Storage recovers: the next commit must not carry the failed ones.
        trigger.store(0, Ordering::SeqCst);
        engine.upsert_points_async(vec![point(3)]).await.unwrap();
    }

    let engine = open(&state).await;
    assert_eq!(engine.point_ids().unwrap(), vec![1, 2, 3]);
    assert!(edges(&engine, 1).is_empty());
}

#[wasm_bindgen_test]
async fn header_write_failure_hides_the_edge_and_its_relation() {
    let state = SharedState::new();
    open(&state)
        .await
        .upsert_points_async(vec![point(1), point(2)])
        .await
        .unwrap();
    {
        // The WAL page is written, then its header is rejected.
        let wal = FailingStorage::new(&state);
        wal.set_meta_fail_at.store(1, Ordering::SeqCst);
        let mut engine = DurableEngine::open_durable(config(), wal, state.mock())
            .await
            .unwrap();
        assert!(engine.add_edge_async(1, 2, "knows", 1.0).await.is_err());
    }

    let mut engine = open(&state).await;
    assert!(edges(&engine, 1).is_empty());
    assert_eq!(engine.config().relation_id("knows"), None);
    engine.add_edge_async(1, 2, "likes", 0.3).await.unwrap();
    drop(engine);
    assert_eq!(
        edges(open(&state).await.engine(), 1),
        vec![(2, "likes".into(), 0.3)]
    );
}
