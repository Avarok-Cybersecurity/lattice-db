#![cfg(not(target_arch = "wasm32"))]
//! A relation first named by a durable edge write keeps its name after a
//! restart: the WAL logs it with the edge, so replay can resolve the edge's
//! relation id. `open_engine` uses a config that declares no relations.

use lattice_test_harness::{make_point, open_engine, SharedState};

#[tokio::test]
async fn relation_named_by_an_edge_write_survives_restart() {
    let state = SharedState::new();
    {
        let mut engine = open_engine(&state).await.unwrap();
        engine
            .upsert_points_async(vec![make_point(1, 4), make_point(2, 4)])
            .await
            .unwrap();
        engine.add_edge_async(1, 2, "knows", 0.8).await.unwrap();
    }

    let engine = open_engine(&state).await.unwrap();
    let edges = engine.get_edges(1).unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].relation, "knows");
    assert_eq!(engine.config().relation_id("knows"), Some(0));
}
