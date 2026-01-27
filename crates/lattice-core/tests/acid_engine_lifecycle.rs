#![cfg(not(target_arch = "wasm32"))]
//! Full engine ACID lifecycle tests
//!
//! Tests upsert/delete/edge operations through the CollectionEngine
//! durable API, verifying crash recovery at the engine level.

use lattice_core::engine::collection::CollectionEngineBuilder;
use lattice_core::types::point::Point;
use lattice_core::wal::{WalEntry, WriteAheadLog};
use lattice_test_harness::{make_config, make_point, open_engine, SharedState};

const DIM: usize = 4;

async fn open_engine_with_edges(
    state: &SharedState,
) -> lattice_core::LatticeResult<
    lattice_core::CollectionEngine<
        lattice_test_harness::MockStorage,
        lattice_test_harness::MockStorage,
    >,
> {
    let config = make_config("test_acid", DIM).with_relation("related_to", 1);
    CollectionEngineBuilder::new(config)
        .with_wal(state.mock())
        .with_data(state.mock())
        .open()
        .await
}

/// Upsert, delete, update, crash, reopen → correct state
#[tokio::test]
async fn test_upsert_delete_recover() {
    let state = SharedState::new();

    // Phase 1: Upsert 100 points, delete 20, update 10
    {
        let mut engine = open_engine(&state).await.unwrap();

        // Upsert 100
        let points: Vec<Point> = (0..100).map(|i| make_point(i, DIM)).collect();
        engine.upsert_points_async(points).await.unwrap();

        // Delete IDs 0-19
        let delete_ids: Vec<u64> = (0..20).collect();
        let deleted = engine.delete_points_async(&delete_ids).await.unwrap();
        assert_eq!(deleted, 20);

        // Update IDs 80-89 with new vectors
        let updated: Vec<Point> = (80..90)
            .map(|i| {
                let vector: Vec<f32> = (0..DIM).map(|d| (i as f32 * 10.0) + d as f32).collect();
                Point::new_vector(i, vector)
            })
            .collect();
        engine.upsert_points_async(updated).await.unwrap();

        // Drop = crash
    }

    // Phase 2: Reopen and verify state
    {
        let engine = open_engine(&state).await.unwrap();

        // Deleted points should be gone
        for id in 0..20 {
            assert!(!engine.point_exists(id), "Point {} should be deleted", id);
        }

        // Remaining points should exist
        for id in 20..100 {
            assert!(engine.point_exists(id), "Point {} should exist", id);
        }

        assert_eq!(engine.point_count(), 80, "80 points should remain");
    }
}

/// Edge operations survive crash and recovery
#[tokio::test]
async fn test_edges_survive_crash() {
    let state = SharedState::new();

    // Phase 1: Add points and edges, then crash
    {
        let mut engine = open_engine_with_edges(&state).await.unwrap();

        // Create two points
        engine
            .upsert_points_async(vec![
                make_point(1, DIM),
                make_point(2, DIM),
                make_point(3, DIM),
            ])
            .await
            .unwrap();

        // Add edges
        engine
            .add_edge_async(1, 2, "related_to", 1.0)
            .await
            .unwrap();
        engine
            .add_edge_async(1, 3, "related_to", 0.5)
            .await
            .unwrap();

        // Drop = crash
    }

    // Phase 2: Verify edges survived
    {
        let engine = open_engine_with_edges(&state).await.unwrap();

        let edges = engine.get_edges(1).unwrap();
        assert_eq!(edges.len(), 2, "Both edges should survive crash");

        // Remove one edge, then crash again
    }

    // Phase 3: Remove edge, crash, verify removal persisted
    {
        let mut engine = open_engine_with_edges(&state).await.unwrap();
        let removed = engine
            .remove_edge_async(1, 3, Some("related_to"))
            .await
            .unwrap();
        assert!(removed, "Edge should have been removed");
        // Drop = crash
    }

    // Phase 4: Verify removal persisted
    {
        let engine = open_engine_with_edges(&state).await.unwrap();
        let edges = engine.get_edges(1).unwrap();
        assert_eq!(edges.len(), 1, "Only one edge should remain after removal");
    }
}

/// WAL abort prevents phantom data from appearing after recovery
#[tokio::test]
async fn test_abort_prevents_phantom_data() {
    let state = SharedState::new();

    // Phase 1: Write upsert + abort at WAL level, then open engine
    {
        let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();

        // Good upsert (LSN 0)
        wal.append(&WalEntry::Upsert {
            points: vec![make_point(1, DIM)],
        })
        .await
        .unwrap();

        // Upsert that will be aborted (LSN 1)
        wal.append(&WalEntry::Upsert {
            points: vec![make_point(999, DIM)],
        })
        .await
        .unwrap();

        // Abort LSN 1
        wal.append(&WalEntry::Abort { aborted_lsn: 1 })
            .await
            .unwrap();

        wal.sync().await.unwrap();
    }

    // Phase 2: Open engine — should replay WAL, skip aborted entry
    {
        let engine = open_engine(&state).await.unwrap();

        assert!(engine.point_exists(1), "Non-aborted point should exist");
        assert!(
            !engine.point_exists(999),
            "Aborted point must NOT exist (phantom data)"
        );
        assert_eq!(engine.point_count(), 1, "Only 1 point should exist");
    }
}
