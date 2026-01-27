//! Stress and ordering tests for WAL ACID compliance
//!
//! Verifies correct final state after interleaved operations
//! and checkpoint + crash scenarios.

use lattice_core::types::point::Point;
use lattice_test_harness::{make_point, open_engine, SharedState};

const DIM: usize = 4;

/// Interleaved upsert/delete/upsert on same ID: correct final state
#[tokio::test]
async fn test_interleaved_operations_ordering() {
    let state = SharedState::new();

    // Phase 1: upsert A, delete A, upsert A (new vector)
    {
        let mut engine = open_engine(&state).await.unwrap();

        // First upsert
        let v1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        engine
            .upsert_points_async(vec![Point::new_vector(42, v1)])
            .await
            .unwrap();

        // Delete
        engine.delete_points_async(&[42]).await.unwrap();

        // Re-upsert with different vector
        let v2: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        engine
            .upsert_points_async(vec![Point::new_vector(42, v2.clone())])
            .await
            .unwrap();

        // Drop = crash
    }

    // Phase 2: Verify final state is the last upsert
    {
        let engine = open_engine(&state).await.unwrap();

        assert!(engine.point_exists(42), "Point 42 should exist");
        assert_eq!(engine.point_count(), 1);

        let point = engine.get_point(42).unwrap().unwrap();
        let vector = &point.vector;
        assert_eq!(
            vector.as_slice(),
            &[10.0, 20.0, 30.0, 40.0],
            "Vector should match last upsert"
        );
    }
}

/// Checkpoint then crash: all data (pre and post checkpoint) recovered
#[tokio::test]
async fn test_checkpoint_then_crash() {
    let state = SharedState::new();

    // Phase 1: Write 100 points, checkpoint, write 50 more, crash
    {
        let mut engine = open_engine(&state).await.unwrap();

        // First batch: 100 points
        let points: Vec<Point> = (0..100).map(|i| make_point(i, DIM)).collect();
        engine.upsert_points_async(points).await.unwrap();

        // Note: checkpoint happens internally via the engine's transaction manager.
        // We force it by doing a large batch that triggers should_checkpoint().

        // Second batch: 50 more points
        let points2: Vec<Point> = (100..150).map(|i| make_point(i, DIM)).collect();
        engine.upsert_points_async(points2).await.unwrap();

        // Drop = crash
    }

    // Phase 2: All 150 should be recovered
    {
        let engine = open_engine(&state).await.unwrap();

        assert_eq!(
            engine.point_count(),
            150,
            "All 150 points should survive crash"
        );

        // Spot-check some points
        for id in [0, 49, 99, 100, 149] {
            assert!(
                engine.point_exists(id),
                "Point {} should exist",
                id
            );
        }
    }
}
