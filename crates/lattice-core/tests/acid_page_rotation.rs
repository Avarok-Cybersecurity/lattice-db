#![cfg(not(target_arch = "wasm32"))]
//! Page rotation edge case tests for WAL durability
//!
//! Verifies that crash during page boundary transitions and
//! multi-page recovery preserve all committed data.

use lattice_core::types::point::Point;
use lattice_core::wal::{WalEntry, WriteAheadLog};
use lattice_test_harness::SharedState;

/// Crash during page rotation: pre-rotation entries survive
#[tokio::test]
async fn test_crash_during_page_rotation() {
    let state = SharedState::new();

    // Phase 1: Write enough entries to span multiple pages (large payloads)
    let large_vec: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let synced_count;
    {
        let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();

        // Write entries with large vectors to fill pages faster
        for i in 0..50 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, large_vec.clone())],
            })
            .await
            .unwrap();
        }
        wal.sync().await.unwrap();
        synced_count = 50;

        // Write more entries that may trigger page rotation
        for i in 50..100 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, large_vec.clone())],
            })
            .await
            .unwrap();
        }
        // Drop = crash during/after rotation
    }

    // Phase 2: Verify synced entries survive
    {
        let wal = WriteAheadLog::open(state.mock()).await.unwrap();
        let entries = wal.read_from(0).await.unwrap();
        assert!(
            entries.len() >= synced_count,
            "At least {} synced entries must survive, got {}",
            synced_count,
            entries.len()
        );

        // Verify LSN ordering
        for window in entries.windows(2) {
            assert!(window[1].0 > window[0].0, "LSNs must be monotonic");
        }
    }
}

/// Recovery across many pages: all entries in order, LSN monotonic
#[tokio::test]
async fn test_recovery_across_many_pages() {
    let state = SharedState::new();
    let large_vec: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let total_entries: u64 = 200;

    // Phase 1: Write enough to span 5+ pages, syncing periodically
    {
        let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();

        for i in 0..total_entries {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, large_vec.clone())],
            })
            .await
            .unwrap();

            // Sync every 20 entries
            if (i + 1) % 20 == 0 {
                wal.sync().await.unwrap();
            }
        }
        wal.sync().await.unwrap();
    }

    // Verify multiple pages were created
    {
        let pages = state.pages.read().unwrap();
        assert!(
            pages.len() > 1,
            "Should span multiple pages, got {} pages",
            pages.len()
        );
    }

    // Phase 2: Crash and recover
    {
        let wal = WriteAheadLog::open(state.mock()).await.unwrap();
        let entries = wal.read_from(0).await.unwrap();

        assert_eq!(
            entries.len(),
            total_entries as usize,
            "All entries must survive"
        );

        // Verify LSN monotonicity
        for (idx, window) in entries.windows(2).enumerate() {
            assert!(
                window[1].0 > window[0].0,
                "LSN not monotonic at index {}: {} >= {}",
                idx,
                window[0].0,
                window[1].0
            );
        }

        // Verify LSNs are sequential
        for (i, (lsn, _)) in entries.iter().enumerate() {
            assert_eq!(*lsn, i as u64, "LSN should equal index");
        }
    }
}
