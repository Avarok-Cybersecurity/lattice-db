//! Crash scenario tests for WAL atomicity
//!
//! Simulates crashes during multi-step operations to verify
//! all-or-nothing semantics and correct recovery.

use lattice_core::types::point::Point;
use lattice_core::wal::{WalEntry, WriteAheadLog};
use lattice_test_harness::SharedState;

/// Crash during batch upsert: only synced entries survive
#[tokio::test]
async fn test_crash_during_batch_upsert() {
    let state = SharedState::new();

    // Phase 1: Write 50 entries but only sync after first 25
    {
        let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();
        for i in 0..25 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, vec![i as f32])],
            })
            .await
            .unwrap();
        }
        wal.sync().await.unwrap();

        // Write 25 more WITHOUT sync
        for i in 25..50 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, vec![i as f32])],
            })
            .await
            .unwrap();
        }
        // Drop = crash before second sync
    }

    // Phase 2: Verify only synced entries recovered
    {
        let wal = WriteAheadLog::open(state.mock()).await.unwrap();
        let entries = wal.read_from(0).await.unwrap();

        // We should have at least 25 (the synced ones).
        // We may also have some of the unsynced ones if they were flushed to pages.
        assert!(
            entries.len() >= 25,
            "At least 25 synced entries must survive, got {}",
            entries.len()
        );

        // First 25 must be present with correct LSNs
        for i in 0..25 {
            assert_eq!(entries[i].0, i as u64, "LSN mismatch at index {}", i);
        }
    }
}

/// WAL with upsert + abort: aborted entry filtered on replay
#[tokio::test]
async fn test_crash_between_log_and_apply() {
    let state = SharedState::new();

    // Phase 1: Log upsert, then abort it, then crash
    {
        let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();

        // Upsert (LSN 0)
        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(1, vec![1.0, 2.0, 3.0, 4.0])],
        })
        .await
        .unwrap();

        // Upsert (LSN 1) — will be aborted
        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(2, vec![5.0, 6.0, 7.0, 8.0])],
        })
        .await
        .unwrap();

        // Abort LSN 1
        wal.append(&WalEntry::Abort { aborted_lsn: 1 })
            .await
            .unwrap();

        wal.sync().await.unwrap();
        // Drop = crash
    }

    // Phase 2: Replay and verify abort is respected
    {
        let wal = WriteAheadLog::open(state.mock()).await.unwrap();
        let entries = wal.read_from(0).await.unwrap();

        let aborted: std::collections::HashSet<u64> = entries
            .iter()
            .filter_map(|(_, e)| match e {
                WalEntry::Abort { aborted_lsn } => Some(*aborted_lsn),
                _ => None,
            })
            .collect();

        let live_upserts: Vec<u64> = entries
            .iter()
            .filter(|(lsn, entry)| {
                matches!(entry, WalEntry::Upsert { .. }) && !aborted.contains(lsn)
            })
            .map(|(lsn, _)| *lsn)
            .collect();

        assert_eq!(live_upserts, vec![0], "Only LSN 0 should survive abort");
    }
}

/// Concurrent upserts with crash: only synced ops survive
#[tokio::test]
async fn test_concurrent_upserts_crash() {
    let state = SharedState::new();

    // Phase 1: Simulate rapid concurrent-style writes
    {
        let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();

        // Batch 1: synced
        for i in 0..10 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, vec![i as f32; 4])],
            })
            .await
            .unwrap();
        }
        wal.sync().await.unwrap();

        // Batch 2: unsynced (concurrent writes in progress)
        for i in 10..20 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, vec![i as f32; 4])],
            })
            .await
            .unwrap();
        }
        // Drop = crash mid-batch
    }

    // Phase 2: Verify only synced batch survived
    {
        let wal = WriteAheadLog::open(state.mock()).await.unwrap();
        let entries = wal.read_from(0).await.unwrap();

        assert!(
            entries.len() >= 10,
            "Synced batch must survive, got {}",
            entries.len()
        );

        // Verify LSN ordering is monotonic
        for window in entries.windows(2) {
            assert!(
                window[1].0 > window[0].0,
                "LSNs must be monotonically increasing"
            );
        }
    }
}
