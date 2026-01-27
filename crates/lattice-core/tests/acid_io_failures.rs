//! I/O failure injection tests for WAL durability
//!
//! Uses FailingStorage to inject I/O errors at precise points in the
//! WAL write path, verifying committed data survives failures.

use lattice_core::types::point::Point;
use lattice_core::wal::{WalEntry, WriteAheadLog};
use lattice_test_harness::{FailingStorage, SharedState};
use std::sync::atomic::Ordering;

/// Committed entries survive a write_page failure on the next write
#[tokio::test]
async fn test_write_page_failure_preserves_committed() {
    let state = SharedState::new();

    // Phase 1: Write and sync 5 entries (committed)
    {
        let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();
        for i in 0..5 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, vec![i as f32])],
            })
            .await
            .unwrap();
        }
        wal.sync().await.unwrap();
    }

    // Phase 2: Attempt more writes with failing storage
    {
        let failing = FailingStorage::new(&state);
        // Fail immediately on next write_page call
        failing.write_page_fail_at.store(1, Ordering::SeqCst);

        let mut wal = WriteAheadLog::open(failing).await.unwrap();
        let result = wal
            .append(&WalEntry::Upsert {
                points: vec![Point::new_vector(99, vec![99.0])],
            })
            .await;
        // The append might succeed (buffered) or fail
        // Either way, the sync should fail
        if result.is_ok() {
            let _ = wal.sync().await; // may fail
        }
        // Drop = crash
    }

    // Phase 3: Reopen and verify original 5 entries intact
    {
        let wal = WriteAheadLog::open(state.mock()).await.unwrap();
        let entries = wal.read_from(0).await.unwrap();
        assert!(
            entries.len() >= 5,
            "At least 5 committed entries must survive, got {}",
            entries.len()
        );
        for i in 0..5 {
            assert_eq!(entries[i].0, i as u64);
        }
    }
}

/// Header (set_meta) failure after page write preserves last good header
#[tokio::test]
async fn test_meta_failure_after_page_write() {
    let state = SharedState::new();

    // Phase 1: Commit baseline
    {
        let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();
        for i in 0..3 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, vec![i as f32])],
            })
            .await
            .unwrap();
        }
        wal.sync().await.unwrap();
    }

    // Phase 2: Fail set_meta during sync
    {
        let failing = FailingStorage::new(&state);
        // set_meta is called during sync for header update — fail on 1st call
        failing.set_meta_fail_at.store(1, Ordering::SeqCst);

        let mut wal = WriteAheadLog::open(failing).await.unwrap();
        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(10, vec![10.0])],
        })
        .await
        .unwrap();
        let _ = wal.sync().await; // should fail at header write
    }

    // Phase 3: Recovery should see at least the original 3
    {
        let wal = WriteAheadLog::open(state.mock()).await.unwrap();
        let entries = wal.read_from(0).await.unwrap();
        assert!(
            entries.len() >= 3,
            "Committed entries must survive header failure, got {}",
            entries.len()
        );
    }
}

/// Sync (fsync) failure preserves previously committed data
#[tokio::test]
async fn test_sync_failure_preserves_committed() {
    let state = SharedState::new();

    // Phase 1: Commit baseline
    {
        let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();
        for i in 0..5 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, vec![i as f32])],
            })
            .await
            .unwrap();
        }
        wal.sync().await.unwrap();
    }

    // Phase 2: Fail sync on next cycle
    {
        let failing = FailingStorage::new(&state);
        failing.sync_fail_at.store(1, Ordering::SeqCst);

        let mut wal = WriteAheadLog::open(failing).await.unwrap();
        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(50, vec![50.0])],
        })
        .await
        .unwrap();
        let _ = wal.sync().await; // sync fails
    }

    // Phase 3: Original 5 intact
    {
        let wal = WriteAheadLog::open(state.mock()).await.unwrap();
        let entries = wal.read_from(0).await.unwrap();
        assert!(
            entries.len() >= 5,
            "Committed entries must survive sync failure, got {}",
            entries.len()
        );
    }
}

/// Transient I/O errors followed by success recovers correctly
#[tokio::test]
async fn test_repeated_failures_then_success() {
    let state = SharedState::new();

    // Phase 1: Commit baseline
    {
        let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();
        for i in 0..3 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, vec![i as f32])],
            })
            .await
            .unwrap();
        }
        wal.sync().await.unwrap();
    }

    // Phase 2: Fail twice, then succeed
    for _ in 0..2 {
        let failing = FailingStorage::new(&state);
        failing.write_page_fail_at.store(1, Ordering::SeqCst);
        let mut wal = WriteAheadLog::open(failing).await.unwrap();
        let _ = wal
            .append(&WalEntry::Upsert {
                points: vec![Point::new_vector(99, vec![99.0])],
            })
            .await;
        // Drop = crash after failure
    }

    // Phase 3: Succeed with clean storage
    {
        let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();
        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(100, vec![100.0])],
        })
        .await
        .unwrap();
        wal.sync().await.unwrap();
    }

    // Phase 4: Verify baseline + successful write
    {
        let wal = WriteAheadLog::open(state.mock()).await.unwrap();
        let entries = wal.read_from(0).await.unwrap();
        assert!(
            entries.len() >= 4,
            "Should have baseline 3 + 1 new entry, got {}",
            entries.len()
        );
    }
}
