//! Crash recovery tests for WAL durability
//!
//! Simulates crash scenarios using MockStorage with shared Arc state.
//! Drop without close simulates a crash; reopen verifies recovery.

use lattice_core::wal::{Lsn, WalEntry, WriteAheadLog};
use lattice_core::types::point::Point;
use lattice_test_harness::SharedState;

#[tokio::test]
async fn test_crash_recovery_after_sync() {
    let state = SharedState::new();

    // Phase 1: Write entries and sync, then "crash" (drop)
    {
        let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();
        for i in 0..10 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, vec![i as f32])],
            })
            .await
            .unwrap();
        }
        wal.sync().await.unwrap();
    }

    // Phase 2: Reopen and verify all entries recovered
    {
        let wal = WriteAheadLog::open(state.mock()).await.unwrap();
        assert_eq!(wal.next_lsn(), 10, "Should recover next_lsn=10 from header");

        let entries = wal.read_from(0).await.unwrap();
        assert_eq!(entries.len(), 10, "All 10 entries should be recoverable");

        for (i, (lsn, _)) in entries.iter().enumerate() {
            assert_eq!(*lsn, i as u64);
        }
    }
}

#[tokio::test]
async fn test_crash_recovery_multiple_syncs() {
    let state = SharedState::new();

    // Phase 1: Multiple sync cycles
    {
        let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();

        for batch in 0..5 {
            for i in 0..3 {
                let id = batch * 3 + i;
                wal.append(&WalEntry::Upsert {
                    points: vec![Point::new_vector(id, vec![id as f32])],
                })
                .await
                .unwrap();
            }
            wal.sync().await.unwrap();
        }
    }

    // Phase 2: Verify recovery
    {
        let wal = WriteAheadLog::open(state.mock()).await.unwrap();
        assert_eq!(wal.next_lsn(), 15);

        let entries = wal.read_from(0).await.unwrap();
        assert_eq!(entries.len(), 15);
    }
}

#[tokio::test]
async fn test_corrupted_entry_skipped() {
    let state = SharedState::new();

    // Phase 1: Write 5 entries
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

    // Phase 2: Corrupt one entry in the page data
    {
        let pages = state.pages.read().unwrap();
        let page_data = pages.get(&0).unwrap().clone();
        drop(pages);

        // Corrupt checksum of first entry (bytes 4-7 are checksum)
        let mut corrupted = page_data;
        if corrupted.len() > 7 {
            corrupted[4] ^= 0xFF;
        }

        state.pages.write().unwrap().insert(0, corrupted);
    }

    // Phase 3: Reopen - should skip corrupted entry, recover rest
    {
        let wal = WriteAheadLog::open(state.mock()).await.unwrap();
        let entries = wal.read_from(0).await.unwrap();

        assert_eq!(entries.len(), 4, "Should recover 4 entries (1 corrupted, skipped)");
        assert_eq!(entries[0].0, 1);
        assert_eq!(entries[3].0, 4);
    }
}

#[tokio::test]
async fn test_abort_entry_skipped_on_replay() {
    let state = SharedState::new();

    // Phase 1: Write entries including an abort
    {
        let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();

        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(1, vec![1.0])],
        })
        .await
        .unwrap();

        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(2, vec![2.0])],
        })
        .await
        .unwrap();

        wal.append(&WalEntry::Abort { aborted_lsn: 1 })
            .await
            .unwrap();

        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(3, vec![3.0])],
        })
        .await
        .unwrap();

        wal.sync().await.unwrap();
    }

    // Phase 2: Read entries and verify abort logic
    {
        let wal = WriteAheadLog::open(state.mock()).await.unwrap();
        let entries = wal.read_from(0).await.unwrap();

        assert_eq!(entries.len(), 4);

        let aborted: std::collections::HashSet<Lsn> = entries
            .iter()
            .filter_map(|(_, e)| match e {
                WalEntry::Abort { aborted_lsn } => Some(*aborted_lsn),
                _ => None,
            })
            .collect();

        assert!(aborted.contains(&1), "LSN 1 should be in aborted set");
        assert!(!aborted.contains(&0), "LSN 0 should NOT be aborted");

        let replay_count = entries
            .iter()
            .filter(|(lsn, entry)| {
                !aborted.contains(lsn)
                    && !matches!(entry, WalEntry::Abort { .. } | WalEntry::Checkpoint { .. })
            })
            .count();

        assert_eq!(replay_count, 2, "Should replay 2 entries (LSN 0 and 3)");
    }
}
