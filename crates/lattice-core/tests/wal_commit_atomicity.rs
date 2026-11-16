//! The shared WAL commits all-or-nothing, on every target.
//!
//! These exercise `WriteAheadLog` itself, which the native and the wasm
//! durable engines both use. `SharedState` is an in-memory store that outlives
//! each `WriteAheadLog`, so dropping one and opening another is a restart;
//! `FailingStorage` wraps the same state and fails a chosen storage call.

use lattice_core::types::point::Point;
use lattice_core::wal::{ArchivedWalEntry, WalEntry, WriteAheadLog};
use lattice_test_harness::{FailingStorage, SharedState};
use std::sync::atomic::Ordering;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

fn upsert(id: u64, dim: usize) -> WalEntry {
    WalEntry::Upsert {
        points: vec![Point::new_vector(id, vec![id as f32; dim])],
    }
}

fn upserted_ids(entries: &[(u64, WalEntry)]) -> Vec<(u64, u64)> {
    entries
        .iter()
        .filter_map(|(lsn, entry)| match entry {
            WalEntry::Upsert { points } => Some((*lsn, points[0].id)),
            _ => None,
        })
        .collect()
}

async fn commit_baseline(state: &SharedState, ids: std::ops::Range<u64>) {
    let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();
    for id in ids {
        wal.append(&upsert(id, 1)).await.unwrap();
    }
    wal.sync().await.unwrap();
}

/// A commit whose page write failed must not ride along with the next one
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn failed_sync_is_not_persisted_by_the_next_commit() {
    let state = SharedState::new();
    commit_baseline(&state, 0..3).await;

    {
        let failing = FailingStorage::new(&state);
        let trigger = failing.write_page_fail_at.clone();
        trigger.store(1, Ordering::SeqCst);
        let mut wal = WriteAheadLog::open(failing).await.unwrap();

        wal.append(&upsert(99, 1)).await.unwrap();
        assert!(
            wal.sync().await.is_err(),
            "the injected failure must surface"
        );
        assert_eq!(wal.next_lsn(), 3, "the failed entry's LSN is given back");

        // The storage recovers; the next commit must carry only its own entry.
        trigger.store(0, Ordering::SeqCst);
        wal.append(&upsert(7, 1)).await.unwrap();
        wal.sync().await.unwrap();
    }

    let wal = WriteAheadLog::open(state.mock()).await.unwrap();
    assert_eq!(
        upserted_ids(&wal.read_from(0).await.unwrap()),
        vec![(0, 0), (1, 1), (2, 2), (3, 7)],
        "the failed entry must not be visible after reopen"
    );
}

/// The page was written but the header naming it was not: not committed
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn page_written_without_its_header_is_not_replayed() {
    let state = SharedState::new();
    commit_baseline(&state, 0..3).await;

    {
        let failing = FailingStorage::new(&state);
        failing.set_meta_fail_at.store(1, Ordering::SeqCst);
        let mut wal = WriteAheadLog::open(failing).await.unwrap();
        wal.append(&upsert(10, 1)).await.unwrap();
        assert!(wal.sync().await.is_err());
    }

    let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();
    assert_eq!(
        upserted_ids(&wal.read_from(0).await.unwrap()),
        vec![(0, 0), (1, 1), (2, 2)]
    );

    // The orphaned bytes are overwritten, not prefixed to the next commit.
    wal.append(&upsert(11, 1)).await.unwrap();
    wal.sync().await.unwrap();
    let wal = WriteAheadLog::open(state.mock()).await.unwrap();
    assert_eq!(
        upserted_ids(&wal.read_from(0).await.unwrap()),
        vec![(0, 0), (1, 1), (2, 2), (3, 11)]
    );
}

/// Deleting pages before a checkpoint must not renumber what remains
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn truncated_log_keeps_its_lsn_numbering() {
    let state = SharedState::new();
    let mut wal = WriteAheadLog::open(state.mock()).await.unwrap();
    // ~40 KB per entry: three 256 KB pages after about twenty entries.
    let mut written = 0;
    while state.pages.read().unwrap().len() < 3 {
        wal.append(&upsert(written, 10_000)).await.unwrap();
        wal.sync().await.unwrap();
        written += 1;
    }
    let checkpoint_lsn = wal.next_lsn();
    wal.checkpoint(checkpoint_lsn).await.unwrap();
    wal.truncate_before(checkpoint_lsn).await.unwrap();
    assert!(
        !state.pages.read().unwrap().contains_key(&0),
        "the first page holds only checkpointed entries and is deleted"
    );
    wal.append(&upsert(1000, 1)).await.unwrap();
    wal.sync().await.unwrap();

    let wal = WriteAheadLog::open(state.mock()).await.unwrap();
    assert_eq!(
        upserted_ids(&wal.read_from(checkpoint_lsn).await.unwrap()),
        vec![(checkpoint_lsn + 1, 1000)],
        "LSNs after truncation must match the ones assigned"
    );
}

/// Adding `DefineRelation` must not change how existing entries archive
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn define_relation_keeps_the_archived_entry_size() {
    // Measured at v0.4.0, before the variant existed. rkyv reads the root
    // object from the end of the buffer, so a larger archived enum would
    // misread every entry already on disk.
    assert_eq!(std::mem::size_of::<ArchivedWalEntry>(), 32);
}
