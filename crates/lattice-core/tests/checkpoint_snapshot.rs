//! Checkpoint snapshots alternate slots, clean up after themselves and never
//! read back damaged data as a collection.
//!
//! `SharedState` is the in-memory store both the writer and reader open, so
//! a later read sees exactly what an earlier write left.

use lattice_core::wal::snapshot::{
    read_snapshot, write_snapshot, SNAPSHOT_CHUNK_BYTES, SNAPSHOT_MANIFEST_KEY,
};
use lattice_core::LatticeError;
use lattice_test_harness::{FailingStorage, SharedState};
use std::sync::atomic::Ordering;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

const PAGE_POINTS_START: u64 = 1_000_000;

fn bytes(len: usize, seed: u8) -> Vec<u8> {
    (0..len).map(|i| (i as u8).wrapping_add(seed)).collect()
}

fn data_pages(state: &SharedState) -> usize {
    let pages = state.pages.read().unwrap();
    pages.keys().filter(|id| **id >= PAGE_POINTS_START).count()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn each_snapshot_replaces_the_last_and_leaves_no_stale_chunks() {
    let state = SharedState::new();
    let storage = state.mock();
    assert!(read_snapshot(&storage).await.unwrap().is_none());

    let large = bytes(SNAPSHOT_CHUNK_BYTES * 2 + 10, 1);
    write_snapshot(&storage, 5, &large).await.unwrap();
    write_snapshot(&storage, 9, &bytes(10, 2)).await.unwrap();
    assert_eq!(
        data_pages(&state),
        4,
        "three chunks in one slot, one in the other"
    );

    // Reuses the large snapshot's slot: its two extra chunks must go.
    let small = bytes(20, 3);
    write_snapshot(&storage, 12, &small).await.unwrap();
    assert_eq!(data_pages(&state), 2);

    let snapshot = read_snapshot(&storage).await.unwrap().unwrap();
    assert_eq!((snapshot.lsn, snapshot.bytes), (12, small));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn a_failed_snapshot_write_keeps_the_previous_one() {
    let state = SharedState::new();
    let first = bytes(SNAPSHOT_CHUNK_BYTES + 1, 4);
    write_snapshot(&state.mock(), 3, &first).await.unwrap();

    let failing = FailingStorage::new(&state);
    failing.write_page_fail_at.store(2, Ordering::SeqCst);
    let second = bytes(SNAPSHOT_CHUNK_BYTES * 3, 5);
    assert!(write_snapshot(&failing, 8, &second).await.is_err());

    let snapshot = read_snapshot(&state.mock()).await.unwrap().unwrap();
    assert_eq!((snapshot.lsn, snapshot.bytes), (3, first));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn a_damaged_snapshot_is_an_error_not_an_empty_collection() {
    let state = SharedState::new();
    write_snapshot(&state.mock(), 3, &bytes(100, 6))
        .await
        .unwrap();
    state
        .pages
        .write()
        .unwrap()
        .get_mut(&PAGE_POINTS_START)
        .unwrap()[7] ^= 0xff;

    let result = read_snapshot(&state.mock()).await;
    assert!(matches!(result, Err(LatticeError::DataCorruption { .. })));

    state.meta.write().unwrap().remove(SNAPSHOT_MANIFEST_KEY);
    assert!(read_snapshot(&state.mock()).await.unwrap().is_none());
}
