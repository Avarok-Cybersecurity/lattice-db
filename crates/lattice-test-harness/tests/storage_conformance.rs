//! The suite passes on the harness's own in-memory store, and fails a store
//! that loses an acknowledged page.

use lattice_test_harness::conformance::controls::DropsLastPageOnSync;
use lattice_test_harness::conformance::crash;
use lattice_test_harness::{run_conformance, SharedState};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn shared_state_conforms() {
    run_conformance(SharedState::new).await.assert_conformant();
}

/// Negative control: a page lost on `sync` must fail the crash check
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn a_store_that_drops_the_last_page_on_sync_fails_the_crash_check() {
    let broken = DropsLastPageOnSync(SharedState::new());
    let outcome = crash::acknowledged_commits_survive_reopen(&broken).await;
    assert!(
        outcome.is_err(),
        "acknowledged commits whose pages were lost passed the crash check"
    );

    let report = run_conformance(|| DropsLastPageOnSync(SharedState::new())).await;
    assert!(report.failed_check("crash::acknowledged_commits_survive_reopen"));
}
