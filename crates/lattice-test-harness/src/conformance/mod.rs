//! A conformance suite for `LatticeStorage` backends
//!
//! Every backend the engine runs on has to behave alike in the ways the WAL
//! and checkpoint snapshots depend on. This suite states those behaviours as
//! checks any backend can be run against: the in-crate ones (memory, disk) and
//! downstream adapters such as a Durable Object's SQLite storage.
//!
//! - [`meta`]: metadata blobs round-trip exactly, per key, and deletes remove
//! - [`pages`]: pages round-trip exactly at WAL size, under any `u64` id
//! - [`sync`]: what was written and synced is there after a reopen
//! - [`crash`]: with the WAL on top, acknowledged commits survive a reopen and
//!   a commit whose storage call failed is never visible after one
//!
//! A backend plugs in through [`StorageFixture`]: each call to
//! [`StorageFixture::open`] is a process restart over the same durable state.
//! [`run_conformance`] runs every check against fresh fixtures and reports
//! each failure by name; checks return their failure rather than panicking, so
//! the suite can run where a panic aborts the whole module (wasm32) and its
//! own negative controls can assert that a broken backend is caught.
//!
//! The failures the crash checks inject return before reaching the backend,
//! so they model a storage call that failed without taking effect. A backend
//! must uphold that itself: a call that returns `Err` must have changed
//! nothing.

pub mod controls;
pub mod crash;
mod fixtures;
pub mod meta;
pub mod pages;
pub mod sync;

pub use fixtures::{Handle, Reopen};

use lattice_core::storage::StorageResult;
use lattice_core::LatticeStorage;
use std::fmt::Display;

/// One check's outcome; `Err` describes what the backend did wrong
pub type CheckResult = Result<(), String>;

/// A backend under test, opened over durable state that outlives each handle
///
/// Dropping a handle and calling `open` again must behave like the process
/// restarting: only what the backend made durable is seen. Each fixture the
/// suite is given must start empty.
#[async_trait::async_trait(?Send)]
pub trait StorageFixture {
    /// The backend's storage handle
    type Storage: LatticeStorage;

    /// Open a handle onto this fixture's durable state
    async fn open(&self) -> StorageResult<Self::Storage>;
}

/// Which checks passed and which failed, with each failure's description
#[derive(Debug, Default)]
pub struct ConformanceReport {
    /// Names of the checks that passed
    pub passed: Vec<&'static str>,
    /// Names and failure descriptions of the checks that failed
    pub failed: Vec<(&'static str, String)>,
}

impl ConformanceReport {
    fn record(&mut self, name: &'static str, result: CheckResult) {
        match result {
            Ok(()) => self.passed.push(name),
            Err(detail) => self.failed.push((name, detail)),
        }
    }

    /// Whether the named check failed
    pub fn failed_check(&self, name: &str) -> bool {
        self.failed.iter().any(|(failed, _)| *failed == name)
    }

    /// Panic listing every failed check, if any failed
    pub fn assert_conformant(&self) {
        if self.failed.is_empty() {
            return;
        }
        let lines: Vec<String> = self
            .failed
            .iter()
            .map(|(name, detail)| format!("  {}: {}", name, detail))
            .collect();
        panic!(
            "{} of {} LatticeStorage conformance checks failed:\n{}",
            self.failed.len(),
            self.failed.len() + self.passed.len(),
            lines.join("\n")
        );
    }
}

/// Run every check, each against a fresh fixture from `make`
pub async fn run_conformance<F, M>(mut make: M) -> ConformanceReport
where
    F: StorageFixture,
    M: FnMut() -> F,
{
    let mut report = ConformanceReport::default();
    macro_rules! run {
        ($($module:ident::$check:ident),* $(,)?) => {
            $(report.record(
                concat!(stringify!($module), "::", stringify!($check)),
                $module::$check(&make()).await,
            );)*
        };
    }
    run!(
        meta::missing_key_is_none,
        meta::value_round_trips_exactly,
        meta::set_replaces_the_whole_value,
        meta::keys_are_independent,
        meta::delete_removes_the_key,
        pages::missing_page_is_not_found,
        pages::page_round_trips_exactly,
        pages::overwrite_replaces_the_whole_page,
        pages::wal_sized_page_round_trips,
        pages::page_ids_are_independent,
        pages::delete_removes_the_page,
        sync::sync_on_an_empty_store_succeeds,
        sync::synced_writes_survive_reopen,
        sync::synced_deletes_survive_reopen,
        crash::acknowledged_commits_survive_reopen,
        crash::failed_page_write_is_not_visible_after_reopen,
        crash::failed_header_write_is_not_visible_after_reopen,
        crash::failed_rotating_page_write_is_not_visible_after_reopen,
        crash::failed_rotating_header_write_is_not_visible_after_reopen,
    );
    report
}

/// Attach what was being done to a storage or WAL error
pub(crate) trait Context<T> {
    fn ctx(self, what: &str) -> Result<T, String>;
}

impl<T, E: Display> Context<T> for Result<T, E> {
    fn ctx(self, what: &str) -> Result<T, String> {
        self.map_err(|e| format!("{} failed: {}", what, e))
    }
}

/// Fail with `detail` unless `holds`
pub(crate) fn ensure(holds: bool, detail: impl FnOnce() -> String) -> CheckResult {
    if holds {
        Ok(())
    } else {
        Err(detail())
    }
}
