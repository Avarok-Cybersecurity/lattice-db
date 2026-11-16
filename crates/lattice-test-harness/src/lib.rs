//! Test infrastructure for LatticeDB ACID disruption testing
//!
//! Provides reusable mocks and helpers for crash recovery,
//! I/O failure injection, and durable engine construction.
//!
//! [`conformance`] is a `LatticeStorage` conformance suite that any backend,
//! including one outside this workspace, can run against itself.
//!
//! The storage doubles and the suite build on every target; the engine helpers are
//! native-only because they use the native engine builder.

pub mod conformance;
pub mod failing_storage;
#[cfg(not(target_arch = "wasm32"))]
pub mod helpers;
pub mod shared_state;

pub use conformance::{run_conformance, ConformanceReport, StorageFixture};
pub use failing_storage::FailingStorage;
#[cfg(not(target_arch = "wasm32"))]
pub use helpers::{make_config, make_point, open_engine};
pub use shared_state::{MockStorage, SharedState};
