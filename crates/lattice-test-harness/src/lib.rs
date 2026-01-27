//! Test infrastructure for LatticeDB ACID disruption testing
//!
//! Provides reusable mocks and helpers for crash recovery,
//! I/O failure injection, and durable engine construction.

pub mod failing_storage;
pub mod helpers;
pub mod shared_state;

pub use failing_storage::FailingStorage;
pub use helpers::{make_config, make_point, open_engine};
pub use shared_state::{MockStorage, SharedState};
