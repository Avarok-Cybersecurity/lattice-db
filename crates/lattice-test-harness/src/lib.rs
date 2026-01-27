//! Test infrastructure for LatticeDB ACID disruption testing
//!
//! Provides reusable mocks and helpers for crash recovery,
//! I/O failure injection, and durable engine construction.

pub mod shared_state;
pub mod failing_storage;
pub mod helpers;

pub use shared_state::{SharedState, MockStorage};
pub use failing_storage::FailingStorage;
pub use helpers::{make_point, make_config, open_engine};
