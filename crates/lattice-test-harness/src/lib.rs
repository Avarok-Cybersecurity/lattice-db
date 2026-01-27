//! Test infrastructure for LatticeDB ACID disruption testing
//!
//! Provides reusable mocks and helpers for crash recovery,
//! I/O failure injection, and durable engine construction.
//!
//! Only available on native targets (not WASM).

#[cfg(not(target_arch = "wasm32"))]
pub mod failing_storage;
#[cfg(not(target_arch = "wasm32"))]
pub mod helpers;
#[cfg(not(target_arch = "wasm32"))]
pub mod shared_state;

#[cfg(not(target_arch = "wasm32"))]
pub use failing_storage::FailingStorage;
#[cfg(not(target_arch = "wasm32"))]
pub use helpers::{make_config, make_point, open_engine};
#[cfg(not(target_arch = "wasm32"))]
pub use shared_state::{MockStorage, SharedState};
