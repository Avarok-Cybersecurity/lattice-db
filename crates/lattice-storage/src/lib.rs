//! LatticeDB Storage Implementations
//!
//! This crate provides concrete implementations of the `LatticeStorage` trait
//! for different storage backends.
//!
//! # Available Backends
//!
//! - `MemStorage`: In-memory HashMap (for testing and ephemeral use)
//! - `DiskStorage`: File-based storage using tokio::fs (native only)
//! - `OpfsStorage`: Browser Origin Private File System (WASM only)

// Configure WASM tests to run in browser
#[cfg(all(target_arch = "wasm32", test))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

pub mod mem;

#[cfg(feature = "native")]
pub mod disk;

#[cfg(feature = "wasm")]
pub mod opfs;

// Re-export the trait from lattice-core
pub use lattice_core::{LatticeStorage, Page, StorageError, StorageResult};

// Re-export implementations
pub use mem::MemStorage;

#[cfg(feature = "native")]
pub use disk::DiskStorage;

#[cfg(feature = "wasm")]
pub use opfs::OpfsStorage;
