//! Engine module - high-level APIs for collection and point management
//!
//! This module provides the main entry points for LatticeDB operations:
//! - `CollectionEngine`: Manages a single collection (points, search, graph)
//! - `LatticeEngine`: Manages multiple collections

pub mod collection;

// Native-only: Async indexer for background HNSW updates
#[cfg(not(target_arch = "wasm32"))]
pub mod async_indexer;

pub use collection::CollectionEngine;
