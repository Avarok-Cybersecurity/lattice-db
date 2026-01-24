//! LatticeDB Core - Pure business logic with no I/O dependencies
//!
//! This crate implements the core functionality of LatticeDB:
//! - Data types (Point, Edge) with zero-copy serialization
//! - HNSW index algorithm for vector similarity search
//! - Graph traversal logic
//! - Collection management
//!
//! # SBIO Architecture
//!
//! This crate follows strict Separation of Business Logic and I/O (SBIO).
//! It defines traits for storage and transport but never imports I/O primitives.
//! All external interactions occur through injected trait implementations.

// Allow some clippy lints that are too pedantic for this crate
#![allow(clippy::needless_range_loop)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::let_and_return)]
#![allow(clippy::or_fun_call)]
#![allow(clippy::vec_init_then_push)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::manual_strip)]
#![allow(dead_code)]
#![allow(clippy::never_loop)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::unnecessary_map_or)]

// Configure WASM tests to run in browser (Node.js has issues with wasm-bindgen-test)
#[cfg(all(target_arch = "wasm32", test))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

pub mod cypher;
pub mod engine;
pub mod error;
pub mod graph;
pub mod index;
pub mod parallel;
pub mod storage;
pub mod sync;
pub mod transport;
pub mod types;

// Re-export commonly used types
pub use engine::collection::{CollectionEngine, EdgeInfo, TraversalResult, UpsertResult};
pub use error::{ConfigError, IndexError, LatticeError, LatticeResult};
pub use index::hnsw::HnswIndex;
pub use index::quantization::QuantizedVector;
pub use storage::{LatticeStorage, Page, StorageError, StorageResult};
pub use transport::{LatticeRequest, LatticeResponse, LatticeTransport};
pub use types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
pub use types::point::{Edge, Point, PointId, Vector};
pub use types::query::{ScrollQuery, ScrollResult, SearchQuery, SearchResult};
pub use types::value::CypherValue;

// Sync utilities
pub use sync::{cmp_f32, cmp_f32_reverse, SyncCell};
#[cfg(not(target_arch = "wasm32"))]
pub use sync::{LockExt, MutexExt};
