//! Storage abstraction layer (SBIO)
//!
//! This module defines the `LatticeStorage` trait that abstracts all storage operations.
//! The core engine uses this trait without knowing the underlying storage medium.
//!
//! # Implementations
//!
//! - `MemStorage`: In-memory HashMap (for testing and ephemeral use)
//! - `DiskStorage`: File-based storage using tokio::fs
//! - `OpfsStorage`: Browser Origin Private File System
//!
//! A backend outside this workspace proves itself with the conformance suite
//! in `latticedb-test-harness` (`lattice_test_harness::conformance`), which
//! runs the checks below and the WAL's crash consistency against it.
//!
//! # The contract the WAL relies on
//!
//! - **Metadata** is a map from key to bytes. A value reads back exactly,
//!   empty included; a key never written, or deleted, reads as `None`.
//!   `set_meta` must be **atomic per key**: a reader sees the whole old value
//!   or the whole new one, never a mix. The WAL commits by writing one key
//!   ([`crate::wal::WAL_HEADER_KEY`]) and a checkpoint switches snapshots by
//!   writing another, so writes of different keys need not be atomic together.
//! - **Pages** are byte blobs under any `u64` id, read back exactly as last
//!   written (no padding, no stale tail). A page never written is
//!   [`StorageError::PageNotFound`], never empty bytes: that is how the WAL and
//!   snapshots tell a missing page from an empty one. Pages must hold at least
//!   [`crate::wal::WAL_PAGE_MAX_BYTES`] (256 KiB); a single commit larger than
//!   that is written as one page of its own size.
//! - **Sync**: once `sync` returns `Ok`, everything written before it survives
//!   a restart. A backend may make writes durable sooner, never later.
//! - **Failures**: a call that returns `Err` must have changed nothing. A
//!   `sync` that fails after its page and header writes already took effect
//!   cannot be undone by the WAL, so on a backend whose writes are durable on
//!   return, `sync` should not fail.
//!
//! # `Send` on wasm32
//!
//! On native targets the trait requires `Send + Sync` and its futures are
//! `Send` (`#[async_trait]`), so engines can move across threads. On
//! `wasm32-unknown-unknown` it is declared with `#[async_trait(?Send)]` and has
//! no `Send + Sync` supertraits: there is one thread, and a backend there
//! usually holds JavaScript handles (`JsValue`, a Durable Object's storage, an
//! OPFS handle) that are not `Send`. An implementation for both targets uses
//! the same attributes as the in-tree backends:
//!
//! ```ignore
//! #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
//! #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
//! impl LatticeStorage for MyStorage { /* ... */ }
//! ```
//!
//! # Cloudflare Durable Objects
//!
//! A Durable Object's transactional storage maps onto this contract, with
//! these constraints:
//!
//! - Use a **SQLite-backed** Durable Object. The key-value API of a KV-backed
//!   object caps values at 128 KiB, below one WAL page; a SQLite-backed one
//!   allows 8 MB of key and value together, which also bounds the largest
//!   single commit. Snapshot chunks are 64 KiB
//!   ([`crate::wal::snapshot::SNAPSHOT_CHUNK_BYTES`]).
//! - Each `set_meta` is one `put` of one key, which the storage writes
//!   atomically, as the per-key atomicity above requires.
//! - Encode page ids losslessly, for example as fixed-width hex string keys:
//!   snapshot chunks already sit above `2^32`, and the trait allows any `u64`,
//!   past the `2^53` a JavaScript number holds exactly.
//! - A failed `put` changes nothing, and writes are durable before the
//!   object's next output, so `sync` can await the storage's own `sync()`.
//! - The wasm32 `DurableEngine` needs no clock, threads or runtime; call
//!   `checkpoint()` from an alarm when `should_checkpoint()` says so.

use async_trait::async_trait;
use thiserror::Error;

/// Raw page of bytes - the fundamental storage unit
///
/// Pages are aligned and sized for efficient I/O operations.
/// The page size is determined by the storage implementation.
pub type Page = Vec<u8>;

/// Storage operation errors
///
/// All errors are explicit - storage operations never silently fail.
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Page not found: {page_id}")]
    PageNotFound { page_id: u64 },

    #[error("I/O error: {message}")]
    Io { message: String },

    #[error("Serialization error: {message}")]
    Serialization { message: String },

    #[error("Storage is read-only")]
    ReadOnly,

    #[error("Storage capacity exceeded")]
    CapacityExceeded,

    /// File format corruption detected
    #[error("Corrupted file at offset {offset}: {reason}")]
    CorruptedFile {
        /// Byte offset where corruption was detected
        offset: usize,
        /// Description of the corruption
        reason: String,
    },

    /// Invalid file format (wrong magic, version, etc.)
    #[error("Invalid file format: expected {expected}, found {found}")]
    InvalidFormat { expected: String, found: String },

    /// Feature not yet implemented
    #[error("Feature not implemented: {feature}")]
    NotImplemented { feature: &'static str },
}

/// Convenience type alias for storage results
pub type StorageResult<T> = Result<T, StorageError>;

// Storage trait definition with platform-specific bounds
// On native: requires Send + Sync for thread safety
// On WASM: no Send/Sync since JavaScript is single-threaded

/// Abstract storage interface (SBIO boundary)
///
/// The behaviour every backend must have is in the [module docs](self); on
/// native targets the trait is `Send + Sync` with `Send` futures.
///
/// This trait abstracts the physical storage medium. The core engine functions
/// identically whether backed by:
/// - In-memory HashMap (testing, ephemeral)
/// - Disk files (server deployment)
/// - Browser OPFS (client-side WASM)
///
/// # Page-Based Model
///
/// Storage is organized into byte pages identified by `u64` IDs.
/// This model maps naturally to:
/// - Memory: `HashMap<u64, Vec<u8>>`
/// - Disk: one file per page
/// - OPFS: offset-based access
///
/// # Metadata
///
/// Separate key-value metadata storage for collection configs,
/// index state, and other non-page data.
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait LatticeStorage: Send + Sync + 'static {
    /// Retrieve metadata by key
    async fn get_meta(&self, key: &str) -> StorageResult<Option<Vec<u8>>>;
    /// Store metadata
    async fn set_meta(&self, key: &str, value: &[u8]) -> StorageResult<()>;
    /// Delete metadata key
    async fn delete_meta(&self, key: &str) -> StorageResult<()>;
    /// Read a page by ID
    async fn read_page(&self, page_id: u64) -> StorageResult<Page>;
    /// Write a page (create or overwrite)
    async fn write_page(&self, page_id: u64, data: &[u8]) -> StorageResult<()>;
    /// Check if a page exists
    async fn page_exists(&self, page_id: u64) -> StorageResult<bool>;
    /// Delete a page
    async fn delete_page(&self, page_id: u64) -> StorageResult<()>;
    /// Flush pending writes to durable storage
    async fn sync(&self) -> StorageResult<()>;
}

/// Abstract storage interface (SBIO boundary), wasm32 form
///
/// The same contract as on native targets (see the [module docs](self)), with
/// `?Send` futures and no `Send + Sync` bounds, so a backend can hold
/// JavaScript handles.
#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait LatticeStorage: 'static {
    /// Retrieve metadata by key
    async fn get_meta(&self, key: &str) -> StorageResult<Option<Vec<u8>>>;
    /// Store metadata
    async fn set_meta(&self, key: &str, value: &[u8]) -> StorageResult<()>;
    /// Delete metadata key
    async fn delete_meta(&self, key: &str) -> StorageResult<()>;
    /// Read a page by ID
    async fn read_page(&self, page_id: u64) -> StorageResult<Page>;
    /// Write a page (create or overwrite)
    async fn write_page(&self, page_id: u64, data: &[u8]) -> StorageResult<()>;
    /// Check if a page exists
    async fn page_exists(&self, page_id: u64) -> StorageResult<bool>;
    /// Delete a page
    async fn delete_page(&self, page_id: u64) -> StorageResult<()>;
    /// Flush pending writes to durable storage
    async fn sync(&self) -> StorageResult<()>;
}
