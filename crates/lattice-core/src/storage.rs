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
/// This trait abstracts the physical storage medium. The core engine functions
/// identically whether backed by:
/// - In-memory HashMap (testing, ephemeral)
/// - Disk files (server deployment)
/// - Browser OPFS (client-side WASM)
///
/// # Page-Based Model
///
/// Storage is organized into fixed-size pages identified by `u64` IDs.
/// This model maps naturally to:
/// - Memory: `HashMap<u64, Vec<u8>>`
/// - Disk: offset = page_id * PAGE_SIZE
/// - OPFS: same offset-based access
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

/// WASM version of storage trait (no Send + Sync bounds)
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
