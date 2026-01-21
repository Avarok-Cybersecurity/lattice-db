//! Core error types for LatticeDB
//!
//! All errors are explicit - no silent failures allowed.

use thiserror::Error;

/// Top-level error type for LatticeDB operations
#[derive(Debug, Error)]
pub enum LatticeError {
    #[error("Storage error: {0}")]
    Storage(#[from] crate::storage::StorageError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Index error: {0}")]
    Index(#[from] IndexError),

    #[error("Cypher query error: {0}")]
    Cypher(#[from] crate::cypher::CypherError),

    #[error("Collection not found: {name}")]
    CollectionNotFound { name: String },

    #[error("Point not found: {id}")]
    PointNotFound { id: u64 },

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Serialization error: {message}")]
    Serialization { message: String },
}

/// Configuration validation errors
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Invalid parameter '{name}': {message}")]
    InvalidParameter { name: &'static str, message: String },

    #[error("Missing required parameter: {name}")]
    MissingParameter { name: &'static str },
}

/// Index-specific errors
#[derive(Debug, Error)]
pub enum IndexError {
    #[error("Index not built - call build() first")]
    NotBuilt,

    #[error("Index is empty - insert points first")]
    Empty,

    #[error("Invalid ef parameter: {ef} (must be >= k={k})")]
    InvalidEf { ef: usize, k: usize },
}

/// Convenience type alias for LatticeDB results
pub type LatticeResult<T> = Result<T, LatticeError>;
