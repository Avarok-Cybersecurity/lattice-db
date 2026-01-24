//! Core error types for LatticeDB
//!
//! All errors are explicit - no silent failures allowed.
//!
//! # Error Code Scheme
//!
//! | Range | Category |
//! |-------|----------|
//! | 10xxx | Storage errors |
//! | 20xxx | Index errors |
//! | 30xxx | Cypher errors |
//! | 40xxx | Configuration errors |
//! | 50xxx | Internal/Runtime errors |
//! | 60xxx | Data corruption errors |

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

    /// Internal error - lock poisoning, invariant violations, thread failures
    ///
    /// These errors indicate bugs or system failures, not user errors.
    /// Error codes in range 50xxx.
    #[error("Internal error [{code}]: {message}")]
    Internal {
        /// Error code (50xxx range)
        code: u32,
        /// Human-readable description
        message: String,
    },

    /// Data corruption detected during read/deserialization
    ///
    /// These errors indicate storage corruption or malformed data.
    /// Error codes in range 60xxx.
    #[error("Data corruption [{code}]: {context}")]
    DataCorruption {
        /// Error code (60xxx range)
        code: u32,
        /// Description of what was corrupted
        context: String,
    },
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

    /// Index has no entry point (empty or corrupted)
    #[error("Index has no entry point - insert points first or rebuild index")]
    NoEntryPoint,

    /// Invalid vector value (NaN or infinite)
    #[error("Invalid vector value at dimension {dimension}: {value_type}")]
    InvalidVectorValue {
        dimension: usize,
        value_type: &'static str,
    },

    /// Graph structure inconsistency during traversal
    #[error("Graph inconsistency: node {node_id} references missing neighbor {neighbor_id}")]
    GraphInconsistency { node_id: u64, neighbor_id: u64 },
}

/// Convenience type alias for LatticeDB results
pub type LatticeResult<T> = Result<T, LatticeError>;
