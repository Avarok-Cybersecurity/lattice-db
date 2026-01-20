//! Graph operations module
//!
//! Provides graph-specific functionality for LatticeDB.
//! The main traversal logic is in `CollectionEngine`, this module
//! provides additional utilities and types.

pub mod traversal;

pub use traversal::{BfsIterator, DfsIterator, GraphPath};
