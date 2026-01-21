//! Core data types for LatticeDB

pub mod collection;
pub mod point;
pub mod query;
pub mod value;

// Re-export SharedStr for use in other modules
pub use value::SharedStr;
