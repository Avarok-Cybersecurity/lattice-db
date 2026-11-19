//! Core data types for LatticeDB

pub mod collection;
pub mod point;
pub mod query;
pub mod sorted_map;
pub mod value;
pub mod vector_index;

// Re-export SharedStr for use in other modules
pub use value::SharedStr;
