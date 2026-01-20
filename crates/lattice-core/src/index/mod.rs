//! Index implementations for vector similarity search
//!
//! This module contains the HNSW index implementation and distance functions.

pub mod distance;
pub mod hnsw;
pub mod layer;
pub mod quantization;
pub mod scann;

// Memory-mapped vectors (native only with mmap feature)
#[cfg(all(feature = "mmap", not(target_arch = "wasm32")))]
pub mod mmap_vectors;

pub use distance::DistanceCalculator;
pub use hnsw::{HnswIndex, PQAccelerator};
pub use quantization::QuantizedVector;
pub use scann::{DistanceTable, PQCode, PQIndex, ProductQuantizer};

#[cfg(all(feature = "mmap", not(target_arch = "wasm32")))]
pub use mmap_vectors::{HybridVectorStore, MmapVectorBuilder, MmapVectorStore, VectorRef};
