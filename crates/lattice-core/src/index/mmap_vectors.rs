//! Memory-mapped vector storage for reduced memory footprint
//!
//! Uses OS-level memory mapping for efficient vector storage:
//! - Vectors are stored in a contiguous file
//! - OS handles paging vectors in/out of memory
//! - Faster startup (no deserialization needed)
//! - Better cache utilization for large datasets
//!
//! # Architecture
//!
//! The mmap store maintains a mapping from PointId to file offset.
//! Vectors are stored contiguously with a header containing metadata.
//!
//! File format:
//! ```text
//! [Header: 24 bytes]
//!   - magic: u64 (0x4C415454_56454353 = "LATTVECS")
//!   - version: u64
//!   - dimension: u64
//! [Index: variable]
//!   - count: u64
//!   - entries: [(point_id: u64, offset: u64), ...]
//! [Vectors: variable]
//!   - vector data: [f32; dimension] per vector
//! ```

use crate::types::point::{PointId, Vector};
use memmap2::{Mmap, MmapMut, MmapOptions};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

/// Magic number for mmap vector files: "LATTVECS"
const MAGIC: u64 = 0x4C415454_56454353;
/// Current file format version
const VERSION: u64 = 1;
/// Header size in bytes
const HEADER_SIZE: usize = 24;

/// Memory-mapped vector storage
///
/// Provides efficient access to vectors stored in a memory-mapped file.
/// The OS handles paging, so only actively-used vectors consume RAM.
pub struct MmapVectorStore {
    /// Memory-mapped file (read-only after initial load)
    mmap: Mmap,
    /// Vector dimension
    dim: usize,
    /// Mapping from PointId to byte offset in the mmap
    offsets: HashMap<PointId, usize>,
}

impl MmapVectorStore {
    /// Open an existing mmap vector file
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Validate header
        if mmap.len() < HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "File too small for header",
            ));
        }

        let magic = u64::from_le_bytes(mmap[0..8].try_into().unwrap());
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid magic number",
            ));
        }

        let version = u64::from_le_bytes(mmap[8..16].try_into().unwrap());
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported version: {}", version),
            ));
        }

        let dim = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;

        // Read index
        let count = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;
        let mut offsets = HashMap::with_capacity(count);

        let index_start = 32;
        for i in 0..count {
            let entry_offset = index_start + i * 16;
            let point_id =
                u64::from_le_bytes(mmap[entry_offset..entry_offset + 8].try_into().unwrap());
            let vector_offset = u64::from_le_bytes(
                mmap[entry_offset + 8..entry_offset + 16]
                    .try_into()
                    .unwrap(),
            ) as usize;
            offsets.insert(point_id, vector_offset);
        }

        Ok(Self { mmap, dim, offsets })
    }

    /// Get a vector by ID (zero-copy access)
    ///
    /// Returns a slice directly into the mmap'd memory.
    /// The OS will page in the data if needed.
    #[inline]
    pub fn get(&self, id: PointId) -> Option<&[f32]> {
        self.offsets.get(&id).map(|&offset| {
            let byte_slice = &self.mmap[offset..offset + self.dim * 4];
            // SAFETY: The file format guarantees proper alignment and the
            // data was written as f32 values. We verify the magic number
            // and version on load.
            unsafe { std::slice::from_raw_parts(byte_slice.as_ptr() as *const f32, self.dim) }
        })
    }

    /// Get vector dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get number of vectors
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Check if a point exists
    pub fn contains(&self, id: PointId) -> bool {
        self.offsets.contains_key(&id)
    }

    /// Iterate over all point IDs
    pub fn ids(&self) -> impl Iterator<Item = PointId> + '_ {
        self.offsets.keys().copied()
    }
}

/// Builder for creating mmap vector files
///
/// Collects vectors in memory, then writes them to a memory-mapped file.
pub struct MmapVectorBuilder {
    /// Vector dimension
    dim: usize,
    /// Accumulated vectors (point_id, vector)
    vectors: Vec<(PointId, Vector)>,
}

impl MmapVectorBuilder {
    /// Create a new builder for vectors of the given dimension
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            vectors: Vec::new(),
        }
    }

    /// Add a vector to the builder
    pub fn add(&mut self, id: PointId, vector: Vector) {
        debug_assert_eq!(
            vector.len(),
            self.dim,
            "Vector dimension mismatch: expected {}, got {}",
            self.dim,
            vector.len()
        );
        self.vectors.push((id, vector));
    }

    /// Add multiple vectors
    pub fn add_batch(&mut self, vectors: impl Iterator<Item = (PointId, Vector)>) {
        for (id, vector) in vectors {
            self.add(id, vector);
        }
    }

    /// Get number of vectors added
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Build the mmap file and return a reader
    pub fn build(self, path: &Path) -> io::Result<MmapVectorStore> {
        // Calculate file size
        let index_size = 8 + self.vectors.len() * 16; // count + entries
        let vectors_size = self.vectors.len() * self.dim * 4;
        let total_size = HEADER_SIZE + index_size + vectors_size;

        // Create and size the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(total_size as u64)?;

        // Memory-map for writing
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Write header
        mmap[0..8].copy_from_slice(&MAGIC.to_le_bytes());
        mmap[8..16].copy_from_slice(&VERSION.to_le_bytes());
        mmap[16..24].copy_from_slice(&(self.dim as u64).to_le_bytes());

        // Write index count
        let count = self.vectors.len() as u64;
        mmap[24..32].copy_from_slice(&count.to_le_bytes());

        // Write index entries and vectors
        let index_start = 32;
        let vectors_start = HEADER_SIZE + index_size;

        for (i, (point_id, vector)) in self.vectors.iter().enumerate() {
            // Index entry
            let entry_offset = index_start + i * 16;
            let vector_offset = vectors_start + i * self.dim * 4;

            mmap[entry_offset..entry_offset + 8].copy_from_slice(&point_id.to_le_bytes());
            mmap[entry_offset + 8..entry_offset + 16]
                .copy_from_slice(&(vector_offset as u64).to_le_bytes());

            // Vector data
            let vector_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(vector.as_ptr() as *const u8, self.dim * 4) };
            mmap[vector_offset..vector_offset + self.dim * 4].copy_from_slice(vector_bytes);
        }

        // Flush to disk
        mmap.flush()?;

        // Re-open as read-only
        MmapVectorStore::open(path)
    }
}

/// Hybrid vector store that uses mmap for cold data and HashMap for hot data
///
/// This provides the best of both worlds:
/// - Mmap for the bulk of vectors (low memory footprint)
/// - HashMap for recently inserted vectors (fast access, no file I/O)
pub struct HybridVectorStore {
    /// Memory-mapped cold storage (optional)
    mmap: Option<MmapVectorStore>,
    /// In-memory hot storage for new insertions
    hot: HashMap<PointId, Vector>,
    /// Vector dimension
    dim: usize,
}

impl HybridVectorStore {
    /// Create a new hybrid store (in-memory only initially)
    pub fn new(dim: usize) -> Self {
        Self {
            mmap: None,
            hot: HashMap::new(),
            dim,
        }
    }

    /// Create with an existing mmap file
    pub fn with_mmap(dim: usize, mmap_path: &Path) -> io::Result<Self> {
        let mmap = MmapVectorStore::open(mmap_path)?;
        if mmap.dim() != dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Dimension mismatch: expected {}, got {}", dim, mmap.dim()),
            ));
        }
        Ok(Self {
            mmap: Some(mmap),
            hot: HashMap::new(),
            dim,
        })
    }

    /// Insert a vector (goes to hot storage)
    pub fn insert(&mut self, id: PointId, vector: Vector) {
        debug_assert_eq!(vector.len(), self.dim);
        self.hot.insert(id, vector);
    }

    /// Get a vector (checks hot storage first, then mmap)
    #[inline]
    pub fn get(&self, id: PointId) -> Option<VectorRef<'_>> {
        // Check hot storage first
        if let Some(v) = self.hot.get(&id) {
            return Some(VectorRef::Owned(v));
        }
        // Fall back to mmap
        if let Some(ref mmap) = self.mmap {
            if let Some(v) = mmap.get(id) {
                return Some(VectorRef::Mmap(v));
            }
        }
        None
    }

    /// Remove a vector from hot storage
    pub fn remove(&mut self, id: PointId) -> Option<Vector> {
        self.hot.remove(&id)
        // Note: We can't remove from mmap - it's immutable
        // Deleted IDs should be tracked separately
    }

    /// Check if a point exists
    pub fn contains(&self, id: PointId) -> bool {
        self.hot.contains_key(&id) || self.mmap.as_ref().map_or(false, |m| m.contains(id))
    }

    /// Get total count (hot + cold)
    pub fn len(&self) -> usize {
        let mmap_count = self.mmap.as_ref().map_or(0, |m| m.len());
        self.hot.len() + mmap_count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.hot.is_empty() && self.mmap.as_ref().map_or(true, |m| m.is_empty())
    }

    /// Get number of vectors in hot storage
    pub fn hot_count(&self) -> usize {
        self.hot.len()
    }

    /// Get number of vectors in cold storage
    pub fn cold_count(&self) -> usize {
        self.mmap.as_ref().map_or(0, |m| m.len())
    }

    /// Compact hot storage to mmap (creates new mmap file)
    ///
    /// This moves all hot vectors to a new mmap file, combining them
    /// with existing mmap vectors if present.
    pub fn compact(&mut self, path: &Path) -> io::Result<()> {
        let mut builder = MmapVectorBuilder::new(self.dim);

        // Add existing mmap vectors
        if let Some(ref mmap) = self.mmap {
            for id in mmap.ids() {
                if let Some(v) = mmap.get(id) {
                    builder.add(id, v.to_vec());
                }
            }
        }

        // Add hot vectors (overwrites mmap if same ID)
        for (&id, vector) in &self.hot {
            builder.add(id, vector.clone());
        }

        // Build new mmap
        self.mmap = Some(builder.build(path)?);
        self.hot.clear();

        Ok(())
    }
}

/// Reference to a vector (either owned or mmap'd)
pub enum VectorRef<'a> {
    /// Reference to owned vector in HashMap
    Owned(&'a Vector),
    /// Reference to mmap'd vector (zero-copy)
    Mmap(&'a [f32]),
}

impl<'a> VectorRef<'a> {
    /// Get the vector as a slice
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        match self {
            VectorRef::Owned(v) => v.as_slice(),
            VectorRef::Mmap(v) => v,
        }
    }
}

impl<'a> AsRef<[f32]> for VectorRef<'a> {
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl<'a> std::ops::Deref for VectorRef<'a> {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn random_vector(dim: usize, seed: u64) -> Vector {
        let mut rng = seed;
        (0..dim)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng as f64 / u64::MAX as f64) as f32
            })
            .collect()
    }

    #[test]
    fn test_mmap_builder_and_store() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_mmap_vectors.bin");

        let dim = 128;
        let count = 100;

        // Build mmap file
        let mut builder = MmapVectorBuilder::new(dim);
        for i in 0..count {
            builder.add(i as u64, random_vector(dim, i as u64));
        }
        let store = builder.build(&path).unwrap();

        // Verify
        assert_eq!(store.len(), count);
        assert_eq!(store.dim(), dim);

        for i in 0..count {
            let v = store.get(i as u64).unwrap();
            assert_eq!(v.len(), dim);
        }

        // Cleanup
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_hybrid_store() {
        let dim = 64;
        let mut store = HybridVectorStore::new(dim);

        // Insert vectors
        for i in 0..50 {
            store.insert(i, random_vector(dim, i));
        }

        assert_eq!(store.len(), 50);
        assert_eq!(store.hot_count(), 50);
        assert_eq!(store.cold_count(), 0);

        // Verify retrieval
        for i in 0..50 {
            let v = store.get(i).unwrap();
            assert_eq!(v.as_slice().len(), dim);
        }

        // Remove a vector
        store.remove(25);
        assert_eq!(store.len(), 49);
        assert!(store.get(25).is_none());
    }

    #[test]
    fn test_hybrid_store_compact() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_hybrid_compact.bin");

        let dim = 64;
        let mut store = HybridVectorStore::new(dim);

        // Insert vectors
        for i in 0..100 {
            store.insert(i, random_vector(dim, i));
        }

        // Compact to mmap
        store.compact(&path).unwrap();

        assert_eq!(store.len(), 100);
        assert_eq!(store.hot_count(), 0);
        assert_eq!(store.cold_count(), 100);

        // Verify retrieval from mmap
        for i in 0..100 {
            let v = store.get(i).unwrap();
            assert_eq!(v.as_slice().len(), dim);
        }

        // Add more hot vectors
        for i in 100..150 {
            store.insert(i, random_vector(dim, i));
        }

        assert_eq!(store.len(), 150);
        assert_eq!(store.hot_count(), 50);
        assert_eq!(store.cold_count(), 100);

        // Cleanup
        fs::remove_file(&path).ok();
    }
}
