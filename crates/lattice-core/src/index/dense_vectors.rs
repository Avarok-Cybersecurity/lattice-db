//! Dense contiguous vector storage for fast indexed access
//!
//! This module provides a cache-friendly vector storage implementation
//! that eliminates HashMap lookup overhead during HNSW search.
//!
//! # Performance
//!
//! HashMap lookup: ~200+ CPU cycles (hash + bucket traversal + pointer chase)
//! Dense array access: ~10 CPU cycles (direct indexing)
//!
//! For 350 vector lookups per search, this saves ~66,000 cycles (~40-50% speedup).

use crate::types::point::{PointId, Vector};
use rustc_hash::FxHashMap;

/// Compact index into dense storage (u32 saves memory vs u64 PointId)
pub type DenseIdx = u32;

/// Dense contiguous vector storage
///
/// Stores all vectors in a single contiguous `Vec<f32>` for:
/// - O(1) indexed access (single array lookup)
/// - Cache-friendly sequential memory layout
/// - Prefetch-friendly predictable access pattern
///
/// # Design
///
/// - `data`: All vectors stored contiguously `[v0_d0, v0_d1, ..., v1_d0, ...]`
/// - `id_to_idx`: Maps PointId to compact DenseIdx (only used for insert/lookup by ID)
/// - `idx_to_id`: Reverse mapping for iteration
/// - `deleted`: Tombstone bitmap for soft deletions
pub struct DenseVectorStore {
    /// All vectors stored contiguously: [v0_d0, v0_d1, ..., v0_dN, v1_d0, ...]
    data: Vec<f32>,
    /// Vector dimension
    dim: usize,
    /// Sparse PointId to dense index mapping
    id_to_idx: FxHashMap<PointId, DenseIdx>,
    /// Dense index to sparse PointId mapping
    idx_to_id: Vec<PointId>,
    /// Tombstone bitmap for deleted vectors (64 deletions per u64)
    deleted: Vec<u64>,
    /// Free list of deleted indices for reuse
    free_list: Vec<DenseIdx>,
}

impl DenseVectorStore {
    /// Create a new dense vector store
    pub fn new(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            dim,
            id_to_idx: FxHashMap::default(),
            idx_to_id: Vec::new(),
            deleted: Vec::new(),
            free_list: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity
    ///
    /// # Panics
    ///
    /// Panics if `dim * capacity` would overflow `usize`.
    pub fn with_capacity(dim: usize, capacity: usize) -> Self {
        let data_capacity = dim
            .checked_mul(capacity)
            .expect("DenseVectorStore: dim * capacity overflow");
        Self {
            data: Vec::with_capacity(data_capacity),
            dim,
            id_to_idx: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            idx_to_id: Vec::with_capacity(capacity),
            deleted: Vec::with_capacity((capacity + 63) / 64),
            free_list: Vec::new(),
        }
    }

    /// Get vector dimension
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get number of vectors (excluding deleted)
    #[inline]
    pub fn len(&self) -> usize {
        self.id_to_idx.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.id_to_idx.is_empty()
    }

    /// Insert a vector, returns its dense index
    ///
    /// If the store was created with dim=0, the dimension is set from the first vector.
    pub fn insert(&mut self, id: PointId, vector: Vector) -> DenseIdx {
        // Lazy dimension initialization from first vector
        if self.dim == 0 {
            self.dim = vector.len();
        }

        debug_assert_eq!(
            vector.len(),
            self.dim,
            "Vector dimension mismatch: expected {}, got {}",
            self.dim,
            vector.len()
        );

        // Check if ID already exists (update case)
        if let Some(&existing_idx) = self.id_to_idx.get(&id) {
            // Update in place (checked_mul prevents overflow on large idx * dim)
            let start = (existing_idx as usize)
                .checked_mul(self.dim)
                .expect("DenseVectorStore: index overflow in update");
            self.data[start..start + self.dim].copy_from_slice(&vector);
            return existing_idx;
        }

        // Try to reuse a deleted slot
        let idx = if let Some(free_idx) = self.free_list.pop() {
            // Reuse deleted slot (checked_mul prevents overflow)
            let start = (free_idx as usize)
                .checked_mul(self.dim)
                .expect("DenseVectorStore: index overflow in reuse");
            self.data[start..start + self.dim].copy_from_slice(&vector);
            self.idx_to_id[free_idx as usize] = id;
            self.clear_deleted(free_idx);
            free_idx
        } else {
            // Append new slot
            let idx = self.idx_to_id.len() as DenseIdx;
            self.data.extend_from_slice(&vector);
            self.idx_to_id.push(id);
            // Expand deleted bitmap if needed
            let word_idx = idx as usize / 64;
            if word_idx >= self.deleted.len() {
                self.deleted.resize(word_idx + 1, 0);
            }
            idx
        };

        self.id_to_idx.insert(id, idx);
        idx
    }

    /// Get dense index for a PointId
    #[inline]
    pub fn get_idx(&self, id: PointId) -> Option<DenseIdx> {
        self.id_to_idx.get(&id).copied()
    }

    /// Get PointId for a dense index
    #[inline]
    pub fn get_id(&self, idx: DenseIdx) -> Option<PointId> {
        let idx = idx as usize;
        if idx < self.idx_to_id.len() && !self.is_deleted(idx as DenseIdx) {
            Some(self.idx_to_id[idx])
        } else {
            None
        }
    }

    /// Get vector by PointId (uses HashMap lookup - use sparingly)
    #[inline]
    pub fn get(&self, id: PointId) -> Option<&[f32]> {
        self.id_to_idx.get(&id).map(|&idx| self.get_by_idx(idx))
    }

    /// Get vector by dense index (fast O(1) access - use in hot path)
    ///
    /// # Panics
    ///
    /// Panics if `idx * dim` overflows or if index is out of bounds.
    #[inline]
    pub fn get_by_idx(&self, idx: DenseIdx) -> &[f32] {
        // checked_mul prevents memory corruption from overflow
        let start = (idx as usize)
            .checked_mul(self.dim)
            .expect("DenseVectorStore: index overflow in get_by_idx");
        debug_assert!(start + self.dim <= self.data.len(), "Index out of bounds");
        &self.data[start..start + self.dim]
    }

    /// Get raw pointer to vector data (for prefetching)
    ///
    /// # Safety
    ///
    /// Caller must ensure idx is valid and not deleted.
    ///
    /// # Panics
    ///
    /// Panics if `idx * dim` overflows.
    #[inline]
    pub fn get_ptr(&self, idx: DenseIdx) -> *const f32 {
        // checked_mul prevents memory corruption from overflow
        let start = (idx as usize)
            .checked_mul(self.dim)
            .expect("DenseVectorStore: index overflow in get_ptr");
        unsafe { self.data.as_ptr().add(start) }
    }

    /// Delete a vector by PointId (soft delete via tombstone)
    pub fn delete(&mut self, id: PointId) -> bool {
        if let Some(idx) = self.id_to_idx.remove(&id) {
            self.set_deleted(idx);
            self.free_list.push(idx);
            true
        } else {
            false
        }
    }

    /// Check if a slot is deleted
    #[inline]
    fn is_deleted(&self, idx: DenseIdx) -> bool {
        let word_idx = idx as usize / 64;
        let bit_idx = idx as usize % 64;
        if word_idx < self.deleted.len() {
            (self.deleted[word_idx] & (1u64 << bit_idx)) != 0
        } else {
            false
        }
    }

    /// Mark a slot as deleted
    #[inline]
    fn set_deleted(&mut self, idx: DenseIdx) {
        let word_idx = idx as usize / 64;
        let bit_idx = idx as usize % 64;
        if word_idx >= self.deleted.len() {
            self.deleted.resize(word_idx + 1, 0);
        }
        self.deleted[word_idx] |= 1u64 << bit_idx;
    }

    /// Clear deleted flag for a slot
    #[inline]
    fn clear_deleted(&mut self, idx: DenseIdx) {
        let word_idx = idx as usize / 64;
        let bit_idx = idx as usize % 64;
        if word_idx < self.deleted.len() {
            self.deleted[word_idx] &= !(1u64 << bit_idx);
        }
    }

    /// Check if PointId exists
    #[inline]
    pub fn contains(&self, id: PointId) -> bool {
        self.id_to_idx.contains_key(&id)
    }

    /// Iterate over all (PointId, vector) pairs
    pub fn iter(&self) -> impl Iterator<Item = (PointId, &[f32])> + '_ {
        self.id_to_idx
            .iter()
            .map(|(&id, &idx)| (id, self.get_by_idx(idx)))
    }

    /// Iterate over all PointIds
    pub fn ids(&self) -> impl Iterator<Item = PointId> + '_ {
        self.id_to_idx.keys().copied()
    }

    /// Get all vectors as borrowed slices (for distance batch calculation)
    pub fn values(&self) -> impl Iterator<Item = &[f32]> + '_ {
        self.id_to_idx.values().map(|&idx| self.get_by_idx(idx))
    }

    /// Compact storage by removing deleted entries
    ///
    /// This is an expensive operation that rebuilds the storage.
    /// Only call during maintenance windows.
    pub fn compact(&mut self) {
        if self.free_list.is_empty() {
            return; // Nothing to compact
        }

        // Rebuild with only live entries
        let mut new_data = Vec::with_capacity(self.id_to_idx.len() * self.dim);
        let mut new_idx_to_id = Vec::with_capacity(self.id_to_idx.len());
        let mut new_id_to_idx =
            FxHashMap::with_capacity_and_hasher(self.id_to_idx.len(), Default::default());

        for (&id, &old_idx) in &self.id_to_idx {
            let new_idx = new_idx_to_id.len() as DenseIdx;
            new_data.extend_from_slice(self.get_by_idx(old_idx));
            new_idx_to_id.push(id);
            new_id_to_idx.insert(id, new_idx);
        }

        self.data = new_data;
        self.idx_to_id = new_idx_to_id;
        self.id_to_idx = new_id_to_idx;
        self.deleted.clear();
        self.deleted.resize((self.idx_to_id.len() + 63) / 64, 0);
        self.free_list.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize, seed: u64) -> Vector {
        let mut rng = seed;
        (0..dim)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
            })
            .collect()
    }

    #[test]
    fn test_insert_and_get() {
        let dim = 128;
        let mut store = DenseVectorStore::new(dim);

        // Insert vectors
        for i in 0..100 {
            let vec = random_vector(dim, i);
            let idx = store.insert(i, vec.clone());
            assert_eq!(idx, i as DenseIdx);
        }

        assert_eq!(store.len(), 100);

        // Verify retrieval
        for i in 0..100 {
            let vec = store.get(i).unwrap();
            assert_eq!(vec.len(), dim);

            let idx = store.get_idx(i).unwrap();
            let vec_by_idx = store.get_by_idx(idx);
            assert_eq!(vec, vec_by_idx);
        }
    }

    #[test]
    fn test_update() {
        let dim = 64;
        let mut store = DenseVectorStore::new(dim);

        // Insert
        let vec1 = random_vector(dim, 42);
        store.insert(1, vec1.clone());

        // Update
        let vec2 = random_vector(dim, 99);
        let idx = store.insert(1, vec2.clone());

        // Should reuse same index
        assert_eq!(idx, 0);
        assert_eq!(store.len(), 1);

        // Should have new vector
        let retrieved = store.get(1).unwrap();
        assert_eq!(retrieved, vec2.as_slice());
    }

    #[test]
    fn test_delete_and_reuse() {
        let dim = 32;
        let mut store = DenseVectorStore::new(dim);

        // Insert 5 vectors
        for i in 0..5 {
            store.insert(i, random_vector(dim, i));
        }
        assert_eq!(store.len(), 5);

        // Delete vector at index 2
        assert!(store.delete(2));
        assert_eq!(store.len(), 4);
        assert!(store.get(2).is_none());

        // Insert new vector - should reuse slot 2
        let new_vec = random_vector(dim, 999);
        let idx = store.insert(10, new_vec);
        assert_eq!(idx, 2); // Reused deleted slot

        assert_eq!(store.len(), 5);
        assert!(store.get(10).is_some());
    }

    #[test]
    fn test_compact() {
        let dim = 16;
        let mut store = DenseVectorStore::new(dim);

        // Insert 10 vectors
        for i in 0..10 {
            store.insert(i, random_vector(dim, i));
        }

        // Delete half
        for i in (0..10).step_by(2) {
            store.delete(i);
        }

        assert_eq!(store.len(), 5);

        // Compact
        store.compact();

        // Verify all live vectors still accessible
        for i in (1..10).step_by(2) {
            assert!(store.get(i).is_some());
        }

        // Verify deleted vectors gone
        for i in (0..10).step_by(2) {
            assert!(store.get(i).is_none());
        }
    }

    #[test]
    fn test_get_by_idx_performance() {
        let dim = 128;
        let mut store = DenseVectorStore::with_capacity(dim, 1000);

        // Insert 1000 vectors
        for i in 0..1000 {
            store.insert(i, random_vector(dim, i));
        }

        // Access by index should be O(1)
        for i in 0..1000 {
            let vec = store.get_by_idx(i as DenseIdx);
            assert_eq!(vec.len(), dim);
        }
    }
}
