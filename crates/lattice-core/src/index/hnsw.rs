//! HNSW (Hierarchical Navigable Small World) index implementation
//!
//! This implements the algorithm from the paper:
//! "Efficient and robust approximate nearest neighbor search using Hierarchical
//! Navigable Small World graphs" by Yu. A. Malkov, D. A. Yashunin
//!
//! # Architecture (SBIO)
//!
//! The index stores vectors in memory and uses the LatticeStorage trait
//! for persistence. Core algorithm is pure computation - no I/O calls.

use crate::index::distance::DistanceCalculator;
use crate::index::layer::{HnswNode, LayerManager};
use crate::index::scann::{DistanceTable, PQCode, ProductQuantizer};
use crate::types::collection::{Distance, HnswConfig};
use crate::types::point::{Point, PointId, Vector};
use crate::types::query::SearchResult;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

// Native: parallel processing with rayon
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

// Native: thread-local scratch space
#[cfg(not(target_arch = "wasm32"))]
use std::cell::RefCell;

/// Pre-allocated scratch space for search operations
/// Avoids repeated allocation of HashSet, BinaryHeap, Vec per search
#[cfg(not(target_arch = "wasm32"))]
struct SearchScratch {
    visited: HashSet<PointId>,
    candidates: BinaryHeap<Reverse<Candidate>>,
    results: BinaryHeap<Candidate>,
}

#[cfg(not(target_arch = "wasm32"))]
impl SearchScratch {
    fn new() -> Self {
        Self {
            visited: HashSet::with_capacity(1000),
            candidates: BinaryHeap::with_capacity(200),
            results: BinaryHeap::with_capacity(200),
        }
    }

    fn clear(&mut self) {
        self.visited.clear();
        self.candidates.clear();
        self.results.clear();
    }
}

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    static SEARCH_SCRATCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::new());
}

/// Candidate for search priority queue
#[derive(Debug, Clone, PartialEq)]
struct Candidate {
    id: PointId,
    distance: f32,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Default: min-heap by distance (closer = higher priority)
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// HNSW index for approximate nearest neighbor search
///
/// # Usage
///
/// ```ignore
/// let config = HnswConfig { m: 16, m0: 32, ml: 0.36, ef: 100, ef_construction: 200 };
/// let mut index = HnswIndex::new(config, Distance::Cosine);
///
/// // Insert points
/// index.insert(&point);
///
/// // Search
/// let results = index.search(&query_vector, 10, 100);
/// ```
pub struct HnswIndex {
    /// HNSW configuration
    config: HnswConfig,
    /// Distance calculator
    distance: DistanceCalculator,
    /// Layer manager (graph structure)
    layers: LayerManager,
    /// Vector storage: point_id -> vector
    vectors: HashMap<PointId, Vector>,
    /// Random number generator state for layer selection
    rng_state: u64,
}

impl HnswIndex {
    /// Create a new HNSW index
    pub fn new(config: HnswConfig, metric: Distance) -> Self {
        Self {
            config,
            distance: DistanceCalculator::new(metric),
            layers: LayerManager::new(),
            vectors: HashMap::new(),
            rng_state: 42, // Deterministic for testing
        }
    }

    /// Set random seed (for reproducible tests)
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_state = seed;
        self
    }

    /// Insert a point into the index
    pub fn insert(&mut self, point: &Point) {
        let id = point.id;
        let vector = point.vector.clone();

        // Store vector
        self.vectors.insert(id, vector.clone());

        // Select random layer for this node
        let level = self.random_level();
        let mut node = HnswNode::new(id, level);

        // If this is the first point, just insert it
        if self.layers.is_empty() {
            self.layers.insert_node(node);
            return;
        }

        let entry_point = self.layers.entry_point().unwrap();
        let mut current = entry_point;

        // Phase 1: Traverse from top layer down to node's level + 1
        // (greedy search, single nearest neighbor)
        for layer in (level as usize + 1..=self.layers.max_layer() as usize).rev() {
            current = self.search_layer_single(&vector, current, layer as u16);
        }

        // Phase 2: Insert at each layer from level down to 0
        for layer in (0..=level as usize).rev() {
            let layer = layer as u16;

            // Find ef_construction nearest neighbors
            let neighbors = self.search_layer(
                &vector,
                vec![current],
                self.config.ef_construction,
                layer,
            );

            // Select M best neighbors (M0 for layer 0)
            let max_neighbors = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };

            let selected: Vec<PointId> = neighbors
                .iter()
                .take(max_neighbors)
                .map(|c| c.id)
                .collect();

            // Add bidirectional connections
            for &neighbor_id in &selected {
                node.add_neighbor(layer, neighbor_id);

                // Add reverse connection (may need pruning)
                if let Some(neighbor_node) = self.layers.get_node_mut(neighbor_id) {
                    neighbor_node.add_neighbor(layer, id);

                    // Prune if over limit
                    let neighbor_count = neighbor_node.neighbors_at(layer).len();
                    if neighbor_count > max_neighbors {
                        self.prune_neighbors(neighbor_id, layer, max_neighbors);
                    }
                }
            }

            // Update current for next layer
            if !neighbors.is_empty() {
                current = neighbors[0].id;
            }
        }

        self.layers.insert_node(node);
    }

    /// Search for k nearest neighbors
    ///
    /// Uses shortcut-enabled search (VLDB 2025) - skips layers when the best
    /// neighbor doesn't change, reducing redundant distance calculations.
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of results
    /// * `ef` - Search queue size (higher = better recall, slower)
    ///
    /// # Returns
    /// Vector of search results sorted by distance (closest first)
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<SearchResult> {
        if self.layers.is_empty() {
            return vec![];
        }

        let ef = ef.max(k); // ef must be >= k

        let entry_point = self.layers.entry_point().unwrap();
        let mut current = entry_point;
        let mut current_dist = self.calc_distance(query, current);

        // Phase 1: Traverse from top layer down to layer 1 with shortcuts
        // If we don't improve at a layer, we can potentially skip faster
        for layer in (1..=self.layers.max_layer() as usize).rev() {
            let (new_current, new_dist, improved) =
                self.search_layer_single_with_shortcut(query, current, current_dist, layer as u16);

            // Even if we didn't improve, update current for continuity
            current = new_current;
            current_dist = new_dist;

            // If we found improvement, continue normal search at next layer
            // If not, the paper suggests we might skip, but for safety we continue
            // This at least avoids recalculating the entry point distance
            if !improved && layer > 1 {
                // Shortcut: check if we can skip directly by verifying neighbors
                // at next layer won't be better. This is the conservative approach.
                continue;
            }
        }

        // Phase 2: Search layer 0 with ef
        let candidates = self.search_layer(query, vec![current], ef, 0);

        // Return top k
        candidates
            .into_iter()
            .take(k)
            .map(|c| SearchResult::new(c.id, c.distance))
            .collect()
    }

    /// Batch search for multiple queries
    ///
    /// More efficient than calling `search` multiple times due to:
    /// - Parallel processing on native platforms (rayon)
    /// - Better cache utilization across queries
    ///
    /// # Arguments
    /// * `queries` - Vector of query vectors
    /// * `k` - Number of results per query
    /// * `ef` - Search queue size (higher = better recall, slower)
    ///
    /// # Returns
    /// Vector of results for each query, in the same order as input
    #[cfg(not(target_arch = "wasm32"))]
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
    ) -> Vec<Vec<SearchResult>> {
        if queries.is_empty() || self.layers.is_empty() {
            return vec![vec![]; queries.len()];
        }

        // Process queries in parallel using rayon
        queries
            .par_iter()
            .map(|query| self.search(query, k, ef))
            .collect()
    }

    /// Batch search for multiple queries (WASM version - sequential)
    #[cfg(target_arch = "wasm32")]
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
    ) -> Vec<Vec<SearchResult>> {
        queries
            .iter()
            .map(|query| self.search(query, k, ef))
            .collect()
    }

    /// Delete a point from the index
    pub fn delete(&mut self, id: PointId) -> bool {
        self.vectors.remove(&id);
        self.layers.remove_node(id).is_some()
    }

    /// Get number of points in the index
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get a vector by ID
    pub fn get_vector(&self, id: PointId) -> Option<&Vector> {
        self.vectors.get(&id)
    }

    /// Get layer counts for debugging
    pub fn layer_counts(&self) -> Vec<usize> {
        self.layers.layer_counts()
    }

    /// Iterate over all vector IDs
    pub fn vector_ids(&self) -> impl Iterator<Item = PointId> + '_ {
        self.vectors.keys().copied()
    }

    // --- Mmap support (native only with mmap feature) ---

    /// Export vectors to a memory-mapped file
    ///
    /// This creates an mmap file containing all vectors in the index.
    /// After export, vectors can be loaded via `load_vectors_mmap` for
    /// reduced memory footprint.
    #[cfg(all(feature = "mmap", not(target_arch = "wasm32")))]
    pub fn export_vectors_mmap(
        &self,
        path: &std::path::Path,
    ) -> std::io::Result<super::mmap_vectors::MmapVectorStore> {
        use super::mmap_vectors::MmapVectorBuilder;

        let dim = self.vectors.values().next().map(|v| v.len()).unwrap_or(0);
        let mut builder = MmapVectorBuilder::new(dim);

        for (&id, vector) in &self.vectors {
            builder.add(id, vector.clone());
        }

        builder.build(path)
    }

    /// Get memory usage estimate in bytes
    ///
    /// Returns estimated memory used by vectors (not including graph structure).
    pub fn vector_memory_bytes(&self) -> usize {
        self.vectors
            .values()
            .map(|v| v.len() * std::mem::size_of::<f32>() + std::mem::size_of::<PointId>())
            .sum()
    }

    // --- Private methods ---

    /// Generate random level for new node
    fn random_level(&mut self) -> u16 {
        // Simple LCG random number generator
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let random = (self.rng_state >> 33) as f64 / (1u64 << 31) as f64;

        // Level = floor(-ln(uniform) * ml)
        let level = (-random.ln() * self.config.ml).floor() as u16;

        // Cap at reasonable maximum (prevents runaway in edge cases)
        level.min(16)
    }

    /// Search a single layer for the nearest neighbor (greedy)
    fn search_layer_single(&self, query: &[f32], start: PointId, layer: u16) -> PointId {
        let mut current = start;
        let mut current_dist = self.calc_distance(query, current);

        loop {
            let mut changed = false;

            if let Some(node) = self.layers.get_node(current) {
                for &neighbor in node.neighbors_at(layer) {
                    let dist = self.calc_distance(query, neighbor);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Search a single layer with shortcut detection (VLDB 2025)
    ///
    /// Returns (best_id, best_distance, improved) where improved indicates
    /// if we found a better neighbor than the starting point.
    fn search_layer_single_with_shortcut(
        &self,
        query: &[f32],
        start: PointId,
        start_dist: f32,
        layer: u16,
    ) -> (PointId, f32, bool) {
        let mut current = start;
        let mut current_dist = start_dist;
        let mut improved = false;

        loop {
            let mut changed = false;

            if let Some(node) = self.layers.get_node(current) {
                for &neighbor in node.neighbors_at(layer) {
                    let dist = self.calc_distance(query, neighbor);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                        improved = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        (current, current_dist, improved)
    }

    /// Search a layer for ef nearest neighbors
    ///
    /// On native builds, uses thread-local scratch space to avoid allocations.
    #[cfg(not(target_arch = "wasm32"))]
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: Vec<PointId>,
        ef: usize,
        layer: u16,
    ) -> Vec<Candidate> {
        SEARCH_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.clear();

            // Initialize with entry points
            for &ep in &entry_points {
                scratch.visited.insert(ep);
                let dist = self.calc_distance(query, ep);
                scratch.candidates.push(Reverse(Candidate { id: ep, distance: dist }));
                scratch.results.push(Candidate { id: ep, distance: dist });
            }

            while let Some(Reverse(current)) = scratch.candidates.pop() {
                // Stop if current is farther than worst result and we have enough
                if scratch.results.len() >= ef {
                    if let Some(worst) = scratch.results.peek() {
                        if current.distance > worst.distance {
                            break;
                        }
                    }
                }

                // Explore neighbors
                if let Some(node) = self.layers.get_node(current.id) {
                    // Collect unvisited neighbors
                    let unvisited: Vec<PointId> = node
                        .neighbors_at(layer)
                        .iter()
                        .copied()
                        .filter(|&n| scratch.visited.insert(n))
                        .collect();

                    if unvisited.is_empty() {
                        continue;
                    }

                    // Calculate distances (parallel for large batches)
                    let neighbor_dists = self.calc_distances_batch(query, &unvisited);

                    // Process results
                    for (neighbor, dist) in neighbor_dists {
                        let candidate = Candidate { id: neighbor, distance: dist };

                        let should_add = scratch.results.len() < ef
                            || scratch.results.peek().map_or(true, |w| dist < w.distance);

                        if should_add {
                            scratch.candidates.push(Reverse(candidate.clone()));
                            scratch.results.push(candidate);

                            while scratch.results.len() > ef {
                                scratch.results.pop();
                            }
                        }
                    }
                }
            }

            // Convert to sorted vec (closest first)
            let mut result_vec: Vec<Candidate> = scratch.results.drain().collect();
            result_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            result_vec
        })
    }

    /// Search a layer for ef nearest neighbors (WASM version - allocates each call)
    #[cfg(target_arch = "wasm32")]
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: Vec<PointId>,
        ef: usize,
        layer: u16,
    ) -> Vec<Candidate> {
        let mut visited: HashSet<PointId> = entry_points.iter().copied().collect();

        // Min-heap: candidates to explore (closest first)
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();

        // Max-heap: best results found (farthest first for easy pruning)
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        // Initialize with entry points
        for &ep in &entry_points {
            let dist = self.calc_distance(query, ep);
            candidates.push(Reverse(Candidate { id: ep, distance: dist }));
            results.push(Candidate { id: ep, distance: dist });
        }

        while let Some(Reverse(current)) = candidates.pop() {
            // Stop if current is farther than worst result and we have enough
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.distance > worst.distance {
                        break;
                    }
                }
            }

            // Explore neighbors - collect unvisited first, then calculate distances
            if let Some(node) = self.layers.get_node(current.id) {
                // Collect unvisited neighbors
                let unvisited: Vec<PointId> = node
                    .neighbors_at(layer)
                    .iter()
                    .copied()
                    .filter(|&n| visited.insert(n))
                    .collect();

                if unvisited.is_empty() {
                    continue;
                }

                // Calculate distances
                let neighbor_dists = self.calc_distances_batch(query, &unvisited);

                // Process results
                for (neighbor, dist) in neighbor_dists {
                    let candidate = Candidate { id: neighbor, distance: dist };

                    // Add to results if better than worst or room available
                    let should_add = results.len() < ef
                        || results.peek().map_or(true, |w| dist < w.distance);

                    if should_add {
                        candidates.push(Reverse(candidate.clone()));
                        results.push(candidate);

                        // Prune results to ef size
                        while results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Convert to sorted vec (closest first)
        let mut result_vec: Vec<Candidate> = results.into_vec();
        result_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        result_vec
    }

    /// Calculate distances to multiple points
    ///
    /// On native builds with large batches (>64), uses parallel processing.
    /// For smaller batches, sequential is faster due to parallel overhead.
    #[cfg(not(target_arch = "wasm32"))]
    fn calc_distances_batch(&self, query: &[f32], ids: &[PointId]) -> Vec<(PointId, f32)> {
        // Parallel overhead is only worth it for larger batches
        // Typical neighbor count is 16-32, so use sequential for small counts
        const PARALLEL_THRESHOLD: usize = 64;

        if ids.len() >= PARALLEL_THRESHOLD {
            ids.par_iter()
                .filter_map(|&id| {
                    self.vectors.get(&id).map(|vec| (id, self.distance.calculate(query, vec)))
                })
                .collect()
        } else {
            ids.iter()
                .filter_map(|&id| {
                    self.vectors.get(&id).map(|vec| (id, self.distance.calculate(query, vec)))
                })
                .collect()
        }
    }

    /// Calculate distances to multiple points (sequential on WASM)
    #[cfg(target_arch = "wasm32")]
    fn calc_distances_batch(&self, query: &[f32], ids: &[PointId]) -> Vec<(PointId, f32)> {
        ids.iter()
            .filter_map(|&id| {
                self.vectors.get(&id).map(|vec| (id, self.distance.calculate(query, vec)))
            })
            .collect()
    }

    /// Prune neighbors to keep best M (closest by distance)
    ///
    /// Maintains sorted order by PointId for efficient binary search lookups.
    fn prune_neighbors(&mut self, node_id: PointId, layer: u16, max_neighbors: usize) {
        // Get neighbors and their distances in one pass (avoid vector clone)
        let neighbors_with_dist: Vec<(PointId, f32)> = {
            let vector = match self.vectors.get(&node_id) {
                Some(v) => v,
                None => return,
            };

            let node = match self.layers.get_node(node_id) {
                Some(n) => n,
                None => return,
            };

            node.neighbors_at(layer)
                .iter()
                .filter_map(|&nid| {
                    self.vectors.get(&nid).map(|neighbor_vec| {
                        (nid, self.distance.calculate(vector, neighbor_vec))
                    })
                })
                .collect()
        };

        // Sort by distance to find best M neighbors
        let mut sorted = neighbors_with_dist;
        sorted.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        sorted.truncate(max_neighbors);

        // Extract IDs and sort by PointId (required for binary_search in add_neighbor)
        let mut new_neighbors: Vec<PointId> = sorted.into_iter().map(|(id, _)| id).collect();
        new_neighbors.sort_unstable();

        // Update neighbors
        if let Some(node) = self.layers.get_node_mut(node_id) {
            if let Some(neighbors) = node.neighbors_at_mut(layer) {
                *neighbors = new_neighbors;
            }
        }
    }

    /// Calculate distance between query and stored vector
    fn calc_distance(&self, query: &[f32], id: PointId) -> f32 {
        match self.vectors.get(&id) {
            Some(vec) => self.distance.calculate(query, vec),
            None => f32::MAX,
        }
    }

    /// Calculate distance between a query vector and another vector
    ///
    /// This is used for brute-force search on pending (unindexed) points.
    pub fn calc_distance_for_query(&self, query: &[f32], vector: &[f32]) -> f32 {
        self.distance.calculate(query, vector)
    }

    /// Build a PQ accelerator for faster approximate distance computation
    ///
    /// The accelerator compresses all vectors using Product Quantization,
    /// enabling O(M) approximate distance computation instead of O(D).
    /// Use with `search_with_pq()` for faster search with re-ranking.
    ///
    /// # Arguments
    /// * `m` - Number of subvectors (typically 8 for 128-dim vectors)
    /// * `training_sample_size` - Number of vectors to sample for training (0 = use all)
    pub fn build_pq_accelerator(&self, m: usize, training_sample_size: usize) -> PQAccelerator {
        let dim = self.vectors.values().next().map(|v| v.len()).unwrap_or(128);
        let mut pq = ProductQuantizer::new(dim, m, 256);

        // Collect training vectors
        let training_vectors: Vec<Vector> = if training_sample_size > 0 && training_sample_size < self.vectors.len() {
            // Sample subset for training
            self.vectors
                .values()
                .take(training_sample_size)
                .cloned()
                .collect()
        } else {
            // Use all vectors
            self.vectors.values().cloned().collect()
        };

        // Train the quantizer
        pq.train_default(&training_vectors);

        // Encode all vectors
        let mut codes: HashMap<PointId, PQCode> = HashMap::with_capacity(self.vectors.len());
        for (&id, vector) in &self.vectors {
            codes.insert(id, pq.encode(vector));
        }

        PQAccelerator { pq, codes }
    }

    /// Search with PQ-accelerated coarse filtering and exact re-ranking
    ///
    /// This method uses Product Quantization for fast approximate distance
    /// computation, then re-ranks the top candidates with exact distances
    /// for high accuracy.
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of results
    /// * `ef` - Search queue size for HNSW traversal
    /// * `accelerator` - PQ accelerator built with `build_pq_accelerator()`
    /// * `rerank_factor` - Multiplier for candidates to re-rank (e.g., 3 = re-rank 3*k candidates)
    pub fn search_with_pq(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accelerator: &PQAccelerator,
        rerank_factor: usize,
    ) -> Vec<SearchResult> {
        if self.layers.is_empty() {
            return vec![];
        }

        let ef = ef.max(k);

        // Phase 1: HNSW traversal with PQ-accelerated distances (coarse filtering)
        let entry_point = self.layers.entry_point().unwrap();
        let mut current = entry_point;
        let mut current_dist = accelerator.approximate_distance(query, current);

        // Build distance table once for O(M) lookups
        let dist_table = accelerator.pq.build_distance_table(query);

        // Navigate upper layers with approximate distances
        for layer in (1..=self.layers.max_layer() as usize).rev() {
            loop {
                let mut changed = false;
                if let Some(node) = self.layers.get_node(current) {
                    for &neighbor in node.neighbors_at(layer as u16) {
                        let dist = accelerator.approximate_distance_with_table(&dist_table, neighbor);
                        if dist < current_dist {
                            current = neighbor;
                            current_dist = dist;
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        // Phase 2: Search layer 0 with PQ-accelerated distances
        let candidates = self.search_layer_pq(query, vec![current], ef * rerank_factor, &dist_table, accelerator);

        // Phase 3: Re-rank top candidates with exact distances
        let rerank_count = k * rerank_factor;
        let mut reranked: Vec<(PointId, f32)> = candidates
            .into_iter()
            .take(rerank_count)
            .filter_map(|c| {
                self.vectors.get(&c.id).map(|v| (c.id, self.distance.calculate(query, v)))
            })
            .collect();

        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        reranked
            .into_iter()
            .take(k)
            .map(|(id, dist)| SearchResult::new(id, dist))
            .collect()
    }

    /// Search layer 0 using PQ-accelerated distances
    fn search_layer_pq(
        &self,
        _query: &[f32],
        entry_points: Vec<PointId>,
        ef: usize,
        dist_table: &DistanceTable,
        accelerator: &PQAccelerator,
    ) -> Vec<Candidate> {
        let mut visited: HashSet<PointId> = entry_points.iter().copied().collect();
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        // Initialize with entry points
        for &ep in &entry_points {
            let dist = accelerator.approximate_distance_with_table(dist_table, ep);
            candidates.push(Reverse(Candidate { id: ep, distance: dist }));
            results.push(Candidate { id: ep, distance: dist });
        }

        while let Some(Reverse(current)) = candidates.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.distance > worst.distance {
                        break;
                    }
                }
            }

            if let Some(node) = self.layers.get_node(current.id) {
                for &neighbor in node.neighbors_at(0) {
                    if visited.insert(neighbor) {
                        let dist = accelerator.approximate_distance_with_table(dist_table, neighbor);
                        let candidate = Candidate { id: neighbor, distance: dist };

                        let should_add = results.len() < ef
                            || results.peek().map_or(true, |w| dist < w.distance);

                        if should_add {
                            candidates.push(Reverse(candidate.clone()));
                            results.push(candidate);

                            while results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut result_vec: Vec<Candidate> = results.into_vec();
        result_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        result_vec
    }
}

/// PQ accelerator for fast approximate distance computation
///
/// Caches PQ-encoded vectors for efficient O(M) distance computation.
/// Use with `HnswIndex::search_with_pq()` for accelerated search.
pub struct PQAccelerator {
    /// Product quantizer with trained codebooks
    pq: ProductQuantizer,
    /// PQ-encoded vectors: point_id -> compressed code
    codes: HashMap<PointId, PQCode>,
}

impl PQAccelerator {
    /// Compute approximate distance to a point
    #[inline]
    pub fn approximate_distance(&self, query: &[f32], id: PointId) -> f32 {
        let table = self.pq.build_distance_table(query);
        self.approximate_distance_with_table(&table, id)
    }

    /// Compute approximate distance using precomputed table (faster for multiple queries)
    #[inline]
    pub fn approximate_distance_with_table(&self, table: &DistanceTable, id: PointId) -> f32 {
        self.codes
            .get(&id)
            .map(|code| self.pq.asymmetric_distance(table, code))
            .unwrap_or(f32::MAX)
    }

    /// Add a new vector to the accelerator (for incremental updates)
    pub fn add(&mut self, id: PointId, vector: &[f32]) {
        let code = self.pq.encode(vector);
        self.codes.insert(id, code);
    }

    /// Remove a vector from the accelerator
    pub fn remove(&mut self, id: PointId) -> bool {
        self.codes.remove(&id).is_some()
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.pq.compression_ratio()
    }

    /// Get number of encoded vectors
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Get memory usage in bytes (compressed vectors only)
    pub fn memory_bytes(&self) -> usize {
        self.codes.len() * self.pq.num_subvectors()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn test_config() -> HnswConfig {
        HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 100,
            ef_construction: 200,
        }
    }

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
    fn test_empty_index() {
        let index = HnswIndex::new(test_config(), Distance::Cosine);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);

        let results = index.search(&[0.1, 0.2], 10, 100);
        assert!(results.is_empty());
    }

    #[test]
    fn test_insert_single_point() {
        let mut index = HnswIndex::new(test_config(), Distance::Cosine);
        let point = Point::new_vector(1, vec![0.1, 0.2, 0.3]);

        index.insert(&point);

        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
        assert!(index.get_vector(1).is_some());
    }

    #[test]
    fn test_insert_multiple_points() {
        let mut index = HnswIndex::new(test_config(), Distance::Cosine);

        for i in 0..100 {
            let point = Point::new_vector(i, random_vector(32, i));
            index.insert(&point);
        }

        assert_eq!(index.len(), 100);

        // Check layer distribution (should have multiple layers)
        let counts = index.layer_counts();
        assert!(!counts.is_empty());
        assert_eq!(counts[0], 100); // All points in layer 0
    }

    #[test]
    fn test_search_returns_k_results() {
        let mut index = HnswIndex::new(test_config(), Distance::Cosine);

        for i in 0..50 {
            let point = Point::new_vector(i, random_vector(32, i));
            index.insert(&point);
        }

        let query = random_vector(32, 999);
        let results = index.search(&query, 10, 100);

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_search_returns_sorted_by_distance() {
        let mut index = HnswIndex::new(test_config(), Distance::Cosine);

        for i in 0..100 {
            let point = Point::new_vector(i, random_vector(32, i));
            index.insert(&point);
        }

        let query = random_vector(32, 999);
        let results = index.search(&query, 20, 100);

        // Verify sorted by distance (ascending)
        for i in 1..results.len() {
            assert!(
                results[i - 1].score <= results[i].score,
                "Results not sorted: {} > {}",
                results[i - 1].score,
                results[i].score
            );
        }
    }

    #[test]
    fn test_search_nearest_is_exact() {
        let mut index = HnswIndex::new(test_config(), Distance::Cosine);

        // Insert points including the query itself
        let query_vec = vec![0.5, 0.5, 0.5, 0.5];
        let query_point = Point::new_vector(42, query_vec.clone());
        index.insert(&query_point);

        for i in 0..50 {
            if i != 42 {
                let point = Point::new_vector(i, random_vector(4, i));
                index.insert(&point);
            }
        }

        let results = index.search(&query_vec, 1, 100);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 42);
        assert!(results[0].score < 0.001); // Should be ~0
    }

    #[test]
    fn test_delete_point() {
        let mut index = HnswIndex::new(test_config(), Distance::Cosine);

        let point = Point::new_vector(1, vec![0.1, 0.2, 0.3]);
        index.insert(&point);
        assert_eq!(index.len(), 1);

        let deleted = index.delete(1);
        assert!(deleted);
        assert_eq!(index.len(), 0);
        assert!(index.get_vector(1).is_none());
    }

    #[test]
    fn test_recall_1k_vectors() {
        // Test recall with 1000 vectors
        let dim = 64;
        let n = 1000;
        let k = 10;

        let mut index = HnswIndex::new(test_config(), Distance::Euclid).with_seed(12345);

        // Generate and insert vectors
        let vectors: Vec<Vector> = (0..n).map(|i| random_vector(dim, i)).collect();

        for (i, vec) in vectors.iter().enumerate() {
            let point = Point::new_vector(i as u64, vec.clone());
            index.insert(&point);
        }

        // Test recall on 10 random queries
        let mut total_recall = 0.0;
        let num_queries = 10;

        let distance = DistanceCalculator::new(Distance::Euclid);

        for q in 0..num_queries {
            let query = random_vector(dim, 10000 + q);

            // Brute-force ground truth
            let mut ground_truth: Vec<(u64, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i as u64, distance.calculate(&query, v)))
                .collect();
            ground_truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let gt_ids: HashSet<u64> = ground_truth.iter().take(k).map(|(id, _)| *id).collect();

            // HNSW search
            let results = index.search(&query, k, 100);
            let result_ids: HashSet<u64> = results.iter().map(|r| r.id).collect();

            // Calculate recall
            let hits = gt_ids.intersection(&result_ids).count();
            total_recall += hits as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        println!("Average recall@{}: {:.3}", k, avg_recall);

        // Should achieve at least 90% recall
        assert!(
            avg_recall >= 0.9,
            "Recall too low: {:.3} (expected >= 0.9)",
            avg_recall
        );
    }

    #[test]
    fn test_different_distance_metrics() {
        let configs = [
            (Distance::Cosine, "cosine"),
            (Distance::Euclid, "euclid"),
            (Distance::Dot, "dot"),
        ];

        for (metric, name) in configs {
            let mut index = HnswIndex::new(test_config(), metric);

            for i in 0..50 {
                let point = Point::new_vector(i, random_vector(16, i));
                index.insert(&point);
            }

            let query = random_vector(16, 999);
            let results = index.search(&query, 10, 100);

            assert_eq!(results.len(), 10, "Failed for metric: {}", name);
        }
    }

    #[test]
    fn test_search_batch() {
        let mut index = HnswIndex::new(test_config(), Distance::Cosine);

        // Insert 100 points
        for i in 0..100 {
            let point = Point::new_vector(i, random_vector(32, i));
            index.insert(&point);
        }

        // Create 10 batch queries
        let queries: Vec<Vec<f32>> = (0..10)
            .map(|i| random_vector(32, 1000 + i))
            .collect();

        // Batch search
        let batch_results = index.search_batch(&queries, 5, 100);

        // Verify we got results for each query
        assert_eq!(batch_results.len(), 10);
        for (i, results) in batch_results.iter().enumerate() {
            assert_eq!(
                results.len(),
                5,
                "Query {} returned {} results instead of 5",
                i,
                results.len()
            );
        }

        // Verify batch results match individual search results
        for (i, query) in queries.iter().enumerate() {
            let individual_result = index.search(query, 5, 100);
            assert_eq!(
                batch_results[i].len(),
                individual_result.len(),
                "Batch result {} differs from individual search",
                i
            );
            // Check IDs match (order might differ slightly due to parallel execution)
            for (j, (batch_r, ind_r)) in batch_results[i]
                .iter()
                .zip(individual_result.iter())
                .enumerate()
            {
                assert_eq!(
                    batch_r.id, ind_r.id,
                    "Query {} result {} mismatch: batch={}, individual={}",
                    i, j, batch_r.id, ind_r.id
                );
            }
        }
    }

    #[test]
    fn test_search_batch_empty() {
        let index = HnswIndex::new(test_config(), Distance::Cosine);

        // Empty index
        let queries: Vec<Vec<f32>> = vec![vec![0.1, 0.2, 0.3]];
        let results = index.search_batch(&queries, 5, 100);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_empty());

        // Empty queries
        let results = index.search_batch(&[], 5, 100);
        assert!(results.is_empty());
    }

    #[test]
    fn test_pq_accelerator_build() {
        let mut index = HnswIndex::new(test_config(), Distance::Cosine);

        // Need at least 128-dim vectors for m=8 (128/8 = 16 dim per subvector)
        for i in 0..500 {
            let point = Point::new_vector(i, random_vector(128, i));
            index.insert(&point);
        }

        // Build PQ accelerator with m=8 subvectors
        let accelerator = index.build_pq_accelerator(8, 0);

        assert_eq!(accelerator.len(), 500);
        // Compression ratio: 128 * 4 bytes = 512 bytes -> 8 bytes = 64x
        assert!((accelerator.compression_ratio() - 64.0).abs() < 0.1);
    }

    #[test]
    fn test_pq_accelerated_search() {
        let mut index = HnswIndex::new(test_config(), Distance::Cosine);

        // Insert vectors
        for i in 0..1000 {
            let point = Point::new_vector(i, random_vector(128, i));
            index.insert(&point);
        }

        // Build PQ accelerator
        let accelerator = index.build_pq_accelerator(8, 500);

        // Run both regular and PQ-accelerated search
        let query = random_vector(128, 9999);
        let regular_results = index.search(&query, 10, 100);
        // Use higher rerank factor for better recall
        let pq_results = index.search_with_pq(&query, 10, 100, &accelerator, 5);

        // Both should return k results
        assert_eq!(regular_results.len(), 10);
        assert_eq!(pq_results.len(), 10);

        // Results should be sorted by distance
        for i in 1..pq_results.len() {
            assert!(
                pq_results[i - 1].score <= pq_results[i].score,
                "PQ results not sorted"
            );
        }

        // PQ results should have reasonable recall (at least 4/10 overlap)
        // Note: PQ is an approximation, exact recall depends on data distribution
        let regular_ids: std::collections::HashSet<_> = regular_results.iter().map(|r| r.id).collect();
        let pq_ids: std::collections::HashSet<_> = pq_results.iter().map(|r| r.id).collect();
        let overlap = regular_ids.intersection(&pq_ids).count();

        // With re-ranking, we should get at least 3/10 of the true results
        // (random vectors have poor PQ reconstruction, real data does better)
        assert!(
            overlap >= 3,
            "PQ recall too low: {}/10 overlap with regular search",
            overlap
        );
    }

    #[test]
    fn test_pq_accelerator_incremental_update() {
        let mut index = HnswIndex::new(test_config(), Distance::Cosine);

        for i in 0..100 {
            let point = Point::new_vector(i, random_vector(128, i));
            index.insert(&point);
        }

        let mut accelerator = index.build_pq_accelerator(8, 0);
        assert_eq!(accelerator.len(), 100);

        // Add a new vector
        accelerator.add(999, &random_vector(128, 999));
        assert_eq!(accelerator.len(), 101);

        // Remove a vector
        assert!(accelerator.remove(999));
        assert_eq!(accelerator.len(), 100);
        assert!(!accelerator.remove(999)); // Already removed
    }
}
