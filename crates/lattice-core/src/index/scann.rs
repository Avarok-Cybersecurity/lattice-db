//! ScaNN-style Product Quantization with Anisotropic Weighting
//!
//! Implements Google's ScaNN approach for fast approximate nearest neighbor search:
//! - Product Quantization (PQ) for compressed vector representation
//! - Anisotropic quantization that weights parallel error more heavily
//! - Asymmetric Distance Computation (ADC) using lookup tables
//!
//! # References
//! - [ScaNN Paper](https://arxiv.org/abs/1908.10396)
//! - [Google Blog](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/)
//!
//! # Architecture
//!
//! ```text
//! Original Vector (128 dims)
//!        │
//!        ▼
//! ┌──────────────────────────────────┐
//! │  Split into M subvectors (M=8)   │
//! │  Each subvector: 128/8 = 16 dims │
//! └──────────────────────────────────┘
//!        │
//!        ▼
//! ┌──────────────────────────────────┐
//! │  Quantize each to nearest        │
//! │  centroid (K=256 centroids)      │
//! └──────────────────────────────────┘
//!        │
//!        ▼
//! Compressed: M bytes (8 bytes for M=8)
//! ```

use crate::types::point::{PointId, Vector};
use std::collections::HashMap;

/// Number of centroids per subquantizer (2^8 = 256)
const DEFAULT_K: usize = 256;
/// Default number of k-means iterations
const DEFAULT_KMEANS_ITERS: usize = 25;
/// Convergence threshold for k-means
const KMEANS_CONVERGENCE: f32 = 1e-6;

/// Product Quantizer for vector compression
///
/// Divides vectors into M subvectors and quantizes each independently
/// using K centroids learned via k-means clustering.
#[derive(Clone)]
pub struct ProductQuantizer {
    /// Number of subvectors
    m: usize,
    /// Dimension of each subvector
    dsub: usize,
    /// Original vector dimension
    dim: usize,
    /// Number of centroids per subquantizer
    k: usize,
    /// Codebooks: m codebooks, each with k centroids of dsub dimensions
    /// Shape: [m][k][dsub]
    codebooks: Vec<Vec<Vec<f32>>>,
    /// Anisotropic weight (higher = more weight on parallel error)
    /// 0.0 = isotropic (standard PQ), 1.0+ = anisotropic
    anisotropic_weight: f32,
}

impl ProductQuantizer {
    /// Create a new product quantizer
    ///
    /// # Arguments
    /// * `dim` - Vector dimension (must be divisible by m)
    /// * `m` - Number of subvectors (typically 8, 16, or 32)
    /// * `k` - Number of centroids per subquantizer (typically 256)
    pub fn new(dim: usize, m: usize, k: usize) -> Self {
        assert!(dim % m == 0, "dim ({}) must be divisible by m ({})", dim, m);
        assert!(k > 0 && k <= 256, "k must be between 1 and 256");

        let dsub = dim / m;

        Self {
            m,
            dsub,
            dim,
            k,
            codebooks: vec![vec![vec![0.0; dsub]; k]; m],
            anisotropic_weight: 0.2, // Default anisotropic weight from ScaNN paper
        }
    }

    /// Create with default parameters for a given dimension
    pub fn default_for_dim(dim: usize) -> Self {
        // Choose M based on dimension
        let m = if dim >= 128 {
            8
        } else if dim >= 64 {
            4
        } else {
            2
        };
        Self::new(dim, m, DEFAULT_K)
    }

    /// Set anisotropic weight
    ///
    /// Higher values weight parallel error more heavily, which improves
    /// inner product / cosine similarity approximation.
    pub fn with_anisotropic_weight(mut self, weight: f32) -> Self {
        self.anisotropic_weight = weight;
        self
    }

    /// Train the quantizer on a set of vectors using k-means
    ///
    /// # Arguments
    /// * `vectors` - Training vectors (should be representative sample)
    /// * `max_iters` - Maximum k-means iterations
    pub fn train(&mut self, vectors: &[Vector], max_iters: usize) {
        if vectors.is_empty() {
            return;
        }

        // Train each subquantizer independently
        for sub_idx in 0..self.m {
            let start = sub_idx * self.dsub;
            let end = start + self.dsub;

            // Extract subvectors for this segment
            let subvectors: Vec<Vec<f32>> =
                vectors.iter().map(|v| v[start..end].to_vec()).collect();

            // Run k-means clustering
            let centroids =
                kmeans_clustering(&subvectors, self.k, max_iters, self.anisotropic_weight);

            self.codebooks[sub_idx] = centroids;
        }
    }

    /// Train with default iterations
    pub fn train_default(&mut self, vectors: &[Vector]) {
        self.train(vectors, DEFAULT_KMEANS_ITERS);
    }

    /// Encode a vector to PQ codes
    ///
    /// Returns M bytes, each representing the nearest centroid index
    /// for the corresponding subvector.
    pub fn encode(&self, vector: &[f32]) -> PQCode {
        debug_assert_eq!(vector.len(), self.dim);

        let mut codes = vec![0u8; self.m];

        for sub_idx in 0..self.m {
            let start = sub_idx * self.dsub;
            let subvec = &vector[start..start + self.dsub];

            // Find nearest centroid
            let mut best_idx = 0;
            let mut best_dist = f32::MAX;

            for (idx, centroid) in self.codebooks[sub_idx].iter().enumerate() {
                let dist = squared_euclidean(subvec, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx;
                }
            }

            // Safety: k is validated to be <= 256 in constructor, so best_idx fits in u8
            debug_assert!(best_idx < 256, "centroid index {} exceeds u8 range", best_idx);
            codes[sub_idx] = best_idx as u8;
        }

        PQCode { codes }
    }

    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, code: &PQCode) -> Vector {
        let mut vector = vec![0.0; self.dim];

        for sub_idx in 0..self.m {
            let centroid_idx = code.codes[sub_idx] as usize;
            let centroid = &self.codebooks[sub_idx][centroid_idx];

            let start = sub_idx * self.dsub;
            vector[start..start + self.dsub].copy_from_slice(centroid);
        }

        vector
    }

    /// Build a distance lookup table for a query vector
    ///
    /// This precomputes distances from query subvectors to all centroids,
    /// enabling O(M) distance computation instead of O(D).
    pub fn build_distance_table(&self, query: &[f32]) -> DistanceTable {
        debug_assert_eq!(query.len(), self.dim);

        // Table shape: [m][k] - distance from query subvector to each centroid
        let mut table = vec![vec![0.0f32; self.k]; self.m];

        for sub_idx in 0..self.m {
            let start = sub_idx * self.dsub;
            let query_sub = &query[start..start + self.dsub];

            for (centroid_idx, centroid) in self.codebooks[sub_idx].iter().enumerate() {
                table[sub_idx][centroid_idx] = squared_euclidean(query_sub, centroid);
            }
        }

        DistanceTable { table }
    }

    /// Build an inner product lookup table for a query vector
    ///
    /// For cosine/dot product similarity, we use inner products instead of L2.
    pub fn build_inner_product_table(&self, query: &[f32]) -> DistanceTable {
        debug_assert_eq!(query.len(), self.dim);

        let mut table = vec![vec![0.0f32; self.k]; self.m];

        for sub_idx in 0..self.m {
            let start = sub_idx * self.dsub;
            let query_sub = &query[start..start + self.dsub];

            for (centroid_idx, centroid) in self.codebooks[sub_idx].iter().enumerate() {
                // Negative inner product (so lower = more similar)
                table[sub_idx][centroid_idx] = -inner_product(query_sub, centroid);
            }
        }

        DistanceTable { table }
    }

    /// Compute asymmetric distance using precomputed table
    ///
    /// This is O(M) instead of O(D) - the key speedup from PQ.
    #[inline]
    pub fn asymmetric_distance(&self, table: &DistanceTable, code: &PQCode) -> f32 {
        let mut dist = 0.0;
        for sub_idx in 0..self.m {
            let centroid_idx = code.codes[sub_idx] as usize;
            dist += table.table[sub_idx][centroid_idx];
        }
        dist
    }

    /// Get number of subvectors
    pub fn num_subvectors(&self) -> usize {
        self.m
    }

    /// Get subvector dimension
    pub fn subvector_dim(&self) -> usize {
        self.dsub
    }

    /// Get number of centroids
    pub fn num_centroids(&self) -> usize {
        self.k
    }

    /// Get compression ratio (original bytes / compressed bytes)
    pub fn compression_ratio(&self) -> f32 {
        (self.dim * 4) as f32 / self.m as f32
    }

    /// Check if trained (has non-zero codebooks)
    pub fn is_trained(&self) -> bool {
        self.codebooks
            .iter()
            .any(|cb| cb.iter().any(|c| c.iter().any(|&v| v != 0.0)))
    }
}

/// Compressed vector representation (M bytes)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PQCode {
    /// Centroid indices for each subvector
    pub codes: Vec<u8>,
}

impl PQCode {
    /// Create from raw bytes
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            codes: bytes.to_vec(),
        }
    }

    /// Get as bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.codes
    }

    /// Size in bytes
    pub fn size(&self) -> usize {
        self.codes.len()
    }
}

/// Precomputed distance table for fast asymmetric distance computation
pub struct DistanceTable {
    /// Distance from query subvector to each centroid
    /// Shape: [m][k]
    table: Vec<Vec<f32>>,
}

impl DistanceTable {
    /// Get distance for a subvector index and centroid index
    #[inline]
    pub fn get(&self, sub_idx: usize, centroid_idx: usize) -> f32 {
        self.table[sub_idx][centroid_idx]
    }
}

/// PQ-compressed index for fast approximate search
///
/// Stores vectors in compressed form and uses ADC for distance computation.
pub struct PQIndex {
    /// Product quantizer
    pq: ProductQuantizer,
    /// Compressed vectors: point_id -> PQ code
    codes: HashMap<PointId, PQCode>,
    /// Original vectors (optional, for re-ranking)
    vectors: Option<HashMap<PointId, Vector>>,
}

impl PQIndex {
    /// Create a new PQ index
    pub fn new(pq: ProductQuantizer) -> Self {
        Self {
            pq,
            codes: HashMap::new(),
            vectors: None,
        }
    }

    /// Create with original vectors stored for re-ranking
    pub fn with_reranking(pq: ProductQuantizer) -> Self {
        Self {
            pq,
            codes: HashMap::new(),
            vectors: Some(HashMap::new()),
        }
    }

    /// Add a vector to the index
    pub fn add(&mut self, id: PointId, vector: &Vector) {
        let code = self.pq.encode(vector);
        self.codes.insert(id, code);

        if let Some(ref mut vectors) = self.vectors {
            vectors.insert(id, vector.clone());
        }
    }

    /// Remove a vector from the index
    pub fn remove(&mut self, id: PointId) -> bool {
        let removed = self.codes.remove(&id).is_some();
        if let Some(ref mut vectors) = self.vectors {
            vectors.remove(&id);
        }
        removed
    }

    /// Search for k nearest neighbors using ADC
    ///
    /// Returns candidates sorted by approximate distance (lower = closer).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(PointId, f32)> {
        // Build distance table
        let table = self.pq.build_distance_table(query);

        // Compute distances to all vectors
        let mut candidates: Vec<(PointId, f32)> = self
            .codes
            .iter()
            .map(|(&id, code)| {
                let dist = self.pq.asymmetric_distance(&table, code);
                (id, dist)
            })
            .collect();

        // Sort by distance
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top k
        candidates.truncate(k);
        candidates
    }

    /// Search with re-ranking using exact distances
    ///
    /// First retrieves top candidates using ADC, then re-ranks with exact distances.
    pub fn search_with_rerank(
        &self,
        query: &[f32],
        k: usize,
        rerank_factor: usize,
    ) -> Vec<(PointId, f32)> {
        let vectors = match &self.vectors {
            Some(v) => v,
            None => return self.search(query, k),
        };

        // Get more candidates for re-ranking
        let candidates = self.search(query, k * rerank_factor);

        // Re-rank with exact distances
        let mut reranked: Vec<(PointId, f32)> = candidates
            .into_iter()
            .filter_map(|(id, _)| {
                vectors.get(&id).map(|v| {
                    let exact_dist = squared_euclidean(query, v);
                    (id, exact_dist)
                })
            })
            .collect();

        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);
        reranked
    }

    /// Get number of vectors
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Get the product quantizer
    pub fn quantizer(&self) -> &ProductQuantizer {
        &self.pq
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let codes_size = self.codes.len() * self.pq.num_subvectors();
        let vectors_size = self
            .vectors
            .as_ref()
            .map_or(0, |v| v.len() * self.pq.dim * std::mem::size_of::<f32>());
        codes_size + vectors_size
    }
}

// --- K-means clustering ---

/// K-means clustering with optional anisotropic weighting
fn kmeans_clustering(
    vectors: &[Vec<f32>],
    k: usize,
    max_iters: usize,
    anisotropic_weight: f32,
) -> Vec<Vec<f32>> {
    if vectors.is_empty() {
        return vec![];
    }

    let dim = vectors[0].len();
    let n = vectors.len();
    let k = k.min(n); // Can't have more centroids than vectors

    // Initialize centroids using k-means++ style initialization
    let mut centroids = kmeans_init(vectors, k);
    let mut assignments = vec![0usize; n];
    let mut prev_inertia = f32::MAX;

    for _iter in 0..max_iters {
        // Assignment step: assign each vector to nearest centroid
        let mut inertia = 0.0;
        for (i, vec) in vectors.iter().enumerate() {
            let mut best_idx = 0;
            let mut best_dist = f32::MAX;

            for (j, centroid) in centroids.iter().enumerate() {
                let dist = anisotropic_distance(vec, centroid, anisotropic_weight);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j;
                }
            }

            assignments[i] = best_idx;
            inertia += best_dist;
        }

        // Check convergence
        if (prev_inertia - inertia).abs() < KMEANS_CONVERGENCE * prev_inertia {
            break;
        }
        prev_inertia = inertia;

        // Update step: recompute centroids
        let mut new_centroids = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];

        for (i, vec) in vectors.iter().enumerate() {
            let cluster = assignments[i];
            counts[cluster] += 1;
            for (d, &v) in vec.iter().enumerate() {
                new_centroids[cluster][d] += v;
            }
        }

        // Normalize
        for (j, centroid) in new_centroids.iter_mut().enumerate() {
            if counts[j] > 0 {
                for v in centroid.iter_mut() {
                    *v /= counts[j] as f32;
                }
            } else {
                // Empty cluster: reinitialize to random vector
                if let Some(random_vec) = vectors.get(j % n) {
                    centroid.copy_from_slice(random_vec);
                }
            }
        }

        centroids = new_centroids;
    }

    centroids
}

/// K-means++ initialization
fn kmeans_init(vectors: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let mut centroids = Vec::with_capacity(k);

    // Simple deterministic initialization: spread evenly through data
    // (Full k-means++ would use random sampling with distance-based probabilities)
    let step = n / k;
    for i in 0..k {
        let idx = (i * step) % n;
        centroids.push(vectors[idx].clone());
    }

    centroids
}

/// Anisotropic distance computation
///
/// Weights parallel component of error more heavily for better inner product approximation.
#[inline]
fn anisotropic_distance(a: &[f32], b: &[f32], weight: f32) -> f32 {
    if weight == 0.0 {
        return squared_euclidean(a, b);
    }

    // Compute norms and inner product
    let mut dot_ab = 0.0;
    let mut norm_a_sq = 0.0;
    let mut norm_b_sq = 0.0;
    let mut diff_sq = 0.0;

    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot_ab += ai * bi;
        norm_a_sq += ai * ai;
        norm_b_sq += bi * bi;
        let d = ai - bi;
        diff_sq += d * d;
    }

    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return diff_sq;
    }

    // Parallel error: difference in direction of a
    // Project b onto a, compute squared error of that projection vs a
    let proj_scale = dot_ab / norm_a_sq;
    let mut parallel_error_sq = 0.0;
    for (&ai, _bi) in a.iter().zip(b.iter()) {
        let parallel_b = ai * proj_scale;
        let parallel_diff = ai - parallel_b;
        parallel_error_sq += parallel_diff * parallel_diff;
    }

    // Combined distance with anisotropic weighting
    let perpendicular_error_sq = diff_sq - parallel_error_sq;
    (1.0 + weight) * parallel_error_sq + perpendicular_error_sq
}

/// Squared Euclidean distance
#[inline]
fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai - bi;
            d * d
        })
        .sum()
}

/// Inner product
#[inline]
fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize, seed: u64) -> Vector {
        let mut rng = seed;
        (0..dim)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5) as f32
            })
            .collect()
    }

    #[test]
    fn test_pq_encode_decode() {
        let dim = 128;
        let m = 8;
        let k = 256;

        let mut pq = ProductQuantizer::new(dim, m, k);

        // Generate training vectors
        let vectors: Vec<Vector> = (0..1000).map(|i| random_vector(dim, i)).collect();
        pq.train_default(&vectors);

        // Test encode/decode
        let original = random_vector(dim, 9999);
        let code = pq.encode(&original);
        let decoded = pq.decode(&code);

        // Check dimensions
        assert_eq!(code.codes.len(), m);
        assert_eq!(decoded.len(), dim);

        // Check reconstruction error is reasonable
        let error: f32 = original
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let rmse = (error / dim as f32).sqrt();

        // RMSE should be relatively small after training
        assert!(rmse < 1.0, "RMSE too high: {}", rmse);
    }

    #[test]
    fn test_pq_compression_ratio() {
        let dim = 128;
        let m = 8;

        let pq = ProductQuantizer::new(dim, m, 256);

        // Original: 128 * 4 = 512 bytes
        // Compressed: 8 bytes
        // Ratio: 64x
        let ratio = pq.compression_ratio();
        assert!((ratio - 64.0).abs() < 0.01);
    }

    #[test]
    fn test_asymmetric_distance() {
        let dim = 128;
        let m = 8;

        let mut pq = ProductQuantizer::new(dim, m, 256);

        // Generate and train
        let vectors: Vec<Vector> = (0..500).map(|i| random_vector(dim, i)).collect();
        pq.train_default(&vectors);

        // Test ADC
        let query = random_vector(dim, 9999);
        let table = pq.build_distance_table(&query);

        let code = pq.encode(&vectors[0]);
        let adc_dist = pq.asymmetric_distance(&table, &code);

        // ADC distance should be non-negative
        assert!(adc_dist >= 0.0);

        // Self-distance should be small
        let self_code = pq.encode(&query);
        let self_dist = pq.asymmetric_distance(&table, &self_code);
        assert!(self_dist < adc_dist || self_dist < 1.0);
    }

    #[test]
    fn test_pq_index_search() {
        let dim = 64;
        let m = 4;
        let n = 500;
        let k = 10;

        let mut pq = ProductQuantizer::new(dim, m, 256);

        // Generate vectors
        let vectors: Vec<Vector> = (0..n).map(|i| random_vector(dim, i as u64)).collect();
        pq.train_default(&vectors);

        // Build index
        let mut index = PQIndex::new(pq);
        for (i, vec) in vectors.iter().enumerate() {
            index.add(i as u64, vec);
        }

        assert_eq!(index.len(), n);

        // Search
        let query = random_vector(dim, 9999);
        let results = index.search(&query, k);

        assert_eq!(results.len(), k);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_pq_index_with_rerank() {
        let dim = 64;
        let m = 4;
        let n = 500;
        let k = 10;

        let mut pq = ProductQuantizer::new(dim, m, 256);

        // Generate vectors
        let vectors: Vec<Vector> = (0..n).map(|i| random_vector(dim, i as u64)).collect();
        pq.train_default(&vectors);

        // Build index with reranking
        let mut index = PQIndex::with_reranking(pq);
        for (i, vec) in vectors.iter().enumerate() {
            index.add(i as u64, vec);
        }

        // Search with reranking
        let query = random_vector(dim, 9999);
        let results = index.search_with_rerank(&query, k, 5);

        assert_eq!(results.len(), k);
    }

    #[test]
    fn test_kmeans_convergence() {
        let dim = 16;
        let n = 100;
        let k = 10;

        let vectors: Vec<Vec<f32>> = (0..n).map(|i| random_vector(dim, i as u64)).collect();

        let centroids = kmeans_clustering(&vectors, k, 50, 0.0);

        assert_eq!(centroids.len(), k);
        assert!(centroids.iter().all(|c| c.len() == dim));
    }
}
