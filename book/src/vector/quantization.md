# Quantization

Quantization reduces memory usage and can accelerate distance computation. LatticeDB supports two quantization methods:

1. **Scalar Quantization (SQ)**: 4x memory reduction (f32 → i8)
2. **Product Quantization (PQ)**: 32-64x memory reduction

## Scalar Quantization

### Overview

Scalar quantization maps each f32 value to an i8 using min-max scaling:

```
quantized[i] = round((original[i] - offset) / scale)
dequantized[i] = quantized[i] * scale + offset

where:
  offset = min(original)
  scale = (max(original) - min(original)) / 255
```

### Memory Savings

| Dimension | Original (f32) | Quantized (i8) | Savings |
|-----------|----------------|----------------|---------|
| 128 | 512 bytes | 136 bytes | 3.8x |
| 256 | 1024 bytes | 264 bytes | 3.9x |
| 512 | 2048 bytes | 520 bytes | 3.9x |
| 1024 | 4096 bytes | 1032 bytes | 4.0x |

The 8-byte overhead is for scale and offset values.

### Usage

```rust
use lattice_core::QuantizedVector;

// Quantize a vector
let original = vec![0.1, 0.5, 0.9, -0.3, 0.0];
let quantized = QuantizedVector::quantize(&original);

// Check memory usage
println!("Original: {} bytes", original.len() * 4);
println!("Quantized: {} bytes", quantized.memory_size());

// Dequantize (lossy)
let recovered = quantized.dequantize();

// Asymmetric distance (quantized DB vector vs f32 query)
let query = vec![0.2, 0.4, 0.8, -0.2, 0.1];
let distance = quantized.dot_distance_asymmetric(&query);
```

### Asymmetric Distance

For search, we compute distance between:
- **Database vectors**: Quantized (memory-efficient)
- **Query vector**: Full precision (accurate)

```rust
impl QuantizedVector {
    // Quantized × f32 distance
    pub fn dot_distance_asymmetric(&self, query: &[f32]) -> f32;
    pub fn euclidean_distance_asymmetric(&self, query: &[f32]) -> f32;
    pub fn cosine_distance_asymmetric(&self, query: &[f32]) -> f32;
}
```

### Accuracy

Scalar quantization maintains good accuracy for most use cases:

| Metric | Recall@10 (128-dim, 100K vectors) |
|--------|-----------------------------------|
| Original (f32) | 99.5% |
| Quantized (i8) | 98.2% |

The 1-2% recall loss is often acceptable given the 4x memory savings.

## Product Quantization (PQ)

### Overview

PQ achieves higher compression by:
1. Splitting vectors into M subvectors
2. Quantizing each subvector to a cluster centroid
3. Storing only the cluster index (1 byte per subvector)

```
Original: [f32; 128] = 512 bytes
Split:    8 subvectors of [f32; 16]
Quantize: 8 cluster indices = 8 bytes
Compression: 64x
```

### Building a PQ Accelerator

```rust
// After building the HNSW index
let accelerator = index.build_pq_accelerator(
    m: 8,                    // 8 subvectors
    training_sample_size: 10000,  // Vectors for training
);

// Check compression
println!("Compression: {}x", accelerator.compression_ratio());
// Output: Compression: 64x (for 128-dim vectors)
```

### PQ-Accelerated Search

PQ enables two-phase search:
1. **Coarse filtering**: Fast approximate distances using PQ codes
2. **Re-ranking**: Exact distances for top candidates

```rust
let results = index.search_with_pq(
    &query,
    k: 10,
    ef: 100,
    &accelerator,
    rerank_factor: 3,  // Re-rank 30 candidates for final 10
);
```

### Distance Table Lookup

For each query, PQ builds a distance table:

```rust
// Precompute distances from query to all centroids
let dist_table = accelerator.pq.build_distance_table(&query);

// O(M) distance lookup instead of O(D)
let approx_dist = accelerator.approximate_distance_with_table(&dist_table, point_id);
```

This reduces distance computation from O(D) to O(M), where M << D.

### PQ Memory Usage

| Dimension | M | Original | PQ Code | Codebook | Total Overhead |
|-----------|---|----------|---------|----------|----------------|
| 128 | 8 | 512 bytes/vec | 8 bytes/vec | 128 KB | 64x compression |
| 256 | 16 | 1024 bytes/vec | 16 bytes/vec | 256 KB | 64x compression |
| 512 | 32 | 2048 bytes/vec | 32 bytes/vec | 512 KB | 64x compression |

### PQ vs SQ Trade-offs

| Aspect | Scalar Quantization | Product Quantization |
|--------|---------------------|----------------------|
| Compression | 4x | 32-64x |
| Accuracy loss | ~1-2% | ~3-5% |
| Build time | Fast (O(N)) | Slow (k-means training) |
| Search speed | Same as f32 | Faster (O(M) distances) |
| Memory per vector | N bytes | M bytes |
| Use case | Moderate memory savings | Maximum compression |

## Combining Quantization Methods

For very large datasets, combine both methods:

```rust
// 1. Quantize database vectors with SQ (4x savings)
let quantized_vectors: Vec<QuantizedVector> = vectors
    .iter()
    .map(|v| QuantizedVector::quantize(v))
    .collect();

// 2. Build PQ accelerator for fast search (additional 16x in search)
let accelerator = index.build_pq_accelerator(8, 10000);

// 3. Two-phase search:
//    - PQ for coarse filtering (very fast)
//    - SQ for re-ranking (accurate enough)
let candidates = index.search_with_pq(&query, 100, 200, &accelerator, 1);
let results = rerank_with_sq(&candidates, &quantized_vectors, &query, k);
```

## When to Use Quantization

| Dataset Size | Recommendation |
|--------------|----------------|
| < 100K | No quantization needed |
| 100K - 1M | Scalar quantization |
| 1M - 10M | Scalar + PQ acceleration |
| > 10M | Full PQ with re-ranking |

## Best Practices

### 1. Train PQ on Representative Data

```rust
// Use a sample that represents your data distribution
let training_sample: Vec<Vector> = dataset
    .iter()
    .step_by(dataset.len() / 10000)
    .cloned()
    .collect();

let accelerator = index.build_pq_accelerator(8, training_sample.len());
```

### 2. Choose M Based on Dimension

```rust
// Rule of thumb: dim / M should be >= 8
let m = match dim {
    d if d <= 64 => 4,
    d if d <= 128 => 8,
    d if d <= 256 => 16,
    d if d <= 512 => 32,
    _ => 64,
};
```

### 3. Re-rank Enough Candidates

```rust
// Higher rerank_factor = better recall, slower search
let rerank_factor = match recall_target {
    r if r >= 0.99 => 5,
    r if r >= 0.95 => 3,
    _ => 2,
};
```

### 4. Update PQ Incrementally

```rust
// When adding new vectors
index.insert(&new_point);
accelerator.add(new_point.id, &new_point.vector);

// When removing vectors
index.delete(point_id);
accelerator.remove(point_id);
```

## Next Steps

- [SIMD Optimization](./simd.md) - Hardware acceleration for distance calculations
- [Performance Tuning](../performance/tuning.md) - Overall optimization strategies
