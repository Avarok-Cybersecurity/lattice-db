# Distance Metrics

LatticeDB supports multiple distance metrics for vector similarity search. This chapter explains each metric and when to use it.

## Available Metrics

### Cosine Distance

```rust
Distance::Cosine
```

Measures the angle between vectors, ignoring magnitude:

```
cosine_distance(a, b) = 1 - (a · b) / (||a|| × ||b||)
```

**Range**: [0, 2]
- 0 = identical direction
- 1 = orthogonal (90°)
- 2 = opposite direction

**Best for**:
- Text embeddings (word2vec, BERT, etc.)
- Normalized vectors
- When magnitude doesn't matter

**Example**:
```rust
let calc = DistanceCalculator::new(Distance::Cosine);

let a = vec![1.0, 0.0];
let b = vec![0.707, 0.707];  // 45° angle

let dist = calc.calculate(&a, &b);
// dist ≈ 0.293 (1 - cos(45°))
```

### Euclidean Distance (L2)

```rust
Distance::Euclid
```

Standard straight-line distance:

```
euclidean_distance(a, b) = sqrt(Σ(aᵢ - bᵢ)²)
```

**Range**: [0, ∞)
- 0 = identical vectors

**Best for**:
- Image embeddings
- Geographic coordinates
- Physical measurements
- When absolute differences matter

**Example**:
```rust
let calc = DistanceCalculator::new(Distance::Euclid);

let a = vec![0.0, 0.0];
let b = vec![3.0, 4.0];

let dist = calc.calculate(&a, &b);
// dist = 5.0 (3-4-5 triangle)
```

### Dot Product Distance

```rust
Distance::Dot
```

Negated dot product (so lower = more similar):

```
dot_distance(a, b) = -(a · b)
```

**Range**: (-∞, ∞)
- More negative = more similar (higher original dot product)

**Best for**:
- Maximum Inner Product Search (MIPS)
- Recommendation systems
- Pre-normalized vectors where you want raw similarity scores

**Example**:
```rust
let calc = DistanceCalculator::new(Distance::Dot);

let a = vec![1.0, 2.0, 3.0];
let b = vec![4.0, 5.0, 6.0];

let dist = calc.calculate(&a, &b);
// dist = -32 (negated: 1*4 + 2*5 + 3*6 = 32)
```

## Choosing a Metric

| Use Case | Recommended Metric | Reason |
|----------|-------------------|--------|
| Text embeddings | Cosine | Angle-based similarity, magnitude-invariant |
| Image embeddings | Euclidean | Pixel-level differences |
| Pre-normalized vectors | Dot | Faster (no normalization needed) |
| Recommendations | Dot | Higher dot product = higher relevance |
| Geographic data | Euclidean | Physical distance |

### Cosine vs Dot Product

If your vectors are **unit-normalized** (||v|| = 1), cosine and dot product are equivalent:

```
For unit vectors: cosine_similarity = dot_product
Therefore: cosine_distance = 1 - dot_product
```

LatticeDB provides a fast path for normalized vectors:

```rust
// Fast cosine distance for pre-normalized vectors (25-30% faster)
let dist = cosine_distance_normalized(&normalized_a, &normalized_b);
```

## Implementation Details

### Distance Calculator

All distance functions are accessed through `DistanceCalculator`:

```rust
use lattice_core::{DistanceCalculator, Distance};

let calc = DistanceCalculator::new(Distance::Cosine);

// Single calculation
let dist = calc.calculate(&vec_a, &vec_b);

// Get the metric type
assert_eq!(calc.metric(), Distance::Cosine);
```

### Lower is Better

All distance functions return values where **lower = more similar**:

```rust
// Identical vectors
let same = calc.calculate(&v, &v);
// same ≈ 0.0 (for all metrics)

// Most similar results first
results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
```

This convention enables consistent use with min-heaps in search algorithms.

### Dimension Requirements

All vectors must have the same dimension:

```rust
let a = vec![1.0, 2.0];
let b = vec![1.0, 2.0, 3.0];

// Panics in debug builds:
// calc.calculate(&a, &b);  // Dimension mismatch!

// In release builds, behavior is undefined
```

The index validates dimensions at insertion time.

## Performance

### SIMD Acceleration

Distance calculations are SIMD-accelerated on supported platforms:

| Platform | Instruction Set | Vectors Processed |
|----------|----------------|-------------------|
| x86_64 | AVX2 + FMA | 8 floats/cycle |
| aarch64 | NEON | 4-16 floats/cycle (4x unrolled) |
| WASM | Scalar | 4 floats (auto-vectorized) |

### Scalar Fallback

For small vectors or unsupported platforms, scalar code with 4x unrolling:

```rust
// Unrolled for better auto-vectorization
for i in 0..chunks {
    let base = i * 4;
    let d0 = a[base] - b[base];
    let d1 = a[base + 1] - b[base + 1];
    let d2 = a[base + 2] - b[base + 2];
    let d3 = a[base + 3] - b[base + 3];
    sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
}
```

### Benchmark Results

Typical throughput for 128-dimensional vectors:

| Metric | Scalar | SIMD (x86) | SIMD (aarch64) |
|--------|--------|------------|----------------|
| Cosine | 120 ns | 25 ns | 20 ns |
| Euclidean | 100 ns | 20 ns | 15 ns |
| Dot | 90 ns | 18 ns | 14 ns |

## Best Practices

### 1. Normalize Early

If using cosine distance, normalize vectors once at insertion:

```rust
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// Normalize before insertion
normalize(&mut embedding);
index.insert(&Point::new_vector(id, embedding));
```

### 2. Use Consistent Metrics

Always use the same metric for insertion and search:

```rust
// Create index with cosine distance
let mut index = HnswIndex::new(config, Distance::Cosine);

// Insertions use cosine distance internally
index.insert(&point);

// Searches use cosine distance
let results = index.search(&query, k, ef);
```

### 3. Check Vector Quality

Validate embeddings before insertion:

```rust
fn validate_vector(v: &[f32]) -> bool {
    // Check for NaN/Inf
    if v.iter().any(|&x| x.is_nan() || x.is_infinite()) {
        return false;
    }

    // Check for zero vectors (problematic for cosine)
    let norm: f32 = v.iter().map(|x| x * x).sum();
    if norm < 1e-10 {
        return false;
    }

    true
}
```

## Next Steps

- [SIMD Optimization](./simd.md) - Hardware-specific acceleration
- [Quantization](./quantization.md) - Memory-efficient distance computation
