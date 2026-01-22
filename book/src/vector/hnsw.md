# HNSW Index

LatticeDB uses **HNSW (Hierarchical Navigable Small World)** for approximate nearest neighbor search. This chapter explains how the algorithm works and how to tune it for your use case.

## Algorithm Overview

HNSW constructs a multi-layer graph where:
- **Upper layers** are sparse and allow quick navigation to the general region
- **Lower layers** are dense and enable precise local search
- **Layer 0** contains all points with the maximum connectivity

```
Layer 2:  ●───────────────────────●  (sparse, fast navigation)
          │                       │
          ▼                       ▼
Layer 1:  ●─────●─────●─────●─────●  (medium density)
          │     │     │     │     │
          ▼     ▼     ▼     ▼     ▼
Layer 0:  ●─●─●─●─●─●─●─●─●─●─●─●─●  (dense, all points)
```

## Search Process

### Phase 1: Top-Down Navigation

Starting from a random entry point at the top layer:

1. Find the nearest neighbor in the current layer (greedy search)
2. Use that node as the entry point for the next layer
3. Repeat until reaching layer 0

```rust
// Simplified pseudocode
let mut current = entry_point;
for layer in (1..=max_layer).rev() {
    current = find_nearest_in_layer(query, current, layer);
}
```

### Phase 2: Layer 0 Search

At layer 0, perform a beam search to find `ef` candidates:

1. Maintain a priority queue of candidates (sorted by distance)
2. Explore neighbors of the closest unvisited candidate
3. Add promising neighbors to the queue
4. Stop when the closest candidate is farther than the farthest result

```rust
// Returns top-k results from ef candidates
let candidates = search_layer(query, entry_points, ef, layer=0);
candidates.into_iter().take(k).collect()
```

## Configuration

### HnswConfig Parameters

```rust
pub struct HnswConfig {
    pub m: usize,               // Connections per node (default: 16)
    pub m0: usize,              // Layer 0 connections (default: 2 * m = 32)
    pub ml: f64,                // Level multiplier (default: 1/ln(m))
    pub ef: usize,              // Search queue size (default: 100)
    pub ef_construction: usize, // Build queue size (default: 200)
}
```

### Parameter Guidelines

| Parameter | Description | Trade-off |
|-----------|-------------|-----------|
| `m` | Connections per node in upper layers | Higher = better recall, more memory |
| `m0` | Connections per node in layer 0 | Higher = better recall, more memory |
| `ef` | Search queue size | Higher = better recall, slower search |
| `ef_construction` | Build queue size | Higher = better index quality, slower build |

### Recommended Values

| Dataset Size | m | m0 | ef_construction | Notes |
|--------------|---|----|-----------------| ----- |
| < 10K | 8 | 16 | 100 | Low memory, fast build |
| 10K - 100K | 16 | 32 | 200 | Balanced (default) |
| 100K - 1M | 24 | 48 | 300 | Higher recall |
| > 1M | 32 | 64 | 400 | Maximum recall |

## Usage

### Basic Search

```rust
use lattice_core::{HnswIndex, Distance, HnswConfig};

// Create index with default config
let config = HnswConfig::default();
let mut index = HnswIndex::new(config, Distance::Cosine);

// Insert points
for point in points {
    index.insert(&point);
}

// Search: find 10 nearest neighbors with ef=100
let results = index.search(&query_vector, k=10, ef=100);
```

### Batch Search

For multiple queries, batch search is more efficient:

```rust
// Prepare query references
let query_refs: Vec<&[f32]> = queries.iter()
    .map(|q| q.as_slice())
    .collect();

// Parallel batch search (uses rayon on native)
let results = index.search_batch(&query_refs, k=10, ef=100);
```

### Adjusting ef at Search Time

`ef` can be adjusted per-query to trade off speed vs recall:

```rust
// Fast search (lower recall)
let quick_results = index.search(&query, 10, ef=50);

// High-recall search (slower)
let accurate_results = index.search(&query, 10, ef=200);
```

## Memory Layout

### Dense Vector Storage

Vectors are stored in a flat, contiguous array for cache efficiency:

```
┌──────────────────────────────────────────────────────┐
│ [v0_d0, v0_d1, ..., v0_dn] [v1_d0, v1_d1, ..., v1_dn] │
└──────────────────────────────────────────────────────┘
     Vector 0 (dim=n)            Vector 1 (dim=n)
```

Access pattern: `&vectors[id * dim .. (id + 1) * dim]`

Benefits:
- **Cache-friendly**: Sequential memory access during index construction
- **SIMD-friendly**: Aligned access for vectorized distance calculations
- **Predictable**: O(1) indexed access via dense array

### Layer Storage

Each layer stores:
- **Node list**: Points present at this layer
- **Neighbor lists**: Connections for each node

```rust
struct HnswNode {
    id: PointId,
    level: u16,  // Highest layer this node appears in
    // neighbors[layer] = Vec<PointId>
    neighbors: SmallVec<[Vec<PointId>; 4]>,
}
```

## Optimizations

### 1. Shortcut Search (VLDB 2025)

If the best neighbor doesn't change at a layer, we can skip redundant distance calculations:

```rust
// Track if we improved at each layer
let (new_current, new_dist, improved) =
    search_layer_single_with_shortcut(query, current, current_dist, layer);

if !improved && layer > 1 {
    // Can potentially skip to next layer faster
    continue;
}
```

### 2. Software Prefetching

Hide memory latency by prefetching future vectors:

```rust
// Prefetch vectors ahead of current iteration
const PREFETCH_DISTANCE: usize = 4;

for (i, &id) in neighbor_ids.iter().enumerate() {
    if i + PREFETCH_DISTANCE < neighbor_ids.len() {
        prefetch_read(vectors.get_ptr(neighbor_ids[i + PREFETCH_DISTANCE]));
    }
    // Calculate distance for current id
    distances.push(calc_distance(query, vectors.get(id)));
}
```

### 3. Thread-Local Scratch Space

Avoid allocation per search by reusing scratch space:

```rust
thread_local! {
    static SCRATCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::new());
}

fn search_layer(&self, ...) -> Vec<Candidate> {
    SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.clear();  // Reuse allocated memory
        // ... perform search ...
    })
}
```

### 4. Vec-Based Results

Use a Vec instead of BinaryHeap for results, with periodic compaction:

```rust
// Add to results
results.push(candidate);

// Compact when 2x over limit (amortizes sort cost)
if results.len() >= ef * 2 {
    results.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    results.truncate(ef);
}
```

## PQ Acceleration

For very large indexes, Product Quantization can accelerate search:

```rust
// Build PQ accelerator (one-time cost)
let accelerator = index.build_pq_accelerator(
    m: 8,  // 8 subvectors for 128-dim
    training_sample_size: 10000,
);

// Search with PQ-accelerated coarse filtering
let results = index.search_with_pq(
    &query,
    k: 10,
    ef: 100,
    &accelerator,
    rerank_factor: 3,  // Re-rank top 30 candidates
);
```

See [Quantization](./quantization.md) for details.

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Insert | O(log N × m × ef_construction) | Per-point amortized |
| Search | O(log N × m × ef) | Per-query |
| Delete | O(m) | Just removes node and edges |
| Memory | O(N × (dim + m × layers)) | Vectors + graph |

Typical search latency (128-dim, 100K vectors, cosine):
- **ef=50**: ~50 µs
- **ef=100**: ~100 µs
- **ef=200**: ~200 µs

## Next Steps

- [Distance Metrics](./distance.md) - Choosing the right distance function
- [Quantization](./quantization.md) - Memory-efficient storage
- [SIMD Optimization](./simd.md) - Hardware acceleration
