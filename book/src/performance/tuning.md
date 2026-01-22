# Tuning Guide

This chapter covers performance optimization strategies for LatticeDB, including HNSW parameters, memory configuration, and query optimization.

## HNSW Parameter Tuning

### Core Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `m` | 16 | 4-64 | Connections per node |
| `m0` | 32 | 8-128 | Layer 0 connections |
| `ef_construction` | 200 | 50-500 | Build quality |
| `ef` | 100 | 10-500 | Search quality |

### Tuning for Recall

Higher recall (accuracy) requires:
- Higher `m` and `m0`
- Higher `ef_construction`
- Higher `ef` at search time

```rust
// High recall configuration
let config = HnswConfig {
    m: 32,
    m0: 64,
    ef_construction: 400,
    ef: 200,
    ml: HnswConfig::recommended_ml(32),
};
```

**Trade-offs:**
- Higher memory usage (more edges per node)
- Slower index construction
- Slower searches (more candidates explored)

### Tuning for Speed

Faster search with acceptable recall:
- Lower `m` and `m0`
- Lower `ef` at search time

```rust
// Fast search configuration
let config = HnswConfig {
    m: 8,
    m0: 16,
    ef_construction: 100,
    ef: 50,
    ml: HnswConfig::recommended_ml(8),
};
```

**Trade-offs:**
- Lower recall (may miss some neighbors)
- Lower memory usage

### Tuning for Memory

Minimize memory footprint:

```rust
// Memory-efficient configuration
let config = HnswConfig {
    m: 8,
    m0: 16,
    ef_construction: 100,
    ef: 100,
    ml: HnswConfig::recommended_ml(8),
};
```

Combine with quantization:
```rust
// Use scalar quantization (4x memory reduction)
let quantized = QuantizedVector::quantize(&vector);
```

### Dataset Size Guidelines

| Dataset Size | m | m0 | ef_construction | Memory/Vector |
|--------------|---|----|-----------------| --------------|
| < 1K | 8 | 16 | 100 | ~200 bytes |
| 1K - 10K | 12 | 24 | 150 | ~300 bytes |
| 10K - 100K | 16 | 32 | 200 | ~400 bytes |
| 100K - 1M | 24 | 48 | 300 | ~600 bytes |
| > 1M | 32 | 64 | 400 | ~800 bytes |

## Search Optimization

### Adjusting ef at Runtime

`ef` can be tuned per-query:

```rust
// Quick search (lower recall)
let fast_results = engine.search(&query.with_ef(50))?;

// High-quality search (higher recall)
let accurate_results = engine.search(&query.with_ef(300))?;
```

### Batch Queries

For multiple queries, use batch search:

```rust
// 5-10x faster than individual searches
let results = index.search_batch(&queries, k=10, ef=100);
```

Benefits:
- Parallel processing (on native)
- Better cache utilization
- Amortized overhead

### Pre-filtering

Filter before vector search when possible:

```rust
// Instead of post-filtering 10K results...
let all_results = engine.search(&query.with_limit(10000))?;
let filtered: Vec<_> = all_results
    .into_iter()
    .filter(|r| r.payload.get("category") == Some("tech"))
    .take(10)
    .collect();

// Pre-filter using graph/index
let tech_ids = engine.get_nodes_by_label("tech")?;
let results = engine.search_among(&query, &tech_ids, k=10)?;
```

## Memory Optimization

### Vector Storage Options

| Option | Memory | Speed | Use Case |
|--------|--------|-------|----------|
| Dense (default) | 100% | Fastest | Small-medium datasets |
| Scalar Quantized | 25% | 95% | Large datasets |
| Product Quantized | 3-5% | 80% | Very large datasets |
| Memory-mapped | Variable | 90% | Larger than RAM |

### Enabling Quantization

```rust
// Scalar quantization
use lattice_core::QuantizedVector;

let quantized_vectors: Vec<QuantizedVector> = vectors
    .iter()
    .map(|v| QuantizedVector::quantize(v))
    .collect();

// Product quantization accelerator
let accelerator = index.build_pq_accelerator(m=8, training_size=10000);
let results = index.search_with_pq(&query, k, ef, &accelerator, rerank=3);
```

### Memory-Mapped Storage (Native)

```rust
// Export vectors to mmap file
index.export_vectors_mmap(Path::new("vectors.mmap"))?;

// Load with mmap (vectors stay on disk, loaded on demand)
let mmap_store = MmapVectorStore::open(Path::new("vectors.mmap"))?;
```

## Storage Optimization

### Choosing a Backend

| Backend | Persistence | Speed | Use Case |
|---------|-------------|-------|----------|
| MemStorage | No | Fastest | Testing, ephemeral |
| DiskStorage | Yes | Fast | Server deployments |
| OpfsStorage | Yes | Medium | Browser persistent |
| IndexedDB | Yes | Slower | Browser fallback |

### Page Size Tuning

For disk storage, larger pages improve sequential access:

```rust
// Default: 4KB pages
let storage = DiskStorage::with_page_size(4096);

// Larger pages for bulk operations
let storage = DiskStorage::with_page_size(64 * 1024); // 64KB
```

## Query Optimization

### Cypher Query Patterns

**Use labels for filtering:**
```cypher
-- Good: Uses label index
MATCH (n:Person) WHERE n.age > 25 RETURN n

-- Less efficient: Full scan
MATCH (n) WHERE n.type = 'Person' AND n.age > 25 RETURN n
```

**Limit early:**
```cypher
-- Good: Limits before ordering
MATCH (n:Person) RETURN n ORDER BY n.name LIMIT 10

-- Less efficient: Orders everything first
MATCH (n:Person) RETURN n ORDER BY n.name
```

**Use parameters:**
```cypher
-- Good: Query can be cached
MATCH (n:Person {name: $name}) RETURN n

-- Less efficient: New query parse each time
MATCH (n:Person {name: 'Alice'}) RETURN n
```

### Hybrid Query Optimization

**Vector-first for similarity:**
```rust
// Good: Vector search narrows candidates
let similar = engine.search(&query.with_limit(100))?;
let expanded = expand_graph(&similar);

// Less efficient: Graph-first with large result set
let all_docs = engine.query("MATCH (n:Document) RETURN n")?;
let similar = filter_by_vector(&all_docs, &query);
```

**Graph-first for structured queries:**
```rust
// Good: Graph query with few results
let authors = engine.query(
    "MATCH (p:Person)-[:AUTHORED]->(d:Document {topic: $topic}) RETURN p",
    params
)?;
let ranked = rank_by_vector(&authors, &query);

// Less efficient: Vector search on entire corpus
let all_similar = engine.search(&query.with_limit(1000))?;
let authors = filter_by_graph(&all_similar);
```

## Monitoring

### Memory Statistics

```rust
let stats = engine.stats();
println!("Vectors: {} ({} bytes)", stats.vector_count, stats.vector_bytes);
println!("Index: {} bytes", stats.index_bytes);
println!("Graph: {} edges", stats.edge_count);
```

### Query Performance

```rust
use std::time::Instant;

let start = Instant::now();
let results = engine.search(&query)?;
let duration = start.elapsed();

println!("Search took {:?}", duration);
println!("Returned {} results", results.len());
```

### Profiling

```bash
# CPU profiling with flamegraph
cargo install flamegraph
cargo flamegraph --bin my_benchmark

# Memory profiling
cargo install heaptrack
heaptrack ./target/release/my_benchmark
```

## Platform-Specific Tips

### Native (Server)

- Enable LTO in release builds
- Use memory-mapped storage for large datasets
- Configure thread pool size based on CPU cores
- Consider NUMA awareness for multi-socket systems

```toml
# Cargo.toml
[profile.release]
lto = true
codegen-units = 1
```

### WASM (Browser)

- Use OPFS for persistent storage
- Limit concurrent operations (single-threaded)
- Offload heavy operations to Web Workers
- Pre-load WASM module during page load

```javascript
// Preload WASM
const wasmPromise = init();

// Later, when needed
await wasmPromise;
const db = await LatticeDB.create(config);
```

## Troubleshooting

### Slow Search

1. Check `ef` parameter (too low = poor recall, too high = slow)
2. Verify SIMD is enabled (`cargo build --features simd`)
3. Profile to identify bottleneck (distance calc vs graph traversal)

### High Memory Usage

1. Consider quantization (4-32x reduction)
2. Use memory-mapped storage
3. Reduce `m` and `m0` parameters
4. Check for memory leaks with profiler

### Poor Recall

1. Increase `ef` at search time
2. Increase `ef_construction` and rebuild index
3. Verify distance metric matches your data
4. Check for data quality issues (zero vectors, outliers)

## Next Steps

- [Benchmarks](./benchmarks.md) - Detailed performance numbers
- [HNSW Index](../vector/hnsw.md) - Algorithm deep dive
