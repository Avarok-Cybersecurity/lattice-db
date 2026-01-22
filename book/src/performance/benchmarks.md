# Benchmarks

LatticeDB is benchmarked against industry-standard databases: **Qdrant** for vector operations and **Neo4j** for graph queries. This chapter presents the benchmark methodology and results.

## Summary

**LatticeDB wins ALL operations against both Qdrant and Neo4j.**

### Vector Operations: LatticeDB vs Qdrant

| Operation | LatticeDB | Qdrant | LatticeDB Advantage |
|-----------|-----------|--------|---------------------|
| **Search** | 106 µs | 150 µs | **1.4x faster** |
| **Upsert** | 0.51 µs | 90 µs | **177x faster** |
| **Retrieve** | 2.61 µs | 135 µs | **52x faster** |
| **Scroll** | 18 µs | 133 µs | **7.4x faster** |

### Graph Operations: LatticeDB vs Neo4j

| Operation | LatticeDB | Neo4j | Speedup |
|-----------|-----------|-------|---------|
| match_all | 60 µs | 3,724 µs | **62x** |
| match_by_label | 58 µs | 3,454 µs | **59x** |
| match_with_limit | 11 µs | 505 µs | **45x** |
| skip_limit | 37 µs | 543 µs | **15x** |
| order_by | 117 µs | 968 µs | **8x** |
| where_property | 107 µs | 622 µs | **6x** |
| where_comparison | 114 µs | 589 µs | **5x** |
| complex_filter | 649 µs | 998 µs | **1.5x** |

## Benchmark Setup

### Hardware

All benchmarks run on:
- **CPU**: Apple M1 Pro (10 cores)
- **RAM**: 16 GB
- **Storage**: NVMe SSD

### Dataset

**Vector Benchmarks:**
- 10,000 vectors
- 128 dimensions
- Cosine distance
- Random normalized vectors

**Graph Benchmarks:**
- 1,000 nodes with labels
- Varied properties (string, integer)
- Cypher queries of increasing complexity

### Competitors

| Database | Version | Configuration |
|----------|---------|---------------|
| Qdrant | 1.7.x | Docker, default settings |
| Neo4j | 5.x | Docker, Community Edition |
| LatticeDB | 0.1 | Native binary, SIMD enabled |

## Vector Benchmark Details

### Search Benchmark

**Query**: Find 10 nearest neighbors from 10,000 vectors

```rust
// LatticeDB
let results = engine.search(&SearchQuery::new(query_vec).with_limit(10))?;

// Qdrant equivalent
POST /collections/{name}/points/search
{"vector": [...], "limit": 10}
```

**Results:**
- LatticeDB: **106 µs** (p50), 142 µs (p99)
- Qdrant: 150 µs (p50), 280 µs (p99)

**Why LatticeDB is faster:**
- SIMD-accelerated distance calculations (4x unrolling on NEON)
- Dense vector storage (cache-friendly)
- Thread-local scratch space (no allocation per search)
- Shortcut-enabled HNSW traversal

### Upsert Benchmark

**Operation**: Insert single point with 128-dim vector

```rust
// LatticeDB
engine.upsert(point)?;

// Qdrant equivalent
PUT /collections/{name}/points
{"points": [{"id": 1, "vector": [...]}]}
```

**Results:**
- LatticeDB: **0.51 µs**
- Qdrant: 90 µs

**177x faster** due to:
- No network overhead (in-process)
- Optimized HNSW insertion with pre-computed distances
- Memory-mapped graph storage

### Retrieve Benchmark

**Operation**: Get point by ID

```rust
// LatticeDB
let point = engine.get_point(id)?;

// Qdrant equivalent
GET /collections/{name}/points/{id}
```

**Results:**
- LatticeDB: **2.61 µs**
- Qdrant: 135 µs

**52x faster** due to:
- Direct HashMap lookup
- No HTTP serialization/deserialization

### Scroll Benchmark

**Operation**: Paginate through all points (100 per page)

```rust
// LatticeDB
let result = engine.scroll(&ScrollQuery::new().with_limit(100))?;

// Qdrant equivalent
POST /collections/{name}/points/scroll
{"limit": 100}
```

**Results:**
- LatticeDB: **18 µs**
- Qdrant: 133 µs

**7.4x faster** due to:
- Sequential memory access
- No HTTP overhead

## Graph Benchmark Details

### match_all

```cypher
MATCH (n) RETURN n LIMIT 100
```

- LatticeDB: **60 µs**
- Neo4j: 3,724 µs
- **62x faster**

### match_by_label

```cypher
MATCH (n:Person) RETURN n LIMIT 100
```

- LatticeDB: **58 µs**
- Neo4j: 3,454 µs
- **59x faster**

### match_with_limit

```cypher
MATCH (n:Person) RETURN n LIMIT 10
```

- LatticeDB: **11 µs**
- Neo4j: 505 µs
- **45x faster**

### skip_limit

```cypher
MATCH (n:Person) RETURN n SKIP 50 LIMIT 20
```

- LatticeDB: **37 µs**
- Neo4j: 543 µs
- **15x faster**

### order_by

```cypher
MATCH (n:Person) RETURN n.name ORDER BY n.name LIMIT 50
```

- LatticeDB: **117 µs**
- Neo4j: 968 µs
- **8x faster**

### where_property

```cypher
MATCH (n:Person) WHERE n.name = 'Alice' RETURN n
```

- LatticeDB: **107 µs**
- Neo4j: 622 µs
- **6x faster**

### where_comparison

```cypher
MATCH (n:Person) WHERE n.age > 25 RETURN n LIMIT 50
```

- LatticeDB: **114 µs**
- Neo4j: 589 µs
- **5x faster**

### complex_filter

```cypher
MATCH (n:Person)
WHERE n.age > 20 AND n.age < 50 AND n.active = true
RETURN n.name, n.age
LIMIT 20
```

- LatticeDB: **649 µs**
- Neo4j: 998 µs
- **1.5x faster**

## Why LatticeDB is Faster

### vs Qdrant

1. **No network overhead**: LatticeDB runs in-process or in-browser
2. **SIMD optimizations**: AVX2/NEON distance calculations
3. **Memory efficiency**: Dense vector storage, thread-local caches
4. **Optimized HNSW**: Shortcut search, prefetching

### vs Neo4j

1. **Lightweight runtime**: No JVM overhead
2. **Efficient data structures**: Rust-native HashMap, Vec
3. **Query compilation**: Direct execution vs interpreted Cypher
4. **Cache-friendly layout**: Sequential memory access

## Running Benchmarks

### Prerequisites

```bash
# Install criterion
cargo install cargo-criterion

# For comparison benchmarks, start competitors
docker run -p 6333:6333 qdrant/qdrant
docker run -p 7474:7474 -p 7687:7687 neo4j
```

### Quick Benchmark

```bash
cargo run -p lattice-bench --release --example quick_vector_bench
```

### Full Criterion Benchmarks

```bash
# Vector operations
cargo bench -p lattice-bench --bench vector_ops

# Graph operations
cargo bench -p lattice-bench --bench cypher_comparison
```

### View Reports

```bash
open target/criterion/report/index.html
```

## Reproducing Results

All benchmarks are reproducible:

```bash
git clone https://github.com/Avarok-Cybersecurity/lattice-db
cd lattice-db

# Run all benchmarks
cargo bench -p lattice-bench

# Results saved to:
# - target/criterion/*/report/index.html (HTML)
# - target/criterion/*/new/estimates.json (JSON)
```

## Next Steps

- [Tuning Guide](./tuning.md) - Optimize for your use case
- [SIMD Optimization](../vector/simd.md) - Hardware acceleration details
