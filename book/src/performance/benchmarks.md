# Benchmarks

LatticeDB is benchmarked against industry-standard databases: **Qdrant** for vector operations and **Neo4j** for graph queries. This chapter presents the benchmark methodology and results.

## Summary

**LatticeDB In-Memory wins ALL operations against both Qdrant and Neo4j.**

### Vector Operations (1,000 points, 128 dimensions)

| Operation | LatticeDB In-Memory¹ | LatticeDB HTTP² | Qdrant HTTP |
|-----------|---------------------|-----------------|-------------|
| **Search** | **77 µs** | **166 µs** | 381 µs |
| **Upsert** | **0.80 µs** | **88 µs** | 306 µs |
| **Retrieve** | **1.5 µs** | **90 µs** | 275 µs |
| **Scroll** | **20 µs** | **130 µs** | 394 µs |

¹ In-memory performance applies to browser/WASM deployments (no network overhead)

² HTTP server uses simd-json, Hyper with pipelining, TCP_NODELAY

### Graph Operations: LatticeDB In-Memory³ vs Neo4j Bolt

| Operation | LatticeDB In-Memory | Neo4j Bolt | Speedup |
|-----------|---------------------|------------|---------|
| match_all | **63 µs** | 3,543 µs | **56x** |
| match_by_label | **57 µs** | 3,689 µs | **65x** |
| match_with_limit | **12 µs** | 610 µs | **51x** |
| order_by | **116 µs** | 953 µs | **8x** |
| where_property | **555 µs** | 2,538 µs | **5x** |

³ In-memory graph queries have no HTTP overhead. For server deployments,
  use `http_graph_profiler` to benchmark LatticeDB HTTP vs Neo4j Bolt.

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

- LatticeDB: **63 µs**
- Neo4j: 3,543 µs
- **56x faster**

### match_by_label

```cypher
MATCH (n:Person) RETURN n LIMIT 100
```

- LatticeDB: **57 µs**
- Neo4j: 3,689 µs
- **65x faster**

### match_with_limit

```cypher
MATCH (n:Person) RETURN n LIMIT 10
```

- LatticeDB: **12 µs**
- Neo4j: 610 µs
- **51x faster**

### order_by

```cypher
MATCH (n:Person) RETURN n.name ORDER BY n.name LIMIT 50
```

- LatticeDB: **116 µs**
- Neo4j: 953 µs
- **8x faster**

### where_property

```cypher
MATCH (n:Person) WHERE n.age > 30 RETURN n
```

- LatticeDB: **555 µs**
- Neo4j: 2,538 µs
- **5x faster**

## Why LatticeDB is Faster

### vs Qdrant

**In-Memory (Browser/WASM):**
1. **No network overhead**: LatticeDB runs in-process or in-browser
2. **SIMD optimizations**: AVX2/NEON distance calculations
3. **Memory efficiency**: Dense vector storage, thread-local caches
4. **Optimized HNSW**: Shortcut search, prefetching

**HTTP Mode (Server):**
1. **Raw Hyper**: Direct HTTP/1.1 with minimal abstraction
2. **simd-json**: SIMD-accelerated JSON parsing/serialization
3. **TCP_NODELAY**: Lower latency with Nagle algorithm disabled
4. **HTTP pipelining**: Concurrent request processing
5. **Zero-copy paths**: Static string allocations, fast response building

### vs Neo4j

**In-Memory Mode (embedded/browser):**
1. **Lightweight runtime**: No JVM overhead
2. **Efficient data structures**: Rust-native HashMap, Vec
3. **Query compilation**: Direct execution vs interpreted Cypher
4. **Cache-friendly layout**: Sequential memory access

**HTTP Mode (server deployments):**
- LatticeDB HTTP uses the same optimized Cypher engine
- Neo4j Bolt is a binary protocol (more efficient than HTTP)
- Use `http_graph_profiler` to compare server deployment performance

## Running Benchmarks

### Prerequisites

```bash
# Install criterion
cargo install cargo-criterion

# For comparison benchmarks, start competitors
docker run -p 6333:6333 qdrant/qdrant
docker run -p 7474:7474 -p 7687:7687 neo4j
```

### Quick Benchmarks

```bash
# Vector operations (HTTP)
cargo run -p lattice-bench --release --example http_profiler

# Graph operations (in-memory)
cargo run -p lattice-bench --release --example graph_profiler

# Graph operations (HTTP vs Bolt)
cargo run -p lattice-bench --release --example http_graph_profiler
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
