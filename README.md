<div align="center">

# LatticeDB

### World's First Production-Grade Hybrid Graph/Vector Database

**Runs in your browser. Zero backend required.**

*Democratizing AI databases for frontend developers*

[![CI](https://github.com/Avarok-Cybersecurity/lattice-db/actions/workflows/ci.yml/badge.svg)](https://github.com/Avarok-Cybersecurity/lattice-db/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-book-blue.svg)](https://Avarok-Cybersecurity.github.io/lattice-db/)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![WASM](https://img.shields.io/badge/wasm-SIMD-blueviolet.svg)](https://webassembly.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![Qdrant Compatible](https://img.shields.io/badge/qdrant-API%20compatible-green.svg)](https://qdrant.tech)
[![Cypher](https://img.shields.io/badge/cypher-query%20language-blue.svg)](https://neo4j.com/docs/cypher-manual/)
[![Memory](https://img.shields.io/badge/memory-2.4MB-brightgreen.svg)](#-ultra-low-footprint)

---

**Browser-Native** | **Graph + Vector Hybrid** | **No Server Costs** | **2.4 MB Memory**

</div>

---

## 📑 Table of Contents

| Section | Description |
|---------|-------------|
| [🎯 Why LatticeDB?](#-why-latticedb) | The problem we solve |
| [⚡ Performance](#-performance) | Benchmark results vs Qdrant & Neo4j |
| [🪶 Ultra-Low Footprint](#-ultra-low-footprint) | 2.4 MB memory, ~500 KB WASM |
| [✨ Features](#-features) | Hybrid graph/vector, platform support |
| [💡 Use Cases](#-use-cases) | RAG, knowledge graphs, AI assistants |
| [🚀 Quick Start](#-quick-start) | Installation & first steps |
| [🏗️ Architecture](#️-architecture) | SBIO pattern & crate structure |
| [⚙️ Optimizations](#️-optimizations) | 8 state-of-the-art techniques |
| [📚 API Reference](#-api-reference) | REST endpoints |
| [🗺️ Roadmap](#️-roadmap) | What's next |
| [🔬 Research](#-research) | Papers we build on |
| [🤝 Contributing](#-contributing) | How to help |
| [📄 License](#-license) | MIT License |

---

## 🎯 Why LatticeDB?

**LatticeDB is the only database that lets you run production-grade vector search AND graph queries entirely in the browser.**

| Problem | Traditional Solution | LatticeDB Solution |
|---------|---------------------|-------------------|
| RAG for web apps | Pay for hosted vector DB | **Run RAG in the frontend** |
| Knowledge graphs | Host Neo4j/Qdrant server | **Zero backend required** |
| Single-user apps | Server for each user | **Data stays on client** |
| Network latency | Round-trips to backend | **Sub-millisecond local access** |

### Who Is This For?

- 🤖 **LLM app developers** - Build RAG-powered apps without server costs
- 🌐 **Frontend developers** - Add semantic search to any web app
- 🚀 **Startups** - Ship faster without infrastructure overhead
- 🔒 **Privacy-conscious apps** - Data never leaves the user's browser

---

## ⚡ Performance

**Optimized for small to medium datasets** - the sweet spot for browser-based applications.

### Target Use Cases

LatticeDB shines for datasets typical in frontend applications:
- **Vectors**: 1K - 50K points (RAG contexts, document collections, user embeddings)
- **Graphs**: 1K - 10K nodes (knowledge graphs, relationship data, user networks)

At these scales, LatticeDB dramatically outperforms server-based solutions by eliminating network latency and running entirely in-process.

### Vector Operations: LatticeDB vs Qdrant

**Benchmark**: 1,000 vectors, 128 dimensions, cosine distance

| Operation | LatticeDB In-Memory¹ | LatticeDB HTTP² | Qdrant HTTP |
|-----------|---------------------|-----------------|-------------|
| **Search** | **77 µs** | **166 µs** | 381 µs |
| **Upsert** | **0.80 µs** | **88 µs** | 306 µs |
| **Retrieve** | **1.5 µs** | **90 µs** | 275 µs |
| **Scroll** | **20 µs** | **130 µs** | 394 µs |

¹ In-memory applies to browser/WASM deployments (no network overhead)
² HTTP server uses simd-json, Hyper with pipelining, TCP_NODELAY

> **LatticeDB wins in ALL deployment modes**: In-memory LatticeDB is **50-100x faster** than HTTP. Even LatticeDB HTTP is **2-3x faster** than Qdrant HTTP.

### Graph Operations: LatticeDB vs Neo4j

**Benchmark**: 1,000 nodes with labels and properties, Cypher queries

| Operation | LatticeDB | Neo4j | Speedup |
|-----------|-----------|-------|---------|
| `MATCH (n) RETURN n LIMIT 100` | **63 µs** | 3,543 µs | **56x** |
| `MATCH (n:Person) RETURN n LIMIT 100` | **57 µs** | 3,689 µs | **65x** |
| `MATCH (n:Person) RETURN n LIMIT 10` | **12 µs** | 610 µs | **51x** |
| `ORDER BY n.name LIMIT 50` | **116 µs** | 953 µs | **8x** |
| `WHERE n.age > 30 RETURN n` | **555 µs** | 2,538 µs | **5x** |

> **LatticeDB wins all graph operations** at 1K nodes. No JVM overhead, native Rust data structures, and direct query execution.

### Scaling Considerations

| Dataset Size | LatticeDB Advantage | Recommendation |
|--------------|---------------------|----------------|
| < 10K | **Excellent** (10-100x faster) | Ideal for browser/embedded use |
| 10K - 50K | **Good** (2-10x faster) | Still great for single-user apps |
| > 50K | **Diminishing** | Consider dedicated vector DB for large datasets |

For datasets exceeding 50K elements, server-based solutions like Qdrant or Neo4j may offer better performance due to their optimized indexing for large-scale workloads.

### Performance Roadmap

LatticeDB HTTP server optimization is ongoing:

- [x] SIMD-accelerated JSON parsing (simd-json)
- [x] Zero-copy request/response handling
- [x] Connection pipelining (HTTP/1.1)
- [ ] Response streaming for large results
- [ ] Binary protocol support (gRPC/protobuf)

Our primary focus remains **in-memory performance** for browser/WASM deployments where LatticeDB excels.

📖 [Full benchmark details](https://Avarok-Cybersecurity.github.io/lattice-db/book/performance/benchmarks.html)

---

## 🪶 Ultra-Low Footprint

LatticeDB is engineered for minimal resource consumption:

| Platform | Metric | Size |
|----------|--------|------|
| **Native** | Runtime Memory (RSS) | **2.4 MB** |
| **Browser (WASM)** | Bundle Size (gzip) | **~500 KB** |
| **Browser (WASM)** | Runtime Memory | **~2-3 MB** |

**Why this matters:**
- 💾 Runs on low-end devices and mobile browsers
- ⚡ Instant startup - no JVM warmup or heavy initialization
- 📱 Ideal for PWAs and offline-first applications
- 🌐 Fast download and parse time in browsers

Compare this to typical database footprints:
- PostgreSQL: ~20-50 MB baseline
- MongoDB: ~100-200 MB baseline
- Neo4j: ~500+ MB (JVM-based)
- Qdrant: ~50-100 MB baseline

LatticeDB delivers **full vector + graph database capabilities in under 3 MB**.

---

## 🔗 Why Hybrid?

**One library for everything your frontend needs.**

Modern AI-powered applications require multiple database capabilities:

| Capability | Traditional Approach | LatticeDB Approach |
|------------|---------------------|-------------------|
| **Semantic Search** | Vector DB (Pinecone, Qdrant) | ✅ Built-in HNSW |
| **Knowledge Graphs** | Graph DB (Neo4j, Dgraph) | ✅ Built-in Cypher |
| **Document Storage** | Key-Value DB (Redis, DynamoDB) | ✅ Built-in Payload |
| **Relationship Queries** | SQL or Graph DB | ✅ Built-in Traversal |

### Why Not Separate Databases?

- 🔌 **Single Dependency** - One import, not three separate databases
- 🎯 **Unified Queries** - Vector similarity + graph traversal in one query
- 📦 **Smaller Bundle** - ~500 KB WASM vs multiple large dependencies
- 🧠 **Simpler Mental Model** - Points have vectors, payloads, AND relationships
- ⚡ **Zero Network Hops** - No coordination between services
- 💰 **No Server Costs** - Everything runs client-side

### The Hybrid Advantage

```javascript
// Find semantically similar documents AND their related concepts
const similar = await db.search({ vector: queryEmbedding, limit: 10 });
const related = await db.query(`
  MATCH (doc:Document)-[:REFERENCES]->(concept:Concept)
  WHERE doc.id IN $docIds
  RETURN DISTINCT concept.name
`, { docIds: similar.map(r => r.id) });
```

With separate databases, this requires:
1. Query vector DB for similar documents
2. Query graph DB for relationships
3. Coordinate results between two systems
4. Handle different data models and APIs

With LatticeDB, it's one library with unified data.

---

## ✨ Features

### Hybrid Graph + Vector

The only embedded database that combines:

```
┌─────────────────────────────────────────────────────────────┐
│                        LatticeDB                              │
│  ┌───────────────────────┐    ┌───────────────────────┐     │
│  │    Vector Engine      │    │     Graph Engine      │     │
│  │  ─────────────────    │    │  ─────────────────    │     │
│  │  • HNSW Index         │    │  • BFS/DFS Traversal  │     │
│  │  • SIMD Distance      │◄──►│  • Cypher Queries     │     │
│  │  • Product Quant.     │    │  • Weighted Edges     │     │
│  │  • Scalar Quant.      │    │  • Relation Types     │     │
│  └───────────────────────┘    └───────────────────────┘     │
│                              ▲                               │
│                              │                               │
│                    Hybrid Queries                            │
│          "Find similar vectors AND their neighbors"          │
└─────────────────────────────────────────────────────────────┘
```

### Platform Support

| Platform | Status | SIMD Support |
|----------|--------|--------------|
| 🌐 **Browser (WASM)** | Production | SIMD128 |
| 🐧 **Linux x86_64** | Production | AVX2/AVX-512 |
| 🍎 **macOS Apple Silicon** | Production | ARM NEON |
| 🪟 **Windows x86_64** | Production | AVX2 |

### API Compatibility

- 🔌 **Qdrant REST API** - Drop-in replacement, use existing SDKs
- 📊 **Cypher Query Language** - Neo4j-compatible graph queries
- 📴 **Service Worker** - Offline-first browser operation *(coming soon)*

---

## 💡 Use Cases

### Frontend RAG (No Backend)

Build LLM-powered apps that run entirely in the browser:

```javascript
import { LatticeDB } from 'lattice-db';

// Initialize in browser
const db = await LatticeDB.init();
await db.createCollection('knowledge', { dimension: 384 });

// User uploads documents → embed → store locally
for (const doc of userDocuments) {
  const embedding = await embed(doc.text);  // Local or API
  await db.upsert('knowledge', [{
    id: doc.id,
    vector: embedding,
    payload: { text: doc.text, source: doc.source }
  }]);
}

// RAG query - zero network latency
const context = await db.search('knowledge', queryEmbedding, 5);
const answer = await llm.generate(query, context);
```

**Benefits:**
- 💰 No server costs for vector storage
- 💾 Data persists in IndexedDB/OPFS
- 📴 Works offline
- ⚡ Sub-millisecond search latency

### Knowledge Graphs with Semantic Search

Combine graph relationships with vector similarity:

```cypher
// Find similar concepts AND their related entities
MATCH (concept:Concept)-[:RELATED_TO]->(related)
WHERE vector_similarity(concept.embedding, $query) > 0.8
RETURN concept, related
ORDER BY vector_similarity(concept.embedding, $query) DESC
LIMIT 10
```

### Personal AI Assistants

Build apps where user data stays on their device:

```javascript
// All data stored locally in browser
const memories = await db.search('memories', currentContext, 10);
const response = await assistant.respond(userMessage, memories);

// Add new memory
await db.upsert('memories', [{
  id: Date.now(),
  vector: await embed(response),
  payload: { conversation: userMessage, response }
}]);
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Avarok-Cybersecurity/lattice-db.git
cd lattice-db

# Build release binary
cargo build --release -p lattice-server

# Run the server (Qdrant-compatible API)
cargo run --release -p lattice-server
```

### Using with Python (Qdrant Client)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Connect to LatticeDB (Qdrant-compatible)
client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="my_vectors",
    vectors_config=VectorParams(size=128, distance=Distance.COSINE),
)

# Insert vectors
client.upsert(
    collection_name="my_vectors",
    points=[
        PointStruct(id=1, vector=[0.1] * 128, payload={"category": "A"}),
        PointStruct(id=2, vector=[0.2] * 128, payload={"category": "B"}),
    ]
)

# Search
results = client.query_points(
    collection_name="my_vectors",
    query=[0.15] * 128,
    limit=10,
)
```

### WASM (Browser)

```javascript
import { LatticeDB } from 'lattice-db';

const db = await LatticeDB.init();
await db.createCollection('vectors', { dimension: 128 });
await db.upsert('vectors', [{ id: 1, vector: new Float32Array(128) }]);
const results = await db.search('vectors', queryVector, 10);
```

### Cypher Query Language

```cypher
// Create nodes with vectors
CREATE (p:Person {name: 'Alice', embedding: [0.1, 0.2, ...]})
CREATE (p:Person {name: 'Bob', embedding: [0.3, 0.4, ...]})

// Create relationships
MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
CREATE (a)-[:KNOWS {since: 2020}]->(b)

// Query with filters
MATCH (p:Person)-[:KNOWS]->(friend)
WHERE p.age > 25
RETURN p.name, friend.name
ORDER BY p.name
LIMIT 10

// Hybrid: vector similarity + graph traversal
MATCH (p:Person)-[:KNOWS*1..2]->(fof)
WHERE vector_similarity(p.embedding, $query) > 0.8
RETURN DISTINCT fof.name
```

---

## 🏗️ Architecture

```
lattice-db/
├── crates/
│   ├── lattice-core/          # Core engine (HNSW, Cypher, SIMD)
│   │   ├── engine/            # Collection management
│   │   ├── index/             # HNSW, ScaNN, distance functions
│   │   ├── cypher/            # Cypher parser & executor
│   │   └── types/             # Point, Query, Config types
│   │
│   ├── lattice-server/        # HTTP server & API
│   │   ├── handlers/          # REST endpoint handlers
│   │   └── router.rs          # Qdrant-compatible routing
│   │
│   └── lattice-wasm/          # Browser WASM bindings
│       └── lib.rs             # JavaScript API
```

### SBIO Architecture

**Separation of Business Logic and I/O** - Core engine never touches filesystem or network.

```
┌─────────────────────────────────────────────────────────────┐
│                      Transport Layer                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Axum HTTP │    │   Service   │    │    WASM     │     │
│  │   Server    │    │   Worker    │    │   Browser   │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    LatticeDB Core Engine                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  HNSW    │  │  Cypher  │  │  Graph   │  │  Filter  │    │
│  │  Index   │  │  Parser  │  │  Ops     │  │  Engine  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                      Storage Layer                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Memory    │    │    MMap     │    │  IndexedDB  │     │
│  │   HashMap   │    │   Files     │    │    OPFS     │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Optimizations

LatticeDB implements **8 state-of-the-art optimizations**:

| Optimization | Technique | Impact |
|--------------|-----------|--------|
| ⚡ **SIMD Distance** | AVX2/NEON/SIMD128 | 4-8x faster cosine |
| 🔗 **HNSW Shortcuts** | VLDB 2025 paper | Skip redundant layers |
| 🧵 **Thread-Local Scratch** | Pre-allocated pools | 10-20% faster search |
| 📦 **Product Quantization** | ScaNN-style | 64x compression |
| 💾 **Memory Mapping** | Zero-copy access | Large dataset support |
| 🔄 **Async Indexing** | Background HNSW updates | Non-blocking upserts |
| 📊 **Batch Search** | Parallel with rayon | High throughput |
| 🗜️ **Scalar Quantization** | int8 vectors | 4x memory reduction |

---

## 📚 API Reference

### Collections

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections` | GET | List all collections |
| `/collections/{name}` | PUT | Create collection |
| `/collections/{name}` | GET | Get collection info |
| `/collections/{name}` | DELETE | Delete collection |

### Points

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections/{name}/points` | PUT | Upsert points |
| `/collections/{name}/points` | POST | Get points by IDs |
| `/collections/{name}/points/delete` | POST | Delete points |
| `/collections/{name}/points/scroll` | POST | Paginate points |

### Search

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections/{name}/points/search` | POST | Vector search |
| `/collections/{name}/points/query` | POST | Query (Qdrant v1.16+) |
| `/collections/{name}/points/search/batch` | POST | Batch search |

### Import/Export (LatticeDB Extension)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections/{name}/export` | GET | Export collection as binary |
| `/collections/{name}/import?mode={mode}` | POST | Import collection (`create`/`replace`/`merge`) |

### Graph Extensions (LatticeDB)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections/{name}/graph/edges` | POST | Add edge between points |
| `/collections/{name}/graph/traverse` | POST | Traverse graph from point |
| `/collections/{name}/graph/query` | POST | Execute Cypher query |

📖 [Full API documentation](https://Avarok-Cybersecurity.github.io/lattice-db/book/api/rest.html)

---

## 🗺️ Roadmap

### ✅ Implemented

- [x] HNSW index with shortcuts (VLDB 2025)
- [x] SIMD distance (AVX2, NEON, WASM SIMD128)
- [x] Cypher query language
- [x] Product Quantization (ScaNN-style)
- [x] Qdrant API compatibility
- [x] WASM browser support

### 🔨 In Progress

- [ ] npm package for easy browser integration
- [ ] IndexedDB/OPFS persistence for WASM
- [ ] Hybrid vector+graph queries in Cypher

### 📋 Planned

| Feature | Impact |
|---------|--------|
| **FP16 Quantization** | 2x memory reduction |
| **Binary Vectors** | 48% faster Hamming |
| **IVF-PQ Hybrid** | Billion-scale support |
| **DiskANN/Vamana** | SSD-based indexing |

---

## 🔬 Research

LatticeDB incorporates techniques from cutting-edge research:

| Paper/Project | Contribution |
|---------------|--------------|
| [HNSW](https://arxiv.org/abs/1603.09320) | Hierarchical graph index |
| [ScaNN](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/) | Anisotropic quantization |
| [VLDB 2025 Shortcuts](https://www.vldb.org/pvldb/vol18/p3518-chen.pdf) | Layer skip optimization |
| [SimSIMD](https://github.com/ashvardanian/SimSIMD) | SIMD best practices |

---

## 🤝 Contributing

We welcome contributions!

```bash
# Run tests
cargo test --all

# Run WASM tests (requires Chrome)
wasm-pack test --headless --chrome crates/lattice-core

# Run benchmarks
cargo bench -p lattice-bench
```

📖 [Contributing guide](https://Avarok-Cybersecurity.github.io/lattice-db/book/contributing/setup.html)

---

## 📄 License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

---

<div align="center">

**Built with 🦀 Rust for the AI-native future**

*The database that runs where your users are*

[📖 Documentation](https://Avarok-Cybersecurity.github.io/lattice-db/) | [📚 API Reference](https://Avarok-Cybersecurity.github.io/lattice-db/api/) | [💬 Discord](https://discord.gg/lattice-db)

</div>
