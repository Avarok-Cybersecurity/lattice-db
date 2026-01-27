# LatticeDB: The Database That Makes AI Apps Actually Work

*A hybrid vector-graph database that runs everywhere—from your browser to the edge*

---

## The Problem No One Talks About

Every developer building AI applications hits the same wall: **context augmentation**.

Your LLM is powerful, but it doesn't know your user's data. So you need:
- **Vector search** for semantic similarity (finding relevant documents)
- **Graph queries** for relationships (user preferences, knowledge graphs)
- **Fast lookups** because users hate waiting

Today, this means running Qdrant or Pinecone for vectors. Neo4j or some graph store for relationships. A cache layer. A sync mechanism. Infrastructure that costs money and requires DevOps expertise.

What if one database handled all of this—and ran entirely in your user's browser?

---

## Introducing LatticeDB

LatticeDB is a hybrid vector and graph database written in Rust that compiles to WebAssembly. It's:

- **Drop-in compatible** with Qdrant's REST API
- **Standard Cypher** for graph queries (like Neo4j)
- **Runs anywhere**: Browser (WASM), native binaries, or as an HTTP server
- **Offline-first**: Works without internet once loaded

```javascript
// Use your existing Qdrant client
import { QdrantClient } from "@qdrant/js-client-rest";

const client = new QdrantClient({ url: "http://localhost:6333" });

// Same API, now it can run in the browser
await client.search("my_collection", {
  vector: [0.1, 0.2, 0.3, ...],
  limit: 10,
});
```

Same code. Different runtime. No migration required.

---

## Why We Built This

Hi, I'm Thomas Braun from [Avarok](https://avarok.net).

Three years ago, I started building AI-powered applications and ran into the same problems everyone does:

1. **Context augmentation is fragmented** — You need vectors, you need graphs, you need them fast
2. **Infrastructure is expensive** — Managed services charge per query, self-hosting requires DevOps
3. **Browser apps are second-class** — Most databases assume a server, leaving PWAs and offline apps behind

The vision was simple: **what if the database could run wherever your code runs?**

Not a compromise. Not a "lite" version. The full database engine, compiled to WebAssembly with SIMD optimizations, storing data in IndexedDB or OPFS.

We wanted to make "vibe coding" AI apps a reality—where developers can focus on the experience, not the infrastructure.

---

## Performance That Actually Matters

Let's be honest about benchmarks. Comparing an in-process database to a network-based service isn't fair. So here's both:

### Fair Comparison: HTTP vs HTTP

When both LatticeDB and Qdrant run as HTTP servers on localhost:

| Operation | LatticeDB | Qdrant | Speedup |
|-----------|-----------|--------|---------|
| Upsert | 115 µs | 287 µs | **2.5x** |
| Search (k=10) | 168 µs | 330 µs | **2.0x** |

*Same playing field. LatticeDB is still 2x faster.*

### Maximum Performance: Embedded vs HTTP

When LatticeDB runs in-process (like in a browser or embedded):

| Operation | LatticeDB | Qdrant (HTTP) | Speedup |
|-----------|-----------|---------------|---------|
| Upsert | 0.76 µs | 287 µs | **377x** |
| Search (k=10) | 84 µs | 330 µs | **4x** |
| Retrieve | 2.2 µs | 306 µs | **139x** |
| Scroll | 18 µs | 398 µs | **22x** |

*No network. No serialization. Direct memory access.*

### Graph Queries: Cypher Performance

LatticeDB also beats Neo4j on graph operations:

| Operation | LatticeDB HTTP | Neo4j Bolt | Speedup |
|-----------|----------------|------------|---------|
| `MATCH (n) RETURN n LIMIT 100` | 85 µs | 1,147 µs | **13x** |
| `MATCH (n:Person) RETURN n LIMIT 10` | 72 µs | 596 µs | **8x** |
| `WHERE n.age > 30 RETURN n` | 965 µs | 3,136 µs | **3x** |

*Both running server protocols (LatticeDB HTTP vs Neo4j Bolt)*

**Why the difference?**
- No HTTP round-trip overhead (in-memory mode)
- No JSON serialization/deserialization
- SIMD-optimized distance calculations
- No JVM overhead (native Rust)
- Cache-friendly memory layout

---

## One Database for Everything

Stop stitching together separate systems:

```
BEFORE:
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Qdrant  │ ──> │  Neo4j   │ ──> │  Redis   │
│ (vectors)│     │ (graph)  │     │ (cache)  │
└──────────┘     └──────────┘     └──────────┘
     ↓                ↓                ↓
   Sync?           Sync?           Sync?
```

```
AFTER:
┌─────────────────────────────────────────────┐
│              LatticeDB                      │
│  ┌─────────────┐    ┌─────────────────┐    │
│  │   Vectors   │    │      Graph      │    │
│  │  (HNSW)     │←──→│    (Cypher)     │    │
│  └─────────────┘    └─────────────────┘    │
└─────────────────────────────────────────────┘
         ↓
   One source of truth
```

Hybrid queries work out of the box:

```cypher
// Find semantically similar products that your friends have purchased
MATCH (user:User {id: $userId})-[:FRIENDS]->(friend)-[:PURCHASED]->(product)
WHERE vector.similarity(product.embedding, $queryVector) > 0.8
RETURN product
LIMIT 10
```

---

## Works in the Browser. Actually.

This isn't a toy implementation. LatticeDB compiles to WASM with SIMD128 optimizations and stores data persistently using:

- **IndexedDB** for compatibility
- **OPFS (Origin Private File System)** for performance

Your users can:
- Search their documents offline
- Build knowledge graphs locally
- Keep data on their device (privacy!)
- Sync to a server when they choose

Perfect for:
- AI-powered note apps
- Local-first productivity tools
- Privacy-focused applications
- PWAs that work on airplanes

---

## Easy Migration

### Coming from Qdrant?

Your client code works unchanged:

```python
from qdrant_client import QdrantClient

# Just point to LatticeDB instead
client = QdrantClient(url="http://localhost:6335")

# Everything else stays the same
client.upsert(
    collection_name="my_collection",
    points=[PointStruct(id=1, vector=[0.1, 0.2, ...], payload={"text": "hello"})]
)
```

### Coming from Neo4j?

Standard Cypher queries work:

```cypher
CREATE (n:Person {name: 'Alice', age: 30})
MATCH (p:Person) WHERE p.age > 25 RETURN p
MATCH (a)-[:KNOWS]->(b) RETURN a, b
```

---

## Try It Now

**Live Demo**: [Chat with LatticeDB in your browser](https://avarok-cybersecurity.github.io/lattice-db/chat/)

No installation. No sign-up. The entire database runs in your browser tab.

**GitHub**: [github.com/avarok-cybersecurity/lattice-db](https://github.com/avarok-cybersecurity/lattice-db)

```bash
# Install from crates.io
cargo install lattice-server

# Run the server
lattice-server
# Now accepting Qdrant-compatible requests on :6334
```

Or download pre-built binaries from [GitHub Releases](https://github.com/avarok-cybersecurity/lattice-db/releases).

For browser builds, use the npm package:
```bash
npm install lattice-db
```

---

## The Vision: A Pillar of AI-Native Apps

We believe the future of AI applications isn't more infrastructure—it's less.

When the database runs in-process:
- **Latency disappears** — Sub-millisecond queries
- **Costs disappear** — No per-query pricing
- **Complexity disappears** — No sync, no cache invalidation

LatticeDB abstracts away the entire "how do I augment my LLM's context?" question. You describe what you want. The database figures out how to get it—fast.

This is what "vibe coding" for AI apps should feel like. You focus on the experience. We handle the infrastructure.

---

## What's Next

- **Data migrators** for seamless Neo4j/Qdrant import
- **Payload encryption** for sensitive data at rest
- **Distributed mode** for multi-user sync (server-hosted)
- **More language bindings** (Python, Go, mobile)

---

## Get Involved

We're building this in the open. Every line of code is MIT licensed.

- **Star the repo** if this resonates: [GitHub](https://github.com/avarok-cybersecurity/lattice-db)
- **Try the demo** and break things: [Live Demo](https://avarok-cybersecurity.github.io/lattice-db/chat/)
- **Open an issue** with your use case
- **Contribute** — Rust developers welcome!

The AI-native app ecosystem needs better primitives. Let's build them together.

---

*Thomas Braun is the founder of Avarok, building infrastructure for the next generation of AI applications.*
