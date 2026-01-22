# Introduction

**LatticeDB** is the world's first production-grade hybrid graph/vector database that runs entirely in the browser with zero backend required.

## What is LatticeDB?

LatticeDB combines two powerful database paradigms into a single, unified engine:

- **Vector Search Engine** - HNSW-based approximate nearest neighbor search with SIMD acceleration
- **Graph Database** - Full Cypher query language support with BFS/DFS traversal

This hybrid approach enables powerful use cases that neither paradigm can achieve alone, such as:

- Finding similar documents AND their relationships
- Semantic search with graph-based re-ranking
- Knowledge graphs with embedding-based similarity

## Key Features

### Browser-Native Execution

LatticeDB compiles to WebAssembly and runs entirely in the browser:

- **Zero server costs** - No backend infrastructure required
- **Sub-millisecond latency** - No network round-trips
- **Privacy by default** - Data never leaves the user's device
- **Offline-capable** - Works without internet connectivity

### Extreme Performance

LatticeDB is faster than industry-standard databases:

| vs Qdrant (Vector) | Speedup |
|-------------------|---------|
| Search | **1.4x faster** |
| Upsert | **177x faster** |
| Retrieve | **52x faster** |
| Scroll | **7.4x faster** |

| vs Neo4j (Graph) | Speedup |
|-----------------|---------|
| Node MATCH | **62x faster** |
| Filter queries | **5-45x faster** |
| ORDER BY | **8x faster** |

### API Compatibility

- **Qdrant-compatible REST API** - Drop-in replacement for existing vector search apps
- **Cypher query language** - Familiar syntax for graph operations
- **Rust, TypeScript, and Python** client libraries

## Use Cases

### RAG Applications

Build retrieval-augmented generation apps that run entirely in the browser:

```
User Query → Vector Search → Context Retrieval → LLM Response
```

### Knowledge Graphs

Create and query knowledge graphs with semantic similarity:

```cypher
MATCH (doc:Document)-[:REFERENCES]->(topic:Topic)
WHERE doc.embedding <-> $query_embedding < 0.5
RETURN doc, topic
```

### Personal AI Assistants

Build privacy-preserving AI assistants where all data stays local:

- Chat history with semantic search
- Personal knowledge bases
- Offline-capable reasoning

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                   LatticeDB                      │
├─────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐            │
│  │   Vector    │    │    Graph    │            │
│  │   Engine    │◄──►│   Engine    │            │
│  │  (HNSW)     │    │  (Cypher)   │            │
│  └─────────────┘    └─────────────┘            │
├─────────────────────────────────────────────────┤
│              Unified Storage Layer               │
│  (MemStorage | DiskStorage | OPFS)              │
└─────────────────────────────────────────────────┘
```

## Getting Started

Ready to dive in? Start with the [Installation](./getting-started/installation.md) guide.

## License

LatticeDB is dual-licensed under MIT and Apache 2.0. Choose whichever license works best for your project.
