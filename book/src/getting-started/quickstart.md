# Quick Start

This guide will help you create your first LatticeDB collection, add vectors and graph data, and run queries.

## Creating a Collection

### Rust

```rust
use lattice_core::{
    CollectionConfig, CollectionEngine, Distance, HnswConfig, VectorConfig,
};

// Create collection configuration
let config = CollectionConfig::new(
    "my_collection",
    VectorConfig::new(128, Distance::Cosine),  // 128-dimensional vectors
    HnswConfig {
        m: 16,                // Connections per node
        m0: 32,               // Layer 0 connections
        ml: 0.36,             // Level multiplier
        ef: 100,              // Search queue size
        ef_construction: 200, // Build quality
    },
);

// Create the collection engine
let mut engine = CollectionEngine::new(config)?;
```

### REST API

```bash
curl -X PUT http://localhost:6333/collections/my_collection \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 128,
      "distance": "Cosine"
    },
    "hnsw_config": {
      "m": 16,
      "ef_construct": 200
    }
  }'
```

## Adding Points (Vectors + Metadata)

### Rust

```rust
use lattice_core::Point;

// Create a point with vector and metadata
let point = Point::new(
    1,  // Point ID
    vec![0.1, 0.2, 0.3, /* ... 128 dimensions */],
)
.with_payload("title", "Introduction to LatticeDB")
.with_payload("category", "documentation");

// Upsert the point
engine.upsert(point)?;
```

### REST API

```bash
curl -X PUT http://localhost:6333/collections/my_collection/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {
        "id": 1,
        "vector": [0.1, 0.2, 0.3],
        "payload": {
          "title": "Introduction to LatticeDB",
          "category": "documentation"
        }
      }
    ]
  }'
```

## Vector Search

### Rust

```rust
use lattice_core::SearchQuery;

// Create a search query
let query = SearchQuery::new(query_vector)
    .with_limit(10)       // Return top 10 results
    .with_ef(100);        // Search quality

// Execute search
let results = engine.search(&query)?;

for result in results {
    println!("ID: {}, Score: {}", result.id, result.score);
}
```

### REST API

```bash
curl -X POST http://localhost:6333/collections/my_collection/points/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3],
    "limit": 10,
    "with_payload": true
  }'
```

## Adding Graph Edges

### Rust

```rust
use lattice_core::Edge;

// Add an edge between two points
engine.add_edge(
    1,  // Source point ID
    Edge::new(2, 0.9, "REFERENCES"),  // Target, weight, relation
)?;
```

### REST API

```bash
curl -X POST http://localhost:6333/collections/my_collection/graph/edges \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": 1,
    "target_id": 2,
    "weight": 0.9,
    "relation": "REFERENCES"
  }'
```

## Cypher Queries

### Rust

```rust
use lattice_core::{CypherHandler, DefaultCypherHandler};
use std::collections::HashMap;

let handler = DefaultCypherHandler::new();

// Execute a Cypher query
let result = handler.query(
    "MATCH (n:Document) WHERE n.category = 'documentation' RETURN n.title",
    &mut engine,
    HashMap::new(),
)?;

for row in result.rows {
    println!("{:?}", row);
}
```

### REST API

```bash
curl -X POST http://localhost:6333/collections/my_collection/graph/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "MATCH (n:Document) WHERE n.category = $cat RETURN n.title",
    "params": {
      "cat": "documentation"
    }
  }'
```

## Hybrid Query Example

Combine vector search with graph traversal:

```rust
// 1. Find similar documents via vector search
let similar = engine.search(&SearchQuery::new(query_vector).with_limit(5))?;

// 2. For each result, find related documents via graph
for result in similar {
    let cypher = format!(
        "MATCH (n)-[:REFERENCES]->(related) WHERE id(n) = {} RETURN related",
        result.id
    );
    let related = handler.query(&cypher, &mut engine, HashMap::new())?;

    println!("Document {} references: {:?}", result.id, related);
}
```

## Complete Example

Here's a complete example that demonstrates the hybrid capabilities:

```rust
use lattice_core::*;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create collection
    let config = CollectionConfig::new(
        "knowledge_base",
        VectorConfig::new(128, Distance::Cosine),
        HnswConfig::default_for_dim(128),
    );
    let mut engine = CollectionEngine::new(config)?;
    let handler = DefaultCypherHandler::new();

    // Add documents with embeddings
    for (id, title, embedding) in documents {
        let point = Point::new(id, embedding)
            .with_payload("title", title)
            .with_label("Document");
        engine.upsert(point)?;
    }

    // Add relationships via Cypher
    handler.query(
        "MATCH (a:Document), (b:Document)
         WHERE a.title = 'Intro' AND b.title = 'Advanced'
         CREATE (a)-[:NEXT]->(b)",
        &mut engine,
        HashMap::new(),
    )?;

    // Hybrid query: similar docs + their neighbors
    let results = engine.search(&SearchQuery::new(query_embedding).with_limit(3))?;

    for result in results {
        let neighbors = handler.query(
            &format!("MATCH (n)-[r]->(m) WHERE id(n) = {} RETURN m, type(r)", result.id),
            &mut engine,
            HashMap::new(),
        )?;
        println!("Doc {}: {:?}", result.id, neighbors);
    }

    Ok(())
}
```

## Next Steps

- [WASM Browser Setup](./wasm.md) - Run LatticeDB in the browser
- [HNSW Index](../vector/hnsw.md) - Understanding vector search
- [Cypher Query Language](../graph/cypher.md) - Full Cypher reference
