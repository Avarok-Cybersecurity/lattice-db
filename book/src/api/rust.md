# Rust API

This chapter documents the Rust API for LatticeDB, covering both the core library and server usage.

## Installation

```toml
# Core library only
[dependencies]
lattice-core = "0.1"

# With storage implementations
lattice-storage = { version = "0.1", features = ["native"] }

# Full server
lattice-server = { version = "0.1", features = ["native"] }
```

## Core Types

### Point

```rust
use lattice_core::{Point, Edge};

// Create a point with vector
let point = Point::new_vector(1, vec![0.1, 0.2, 0.3]);

// Add payload (builder pattern)
let point = Point::new_vector(1, vec![0.1, 0.2, 0.3])
    .with_payload("title", "My Document")
    .with_payload("score", 0.95)
    .with_payload("tags", vec!["rust", "database"])
    .with_label("Document");

// Access fields
let id: PointId = point.id;
let vector: &[f32] = &point.vector;
let title: Option<&CypherValue> = point.payload.get("title");
```

### Edge

```rust
use lattice_core::Edge;

// Create an edge
let edge = Edge::new(
    target_id: 2,
    weight: 0.9,
    relation: "REFERENCES",
);

// Access fields
let target: PointId = edge.target;
let weight: f32 = edge.weight;
let relation: &str = &edge.relation;
```

### Vector

```rust
use lattice_core::Vector;

// Vector is an alias for Vec<f32>
let vector: Vector = vec![0.1, 0.2, 0.3, 0.4];

// Create from iterator
let vector: Vector = (0..128)
    .map(|i| (i as f32) / 128.0)
    .collect();
```

## Configuration

### CollectionConfig

```rust
use lattice_core::{CollectionConfig, VectorConfig, HnswConfig, Distance};

let config = CollectionConfig::new(
    "my_collection",
    VectorConfig::new(128, Distance::Cosine),
    HnswConfig {
        m: 16,
        m0: 32,
        ml: 0.36,
        ef: 100,
        ef_construction: 200,
    },
);
```

### HnswConfig Helpers

```rust
use lattice_core::HnswConfig;

// Default config
let config = HnswConfig::default();

// Recommended ml for given m
let ml = HnswConfig::recommended_ml(16);  // ~0.36

// Config optimized for dimension
let config = HnswConfig::default_for_dim(128);
```

## Collection Engine

### Creating an Engine

```rust
use lattice_core::CollectionEngine;
use lattice_storage::MemStorage;

// With in-memory storage
let storage = MemStorage::new();
let mut engine = CollectionEngine::new(config, storage)?;
```

### Upsert Operations

```rust
// Single point
engine.upsert(point)?;

// Batch upsert
let points = vec![point1, point2, point3];
for point in points {
    engine.upsert(point)?;
}

// Upsert returns info
let result = engine.upsert(point)?;
println!("Upserted point {}, was_insert: {}", result.id, result.was_insert);
```

### Search Operations

```rust
use lattice_core::SearchQuery;

// Basic search
let query = SearchQuery::new(query_vector)
    .with_limit(10);

let results = engine.search(&query)?;

for result in results {
    println!("ID: {}, Score: {}", result.id, result.score);
}

// Search with ef parameter
let query = SearchQuery::new(query_vector)
    .with_limit(10)
    .with_ef(200);  // Higher ef = better recall

// Search returning payloads
let results = engine.search_with_payload(&query)?;

for result in results {
    let payload = result.payload.as_ref();
    println!("Title: {:?}", payload.and_then(|p| p.get("title")));
}
```

### Retrieve Operations

```rust
// Get single point
let point = engine.get_point(42)?;

// Get multiple points
let points = engine.get_points(&[1, 2, 3])?;

// Check existence
if engine.has_point(42)? {
    println!("Point exists");
}
```

### Delete Operations

```rust
// Delete single point
let deleted = engine.delete(42)?;

// Delete multiple points
for id in [1, 2, 3] {
    engine.delete(id)?;
}
```

### Scroll (Pagination)

```rust
use lattice_core::ScrollQuery;

let query = ScrollQuery::new()
    .with_limit(100)
    .with_offset(0);

let result = engine.scroll(&query)?;

for point in result.points {
    println!("Point: {}", point.id);
}

// Iterate all points
let mut offset = 0;
loop {
    let result = engine.scroll(
        &ScrollQuery::new().with_limit(100).with_offset(offset)
    )?;

    if result.points.is_empty() {
        break;
    }

    for point in &result.points {
        process(point);
    }

    offset += result.points.len();
}
```

## Graph Operations

### Adding Edges

```rust
use lattice_core::Edge;

// Add single edge
engine.add_edge(1, Edge::new(2, 0.9, "REFERENCES"))?;

// Add multiple edges from same source
engine.add_edge(1, Edge::new(3, 0.7, "CITES"))?;
engine.add_edge(1, Edge::new(4, 0.8, "RELATED_TO"))?;
```

### Getting Neighbors

```rust
// Get all outgoing edges
let neighbors = engine.get_neighbors(1)?;

for edge in neighbors {
    println!("→ {} (weight: {}, type: {})",
        edge.target, edge.weight, edge.relation);
}

// Filter by relation type
let references = engine.get_neighbors_by_type(1, "REFERENCES")?;
```

### Graph Traversal

```rust
use lattice_core::graph::{BfsIterator, DfsIterator};

// BFS from node 1, max depth 3
let get_neighbors = |id| {
    engine.get_neighbors(id)
        .map(|edges| edges.iter().map(|e| e.target).collect())
        .unwrap_or_default()
};

for (node_id, depth) in BfsIterator::new(1, 3, get_neighbors) {
    println!("Node {} at depth {}", node_id, depth);
}

// DFS traversal
for (node_id, depth) in DfsIterator::new(1, 5, get_neighbors) {
    println!("Visiting {} at depth {}", node_id, depth);
}
```

## Cypher Queries

```rust
use lattice_core::cypher::{CypherHandler, DefaultCypherHandler};
use std::collections::HashMap;

let handler = DefaultCypherHandler::new();

// Simple query
let result = handler.query(
    "MATCH (n:Person) RETURN n.name",
    &mut engine,
    HashMap::new(),
)?;

// Query with parameters
let mut params = HashMap::new();
params.insert("min_age".into(), CypherValue::Int(25));
params.insert("category".into(), CypherValue::String("tech".into()));

let result = handler.query(
    "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name, n.age",
    &mut engine,
    params,
)?;

// Process results
println!("Columns: {:?}", result.columns);
for row in result.rows {
    let name = row.get("name");
    let age = row.get("age");
    println!("{:?}, {:?}", name, age);
}

// Check stats
println!("Execution time: {:?}", result.stats.execution_time);
println!("Rows returned: {}", result.stats.rows_returned);
```

## HNSW Index Direct Access

```rust
use lattice_core::{HnswIndex, Distance, HnswConfig};

// Create index directly (without storage)
let config = HnswConfig::default();
let mut index = HnswIndex::new(config, Distance::Cosine);

// Insert points
index.insert(&point);

// Search
let results = index.search(&query_vector, k=10, ef=100);

// Batch search
let queries: Vec<&[f32]> = query_vectors.iter()
    .map(|v| v.as_slice())
    .collect();

let batch_results = index.search_batch(&queries, k=10, ef=100);

// Get statistics
println!("Index size: {}", index.len());
println!("Layer counts: {:?}", index.layer_counts());
println!("Memory: {} bytes", index.vector_memory_bytes());
```

## Storage Implementations

### MemStorage

```rust
use lattice_storage::MemStorage;

let storage = MemStorage::new();
// Fast, ephemeral, for testing
```

### DiskStorage (Native)

```rust
use lattice_storage::DiskStorage;
use std::path::Path;

let storage = DiskStorage::new(Path::new("./data"))?;
// Persistent, uses tokio::fs
```

### OpfsStorage (WASM)

```rust
#[cfg(target_arch = "wasm32")]
use lattice_storage::OpfsStorage;

let storage = OpfsStorage::new().await?;
// Browser persistent storage
```

## Server Usage

### Starting the Server

```rust
use lattice_server::{
    axum_transport::AxumTransport,
    router::{new_app_state, route},
};

#[tokio::main]
async fn main() {
    let state = new_app_state();
    let transport = AxumTransport::new("0.0.0.0:6333");

    transport.serve(move |request| {
        let state = state.clone();
        async move { route(state, request).await }
    }).await.unwrap();
}
```

### Custom Handlers

```rust
use lattice_server::handlers;
use lattice_core::{LatticeRequest, LatticeResponse};

async fn custom_route(
    state: &AppState,
    request: &LatticeRequest,
) -> LatticeResponse {
    // Access collection
    let collections = state.collections.read().await;
    let engine = collections.get("my_collection")?;

    // Process request
    let result = engine.search(&query)?;

    // Return response
    LatticeResponse::ok(serde_json::to_vec(&result)?)
}
```

## Error Handling

```rust
use lattice_core::{LatticeError, LatticeResult};

fn process() -> LatticeResult<()> {
    let engine = CollectionEngine::new(config, storage)?;

    match engine.get_point(999) {
        Ok(point) => println!("Found: {:?}", point),
        Err(LatticeError::NotFound { .. }) => println!("Not found"),
        Err(e) => return Err(e),
    }

    Ok(())
}

// Error types
match error {
    LatticeError::NotFound { resource, id } => ...,
    LatticeError::InvalidConfig { message } => ...,
    LatticeError::Storage(storage_error) => ...,
    LatticeError::Cypher(cypher_error) => ...,
}
```

## Next Steps

- [TypeScript API](./typescript.md) - Browser/Node.js client
- [REST API](./rest.md) - HTTP endpoints
