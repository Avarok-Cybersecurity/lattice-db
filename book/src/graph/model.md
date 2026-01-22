# Graph Model

LatticeDB implements a **property graph model** where nodes (points) can have properties and labeled edges connecting them. This chapter explains the graph data model and how to work with it.

## Core Concepts

### Nodes (Points)

In LatticeDB, nodes are called **Points**. Each point has:

- **ID**: Unique 64-bit identifier (`PointId`)
- **Vector**: Dense embedding for similarity search
- **Payload**: Key-value properties
- **Labels**: Optional node type labels

```rust
use lattice_core::{Point, Edge};

// Create a node with vector and properties
let point = Point::new_vector(1, vec![0.1, 0.2, 0.3])
    .with_payload("name", "Alice")
    .with_payload("age", 30)
    .with_label("Person");
```

### Edges

Edges connect nodes with optional:

- **Weight**: f32 similarity/relevance score
- **Relation type**: String label for the relationship

```rust
// Edge from node 1 to node 2
let edge = Edge::new(2, 0.9, "KNOWS");
// Fields: target_id, weight, relation_type
```

### Graph Structure

The graph is stored as an **adjacency list**:

```
Point 1 → [(Edge to 2), (Edge to 3)]
Point 2 → [(Edge to 3)]
Point 3 → [(Edge to 1)]
```

This enables efficient:
- **O(1)** neighbor lookup by source node
- **O(E/N)** average edge retrieval
- **O(1)** edge insertion

## Working with Graphs

### Adding Edges

```rust
use lattice_core::{CollectionEngine, Edge};

// Add a single edge
engine.add_edge(1, Edge::new(2, 0.9, "REFERENCES"))?;

// Add multiple edges
engine.add_edge(1, Edge::new(3, 0.7, "REFERENCES"))?;
engine.add_edge(2, Edge::new(3, 0.8, "CITES"))?;
```

### Getting Neighbors

```rust
// Get all outgoing edges from node 1
let neighbors = engine.get_neighbors(1)?;

for edge in neighbors {
    println!("→ {} (weight: {}, type: {})",
        edge.target, edge.weight, edge.relation);
}
```

### REST API

```bash
# Add an edge
curl -X POST http://localhost:6333/collections/my_collection/graph/edges \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": 1,
    "target_id": 2,
    "weight": 0.9,
    "relation": "REFERENCES"
  }'

# Get neighbors
curl http://localhost:6333/collections/my_collection/graph/neighbors/1
```

## Edge Properties

### Weight

Edge weight is a f32 value typically representing:
- **Similarity**: Higher = more similar (0.0 to 1.0)
- **Relevance**: Higher = more relevant
- **Distance**: Lower = closer (depending on use case)

```rust
// High-confidence relationship
Edge::new(2, 0.95, "CONFIRMED_MATCH");

// Lower confidence
Edge::new(3, 0.6, "POSSIBLE_MATCH");
```

### Relation Types

Relation types are strings that categorize edges:

```rust
// Document relationships
Edge::new(2, 0.9, "REFERENCES");
Edge::new(3, 0.8, "CITES");
Edge::new(4, 0.7, "RELATED_TO");

// Social relationships
Edge::new(2, 1.0, "KNOWS");
Edge::new(3, 1.0, "WORKS_WITH");

// Hierarchical relationships
Edge::new(2, 1.0, "PARENT_OF");
Edge::new(3, 1.0, "CHILD_OF");
```

## Directionality

Edges in LatticeDB are **directed**:

```
Node A --[KNOWS]--> Node B
```

This means:
- Edge A→B exists
- Edge B→A does NOT automatically exist

To create bidirectional relationships:

```rust
// Bidirectional KNOWS relationship
engine.add_edge(1, Edge::new(2, 1.0, "KNOWS"))?;
engine.add_edge(2, Edge::new(1, 1.0, "KNOWS"))?;
```

Or use Cypher with variable-length patterns:

```cypher
// Match edges in either direction
MATCH (a)-[:KNOWS]-(b)
WHERE id(a) = 1
RETURN b
```

## Labels (Node Types)

Labels categorize nodes:

```rust
let person = Point::new_vector(1, embedding)
    .with_label("Person")
    .with_label("Employee");  // Multiple labels

let company = Point::new_vector(2, embedding)
    .with_label("Company");
```

Query by label:

```cypher
-- Find all Person nodes
MATCH (n:Person) RETURN n

-- Find Person nodes that are also Employees
MATCH (n:Person:Employee) RETURN n
```

## Hybrid Queries

The power of LatticeDB is combining vector search with graph traversal:

### Vector Search → Graph Expansion

```rust
// 1. Find similar documents via vector search
let similar = engine.search(&SearchQuery::new(query_vector).with_limit(5))?;

// 2. Expand each result via graph traversal
for result in similar {
    let references = engine.get_neighbors_by_type(result.id, "REFERENCES")?;
    println!("Document {} references: {:?}", result.id, references);
}
```

### Cypher with Vector Predicates

```cypher
-- Find similar documents and their references
MATCH (doc:Document)-[:REFERENCES]->(ref:Document)
WHERE doc.embedding <-> $query_embedding < 0.5
RETURN doc.title, ref.title
```

### Graph → Vector Reranking

```rust
// 1. Get candidates from graph traversal
let cypher_results = handler.query(
    "MATCH (n:Person)-[:WORKS_AT]->(c:Company {name: 'Acme'}) RETURN n",
    &mut engine,
    params,
)?;

// 2. Rerank by vector similarity
let mut candidates: Vec<_> = cypher_results.rows
    .iter()
    .filter_map(|row| {
        let id = row.get("n")?.as_node_id()?;
        let vec = engine.get_vector(id)?;
        let dist = distance.calculate(&query_vec, vec);
        Some((id, dist))
    })
    .collect();

candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
```

## Memory Layout

### Adjacency Storage

Edges are stored in a `HashMap<PointId, Vec<Edge>>`:

```
┌────────────────────────────────────────┐
│ Node 1 → [Edge(2, 0.9), Edge(3, 0.7)]  │
│ Node 2 → [Edge(3, 0.8)]                │
│ Node 3 → [Edge(1, 0.5)]                │
└────────────────────────────────────────┘
```

Memory per edge: ~20 bytes (target_id + weight + relation string pointer)

### Index Structure

For efficient traversal, LatticeDB maintains:

1. **Forward index**: Source → [Edges] (primary storage)
2. **Point lookup**: ID → Point (for properties)
3. **Label index**: Label → [PointIds] (for label queries)

## Best Practices

### 1. Use Meaningful Relation Types

```rust
// Good: Specific, queryable
Edge::new(2, 0.9, "AUTHORED_BY");
Edge::new(3, 0.8, "PUBLISHED_IN");

// Bad: Generic, hard to filter
Edge::new(2, 0.9, "RELATED");
```

### 2. Normalize Weights

```rust
// Good: Consistent 0-1 scale
Edge::new(2, 0.95, "HIGH_CONFIDENCE");
Edge::new(3, 0.60, "MEDIUM_CONFIDENCE");

// Bad: Inconsistent scales
Edge::new(2, 100.0, "TYPE_A");
Edge::new(3, 0.6, "TYPE_B");
```

### 3. Consider Edge Density

High edge counts per node can impact traversal performance:

| Edges per Node | Performance | Use Case |
|----------------|-------------|----------|
| 1-10 | Excellent | Typical relationships |
| 10-100 | Good | Dense graphs |
| 100+ | Consider filtering | Social networks |

### 4. Batch Edge Operations

```rust
// Good: Batch insert
let edges = vec![
    (1, Edge::new(2, 0.9, "TYPE")),
    (1, Edge::new(3, 0.8, "TYPE")),
    (2, Edge::new(3, 0.7, "TYPE")),
];
for (source, edge) in edges {
    engine.add_edge(source, edge)?;
}

// Even better: Use Cypher CREATE for multiple edges
handler.query(
    "UNWIND $edges AS e CREATE (a)-[r:TYPE {weight: e.weight}]->(b)
     WHERE id(a) = e.source AND id(b) = e.target",
    &mut engine,
    params,
)?;
```

## Next Steps

- [Cypher Query Language](./cypher.md) - Graph query syntax
- [Traversal Algorithms](./traversal.md) - BFS, DFS, and path finding
