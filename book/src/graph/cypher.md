# Cypher Query Language

LatticeDB implements a subset of the **openCypher** query language for graph operations. This chapter covers the supported syntax and patterns.

## Query Structure

A typical Cypher query follows this structure:

```cypher
MATCH <pattern>
WHERE <predicate>
RETURN <expression>
ORDER BY <expression>
SKIP <number>
LIMIT <number>
```

## MATCH Clause

### Node Patterns

Match all nodes:
```cypher
MATCH (n) RETURN n
```

Match nodes with a label:
```cypher
MATCH (n:Person) RETURN n
```

Match nodes with multiple labels:
```cypher
MATCH (n:Person:Employee) RETURN n
```

Match nodes with properties:
```cypher
MATCH (n:Person {name: "Alice"}) RETURN n
```

### Relationship Patterns

Match outgoing relationships:
```cypher
MATCH (a)-[r]->(b) RETURN a, r, b
```

Match with relationship type:
```cypher
MATCH (a)-[r:KNOWS]->(b) RETURN a, b
```

Match with multiple relationship types:
```cypher
MATCH (a)-[r:KNOWS|WORKS_WITH]->(b) RETURN a, b
```

Match in either direction:
```cypher
MATCH (a)-[r:KNOWS]-(b) RETURN a, b
```

### Variable-Length Paths

Match paths of specific length:
```cypher
MATCH (a)-[*2]->(b) RETURN a, b  -- Exactly 2 hops
```

Match paths within a range:
```cypher
MATCH (a)-[*1..3]->(b) RETURN a, b  -- 1 to 3 hops
```

Match paths up to a limit:
```cypher
MATCH (a)-[*..5]->(b) RETURN a, b  -- Up to 5 hops
```

## WHERE Clause

### Comparison Operators

```cypher
WHERE n.age > 25
WHERE n.age >= 25
WHERE n.age < 30
WHERE n.age <= 30
WHERE n.name = "Alice"
WHERE n.name <> "Bob"
```

### Logical Operators

```cypher
WHERE n.age > 25 AND n.active = true
WHERE n.role = "admin" OR n.role = "superuser"
WHERE NOT n.deleted
```

### String Matching

```cypher
WHERE n.name STARTS WITH "Al"
WHERE n.email ENDS WITH "@example.com"
WHERE n.description CONTAINS "important"
```

### List Membership

```cypher
WHERE n.status IN ["active", "pending"]
WHERE n.category IN $allowed_categories
```

### NULL Checks

```cypher
WHERE n.email IS NOT NULL
WHERE n.deleted IS NULL
```

### Property Existence

```cypher
WHERE exists(n.email)
WHERE NOT exists(n.deleted_at)
```

## RETURN Clause

### Return Nodes and Properties

```cypher
RETURN n                    -- Return entire node
RETURN n.name              -- Return single property
RETURN n.name, n.age       -- Return multiple properties
RETURN n.name AS fullName  -- Alias
```

### Aggregations

```cypher
RETURN count(n)                    -- Count
RETURN count(DISTINCT n.category)  -- Distinct count
RETURN sum(n.amount)               -- Sum
RETURN avg(n.score)                -- Average
RETURN min(n.created_at)           -- Minimum
RETURN max(n.updated_at)           -- Maximum
RETURN collect(n.name)             -- Collect into list
```

### Grouping

```cypher
MATCH (n:Person)
RETURN n.department, count(n) AS count
```

## ORDER BY, SKIP, LIMIT

### Sorting

```cypher
ORDER BY n.name              -- Ascending (default)
ORDER BY n.name ASC          -- Explicit ascending
ORDER BY n.name DESC         -- Descending
ORDER BY n.last_name, n.first_name  -- Multiple columns
```

### Pagination

```cypher
SKIP 10 LIMIT 20            -- Skip first 10, return next 20
LIMIT 100                   -- Return first 100
```

## CREATE Clause

### Create Nodes

```cypher
CREATE (n:Person {name: "Alice", age: 30})
```

### Create Relationships

```cypher
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Bob"})
CREATE (a)-[:KNOWS {since: 2020}]->(b)
```

### Create with RETURN

```cypher
CREATE (n:Person {name: "Charlie"})
RETURN n
```

## DELETE Clause

### Delete Nodes

```cypher
MATCH (n:Person {name: "DeleteMe"})
DELETE n
```

### Delete Relationships

```cypher
MATCH (a)-[r:KNOWS]->(b)
WHERE a.name = "Alice" AND b.name = "Bob"
DELETE r
```

### DETACH DELETE (Nodes + Edges)

```cypher
MATCH (n:Person {name: "DeleteMe"})
DETACH DELETE n  -- Deletes node and all its relationships
```

## SET Clause

### Update Properties

```cypher
MATCH (n:Person {name: "Alice"})
SET n.age = 31
```

### Set Multiple Properties

```cypher
MATCH (n:Person {name: "Alice"})
SET n.age = 31, n.updated = true
```

### Remove Property

```cypher
MATCH (n:Person {name: "Alice"})
SET n.temp = NULL  -- Removes the property
```

## Parameters

Use `$` prefix for query parameters:

```cypher
MATCH (n:Person {name: $name})
WHERE n.age > $min_age
RETURN n
LIMIT $limit
```

### Rust Usage

```rust
use std::collections::HashMap;
use lattice_core::CypherValue;

let mut params = HashMap::new();
params.insert("name".to_string(), CypherValue::String("Alice".into()));
params.insert("min_age".to_string(), CypherValue::Int(25));
params.insert("limit".to_string(), CypherValue::Int(10));

let result = handler.query(
    "MATCH (n:Person {name: $name}) WHERE n.age > $min_age RETURN n LIMIT $limit",
    &mut engine,
    params,
)?;
```

### REST API

```bash
curl -X POST http://localhost:6333/collections/my_collection/graph/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "MATCH (n:Person {name: $name}) RETURN n",
    "params": {
      "name": "Alice"
    }
  }'
```

## Query Examples

### Find Connected Nodes

```cypher
-- Find all people Alice knows
MATCH (alice:Person {name: "Alice"})-[:KNOWS]->(friend)
RETURN friend.name
```

### Find Paths

```cypher
-- Find path from Alice to Bob (up to 4 hops)
MATCH path = (alice:Person {name: "Alice"})-[*1..4]->(bob:Person {name: "Bob"})
RETURN path
```

### Aggregation Query

```cypher
-- Count employees by department
MATCH (e:Employee)-[:WORKS_IN]->(d:Department)
RETURN d.name AS department, count(e) AS employee_count
ORDER BY employee_count DESC
```

### Subgraph Extraction

```cypher
-- Get a subgraph around a node
MATCH (center:Person {name: "Alice"})-[r*1..2]-(connected)
RETURN center, r, connected
```

### Hybrid Vector + Graph

```cypher
-- Find similar documents and their authors
MATCH (doc:Document)-[:AUTHORED_BY]->(author:Person)
WHERE doc.embedding <-> $query_embedding < 0.5
RETURN doc.title, author.name
ORDER BY doc.embedding <-> $query_embedding
LIMIT 10
```

## Query Execution

### Architecture

```
Query String
    │
    ▼
┌─────────────┐
│   Parser    │  Pest grammar → AST
└─────────────┘
    │
    ▼
┌─────────────┐
│  Planner    │  AST → Logical Plan
└─────────────┘
    │
    ▼
┌─────────────┐
│  Executor   │  Execute against storage
└─────────────┘
    │
    ▼
  Results
```

### Handler API

```rust
use lattice_core::cypher::{CypherHandler, DefaultCypherHandler};

let handler = DefaultCypherHandler::new();

// Execute a query
let result = handler.query(
    "MATCH (n:Person) RETURN n.name",
    &mut engine,
    HashMap::new(),
)?;

// Process results
for row in result.rows {
    if let Some(name) = row.get("name") {
        println!("Name: {:?}", name);
    }
}

// Check execution stats
println!("Rows returned: {}", result.stats.rows_returned);
println!("Execution time: {:?}", result.stats.execution_time);
```

## Limitations

Current implementation supports a subset of openCypher:

| Feature | Status | Notes |
|---------|--------|-------|
| MATCH | ✅ | Single and multi-pattern |
| WHERE | ✅ | Basic predicates |
| RETURN | ✅ | Properties, aliases, aggregates |
| CREATE | ✅ | Nodes and relationships |
| DELETE | ✅ | With DETACH |
| SET | ✅ | Property updates |
| LIMIT/SKIP | ✅ | Pagination |
| ORDER BY | ✅ | Single/multi column |
| WITH | ⚠️ | Basic support |
| UNWIND | ⚠️ | Limited |
| MERGE | ❌ | Not yet implemented |
| FOREACH | ❌ | Not yet implemented |
| CALL | ❌ | Procedures not supported |

## Next Steps

- [Traversal Algorithms](./traversal.md) - BFS, DFS, and path finding
- [Graph Model](./model.md) - Understanding the data model
