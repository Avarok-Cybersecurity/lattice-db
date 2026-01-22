# Traversal Algorithms

LatticeDB provides iterators for graph traversal, enabling efficient exploration of connected nodes. This chapter covers BFS, DFS, and path finding.

## Traversal Types

| Algorithm | Order | Use Case |
|-----------|-------|----------|
| BFS | Level by level | Shortest paths, nearest neighbors |
| DFS | Deep first | Exhaustive search, topological sort |

## Breadth-First Search (BFS)

BFS visits nodes level by level, finding shortest paths first.

### Algorithm

```
Start: [A]
Level 0: A
Level 1: B, C (neighbors of A)
Level 2: D, E (neighbors of B, C)
...
```

### Usage

```rust
use lattice_core::graph::BfsIterator;

// Define neighbor lookup function
let get_neighbors = |node_id| {
    engine.get_neighbors(node_id)
        .map(|edges| edges.iter().map(|e| e.target).collect())
        .unwrap_or_default()
};

// Create BFS iterator
let bfs = BfsIterator::new(
    start_node,    // Starting node ID
    max_depth: 3,  // Maximum traversal depth
    get_neighbors,
);

// Iterate over nodes with depth
for (node_id, depth) in bfs {
    println!("Node {} at depth {}", node_id, depth);
}
```

### Example: Find All Nodes Within 2 Hops

```rust
let within_2_hops: Vec<PointId> = BfsIterator::new(start, 2, get_neighbors)
    .map(|(id, _)| id)
    .collect();
```

### Example: Level-Grouped Results

```rust
let mut levels: HashMap<usize, Vec<PointId>> = HashMap::new();

for (node_id, depth) in BfsIterator::new(start, 5, get_neighbors) {
    levels.entry(depth).or_default().push(node_id);
}

for level in 0..=5 {
    if let Some(nodes) = levels.get(&level) {
        println!("Level {}: {} nodes", level, nodes.len());
    }
}
```

## Depth-First Search (DFS)

DFS explores as far as possible along each branch before backtracking.

### Algorithm

```
Start: [A]
Visit A → B → D (go deep)
Backtrack to B → E
Backtrack to A → C → F
...
```

### Usage

```rust
use lattice_core::graph::DfsIterator;

let dfs = DfsIterator::new(
    start_node,
    max_depth: 10,
    get_neighbors,
);

for (node_id, depth) in dfs {
    println!("Visiting node {} at depth {}", node_id, depth);
}
```

### Example: Find All Reachable Nodes

```rust
let reachable: HashSet<PointId> = DfsIterator::new(start, usize::MAX, get_neighbors)
    .map(|(id, _)| id)
    .collect();
```

### Example: Path Recording

```rust
let mut paths: Vec<Vec<PointId>> = Vec::new();
let mut current_path = Vec::new();

for (node_id, depth) in DfsIterator::new(start, 5, get_neighbors) {
    // Truncate path to current depth
    current_path.truncate(depth);
    current_path.push(node_id);

    // If leaf node, save path
    if get_neighbors(node_id).is_empty() {
        paths.push(current_path.clone());
    }
}
```

## GraphPath

`GraphPath` represents a sequence of nodes with total weight.

### Creation

```rust
use lattice_core::graph::GraphPath;

// Start a path
let path = GraphPath::new(1);
assert_eq!(path.len(), 0);  // No edges yet

// Extend the path
let path = path.extend(2, 0.5);  // Add node 2 with edge weight 0.5
assert_eq!(path.len(), 1);  // One edge

let path = path.extend(3, 0.3);
assert_eq!(path.len(), 2);
assert_eq!(path.total_weight, 0.8);
```

### Properties

```rust
let path = GraphPath::new(1)
    .extend(2, 0.5)
    .extend(3, 0.3);

assert_eq!(path.start(), Some(1));
assert_eq!(path.end(), Some(3));
assert_eq!(path.nodes, vec![1, 2, 3]);
assert_eq!(path.total_weight, 0.8);
```

## Shortest Path (Dijkstra)

For weighted shortest paths, combine BFS with weight tracking:

```rust
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Reverse;

fn shortest_path(
    start: PointId,
    end: PointId,
    get_edges: impl Fn(PointId) -> Vec<Edge>,
) -> Option<GraphPath> {
    let mut distances: HashMap<PointId, f32> = HashMap::new();
    let mut predecessors: HashMap<PointId, (PointId, f32)> = HashMap::new();
    let mut heap = BinaryHeap::new();

    distances.insert(start, 0.0);
    heap.push(Reverse((0.0f32, start)));

    while let Some(Reverse((dist, node))) = heap.pop() {
        if node == end {
            // Reconstruct path
            return Some(reconstruct_path(start, end, &predecessors));
        }

        if dist > *distances.get(&node).unwrap_or(&f32::MAX) {
            continue;
        }

        for edge in get_edges(node) {
            let new_dist = dist + edge.weight;
            if new_dist < *distances.get(&edge.target).unwrap_or(&f32::MAX) {
                distances.insert(edge.target, new_dist);
                predecessors.insert(edge.target, (node, edge.weight));
                heap.push(Reverse((new_dist, edge.target)));
            }
        }
    }

    None  // No path found
}

fn reconstruct_path(
    start: PointId,
    end: PointId,
    predecessors: &HashMap<PointId, (PointId, f32)>,
) -> GraphPath {
    let mut path = GraphPath::new(end);
    let mut current = end;

    while current != start {
        if let Some(&(pred, weight)) = predecessors.get(&current) {
            path = GraphPath::new(pred).extend(current, weight);
            current = pred;
        } else {
            break;
        }
    }

    // Reverse the path (we built it backwards)
    GraphPath {
        nodes: path.nodes.into_iter().rev().collect(),
        total_weight: path.total_weight,
    }
}
```

## Filtered Traversal

Filter edges during traversal:

### By Relation Type

```rust
let get_knows_neighbors = |node_id| {
    engine.get_neighbors(node_id)
        .map(|edges| {
            edges.iter()
                .filter(|e| e.relation == "KNOWS")
                .map(|e| e.target)
                .collect()
        })
        .unwrap_or_default()
};

let social_network: Vec<_> = BfsIterator::new(start, 3, get_knows_neighbors).collect();
```

### By Weight Threshold

```rust
let get_strong_connections = |node_id| {
    engine.get_neighbors(node_id)
        .map(|edges| {
            edges.iter()
                .filter(|e| e.weight > 0.8)  // Only strong connections
                .map(|e| e.target)
                .collect()
        })
        .unwrap_or_default()
};
```

### By Node Property

```rust
let get_active_neighbors = |node_id| {
    engine.get_neighbors(node_id)
        .map(|edges| {
            edges.iter()
                .filter(|e| {
                    // Check if target node is active
                    engine.get_point(e.target)
                        .and_then(|p| p.payload.get("active"))
                        .map(|v| v.as_bool().unwrap_or(false))
                        .unwrap_or(false)
                })
                .map(|e| e.target)
                .collect()
        })
        .unwrap_or_default()
};
```

## Cypher Traversal

Cypher queries compile to traversal operations:

### Single Hop

```cypher
MATCH (a:Person)-[:KNOWS]->(b)
WHERE a.name = "Alice"
RETURN b
```

Compiles to:
```rust
let alice = find_by_label_and_property("Person", "name", "Alice");
let friends = get_neighbors_by_type(alice, "KNOWS");
```

### Variable Length

```cypher
MATCH (a:Person)-[:KNOWS*1..3]->(b)
WHERE a.name = "Alice"
RETURN DISTINCT b
```

Compiles to:
```rust
let alice = find_by_label_and_property("Person", "name", "Alice");
let mut results = HashSet::new();

for (node, depth) in BfsIterator::new(alice, 3, get_knows_neighbors) {
    if depth >= 1 && depth <= 3 {
        results.insert(node);
    }
}
```

## Performance Considerations

### Traversal Complexity

| Algorithm | Time | Space | Notes |
|-----------|------|-------|-------|
| BFS | O(V + E) | O(V) | Queue-based, finds shortest |
| DFS | O(V + E) | O(V) | Stack-based, memory efficient |
| Dijkstra | O((V + E) log V) | O(V) | Weighted shortest path |

### Optimization Tips

**1. Limit Depth**
```rust
// Good: Bounded traversal
BfsIterator::new(start, 3, get_neighbors)

// Risky: Unbounded can be slow on dense graphs
BfsIterator::new(start, usize::MAX, get_neighbors)
```

**2. Early Termination**
```rust
// Stop when target found
for (node, _) in BfsIterator::new(start, 10, get_neighbors) {
    if node == target {
        break;
    }
}
```

**3. Batch Neighbor Lookups**
```rust
// Prefetch neighbors for nodes at current level
let level_nodes: Vec<_> = current_level.collect();
let all_neighbors: HashMap<_, _> = level_nodes.iter()
    .map(|&id| (id, engine.get_neighbors(id)))
    .collect();
```

**4. Use Indexes**
```rust
// For filtered traversal, use label index
let persons = engine.get_nodes_by_label("Person")?;
for person in persons {
    // Already filtered to Person label
}
```

## Next Steps

- [Graph Model](./model.md) - Understanding nodes and edges
- [Cypher Language](./cypher.md) - Declarative graph queries
