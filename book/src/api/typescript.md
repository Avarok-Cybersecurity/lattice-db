# TypeScript API

LatticeDB provides a TypeScript/JavaScript client for browser and Node.js environments. The client works with both the WASM build (in-browser) and the REST API (remote server).

## Installation

```bash
npm install lattice-db
```

## Quick Start

### Browser (WASM)

```typescript
import { LatticeDB } from 'lattice-db';

async function main() {
  // Initialize WASM module and get database instance
  const db = await LatticeDB.init();

  // Create a collection
  db.createCollection('my_collection', {
    vectors: { size: 128, distance: 'Cosine' }
  });

  // Add points
  db.upsert('my_collection', [
    {
      id: 1,
      vector: new Float32Array(128).fill(0.1),
      payload: { title: 'Hello World' }
    }
  ]);

  // Search
  const results = db.search(
    'my_collection',
    new Float32Array(128).fill(0.1),
    10
  );

  console.log('Results:', results);
}

main();
```

### Node.js (REST Client)

```typescript
import { LatticeClient } from 'lattice-db';

const client = new LatticeClient('http://localhost:6333');

// Create collection
await client.createCollection('my_collection', {
  vectors: { size: 128, distance: 'Cosine' }
});

// Upsert points
await client.upsert('my_collection', {
  points: [
    { id: 1, vector: [...], payload: { title: 'Doc 1' } }
  ]
});

// Search
const results = await client.search('my_collection', {
  query: [...],
  limit: 10
});
```

## WASM API

### Initialization

```typescript
import { LatticeDB } from 'lattice-db';

// Initialize WASM and get database instance
const db = await LatticeDB.init();

// Or with custom WASM path
const db = await LatticeDB.init('/path/to/lattice.wasm');
```

### Creating a Collection

```typescript
// Create a collection with default HNSW config
db.createCollection('my_collection', {
  vectors: { size: 128, distance: 'Cosine' }
});

// Create with custom HNSW config
db.createCollection('my_collection', {
  vectors: { size: 128, distance: 'Cosine' },
  hnsw_config: { m: 16, ef_construct: 200 }
});
```

### Upsert Points

```typescript
// Batch upsert (always array)
db.upsert('my_collection', [
  {
    id: 1,
    vector: new Float32Array([0.1, 0.2, ...]),
    payload: { title: 'My Document', tags: ['rust', 'database'] }
  },
  {
    id: 2,
    vector: new Float32Array([0.3, 0.4, ...]),
    payload: { title: 'Another Doc' }
  }
]);
```

### Search

```typescript
// Basic search
const results = db.search(
  'my_collection',
  queryVector,  // Float32Array
  10            // limit
);

// Search with options (snake_case for WASM binding)
const results = db.search('my_collection', queryVector, 10, {
  with_payload: true,   // Include payload in results (default: true)
  with_vector: false,   // Include vector in results (default: false)
  score_threshold: 0.5  // Optional minimum score filter
});

// Results format
for (const result of results) {
  console.log(`ID: ${result.id}, Score: ${result.score}`);
  console.log(`Payload: ${JSON.stringify(result.payload)}`);
}
```

### Retrieve Points

```typescript
// Get multiple points by ID
const points = db.getPoints('my_collection',
  BigUint64Array.from([1n, 2n, 3n]),
  true,  // withPayload
  false  // withVector
);
```

### Delete Points

```typescript
// Delete by IDs
db.deletePoints('my_collection', BigUint64Array.from([1n, 2n, 3n]));
```

### Graph Operations

```typescript
// Add edge
db.addEdge('my_collection', 1n, 2n, 'REFERENCES', 0.9);

// Traverse graph
const result = db.traverse('my_collection', 1n, 2, ['REFERENCES']);

// Cypher query
const result = db.query(
  'my_collection',
  'MATCH (n:Person) WHERE n.age > $minAge RETURN n.name',
  { minAge: 25 }
);

for (const row of result.rows) {
  console.log(row['n.name']);
}
```

### Collection Management

```typescript
// List all collections
const collections = db.listCollections();

// Get collection info
const info = db.getCollection('my_collection');
console.log(`Points: ${info.vectors_count}`);

// Delete collection
db.deleteCollection('my_collection');
```

## REST Client

### Creating a Client

```typescript
import { LatticeClient } from 'lattice-db';

const client = new LatticeClient('http://localhost:6333', {
  timeout: 30000,  // Request timeout in ms
  headers: {       // Custom headers
    'Authorization': 'Bearer token'
  }
});
```

### Collections

```typescript
// Create (REST API uses snake_case)
await client.createCollection('docs', {
  vectors: { size: 128, distance: 'Cosine' },
  hnsw_config: { m: 16, ef_construct: 200 }
});

// Get info
const info = await client.getCollection('docs');
console.log(`Vectors: ${info.vectors_count}`);

// List all
const collections = await client.listCollections();

// Delete
await client.deleteCollection('docs');
```

### Points

```typescript
// Upsert
await client.upsert('docs', {
  points: [
    { id: 1, vector: [...], payload: { title: 'Doc 1' } }
  ]
});

// Get
const point = await client.getPoint('docs', 1);

// Delete
await client.deletePoints('docs', { points: [1, 2, 3] });
```

### Search

```typescript
const results = await client.search('docs', {
  query: [...],
  limit: 10,
  filter: {
    must: [
      { key: 'category', match: { value: 'tech' } }
    ]
  },
  with_payload: true  // Note: REST API uses snake_case
});
```

### Scroll

```typescript
// First page
let result = await client.scroll('docs', {
  limit: 100,
  with_payload: true  // Note: REST API uses snake_case
});

// Subsequent pages
while (result.next_page_offset !== null) {
  result = await client.scroll('docs', {
    limit: 100,
    offset: result.next_page_offset
  });
  // Process result.points
}
```

### Graph

```typescript
// Add edge
await client.addEdge('docs', {
  sourceId: 1,
  targetId: 2,
  weight: 0.9,
  relation: 'REFERENCES'
});

// Get neighbors
const neighbors = await client.getNeighbors('docs', 1);

// Cypher query
const result = await client.cypherQuery('docs', {
  query: 'MATCH (n:Person) RETURN n.name',
  parameters: {}
});
```

## React Integration

### Hook

```typescript
import { useEffect, useState } from 'react';
import { LatticeDB } from 'lattice-db';

function useLatticeDB() {
  const [db, setDb] = useState<LatticeDB | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function initialize() {
      try {
        const instance = await LatticeDB.init();
        setDb(instance);
      } catch (e) {
        setError(e as Error);
      } finally {
        setLoading(false);
      }
    }
    initialize();
  }, []);

  return { db, loading, error };
}
```

### Component

```tsx
function SearchComponent() {
  const { db, loading } = useLatticeDB();
  const [results, setResults] = useState<SearchResult[]>([]);

  // Create collection on first load
  useEffect(() => {
    if (!db) return;
    try {
      db.createCollection('docs', {
        vectors: { size: 128, distance: 'Cosine' }
      });
    } catch {
      // Collection may already exist
    }
  }, [db]);

  const handleSearch = (queryVector: Float32Array) => {
    if (!db) return;
    const searchResults = db.search('docs', Array.from(queryVector), 10);
    setResults(searchResults);
  };

  if (loading) return <div>Loading...</div>;

  return (
    <div>
      <SearchInput onSearch={handleSearch} />
      <ResultsList results={results} />
    </div>
  );
}
```

## Vue Integration

```vue
<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { LatticeDB } from 'lattice-db';

const db = ref<LatticeDB | null>(null);
const results = ref<SearchResult[]>([]);

onMounted(async () => {
  db.value = await LatticeDB.init();

  // Create collection (will throw if already exists)
  try {
    db.value.createCollection('docs', {
      vectors: { size: 128, distance: 'Cosine' }
    });
  } catch {
    // Collection may already exist
  }
});

function search(vector: Float32Array) {
  if (!db.value) return;
  results.value = db.value.search('docs', Array.from(vector), 10);
}
</script>

<template>
  <SearchInput @search="search" />
  <ResultsList :results="results" />
</template>
```

## TypeScript Types

```typescript
// Distance metrics
type DistanceMetric = 'Cosine' | 'Euclid' | 'Dot';

// Collection configuration
interface CollectionConfig {
  vectors: { size: number; distance: DistanceMetric };
  hnsw_config?: { m: number; m0?: number; ef_construct: number };
}

// Point to upsert
interface Point {
  id: number;
  vector: number[];
  payload?: Record<string, unknown>;
}

// Search options
interface SearchOptions {
  with_payload?: boolean;
  with_vector?: boolean;
  score_threshold?: number;
}

// Search result
interface SearchResult {
  id: number;
  score: number;
  payload?: Record<string, unknown>;
  vector?: number[];
}

// Graph traversal result
interface TraversalResult {
  nodes: { id: number; depth: number; payload?: Record<string, unknown> }[];
  edges: { from: number; to: number; relation: string; weight: number }[];
}

// Cypher query result
interface CypherResult {
  columns: string[];
  rows: unknown[][];
}
```

## Error Handling

```typescript
try {
  db.search('my_collection', queryVector, 10);
} catch (error) {
  if (error instanceof Error) {
    if (error.message.includes('not found')) {
      console.log('Collection not found');
    } else if (error.message.includes('dimension')) {
      console.log('Vector dimension mismatch');
    } else {
      console.log('Error:', error.message);
    }
  }
}
```

## Performance Tips

### Batch Upserts

```typescript
// Good: Single call for multiple points
db.upsert('my_collection', [
  { id: 1, vector: [...], payload: { title: 'Doc 1' } },
  { id: 2, vector: [...], payload: { title: 'Doc 2' } },
  { id: 3, vector: [...], payload: { title: 'Doc 3' } }
]);

// Bad: Multiple calls
for (const point of points) {
  db.upsert('my_collection', [point]);
}
```

### Web Worker

Move LatticeDB to a Web Worker to keep the main thread responsive:

```typescript
// worker.ts
import { LatticeDB } from 'lattice-db';

let db: LatticeDB;

self.onmessage = async ({ data }) => {
  if (data.type === 'init') {
    db = await LatticeDB.init();
    db.createCollection('docs', {
      vectors: { size: data.vectorSize, distance: 'Cosine' }
    });
    self.postMessage({ type: 'ready' });
  }

  if (data.type === 'search') {
    const results = db.search('docs', data.vector, data.limit);
    self.postMessage({ type: 'results', results });
  }
};
```

## Next Steps

- [REST API](./rest.md) - HTTP endpoints
- [WASM Browser Setup](../getting-started/wasm.md) - Detailed browser guide
