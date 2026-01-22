# TypeScript API

LatticeDB provides a TypeScript/JavaScript client for browser and Node.js environments. The client works with both the WASM build (in-browser) and the REST API (remote server).

## Installation

```bash
npm install @lattice-db/client
```

## Quick Start

### Browser (WASM)

```typescript
import init, { LatticeDB } from '@lattice-db/client';

async function main() {
  // Initialize WASM module
  await init();

  // Create database instance
  const db = await LatticeDB.create({
    name: 'my_collection',
    vectorSize: 128,
    distance: 'cosine'
  });

  // Add a point
  await db.upsert({
    id: 1,
    vector: new Float32Array(128).fill(0.1),
    payload: { title: 'Hello World' }
  });

  // Search
  const results = await db.search({
    vector: new Float32Array(128).fill(0.1),
    limit: 10
  });

  console.log('Results:', results);
}

main();
```

### Node.js (REST Client)

```typescript
import { LatticeClient } from '@lattice-db/client';

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
import init, { LatticeDB } from '@lattice-db/client';

// Initialize WASM (required once)
await init();

// Or with custom WASM path
await init('/path/to/lattice_db.wasm');
```

### Creating a Database

```typescript
const db = await LatticeDB.create({
  name: 'collection_name',      // Required
  vectorSize: 128,              // Required
  distance: 'cosine',           // 'cosine' | 'euclidean' | 'dot'
  storage: 'memory',            // 'memory' | 'opfs' | 'indexeddb'
  hnsw: {                       // Optional HNSW config
    m: 16,
    m0: 32,
    efConstruction: 200
  }
});
```

### Upsert Points

```typescript
// Single point
await db.upsert({
  id: 1,
  vector: new Float32Array([0.1, 0.2, ...]),
  payload: {
    title: 'My Document',
    tags: ['rust', 'database']
  }
});

// Batch upsert
await db.upsertBatch([
  { id: 1, vector: vec1, payload: { title: 'Doc 1' } },
  { id: 2, vector: vec2, payload: { title: 'Doc 2' } },
  { id: 3, vector: vec3, payload: { title: 'Doc 3' } }
]);
```

### Search

```typescript
const results = await db.search({
  vector: queryVector,
  limit: 10,
  ef: 100,  // Optional: search quality
  withPayload: true,
  withVector: false
});

// Results format
for (const result of results) {
  console.log(`ID: ${result.id}, Score: ${result.score}`);
  console.log(`Payload: ${JSON.stringify(result.payload)}`);
}
```

### Retrieve Points

```typescript
// Single point
const point = await db.get(42);

// Multiple points
const points = await db.getMany([1, 2, 3]);

// Check existence
const exists = await db.has(42);
```

### Delete Points

```typescript
// Single point
await db.delete(42);

// Multiple points
await db.deleteMany([1, 2, 3]);
```

### Graph Operations

```typescript
// Add edge
await db.addEdge({
  sourceId: 1,
  targetId: 2,
  weight: 0.9,
  relation: 'REFERENCES'
});

// Get neighbors
const neighbors = await db.getNeighbors(1);

// Cypher query
const result = await db.query(
  'MATCH (n:Person) WHERE n.age > $minAge RETURN n.name',
  { minAge: 25 }
);

for (const row of result.rows) {
  console.log(row['n.name']);
}
```

### Memory Statistics

```typescript
const stats = db.memoryStats();
console.log(`Vectors: ${stats.vectorBytes} bytes`);
console.log(`Index: ${stats.indexBytes} bytes`);
console.log(`Total: ${stats.totalBytes} bytes`);
```

## REST Client

### Creating a Client

```typescript
import { LatticeClient } from '@lattice-db/client';

const client = new LatticeClient('http://localhost:6333', {
  timeout: 30000,  // Request timeout in ms
  headers: {       // Custom headers
    'Authorization': 'Bearer token'
  }
});
```

### Collections

```typescript
// Create
await client.createCollection('docs', {
  vectors: { size: 128, distance: 'Cosine' },
  hnswConfig: { m: 16, efConstruct: 200 }
});

// Get info
const info = await client.getCollection('docs');
console.log(`Vectors: ${info.vectorsCount}`);

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
  withPayload: true
});
```

### Scroll

```typescript
// First page
let result = await client.scroll('docs', {
  limit: 100,
  withPayload: true
});

// Subsequent pages
while (result.nextPageOffset !== null) {
  result = await client.scroll('docs', {
    limit: 100,
    offset: result.nextPageOffset
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
  params: {}
});
```

## React Integration

### Hook

```typescript
import { useEffect, useState } from 'react';
import init, { LatticeDB } from '@lattice-db/client';

function useLatticeDB(config: LatticeConfig) {
  const [db, setDb] = useState<LatticeDB | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function initialize() {
      try {
        await init();
        const instance = await LatticeDB.create(config);
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
  const { db, loading } = useLatticeDB({
    name: 'docs',
    vectorSize: 128,
    distance: 'cosine'
  });
  const [results, setResults] = useState([]);

  const handleSearch = async (queryVector: Float32Array) => {
    if (!db) return;
    const searchResults = await db.search({
      vector: queryVector,
      limit: 10
    });
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
import init, { LatticeDB } from '@lattice-db/client';

const db = ref<LatticeDB | null>(null);
const results = ref([]);

onMounted(async () => {
  await init();
  db.value = await LatticeDB.create({
    name: 'docs',
    vectorSize: 128,
    distance: 'cosine'
  });
});

async function search(vector: Float32Array) {
  if (!db.value) return;
  results.value = await db.value.search({
    vector,
    limit: 10
  });
}
</script>

<template>
  <SearchInput @search="search" />
  <ResultsList :results="results" />
</template>
```

## TypeScript Types

```typescript
interface Point {
  id: number;
  vector: Float32Array | number[];
  payload?: Record<string, any>;
}

interface SearchParams {
  vector: Float32Array | number[];
  limit: number;
  ef?: number;
  filter?: Filter;
  withPayload?: boolean;
  withVector?: boolean;
}

interface SearchResult {
  id: number;
  score: number;
  payload?: Record<string, any>;
  vector?: Float32Array;
}

interface Edge {
  target: number;
  weight: number;
  relation: string;
}

interface CypherResult {
  columns: string[];
  rows: Record<string, any>[];
  stats: {
    nodesScanned: number;
    rowsReturned: number;
    executionTimeMs: number;
  };
}
```

## Error Handling

```typescript
try {
  await db.search({ vector, limit: 10 });
} catch (error) {
  if (error instanceof LatticeError) {
    switch (error.code) {
      case 'NOT_FOUND':
        console.log('Collection not found');
        break;
      case 'INVALID_VECTOR':
        console.log('Vector dimension mismatch');
        break;
      default:
        console.log('Error:', error.message);
    }
  }
}
```

## Performance Tips

### Use TypedArrays

```typescript
// Good: Zero-copy transfer to WASM
const vector = new Float32Array([0.1, 0.2, 0.3]);

// Bad: Requires conversion
const vector = [0.1, 0.2, 0.3];
```

### Batch Operations

```typescript
// Good: Single WASM call
await db.upsertBatch(points);

// Bad: Multiple WASM calls
for (const point of points) {
  await db.upsert(point);
}
```

### Web Worker

```typescript
// worker.ts
import init, { LatticeDB } from '@lattice-db/client';

let db: LatticeDB;

self.onmessage = async ({ data }) => {
  if (data.type === 'init') {
    await init();
    db = await LatticeDB.create(data.config);
    self.postMessage({ type: 'ready' });
  }

  if (data.type === 'search') {
    const results = await db.search(data.params);
    self.postMessage({ type: 'results', results });
  }
};
```

## Next Steps

- [REST API](./rest.md) - HTTP endpoints
- [WASM Browser Setup](../getting-started/wasm.md) - Detailed browser guide
