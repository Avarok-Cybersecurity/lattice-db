# WASM Browser Setup

LatticeDB compiles to WebAssembly (WASM) and runs entirely in the browser. This guide covers integration options and best practices.

## Quick Setup

### ES Modules

```html
<!DOCTYPE html>
<html>
<head>
  <title>LatticeDB Demo</title>
</head>
<body>
  <script type="module">
    import init, { LatticeDB } from './lattice_db.js';

    async function main() {
      // Initialize WASM module
      await init();

      // Create database instance
      const db = new LatticeDB();

      // Create a collection
      db.createCollection('my_collection', {
        vectors: { size: 128, distance: 'Cosine' }
      });

      // Add points
      db.upsert('my_collection', [
        {
          id: 1,
          vector: Array.from({ length: 128 }, () => 0.1),
          payload: { title: 'Hello World' }
        }
      ]);

      // Search
      const results = db.search(
        'my_collection',
        Array.from({ length: 128 }, () => 0.1),
        10  // limit
      );

      console.log('Results:', results);
    }

    main();
  </script>
</body>
</html>
```

### Service Worker (Planned)

> **Note**: Service Worker transport is planned for a future release.
> Currently, use the direct LatticeDB API from the main thread or a Web Worker.

## Storage

LatticeDB currently uses **in-memory storage** by default. All data is lost when the page reloads.

Future releases will support:
- Origin Private File System (OPFS) for persistent storage
- IndexedDB fallback for older browsers

For now, if you need persistence, consider serializing your data to localStorage or IndexedDB separately.

## Framework Integration

### React

```jsx
import { useEffect, useState, useRef } from 'react';
import init, { LatticeDB } from 'lattice-db';

function useLatticeDB(collectionName, vectorSize) {
  const dbRef = useRef(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    async function initialize() {
      await init();
      const db = new LatticeDB();
      db.createCollection(collectionName, {
        vectors: { size: vectorSize, distance: 'Cosine' }
      });
      dbRef.current = db;
      setReady(true);
    }
    initialize();
  }, [collectionName, vectorSize]);

  return { db: dbRef.current, ready };
}

function SearchComponent() {
  const { db, ready } = useLatticeDB('my_collection', 128);
  const [results, setResults] = useState([]);

  const handleSearch = async (query) => {
    if (!db) return;
    const embedding = await getEmbedding(query); // Your embedding function
    const searchResults = db.search('my_collection', embedding, 10);
    setResults(searchResults);
  };

  if (!ready) return <div>Loading...</div>;

  return (
    <div>
      <input onChange={(e) => handleSearch(e.target.value)} />
      <ul>
        {results.map(r => <li key={r.id}>{r.payload?.title}</li>)}
      </ul>
    </div>
  );
}
```

### Vue

```vue
<script setup>
import { ref, onMounted } from 'vue';
import init, { LatticeDB } from 'lattice-db';

const db = ref(null);
const results = ref([]);

onMounted(async () => {
  await init();
  const instance = new LatticeDB();
  instance.createCollection('my_collection', {
    vectors: { size: 128, distance: 'Cosine' }
  });
  db.value = instance;
});

function search(query) {
  const embedding = getEmbedding(query); // Your embedding function
  results.value = db.value.search('my_collection', embedding, 10);
}
</script>

<template>
  <input @input="search($event.target.value)" />
  <ul>
    <li v-for="r in results" :key="r.id">{{ r.payload?.title }}</li>
  </ul>
</template>
```

## Performance Tips

### 1. Use TypedArrays

Always use `Float32Array` for vectors:

```javascript
// Good - zero-copy transfer to WASM
const vector = new Float32Array([0.1, 0.2, 0.3, ...]);

// Bad - requires conversion
const vector = [0.1, 0.2, 0.3, ...];
```

### 2. Batch Operations

Batch upserts for better performance:

```javascript
// Good - single WASM call with multiple points
db.upsert('my_collection', [point1, point2, point3]);

// Bad - multiple WASM calls
for (const point of points) {
  db.upsert('my_collection', [point]);
}
```

### 3. Web Workers

Offload heavy operations to a Web Worker:

```javascript
// worker.js
import init, { LatticeDB } from 'lattice-db';

let db;

self.onmessage = async ({ data }) => {
  if (data.type === 'init') {
    await init();
    db = new LatticeDB();
    db.createCollection(data.collection, data.config);
    self.postMessage({ type: 'ready' });
  }

  if (data.type === 'search') {
    const results = db.search(data.collection, data.vector, data.limit);
    self.postMessage({ type: 'results', results });
  }
};
```

### 4. SIMD

WASM SIMD is enabled by default for 4-8x faster distance calculations. Ensure your bundler preserves SIMD instructions:

```javascript
// vite.config.js
export default {
  optimizeDeps: {
    exclude: ['lattice-db']  // Don't transform WASM
  }
};
```

## Browser Compatibility

| Browser | WASM | SIMD | OPFS | Service Workers |
|---------|------|------|------|-----------------|
| Chrome 89+ | ✅ | ✅ | ✅ | ✅ |
| Firefox 89+ | ✅ | ✅ | ✅ | ✅ |
| Safari 15+ | ✅ | ✅ | ✅ | ✅ |
| Edge 89+ | ✅ | ✅ | ✅ | ✅ |

## Debugging

### Memory Usage

Monitor WASM memory consumption:

```javascript
const stats = db.memoryStats();
console.log(`Vectors: ${stats.vectorBytes} bytes`);
console.log(`Index: ${stats.indexBytes} bytes`);
console.log(`Total: ${stats.totalBytes} bytes`);
```

### Performance Profiling

Use the browser's Performance API:

```javascript
performance.mark('search-start');
const results = await db.search(query);
performance.mark('search-end');
performance.measure('search', 'search-start', 'search-end');

const [measure] = performance.getEntriesByName('search');
console.log(`Search took ${measure.duration}ms`);
```

## Next Steps

- [Architecture Overview](../architecture/overview.md) - How LatticeDB works internally
- [Performance Tuning](../performance/tuning.md) - Optimization strategies
