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
  </script>
</body>
</html>
```

### Service Worker

> **Note**: Service Worker transport (fetch interception) is not yet implemented.
> The examples below show the planned API. Currently, use the direct
> LatticeDB API from the main thread or a Web Worker.

For persistent storage and background processing:

```javascript
// sw.js - Service Worker
import init, { LatticeDB } from './lattice_db.js';

let db = null;

self.addEventListener('install', async (event) => {
  event.waitUntil(async () => {
    await init();
    db = await LatticeDB.create({
      name: 'persistent_collection',
      vectorSize: 128,
      storage: 'opfs'  // Use Origin Private File System
    });
  });
});

self.addEventListener('message', async (event) => {
  const { type, data } = event.data;

  switch (type) {
    case 'upsert':
      await db.upsert(data);
      event.ports[0].postMessage({ success: true });
      break;
    case 'search':
      const results = await db.search(data);
      event.ports[0].postMessage({ results });
      break;
  }
});
```

Register the service worker:

```javascript
// main.js
if ('serviceWorker' in navigator) {
  const registration = await navigator.serviceWorker.register('/sw.js');

  // Send messages to the service worker
  async function search(vector) {
    const channel = new MessageChannel();
    registration.active.postMessage(
      { type: 'search', data: { vector, limit: 10 } },
      [channel.port2]
    );
    return new Promise(resolve => {
      channel.port1.onmessage = (e) => resolve(e.data.results);
    });
  }
}
```

## Storage Options

### In-Memory (Default)

Fast but not persistent across page reloads:

```javascript
const db = await LatticeDB.create({
  name: 'temp_collection',
  storage: 'memory'
});
```

### Origin Private File System (OPFS)

Persistent browser storage with high performance:

```javascript
const db = await LatticeDB.create({
  name: 'persistent_collection',
  storage: 'opfs'
});
```

OPFS provides:
- Persistent storage across sessions
- Better performance than IndexedDB
- File system-like API
- Available in modern browsers (Chrome 86+, Firefox 111+, Safari 15.2+)

### IndexedDB Fallback

For older browsers without OPFS:

```javascript
const db = await LatticeDB.create({
  name: 'fallback_collection',
  storage: 'indexeddb'
});
```

## Framework Integration

### React

```jsx
import { useEffect, useState } from 'react';
import init, { LatticeDB } from '@lattice-db/client';

function useLatticeDB(config) {
  const [db, setDb] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function initialize() {
      await init();
      const instance = await LatticeDB.create(config);
      setDb(instance);
      setLoading(false);
    }
    initialize();
  }, []);

  return { db, loading };
}

function SearchComponent() {
  const { db, loading } = useLatticeDB({
    name: 'my_collection',
    vectorSize: 128
  });
  const [results, setResults] = useState([]);

  const handleSearch = async (query) => {
    if (!db) return;
    const embedding = await getEmbedding(query); // Your embedding function
    const searchResults = await db.search({ vector: embedding, limit: 10 });
    setResults(searchResults);
  };

  if (loading) return <div>Loading...</div>;

  return (
    <div>
      <input onChange={(e) => handleSearch(e.target.value)} />
      <ul>
        {results.map(r => <li key={r.id}>{r.payload.title}</li>)}
      </ul>
    </div>
  );
}
```

### Vue

```vue
<script setup>
import { ref, onMounted } from 'vue';
import init, { LatticeDB } from '@lattice-db/client';

const db = ref(null);
const results = ref([]);

onMounted(async () => {
  await init();
  db.value = await LatticeDB.create({
    name: 'my_collection',
    vectorSize: 128
  });
});

async function search(query) {
  const embedding = await getEmbedding(query);
  results.value = await db.value.search({ vector: embedding, limit: 10 });
}
</script>

<template>
  <input @input="search($event.target.value)" />
  <ul>
    <li v-for="r in results" :key="r.id">{{ r.payload.title }}</li>
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
// Good - single WASM call
await db.upsertBatch(points);

// Bad - multiple WASM calls
for (const point of points) {
  await db.upsert(point);
}
```

### 3. Web Workers

Offload heavy operations to a Web Worker:

```javascript
// worker.js
import init, { LatticeDB } from '@lattice-db/client';

let db;

self.onmessage = async ({ data }) => {
  if (data.type === 'init') {
    await init();
    db = await LatticeDB.create(data.config);
    self.postMessage({ type: 'ready' });
  }

  if (data.type === 'search') {
    const results = await db.search(data.query);
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
    exclude: ['@lattice-db/client']  // Don't transform WASM
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
