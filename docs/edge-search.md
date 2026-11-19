# Site search at the edge

LatticeDB can answer search for a static site from a snapshot built at deploy
time. The snapshot can be served from a page, a service worker or a
Cloudflare Worker, all through the same Qdrant-compatible REST API. This page
covers the collection shapes, the queries, the hosts and the cold-load
numbers.

## Collection shapes

| Shape | Create body | Snapshot holds |
|---|---|---|
| Lexical only | `{"text_index": {"field": "text"}}` | payloads + BM25 postings |
| Flat, compact vectors | `{"vectors": {"size": 768, "distance": "Cosine", "index": "flat", "storage": "f16"}, "text_index": {...}}` | payloads + vectors + postings |
| HNSW | `{"vectors": {"size": 768, "distance": "Cosine"}, "text_index": {...}}` | payloads + vectors + graph + postings |

- **No `vectors`** means no dense index at all: only BM25 queries work. Vector
  search, vector queries, batch search and upserting a point with a vector
  each return `400` with a message saying the collection "has no dense
  vectors". An `embedding` model also requires `vectors`.
- **`index`**: `"hnsw"` (the default) or `"flat"`. Flat is an exact
  brute-force scan. It needs no graph to build or load, and at a few thousand
  points it costs about as much per query as HNSW.
- **`storage`**: `"f32"` (the default), `"f16"` or `"int8"`. Vectors are
  rounded to this precision when written: f16 keeps about 3 significant
  digits, and int8 uses a per-vector power-of-two scale. Snapshots then store
  the compact form losslessly. Search computes in f32.

## Queries

`POST /collections/{name}/points/query`:

```jsonc
// BM25, no embedder needed
{ "query": { "text": "free plan limits", "model": "bm25" }, "limit": 10 }

// Dense from text: needs an embedder registered for the collection's embedding.model
{ "query": { "text": "free plan limits", "model": "bge-base" }, "limit": 10 }

// Hybrid: BM25 + dense, fused with Reciprocal Rank Fusion (k = 60 unless {"rrf": {"k": n}})
{
  "prefetch": [
    { "query": { "text": "free plan limits", "model": "bm25" }, "limit": 20 },
    { "query": { "text": "free plan limits", "model": "bge-base" }, "limit": 20 }
  ],
  "query": { "fusion": "rrf" },
  "limit": 10
}
```

A database opened with `readOnly: true` answers every read (search, scroll,
BM25, dense-from-text and hybrid queries), and refuses writes with `403`.

## Hosts (JavaScript package)

Every host runs the same router; there is one way to add an embedder
(`setEmbedder` in WASM, which `embedders` options call):

```ts
import { LatticeHost } from 'lattice-db';

const host = await LatticeHost.create({
  readOnly: true,
  embedders: [{ model: 'bge-base', dim: 768, embed: async (texts) => myEmbed(texts) }],
});
host.importSnapshot(bytes, 'docs', 'create');
const api = host.fetchHandler({ prefix: '/api/lattice' }); // api.handleFetch(request)
```

- **Page**: `host.route(method, path, body)`.
- **Service worker**: `api.handleFetch(event.request)`; see
  `packages/lattice-db-js/examples/service-worker.js`.
- **Cloudflare Worker**: `createLatticeHandler({ wasm, snapshotBytes | snapshotUrl,
  collection, prefix, readOnly, embedders? })`; see
  `packages/lattice-db-js/examples/cloudflare-worker.ts`. With `snapshotBytes`
  (a bundled Data module), `await lattice.ready({ warmup: { query, runs } })`
  at module top level installs the snapshot during startup. It also compiles
  the query path, because V8 compiles WASM functions on first call. Without
  that warm-up, the first request pays the compile.

## Snapshot format

`GET /collections/{name}/export` writes format 2:

```
"LTDB" | u32 format = 2 | u32 config_len | config JSON | u64 body_len | pad to 16 | rkyv body
```

The body holds the points (without vectors), the vectors in their storage
encoding, the HNSW graph and the BM25 postings, all as columnar arrays, so a
load copies them instead of rebuilding indexes. Every restored reference is
validated, because the bytes are untrusted.

Format 1 snapshots, written by earlier releases, still load; their indexes are
rebuilt. A snapshot whose format is newer than the build fails with
`Unsupported snapshot format N: this build reads formats 1 and 2`.

## Cold load

The corpus is site-like chunks of 200 words with Zipf-like vocabulary, with
768-d vectors and a BM25 index on `text`. The query is BM25 alone (lexical) or
BM25 plus a dense vector fused with RRF (all others). Reproduce:

```sh
cd packages/lattice-db-js
npm run build:wasm
node scripts/bench/node.mjs --out /tmp/lattice-bench   # Node >= 22.18
node scripts/bench/workerd.mjs /tmp/lattice-bench      # wrangler dev, local only
```

Node 22 is single-threaded here, so wall time is effectively CPU time. These
are medians of 5 runs, release WASM, on an Apple M-series machine:

| config | chunks | snapshot MiB | load ms | first query ms | load + first ms | warm query ms |
|---|---|---|---|---|---|---|
| lexical | 1000 | 2.55 | 1.99 | 0.06 | 2.04 | 0.05 |
| flat-f32 | 1000 | 5.48 | 1.28 | 0.71 | 2.01 | 0.68 |
| flat-f16 | 1000 | 4.02 | 1.22 | 0.68 | 1.91 | 0.67 |
| flat-int8 | 1000 | 3.28 | 1.07 | 0.67 | 1.76 | 0.67 |
| hnsw | 1000 | 5.76 | 1.50 | 0.72 | 2.23 | 0.70 |
| lexical | 2000 | 5.04 | 1.65 | 0.07 | 1.71 | 0.05 |
| flat-f32 | 2000 | 10.90 | 1.88 | 1.29 | 3.33 | 1.23 |
| flat-f16 | 2000 | 7.97 | 2.28 | 1.25 | 3.55 | 1.21 |
| flat-int8 | 2000 | 6.50 | 2.08 | 1.38 | 3.49 | 1.25 |
| hnsw | 2000 | 11.45 | 2.48 | 1.49 | 4.16 | 1.30 |

In Node the module instance is already warm when these are taken.

In workerd (`wrangler dev`, local), each figure is the handler's own time
measured inside the isolate, and is the median of three fresh isolates. The
machine was heavily loaded (load average about 100), so treat these as upper
bounds; workerd also rounds to whole milliseconds.

| config | chunks | load + instantiate + query in request 1 | request 1 after `ready()` + warm-up | warm |
|---|---|---|---|---|
| lexical | 1000 / 2000 | 14 / 16 | 6 / 5 | 0-1 |
| flat-f16 | 1000 / 2000 | 16 / 22 | 6 / 6 | 1 |
| flat-int8 | 1000 / 2000 | 17 / 23 | 7 / 7 | 1 |
| flat-f32 | 1000 / 2000 | 17 / 25 | 7 / 7 | 1-2 |
| hnsw | 1000 / 2000 | 20 / 31 | 10 / 8 | 1-2 |

On an idle run, request 1 after `ready()` with 8 warm-up queries measured 3-4
ms for flat-f16 at 2K. Doing everything in request 1 does not fit the 10 ms
Free-plan CPU limit. Loading during startup does.

Timers do not advance at global scope, so startup cost could not be measured
in workerd. Node's "load" column (about 1-2.5 ms) plus WASM instantiation is
the estimate, well within the 1 s startup limit.

## Cloudflare Workers limits that apply

From <https://developers.cloudflare.com/workers/platform/limits/>, read
2026-11-19:

- **CPU per request**: 10 ms on Free. On Paid the default is 30 s, up to 5 min.
  Waiting on network requests does not count.
- **Startup**: a Worker must parse and run its global scope within 1 s.
- **Worker size**: 64 MiB uncompressed on Free and Paid. Only the uncompressed
  size counts, and that includes the bundled snapshot and the WASM (about 1.5
  MB).
- **Memory**: 128 MB per isolate, including the JavaScript heap and WebAssembly
  allocations.
- **Static assets**: 25 MiB per file, which is the `snapshotUrl` route.

Global scope cannot do I/O (fetch, timers or random values), so a snapshot
fetched from assets can only load inside a request. A bundled Data module is
already in memory at startup.
