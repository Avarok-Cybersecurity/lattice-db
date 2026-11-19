// LatticeDB on Cloudflare Workers: static site from ASSETS, search API under
// /api/lattice, answered from a snapshot bundled into the Worker.
//
// wrangler.jsonc:
//   {
//     "main": "cloudflare-worker.ts",
//     "assets": { "directory": "./public", "binding": "ASSETS" },
//     "rules": [{ "type": "Data", "globs": ["**/*.lattice"], "fallthrough": false }]
//   }
//
// corpus.lattice is produced at build time by `GET /collections/docs/export`
// (or LatticeHost.exportSnapshot). A Data module is an ArrayBuffer available
// at global scope, so the snapshot is installed during Worker startup (1 s
// budget) instead of in the first request (10 ms CPU on the Free plan). It
// counts toward the 64 MiB Worker size limit; 2K chunks x 768-d is 5-11 MiB
// depending on `vectors.storage` (see docs/edge-search.md).
//
// Snapshots larger than the bundle can carry can be served from static assets
// instead (`snapshotUrl: '/corpus.lattice'`, 25 MiB per asset); those load on
// the first request, since global scope cannot do I/O.

import wasm from 'lattice-db/wasm/lattice_server_bg.wasm';
import corpus from './corpus.lattice';
import { createLatticeHandler, type LatticeWorkerEnv } from 'lattice-db';

const lattice = createLatticeHandler({
  wasm,
  snapshotBytes: new Uint8Array(corpus as ArrayBuffer),
  collection: 'docs',
  prefix: '/api/lattice',
  readOnly: true,
  // Only needed for dense text queries ({"text", "model": "<collection model>"});
  // BM25 queries ({"model": "bm25"}) need no embedder. For example, Workers AI:
  // embedders: [{ model: '@cf/baai/bge-base-en-v1.5', dim: 768, embed: (texts) => ... }],
});

// Instantiate WASM, install the snapshot and compile the query path now.
await lattice.ready({
  warmup: { query: JSON.stringify({ query: { text: 'warm up', model: 'bm25' } }), runs: 8 },
});

interface Env extends LatticeWorkerEnv {
  ASSETS: { fetch(request: Request): Promise<Response> };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return (await lattice.fetch(request, env)) ?? env.ASSETS.fetch(request);
  },
};
