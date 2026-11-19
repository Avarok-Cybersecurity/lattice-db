// LatticeDB on Cloudflare Workers: static site from ASSETS, search API under
// /api/lattice served from a snapshot shipped alongside the site.
//
// wrangler.jsonc:
//   { "main": "cloudflare-worker.ts", "assets": { "directory": "./public", "binding": "ASSETS" } }
// public/corpus.lattice is produced by `GET /collections/docs/export` (or
// LatticeHost.exportSnapshot) at build time.

import wasm from 'lattice-db/wasm/lattice_server_bg.wasm';
import { createLatticeHandler, type LatticeWorkerEnv } from 'lattice-db';

const lattice = createLatticeHandler({
  wasm,
  snapshotUrl: '/corpus.lattice',
  collection: 'docs',
  prefix: '/api/lattice',
  readOnly: true,
  // embed: (model, texts) => ...  // e.g. Workers AI; without it text queries get 501
});

interface Env extends LatticeWorkerEnv {
  ASSETS: { fetch(request: Request): Promise<Response> };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return (await lattice.fetch(request, env)) ?? env.ASSETS.fetch(request);
  },
};
