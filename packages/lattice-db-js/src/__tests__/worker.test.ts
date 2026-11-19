/**
 * @jest-environment node
 *
 * The Cloudflare Worker handler with real Request/Response and the real WASM
 * build. `env.ASSETS` is a stand-in for the Workers static-assets binding — the
 * one piece of platform I/O that does not exist in Node — and records calls so
 * the lazy, once-per-isolate load is observable.
 */

import { describe, it, expect } from '@jest/globals';
import { createLatticeHandler, type LatticeWorkerEnv } from '../worker';
import { COLLECTION, SEARCH_BETA, firstId, seededSnapshot, textSnapshot, wasmBytes } from './fixtures';

const BASE = 'https://site.example/api/lattice';

function assets(serve: () => Response): { env: LatticeWorkerEnv; urls: string[] } {
  const urls: string[] = [];
  const env: LatticeWorkerEnv = {
    ASSETS: {
      fetch: async (request: Request): Promise<Response> => {
        urls.push(request.url);
        return serve();
      },
    },
  };
  return { env, urls };
}

function search(): Request {
  return new Request(`${BASE}/collections/${COLLECTION}/points/search`, {
    method: 'POST',
    body: SEARCH_BETA,
  });
}

const common = {
  collection: COLLECTION,
  prefix: '/api/lattice',
  readOnly: true,
} as const;

describe('createLatticeHandler', () => {
  it('loads the snapshot from ASSETS once, lazily, and serves reads', async () => {
    const snapshot = await seededSnapshot();
    const { env, urls } = assets(() => new Response(new Uint8Array(snapshot)));
    const lattice = createLatticeHandler({ ...common, wasm: wasmBytes(), snapshotUrl: '/c.lattice' });

    expect(await lattice.fetch(new Request('https://site.example/index.html'), env)).toBeNull();
    expect(urls).toEqual([]);

    const [a, b] = await Promise.all([lattice.fetch(search(), env), lattice.fetch(search(), env)]);
    expect(a?.status).toBe(200);
    expect(b?.status).toBe(200);
    expect(firstId(await a?.json())).toBe(2);
    expect(urls).toEqual(['https://site.example/c.lattice']);

    const write = new Request(`${BASE}/collections/${COLLECTION}`, { method: 'DELETE' });
    expect((await lattice.fetch(write, env))?.status).toBe(403);
    expect(urls).toHaveLength(1);
  });

  it('answers 503 on a failed load and retries on the next request', async () => {
    const snapshot = await seededSnapshot();
    let ok = false;
    const { env, urls } = assets(() =>
      ok ? new Response(new Uint8Array(snapshot)) : new Response('gone', { status: 404 })
    );
    const lattice = createLatticeHandler({
      ...common,
      collection: 'retry',
      wasm: wasmBytes(),
      snapshotUrl: '/c.lattice',
    });
    const req = () => new Request(`${BASE}/collections/retry`);
    const failed = await lattice.fetch(req(), env);
    expect(failed?.status).toBe(503);
    expect(await failed?.text()).toMatch(/returned 404/);
    ok = true;
    expect((await lattice.fetch(req(), env))?.status).toBe(200);
    expect(urls).toHaveLength(2);
  });

  it('requires the ASSETS binding for snapshotUrl', async () => {
    const lattice = createLatticeHandler({
      ...common,
      collection: 'noassets',
      wasm: wasmBytes(),
      snapshotUrl: '/c.lattice',
    });
    const res = await lattice.fetch(new Request(`${BASE}/collections`), {});
    expect(res?.status).toBe(503);
    expect(await res?.text()).toMatch(/ASSETS/);
  });

  it('serves bundled snapshot bytes; a dense text query without an embedder is 503', async () => {
    const lattice = createLatticeHandler({
      ...common,
      collection: 'bundled',
      wasm: wasmBytes(),
      snapshotBytes: await textSnapshot(),
    });
    const query = new Request(`${BASE}/collections/bundled/points/query`, {
      method: 'POST',
      body: JSON.stringify({ query: { text: 'hello', model: 'm' } }),
    });
    expect((await lattice.fetch(query, {}))?.status).toBe(503);
  });

  it('ready() installs bundled bytes before any request', async () => {
    const lattice = createLatticeHandler({
      ...common,
      collection: 'eager',
      wasm: wasmBytes(),
      snapshotBytes: await seededSnapshot(),
    });
    await lattice.ready();
    const res = await lattice.fetch(
      new Request(`${BASE}/collections/eager/points/search`, { method: 'POST', body: SEARCH_BETA }),
      {}
    );
    expect(firstId(await res?.json())).toBe(2);
    const lazy = createLatticeHandler({ ...common, wasm: wasmBytes(), snapshotUrl: '/c.lattice' });
    await expect(lazy.ready()).rejects.toThrow(/snapshotBytes/);
  });

  it('registers embedders passed in the options', async () => {
    const lattice = createLatticeHandler({
      ...common,
      collection: 'embedded',
      wasm: wasmBytes(),
      snapshotBytes: await textSnapshot(),
      embedders: [{ model: 'm', dim: 3, embed: (texts) => texts.map(() => [0, 0, 1]) }],
    });
    const query = new Request(`${BASE}/collections/embedded/points/query`, {
      method: 'POST',
      body: JSON.stringify({ query: { text: 'hello', model: 'm' }, limit: 1 }),
    });
    const res = await lattice.fetch(query, {});
    expect(res?.status).toBe(200);
    expect(firstId(await res?.json())).toBe(3);
  });
});
