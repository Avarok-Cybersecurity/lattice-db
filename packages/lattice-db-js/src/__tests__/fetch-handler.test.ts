/**
 * @jest-environment node
 *
 * The service-worker helper with real Request/Response objects and the real
 * WASM build. A service worker calls exactly these two methods from its
 * `fetch` listener (see examples/service-worker.js).
 */

import { describe, it, expect } from '@jest/globals';
import type { EmbedFunction } from '../wasm-types';
import { COLLECTION, SEARCH_BETA, firstId, seededHost, textHost } from './fixtures';

const ORIGIN = 'https://site.example';
const BASE = `${ORIGIN}/api/lattice`;

function post(path: string, body: string): Request {
  return new Request(`${BASE}${path}`, {
    method: 'POST',
    body,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('fetchHandler', () => {
  it('claims only URLs under its prefix, on a segment boundary', async () => {
    const api = (await seededHost()).fetchHandler({ prefix: '/api/lattice' });
    expect(api.prefix).toBe('/api/lattice');
    expect(api.handles(`${BASE}/collections`)).toBe(true);
    expect(api.handles(BASE)).toBe(true);
    expect(api.handles(`${ORIGIN}/api/latticex/collections`)).toBe(false);
    expect(api.handles(`${ORIGIN}/index.html`)).toBe(false);
    expect(api.handles('not a url')).toBe(false);
  });

  it('turns a Request into a routed Response', async () => {
    const api = (await seededHost()).fetchHandler({ prefix: '/api/lattice' });
    const res = await api.handleFetch(post(`/collections/${COLLECTION}/points/search`, SEARCH_BETA));
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toBe('application/json');
    expect(res.headers.get('x-content-type-options')).toBe('nosniff');
    expect(res.headers.get('access-control-allow-origin')).toBeNull();
    expect(firstId(await res.json())).toBe(2);

    const list = await api.handleFetch(new Request(`${BASE}/collections`));
    expect(list.status).toBe(200);
    const outside = await api.handleFetch(new Request(`${ORIGIN}/elsewhere`));
    expect(outside.status).toBe(404);
  });

  it('serves the binary export with its own content type', async () => {
    const api = (await seededHost()).fetchHandler({ prefix: '/api/lattice' });
    const res = await api.handleFetch(new Request(`${BASE}/collections/${COLLECTION}/export`));
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toBe('application/octet-stream');
    expect((await res.arrayBuffer()).byteLength).toBeGreaterThan(16);
  });

  it('answers CORS preflights and tags responses when configured', async () => {
    const api = (await seededHost()).fetchHandler({ prefix: '/api/lattice', corsOrigin: '*' });
    const pre = await api.handleFetch(new Request(`${BASE}/collections`, { method: 'OPTIONS' }));
    expect(pre.status).toBe(204);
    expect(pre.headers.get('access-control-allow-methods')).toMatch(/POST/);
    const res = await api.handleFetch(new Request(`${BASE}/ping`));
    expect(res.headers.get('access-control-allow-origin')).toBe('*');
  });

  it('rejects bad options up front', async () => {
    const host = await seededHost();
    expect(() => host.fetchHandler({ prefix: '/api/' })).toThrow(/prefix/);
    expect(() => host.fetchHandler({ prefix: 'api' })).toThrow(/prefix/);
  });
});

describe('text queries are answered by the router', () => {
  const path = `/collections/${COLLECTION}/points/query`;
  const query = (q: unknown): string => JSON.stringify({ query: q, limit: 1 });

  it('answers bm25 queries with no embedder at all', async () => {
    const api = (await textHost()).fetchHandler({ prefix: '/api/lattice' });
    const res = await api.handleFetch(post(path, query({ text: 'beta', model: 'bm25' })));
    expect(res.status).toBe(200);
    expect(firstId(await res.json())).toBe(2);
    const nested = query({ nearest: { text: 'gamma', model: 'bm25' } });
    expect(firstId(await (await api.handleFetch(post(path, nested))).json())).toBe(3);
  });

  it('refuses a dense text query whose model has no registered embedder', async () => {
    const api = (await textHost()).fetchHandler({ prefix: '/api/lattice', corsOrigin: '*' });
    const res = await api.handleFetch(post(path, query({ text: 'beta', model: 'm' })));
    expect(res.status).toBe(503);
    expect(res.headers.get('access-control-allow-origin')).toBe('*');
    expect(((await res.json()) as { message: string }).message).toMatch(/no embedder/);
    const noModel = await api.handleFetch(post(path, query({ text: 'beta' })));
    expect(noModel.status).toBe(400);
  });

  it('embeds through an embedder registered on the host, for every handler', async () => {
    const calls: string[][] = [];
    const embed: EmbedFunction = async (texts) => {
      calls.push(texts);
      return texts.map(() => new Float32Array([0, 1, 0]));
    };
    const host = await textHost();
    const api = host.fetchHandler({ prefix: '/api/lattice' });
    host.setEmbedder({ model: 'm', dim: 3, embed });
    const res = await api.handleFetch(post(path, query({ text: 'anything', model: 'm' })));
    expect(res.status).toBe(200);
    expect(firstId(await res.json())).toBe(2);
    expect(calls).toEqual([['anything']]);

    const hybrid = JSON.stringify({
      prefetch: [
        { query: { text: 'gamma', model: 'bm25' }, limit: 3 },
        { query: { text: 'gamma', model: 'm' }, limit: 3 },
      ],
      query: { fusion: 'rrf' },
      limit: 3,
    });
    const fused = await api.handleFetch(post(path, hybrid));
    expect(fused.status).toBe(200);
    expect(calls).toHaveLength(2);
  });

  it('rejects an embedder returning the wrong dimension with 502', async () => {
    const host = await textHost();
    host.setEmbedder({ model: 'm', dim: 3, embed: (texts) => texts.map(() => [1, 0]) });
    const api = host.fetchHandler({ prefix: '/api/lattice' });
    const res = await api.handleFetch(post(path, query({ text: 'x', model: 'm' })));
    expect(res.status).toBe(502);
  });

  it('refuses to register the reserved bm25 model', async () => {
    const host = await textHost();
    expect(() => host.setEmbedder({ model: 'bm25', dim: 3, embed: () => [] })).toThrow(/reserved/);
  });
});
