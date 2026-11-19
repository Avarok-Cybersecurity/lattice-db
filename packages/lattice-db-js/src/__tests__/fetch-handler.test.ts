/**
 * @jest-environment node
 *
 * The service-worker helper with real Request/Response objects and the real
 * WASM build. A service worker calls exactly these two methods from its
 * `fetch` listener (see examples/service-worker.js).
 */

import { describe, it, expect } from '@jest/globals';
import type { Embedder } from '../text-query';
import { COLLECTION, SEARCH_BETA, firstId, seededHost } from './fixtures';

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

describe('text queries', () => {
  const textQuery = JSON.stringify({ query: { text: 'second', model: 'm' }, limit: 1 });
  const path = `/collections/${COLLECTION}/points/query`;

  it('answers 501 when no embedder is configured', async () => {
    const api = (await seededHost()).fetchHandler({ prefix: '/api/lattice', corsOrigin: '*' });
    const res = await api.handleFetch(post(path, textQuery));
    expect(res.status).toBe(501);
    expect(res.headers.get('access-control-allow-origin')).toBe('*');
    expect(((await res.json()) as { message: string }).message).toMatch(/embedder/);
  });

  it('embeds through the configured hook and forwards a vector query', async () => {
    const calls: Array<[string, string[]]> = [];
    const embed: Embedder = async (model, texts) => {
      calls.push([model, texts]);
      return texts.map(() => new Float32Array([0, 1, 0]));
    };
    const api = (await seededHost()).fetchHandler({ prefix: '/api/lattice', embed });
    const res = await api.handleFetch(post(path, textQuery));
    expect(res.status).toBe(200);
    expect(firstId(await res.json())).toBe(2);
    expect(calls).toEqual([['m', ['second']]]);

    const nested = JSON.stringify({ query: { nearest: { text: 'x', model: 'm' } }, limit: 1 });
    expect((await api.handleFetch(post(path, nested))).status).toBe(200);
    const noModel = JSON.stringify({ query: { text: 'x' }, limit: 1 });
    expect((await api.handleFetch(post(path, noModel))).status).toBe(400);
  });

  it('leaves vector queries untouched', async () => {
    const embed: Embedder = async () => {
      throw new Error('must not be called');
    };
    const api = (await seededHost()).fetchHandler({ prefix: '/api/lattice', embed });
    const res = await api.handleFetch(post(path, JSON.stringify({ query: [1, 0, 0], limit: 1 })));
    expect(res.status).toBe(200);
    expect(firstId(await res.json())).toBe(1);
  });
});
