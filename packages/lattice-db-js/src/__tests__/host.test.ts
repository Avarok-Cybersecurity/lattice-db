/**
 * @jest-environment node
 *
 * LatticeHost against the real WASM build: snapshot/JSONL import and route().
 */

import { describe, it, expect } from '@jest/globals';
import { LatticeHost } from '../host';
import {
  COLLECTION,
  SEARCH_BETA,
  decodeJson,
  firstId,
  seededHost,
  seededSnapshot,
  wasmBytes,
} from './fixtures';

const JSONL = [
  JSON.stringify({
    t: 'header',
    format: 'lattice-jsonl',
    version: 1,
    config: {
      name: 'site',
      vectors: { size: 2, distance: 'cosine' },
      hnsw: { m: 16, m0: 32, ml: 0.36, ef: 64, ef_construction: 100 },
    },
    vectors: 'inline',
    dim: 2,
    points: 2,
    edges: 1,
  }),
  JSON.stringify({ t: 'point', id: 1, vector: [1, 0], payload: { title: 'one' } }),
  JSON.stringify({ t: 'point', id: 2, vector: [0, 1] }),
  JSON.stringify({ t: 'edge', from: 1, to: 2, rel: 'links', w: 0.5 }),
].join('\n');

describe('route()', () => {
  it('drives the native router with string, byte and absent bodies', async () => {
    const host = await seededHost();
    const ping = await host.route('get', '/ping');
    expect(ping.status).toBe(200);
    expect(ping.headers['Content-Type']).toBe('application/json');

    const bytes = new TextEncoder().encode(SEARCH_BETA);
    const search = await host.route('POST', `/collections/${COLLECTION}/points/search`, bytes);
    expect(search.status).toBe(200);
    expect(search.body).toBeInstanceOf(Uint8Array);
    expect(firstId(decodeJson(search.body))).toBe(2);

    expect((await host.route('GET', '/nope')).status).toBe(404);
    expect((await host.route('PUT', `/collections/${COLLECTION}`, '{')).status).toBe(400);
  });

  it('rejects invalid arguments instead of guessing', async () => {
    const host = await seededHost();
    await expect(host.route('', '/ping')).rejects.toMatch(/method/);
    await expect(host.route('GET', 'ping')).rejects.toMatch(/must start/);
    const notBytes = 42 as unknown as Uint8Array;
    await expect(host.route('POST', '/ping', notBytes)).rejects.toMatch(/Expected/);
  });
});

describe('snapshots', () => {
  it('round-trips into a read-only host, accepting a view at an offset', async () => {
    const snapshot = await seededSnapshot();
    const padded = new Uint8Array(snapshot.length + 13);
    padded.set(snapshot, 7);
    const view = padded.subarray(7, 7 + snapshot.length);

    const host = await LatticeHost.create({ readOnly: true, wasm: wasmBytes() });
    const result = host.importSnapshot(view, 'copy', 'create');
    expect(result.points_imported).toBe(3);
    const search = await host.route('POST', '/collections/copy/points/search', SEARCH_BETA);
    expect(firstId(decodeJson(search.body))).toBe(2);
    expect(host.exportSnapshot('copy').length).toBe(snapshot.length);
  });

  it('accepts a DataView and a bare ArrayBuffer, refuses text', async () => {
    const snapshot = await seededSnapshot();
    const host = await LatticeHost.create({ readOnly: false, wasm: wasmBytes() });
    const copy = snapshot.slice();
    host.importSnapshot(new DataView(copy.buffer), 'a', 'create');
    host.importSnapshot(snapshot.slice().buffer, 'b', 'create');
    const text = 'nope' as unknown as Uint8Array;
    expect(() => host.importSnapshot(text, 'c', 'create')).toThrow(/Expected/);
    expect(() => host.importSnapshot(snapshot, 'a', 'create')).toThrow(/already exists/);
    expect(() => host.importSnapshot(snapshot, 'a', 'bogus' as 'create')).toThrow(/mode/);
  });

  it('exportSnapshot throws for a missing collection', async () => {
    const host = await LatticeHost.create({ readOnly: false, wasm: wasmBytes() });
    expect(() => host.exportSnapshot('missing')).toThrow(/not found/);
  });
});

describe('importJsonl', () => {
  it('installs the dump under its own name, from text or bytes', async () => {
    const host = await LatticeHost.create({ readOnly: true, wasm: wasmBytes() });
    expect(host.importJsonl(JSONL, 'create').points_imported).toBe(2);
    const traverse = await host.route(
      'POST',
      '/collections/site/graph/traverse',
      JSON.stringify({ start_id: 1, max_depth: 1 })
    );
    expect(traverse.status).toBe(200);
    expect(() => host.importJsonl(new TextEncoder().encode(JSONL), 'create')).toThrow(/exists/);
    expect(host.importJsonl(new TextEncoder().encode(JSONL), 'replace').points_imported).toBe(2);
  });

  it('refuses a truncated dump', async () => {
    const host = await LatticeHost.create({ readOnly: false, wasm: wasmBytes() });
    const cut = JSONL.split('\n').slice(0, 2).join('\n');
    expect(() => host.importJsonl(cut, 'create')).toThrow(/truncated/);
  });
});

describe('read-only', () => {
  it('refuses REST writes with 403 while serving reads', async () => {
    const host = await LatticeHost.create({ readOnly: true, wasm: wasmBytes() });
    host.importSnapshot(await seededSnapshot(), COLLECTION, 'create');
    const write = await host.route('PUT', `/collections/${COLLECTION}/points`, '{"points":[]}');
    expect(write.status).toBe(403);
    expect(new TextDecoder().decode(write.body)).toMatch(/Read-only/);
    expect((await host.route('DELETE', `/collections/${COLLECTION}`)).status).toBe(403);
    const cypher = JSON.stringify({ query: 'MATCH (n) RETURN n.id' });
    expect((await host.route('POST', `/collections/${COLLECTION}/graph/query`, cypher)).status).toBe(
      200
    );
  });
});
