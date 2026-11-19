/**
 * Shared fixtures for tests that run the real WASM build in Node.
 * Requires `npm run build:wasm` (CI builds it before `npm test`).
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { LatticeHost } from '../host';

/** The built binary, read from disk: a Node `Buffer`, i.e. a pooled Uint8Array view. */
export function wasmBytes(): Uint8Array {
  return readFileSync(join(process.cwd(), 'wasm', 'lattice_server_bg.wasm'));
}

export const COLLECTION = 'docs';

/** Three orthogonal-ish 3-d points with titles. */
export const UPSERT = JSON.stringify({
  points: [
    { id: 1, vector: [1, 0, 0], payload: { title: 'alpha' } },
    { id: 2, vector: [0, 1, 0], payload: { title: 'beta' } },
    { id: 3, vector: [0, 0, 1], payload: { title: 'gamma' } },
  ],
});

export const SEARCH_BETA = JSON.stringify({ vector: [0, 1, 0], limit: 1 });

/** A writable host holding the seeded collection. */
export async function seededHost(): Promise<LatticeHost> {
  const host = await LatticeHost.create({ readOnly: false, wasm: wasmBytes() });
  const created = await host.route(
    'PUT',
    `/collections/${COLLECTION}`,
    JSON.stringify({ vectors: { size: 3, distance: 'Cosine' } })
  );
  expectStatus(created.status, 200);
  expectStatus((await host.route('PUT', `/collections/${COLLECTION}/points`, UPSERT)).status, 200);
  return host;
}

/** Snapshot bytes of the seeded collection. */
export async function seededSnapshot(): Promise<Uint8Array> {
  return (await seededHost()).exportSnapshot(COLLECTION);
}

export function decodeJson(body: Uint8Array): unknown {
  return JSON.parse(new TextDecoder().decode(body));
}

/** First result id of a search/query response body. */
export function firstId(json: unknown): number | undefined {
  if (typeof json !== 'object' || json === null) return undefined;
  const result = (json as { result?: unknown }).result;
  const list = Array.isArray(result)
    ? result
    : (result as { points?: unknown } | undefined)?.points;
  if (!Array.isArray(list) || list.length === 0) return undefined;
  const id = (list[0] as { id?: unknown }).id;
  return typeof id === 'number' ? id : undefined;
}

function expectStatus(actual: number, expected: number): void {
  if (actual !== expected) throw new Error(`expected status ${expected}, got ${actual}`);
}
