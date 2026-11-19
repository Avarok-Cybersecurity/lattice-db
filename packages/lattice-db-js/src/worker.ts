/**
 * Cloudflare Worker host: serve a read-mostly corpus from a binary snapshot.
 *
 * The snapshot is loaded once per isolate, lazily, on the first request under
 * the prefix; later requests reuse it. A failed load is answered with 503 and
 * retried on the next request rather than cached.
 *
 * @example
 * ```typescript
 * import wasm from 'lattice-db/wasm/lattice_server_bg.wasm';
 * import { createLatticeHandler } from 'lattice-db';
 *
 * const lattice = createLatticeHandler({
 *   wasm, snapshotUrl: '/corpus.lattice', collection: 'docs',
 *   prefix: '/api/lattice', readOnly: true,
 * });
 * export default {
 *   async fetch(request: Request, env: LatticeWorkerEnv): Promise<Response> {
 *     return (await lattice.fetch(request, env)) ?? env.ASSETS!.fetch(request);
 *   },
 * };
 * ```
 */

import { LatticeHost, type LatticeFetchHandler } from './host';
import type { Embedder } from './text-query';
import type { ByteInput, WasmInput } from './wasm-types';

/** The bindings this handler reads from a Worker's `env`. */
export interface LatticeWorkerEnv {
  /** Static-assets binding; required when using `snapshotUrl`. */
  ASSETS?: { fetch(request: Request): Promise<Response> };
}

interface CommonOptions {
  /** Precompiled module (Workers cannot compile WASM from bytes at runtime). */
  wasm: WasmInput;
  /** Collection name the snapshot is installed under. */
  collection: string;
  /** Mount path such as `/api/lattice`, or `''` for every path. */
  prefix: string;
  /** Refuse mutating endpoints with 403. */
  readOnly: boolean;
  /** `Access-Control-Allow-Origin` value; omit for same-origin use. */
  corsOrigin?: string;
  /** Embeds text queries; without it they are answered with 501. */
  embed?: Embedder;
}

/** Load the snapshot from the Worker's static assets (resolved against the request URL). */
export interface SnapshotFromAssets extends CommonOptions {
  snapshotUrl: string;
  snapshotBytes?: never;
}

/** Use snapshot bytes already in hand (e.g. bundled). */
export interface SnapshotFromBytes extends CommonOptions {
  snapshotBytes: ByteInput;
  snapshotUrl?: never;
}

export type LatticeWorkerOptions = SnapshotFromAssets | SnapshotFromBytes;

export interface LatticeWorkerHandler {
  /**
   * Answer a request under the prefix, or resolve to `null` when the request
   * is outside it (so the Worker can fall through to its assets).
   */
  fetch(request: Request, env: LatticeWorkerEnv): Promise<Response | null>;
}

export function createLatticeHandler(options: LatticeWorkerOptions): LatticeWorkerHandler {
  let mounted: Promise<Mounted> | null = null;
  let corpus: Promise<void> | null = null;

  // Phase 1 (cheap, once): instantiate WASM and the handler — enough to match the prefix.
  const mount = (): Promise<Mounted> => {
    mounted ??= mountHandler(options).catch((error: unknown) => {
      mounted = null;
      throw error;
    });
    return mounted;
  };

  // Phase 2 (once): load the corpus into the same database.
  const load = (m: Mounted, request: Request, env: LatticeWorkerEnv): Promise<void> => {
    corpus ??= loadCorpus(options, m.host, request, env).catch((error: unknown) => {
      corpus = null;
      throw error;
    });
    return corpus;
  };

  return {
    async fetch(request: Request, env: LatticeWorkerEnv): Promise<Response | null> {
      const m = await mount();
      if (!m.api.handles(request.url)) return null;
      try {
        await load(m, request, env);
      } catch (error: unknown) {
        const reason = error instanceof Error ? error.message : String(error);
        return m.api.errorResponse(503, `LatticeDB corpus unavailable: ${reason}`);
      }
      return m.api.handleFetch(request);
    },
  };
}

interface Mounted {
  host: LatticeHost;
  api: LatticeFetchHandler;
}

async function mountHandler(options: LatticeWorkerOptions): Promise<Mounted> {
  const host = await LatticeHost.create({ readOnly: options.readOnly, wasm: options.wasm });
  const api = host.fetchHandler({
    prefix: options.prefix,
    corsOrigin: options.corsOrigin,
    embed: options.embed,
  });
  return { host, api };
}

async function loadCorpus(
  options: LatticeWorkerOptions,
  host: LatticeHost,
  request: Request,
  env: LatticeWorkerEnv
): Promise<void> {
  const bytes = options.snapshotBytes ?? (await fetchSnapshot(options.snapshotUrl, request, env));
  host.importSnapshot(bytes, options.collection, 'create');
}

async function fetchSnapshot(
  url: string | undefined,
  request: Request,
  env: LatticeWorkerEnv
): Promise<Uint8Array> {
  if (url === undefined) throw new Error('snapshotUrl or snapshotBytes is required');
  if (env.ASSETS === undefined) throw new Error('env.ASSETS binding is required for snapshotUrl');
  const response = await env.ASSETS.fetch(new Request(new URL(url, request.url)));
  if (!response.ok) throw new Error(`snapshot fetch ${url} returned ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}
