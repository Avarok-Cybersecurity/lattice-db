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
import type { ByteInput, EmbedderRegistration, WasmInput } from './wasm-types';

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
  /** Text embedders for dense text queries; `bm25` queries need none. */
  embedders?: readonly EmbedderRegistration[];
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
   * Instantiate WASM and install `snapshotBytes` now. Await it at module top
   * level so the work counts against the Worker's startup budget instead of
   * the first request's CPU time. Rejects for `snapshotUrl` (no I/O is
   * possible outside a request).
   */
  ready(): Promise<void>;
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
  const load = (m: Mounted, bytes: () => Promise<ByteInput>): Promise<void> => {
    corpus ??= bytes()
      .then((b) => {
        m.host.importSnapshot(b, options.collection, 'create');
      })
      .catch((error: unknown) => {
        corpus = null;
        throw error;
      });
    return corpus;
  };

  return {
    async ready(): Promise<void> {
      const bundled = options.snapshotBytes;
      if (bundled === undefined) {
        throw new Error('ready() needs snapshotBytes; snapshotUrl loads on the first request');
      }
      await load(await mount(), async () => bundled);
    },
    async fetch(request: Request, env: LatticeWorkerEnv): Promise<Response | null> {
      const m = await mount();
      if (!m.api.handles(request.url)) return null;
      try {
        await load(m, () => snapshotSource(options, request, env));
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
  const host = await LatticeHost.create({
    readOnly: options.readOnly,
    wasm: options.wasm,
    embedders: options.embedders,
  });
  const api = host.fetchHandler({ prefix: options.prefix, corsOrigin: options.corsOrigin });
  return { host, api };
}

function snapshotSource(
  options: LatticeWorkerOptions,
  request: Request,
  env: LatticeWorkerEnv
): Promise<ByteInput> {
  const bundled = options.snapshotBytes;
  return bundled === undefined
    ? fetchSnapshot(options.snapshotUrl, request, env)
    : Promise.resolve(bundled);
}

async function fetchSnapshot(
  url: string | undefined,
  request: Request,
  env: LatticeWorkerEnv
): Promise<ByteInput> {
  if (url === undefined) throw new Error('snapshotUrl or snapshotBytes is required');
  if (env.ASSETS === undefined) throw new Error('env.ASSETS binding is required for snapshotUrl');
  const response = await env.ASSETS.fetch(new Request(new URL(url, request.url)));
  if (!response.ok) throw new Error(`snapshot fetch ${url} returned ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}
