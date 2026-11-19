/**
 * LatticeHost — serve the native REST API from a page, a service worker or a
 * Cloudflare Worker.
 *
 * @example Service worker
 * ```typescript
 * const host = await LatticeHost.create({ readOnly: true });
 * const snapshot = new Uint8Array(await (await fetch('/corpus.lattice')).arrayBuffer());
 * host.importSnapshot(snapshot, 'docs', 'create');
 * const api = host.fetchHandler({ prefix: '/api/lattice' });
 * self.addEventListener('fetch', (e) => {
 *   if (api.handles(e.request.url)) e.respondWith(api.handleFetch(e.request));
 * });
 * ```
 */

import type { ApiResponse } from './types';
import { loadWasm } from './wasm-loader';
import type {
  ByteInput,
  EmbedderRegistration,
  ImportMode,
  ImportResult,
  RawFetchHandler,
  RawFetchOptions,
  RawLatticeDB,
  RouteResponse,
  WasmInput,
} from './wasm-types';

export interface LatticeHostOptions {
  /** Refuse every mutating REST endpoint with 403 (host imports still work). */
  readOnly: boolean;
  /** WASM binary source; see {@link loadWasm}. */
  wasm?: WasmInput;
  /**
   * Text embedders to register with the router (see {@link LatticeHost.setEmbedder}).
   * Lexical (`"model": "bm25"`) queries need none.
   */
  embedders?: readonly EmbedderRegistration[];
}

/** Options for {@link LatticeHost.fetchHandler}. */
export type FetchHandlerOptions = RawFetchOptions;

/** A `fetch`-event handler over one database. */
export type LatticeFetchHandler = RawFetchHandler;

export class LatticeHost {
  private constructor(private readonly db: RawLatticeDB) {}

  static async create(options: LatticeHostOptions): Promise<LatticeHost> {
    const glue = await loadWasm(options.wasm);
    const host = new LatticeHost(new glue.LatticeDB({ readOnly: options.readOnly }));
    for (const registration of options.embedders ?? []) host.setEmbedder(registration);
    return host;
  }

  /**
   * Register (or replace) the embedder for `registration.model`. Text queries
   * naming that model are embedded inside the router, on every host (route,
   * fetch handler, service worker, Worker); an unregistered model answers 503.
   */
  setEmbedder(registration: EmbedderRegistration): void {
    this.db.setEmbedder(registration.model, registration.dim, registration.embed);
  }

  /** Import a `lattice-jsonl` dump into the collection its header names. */
  importJsonl(data: string | ByteInput, mode: ImportMode): ImportResult {
    return unwrap(this.db.importJsonl(data, mode));
  }

  /** Import a binary snapshot (from `exportSnapshot` or `GET …/export`). */
  importSnapshot(bytes: ByteInput, name: string, mode: ImportMode): ImportResult {
    return unwrap(this.db.importSnapshot(bytes, name, mode));
  }

  /** Export a collection as a binary snapshot. */
  exportSnapshot(name: string): Uint8Array {
    return this.db.exportSnapshot(name);
  }

  /**
   * Drive one request through the native router. HTTP errors resolve as
   * statuses; only invalid arguments reject.
   */
  async route(method: string, path: string, body?: string | ByteInput): Promise<RouteResponse> {
    return this.db.route(method, path, body);
  }

  /** Create a fetch handler mounted at `options.prefix`. */
  fetchHandler(options: FetchHandlerOptions): LatticeFetchHandler {
    return this.db.fetchHandler(
      options.corsOrigin === undefined
        ? { prefix: options.prefix }
        : { prefix: options.prefix, corsOrigin: options.corsOrigin }
    );
  }
}

function unwrap<T>(response: ApiResponse<T>): T {
  if (response.result === undefined) {
    throw new Error(response.error ?? `LatticeDB returned status ${response.status}`);
  }
  return response.result;
}
