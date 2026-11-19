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
import { mayCarryTextQuery, resolveTextQuery, type Embedder } from './text-query';
import type {
  ByteInput,
  ImportMode,
  ImportResult,
  RawFetchHandler,
  RawLatticeDB,
  RouteResponse,
  WasmInput,
} from './wasm-types';

export interface LatticeHostOptions {
  /** Refuse every mutating REST endpoint with 403 (host imports still work). */
  readOnly: boolean;
  /** WASM binary source; see {@link loadWasm}. */
  wasm?: WasmInput;
}

export interface FetchHandlerOptions {
  /** Mount path such as `/api/lattice`, or `''` for every path. */
  prefix: string;
  /** `Access-Control-Allow-Origin` value; omit for same-origin use. */
  corsOrigin?: string;
  /** Embeds text queries; without it they are answered with 501. */
  embed?: Embedder;
}

/** A `fetch`-event handler over one database. */
export interface LatticeFetchHandler {
  readonly prefix: string;
  /** Whether an absolute URL is under the prefix. */
  handles(url: string): boolean;
  /** Answer a request (HTTP errors are statuses, never rejections). */
  handleFetch(request: Request): Promise<Response>;
  /** An error `Response` carrying the same security/CORS headers as routed ones. */
  errorResponse(status: number, message: string): Response;
}

export class LatticeHost {
  private constructor(private readonly db: RawLatticeDB) {}

  static async create(options: LatticeHostOptions): Promise<LatticeHost> {
    const glue = await loadWasm(options.wasm);
    return new LatticeHost(new glue.LatticeDB({ readOnly: options.readOnly }));
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
    const raw = this.db.fetchHandler(
      options.corsOrigin === undefined
        ? { prefix: options.prefix }
        : { prefix: options.prefix, corsOrigin: options.corsOrigin }
    );
    return wrapFetchHandler(raw, options.embed);
  }
}

function wrapFetchHandler(raw: RawFetchHandler, embed: Embedder | undefined): LatticeFetchHandler {
  return {
    prefix: raw.prefix,
    handles: (url: string): boolean => raw.handles(url),
    errorResponse: (status: number, message: string): Response =>
      raw.errorResponse(status, message),
    handleFetch: async (request: Request): Promise<Response> => {
      const pathname = new URL(request.url).pathname;
      if (!raw.handles(request.url) || !mayCarryTextQuery(request.method, pathname)) {
        return raw.handleFetch(request);
      }
      const outcome = await resolveTextQuery(await request.clone().text(), embed);
      switch (outcome.kind) {
        case 'pass':
          return raw.handleFetch(request);
        case 'reject':
          return raw.errorResponse(outcome.status, outcome.message);
        case 'rewrite':
          return raw.handleFetch(new Request(request, { body: outcome.body }));
      }
    },
  };
}

function unwrap<T>(response: ApiResponse<T>): T {
  if (response.result === undefined) {
    throw new Error(response.error ?? `LatticeDB returned status ${response.status}`);
  }
  return response.result;
}
