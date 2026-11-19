/**
 * Types for the host-facing surface of the wasm-bindgen module
 * (`crates/lattice-server/src/wasm_{route,import,fetch}.rs`).
 */

import type { ApiResponse } from './types';

/** Anything the WASM glue can instantiate from. */
export type WasmInput =
  | string
  | URL
  | Request
  | Response
  | ArrayBuffer
  | ArrayBufferView
  | WebAssembly.Module;

/** Bytes accepted by the bindings: any view (checked with `ArrayBuffer.isView`) or buffer. */
export type ByteInput = ArrayBufferView | ArrayBuffer;

/** How an import treats an existing collection of the same name. */
export type ImportMode = 'create' | 'replace' | 'merge';

/** Result of `importJsonl` / `importSnapshot`. */
export interface ImportResult {
  points_imported: number;
  points_skipped: number;
  dimension: number;
  mode: ImportMode;
}

/** Plain-object response of `route()`. */
export interface RouteResponse {
  status: number;
  headers: Record<string, string>;
  body: Uint8Array;
}

/** Options for `new LatticeDB(options)` in the WASM module. */
export interface RawDBOptions {
  readOnly: boolean;
}

/** Options for `fetchHandler()` in the WASM module. */
export interface RawFetchOptions {
  /** Mount path such as `/api/lattice`, or `''` for every path. */
  prefix: string;
  /** `Access-Control-Allow-Origin` value; omit for same-origin use. */
  corsOrigin?: string;
}

/** A `fetch`-event handler over one database. */
export interface RawFetchHandler {
  readonly prefix: string;
  /** Whether an absolute URL is under the prefix. */
  handles(url: string): boolean;
  /** Answer a request (HTTP errors are statuses, never rejections). */
  handleFetch(request: Request): Promise<Response>;
  /** An error `Response` carrying the same security/CORS headers as routed ones. */
  errorResponse(status: number, message: string): Response;
}

/**
 * Turns texts into vectors, one per text and in order. May be async (e.g. a
 * call to an embeddings API or Workers AI).
 */
export type EmbedFunction = (
  texts: string[]
) => ReadonlyArray<Float32Array | number[]> | Promise<ReadonlyArray<Float32Array | number[]>>;

/**
 * A text embedder registered with the router. Collections whose
 * `embedding.model` equals `model` answer `{"query": {"text", "model"}}`
 * through it; `bm25` queries never need one.
 */
export interface EmbedderRegistration {
  model: string;
  /** Vector size the function returns; must equal the collection's `vectors.size`. */
  dim: number;
  embed: EmbedFunction;
}

export interface RawLatticeDB {
  setEmbedder(model: string, dim: number, embed: EmbedFunction): void;
  importJsonl(data: string | ByteInput, mode: ImportMode): ApiResponse<ImportResult>;
  importSnapshot(bytes: ByteInput, name: string, mode: ImportMode): ApiResponse<ImportResult>;
  exportSnapshot(name: string): Uint8Array;
  route(method: string, path: string, body?: string | ByteInput): Promise<RouteResponse>;
  fetchHandler(options: RawFetchOptions): RawFetchHandler;
}

export interface HostWasmModule {
  default: (input?: { module_or_path: WasmInput }) => Promise<unknown>;
  LatticeDB: new (options?: RawDBOptions) => RawLatticeDB;
}
