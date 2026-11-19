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
  prefix: string;
  corsOrigin?: string;
}

export interface RawFetchHandler {
  readonly prefix: string;
  handles(url: string): boolean;
  handleFetch(request: Request): Promise<Response>;
  errorResponse(status: number, message: string): Response;
}

export interface RawLatticeDB {
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
