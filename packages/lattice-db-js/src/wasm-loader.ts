/**
 * Loads and instantiates the wasm-bindgen module once per realm.
 *
 * Shared by the `LatticeDB` wrapper and the hosts (page, service worker,
 * Cloudflare Worker) so the glue is imported and initialised in one place.
 */

import type { HostWasmModule, WasmInput } from './wasm-types';

let loading: Promise<HostWasmModule> | null = null;

/**
 * Import the glue and instantiate the WASM binary.
 *
 * @param input - Where to get the binary: a URL/path, bytes, a `Response`, or
 *   a precompiled `WebAssembly.Module` (required on Cloudflare Workers, which
 *   forbid compiling WASM from bytes at runtime). Omit to let the glue resolve
 *   `lattice_server_bg.wasm` next to itself.
 *
 * Only the first call's `input` is used; later calls share that instance. A
 * failed load is not cached, so a later call can retry.
 */
export function loadWasm(input?: WasmInput): Promise<HostWasmModule> {
  if (loading === null) {
    loading = instantiate(input).catch((error: unknown) => {
      loading = null;
      throw error;
    });
  }
  return loading;
}

async function instantiate(input?: WasmInput): Promise<HostWasmModule> {
  const glue = (await import('../wasm/lattice_server.js')) as unknown as HostWasmModule;
  if (input === undefined) {
    await glue.default();
  } else {
    await glue.default({ module_or_path: input });
  }
  return glue;
}
