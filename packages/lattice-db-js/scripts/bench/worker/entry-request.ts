// Benchmark Worker, request mode: WASM instantiation and the snapshot load
// both happen inside the first request (they count against its CPU time).
// `snapshot.lattice` (a Data module) is copied here by scripts/bench/workerd.mjs.
import wasm from '../../../wasm/lattice_server_bg.wasm';
import snapshot from './snapshot.lattice';
import { createLatticeHandler } from '../../../src/index';
import { timed } from './timed';

export default timed(
  createLatticeHandler({
    wasm,
    collection: 'c',
    prefix: '/api',
    readOnly: true,
    snapshotBytes: new Uint8Array(snapshot as ArrayBuffer),
  })
);
