// Benchmark Worker, startup mode: WASM instantiation, the snapshot load and
// eight warm-up queries run at module top level (the 1 s startup budget), so
// the first request only queries. Timers do not advance at top level, so the
// startup cost itself is not observable here (Node's numbers stand in).
// `snapshot.lattice` and `warmup.txt` are copied here by scripts/bench/workerd.mjs.
import wasm from '../../../wasm/lattice_server_bg.wasm';
import snapshot from './snapshot.lattice';
import { createLatticeHandler } from '../../../src/index';
import warmupQuery from './warmup.txt';
import { timed } from './timed';

const lattice = createLatticeHandler({
  wasm,
  collection: 'c',
  prefix: '/api',
  readOnly: true,
  snapshotBytes: new Uint8Array(snapshot as ArrayBuffer),
});
await lattice.ready({ warmup: { query: warmupQuery, runs: 8 } });

export default timed(lattice);
