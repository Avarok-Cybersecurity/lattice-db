// Snapshot import cost in Node, using the real WASM build.
//
//   npm run build:wasm && node scripts/bench-snapshot.mjs 1000 10000
//
// Reports, per corpus size (768-d f32): WASM instantiate, snapshot size,
// importSnapshot (first load), and steady-state search/route latency — the
// per-request cost after the first load.
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import init, { LatticeDB } from '../wasm/lattice_server.js';

const DIM = 768;
const sizes = process.argv.slice(2).map(Number);
if (sizes.length === 0 || sizes.some((n) => !Number.isInteger(n) || n <= 0)) {
  throw new Error('usage: node scripts/bench-snapshot.mjs <points> [<points> ...]');
}

let seed = 42;
const rand = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32) * 2 - 1;
const vector = () => Array.from({ length: DIM }, rand);

const t0 = performance.now();
await init({ module_or_path: readFileSync(new URL('../wasm/lattice_server_bg.wasm', import.meta.url)) });
console.log(`wasm instantiate: ${(performance.now() - t0).toFixed(1)} ms`);

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const time = (fn) => {
  const s = performance.now();
  fn();
  return performance.now() - s;
};

for (const n of sizes) {
  const source = new LatticeDB();
  source.createCollection('c', { vectors: { size: DIM, distance: 'Cosine' } });
  for (let i = 0; i < n; i += 500) {
    const batch = [];
    for (let id = i; id < Math.min(n, i + 500); id++) {
      batch.push({ id: id + 1, vector: vector(), payload: { title: `doc ${id}`, url: `/p/${id}` } });
    }
    source.upsert('c', batch);
  }
  const snapshot = source.exportSnapshot('c');
  source.free();

  const imports = [];
  for (let run = 0; run < 3; run++) {
    const db = new LatticeDB({ readOnly: true });
    imports.push(time(() => db.importSnapshot(snapshot, 'c', 'create')));
    db.free();
  }

  const db = new LatticeDB({ readOnly: true });
  db.importSnapshot(snapshot, 'c', 'create');
  const body = JSON.stringify({ vector: vector(), limit: 10, with_payload: true });
  const searches = [];
  for (let q = 0; q < 50; q++) {
    const s = performance.now();
    const res = await db.route('POST', '/collections/c/points/search', body);
    searches.push(performance.now() - s);
    if (res.status !== 200) throw new Error(`search returned ${res.status}`);
  }
  console.log(
    `${n} pts: snapshot ${(snapshot.length / 1048576).toFixed(1)} MiB, ` +
      `importSnapshot median ${median(imports).toFixed(1)} ms (runs ${imports.map((x) => x.toFixed(0)).join('/')}), ` +
      `route search p50 ${median(searches).toFixed(2)} ms, max ${Math.max(...searches).toFixed(2)} ms`
  );
  db.free();
}
