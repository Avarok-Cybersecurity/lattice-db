// Cold load and query latency in Node, on the real WASM build.
//
//   npm run build:wasm               (Node >= 22.18: corpus.ts is type-stripped)
//   node scripts/bench/node.mjs [--out fixtures-dir] [sizes...]   (default sizes: 1000 2000)
//
// Per configuration and size: snapshot bytes, cold load (new LatticeDB +
// importSnapshot), first query, and warm query (median of 50). Node is
// single-threaded here, so wall time ~ CPU time. `--out` also writes the
// snapshots and query bodies for the workerd benchmark.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import init, { LatticeDB } from '../../wasm/lattice_server.js';
import { CONFIGS, buildSnapshot, queryBody } from './corpus.ts';

const args = process.argv.slice(2);
const outAt = args.indexOf('--out');
const outDir = outAt >= 0 ? args[outAt + 1] : null;
const sizes = args.filter((_, i) => outAt < 0 || (i !== outAt && i !== outAt + 1)).map(Number);
if (sizes.length === 0) sizes.push(1000, 2000);
if (sizes.some((n) => !Number.isInteger(n) || n <= 0)) throw new Error('sizes must be positive integers');

await init({ module_or_path: readFileSync(new URL('../../wasm/lattice_server_bg.wasm', import.meta.url)) });
if (outDir !== null) mkdirSync(outDir, { recursive: true });

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const ms = (x) => x.toFixed(2);
const rows = [];

for (const n of sizes) {
  for (const config of Object.keys(CONFIGS)) {
    const builder = new LatticeDB();
    const snapshot = await buildSnapshot(builder, config, n);
    builder.free();
    const body = queryBody(config);
    if (outDir !== null) {
      writeFileSync(join(outDir, `${config}-${n}.lattice`), snapshot);
      writeFileSync(join(outDir, `${config}-${n}.query.json`), body);
    }
    const loads = [];
    const firsts = [];
    let warm = [];
    for (let run = 0; run < 5; run++) {
      const t0 = performance.now();
      const db = new LatticeDB({ readOnly: true });
      db.importSnapshot(snapshot, 'c', 'create');
      const t1 = performance.now();
      const first = await db.route('POST', '/collections/c/points/query', body);
      const t2 = performance.now();
      if (first.status !== 200) throw new Error(`${config}: query returned ${first.status}`);
      loads.push(t1 - t0);
      firsts.push(t2 - t1);
      if (run === 0) {
        for (let q = 0; q < 50; q++) {
          const s = performance.now();
          await db.route('POST', '/collections/c/points/query', body);
          warm.push(performance.now() - s);
        }
      }
      db.free();
    }
    rows.push({
      config,
      n,
      mib: (snapshot.length / 1048576).toFixed(2),
      load: ms(median(loads)),
      first: ms(median(firsts)),
      warm: ms(median(warm)),
      coldTotal: ms(median(loads.map((l, i) => l + firsts[i]))),
    });
    warm = [];
  }
}

console.log('| config | chunks | snapshot MiB | cold load ms | first query ms | load+first ms | warm query ms |');
console.log('|---|---|---|---|---|---|---|');
for (const r of rows) {
  console.log(`| ${r.config} | ${r.n} | ${r.mib} | ${r.load} | ${r.first} | ${r.coldTotal} | ${r.warm} |`);
}
