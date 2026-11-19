// Cold load and query latency in workerd, via `wrangler dev` (local only;
// nothing is deployed and no Cloudflare account is used).
//
//   node scripts/bench/node.mjs --out <dir>      # writes the snapshots
//   node scripts/bench/workerd.mjs <dir> [port]
//
// For every snapshot in <dir>, both Worker entries in scripts/bench/worker/
// are started in a fresh workerd, three times each (median reported):
//   request: WASM instantiation + snapshot load + query all in request 1
//   startup: instantiation, load and 8 warm-up queries at top level
// `x-handler-ms` is measured inside the isolate (local workerd advances
// performance.now during execution; production freezes it until I/O), so it
// is the handler's own time with no network or wrangler proxy overhead,
// rounded by workerd to whole milliseconds.
import { spawn } from 'node:child_process';
import { copyFileSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const [dir, portArg] = process.argv.slice(2);
if (dir === undefined) throw new Error('usage: workerd.mjs <snapshot-dir> [port]');
const port = Number(portArg ?? 8791);
const workerDir = fileURLToPath(new URL('./worker/', import.meta.url));
const base = `http://127.0.0.1:${port}`;

async function waitForServer(child) {
  for (let i = 0; i < 120; i++) {
    if (child.exitCode !== null) throw new Error(`wrangler exited with ${child.exitCode}`);
    try {
      if ((await fetch(`${base}/noop`)).ok) return;
    } catch {
      // Not listening yet.
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error('wrangler dev did not start within 60 s');
}

async function query(body) {
  const res = await fetch(`${base}/api/collections/c/points/query`, { method: 'POST', body });
  await res.arrayBuffer();
  if (res.status !== 200) throw new Error(`query returned ${res.status}`);
  return Number(res.headers.get('x-handler-ms'));
}

async function run(entry, body) {
  const log = [];
  const child = spawn(
    'npx',
    ['-y', 'wrangler@4', 'dev', entry, '--port', String(port), '--ip', '127.0.0.1', '--log-level', 'warn'],
    { cwd: workerDir, env: { ...process.env, WRANGLER_SEND_METRICS: 'false' }, stdio: ['ignore', 'pipe', 'pipe'] }
  );
  child.stdout.on('data', (d) => log.push(String(d)));
  child.stderr.on('data', (d) => log.push(String(d)));
  try {
    await waitForServer(child);
    const first = await query(body);
    const warm = [];
    for (let i = 0; i < 30; i++) warm.push(await query(body));
    warm.sort((a, b) => a - b);
    return { first, warm: warm[Math.floor(warm.length / 2)] };
  } catch (error) {
    throw new Error(`${entry}: ${error instanceof Error ? error.message : String(error)}\n${log.join('')}`);
  } finally {
    if (child.exitCode === null) {
      const exited = new Promise((r) => child.once('exit', r));
      child.kill('SIGINT');
      await exited;
    }
  }
}

console.log('| config | chunks | MiB | request mode: first request ms | startup mode: first request ms | warm ms |');
console.log('|---|---|---|---|---|---|');
const snapshots = readdirSync(dir).filter((f) => f.endsWith('.lattice')).sort();
for (const file of snapshots) {
  const name = file.replace(/\.lattice$/, '');
  const [, config, n] = /^(.*)-(\d+)$/.exec(name) ?? [];
  copyFileSync(join(dir, file), join(workerDir, 'snapshot.lattice'));
  const body = readFileSync(join(dir, `${name}.query.json`), 'utf8');
  writeFileSync(join(workerDir, 'warmup.txt'), body);
  // Each first request is one sample from a fresh isolate: take the median of three.
  const firsts = async (entry) => {
    const runs = [await run(entry, body), await run(entry, body), await run(entry, body)];
    runs.sort((a, b) => a.first - b.first);
    return runs[1];
  };
  const request = await firsts('entry-request.ts');
  const startup = await firsts('entry-startup.ts');
  const mib = (statSync(join(dir, file)).size / 1048576).toFixed(2);
  console.log(
    `| ${config} | ${n} | ${mib} | ${request.first} | ${startup.first} | ${startup.warm} |`
  );
}
