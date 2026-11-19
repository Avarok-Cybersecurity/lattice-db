/**
 * @jest-environment node
 *
 * Regression bound on cold load: 1,000 chunks of 768-d vectors with an HNSW
 * graph and a BM25 index must import and answer a hybrid query quickly on the
 * release WASM build. Measured ~2 ms (load + first query) in Node on an
 * M-series Mac (scripts/bench/node.mjs); the bound is far looser so shared CI
 * runners pass. Negative control: restoring without the persisted graph
 * (rebuilding it on load) measured 884 ms and failed this test.
 */

import { describe, it, expect } from '@jest/globals';
import { LatticeHost } from '../host';
import { buildSnapshot, queryBody } from '../../scripts/bench/corpus';
import { wasmBytes } from './fixtures';

const LOAD_AND_QUERY_BOUND_MS = 150;
const SNAPSHOT_BOUND_BYTES = 7 * 1024 * 1024;

describe('cold load of the 1K-chunk fixture', () => {
  it('stays small and loads within the bound', async () => {
    const wasm = wasmBytes();
    const writer = await LatticeHost.create({ readOnly: false, wasm });
    const snapshot = await buildSnapshot(writer, 'hnsw', 1000);
    expect(snapshot.length).toBeLessThan(SNAPSHOT_BOUND_BYTES);

    const body = queryBody('hnsw');
    const samples: number[] = [];
    for (let run = 0; run < 3; run++) {
      const reader = await LatticeHost.create({ readOnly: true, wasm });
      const start = performance.now();
      reader.importSnapshot(snapshot, 'c', 'create');
      const response = await reader.route('POST', '/collections/c/points/query', body);
      samples.push(performance.now() - start);
      expect(response.status).toBe(200);
    }
    samples.sort((a, b) => a - b);
    expect(samples[1]).toBeLessThan(LOAD_AND_QUERY_BOUND_MS);
  }, 120_000);
});
