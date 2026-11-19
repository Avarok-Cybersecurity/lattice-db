// Deterministic site-search corpus and snapshots for the cold-load benchmarks
// (scripts/bench/*.mjs, run by Node >= 22.18 with type stripping) and the
// cold-load regression test (src/__tests__/cold-load.test.ts).
//
// A "chunk" is 200 words of Zipf-like vocabulary (like prose) plus a 768-d
// vector, with title/url payload. Each configuration is a collection shape a
// static site can ship; `buildSnapshot` returns the exported bytes.

export const DIM = 768;
const VOCAB = 6000;
const WORDS_PER_CHUNK = 200;

/** Collection create bodies, by configuration name. */
export const CONFIGS: Record<string, object> = {
  lexical: { text_index: { field: 'text' } },
  'flat-f32': {
    vectors: { size: DIM, distance: 'Cosine', index: 'flat', storage: 'f32' },
    text_index: { field: 'text' },
  },
  'flat-f16': {
    vectors: { size: DIM, distance: 'Cosine', index: 'flat', storage: 'f16' },
    text_index: { field: 'text' },
  },
  'flat-int8': {
    vectors: { size: DIM, distance: 'Cosine', index: 'flat', storage: 'int8' },
    text_index: { field: 'text' },
  },
  hnsw: { vectors: { size: DIM, distance: 'Cosine' }, text_index: { field: 'text' } },
};

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

function word(rand: () => number): string {
  return `w${Math.floor(VOCAB * rand() ** 3)}`;
}

export function vector(rand: () => number): number[] {
  return Array.from({ length: DIM }, () => rand() * 2 - 1);
}

export interface Chunk {
  id: number;
  text: string;
  vector: number[];
}

export function chunks(n: number): Chunk[] {
  const rand = rng(42);
  return Array.from({ length: n }, (_, i) => ({
    id: i + 1,
    text: Array.from({ length: WORDS_PER_CHUNK }, () => word(rand)).join(' '),
    vector: vector(rand),
  }));
}

/** The query each configuration answers: BM25 alone, or BM25 + dense fused with RRF. */
export function queryBody(config: string): string {
  const rand = rng(7);
  const text = 'w12 w340 w57 w2001';
  const lexical = { query: { text, model: 'bm25' }, limit: 10, with_payload: true };
  if (config === 'lexical') return JSON.stringify(lexical);
  return JSON.stringify({
    prefetch: [
      { query: { text, model: 'bm25' }, limit: 20 },
      { query: vector(rand), limit: 20 },
    ],
    query: { fusion: 'rrf' },
    limit: 10,
    with_payload: true,
  });
}

/** What `buildSnapshot` needs: the raw wasm `LatticeDB` or a `LatticeHost`. */
export interface SnapshotSource {
  route(method: string, path: string, body?: string): Promise<{ status: number; body: Uint8Array }>;
  exportSnapshot(name: string): Uint8Array;
}

/** Create collection `c` in `db` with `config`, upsert `n` chunks, and export it. */
export async function buildSnapshot(db: SnapshotSource, config: string, n: number): Promise<Uint8Array> {
  const create = CONFIGS[config];
  if (create === undefined) throw new Error(`unknown configuration ${config}`);
  const ok = (res: { status: number; body: Uint8Array }, what: string): void => {
    if (res.status !== 200) throw new Error(`${what}: ${res.status} ${new TextDecoder().decode(res.body)}`);
  };
  ok(await db.route('PUT', '/collections/c', JSON.stringify(create)), 'create');
  const docs = chunks(n);
  for (let i = 0; i < n; i += 250) {
    const points = docs.slice(i, i + 250).map((d) => ({
      id: d.id,
      ...(config === 'lexical' ? {} : { vector: d.vector }),
      payload: { text: d.text, title: `Page ${d.id}`, url: `/docs/${d.id}` },
    }));
    ok(await db.route('PUT', '/collections/c/points', JSON.stringify({ points })), 'upsert');
  }
  return db.exportSnapshot('c');
}
