# Changelog

All notable changes to LatticeDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Text search without embeddings (BM25)**: collections accept
  `"text_index": {"field": "text", "k1": 1.2, "b": 0.75}`; the index is kept
  current on upsert/delete, persisted in snapshots, and rebuilt on WAL and
  JSONL load.
- **Lexical-only collections**: omit `vectors` for BM25 only (no HNSW). Vector
  search/query/upsert-with-vector on them return a 400 naming the cause.
- **Flat and compact vector indexes**: `vectors.index` = `"hnsw"` (default) |
  `"flat"` (exact scan); `vectors.storage` = `"f32"` (default) | `"f16"` |
  `"int8"` (vectors rounded on write, stored compactly in snapshots).
- **Snapshot format 2**: the HNSW graph, BM25 postings and encoded vectors are
  persisted, so loads copy arrays instead of rebuilding (1K chunks x 768-d with
  HNSW: ~1.5 ms vs ~880 ms). Format 1 snapshots still load (indexes rebuilt);
  newer formats fail with `Unsupported snapshot format N`.
- **JS hosts**: `LatticeHost` (`route`, `fetchHandler` for service workers,
  `embedders` / `setEmbedder`) and `createLatticeHandler` for Cloudflare Workers
  (`snapshotBytes` + `ready({ warmup })` at startup, or lazy `snapshotUrl`).
  Read-only mode answers BM25, dense-from-text and hybrid queries.
  See `docs/edge-search.md`; benchmarks in `packages/lattice-db-js/scripts/bench`.
  `POST /collections/{name}/points/query` answers `{"query": {"text": "...", "model": "bm25"}}`.
- **Hybrid search**: `prefetch` + `{"fusion": "rrf"}` / `{"rrf": {"k": n}}`
  (Reciprocal Rank Fusion, k = 60 unless given).
- **Text embedding**: `"embedding": {"model": "..."}` records the model behind a
  collection's vectors; text queries naming another model are rejected. Native
  OpenAI-compatible client behind the `openai-embedder` feature
  (`LATTICE_EMBEDDER_*`), and `LatticeDB.setEmbedder` / `queryPoints` in wasm.
- Error response bodies are now valid JSON when the message contains quotes.
- **Import/Export Endpoints**: Binary collection backup and restore
  - `GET /collections/{name}/export` - Export to rkyv binary format
  - `POST /collections/{name}/import?mode={create|replace|merge}` - Import with mode selection
- **Graph Traverse Endpoint**: `POST /collections/{name}/graph/traverse` for BFS/DFS traversal
- **Authentication**: API key and Bearer token support via environment variables
- **Rate Limiting**: Token bucket algorithm with configurable limits (100 req/s default)

### Security

- Self-loop detection in graph edge insertion
- Checked arithmetic in serialization to prevent overflow
- BFS/DFS frontier size limits to prevent memory exhaustion
- HNSW insert returns Result for proper error handling
- Path length validation in Cypher parser (max depth: 100)

### Fixed

- **DiskStorage pages no longer overwrite each other**: pages lived in one `pages.dat` at
  `page_id * page_size` (4 KiB), but a WAL page grows to 256 KiB and one synced batch can exceed
  that, so each page rotation wrote over the tail of the page before it and WAL replay silently
  skipped the clobbered entries (lost upserts, deletes and stale state after reopen). Each page
  is now its own file under `pages/`, written via a temp file and rename, and read back exactly.
  `DiskStorage::new`'s size argument is now the largest page it writes (`CapacityExceeded`
  beyond it; `with_defaults` = 1 GiB). A directory holding the old `pages.dat` is refused with
  `StorageError::InvalidFormat` and must be rebuilt. lattice-server now calls `init()` on its
  storages, so it loads the WAL header on restart and applies that refusal.

- Dense vector operations now return `Option`/`Result` instead of panicking

## [0.5.0] - Unreleased

Durable collections on wasm32, for Cloudflare Durable Objects, on a WAL whose commits are now
all-or-nothing. A minor bump: `WalEntry` gains a variant and the test harness's storage doubles
change behaviour.

### Added

- **wasm32 `DurableEngine<W, D>`**: the wasm collection engine over any `LatticeStorage`, with the
  native engine's constructors (`open_durable`, `open_durable_unified`) and `*_async` writes. Each
  write appends, syncs and only then applies, so `Ok` means durable. `checkpoint()` stores the
  collection as 64 KiB snapshot chunks in two alternating slots behind a one-key manifest; opening
  loads the snapshot and replays the WAL from its LSN. No clock, thread or runtime is used, so it
  runs inside a Durable Object; call `checkpoint()` from an alarm when `should_checkpoint()` says so.
- **`WalEntry::DefineRelation`**: a relation first used by `add_edge_async` is logged with the edge
  in one commit, so edges keep their relation name across restarts. It is the last variant, so
  existing tags and the 32-byte archived size are unchanged; an older build skips the entry as
  undecodable.
- **`wal::replay`**: one replay core over a `WalApply` trait, used by both engines.
- **`LatticeStorage` conformance suite** in `latticedb-test-harness`
  (`lattice_test_harness::conformance`): a backend implements `StorageFixture` and runs
  `run_conformance`, which checks metadata blobs, pages (exact bytes, WAL-sized, any `u64` id),
  sync and reopen, and the WAL's crash consistency over the backend (acknowledged commits survive
  a restart; a commit whose page or header write failed never appears after one). `MemStorage`
  (native and wasm32), `DiskStorage` and the harness's `SharedState` run it; a store that drops
  its last page on `sync` is a negative control that must fail it. The harness is not on
  crates.io; depend on it by git tag.
- `wal::WAL_PAGE_MAX_BYTES` and `wal::WAL_HEADER_KEY` are public, and the `LatticeStorage` docs
  state the contract the WAL relies on, the wasm32 `?Send` form of the trait, and how to back it
  with a SQLite-backed Durable Object.

### Fixed

- **A failed commit could be persisted by the next one**: a failed `sync` kept its entry in the
  write buffer, page cache and header, so the next successful commit wrote it out. `sync` now
  restores the last durable state on error and gives the LSNs back.
- **A page written without its header was replayed**: reads now stop at the header's committed
  length, and a page shorter than the header records is reported as corruption.
- **LSNs after truncation were renumbered from 0**, so replay from a checkpoint skipped surviving
  entries. Each page is numbered from the first LSN the header records for it. Truncation persists
  the shortened page list before deleting pages; a missing listed page is an error, not empty.

### Changed

- `lattice_test_harness::MockStorage::read_page` returns `StorageError::PageNotFound` for a page
  never written, as every real backend does, instead of empty bytes. `FailingStorage` is generic
  over the backend it wraps (`FailingStorage::wrap`); `FailingStorage::new(&SharedState)` is
  unchanged.

## [0.4.0] - 2026-10-28

Cypher read queries now evaluate full expressions. This is a minor bump rather than a
patch because the public `Query` AST and the planner's operators changed.

### Breaking

- `cypher::ast::Query` is now `clauses: Vec<Clause>` plus `return_clause`. The single
  `match_clause`, `optional_matches`, `where_clause`, `with_clause` and `order_by` fields are
  gone; each `MATCH`, `OPTIONAL MATCH`, `UNWIND` and `WITH` is a `Clause` in query order.
  `ReturnClause` now carries `order_by`, `skip` and `limit`, and `WithClause` wraps a
  `ReturnClause` projection.
- `Expr` gains `HasLabels`, `Index`, `Slice` and `Scalar`; `CypherError` gains
  `UnknownFunction`, `InvalidArgumentCount`, `InvalidAggregation` and `MissingParameter`.
- `LogicalOp` gains `Aggregate`, `Argument`, `Cartesian` and `Unwind`, and `OptionalExpand`
  is replaced by `OptionalMatch`.

### Added

- Expressions: property access on nodes, relationships and maps; comparisons and
  `AND`/`OR`/`XOR`/`NOT` in three-valued logic; `IN`, `STARTS WITH`, `ENDS WITH`,
  `CONTAINS`, `IS [NOT] NULL`; checked integer arithmetic and `^`; string and list `+`;
  simple and generic `CASE`; list and map literals, indexing and slicing; parameters.
- 34 scalar functions (`labels`, `type`, `id`, `keys`, `properties`, `size`, `length`,
  `exists`, `coalesce`, `toString`, `toInteger`, `toFloat`, `toLower`, `toUpper`, `head`,
  `last`, ...), resolved once at plan time with their arity checked.
- Aggregates `count(*)`, `count(x)`, `count(DISTINCT x)`, `collect`, `min`, `max`, `sum`,
  `avg`, with implicit grouping on the non-aggregate items.
- Clauses: `MATCH`, `OPTIONAL MATCH` (left outer join), `UNWIND`, `WITH ... WHERE`, and
  `RETURN` with `DISTINCT`, `ORDER BY` on any expression, `SKIP` and `LIMIT`, applied in
  openCypher order (project, DISTINCT, ORDER BY, SKIP, LIMIT).
- A pattern whose first node is unbound and whose last node is bound is planned from the
  bound end, as one expansion per input row instead of a scan joined with every row.

Anything outside this subset is refused with an error that names it, never answered
wrongly: `MERGE`, `SET`, `CALL`, `UNION`, `shortestPath`, named paths, list and pattern
comprehensions, subqueries, `reduce`, `=~`, `id()` of a relationship, and a variable bound
to a variable-length relationship.

### Changed

- An unaliased result column is named by its expression text (`m.id`, `count(*)`).
- Float division or modulo by zero is an error, like integer division by zero.
- A single-hop match returns one row per matching relationship, so parallel relationships
  and self-loops now appear.
- A variable-length pattern returns each node in the hop range once, at its shortest
  distance, not one row per path.
- `SKIP` and `LIMIT` take integer literals only.
- An `ORDER BY` key that fails to evaluate is an error instead of a null.

### Fixed

These returned wrong answers on 0.3.x:

- `a - b` was parsed as `a + b`, `/` and `%` as `*`, and unary minus was dropped.
- `CONTAINS` was evaluated as `IN`, and map literals evaluated to `{}`.
- `WHERE x < 15` kept rows where `x` is null.
- `(a)<-[]-(b)` followed outgoing relationships.
- `DISTINCT` and `LIMIT` ran before the projection.
- The `ORDER BY` fast paths sorted on the wrong column for a key on another variable, read a
  payload `id` instead of the point id for `n.id`, and put nulls first; nulls now sort last
  ascending and first descending, as on the generic path.
- A query naming a label on any pattern node after the first failed with
  `Unsupported feature: Complex expression evaluation`.
- `IS NULL` and `count(*)` did not parse.

### Security

- utoipa 6 / utoipa-swagger-ui 10 move the `openapi` feature's build dependency from zip
  3.0.0 to zip 9 (RUSTSEC-2026-0367, -0368, -0369).

## [0.1.0] - 2025-01-22

### Added

- **Hybrid Graph + Vector Database**: First release combining HNSW vector search with Cypher graph queries
- **WASM Browser Support**: Full database runs in the browser with SIMD acceleration
- **Qdrant API Compatibility**: Drop-in replacement using existing Qdrant SDKs
- **Cypher Query Language**: Neo4j-compatible graph queries
- **SIMD Acceleration**: AVX2/AVX-512 (x86), NEON (ARM), SIMD128 (WASM)
- **Product Quantization**: ScaNN-style compression for memory efficiency
- **Scalar Quantization**: int8 vector compression
- **HNSW Shortcuts**: VLDB 2025 optimization for faster search
- **Async Indexing**: Non-blocking background index updates
- **Multi-platform Support**: Linux, macOS, Windows, Browser (WASM)

### Performance

- Vector search: 1.4x faster than Qdrant
- Vector upsert: 177x faster than Qdrant
- Graph queries: 5-62x faster than Neo4j
- Ultra-low memory footprint: 2.4 MB runtime

### Documentation

- mdbook documentation with architecture guides
- API reference documentation
- Performance benchmarks and tuning guides

[0.1.0]: https://github.com/Avarok-Cybersecurity/lattice-db/releases/tag/v0.1.0
