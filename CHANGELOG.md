# Changelog

All notable changes to LatticeDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
