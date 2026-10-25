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
