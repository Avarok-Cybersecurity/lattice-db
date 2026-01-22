# Changelog

All notable changes to LatticeDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
