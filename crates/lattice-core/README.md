# LatticeDB Core

Core library for LatticeDB - a high-performance Vector + Graph database.

## Features

- **HNSW Index** - Hierarchical Navigable Small World graph for approximate nearest neighbor search
- **Graph Operations** - BFS/DFS traversal, edge management
- **Scalar Quantization** - Int8 quantization for 4x memory reduction
- **SIMD Distance Functions** - Optimized cosine, euclidean, and dot product calculations
- **WASM Support** - Full compilation and test support for WebAssembly

## Usage

```rust
use lattice_core::engine::CollectionEngine;
use lattice_core::types::collection::{CollectionConfig, VectorConfig, HnswConfig, Distance};
use lattice_core::types::point::Point;
use lattice_core::types::query::SearchQuery;

// Create a collection
let config = CollectionConfig::new(
    "my_collection",
    VectorConfig::new(128, Distance::Cosine),
    HnswConfig::default(),
);
let mut engine = CollectionEngine::new(config)?;

// Insert points
let point = Point::new_vector(1, vec![0.1; 128]);
engine.upsert_points(vec![point])?;

// Search
let query = SearchQuery::new(vec![0.1; 128], 10);
let results = engine.search(query)?;
```

## Running Tests

### Native Tests

```bash
# Run all tests
cargo test -p lattice-core

# Run with specific features
cargo test -p lattice-core --features simd
```

### WASM Tests

LatticeDB Core supports running tests on the `wasm32-unknown-unknown` target using `wasm-pack` and `wasm-bindgen-test`.

#### Prerequisites

```bash
# Install wasm-pack
cargo install wasm-pack

# Install Chrome (for headless testing)
# macOS: brew install --cask google-chrome
# Linux: apt install chromium-browser
```

#### Running WASM Tests

```bash
# Run tests in headless Chrome (recommended)
wasm-pack test --headless --chrome crates/lattice-core

# Run tests in headless Firefox
wasm-pack test --headless --firefox crates/lattice-core
```

**Note:** Node.js testing (`wasm-pack test --node`) has known issues with `wasm-bindgen-test` and may crash. Browser-based testing is recommended.

#### How WASM Tests Work

All test modules use conditional compilation to support both native and WASM targets:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // On WASM, shadow the test attribute with wasm_bindgen_test
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]  // Works on both native and WASM
    fn test_example() {
        assert!(true);
    }
}
```

Integration tests also configure browser execution:

```rust
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `simd` | Enable SIMD-accelerated distance calculations |
| `quantization` | Enable scalar quantization for memory-efficient storage |

## Building for WASM

```bash
# Check compilation
cargo check -p lattice-core --target wasm32-unknown-unknown

# Build
cargo build -p lattice-core --target wasm32-unknown-unknown --release
```

## Architecture

```
lattice-core/
├── src/
│   ├── engine/         # Collection engine (point ops, search, graph)
│   ├── graph/          # Graph traversal (BFS, DFS)
│   ├── index/          # HNSW index, distance functions, quantization
│   ├── types/          # Collection, Point, Query types
│   ├── error.rs        # Error types
│   ├── transport.rs    # LatticeTransport trait
│   └── lib.rs
└── tests/
    └── integration_test.rs
```
