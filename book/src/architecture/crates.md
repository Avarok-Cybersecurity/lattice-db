# Crate Structure

LatticeDB is organized as a Cargo workspace with multiple crates, each with a specific responsibility. This chapter explains the purpose and contents of each crate.

## Workspace Overview

```
lattice-db/
├── Cargo.toml              # Workspace configuration
├── crates/
│   ├── lattice-core/       # Pure business logic
│   ├── lattice-storage/    # Storage implementations
│   ├── lattice-server/     # HTTP/WASM server
│   └── lattice-bench/      # Benchmarks
└── book/                   # This documentation
```

## Dependency Graph

```
                    ┌─────────────────┐
                    │  lattice-bench  │
                    │  (benchmarks)   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │ lattice-server  │           │ lattice-storage │
    │ (HTTP/WASM API) │           │ (Storage impls) │
    └────────┬────────┘           └────────┬────────┘
             │                             │
             └──────────────┬──────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │  lattice-core   │
                  │ (Pure business  │
                  │     logic)      │
                  └─────────────────┘
```

## lattice-core

**Purpose**: Contains all business logic with zero I/O dependencies.

### Modules

| Module | Description |
|--------|-------------|
| `types/` | Data structures: `Point`, `Edge`, `Vector`, configs |
| `index/` | HNSW algorithm, distance metrics, quantization |
| `graph/` | Adjacency storage, traversal algorithms |
| `cypher/` | Cypher parser and query executor |
| `engine/` | `CollectionEngine` that orchestrates everything |
| `storage.rs` | `LatticeStorage` trait definition |
| `transport.rs` | `LatticeTransport` trait definition |
| `error.rs` | Error types and `Result` aliases |

### Key Types

```rust
// Re-exported from lattice-core
pub use engine::collection::CollectionEngine;
pub use index::hnsw::HnswIndex;
pub use storage::{LatticeStorage, StorageError};
pub use transport::{LatticeRequest, LatticeResponse, LatticeTransport};
pub use types::collection::{CollectionConfig, Distance, HnswConfig};
pub use types::point::{Edge, Point, PointId, Vector};
pub use types::query::{SearchQuery, SearchResult};
```

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `simd` | SIMD-accelerated distance calculations | Enabled |

### Dependencies

`lattice-core` has minimal dependencies to stay portable:

```toml
[dependencies]
async-trait = "0.1"
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }

# SIMD for distance calculations (optional)
wide = { version = "0.7", optional = true }
```

**No I/O crates**: No `tokio`, `std::fs`, `reqwest`, or `web_sys`.

---

## lattice-storage

**Purpose**: Platform-specific `LatticeStorage` implementations.

### Implementations

| Type | Platform | Description |
|------|----------|-------------|
| `MemStorage` | All | In-memory HashMap, for testing |
| `DiskStorage` | Native | File-based using `tokio::fs` |
| `OpfsStorage` | WASM | Browser Origin Private File System |

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `native` | Enables `DiskStorage` | Disabled |
| `wasm` | Enables `OpfsStorage` | Disabled |

### Usage

```toml
# Server application
[dependencies]
lattice-storage = { version = "0.1", features = ["native"] }

# Browser application
[dependencies]
lattice-storage = { version = "0.1", features = ["wasm"] }
```

### Code Example

```rust
use lattice_storage::MemStorage;

// In-memory storage for testing
let storage = MemStorage::new();
storage.write_page(0, b"hello").await?;
let page = storage.read_page(0).await?;
```

---

## lattice-server

**Purpose**: HTTP API and transport implementations.

### Modules

| Module | Description |
|--------|-------------|
| `dto/` | Data Transfer Objects (JSON serialization) |
| `handlers/` | Request handlers for each endpoint |
| `router.rs` | Route matching and dispatch |
| `axum_transport.rs` | Native HTTP server (Axum) |
| `service_worker.rs` | WASM fetch event handler |
| `openapi.rs` | OpenAPI documentation generator |

### REST API Endpoints

```
PUT    /collections/{name}              Create collection
GET    /collections/{name}              Get collection info
DELETE /collections/{name}              Delete collection

PUT    /collections/{name}/points       Upsert points
POST   /collections/{name}/points/query Vector search
POST   /collections/{name}/points/scroll Paginated retrieval

POST   /collections/{name}/graph/edges  Add graph edges
POST   /collections/{name}/graph/query  Cypher query
```

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `native` | Enables `AxumTransport` | Disabled |
| `wasm` | Enables `ServiceWorkerTransport` | Disabled |
| `openapi` | Enables OpenAPI documentation | Disabled |

### Usage

```rust
// Native server
use lattice_server::{axum_transport::AxumTransport, router::*};

#[tokio::main]
async fn main() {
    let state = new_app_state();
    let transport = AxumTransport::new("0.0.0.0:6333");

    transport.serve(move |request| {
        let state = state.clone();
        async move { route(state, request).await }
    }).await.unwrap();
}
```

---

## lattice-bench

**Purpose**: Benchmarks comparing LatticeDB to Qdrant and Neo4j.

### Benchmarks

| Benchmark | Description |
|-----------|-------------|
| `vector_ops` | Vector operations (search, upsert, retrieve, scroll) |
| `cypher_comparison` | Cypher queries vs Neo4j |
| `quick_vector_bench` | Fast iteration benchmark |

### Running Benchmarks

```bash
# All benchmarks
cargo bench -p lattice-bench

# Specific benchmark
cargo bench -p lattice-bench --bench vector_ops

# Quick iteration
cargo run -p lattice-bench --release --example quick_vector_bench
```

### Output

Benchmarks use Criterion and output:
- Console summary with mean/stddev
- HTML reports in `target/criterion/`
- JSON data for CI integration

---

## Building for Different Platforms

### Native Server

```bash
cargo build --release -p lattice-server --features native
```

### WASM Browser

```bash
# Install wasm-pack
cargo install wasm-pack

# Build WASM package
wasm-pack build crates/lattice-server \
    --target web \
    --out-dir pkg \
    --no-default-features \
    --features wasm
```

### All Tests

```bash
# Native tests
cargo test --workspace

# WASM tests (requires Chrome)
wasm-pack test --headless --chrome crates/lattice-core
```

## Adding New Features

### New Storage Backend

1. Create implementation in `lattice-storage/src/`:

```rust
// my_storage.rs
pub struct MyStorage { ... }

impl LatticeStorage for MyStorage {
    async fn read_page(&self, page_id: u64) -> StorageResult<Page> { ... }
    // ... other methods
}
```

2. Add feature flag in `Cargo.toml`:

```toml
[features]
my-backend = ["some-dependency"]

[dependencies]
some-dependency = { version = "1.0", optional = true }
```

3. Conditionally export:

```rust
#[cfg(feature = "my-backend")]
pub mod my_storage;

#[cfg(feature = "my-backend")]
pub use my_storage::MyStorage;
```

### New API Endpoint

1. Add handler in `lattice-server/src/handlers/`:

```rust
pub async fn my_handler<S: LatticeStorage>(
    state: &AppState<S>,
    request: &LatticeRequest,
) -> LatticeResponse {
    // Handle request
}
```

2. Add route in `router.rs`:

```rust
("POST", path) if path.ends_with("/my-endpoint") => {
    my_handler(state, request).await
}
```

3. Add DTO types if needed in `dto/`.

## Next Steps

- [HNSW Index](../vector/hnsw.md) - How vector search works
- [Cypher Language](../graph/cypher.md) - Graph query implementation
