# Architecture Overview

LatticeDB is designed from the ground up for **cross-platform execution**: the same core logic runs identically on native servers and in WebAssembly browsers. This chapter explains the key architectural decisions that make this possible.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐         ┌───────────────────┐            │
│  │   REST Handlers   │         │  Cypher Executor  │            │
│  │  (Qdrant-compat)  │         │  (Graph Queries)  │            │
│  └─────────┬─────────┘         └─────────┬─────────┘            │
│            │                             │                       │
│            └──────────┬──────────────────┘                       │
│                       ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    CollectionEngine                          ││
│  │  ┌─────────────────┐       ┌─────────────────┐              ││
│  │  │   HNSW Index    │◄─────►│   Graph Store   │              ││
│  │  │ (Vector Search) │       │   (Adjacency)   │              ││
│  │  └─────────────────┘       └─────────────────┘              ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                       Abstraction Boundary                       │
│  ┌─────────────────┐         ┌─────────────────┐                │
│  │ LatticeStorage  │         │ LatticeTransport│                │
│  │     (trait)     │         │     (trait)     │                │
│  └────────┬────────┘         └────────┬────────┘                │
├───────────┼────────────────────────────┼────────────────────────┤
│           ▼                            ▼                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Platform Implementations                    ││
│  │                                                              ││
│  │  Native:                     WASM:                           ││
│  │  ├─ DiskStorage              ├─ OpfsStorage                  ││
│  │  └─ AxumTransport            └─ ServiceWorkerTransport       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Core Design Principles

### 1. Zero I/O in Core Logic

The `lattice-core` crate contains all business logic but **never imports I/O primitives**:

- No `std::fs` or file operations
- No `tokio::net` or networking
- No `web_sys` or browser APIs

Instead, core logic defines traits (`LatticeStorage`, `LatticeTransport`) that abstract these operations. Platform-specific crates provide concrete implementations.

### 2. Page-Based Storage Model

All persistent data is organized into fixed-size **pages**:

```rust
// Storage is just pages and metadata
trait LatticeStorage {
    async fn read_page(&self, page_id: u64) -> StorageResult<Page>;
    async fn write_page(&self, page_id: u64, data: &[u8]) -> StorageResult<()>;
    async fn get_meta(&self, key: &str) -> StorageResult<Option<Vec<u8>>>;
    async fn set_meta(&self, key: &str, value: &[u8]) -> StorageResult<()>;
}
```

This maps naturally to different backends:

| Backend | Page Access Pattern |
|---------|---------------------|
| Memory | `HashMap<u64, Vec<u8>>` |
| Disk | `offset = page_id * PAGE_SIZE` |
| OPFS | Same offset-based access |

### 3. Async-First Design

All storage and transport operations are `async`:

```rust
// Works with tokio (native) or wasm-bindgen-futures (browser)
let page = storage.read_page(42).await?;
```

Platform differences are handled via conditional compilation:

```rust
// Native: requires Send + Sync for thread safety
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait LatticeStorage: Send + Sync { ... }

// WASM: single-threaded, no Send bounds
#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait LatticeStorage { ... }
```

## Data Flow

### Vector Search Flow

```
Query Vector
     │
     ▼
┌─────────────┐
│ HNSW Index  │ ← Hierarchical navigation
└─────────────┘
     │
     ▼
┌─────────────┐
│   Layer 0   │ ← Dense candidate selection
└─────────────┘
     │
     ▼
┌─────────────┐
│  Distance   │ ← SIMD-accelerated comparison
│ Calculation │
└─────────────┘
     │
     ▼
┌─────────────┐
│  Top-K      │ ← Priority queue extraction
│  Results    │
└─────────────┘
```

### Hybrid Query Flow

```
Cypher Query: MATCH (n)-[:SIMILAR]->()
              WHERE n.embedding <-> $query < 0.5
     │
     ▼
┌─────────────────┐
│  Cypher Parser  │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Vector Predicate│ ← Recognized as embedding comparison
└─────────────────┘
     │
     ├─────────────────────────┐
     ▼                         ▼
┌─────────────┐        ┌─────────────┐
│ HNSW Search │        │Graph Traverse│
│ (candidates)│        │ (neighbors) │
└─────────────┘        └─────────────┘
     │                         │
     └───────────┬─────────────┘
                 ▼
         ┌─────────────┐
         │   Merge &   │
         │   Filter    │
         └─────────────┘
                 │
                 ▼
         ┌─────────────┐
         │   Results   │
         └─────────────┘
```

## Memory Layout

### Dense Vector Storage

Vectors are stored in a flat, cache-friendly layout:

```
┌────────────────────────────────────────────────────────┐
│ Vector 0: [f32; DIM] │ Vector 1: [f32; DIM] │ ...      │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
           Access: base_ptr + (id * DIM * sizeof(f32))
```

This enables:
- **Cache-efficient sequential access** during index construction
- **SIMD-friendly alignment** for distance calculations
- **Zero-copy access** via memory mapping (native) or typed arrays (WASM)

### HNSW Layer Structure

```
Layer 2:  [sparse entry points]
              │
              ▼
Layer 1:  [medium density nodes]
              │
              ▼
Layer 0:  [all nodes, dense connections]
```

Each layer stores:
- Node IDs present at that layer
- Neighbor lists (connections to other nodes)
- Entry points for search initialization

## Thread Safety (Native)

On native platforms, LatticeDB uses concurrent data structures:

```rust
// Read-write lock for the HNSW index
let index = RwLock<HnswIndex>;

// Multiple readers can search simultaneously
let guard = index.read().await;
let results = guard.search(&query);

// Single writer for mutations
let mut guard = index.write().await;
guard.insert(point);
```

On WASM, all operations are single-threaded, but the same code works because Rust's type system handles the `Send`/`Sync` bounds at compile time.

## Error Handling

All operations return explicit `Result` types:

```rust
pub enum StorageError {
    PageNotFound { page_id: u64 },
    Io { message: String },
    Serialization { message: String },
    ReadOnly,
    CapacityExceeded,
}

pub type StorageResult<T> = Result<T, StorageError>;
```

Errors are never silently swallowed. The `?` operator propagates errors up the call stack, and handlers convert them to appropriate HTTP status codes.

## Next Steps

- [SBIO Pattern](./sbio.md) - Deep dive into the I/O abstraction pattern
- [Crate Structure](./crates.md) - Detailed breakdown of each crate
