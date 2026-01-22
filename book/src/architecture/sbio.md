# SBIO Pattern

**Separation of Business Logic and I/O (SBIO)** is the architectural pattern that enables LatticeDB to run on both native servers and WebAssembly browsers from a single codebase.

## The Problem

Traditional database code mixes business logic with I/O operations:

```rust
// Bad: Business logic directly performs I/O
fn search(&self, query: &[f32]) -> Vec<SearchResult> {
    // Direct file system access - won't work in browser!
    let index_data = std::fs::read("index.bin").unwrap();
    let index: HnswIndex = deserialize(&index_data);

    // Direct network call - different API on each platform
    let embeddings = reqwest::get("http://model-server/embed")
        .await.unwrap();

    index.search(query)
}
```

This code has several problems:
1. **Platform-specific APIs**: `std::fs` doesn't exist in WASM
2. **Tight coupling**: Can't swap storage backends
3. **Untestable**: Hard to mock file system in tests
4. **Error handling**: `unwrap()` hides failures

## The Solution

SBIO separates concerns into three layers:

```
┌─────────────────────────────────────────┐
│        Business Logic (Pure)             │
│   - No I/O imports                       │
│   - Defines traits for dependencies      │
│   - Contains all algorithms              │
├─────────────────────────────────────────┤
│        Abstraction Boundary              │
│   - LatticeStorage trait                 │
│   - LatticeTransport trait               │
├─────────────────────────────────────────┤
│     Platform Implementations             │
│   - DiskStorage / OpfsStorage            │
│   - AxumTransport / ServiceWorker        │
└─────────────────────────────────────────┘
```

## Storage Trait

The `LatticeStorage` trait abstracts all persistent storage:

```rust
/// Abstract storage interface (SBIO boundary)
#[async_trait]
pub trait LatticeStorage: Send + Sync {
    /// Retrieve metadata by key
    async fn get_meta(&self, key: &str) -> StorageResult<Option<Vec<u8>>>;

    /// Store metadata
    async fn set_meta(&self, key: &str, value: &[u8]) -> StorageResult<()>;

    /// Read a page by ID
    async fn read_page(&self, page_id: u64) -> StorageResult<Page>;

    /// Write a page (create or overwrite)
    async fn write_page(&self, page_id: u64, data: &[u8]) -> StorageResult<()>;

    /// Flush pending writes to durable storage
    async fn sync(&self) -> StorageResult<()>;
}
```

### Why Pages?

The page-based model is chosen because it maps naturally to all storage backends:

| Backend | Page Implementation |
|---------|---------------------|
| **MemStorage** | `HashMap<u64, Vec<u8>>` - pages are hash map entries |
| **DiskStorage** | File with `offset = page_id * PAGE_SIZE` |
| **OpfsStorage** | OPFS file with same offset calculation |
| **IndexedDB** | Could use page_id as object store key |

This abstraction is low-level enough to be efficient but high-level enough to hide platform differences.

## Transport Trait

The `LatticeTransport` trait abstracts the "server" concept:

```rust
/// Abstract transport interface (SBIO boundary)
#[async_trait]
pub trait LatticeTransport: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Start serving requests
    async fn serve<H, Fut>(self, handler: H) -> Result<(), Self::Error>
    where
        H: Fn(LatticeRequest) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = LatticeResponse> + Send + 'static;
}
```

The request/response types are platform-agnostic:

```rust
pub struct LatticeRequest {
    pub method: String,      // GET, POST, PUT, DELETE
    pub path: String,        // /collections/{name}/points
    pub body: Vec<u8>,       // JSON payload
    pub headers: HashMap<String, String>,
}

pub struct LatticeResponse {
    pub status: u16,         // HTTP status code
    pub body: Vec<u8>,       // JSON response
    pub headers: HashMap<String, String>,
}
```

## Platform Implementations

### Native (Server)

```rust
// DiskStorage: File-based storage using tokio::fs
pub struct DiskStorage {
    data_file: tokio::fs::File,
    meta_file: tokio::fs::File,
}

impl LatticeStorage for DiskStorage {
    async fn read_page(&self, page_id: u64) -> StorageResult<Page> {
        let offset = page_id * PAGE_SIZE;
        self.data_file.seek(SeekFrom::Start(offset)).await?;
        let mut buf = vec![0u8; PAGE_SIZE];
        self.data_file.read_exact(&mut buf).await?;
        Ok(buf)
    }
}

// AxumTransport: HTTP server using Axum
pub struct AxumTransport {
    bind_addr: SocketAddr,
}

impl LatticeTransport for AxumTransport {
    async fn serve<H, Fut>(self, handler: H) -> Result<(), Self::Error> {
        let app = Router::new()
            .fallback(move |req| convert_and_call(handler.clone(), req));
        axum::Server::bind(&self.bind_addr)
            .serve(app.into_make_service())
            .await
    }
}
```

### WASM (Browser)

```rust
// OpfsStorage: Origin Private File System
pub struct OpfsStorage {
    root: web_sys::FileSystemDirectoryHandle,
}

impl LatticeStorage for OpfsStorage {
    async fn read_page(&self, page_id: u64) -> StorageResult<Page> {
        let file = self.root.get_file_handle("data.bin").await?;
        let blob = file.get_file().await?;
        let offset = page_id * PAGE_SIZE;
        let slice = blob.slice_with_i32_and_i32(offset, offset + PAGE_SIZE)?;
        let array_buffer = slice.array_buffer().await?;
        Ok(js_sys::Uint8Array::new(&array_buffer).to_vec())
    }
}

// ServiceWorkerTransport: Fetch event interception
pub struct ServiceWorkerTransport;

impl LatticeTransport for ServiceWorkerTransport {
    async fn serve<H, Fut>(self, handler: H) -> Result<(), Self::Error> {
        let closure = Closure::wrap(Box::new(move |event: FetchEvent| {
            let request = convert_fetch_to_lattice(event.request());
            let future = handler(request);
            event.respond_with(&future_to_promise(async move {
                let response = future.await;
                convert_lattice_to_fetch(response)
            }));
        }));

        // Register for fetch events
        js_sys::global()
            .add_event_listener_with_callback("fetch", closure.as_ref())
    }
}
```

## WASM Conditional Compilation

The traits have different bounds for native vs WASM:

```rust
// Native: requires Send + Sync for multi-threaded runtime
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait LatticeStorage: Send + Sync { ... }

// WASM: single-threaded, no Send bounds needed
#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait LatticeStorage { ... }
```

The `?Send` annotation tells `async_trait` that the futures don't need to be `Send`, which is required because JavaScript's `Promise` is not `Send`.

## Benefits

### 1. Testability

Business logic can be tested with `MemStorage`:

```rust
#[tokio::test]
async fn test_search() {
    let storage = MemStorage::new();
    let engine = CollectionEngine::new(config, storage);

    engine.upsert(point).await.unwrap();
    let results = engine.search(&query).await.unwrap();

    assert_eq!(results.len(), 1);
}
```

No file system setup, no cleanup, no flaky tests.

### 2. Single Codebase

The same `CollectionEngine` code works everywhere:

```rust
// Server
let engine = CollectionEngine::new(config, DiskStorage::new(path));

// Browser
let engine = CollectionEngine::new(config, OpfsStorage::new());

// Tests
let engine = CollectionEngine::new(config, MemStorage::new());
```

### 3. Explicit Dependencies

All I/O dependencies are visible in function signatures:

```rust
// Clear: this function needs storage
async fn build_index<S: LatticeStorage>(storage: &S) { ... }

// Hidden: what I/O does this do?
async fn build_index() { ... }  // Bad!
```

### 4. Error Propagation

Explicit `Result` types force error handling:

```rust
pub enum StorageError {
    PageNotFound { page_id: u64 },
    Io { message: String },
    Serialization { message: String },
    ReadOnly,
    CapacityExceeded,
}

// Errors propagate via ?
let page = storage.read_page(42)?;
```

## Common Patterns

### Dependency Injection

```rust
pub struct CollectionEngine<S: LatticeStorage> {
    storage: S,
    index: HnswIndex,
}

impl<S: LatticeStorage> CollectionEngine<S> {
    pub fn new(config: CollectionConfig, storage: S) -> Self {
        Self { storage, index: HnswIndex::new(config) }
    }
}
```

### Factory Functions

```rust
// Platform-specific factory
#[cfg(not(target_arch = "wasm32"))]
pub fn create_storage(path: &Path) -> impl LatticeStorage {
    DiskStorage::new(path)
}

#[cfg(target_arch = "wasm32")]
pub fn create_storage() -> impl LatticeStorage {
    OpfsStorage::new()
}
```

## Next Steps

- [Crate Structure](./crates.md) - How the codebase is organized
- [HNSW Index](../vector/hnsw.md) - Vector search implementation
