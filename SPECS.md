LatticeDB Technical SpecificationProject Name: LatticeDBCrate Name: lattice-dbVersion: 0.1.0 (Draft)Tagline: The everywhere database for semantic connections.1. Executive SummaryLatticeDB is a high-performance, converged Vector and Graph database written in pure Rust. It is designed to run anywhere: from high-end cloud servers to the constrained environment of a web browser (WASM).Key Differentiators:Drop-in Qdrant Compatibility: Existing Qdrant SDKs and applications work without modification.Browser-Native: Runs entirely client-side using Web Assembly (WASM) and Origin Private File System (OPFS) for storage.SBIO Architecture: Strict Separation of Business Logic and I/O allows swapping storage backends (Memory, Disk, OPFS) and Transport layers (HTTP, Service Worker) without changing core logic.Zero-Copy: Utilizes rkyv for guaranteed zero-copy deserialization, ensuring maximum throughput.2. Core Principles2.1 SBIO (Separation of Business Logic & I/O)The lattice-core crate must never import std::fs, tokio::net, or web_sys. All external interactions occur through injected Traits.Business Logic: "Given a Query, return a Result."I/O (Abstracted): "Store this page of bytes," "Listen for requests."2.2 Performance FirstZero-Copy: Data read from storage is mapped directly into memory structures using rkyv. No parsing steps.SIMD: Vector math uses std::simd / simd-adler32.Allocations: Minimized heap allocations in the hot path (search).2.3 TDD (Test-Driven Development)Development follows a strict cycle:Define the Interface (Trait).Write a failing test against the In-Memory implementation.Implement the logic to pass the test.Abstract for specific backends (OPFS/Disk).3. Data Architecture3.1 The "Super-Point" ModelThe fundamental unit of storage is the Point. It combines vector data, payload metadata, and graph connections.Serialization: rkyv (Guaranteed Zero-Copy).Alignment: repr(C) for consistent memory layout across WASM boundaries.use rkyv::{Archive, Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;

// Type Aliases for clarity
pub type PointId = u64; // UUIDs are hashed to u64 internally for performance
pub type Vector = Vec<f32>; // Or quantized u8

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[repr(C)] 
pub struct Point {
    /// Unique identifier
    pub id: PointId,

    /// High-dimensional vector data
    pub vector: Vector,

    /// JSON-like payload for filtering
    /// Stored as raw bytes (serialized JSON) to delay parsing until needed
    pub payload: HashMap<String, Vec<u8>>,

    /// OPTIONAL: Graph Connectivity
    /// If None, this is a pure VectorDB node.
    /// SmallVec optimization: stores first 4 edges inline to avoid heap pointer chasing.
    pub outgoing_edges: Option<SmallVec<[Edge; 4]>>,
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[repr(C)]
pub struct Edge {
    pub target_id: PointId,
    pub weight: f32,
    /// Relation type ID (mapped to a string dictionary in collection metadata)
    pub relation_id: u16, 
}
4. Abstraction Interfaces (SBIO)4.1 Storage Layer (LatticeStorage)This trait abstracts the physical medium. It allows the core engine to function identically whether running on an ephemeral In-Memory map, a Server Disk, or Browser OPFS.use async_trait::async_trait;
use anyhow::Result;

/// Represents a raw block of aligned memory
pub type Page = Vec<u8>; 

#[async_trait]
pub trait LatticeStorage: Send + Sync {
    /// Retrieve collection metadata (config, optimizer state)
    async fn get_meta(&self, key: &str) -> Result<Option<Vec<u8>>>;
    
    /// Persist collection metadata
    async fn set_meta(&self, key: &str, value: &[u8]) -> Result<()>;

    /// Read a specific page/chunk of data by ID
    /// In OPFS: Maps to a read at offset (page_id * PAGE_SIZE)
    async fn read_page(&self, page_id: u64) -> Result<Page>;

    /// Write a specific page/chunk of data
    async fn write_page(&self, page_id: u64, data: &[u8]) -> Result<()>;

    /// Check if a page exists
    async fn page_exists(&self, page_id: u64) -> Result<bool>;
    
    /// Delete a page (reclaim space)
    async fn delete_page(&self, page_id: u64) -> Result<()>;
}
Required Implementations:MemStorage: Backed by HashMap<u64, Vec<u8>>. Used for TDD and Unit Tests.OpfsStorage: Backed by web_sys::FileSystemSyncAccessHandle (inside a Worker).DiskStorage: Backed by tokio::fs or Memory-Mapped files.4.2 Transport Layer (LatticeTransport)This trait abstracts the concept of a "Server." It allows the application to listen on a TCP port (Server) or intercept generic fetch requests (Browser).use async_trait::async_trait;
use anyhow::Result;

/// A generic Request object (simplified http::Request)
pub struct LatticeRequest {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
    pub headers: HashMap<String, String>,
}

/// A generic Response object
pub struct LatticeResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

#[async_trait]
pub trait LatticeTransport {
    /// Starts the listener.
    /// @param handler: A function closure that processes a request and returns a response.
    async fn serve<F>(self, handler: F) -> Result<()>
    where
        F: Fn(LatticeRequest) -> impl Future<Output = LatticeResponse> + Send + Sync + 'static;
}
Required Implementations:AxumTransport: Standard HTTP server for Linux/macOS/Windows.ServiceWorkerTransport: Intercepts browser fetch events targeting a virtual domain.5. API Specification5.1 Qdrant Compatibility (V1)LatticeDB acts as a drop-in replacement. It must serialize/deserialize the standard Qdrant JSON schemas.Priority Endpoints:PUT /collections/{name} - Create CollectionGET /collections/{name} - Get InfoPUT /collections/{name}/points - Upsert PointsPOST /collections/{name}/points/search - Vector SearchPOST /collections/{name}/points/scroll - Pagination/Listing5.2 Graph Extensions (Custom)These endpoints extend Qdrant functionality while remaining JSON-compatible.POST /collections/{name}/graph/traverseBody: { "start_point": 123, "max_depth": 2, "relations": ["acted_in"] }POST /collections/{name}/points/{id}/connectBody: { "target": 456, "relation": "is_similar_to", "weight": 0.9 }6. Implementation Roadmap (TDD)

Phase 1: The Core Engine (In-Memory) ✅ COMPLETE
Goal: A fully functional Vector+Graph engine running in RAM.
- ✅ Scaffolding: Cargo workspace (core, server, bench, wasm)
- ✅ Storage TDD: MemStorage implemented and tested
- ✅ Data TDD: Point with rkyv serialization, zero-copy deserialization
- ✅ Index Logic: HNSW index with async background indexing (native)
- ✅ Graph Support: Edges, traversal, and Cypher query language
- ✅ Search Test: Recall benchmarks > 0.95 verified

Phase 2: The Browser Persistence (OPFS) 🟡 PARTIAL
Goal: Persistent storage in the browser.
- ✅ WASM Setup: wasm-bindgen, wasm-pack builds working
- ✅ WASM Core: In-memory CollectionEngine compiles to WASM
- 🔲 OPFS Implementation: OpfsStorage (planned)
- 🔲 Persistence Test: Cross-session data verification (planned)

Phase 3: The API & Transport 🟡 PARTIAL
Goal: Speak HTTP and Qdrant JSON.
- ✅ Router Logic: Full Qdrant-compatible REST API
- ✅ Axum Server: Native HTTP server with rate limiting, TLS, auth
- ✅ Graph Extensions: /graph/edges, /graph/neighbors, /graph/query
- ✅ Cypher Support: Subset of openCypher via /graph/query
- 🔲 Service Worker: Transport layer (planned)
- ✅ Integration: Qdrant-compatible client tested

Phase 4: Optimization ✅ MOSTLY COMPLETE
Goal: Ultra-high performance.
- ✅ SIMD: Optional `simd` feature for accelerated distance calculations
- ✅ Parallel Sort: Rayon-based parallel sorting with adaptive thresholds
- ✅ Async Indexing: Background HNSW updates for fast inserts
- ✅ Zero-Copy: rkyv throughout, Arc<Point> caching in executor
- ✅ Batch Operations: search_batch, batch_extract_properties
- 🔲 Quantization: Int8 scalar quantization (planned)
