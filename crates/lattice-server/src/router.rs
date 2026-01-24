//! Request router - maps HTTP requests to commands
//!
//! Maps Qdrant-compatible endpoints to internal handlers.

use crate::dto::{
    AddEdgeRequest, BatchSearchRequest, CreateCollectionRequest, CypherQueryRequest,
    DeletePointsRequest, GetPointsRequest, QueryRequest, ScrollRequest, SearchRequest,
    TraverseRequest, UpsertPointsRequest,
};
use crate::handlers::{collections, points, search};
use lattice_core::{CollectionEngine, LatticeRequest, LatticeResponse};
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::{debug, instrument, warn};

/// Application state shared across all requests
pub type AppState = Arc<AppStateInner>;

/// Per-collection engine wrapped in its own lock for fine-grained concurrency
pub type CollectionHandle = Arc<RwLock<CollectionEngine>>;

/// Server configuration (PCND: all fields explicit)
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Maximum points per upsert request (prevents memory exhaustion)
    pub max_upsert_batch_size: usize,
    /// Maximum points per delete request
    pub max_delete_batch_size: usize,
    /// Maximum IDs per get request
    pub max_get_batch_size: usize,
    /// Maximum queries per batch search request
    pub max_search_batch_size: usize,
}

impl ServerConfig {
    /// Create config with enterprise defaults
    ///
    /// These defaults are chosen for production safety:
    /// - 10,000 points per upsert (reasonable for high-dim vectors)
    /// - 10,000 IDs per delete/get
    /// - 100 queries per batch search
    pub fn production() -> Self {
        Self {
            max_upsert_batch_size: 10_000,
            max_delete_batch_size: 10_000,
            max_get_batch_size: 10_000,
            max_search_batch_size: 100,
        }
    }

    /// Create config with no limits (for testing only)
    #[cfg(test)]
    pub fn unlimited() -> Self {
        Self {
            max_upsert_batch_size: usize::MAX,
            max_delete_batch_size: usize::MAX,
            max_get_batch_size: usize::MAX,
            max_search_batch_size: usize::MAX,
        }
    }
}

/// Inner application state
///
/// Uses per-collection locking to minimize contention. The outer RwLock is only
/// held briefly to look up or insert collections. Operations on individual
/// collections use the inner per-collection RwLock.
pub struct AppStateInner {
    /// Collection engines keyed by name, each with its own RwLock
    pub collections: RwLock<std::collections::HashMap<String, CollectionHandle>>,
    /// Server configuration
    pub config: ServerConfig,
}

impl AppStateInner {
    /// Create new application state with production config
    pub fn new() -> Self {
        Self {
            collections: RwLock::new(std::collections::HashMap::new()),
            config: ServerConfig::production(),
        }
    }

    /// Create new application state with custom config
    pub fn with_config(config: ServerConfig) -> Self {
        Self {
            collections: RwLock::new(std::collections::HashMap::new()),
            config,
        }
    }

    /// Get a collection handle (fast - only holds outer lock briefly)
    pub fn get_collection(&self, name: &str) -> Option<CollectionHandle> {
        self.collections.read().get(name).cloned()
    }

    /// List collection names
    pub fn list_collection_names(&self) -> Vec<String> {
        self.collections.read().keys().cloned().collect()
    }

    /// Insert a new collection (requires write lock on outer HashMap)
    pub fn insert_collection(&self, name: String, engine: CollectionEngine) -> bool {
        let mut collections = self.collections.write();
        if collections.contains_key(&name) {
            return false;
        }
        collections.insert(name, Arc::new(RwLock::new(engine)));
        true
    }

    /// Remove a collection (requires write lock on outer HashMap)
    pub fn remove_collection(&self, name: &str) -> bool {
        self.collections.write().remove(name).is_some()
    }
}

impl Default for AppStateInner {
    fn default() -> Self {
        Self::new()
    }
}

/// Create new shared application state
pub fn new_app_state() -> AppState {
    Arc::new(AppStateInner::new())
}

/// Route a request to the appropriate handler
///
/// # Endpoints (Qdrant-compatible)
///
/// Collections:
/// - `PUT /collections/{name}` - Create collection
/// - `GET /collections/{name}` - Get collection info
/// - `DELETE /collections/{name}` - Delete collection
/// - `GET /collections` - List collections
///
/// Points:
/// - `PUT /collections/{name}/points` - Upsert points
/// - `POST /collections/{name}/points/search` - Search (legacy)
/// - `POST /collections/{name}/points/query` - Query (Qdrant v1.16+)
/// - `POST /collections/{name}/points/scroll` - Scroll
/// - `POST /collections/{name}/points` - Get points by IDs
/// - `POST /collections/{name}/points/delete` - Delete points
///
/// Graph (LatticeDB extensions):
/// - `POST /collections/{name}/graph/edges` - Add edge
/// - `POST /collections/{name}/graph/traverse` - Traverse graph
/// - `POST /collections/{name}/graph/query` - Execute Cypher query
#[instrument(skip(state, request), fields(method = %request.method, path = %request.path))]
pub async fn route(state: AppState, request: LatticeRequest) -> LatticeResponse {
    // Method is already uppercase from transport layer
    let method = &request.method;
    let path = request.path.trim_end_matches('/');
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

    debug!(method = %method, path = %path, "Processing request");

    let response = match (method.as_str(), segments.as_slice()) {
        // === Health/Diagnostics ===

        // GET /ping - Minimal endpoint for baseline HTTP overhead measurement
        ("GET", ["ping"]) => LatticeResponse::ok(b"{\"status\":\"ok\"}".to_vec()),

        // === Collections ===

        // GET /collections - List all collections
        ("GET", ["collections"]) => collections::list_collections(&state),

        // PUT /collections/{name} - Create collection
        ("PUT", ["collections", name]) => {
            match parse_body::<CreateCollectionRequest>(&request.body) {
                Ok(req) => collections::create_collection(&state, name, req),
                Err(e) => e,
            }
        }

        // GET /collections/{name} - Get collection info
        ("GET", ["collections", name]) => collections::get_collection(&state, name),

        // DELETE /collections/{name} - Delete collection
        ("DELETE", ["collections", name]) => collections::delete_collection(&state, name),

        // === Points ===

        // PUT /collections/{name}/points - Upsert points
        ("PUT", ["collections", name, "points"]) => {
            match parse_body::<UpsertPointsRequest>(&request.body) {
                Ok(req) => points::upsert_points(&state, name, req),
                Err(e) => e,
            }
        }

        // POST /collections/{name}/points/search - Search (legacy)
        ("POST", ["collections", name, "points", "search"]) => {
            match parse_body::<SearchRequest>(&request.body) {
                Ok(req) => search::search_points(&state, name, req),
                Err(e) => e,
            }
        }

        // POST /collections/{name}/points/search/batch - Batch search
        ("POST", ["collections", name, "points", "search", "batch"]) => {
            match parse_body::<BatchSearchRequest>(&request.body) {
                Ok(req) => search::search_batch(&state, name, req),
                Err(e) => e,
            }
        }

        // POST /collections/{name}/points/query - Query (Qdrant v1.16+)
        ("POST", ["collections", name, "points", "query"]) => {
            match parse_body::<QueryRequest>(&request.body) {
                Ok(req) => search::query_points(&state, name, req.into()),
                Err(e) => e,
            }
        }

        // POST /collections/{name}/points/scroll - Scroll
        ("POST", ["collections", name, "points", "scroll"]) => {
            match parse_body::<ScrollRequest>(&request.body) {
                Ok(req) => search::scroll_points(&state, name, req),
                Err(e) => e,
            }
        }

        // POST /collections/{name}/points - Get points by IDs
        ("POST", ["collections", name, "points"]) => {
            match parse_body::<GetPointsRequest>(&request.body) {
                Ok(req) => points::get_points(&state, name, req),
                Err(e) => e,
            }
        }

        // POST /collections/{name}/points/delete - Delete points
        ("POST", ["collections", name, "points", "delete"]) => {
            match parse_body::<DeletePointsRequest>(&request.body) {
                Ok(req) => points::delete_points(&state, name, req),
                Err(e) => e,
            }
        }

        // === Graph Extensions ===

        // POST /collections/{name}/graph/edges - Add edge
        ("POST", ["collections", name, "graph", "edges"]) => {
            match parse_body::<AddEdgeRequest>(&request.body) {
                Ok(req) => points::add_edge(&state, name, req),
                Err(e) => e,
            }
        }

        // POST /collections/{name}/graph/traverse - Traverse graph
        ("POST", ["collections", name, "graph", "traverse"]) => {
            match parse_body::<TraverseRequest>(&request.body) {
                Ok(req) => search::traverse_graph(&state, name, req),
                Err(e) => e,
            }
        }

        // POST /collections/{name}/graph/query - Execute Cypher query
        ("POST", ["collections", name, "graph", "query"]) => {
            match parse_body::<CypherQueryRequest>(&request.body) {
                Ok(req) => search::cypher_query(&state, name, req),
                Err(e) => e,
            }
        }

        // === Fallback ===
        _ => {
            warn!(method = %method, path = %path, "Unknown endpoint");
            LatticeResponse::not_found(&format!("Unknown endpoint: {} {}", method, path))
        }
    };

    // Log response status
    if response.status >= 400 {
        warn!(status = response.status, method = %method, path = %path, "Request failed");
    } else {
        debug!(status = response.status, "Request completed");
    }

    response
}

/// Parse JSON body into a DTO using simd-json when available
#[cfg(feature = "simd-json")]
fn parse_body<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, LatticeResponse> {
    // simd-json requires mutable input for in-place parsing
    let mut buf = body.to_vec();
    simd_json::from_slice(&mut buf)
        .map_err(|e| LatticeResponse::bad_request(&format!("Invalid JSON: {}", e)))
}

#[cfg(not(feature = "simd-json"))]
fn parse_body<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, LatticeResponse> {
    serde_json::from_slice(body)
        .map_err(|e| LatticeResponse::bad_request(&format!("Invalid JSON: {}", e)))
}

/// Serialize response as JSON using simd-json when available
#[cfg(feature = "simd-json")]
pub fn json_response<T: serde::Serialize>(value: &T) -> LatticeResponse {
    match simd_json::to_vec(value) {
        Ok(body) => LatticeResponse::ok(body),
        Err(e) => LatticeResponse::internal_error(&format!("Serialization error: {}", e)),
    }
}

#[cfg(not(feature = "simd-json"))]
pub fn json_response<T: serde::Serialize>(value: &T) -> LatticeResponse {
    match serde_json::to_vec(value) {
        Ok(body) => LatticeResponse::ok(body),
        Err(e) => LatticeResponse::internal_error(&format!("Serialization error: {}", e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    // Async test macro for both native (tokio) and WASM (wasm_bindgen_test)
    macro_rules! async_test {
        ($name:ident, $body:expr) => {
            #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
            async fn $name() {
                $body
            }
        };
    }

    #[test]
    fn test_new_app_state() {
        let state = new_app_state();
        let collections = state.collections.read();
        assert!(collections.is_empty());
    }

    async_test!(test_unknown_endpoint, {
        let state = new_app_state();
        let request = LatticeRequest::new("GET", "/unknown/path");
        let response = route(state, request).await;
        assert_eq!(response.status, 404);
    });

    async_test!(test_list_empty_collections, {
        let state = new_app_state();
        let request = LatticeRequest::new("GET", "/collections");
        let response = route(state, request).await;
        assert_eq!(response.status, 200);
    });

    async_test!(test_invalid_json, {
        let state = new_app_state();
        let request =
            LatticeRequest::new("PUT", "/collections/test").with_body(b"not valid json".to_vec());
        let response = route(state, request).await;
        assert_eq!(response.status, 400);
    });
}
