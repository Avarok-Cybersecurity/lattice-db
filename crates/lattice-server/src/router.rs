//! Request router - maps HTTP requests to commands
//!
//! Maps Qdrant-compatible endpoints to internal handlers.

use crate::dto::{
    AddEdgeRequest, BatchSearchRequest, CreateCollectionRequest, CypherQueryRequest,
    DeletePointsRequest, GetPointsRequest, QueryRequest, ScrollRequest, SearchRequest,
    TraverseRequest, UpsertPointsRequest,
};
use crate::handlers::{collections, points, search};
use lattice_core::{LatticeRequest, LatticeResponse};
use std::sync::Arc;

/// Application state shared across all requests
pub type AppState = Arc<AppStateInner>;

/// Inner application state
pub struct AppStateInner {
    /// Collection engines keyed by name
    pub collections: std::sync::RwLock<
        std::collections::HashMap<String, lattice_core::CollectionEngine>,
    >,
}

impl AppStateInner {
    /// Create new application state
    pub fn new() -> Self {
        Self {
            collections: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
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
pub async fn route(state: AppState, request: LatticeRequest) -> LatticeResponse {
    let method = request.method.to_uppercase();
    let path = request.path.trim_end_matches('/');
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

    match (method.as_str(), segments.as_slice()) {
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
        _ => LatticeResponse::not_found(&format!("Unknown endpoint: {} {}", method, path)),
    }
}

/// Parse JSON body into a DTO
fn parse_body<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, LatticeResponse> {
    serde_json::from_slice(body).map_err(|e| {
        LatticeResponse::bad_request(&format!("Invalid JSON: {}", e))
    })
}

/// Serialize response as JSON
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
        let collections = state.collections.read().unwrap();
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
        let request = LatticeRequest::new("PUT", "/collections/test")
            .with_body(b"not valid json".to_vec());
        let response = route(state, request).await;
        assert_eq!(response.status, 400);
    });
}
