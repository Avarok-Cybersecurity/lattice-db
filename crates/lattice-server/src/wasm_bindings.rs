//! WASM bindings for browser JavaScript API
//!
//! This module exposes LatticeDB functionality through wasm-bindgen,
//! enabling direct usage from JavaScript/TypeScript in browsers.
//!
//! # Example (JavaScript)
//!
//! ```javascript
//! import init, { LatticeDB } from 'lattice-db';
//!
//! await init();
//! const db = new LatticeDB();
//!
//! // Create a collection
//! db.createCollection('docs', { vectors: { size: 128, distance: 'Cosine' } });
//!
//! // Insert vectors
//! db.upsert('docs', [
//!     { id: 1, vector: [...], payload: { title: 'Hello' } }
//! ]);
//!
//! // Search
//! const results = db.search('docs', queryVector, 10);
//! ```

use crate::dto::{
    AddEdgeRequest, CreateCollectionRequest, CypherQueryRequest, DeletePointsRequest,
    GetPointsRequest, PointStruct, ScrollRequest, SearchRequest, TraverseRequest,
    UpsertPointsRequest,
};
use crate::handlers::{collections, points, search};
use crate::router::{new_app_state, AppState};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// LatticeDB browser database instance
///
/// Main entry point for browser applications. Wraps the internal state
/// and provides a JavaScript-friendly API.
#[wasm_bindgen]
pub struct LatticeDB {
    state: AppState,
}

#[wasm_bindgen]
impl LatticeDB {
    /// Create a new LatticeDB instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // Initialize console_error_panic_hook for better error messages in browser
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        Self {
            state: new_app_state(),
        }
    }

    // === Collections ===

    /// Create a new collection
    ///
    /// # Arguments
    /// * `name` - Collection name
    /// * `config` - Configuration object: `{ vectors: { size: number, distance: string }, hnsw_config?: {...} }`
    ///
    /// # Returns
    /// Result object with status
    #[wasm_bindgen(js_name = createCollection)]
    pub fn create_collection(&mut self, name: &str, config: JsValue) -> Result<JsValue, JsValue> {
        let request: CreateCollectionRequest = serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

        let response = collections::create_collection(&self.state, name, request);
        response_to_js(response)
    }

    /// List all collections
    ///
    /// # Returns
    /// Array of collection names
    #[wasm_bindgen(js_name = listCollections)]
    pub fn list_collections(&self) -> Result<JsValue, JsValue> {
        let response = collections::list_collections(&self.state);
        response_to_js(response)
    }

    /// Get collection info
    ///
    /// # Arguments
    /// * `name` - Collection name
    ///
    /// # Returns
    /// Collection info object
    #[wasm_bindgen(js_name = getCollection)]
    pub fn get_collection(&self, name: &str) -> Result<JsValue, JsValue> {
        let response = collections::get_collection(&self.state, name);
        response_to_js(response)
    }

    /// Delete a collection
    ///
    /// # Arguments
    /// * `name` - Collection name to delete
    ///
    /// # Returns
    /// Boolean indicating success
    #[wasm_bindgen(js_name = deleteCollection)]
    pub fn delete_collection(&mut self, name: &str) -> Result<bool, JsValue> {
        let response = collections::delete_collection(&self.state, name);
        if response.status == 200 {
            Ok(true)
        } else if response.status == 404 {
            Ok(false)
        } else {
            Err(JsValue::from_str(&String::from_utf8_lossy(&response.body)))
        }
    }

    // === Points ===

    /// Upsert points into a collection
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `points` - Array of points: `[{ id: number, vector: number[], payload?: object }, ...]`
    ///
    /// # Returns
    /// Upsert result with operation status
    #[wasm_bindgen]
    pub fn upsert(&mut self, collection: &str, points_js: JsValue) -> Result<JsValue, JsValue> {
        let points: Vec<PointInput> = serde_wasm_bindgen::from_value(points_js)
            .map_err(|e| JsValue::from_str(&format!("Invalid points: {}", e)))?;

        let request = UpsertPointsRequest {
            points: points.into_iter().map(|p| p.into()).collect(),
        };

        let response = points::upsert_points(&self.state, collection, request);
        response_to_js(response)
    }

    /// Get points by IDs
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `ids` - Array of point IDs
    /// * `with_payload` - Include payload in results (default: true)
    /// * `with_vector` - Include vector in results (default: false)
    ///
    /// # Returns
    /// Array of point records
    #[wasm_bindgen(js_name = getPoints)]
    pub fn get_points(
        &self,
        collection: &str,
        ids: Vec<u64>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<JsValue, JsValue> {
        let request = GetPointsRequest {
            ids,
            with_payload: with_payload.unwrap_or(true),
            with_vector: with_vector.unwrap_or(false),
        };

        let response = points::get_points(&self.state, collection, request);
        response_to_js(response)
    }

    /// Delete points by IDs
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `ids` - Array of point IDs to delete
    ///
    /// # Returns
    /// Number of points deleted
    #[wasm_bindgen(js_name = deletePoints)]
    pub fn delete_points(&mut self, collection: &str, ids: Vec<u64>) -> Result<JsValue, JsValue> {
        let request = DeletePointsRequest { points: ids };
        let response = points::delete_points(&self.state, collection, request);
        response_to_js(response)
    }

    // === Search ===

    /// Search for nearest neighbors
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `vector` - Query vector
    /// * `limit` - Maximum number of results
    /// * `options` - Optional search options: `{ with_payload?: boolean, with_vector?: boolean, score_threshold?: number }`
    ///
    /// # Returns
    /// Array of search results: `[{ id: number, score: number, payload?: object, vector?: number[] }, ...]`
    #[wasm_bindgen]
    pub fn search(
        &self,
        collection: &str,
        vector: Vec<f32>,
        limit: usize,
        options: Option<JsValue>,
    ) -> Result<JsValue, JsValue> {
        let opts: SearchOptions = if let Some(opts_js) = options {
            serde_wasm_bindgen::from_value(opts_js)
                .map_err(|e| JsValue::from_str(&format!("Invalid options: {}", e)))?
        } else {
            SearchOptions::default()
        };

        let request = SearchRequest {
            vector,
            limit,
            with_payload: opts.with_payload.unwrap_or(true),
            with_vector: opts.with_vector.unwrap_or(false),
            params: None,
            score_threshold: opts.score_threshold,
        };

        let response = search::search_points(&self.state, collection, request);
        response_to_js(response)
    }

    /// Scroll through all points
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `options` - Scroll options: `{ limit?: number, offset?: number, with_payload?: boolean, with_vector?: boolean }`
    ///
    /// # Returns
    /// Scroll result with points and next offset
    #[wasm_bindgen]
    pub fn scroll(&self, collection: &str, options: Option<JsValue>) -> Result<JsValue, JsValue> {
        let opts: ScrollOptions = if let Some(opts_js) = options {
            serde_wasm_bindgen::from_value(opts_js)
                .map_err(|e| JsValue::from_str(&format!("Invalid options: {}", e)))?
        } else {
            ScrollOptions::default()
        };

        let request = ScrollRequest {
            limit: opts.limit.unwrap_or(10),
            offset: opts.offset,
            with_payload: opts.with_payload.unwrap_or(true),
            with_vector: opts.with_vector.unwrap_or(false),
        };

        let response = search::scroll_points(&self.state, collection, request);
        response_to_js(response)
    }

    // === Graph ===

    /// Add an edge between two points
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `from_id` - Source point ID
    /// * `to_id` - Target point ID
    /// * `relation` - Relation type name
    /// * `weight` - Optional edge weight (default: 1.0)
    #[wasm_bindgen(js_name = addEdge)]
    pub fn add_edge(
        &mut self,
        collection: &str,
        from_id: u64,
        to_id: u64,
        relation: &str,
        weight: Option<f32>,
    ) -> Result<(), JsValue> {
        let request = AddEdgeRequest {
            from_id,
            to_id,
            relation: relation.to_string(),
            weight: weight.unwrap_or(1.0),
        };

        let response = points::add_edge(&self.state, collection, request);
        if response.status == 200 {
            Ok(())
        } else {
            Err(JsValue::from_str(&String::from_utf8_lossy(&response.body)))
        }
    }

    /// Traverse the graph from a starting point
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `start_id` - Starting point ID
    /// * `max_depth` - Maximum traversal depth
    /// * `relations` - Optional array of relation types to filter
    ///
    /// # Returns
    /// Traversal result with visited nodes and edges
    #[wasm_bindgen]
    pub fn traverse(
        &self,
        collection: &str,
        start_id: u64,
        max_depth: usize,
        relations: Option<Vec<String>>,
    ) -> Result<JsValue, JsValue> {
        let request = TraverseRequest {
            start_id,
            max_depth,
            relations,
        };

        let response = search::traverse_graph(&self.state, collection, request);
        response_to_js(response)
    }

    /// Execute a Cypher query
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `cypher` - Cypher query string
    /// * `parameters` - Optional query parameters
    ///
    /// # Returns
    /// Query result with columns and rows
    #[wasm_bindgen]
    pub fn query(
        &self,
        collection: &str,
        cypher: &str,
        parameters: Option<JsValue>,
    ) -> Result<JsValue, JsValue> {
        let params: HashMap<String, serde_json::Value> = if let Some(params_js) = parameters {
            serde_wasm_bindgen::from_value(params_js)
                .map_err(|e| JsValue::from_str(&format!("Invalid parameters: {}", e)))?
        } else {
            HashMap::new()
        };

        let request = CypherQueryRequest {
            query: cypher.to_string(),
            parameters: params,
        };

        let response = search::cypher_query(&self.state, collection, request);
        response_to_js(response)
    }
}

impl Default for LatticeDB {
    fn default() -> Self {
        Self::new()
    }
}

// === Helper Types ===

/// Input point structure for JavaScript interop
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PointInput {
    id: u64,
    vector: Vec<f32>,
    #[serde(default)]
    payload: Option<HashMap<String, serde_json::Value>>,
}

impl From<PointInput> for PointStruct {
    fn from(p: PointInput) -> Self {
        PointStruct {
            id: p.id,
            vector: p.vector,
            payload: p.payload,
        }
    }
}

/// Search options for JavaScript interop
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct SearchOptions {
    with_payload: Option<bool>,
    with_vector: Option<bool>,
    score_threshold: Option<f32>,
}

/// Scroll options for JavaScript interop
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ScrollOptions {
    limit: Option<usize>,
    offset: Option<u64>,
    with_payload: Option<bool>,
    with_vector: Option<bool>,
}

/// Convert LatticeResponse to JsValue
fn response_to_js(response: lattice_core::LatticeResponse) -> Result<JsValue, JsValue> {
    if response.status >= 200 && response.status < 300 {
        // Parse the JSON body and convert to JsValue
        let value: serde_json::Value = serde_json::from_slice(&response.body)
            .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;
        serde_wasm_bindgen::to_value(&value)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    } else {
        // Return error with message from body
        let msg = String::from_utf8_lossy(&response.body);
        Err(JsValue::from_str(&msg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn test_create_lattice_db() {
        let _db = LatticeDB::new();
    }
}
