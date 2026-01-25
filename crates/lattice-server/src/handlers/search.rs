//! Search handlers
//!
//! Handles search, scroll, graph traversal, and Cypher query operations.

use crate::dto::{
    ApiResponse, BatchSearchRequest, CypherQueryRequest, CypherQueryResponse, CypherQueryStats,
    EdgeRecord, PointRecord, QueryResponse, ScoredPoint, ScrollRequest, ScrollResult,
    SearchRequest, TraversalResult, TraverseRequest,
};
#[cfg(feature = "openapi")]
use crate::dto::{
    ApiResponseCypherQueryResponse, ApiResponseScrollResult, ApiResponseTraversalResult,
    ApiResponseVecScoredPoint,
};
use crate::router::{json_response, AppState};
use lattice_core::cypher::{CypherHandler, DefaultCypherHandler};
use lattice_core::CypherValue;
use lattice_core::{LatticeResponse, ScrollQuery, SearchQuery};
use std::collections::HashMap;

// Use wasmtimer for WASM, std::time for native
#[cfg(target_arch = "wasm32")]
use wasmtimer::std::Instant;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

/// Search for nearest neighbors
#[cfg_attr(feature = "openapi", utoipa::path(
    post,
    path = "/collections/{collection_name}/points/search",
    tag = "Search",
    params(
        ("collection_name" = String, Path, description = "Collection name")
    ),
    request_body = SearchRequest,
    responses(
        (status = 200, description = "Search results", body = ApiResponseVecScoredPoint),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Collection not found")
    )
))]
pub fn search_points(
    state: &AppState,
    collection_name: &str,
    request: SearchRequest,
) -> LatticeResponse {
    let handle = match state.get_collection(collection_name) {
        Some(h) => h,
        None => {
            return LatticeResponse::not_found(&format!(
                "Collection '{}' not found",
                collection_name
            ))
        }
    };
    let engine = handle.read();

    // Build search query
    let mut query = SearchQuery::new(request.vector, request.limit);

    if request.with_payload {
        query = query.include_payload();
    }
    if request.with_vector {
        query = query.include_vector();
    }
    if let Some(threshold) = request.score_threshold {
        query = query.with_score_threshold(threshold);
    }
    if let Some(params) = request.params {
        if let Some(ef) = params.ef {
            query = query.with_ef(ef);
        }
    }

    // Execute search
    match engine.search(query) {
        Ok(results) => {
            let scored_points: Vec<ScoredPoint> = results
                .into_iter()
                .map(|r| ScoredPoint {
                    id: r.id,
                    version: 0, // LatticeDB doesn't track versions
                    score: r.score,
                    payload: r.payload.map(json_value_to_map),
                    vector: r.vector,
                })
                .collect();

            json_response(&ApiResponse::ok(scored_points))
        }
        Err(e) => LatticeResponse::bad_request(&format!("Search failed: {}", e)),
    }
}

/// Batch search for multiple queries (parallel processing)
///
/// More efficient than calling search multiple times.
pub fn search_batch(
    state: &AppState,
    collection_name: &str,
    request: BatchSearchRequest,
) -> LatticeResponse {
    // Validate batch size to prevent memory exhaustion
    let max_batch = state.config.max_search_batch_size;
    if request.searches.len() > max_batch {
        return LatticeResponse::bad_request(&format!(
            "Batch size {} exceeds maximum allowed {} queries per request",
            request.searches.len(),
            max_batch
        ));
    }

    let handle = match state.get_collection(collection_name) {
        Some(h) => h,
        None => {
            return LatticeResponse::not_found(&format!(
                "Collection '{}' not found",
                collection_name
            ))
        }
    };
    let engine = handle.read();

    // Convert SearchRequest DTOs to SearchQuery
    let queries: Vec<SearchQuery> = request
        .searches
        .iter()
        .map(|req| {
            let mut query = SearchQuery::new(req.vector.clone(), req.limit);
            if req.with_payload {
                query = query.include_payload();
            }
            if req.with_vector {
                query = query.include_vector();
            }
            if let Some(threshold) = req.score_threshold {
                query = query.with_score_threshold(threshold);
            }
            if let Some(ref params) = req.params {
                if let Some(ef) = params.ef {
                    query = query.with_ef(ef);
                }
            }
            query
        })
        .collect();

    // Execute batch search
    match engine.search_batch(queries) {
        Ok(all_results) => {
            // Convert each result set to ScoredPoint
            let batch_results: Vec<Vec<ScoredPoint>> = all_results
                .into_iter()
                .map(|results| {
                    results
                        .into_iter()
                        .map(|r| ScoredPoint {
                            id: r.id,
                            version: 0,
                            score: r.score,
                            payload: r.payload.map(json_value_to_map),
                            vector: r.vector,
                        })
                        .collect()
                })
                .collect();

            json_response(&ApiResponse::ok(batch_results))
        }
        Err(e) => LatticeResponse::bad_request(&format!("Batch search failed: {}", e)),
    }
}

/// Query points (Qdrant v1.16+ unified search endpoint)
///
/// This is the newer unified search endpoint. It wraps results in a QueryResponse.
pub fn query_points(
    state: &AppState,
    collection_name: &str,
    request: SearchRequest,
) -> LatticeResponse {
    let handle = match state.get_collection(collection_name) {
        Some(h) => h,
        None => {
            return LatticeResponse::not_found(&format!(
                "Collection '{}' not found",
                collection_name
            ))
        }
    };
    let engine = handle.read();

    // Build search query
    let mut query = SearchQuery::new(request.vector, request.limit);

    if request.with_payload {
        query = query.include_payload();
    }
    if request.with_vector {
        query = query.include_vector();
    }
    if let Some(threshold) = request.score_threshold {
        query = query.with_score_threshold(threshold);
    }
    if let Some(params) = request.params {
        if let Some(ef) = params.ef {
            query = query.with_ef(ef);
        }
    }

    // Execute search
    match engine.search(query) {
        Ok(results) => {
            let scored_points: Vec<ScoredPoint> = results
                .into_iter()
                .map(|r| ScoredPoint {
                    id: r.id,
                    version: 0, // LatticeDB doesn't track versions
                    score: r.score,
                    payload: r.payload.map(json_value_to_map),
                    vector: r.vector,
                })
                .collect();

            // Wrap in QueryResponse for v1.16+ compatibility
            json_response(&ApiResponse::ok(QueryResponse {
                points: scored_points,
            }))
        }
        Err(e) => LatticeResponse::bad_request(&format!("Search failed: {}", e)),
    }
}

/// Scroll through points (paginated retrieval)
#[cfg_attr(feature = "openapi", utoipa::path(
    post,
    path = "/collections/{collection_name}/points/scroll",
    tag = "Search",
    params(
        ("collection_name" = String, Path, description = "Collection name")
    ),
    request_body = ScrollRequest,
    responses(
        (status = 200, description = "Scroll results", body = ApiResponseScrollResult),
        (status = 404, description = "Collection not found")
    )
))]
pub fn scroll_points(
    state: &AppState,
    collection_name: &str,
    request: ScrollRequest,
) -> LatticeResponse {
    // Validate scroll limit (DoS protection)
    const MAX_SCROLL_LIMIT: usize = 10_000;
    if request.limit == 0 {
        return LatticeResponse::bad_request("Scroll limit must be at least 1");
    }
    if request.limit > MAX_SCROLL_LIMIT {
        return LatticeResponse::bad_request(&format!(
            "Scroll limit {} exceeds maximum of {}",
            request.limit, MAX_SCROLL_LIMIT
        ));
    }

    let handle = match state.get_collection(collection_name) {
        Some(h) => h,
        None => {
            return LatticeResponse::not_found(&format!(
                "Collection '{}' not found",
                collection_name
            ))
        }
    };
    let engine = handle.read();

    // Build scroll query
    let mut query = ScrollQuery::new(request.limit);

    if let Some(offset) = request.offset {
        query = query.with_offset(offset);
    }
    if request.with_payload {
        query = query.include_payload();
    }
    if request.with_vector {
        query = query.include_vector();
    }

    // Execute scroll
    let result = match engine.scroll(query) {
        Ok(r) => r,
        Err(e) => return LatticeResponse::bad_request(&format!("Scroll failed: {}", e)),
    };

    let scroll_result = ScrollResult {
        points: result
            .points
            .into_iter()
            .map(|p| PointRecord {
                id: p.id,
                payload: p.payload.map(json_value_to_map),
                vector: p.vector,
            })
            .collect(),
        next_page_offset: result.next_offset,
    };

    json_response(&ApiResponse::ok(scroll_result))
}

/// Traverse the graph from a starting point
#[cfg_attr(feature = "openapi", utoipa::path(
    post,
    path = "/collections/{collection_name}/graph/traverse",
    tag = "Graph",
    params(
        ("collection_name" = String, Path, description = "Collection name")
    ),
    request_body = TraverseRequest,
    responses(
        (status = 200, description = "Traversal results", body = ApiResponseTraversalResult),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Collection not found")
    )
))]
pub fn traverse_graph(
    state: &AppState,
    collection_name: &str,
    request: TraverseRequest,
) -> LatticeResponse {
    let handle = match state.get_collection(collection_name) {
        Some(h) => h,
        None => {
            return LatticeResponse::not_found(&format!(
                "Collection '{}' not found",
                collection_name
            ))
        }
    };
    let engine = handle.read();

    // Convert relation strings to references for the API
    let relations: Option<Vec<&str>> = request
        .relations
        .as_ref()
        .map(|rels| rels.iter().map(|s| s.as_str()).collect());

    // Execute traversal
    match engine.traverse(request.start_id, request.max_depth, relations.as_deref()) {
        Ok(result) => {
            // Build edge records from paths, resolving relation_id to name
            let config = engine.config();
            let mut edges = Vec::new();
            for path in &result.paths {
                if path.path.len() >= 2 {
                    let from_id = path.path[path.path.len() - 2];
                    let relation = config
                        .relation_name(path.relation_id)
                        .unwrap_or("unknown")
                        .to_string();
                    edges.push(EdgeRecord {
                        from_id,
                        to_id: path.target_id,
                        relation,
                        weight: path.weight,
                    });
                }
            }

            let traversal_result = TraversalResult {
                visited: result.paths.iter().map(|p| p.target_id).collect(),
                edges,
                max_depth_reached: result.paths.iter().map(|p| p.depth).max().unwrap_or(0),
            };

            json_response(&ApiResponse::ok(traversal_result))
        }
        Err(e) => LatticeResponse::bad_request(&format!("Traversal failed: {}", e)),
    }
}

/// Execute a Cypher query (LatticeDB extension)
#[cfg_attr(feature = "openapi", utoipa::path(
    post,
    path = "/collections/{collection_name}/graph/query",
    tag = "Graph",
    params(
        ("collection_name" = String, Path, description = "Collection name")
    ),
    request_body = CypherQueryRequest,
    responses(
        (status = 200, description = "Query results", body = ApiResponseCypherQueryResponse),
        (status = 400, description = "Invalid query"),
        (status = 404, description = "Collection not found")
    )
))]
pub fn cypher_query(
    state: &AppState,
    collection_name: &str,
    request: CypherQueryRequest,
) -> LatticeResponse {
    // Validate query length to prevent DoS via parser exhaustion
    const MAX_QUERY_LENGTH: usize = 100_000; // 100KB
    if request.query.len() > MAX_QUERY_LENGTH {
        return LatticeResponse::bad_request(&format!(
            "Query length {} exceeds maximum of {} bytes",
            request.query.len(),
            MAX_QUERY_LENGTH
        ));
    }

    // Validate parameter count (DoS protection)
    const MAX_PARAMETERS: usize = 10_000;
    if request.parameters.len() > MAX_PARAMETERS {
        return LatticeResponse::bad_request(&format!(
            "Parameter count {} exceeds maximum of {}",
            request.parameters.len(),
            MAX_PARAMETERS
        ));
    }

    let handle = match state.get_collection(collection_name) {
        Some(h) => h,
        None => {
            return LatticeResponse::not_found(&format!(
                "Collection '{}' not found",
                collection_name
            ))
        }
    };
    let mut engine = handle.write();

    // Create the Cypher handler
    let handler = DefaultCypherHandler::new();

    // Convert JSON parameters to CypherValue
    let parameters: HashMap<String, CypherValue> = request
        .parameters
        .into_iter()
        .map(|(k, v)| (k, json_to_cypher_value(v)))
        .collect();

    // Record start time
    let start = Instant::now();

    // Execute the query
    match handler.query(&request.query, &mut engine, parameters) {
        Ok(result) => {
            let execution_time_ms = start.elapsed().as_millis() as u64;

            // Convert CypherValue rows to JSON
            let rows: Vec<Vec<serde_json::Value>> = result
                .rows
                .into_iter()
                .map(|row| row.into_iter().map(cypher_value_to_json).collect())
                .collect();

            let response = CypherQueryResponse {
                columns: result.columns,
                rows,
                stats: CypherQueryStats {
                    nodes_created: result.stats.nodes_created as u64,
                    relationships_created: result.stats.relationships_created as u64,
                    nodes_deleted: result.stats.nodes_deleted as u64,
                    relationships_deleted: result.stats.relationships_deleted as u64,
                    properties_set: result.stats.properties_set as u64,
                    execution_time_ms,
                },
            };

            json_response(&ApiResponse::ok(response))
        }
        Err(e) => LatticeResponse::bad_request(&format!("Cypher query failed: {}", e)),
    }
}

/// Maximum JSON nesting depth to prevent stack overflow
const MAX_JSON_DEPTH: usize = 64;

/// Convert JSON value to CypherValue with depth limit
fn json_to_cypher_value(value: serde_json::Value) -> CypherValue {
    json_to_cypher_value_with_depth(value, 0)
}

/// Convert JSON value to CypherValue with depth tracking
fn json_to_cypher_value_with_depth(value: serde_json::Value, depth: usize) -> CypherValue {
    if depth > MAX_JSON_DEPTH {
        // Return Null for excessively nested values (DoS protection)
        return CypherValue::Null;
    }

    match value {
        serde_json::Value::Null => CypherValue::Null,
        serde_json::Value::Bool(b) => CypherValue::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                CypherValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                CypherValue::Float(f)
            } else {
                CypherValue::Null
            }
        }
        serde_json::Value::String(s) => CypherValue::String(s.into()),
        serde_json::Value::Array(arr) => CypherValue::List(
            arr.into_iter()
                .map(|v| json_to_cypher_value_with_depth(v, depth + 1))
                .collect(),
        ),
        serde_json::Value::Object(obj) => CypherValue::Map(
            obj.into_iter()
                .map(|(k, v)| (k.into(), json_to_cypher_value_with_depth(v, depth + 1)))
                .collect(),
        ),
    }
}

/// Convert CypherValue to JSON value with depth limit
fn cypher_value_to_json(value: CypherValue) -> serde_json::Value {
    cypher_value_to_json_with_depth(value, 0)
}

/// Convert CypherValue to JSON value with depth tracking
fn cypher_value_to_json_with_depth(value: CypherValue, depth: usize) -> serde_json::Value {
    if depth > MAX_JSON_DEPTH {
        // Return null for excessively nested values (DoS protection)
        return serde_json::Value::Null;
    }

    match value {
        CypherValue::Null => serde_json::Value::Null,
        CypherValue::Bool(b) => serde_json::Value::Bool(b),
        CypherValue::Int(i) => serde_json::Value::Number(i.into()),
        CypherValue::Float(f) => serde_json::json!(f),
        CypherValue::String(s) => serde_json::Value::String(s.to_string()),
        CypherValue::Bytes(bytes) => {
            // Encode as base64 string
            serde_json::Value::String(base64_encode(&bytes))
        }
        CypherValue::Date { year, month, day } => {
            serde_json::json!({ "year": year, "month": month, "day": day })
        }
        CypherValue::Time {
            hour,
            minute,
            second,
            nanos,
        } => {
            serde_json::json!({ "hour": hour, "minute": minute, "second": second, "nanos": nanos })
        }
        CypherValue::DateTime { date, time } => {
            serde_json::json!({
                "date": cypher_value_to_json_with_depth(*date, depth + 1),
                "time": cypher_value_to_json_with_depth(*time, depth + 1)
            })
        }
        CypherValue::Duration {
            months,
            days,
            nanos,
        } => {
            serde_json::json!({ "months": months, "days": days, "nanos": nanos })
        }
        CypherValue::Point2D { x, y, srid } => {
            serde_json::json!({ "x": x, "y": y, "srid": srid })
        }
        CypherValue::Point3D { x, y, z, srid } => {
            serde_json::json!({ "x": x, "y": y, "z": z, "srid": srid })
        }
        CypherValue::List(items) => serde_json::Value::Array(
            items
                .into_iter()
                .map(|v| cypher_value_to_json_with_depth(v, depth + 1))
                .collect(),
        ),
        CypherValue::Map(entries) => {
            let obj: serde_json::Map<String, serde_json::Value> = entries
                .into_iter()
                .map(|(k, v): (String, CypherValue)| {
                    (k, cypher_value_to_json_with_depth(v, depth + 1))
                })
                .collect();
            serde_json::Value::Object(obj)
        }
        CypherValue::NodeRef(id) => serde_json::json!({ "_node_id": id }),
        CypherValue::RelationshipRef(id) => serde_json::json!({ "_relationship_id": id }),
        CypherValue::Path(ids) => {
            serde_json::json!({ "_path": ids.into_iter().collect::<Vec<_>>() })
        }
    }
}

/// Simple base64 encoding for bytes
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();

    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        result.push(ALPHABET[b0 >> 2] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);

        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(ALPHABET[b2 & 0x3f] as char);
        } else {
            result.push('=');
        }
    }

    result
}

/// Convert serde_json::Value to HashMap
fn json_value_to_map(value: serde_json::Value) -> HashMap<String, serde_json::Value> {
    match value {
        serde_json::Value::Object(map) => map.into_iter().collect(),
        _ => HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dto::{
        CreateCollectionRequest, PointStruct, SearchParams, UpsertPointsRequest, VectorParams,
    };
    use crate::handlers::{collections::create_collection, points::upsert_points};
    use crate::router::new_app_state;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn setup_collection_with_points(state: &AppState) {
        // Create collection
        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 4,
                distance: "Cosine".to_string(),
            },
            hnsw_config: None,
        };
        create_collection(state, "test", request);

        // Insert points
        let request = UpsertPointsRequest {
            points: (0..20)
                .map(|i| PointStruct {
                    id: i,
                    vector: vec![i as f32 * 0.1, i as f32 * 0.05, 0.5, 0.5],
                    payload: None,
                })
                .collect(),
        };
        upsert_points(state, "test", request);
    }

    #[test]
    fn test_search() {
        let state = new_app_state();
        setup_collection_with_points(&state);

        let request = SearchRequest {
            vector: vec![0.1, 0.05, 0.5, 0.5],
            limit: 5,
            with_payload: true,
            with_vector: false,
            params: None,
            score_threshold: None,
        };

        let response = search_points(&state, "test", request);
        assert_eq!(response.status, 200);

        let body: ApiResponse<Vec<ScoredPoint>> = serde_json::from_slice(&response.body).unwrap();
        let results = body.result.unwrap();
        assert_eq!(results.len(), 5);

        // Results should be sorted by score
        for i in 1..results.len() {
            assert!(results[i - 1].score <= results[i].score);
        }
    }

    #[test]
    fn test_search_with_ef() {
        let state = new_app_state();
        setup_collection_with_points(&state);

        let request = SearchRequest {
            vector: vec![0.1, 0.05, 0.5, 0.5],
            limit: 5,
            with_payload: false,
            with_vector: true,
            params: Some(SearchParams { ef: Some(50) }),
            score_threshold: None,
        };

        let response = search_points(&state, "test", request);
        assert_eq!(response.status, 200);
    }

    #[test]
    fn test_scroll() {
        let state = new_app_state();
        setup_collection_with_points(&state);

        // First page
        let request = ScrollRequest {
            limit: 10,
            offset: None,
            with_payload: true,
            with_vector: false,
        };

        let response = scroll_points(&state, "test", request);
        assert_eq!(response.status, 200);

        let body: ApiResponse<ScrollResult> = serde_json::from_slice(&response.body).unwrap();
        let result = body.result.unwrap();
        assert_eq!(result.points.len(), 10);
        assert!(result.next_page_offset.is_some());

        // Second page
        let request = ScrollRequest {
            limit: 10,
            offset: result.next_page_offset,
            with_payload: true,
            with_vector: false,
        };

        let response = scroll_points(&state, "test", request);
        let body: ApiResponse<ScrollResult> = serde_json::from_slice(&response.body).unwrap();
        let result = body.result.unwrap();
        assert_eq!(result.points.len(), 10);
    }

    #[test]
    fn test_collection_not_found() {
        let state = new_app_state();

        let request = SearchRequest {
            vector: vec![0.1, 0.2, 0.3, 0.4],
            limit: 5,
            with_payload: false,
            with_vector: false,
            params: None,
            score_threshold: None,
        };

        let response = search_points(&state, "nonexistent", request);
        assert_eq!(response.status, 404);
    }
}
