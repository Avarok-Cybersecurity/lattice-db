//! Search handlers
//!
//! Handles search, scroll, and graph traversal operations.

use crate::dto::{
    ApiResponse, BatchSearchRequest, EdgeRecord, PointRecord, QueryResponse, ScoredPoint,
    ScrollRequest, ScrollResult, SearchRequest, TraversalResult, TraverseRequest,
};
#[cfg(feature = "openapi")]
use crate::dto::{
    ApiResponseScrollResult, ApiResponseTraversalResult, ApiResponseVecScoredPoint,
};
use crate::router::{json_response, AppState};
use lattice_core::{LatticeResponse, ScrollQuery, SearchQuery};
use std::collections::HashMap;

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
    let collections = state.collections.read().unwrap();

    let engine = match collections.get(collection_name) {
        Some(e) => e,
        None => {
            return LatticeResponse::not_found(&format!(
                "Collection '{}' not found",
                collection_name
            ))
        }
    };

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
    let collections = state.collections.read().unwrap();

    let engine = match collections.get(collection_name) {
        Some(e) => e,
        None => {
            return LatticeResponse::not_found(&format!(
                "Collection '{}' not found",
                collection_name
            ))
        }
    };

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
    let collections = state.collections.read().unwrap();

    let engine = match collections.get(collection_name) {
        Some(e) => e,
        None => {
            return LatticeResponse::not_found(&format!(
                "Collection '{}' not found",
                collection_name
            ))
        }
    };

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
    let collections = state.collections.read().unwrap();

    let engine = match collections.get(collection_name) {
        Some(e) => e,
        None => {
            return LatticeResponse::not_found(&format!(
                "Collection '{}' not found",
                collection_name
            ))
        }
    };

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
    let result = engine.scroll(query);

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
    let collections = state.collections.read().unwrap();

    let engine = match collections.get(collection_name) {
        Some(e) => e,
        None => {
            return LatticeResponse::not_found(&format!(
                "Collection '{}' not found",
                collection_name
            ))
        }
    };

    // Convert relation strings to references for the API
    let relations: Option<Vec<&str>> = request
        .relations
        .as_ref()
        .map(|rels| rels.iter().map(|s| s.as_str()).collect());

    // Execute traversal
    match engine.traverse(request.start_id, request.max_depth, relations.as_deref()) {
        Ok(result) => {
            // Build edge records from paths
            let mut edges = Vec::new();
            for path in &result.paths {
                if path.path.len() >= 2 {
                    let from_id = path.path[path.path.len() - 2];
                    edges.push(EdgeRecord {
                        from_id,
                        to_id: path.target_id,
                        relation: "traversed".to_string(), // Simplified - would need edge info
                        weight: path.weight,
                    });
                }
            }

            let traversal_result = TraversalResult {
                visited: result.paths.iter().map(|p| p.target_id).collect(),
                edges,
                max_depth_reached: result
                    .paths
                    .iter()
                    .map(|p| p.depth)
                    .max()
                    .unwrap_or(0),
            };

            json_response(&ApiResponse::ok(traversal_result))
        }
        Err(e) => LatticeResponse::bad_request(&format!("Traversal failed: {}", e)),
    }
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
    use crate::dto::{CreateCollectionRequest, PointStruct, SearchParams, UpsertPointsRequest, VectorParams};
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
