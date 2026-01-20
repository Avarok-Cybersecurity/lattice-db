//! Point management handlers
//!
//! Handles point CRUD operations and graph edges.

use crate::dto::{
    AddEdgeRequest, AddEdgeResult, ApiResponse, DeletePointsRequest, GetPointsRequest,
    PointRecord, UpdateResult, UpsertPointsRequest, UpsertResult,
};
#[cfg(feature = "openapi")]
use crate::dto::{
    ApiResponseAddEdgeResult, ApiResponseUpdateResult, ApiResponseUpsertResult,
    ApiResponseVecPointRecord,
};
use crate::router::{json_response, AppState};
use lattice_core::{LatticeResponse, Point};
use std::collections::HashMap;

/// Upsert points into a collection
#[cfg_attr(feature = "openapi", utoipa::path(
    put,
    path = "/collections/{collection_name}/points",
    tag = "Points",
    params(
        ("collection_name" = String, Path, description = "Collection name")
    ),
    request_body = UpsertPointsRequest,
    responses(
        (status = 200, description = "Points upserted", body = ApiResponseUpsertResult),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Collection not found")
    )
))]
pub fn upsert_points(
    state: &AppState,
    collection_name: &str,
    request: UpsertPointsRequest,
) -> LatticeResponse {
    let mut collections = state.collections.write().unwrap();

    let engine = match collections.get_mut(collection_name) {
        Some(e) => e,
        None => {
            return LatticeResponse::not_found(&format!(
                "Collection '{}' not found",
                collection_name
            ))
        }
    };

    // Convert DTO points to core Points
    let points: Vec<Point> = request
        .points
        .into_iter()
        .map(|p| {
            let mut point = Point::new_vector(p.id, p.vector);
            if let Some(payload) = p.payload {
                for (key, value) in payload {
                    // Serialize JSON value to bytes
                    if let Ok(bytes) = serde_json::to_vec(&value) {
                        point = point.with_field(&key, bytes);
                    }
                }
            }
            point
        })
        .collect();

    // Upsert into engine
    match engine.upsert_points(points) {
        Ok(_) => {
            json_response(&ApiResponse::ok(UpsertResult {
                operation_id: 0,
                status: "completed".to_string(),
            }))
        }
        Err(e) => LatticeResponse::bad_request(&format!("Upsert failed: {}", e)),
    }
}

/// Get points by IDs
#[cfg_attr(feature = "openapi", utoipa::path(
    post,
    path = "/collections/{collection_name}/points",
    tag = "Points",
    params(
        ("collection_name" = String, Path, description = "Collection name")
    ),
    request_body = GetPointsRequest,
    responses(
        (status = 200, description = "Points retrieved", body = ApiResponseVecPointRecord),
        (status = 404, description = "Collection not found")
    )
))]
pub fn get_points(
    state: &AppState,
    collection_name: &str,
    request: GetPointsRequest,
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

    let results = engine.get_points(&request.ids);

    let points: Vec<PointRecord> = request
        .ids
        .iter()
        .zip(results.into_iter())
        .filter_map(|(&id, opt)| {
            opt.map(|point| PointRecord {
                id,
                payload: if request.with_payload {
                    Some(payload_to_json(&point.payload))
                } else {
                    None
                },
                vector: if request.with_vector {
                    Some(point.vector)
                } else {
                    None
                },
            })
        })
        .collect();

    json_response(&ApiResponse::ok(points))
}

/// Delete points by IDs
#[cfg_attr(feature = "openapi", utoipa::path(
    post,
    path = "/collections/{collection_name}/points/delete",
    tag = "Points",
    params(
        ("collection_name" = String, Path, description = "Collection name")
    ),
    request_body = DeletePointsRequest,
    responses(
        (status = 200, description = "Points deleted", body = ApiResponseUpdateResult),
        (status = 404, description = "Collection not found")
    )
))]
pub fn delete_points(
    state: &AppState,
    collection_name: &str,
    request: DeletePointsRequest,
) -> LatticeResponse {
    let mut collections = state.collections.write().unwrap();

    let engine = match collections.get_mut(collection_name) {
        Some(e) => e,
        None => {
            return LatticeResponse::not_found(&format!(
                "Collection '{}' not found",
                collection_name
            ))
        }
    };

    let _deleted = engine.delete_points(&request.points);

    json_response(&ApiResponse::ok(UpdateResult {
        operation_id: 0,
        status: "completed".to_string(),
    }))
}

/// Add an edge between two points (graph extension)
#[cfg_attr(feature = "openapi", utoipa::path(
    post,
    path = "/collections/{collection_name}/graph/edges",
    tag = "Graph",
    params(
        ("collection_name" = String, Path, description = "Collection name")
    ),
    request_body = AddEdgeRequest,
    responses(
        (status = 200, description = "Edge added", body = ApiResponseAddEdgeResult),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Collection not found")
    )
))]
pub fn add_edge(
    state: &AppState,
    collection_name: &str,
    request: AddEdgeRequest,
) -> LatticeResponse {
    let mut collections = state.collections.write().unwrap();

    let engine = match collections.get_mut(collection_name) {
        Some(e) => e,
        None => {
            return LatticeResponse::not_found(&format!(
                "Collection '{}' not found",
                collection_name
            ))
        }
    };

    match engine.add_edge(request.from_id, request.to_id, &request.relation, request.weight) {
        Ok(_) => json_response(&ApiResponse::ok(AddEdgeResult {
            status: "created".to_string(),
        })),
        Err(e) => LatticeResponse::bad_request(&format!("Failed to add edge: {}", e)),
    }
}

/// Convert payload HashMap to JSON Value
fn payload_to_json(payload: &HashMap<String, Vec<u8>>) -> HashMap<String, serde_json::Value> {
    payload
        .iter()
        .filter_map(|(k, v)| {
            serde_json::from_slice(v)
                .ok()
                .map(|val| (k.clone(), val))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dto::{CreateCollectionRequest, PointStruct, VectorParams};
    use crate::handlers::collections::create_collection;
    use crate::router::new_app_state;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn setup_collection(state: &AppState) {
        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 4,
                distance: "Cosine".to_string(),
            },
            hnsw_config: None,
        };
        create_collection(state, "test", request);
    }

    #[test]
    fn test_upsert_and_get_points() {
        let state = new_app_state();
        setup_collection(&state);

        // Upsert
        let request = UpsertPointsRequest {
            points: vec![
                PointStruct {
                    id: 1,
                    vector: vec![0.1, 0.2, 0.3, 0.4],
                    payload: None,
                },
                PointStruct {
                    id: 2,
                    vector: vec![0.5, 0.6, 0.7, 0.8],
                    payload: None,
                },
            ],
        };

        let response = upsert_points(&state, "test", request);
        assert_eq!(response.status, 200);

        // Get
        let request = GetPointsRequest {
            ids: vec![1, 2],
            with_payload: true,
            with_vector: true,
        };

        let response = get_points(&state, "test", request);
        assert_eq!(response.status, 200);

        let body: ApiResponse<Vec<PointRecord>> = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body.result.unwrap().len(), 2);
    }

    #[test]
    fn test_delete_points() {
        let state = new_app_state();
        setup_collection(&state);

        // Upsert
        let request = UpsertPointsRequest {
            points: vec![PointStruct {
                id: 1,
                vector: vec![0.1, 0.2, 0.3, 0.4],
                payload: None,
            }],
        };
        upsert_points(&state, "test", request);

        // Delete
        let request = DeletePointsRequest { points: vec![1] };
        let response = delete_points(&state, "test", request);
        assert_eq!(response.status, 200);

        // Verify gone
        let request = GetPointsRequest {
            ids: vec![1],
            with_payload: false,
            with_vector: false,
        };
        let response = get_points(&state, "test", request);
        let body: ApiResponse<Vec<PointRecord>> = serde_json::from_slice(&response.body).unwrap();
        assert!(body.result.unwrap().is_empty());
    }

    #[test]
    fn test_add_edge() {
        let state = new_app_state();
        setup_collection(&state);

        // Upsert two points
        let request = UpsertPointsRequest {
            points: vec![
                PointStruct {
                    id: 1,
                    vector: vec![0.1, 0.2, 0.3, 0.4],
                    payload: None,
                },
                PointStruct {
                    id: 2,
                    vector: vec![0.5, 0.6, 0.7, 0.8],
                    payload: None,
                },
            ],
        };
        upsert_points(&state, "test", request);

        // Add edge
        let request = AddEdgeRequest {
            from_id: 1,
            to_id: 2,
            relation: "similar".to_string(),
            weight: 0.9,
        };

        let response = add_edge(&state, "test", request);
        assert_eq!(response.status, 200);
    }

    #[test]
    fn test_collection_not_found() {
        let state = new_app_state();

        let request = GetPointsRequest {
            ids: vec![1],
            with_payload: false,
            with_vector: false,
        };

        let response = get_points(&state, "nonexistent", request);
        assert_eq!(response.status, 404);
    }
}
