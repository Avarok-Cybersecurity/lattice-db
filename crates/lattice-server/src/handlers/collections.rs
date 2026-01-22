//! Collection management handlers
//!
//! Handles collection CRUD operations.

use crate::dto::{
    ApiResponse, CollectionConfigResponse, CollectionDescription, CollectionInfo,
    CollectionParamsResponse, CollectionStatus, CollectionsResponse, CreateCollectionRequest,
    HnswConfigResponse, OptimizersConfigResponse, OptimizersStatus, VectorParamsResponse,
};
#[cfg(feature = "openapi")]
use crate::dto::{
    ApiResponseBoolResult, ApiResponseCollectionInfo, ApiResponseCollectionsResponse,
};
use crate::router::{json_response, AppState};
use lattice_core::{
    CollectionConfig, CollectionEngine, Distance, HnswConfig, LatticeResponse, VectorConfig,
};
use std::collections::HashMap;

/// List all collections
#[cfg_attr(feature = "openapi", utoipa::path(
    get,
    path = "/collections",
    tag = "Collections",
    responses(
        (status = 200, description = "List of collections", body = ApiResponseCollectionsResponse)
    )
))]
pub fn list_collections(state: &AppState) -> LatticeResponse {
    let collections = state.collections.read().unwrap();

    let descriptions: Vec<CollectionDescription> = collections
        .keys()
        .map(|name| CollectionDescription { name: name.clone() })
        .collect();

    json_response(&ApiResponse::ok(CollectionsResponse {
        collections: descriptions,
    }))
}

/// Create a new collection
#[cfg_attr(feature = "openapi", utoipa::path(
    put,
    path = "/collections/{name}",
    tag = "Collections",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    request_body = CreateCollectionRequest,
    responses(
        (status = 200, description = "Collection created", body = ApiResponseBoolResult),
        (status = 400, description = "Invalid request")
    )
))]
pub fn create_collection(
    state: &AppState,
    name: &str,
    request: CreateCollectionRequest,
) -> LatticeResponse {
    let mut collections = state.collections.write().unwrap();

    // Check if collection already exists
    if collections.contains_key(name) {
        return LatticeResponse::bad_request(&format!("Collection '{}' already exists", name));
    }

    // Parse distance metric
    let distance = match request.vectors.distance.to_lowercase().as_str() {
        "cosine" => Distance::Cosine,
        "euclid" | "euclidean" | "l2" => Distance::Euclid,
        "dot" | "dotproduct" => Distance::Dot,
        other => {
            return LatticeResponse::bad_request(&format!(
                "Unknown distance metric: '{}'. Use 'Cosine', 'Euclid', or 'Dot'",
                other
            ));
        }
    };

    // Build HNSW config (PCND: require explicit values or use recommended)
    let hnsw_config = if let Some(hnsw) = request.hnsw_config {
        let m = hnsw.m;
        HnswConfig {
            m,
            m0: hnsw.m0.unwrap_or(m * 2),
            ml: hnsw.ml.unwrap_or_else(|| HnswConfig::recommended_ml(m)),
            ef: hnsw.ef_construct, // Use ef_construct as default ef
            ef_construction: hnsw.ef_construct,
        }
    } else {
        // Reasonable defaults for API users (PCND: explicit in API layer)
        // ef_construction=100 matches Qdrant's default for balanced build speed/quality
        HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 100,
            ef_construction: 100,
        }
    };

    // Create collection config
    let config = CollectionConfig::new(
        name,
        VectorConfig::new(request.vectors.size, distance),
        hnsw_config,
    );

    // Create collection engine
    let engine = match CollectionEngine::new(config) {
        Ok(e) => e,
        Err(e) => return LatticeResponse::bad_request(&format!("Invalid config: {}", e)),
    };

    collections.insert(name.to_string(), engine);

    // Qdrant returns plain bool for create/delete operations
    json_response(&ApiResponse::ok(true))
}

/// Get collection info
#[cfg_attr(feature = "openapi", utoipa::path(
    get,
    path = "/collections/{name}",
    tag = "Collections",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    responses(
        (status = 200, description = "Collection info", body = ApiResponseCollectionInfo),
        (status = 404, description = "Collection not found")
    )
))]
pub fn get_collection(state: &AppState, name: &str) -> LatticeResponse {
    let collections = state.collections.read().unwrap();

    let engine = match collections.get(name) {
        Some(e) => e,
        None => return LatticeResponse::not_found(&format!("Collection '{}' not found", name)),
    };

    let config = engine.config();
    let point_count = engine.point_count() as u64;

    // Build Qdrant-compatible CollectionInfo response
    let info = CollectionInfo {
        status: CollectionStatus::Green,
        optimizer_status: OptimizersStatus::Ok,
        vectors_count: Some(point_count),
        indexed_vectors_count: Some(point_count),
        points_count: Some(point_count),
        segments_count: 1,
        config: CollectionConfigResponse {
            params: CollectionParamsResponse {
                vectors: VectorParamsResponse {
                    size: config.vectors.size,
                    distance: format!("{:?}", config.vectors.distance),
                    on_disk: None,
                },
                shard_number: Some(1),
                sharding_method: None,
                replication_factor: Some(1),
                write_consistency_factor: Some(1),
                on_disk_payload: Some(false),
            },
            hnsw_config: HnswConfigResponse {
                m: config.hnsw.m,
                ef_construct: config.hnsw.ef_construction,
                full_scan_threshold: 10000, // Qdrant default
                max_indexing_threads: Some(0),
                on_disk: None,
                payload_m: None,
            },
            optimizer_config: OptimizersConfigResponse::default(),
            wal_config: None,
            quantization_config: None,
        },
        payload_schema: HashMap::new(),
    };

    json_response(&ApiResponse::ok(info))
}

/// Delete a collection
#[cfg_attr(feature = "openapi", utoipa::path(
    delete,
    path = "/collections/{name}",
    tag = "Collections",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    responses(
        (status = 200, description = "Collection deleted", body = ApiResponseBoolResult),
        (status = 404, description = "Collection not found")
    )
))]
pub fn delete_collection(state: &AppState, name: &str) -> LatticeResponse {
    let mut collections = state.collections.write().unwrap();

    if collections.remove(name).is_some() {
        // Qdrant returns plain bool for create/delete operations
        json_response(&ApiResponse::ok(true))
    } else {
        LatticeResponse::not_found(&format!("Collection '{}' not found", name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dto::VectorParams;
    use crate::router::new_app_state;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn test_create_and_list_collection() {
        let state = new_app_state();

        // Create
        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 128,
                distance: "Cosine".to_string(),
            },
            hnsw_config: None,
        };

        let response = create_collection(&state, "test", request);
        assert_eq!(response.status, 200);

        // List
        let response = list_collections(&state);
        assert_eq!(response.status, 200);

        let body: ApiResponse<CollectionsResponse> =
            serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body.result.unwrap().collections.len(), 1);
    }

    #[test]
    fn test_create_duplicate_collection() {
        let state = new_app_state();

        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 128,
                distance: "Cosine".to_string(),
            },
            hnsw_config: None,
        };

        // First create succeeds
        let response = create_collection(&state, "test", request.clone());
        assert_eq!(response.status, 200);

        // Second create fails
        let response = create_collection(&state, "test", request);
        assert_eq!(response.status, 400);
    }

    #[test]
    fn test_get_collection_not_found() {
        let state = new_app_state();
        let response = get_collection(&state, "nonexistent");
        assert_eq!(response.status, 404);
    }

    #[test]
    fn test_delete_collection() {
        let state = new_app_state();

        // Create
        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 128,
                distance: "Cosine".to_string(),
            },
            hnsw_config: None,
        };
        create_collection(&state, "test", request);

        // Delete
        let response = delete_collection(&state, "test");
        assert_eq!(response.status, 200);

        // Verify gone
        let response = get_collection(&state, "test");
        assert_eq!(response.status, 404);
    }

    #[test]
    fn test_invalid_distance() {
        let state = new_app_state();

        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 128,
                distance: "InvalidMetric".to_string(),
            },
            hnsw_config: None,
        };

        let response = create_collection(&state, "test", request);
        assert_eq!(response.status, 400);
    }
}
