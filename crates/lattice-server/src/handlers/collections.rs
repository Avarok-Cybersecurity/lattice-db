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
use tracing::{info, warn};

/// Maximum allowed collection name length
const MAX_COLLECTION_NAME_LENGTH: usize = 255;

/// Validate collection name to prevent path traversal and injection attacks
///
/// Returns `Ok(())` if valid, or an error message if invalid.
fn validate_collection_name(name: &str) -> Result<(), &'static str> {
    // Check empty
    if name.is_empty() {
        return Err("Collection name cannot be empty");
    }

    // Check length
    if name.len() > MAX_COLLECTION_NAME_LENGTH {
        return Err("Collection name exceeds maximum length (255 characters)");
    }

    // Check for null bytes
    if name.contains('\0') {
        return Err("Collection name cannot contain null bytes");
    }

    // Check for path traversal sequences
    if name.contains("..") || name.contains('/') || name.contains('\\') {
        return Err("Collection name cannot contain path traversal characters");
    }

    // Only allow alphanumeric, underscore, hyphen, and dot (but not starting with dot)
    if name.starts_with('.') {
        return Err("Collection name cannot start with a dot");
    }

    for c in name.chars() {
        if !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != '.' {
            return Err("Collection name can only contain alphanumeric characters, underscores, hyphens, and dots");
        }
    }

    Ok(())
}

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
    let descriptions: Vec<CollectionDescription> = state
        .list_collection_names()
        .into_iter()
        .map(|name| CollectionDescription { name })
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
    // Validate collection name (security: prevent path traversal)
    if let Err(e) = validate_collection_name(name) {
        return LatticeResponse::bad_request(e);
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

    // Validate vector dimension bounds (DoS protection: prevent memory exhaustion)
    const MAX_VECTOR_DIM: usize = 65536;
    if request.vectors.size == 0 {
        return LatticeResponse::bad_request("Vector dimension must be at least 1");
    }
    if request.vectors.size > MAX_VECTOR_DIM {
        return LatticeResponse::bad_request(&format!(
            "Vector dimension {} exceeds maximum of {}",
            request.vectors.size, MAX_VECTOR_DIM
        ));
    }

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

    // Insert with per-collection locking (insert_collection checks for duplicates and limits)
    match state.insert_collection(name.to_string(), engine) {
        Ok(true) => {} // Successfully created
        Ok(false) => {
            warn!(collection = name, "Collection already exists");
            return LatticeResponse::bad_request(&format!("Collection '{}' already exists", name));
        }
        Err(msg) => {
            warn!(
                collection = name,
                error = msg,
                "Failed to create collection"
            );
            return LatticeResponse::bad_request(msg);
        }
    }

    info!(
        collection = name,
        vector_dim = request.vectors.size,
        distance = %request.vectors.distance,
        "Collection created"
    );

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
    let handle = match state.get_collection(name) {
        Some(h) => h,
        None => return LatticeResponse::not_found(&format!("Collection '{}' not found", name)),
    };
    let engine = handle.read();

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
    if state.remove_collection(name) {
        info!(collection = name, "Collection deleted");
        // Qdrant returns plain bool for create/delete operations
        json_response(&ApiResponse::ok(true))
    } else {
        warn!(collection = name, "Collection not found for deletion");
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

    #[test]
    fn test_collection_name_validation() {
        // Valid names
        assert!(validate_collection_name("test").is_ok());
        assert!(validate_collection_name("my_collection").is_ok());
        assert!(validate_collection_name("collection-123").is_ok());
        assert!(validate_collection_name("v1.0.0").is_ok());

        // Invalid: empty
        assert!(validate_collection_name("").is_err());

        // Invalid: path traversal
        assert!(validate_collection_name("../etc").is_err());
        assert!(validate_collection_name("foo/bar").is_err());
        assert!(validate_collection_name("foo\\bar").is_err());
        assert!(validate_collection_name("..").is_err());

        // Invalid: starts with dot
        assert!(validate_collection_name(".hidden").is_err());

        // Invalid: special characters
        assert!(validate_collection_name("test@name").is_err());
        assert!(validate_collection_name("test name").is_err());
        assert!(validate_collection_name("test\0name").is_err());
    }

    #[test]
    fn test_create_collection_with_invalid_name() {
        let state = new_app_state();

        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 128,
                distance: "Cosine".to_string(),
            },
            hnsw_config: None,
        };

        // Path traversal attempt
        let response = create_collection(&state, "../etc/passwd", request.clone());
        assert_eq!(response.status, 400);

        // Empty name
        let response = create_collection(&state, "", request);
        assert_eq!(response.status, 400);
    }
}
