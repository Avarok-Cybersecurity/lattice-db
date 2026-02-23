//! Collection management handlers
//!
//! Handles collection CRUD operations.

use crate::dto::{
    ApiResponse, CollectionConfigResponse, CollectionDescription, CollectionInfo,
    CollectionParamsResponse, CollectionStatus, CollectionsResponse, CreateCollectionRequest,
    HnswConfigResponse, OptimizersConfigResponse, OptimizersStatus, VectorParamsResponse,
};
#[cfg(feature = "native")]
use crate::engine_wrapper::AnyEngine;
#[cfg(feature = "openapi")]
use crate::dto::{
    ApiResponseBoolResult, ApiResponseCollectionInfo, ApiResponseCollectionsResponse,
};
use crate::router::{json_response, AppState, InsertError};
use lattice_core::{
    CollectionConfig, Distance, DurabilityMode, HnswConfig, LatticeResponse, VectorConfig,
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
pub async fn create_collection(
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
        // Validate m (DoS protection: prevents excessive memory usage per node)
        const MAX_M: usize = 256;
        if hnsw.m == 0 {
            return LatticeResponse::bad_request("m must be at least 1");
        }
        if hnsw.m > MAX_M {
            return LatticeResponse::bad_request(&format!(
                "m {} exceeds maximum of {}",
                hnsw.m, MAX_M
            ));
        }

        // Validate ef_construct (DoS protection: prevents OOM during index build)
        const MAX_EF_CONSTRUCT: usize = 100_000;
        if hnsw.ef_construct == 0 {
            return LatticeResponse::bad_request("ef_construct must be at least 1");
        }
        if hnsw.ef_construct > MAX_EF_CONSTRUCT {
            return LatticeResponse::bad_request(&format!(
                "ef_construct {} exceeds maximum of {}",
                hnsw.ef_construct, MAX_EF_CONSTRUCT
            ));
        }

        // Validate m0 if provided (must be >= m for valid HNSW structure)
        if let Some(m0) = hnsw.m0 {
            if m0 < hnsw.m {
                return LatticeResponse::bad_request(&format!(
                    "m0 ({}) must be >= m ({})",
                    m0, hnsw.m
                ));
            }
            const MAX_M0: usize = 512;
            if m0 > MAX_M0 {
                return LatticeResponse::bad_request(&format!(
                    "m0 {} exceeds maximum of {}",
                    m0, MAX_M0
                ));
            }
        }

        // Validate ml if provided (must be finite and positive)
        if let Some(ml) = hnsw.ml {
            if !ml.is_finite() || ml <= 0.0 {
                return LatticeResponse::bad_request("ml must be a finite positive number");
            }
        }

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

    // Parse durability mode
    let durability = match request.durability.as_deref() {
        Some("durable") => DurabilityMode::Durable,
        Some("ephemeral") | None => DurabilityMode::Ephemeral,
        Some(other) => {
            return LatticeResponse::bad_request(&format!(
                "Unknown durability mode: '{}'. Use 'ephemeral' or 'durable'",
                other
            ));
        }
    };

    // Create collection config
    let config = CollectionConfig::new(
        name,
        VectorConfig::new(request.vectors.size, distance),
        hnsw_config,
    )
    .with_durability(durability);

    // Create engine: durable if data_dir is set and durability=durable, else ephemeral
    #[cfg(feature = "native")]
    let engine = {
        if durability == DurabilityMode::Durable {
            if let Some(ref data_dir) = state.config.data_dir {
                let col_dir = data_dir.join(name);
                let wal_dir = col_dir.join("wal");
                let data_sub = col_dir.join("data");

                // Create directories
                if let Err(e) = tokio::fs::create_dir_all(&wal_dir).await {
                    return LatticeResponse::internal_error(&format!(
                        "Failed to create WAL directory: {}",
                        e
                    ));
                }
                if let Err(e) = tokio::fs::create_dir_all(&data_sub).await {
                    return LatticeResponse::internal_error(&format!(
                        "Failed to create data directory: {}",
                        e
                    ));
                }

                // Save config.json for restart discovery
                let config_json = match serde_json::to_vec_pretty(&config) {
                    Ok(j) => j,
                    Err(e) => {
                        return LatticeResponse::internal_error(&format!(
                            "Failed to serialize config: {}",
                            e
                        ))
                    }
                };
                if let Err(e) = tokio::fs::write(col_dir.join("config.json"), &config_json).await {
                    return LatticeResponse::internal_error(&format!(
                        "Failed to write config: {}",
                        e
                    ));
                }

                let wal_storage = lattice_storage::DiskStorage::new(wal_dir, 4096);
                let data_storage = lattice_storage::DiskStorage::new(data_sub, 4096);

                match AnyEngine::open_durable(config, wal_storage, data_storage).await {
                    Ok(e) => e,
                    Err(e) => {
                        return LatticeResponse::internal_error(&format!(
                            "Failed to open durable engine: {}",
                            e
                        ))
                    }
                }
            } else {
                // No data_dir configured; fall back to ephemeral
                match AnyEngine::new_ephemeral(config) {
                    Ok(e) => e,
                    Err(e) => {
                        return LatticeResponse::bad_request(&format!("Invalid config: {}", e))
                    }
                }
            }
        } else {
            match AnyEngine::new_ephemeral(config) {
                Ok(e) => e,
                Err(e) => return LatticeResponse::bad_request(&format!("Invalid config: {}", e)),
            }
        }
    };

    #[cfg(not(feature = "native"))]
    let engine = match lattice_core::CollectionEngine::new(config) {
        Ok(e) => e,
        Err(e) => return LatticeResponse::bad_request(&format!("Invalid config: {}", e)),
    };

    // Insert with per-collection locking (insert_collection checks for duplicates and capacity)
    if let Err(e) = state.insert_collection(name.to_string(), engine) {
        return match e {
            InsertError::AlreadyExists => {
                warn!(collection = name, "Collection already exists");
                LatticeResponse::bad_request(&format!("Collection '{}' already exists", name))
            }
            InsertError::AtCapacity(max) => {
                warn!(
                    collection = name,
                    max_collections = max,
                    "Collection limit reached"
                );
                LatticeResponse::bad_request(&format!(
                    "Maximum collection limit ({}) reached. Delete unused collections first.",
                    max
                ))
            }
        };
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
pub async fn get_collection(state: &AppState, name: &str) -> LatticeResponse {
    let handle = match state.get_collection(name) {
        Some(h) => h,
        None => return LatticeResponse::not_found(&format!("Collection '{}' not found", name)),
    };
    let engine = handle.read().await;

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
            durability: Some(match config.durability {
                DurabilityMode::Ephemeral => "ephemeral".to_string(),
                DurabilityMode::Durable => "durable".to_string(),
            }),
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

    async_test!(test_create_and_list_collection, {
        let state = new_app_state();

        // Create
        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 128,
                distance: "Cosine".to_string(),
            },
            hnsw_config: None,
            durability: None,
        };

        let response = create_collection(&state, "test", request).await;
        assert_eq!(response.status, 200);

        // List
        let response = list_collections(&state);
        assert_eq!(response.status, 200);

        let body: ApiResponse<CollectionsResponse> =
            serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body.result.unwrap().collections.len(), 1);
    });

    async_test!(test_create_duplicate_collection, {
        let state = new_app_state();

        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 128,
                distance: "Cosine".to_string(),
            },
            hnsw_config: None,
            durability: None,
        };

        // First create succeeds
        let response = create_collection(&state, "test", request.clone()).await;
        assert_eq!(response.status, 200);

        // Second create fails
        let response = create_collection(&state, "test", request).await;
        assert_eq!(response.status, 400);
    });

    async_test!(test_get_collection_not_found, {
        let state = new_app_state();
        let response = get_collection(&state, "nonexistent").await;
        assert_eq!(response.status, 404);
    });

    async_test!(test_delete_collection, {
        let state = new_app_state();

        // Create
        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 128,
                distance: "Cosine".to_string(),
            },
            hnsw_config: None,
            durability: None,
        };
        create_collection(&state, "test", request).await;

        // Delete
        let response = delete_collection(&state, "test");
        assert_eq!(response.status, 200);

        // Verify gone
        let response = get_collection(&state, "test").await;
        assert_eq!(response.status, 404);
    });

    async_test!(test_invalid_distance, {
        let state = new_app_state();

        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 128,
                distance: "InvalidMetric".to_string(),
            },
            hnsw_config: None,
            durability: None,
        };

        let response = create_collection(&state, "test", request).await;
        assert_eq!(response.status, 400);
    });

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

    async_test!(test_create_collection_with_invalid_name, {
        let state = new_app_state();

        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 128,
                distance: "Cosine".to_string(),
            },
            hnsw_config: None,
            durability: None,
        };

        // Path traversal attempt
        let response = create_collection(&state, "../etc/passwd", request.clone()).await;
        assert_eq!(response.status, 400);

        // Empty name
        let response = create_collection(&state, "", request).await;
        assert_eq!(response.status, 400);
    });
}
