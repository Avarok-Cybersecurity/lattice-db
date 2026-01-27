//! Import/Export handlers for collection backup and restore
//!
//! Provides binary import/export using rkyv serialization.

use crate::dto::{ApiResponse, ImportMode, ImportResult};
use crate::router::{json_response, AppState, InsertError};
use lattice_core::{CollectionEngine, LatticeResponse};
use std::collections::HashMap;
use tracing::{info, warn};

/// Maximum import payload size (1GB)
const MAX_IMPORT_SIZE: usize = 1024 * 1024 * 1024;

/// Binary format version for future compatibility
const FORMAT_VERSION: &str = "1";

/// Export a collection as binary data
///
/// Returns the serialized collection using rkyv format.
pub fn export_collection(state: &AppState, name: &str) -> LatticeResponse {
    let handle = match state.get_collection(name) {
        Some(h) => h,
        None => return LatticeResponse::not_found(&format!("Collection '{}' not found", name)),
    };

    let engine = handle.read();

    // Serialize the collection
    let bytes = match engine.to_bytes() {
        Ok(b) => b,
        Err(e) => {
            warn!(collection = name, error = %e, "Export serialization failed");
            return LatticeResponse::internal_error(&format!("Serialization failed: {}", e));
        }
    };

    let point_count = engine.point_count();
    let dimension = engine.config().vectors.size;

    info!(
        collection = name,
        points = point_count,
        bytes = bytes.len(),
        "Collection exported"
    );

    // Build binary response with metadata headers
    let mut headers = HashMap::new();
    headers.insert("Content-Type".into(), "application/octet-stream".into());
    headers.insert("X-Lattice-Format-Version".into(), FORMAT_VERSION.into());
    headers.insert("X-Lattice-Point-Count".into(), point_count.to_string());
    headers.insert("X-Lattice-Dimension".into(), dimension.to_string());
    headers.insert(
        "Content-Disposition".into(),
        format!("attachment; filename=\"{}.lattice\"", name),
    );

    LatticeResponse {
        status: 200,
        body: bytes,
        headers,
    }
}

/// Import a collection from binary data
///
/// Supports three modes (PCND: mode is required):
/// - `create`: Create new collection, fail if exists
/// - `replace`: Drop existing collection and create new
/// - `merge`: Merge points into existing collection (skip duplicates)
pub fn import_collection(
    state: &AppState,
    name: &str,
    mode: ImportMode,
    data: Vec<u8>,
) -> LatticeResponse {
    // Validate size limit
    if data.len() > MAX_IMPORT_SIZE {
        return LatticeResponse::bad_request(&format!(
            "Payload size {} exceeds maximum of {} bytes",
            data.len(),
            MAX_IMPORT_SIZE
        ));
    }

    // Minimum size check (at least 4 bytes for config_len)
    if data.len() < 4 {
        return LatticeResponse::bad_request("Invalid import data: too small");
    }

    match mode {
        ImportMode::Create => import_create(state, name, &data),
        ImportMode::Replace => import_replace(state, name, &data),
        ImportMode::Merge => import_merge(state, name, &data),
    }
}

/// Import in create mode: fail if collection exists
fn import_create(state: &AppState, name: &str, data: &[u8]) -> LatticeResponse {
    // Check if collection already exists
    if state.get_collection(name).is_some() {
        return LatticeResponse::error(
            409,
            &format!(
                "Collection '{}' already exists. Use mode=replace or mode=merge",
                name
            ),
        );
    }

    // Deserialize the collection
    let engine = match CollectionEngine::from_bytes(data) {
        Ok(e) => e,
        Err(e) => {
            warn!(collection = name, error = %e, "Import deserialization failed");
            return LatticeResponse::bad_request(&format!("Invalid import data: {}", e));
        }
    };

    let point_count = engine.point_count();
    let dimension = engine.config().vectors.size;

    // Insert the collection
    if let Err(e) = state.insert_collection(name.to_string(), engine) {
        return match e {
            InsertError::AlreadyExists => {
                LatticeResponse::error(409, &format!("Collection '{}' already exists", name))
            }
            InsertError::AtCapacity(max) => {
                LatticeResponse::bad_request(&format!("Maximum collection limit ({}) reached", max))
            }
        };
    }

    info!(
        collection = name,
        points = point_count,
        mode = "create",
        "Collection imported"
    );

    json_response(&ApiResponse::ok(ImportResult {
        points_imported: point_count,
        points_skipped: 0,
        dimension,
        mode: ImportMode::Create,
    }))
}

/// Import in replace mode: drop existing and create new
fn import_replace(state: &AppState, name: &str, data: &[u8]) -> LatticeResponse {
    // Remove existing collection if present
    let _ = state.remove_collection(name);

    // Deserialize the collection
    let engine = match CollectionEngine::from_bytes(data) {
        Ok(e) => e,
        Err(e) => {
            warn!(collection = name, error = %e, "Import deserialization failed");
            return LatticeResponse::bad_request(&format!("Invalid import data: {}", e));
        }
    };

    let point_count = engine.point_count();
    let dimension = engine.config().vectors.size;

    // Insert the collection
    if let Err(e) = state.insert_collection(name.to_string(), engine) {
        return match e {
            InsertError::AlreadyExists => {
                // Race condition: another request created the collection
                LatticeResponse::error(
                    409,
                    &format!("Collection '{}' was created concurrently", name),
                )
            }
            InsertError::AtCapacity(max) => {
                LatticeResponse::bad_request(&format!("Maximum collection limit ({}) reached", max))
            }
        };
    }

    info!(
        collection = name,
        points = point_count,
        mode = "replace",
        "Collection imported"
    );

    json_response(&ApiResponse::ok(ImportResult {
        points_imported: point_count,
        points_skipped: 0,
        dimension,
        mode: ImportMode::Replace,
    }))
}

/// Import in merge mode: add new points to existing collection
fn import_merge(state: &AppState, name: &str, data: &[u8]) -> LatticeResponse {
    // Get existing collection
    let handle = match state.get_collection(name) {
        Some(h) => h,
        None => {
            return LatticeResponse::not_found(&format!(
                "Collection '{}' not found. Use mode=create for new collections",
                name
            ))
        }
    };

    // Deserialize the import data
    let import_engine = match CollectionEngine::from_bytes(data) {
        Ok(e) => e,
        Err(e) => {
            warn!(collection = name, error = %e, "Import deserialization failed");
            return LatticeResponse::bad_request(&format!("Invalid import data: {}", e));
        }
    };

    let import_dimension = import_engine.config().vectors.size;

    // Get existing dimension for validation
    let existing_dimension = {
        let engine = handle.read();
        engine.config().vectors.size
    };

    // Validate dimension match
    if import_dimension != existing_dimension {
        return LatticeResponse::bad_request(&format!(
            "Dimension mismatch: existing collection has dimension {}, import has {}",
            existing_dimension, import_dimension
        ));
    }

    // Get all point IDs from import engine, then get the points
    let import_ids = match import_engine.point_ids() {
        Ok(ids) => ids,
        Err(e) => {
            warn!(collection = name, error = %e, "Failed to get import point IDs");
            return LatticeResponse::bad_request(&format!("Failed to read import data: {}", e));
        }
    };

    let import_points = match import_engine.get_points(&import_ids) {
        Ok(pts) => pts,
        Err(e) => {
            warn!(collection = name, error = %e, "Failed to get import points");
            return LatticeResponse::bad_request(&format!("Failed to read import data: {}", e));
        }
    };

    let mut points_imported = 0;
    let mut points_skipped = 0;

    // Merge points
    {
        let mut engine = handle.write();
        for maybe_point in import_points {
            let point = match maybe_point {
                Some(p) => p,
                None => {
                    points_skipped += 1;
                    continue;
                }
            };

            // Check if point already exists
            let exists = match engine.get_point(point.id) {
                Ok(Some(_)) => true,
                Ok(None) => false,
                Err(_) => false,
            };

            if exists {
                points_skipped += 1;
            } else {
                // Insert new point
                if engine.upsert_points(vec![point]).is_ok() {
                    points_imported += 1;
                } else {
                    points_skipped += 1;
                }
            }
        }
    }

    info!(
        collection = name,
        imported = points_imported,
        skipped = points_skipped,
        mode = "merge",
        "Collection merge completed"
    );

    json_response(&ApiResponse::ok(ImportResult {
        points_imported,
        points_skipped,
        dimension: existing_dimension,
        mode: ImportMode::Merge,
    }))
}

/// Parse import mode from query string
///
/// Returns `None` if mode parameter is missing (PCND: required parameter).
pub fn parse_import_mode(query: &str) -> Result<ImportMode, LatticeResponse> {
    // Parse query string: "mode=create" or "?mode=create"
    let query = query.trim_start_matches('?');

    for pair in query.split('&') {
        let mut kv = pair.splitn(2, '=');
        if let (Some(key), Some(value)) = (kv.next(), kv.next()) {
            if key == "mode" {
                return match value.to_lowercase().as_str() {
                    "create" => Ok(ImportMode::Create),
                    "replace" => Ok(ImportMode::Replace),
                    "merge" => Ok(ImportMode::Merge),
                    other => Err(LatticeResponse::bad_request(&format!(
                        "Invalid import mode '{}'. Use 'create', 'replace', or 'merge'",
                        other
                    ))),
                };
            }
        }
    }

    // Mode is required (PCND)
    Err(LatticeResponse::bad_request(
        "Missing required 'mode' query parameter. Use mode=create, mode=replace, or mode=merge",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dto::CreateCollectionRequest;
    use crate::dto::VectorParams;
    use crate::handlers::collections::create_collection;
    use crate::router::new_app_state;
    use lattice_core::Point;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn test_parse_import_mode() {
        assert_eq!(
            parse_import_mode("mode=create").unwrap(),
            ImportMode::Create
        );
        assert_eq!(
            parse_import_mode("mode=replace").unwrap(),
            ImportMode::Replace
        );
        assert_eq!(parse_import_mode("mode=merge").unwrap(), ImportMode::Merge);
        assert_eq!(
            parse_import_mode("?mode=create").unwrap(),
            ImportMode::Create
        );
        assert_eq!(
            parse_import_mode("foo=bar&mode=replace").unwrap(),
            ImportMode::Replace
        );

        // Missing mode
        assert!(parse_import_mode("").is_err());
        assert!(parse_import_mode("foo=bar").is_err());

        // Invalid mode
        assert!(parse_import_mode("mode=invalid").is_err());
    }

    #[test]
    fn test_export_not_found() {
        let state = new_app_state();
        let response = export_collection(&state, "nonexistent");
        assert_eq!(response.status, 404);
    }

    #[test]
    fn test_export_import_roundtrip() {
        let state = new_app_state();

        // Create collection
        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 4,
                distance: "Cosine".to_string(),
            },
            hnsw_config: None,
            durability: None,
        };
        let response = create_collection(&state, "test", request);
        assert_eq!(response.status, 200);

        // Add a point
        {
            let handle = state.get_collection("test").unwrap();
            let mut engine = handle.write();
            let point = Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4]);
            engine.upsert_points(vec![point]).unwrap();
        }

        // Export
        let response = export_collection(&state, "test");
        assert_eq!(response.status, 200);
        assert_eq!(
            response.headers.get("Content-Type").unwrap(),
            "application/octet-stream"
        );
        assert_eq!(
            response.headers.get("X-Lattice-Format-Version").unwrap(),
            "1"
        );
        assert_eq!(response.headers.get("X-Lattice-Point-Count").unwrap(), "1");

        let exported_data = response.body;

        // Import to new collection (create mode)
        let response =
            import_collection(&state, "test2", ImportMode::Create, exported_data.clone());
        assert_eq!(response.status, 200);

        // Verify imported collection
        let handle = state.get_collection("test2").unwrap();
        let engine = handle.read();
        assert_eq!(engine.point_count(), 1);
        assert!(engine.get_point(1).unwrap().is_some());
    }

    #[test]
    fn test_import_create_conflict() {
        let state = new_app_state();

        // Create collection
        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 4,
                distance: "Cosine".to_string(),
            },
            hnsw_config: None,
            durability: None,
        };
        create_collection(&state, "test", request);

        // Export
        let response = export_collection(&state, "test");
        let exported_data = response.body;

        // Try to import with create mode (should fail - already exists)
        let response = import_collection(&state, "test", ImportMode::Create, exported_data);
        assert_eq!(response.status, 409);
    }

    #[test]
    fn test_import_replace() {
        let state = new_app_state();

        // Create collection with 1 point
        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 4,
                distance: "Cosine".to_string(),
            },
            hnsw_config: None,
            durability: None,
        };
        create_collection(&state, "test", request);
        {
            let handle = state.get_collection("test").unwrap();
            let mut engine = handle.write();
            engine
                .upsert_points(vec![Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4])])
                .unwrap();
        }

        // Export
        let response = export_collection(&state, "test");
        let exported_data = response.body;

        // Add another point to the collection
        {
            let handle = state.get_collection("test").unwrap();
            let mut engine = handle.write();
            engine
                .upsert_points(vec![Point::new_vector(2, vec![0.5, 0.6, 0.7, 0.8])])
                .unwrap();
        }

        // Import with replace mode (should reset to 1 point)
        let response = import_collection(&state, "test", ImportMode::Replace, exported_data);
        assert_eq!(response.status, 200);

        // Verify replaced collection has only 1 point
        let handle = state.get_collection("test").unwrap();
        let engine = handle.read();
        assert_eq!(engine.point_count(), 1);
        assert!(engine.get_point(1).unwrap().is_some());
        assert!(engine.get_point(2).unwrap().is_none());
    }

    #[test]
    fn test_import_merge() {
        let state = new_app_state();

        // Create collection with 1 point (id=1)
        let request = CreateCollectionRequest {
            vectors: VectorParams {
                size: 4,
                distance: "Cosine".to_string(),
            },
            hnsw_config: None,
            durability: None,
        };
        create_collection(&state, "test", request.clone());
        {
            let handle = state.get_collection("test").unwrap();
            let mut engine = handle.write();
            engine
                .upsert_points(vec![Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4])])
                .unwrap();
        }

        // Create another collection with 2 points (id=1, id=2)
        create_collection(&state, "source", request);
        {
            let handle = state.get_collection("source").unwrap();
            let mut engine = handle.write();
            engine
                .upsert_points(vec![Point::new_vector(1, vec![0.9, 0.9, 0.9, 0.9])])
                .unwrap();
            engine
                .upsert_points(vec![Point::new_vector(2, vec![0.5, 0.6, 0.7, 0.8])])
                .unwrap();
        }

        // Export source
        let response = export_collection(&state, "source");
        let exported_data = response.body;

        // Import with merge mode (should add id=2, skip id=1)
        let response = import_collection(&state, "test", ImportMode::Merge, exported_data);
        assert_eq!(response.status, 200);

        let body: ApiResponse<ImportResult> = serde_json::from_slice(&response.body).unwrap();
        let result = body.result.unwrap();
        assert_eq!(result.points_imported, 1);
        assert_eq!(result.points_skipped, 1);

        // Verify merged collection
        let handle = state.get_collection("test").unwrap();
        let engine = handle.read();
        assert_eq!(engine.point_count(), 2);
        // Original point should be preserved (not overwritten)
        let point1 = engine.get_point(1).unwrap().unwrap();
        assert_eq!(point1.vector[0], 0.1);
    }

    #[test]
    fn test_import_merge_not_found() {
        let state = new_app_state();
        let response =
            import_collection(&state, "nonexistent", ImportMode::Merge, vec![0, 0, 0, 0]);
        assert_eq!(response.status, 404);
    }

    #[test]
    fn test_import_invalid_data() {
        let state = new_app_state();

        // Too small
        let response = import_collection(&state, "test", ImportMode::Create, vec![1, 2]);
        assert_eq!(response.status, 400);
    }
}
