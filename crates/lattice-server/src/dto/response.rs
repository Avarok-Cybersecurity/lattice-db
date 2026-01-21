//! Response DTOs for Qdrant-compatible API
//!
//! These structures match the Qdrant JSON schema exactly for drop-in compatibility
//! with existing Qdrant clients.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Standard API response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Processing time in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time: Option<f64>,
}

impl<T> ApiResponse<T> {
    /// Create a success response
    pub fn ok(result: T) -> Self {
        Self {
            status: "ok".to_string(),
            result: Some(result),
            error: None,
            time: None,
        }
    }

    /// Create a success response with timing
    pub fn ok_with_time(result: T, time_secs: f64) -> Self {
        Self {
            status: "ok".to_string(),
            result: Some(result),
            error: None,
            time: Some(time_secs),
        }
    }

    /// Create an error response
    pub fn error(message: impl Into<String>) -> ApiResponse<()> {
        ApiResponse {
            status: "error".to_string(),
            result: None,
            error: Some(message.into()),
            time: None,
        }
    }
}

// === Collection Responses ===

/// Collection info response (Qdrant-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CollectionInfo {
    /// Collection status: "green", "yellow", "grey", "red"
    pub status: CollectionStatus,
    /// Optimizer status: "ok" or error object
    pub optimizer_status: OptimizersStatus,
    /// Total vectors count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vectors_count: Option<u64>,
    /// Indexed vectors count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexed_vectors_count: Option<u64>,
    /// Points count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub points_count: Option<u64>,
    /// Segments count
    pub segments_count: u64,
    /// Collection config
    pub config: CollectionConfigResponse,
    /// Payload field schemas (empty for LatticeDB)
    pub payload_schema: HashMap<String, PayloadIndexInfo>,
}

/// Collection status enum (Qdrant-compatible)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum CollectionStatus {
    Green,
    Yellow,
    Grey,
    Red,
}

/// Optimizer status - either "ok" string or error object (Qdrant-compatible)
///
/// Qdrant uses a union type here. We serialize "ok" as a plain string.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum OptimizersStatus {
    Ok,
    Error { error: String },
}

impl Default for OptimizersStatus {
    fn default() -> Self {
        Self::Ok
    }
}

impl Serialize for OptimizersStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            OptimizersStatus::Ok => serializer.serialize_str("ok"),
            OptimizersStatus::Error { error } => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("error", error)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for OptimizersStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};

        struct OptimizersStatusVisitor;

        impl<'de> Visitor<'de> for OptimizersStatusVisitor {
            type Value = OptimizersStatus;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("'ok' string or object with 'error' field")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value == "ok" {
                    Ok(OptimizersStatus::Ok)
                } else {
                    Err(de::Error::unknown_variant(value, &["ok"]))
                }
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut error = None;
                while let Some(key) = map.next_key::<String>()? {
                    if key == "error" {
                        error = Some(map.next_value()?);
                    } else {
                        let _ = map.next_value::<serde::de::IgnoredAny>()?;
                    }
                }
                match error {
                    Some(e) => Ok(OptimizersStatus::Error { error: e }),
                    None => Err(de::Error::missing_field("error")),
                }
            }
        }

        deserializer.deserialize_any(OptimizersStatusVisitor)
    }
}

/// Payload index info (Qdrant-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PayloadIndexInfo {
    pub data_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
    pub points: u64,
}

/// Collection configuration in response format (Qdrant-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CollectionConfigResponse {
    pub params: CollectionParamsResponse,
    pub hnsw_config: HnswConfigResponse,
    pub optimizer_config: OptimizersConfigResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wal_config: Option<WalConfigResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization_config: Option<serde_json::Value>,
}

/// Collection parameters in response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CollectionParamsResponse {
    pub vectors: VectorParamsResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shard_number: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sharding_method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replication_factor: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub write_consistency_factor: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub on_disk_payload: Option<bool>,
}

/// Vector parameters in response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct VectorParamsResponse {
    pub size: usize,
    pub distance: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub on_disk: Option<bool>,
}

/// HNSW config in response (Qdrant-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct HnswConfigResponse {
    pub m: usize,
    pub ef_construct: usize,
    pub full_scan_threshold: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_indexing_threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub on_disk: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload_m: Option<usize>,
}

/// Optimizers config in response (Qdrant-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct OptimizersConfigResponse {
    pub deleted_threshold: f64,
    pub vacuum_min_vector_number: usize,
    pub default_segment_number: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_segment_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memmap_threshold: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexing_threshold: Option<usize>,
    pub flush_interval_sec: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_optimization_threads: Option<usize>,
}

impl Default for OptimizersConfigResponse {
    fn default() -> Self {
        Self {
            deleted_threshold: 0.2,
            vacuum_min_vector_number: 1000,
            default_segment_number: 0,
            max_segment_size: None,
            memmap_threshold: None,
            indexing_threshold: Some(20000),
            flush_interval_sec: 5,
            max_optimization_threads: None,
        }
    }
}

/// WAL config in response (Qdrant-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct WalConfigResponse {
    pub wal_capacity_mb: usize,
    pub wal_segments_ahead: usize,
}

// === Point Responses ===

/// Point with score (search result)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ScoredPoint {
    /// Point ID
    pub id: u64,
    /// Point version (Qdrant-compatible)
    pub version: u64,
    /// Similarity score
    pub score: f32,
    /// Optional payload
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<HashMap<String, serde_json::Value>>,
    /// Optional vector
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
}

/// Point record (for get/scroll)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PointRecord {
    /// Point ID
    pub id: u64,
    /// Optional payload
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<HashMap<String, serde_json::Value>>,
    /// Optional vector
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
}

/// Scroll response with pagination
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ScrollResult {
    /// Retrieved points
    pub points: Vec<PointRecord>,
    /// Next page offset (None if no more results)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_page_offset: Option<u64>,
}

/// Upsert operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct UpsertResult {
    /// Operation ID (for tracking)
    pub operation_id: u64,
    /// Status: "acknowledged" or "completed"
    pub status: String,
}

/// Query response for /points/query endpoint (Qdrant v1.16+)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct QueryResponse {
    /// Matched points with scores
    pub points: Vec<ScoredPoint>,
}

/// Update (delete) operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct UpdateResult {
    /// Operation ID
    pub operation_id: u64,
    /// Status
    pub status: String,
}

// === Graph Extension Responses ===

/// Edge in graph traversal result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct EdgeRecord {
    /// Source point ID
    pub from_id: u64,
    /// Target point ID
    pub to_id: u64,
    /// Relation type
    pub relation: String,
    /// Edge weight
    pub weight: f32,
}

/// Graph traversal result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TraversalResult {
    /// Visited point IDs in traversal order
    pub visited: Vec<u64>,
    /// Edges traversed
    pub edges: Vec<EdgeRecord>,
    /// Depth reached
    pub max_depth_reached: usize,
}

/// Add edge result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct AddEdgeResult {
    pub status: String,
}

// === Cypher Query Response (LatticeDB extension) ===

/// Cypher query response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CypherQueryResponse {
    /// Column names from the RETURN clause
    pub columns: Vec<String>,
    /// Result rows (each row contains values for each column)
    pub rows: Vec<Vec<serde_json::Value>>,
    /// Query execution statistics
    pub stats: CypherQueryStats,
}

/// Cypher query execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CypherQueryStats {
    /// Number of nodes created
    pub nodes_created: u64,
    /// Number of relationships created
    pub relationships_created: u64,
    /// Number of nodes deleted
    pub nodes_deleted: u64,
    /// Number of relationships deleted
    pub relationships_deleted: u64,
    /// Number of properties set
    pub properties_set: u64,
    /// Query execution time in milliseconds
    pub execution_time_ms: u64,
}

// === List Collections Response ===

/// Single collection description
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CollectionDescription {
    pub name: String,
}

/// List of collections
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CollectionsResponse {
    pub collections: Vec<CollectionDescription>,
}

// === OpenAPI Type Aliases ===
//
// These concrete types are used for OpenAPI documentation since utoipa
// doesn't handle generic types well in path responses.

#[cfg(feature = "openapi")]
pub mod openapi_types {
    use super::*;
    use utoipa::ToSchema;

    /// API response containing collections list
    #[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
    pub struct ApiResponseCollectionsResponse {
        pub status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<CollectionsResponse>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub time: Option<f64>,
    }

    /// API response containing collection info
    #[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
    pub struct ApiResponseCollectionInfo {
        pub status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<CollectionInfo>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub time: Option<f64>,
    }

    /// API response containing boolean result (Qdrant-compatible: plain bool)
    #[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
    pub struct ApiResponseBoolResult {
        pub status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub time: Option<f64>,
    }

    /// API response containing upsert result
    #[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
    pub struct ApiResponseUpsertResult {
        pub status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<UpsertResult>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub time: Option<f64>,
    }

    /// API response containing update result
    #[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
    pub struct ApiResponseUpdateResult {
        pub status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<UpdateResult>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub time: Option<f64>,
    }

    /// API response containing scroll result
    #[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
    pub struct ApiResponseScrollResult {
        pub status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<ScrollResult>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub time: Option<f64>,
    }

    /// API response containing add edge result
    #[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
    pub struct ApiResponseAddEdgeResult {
        pub status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<AddEdgeResult>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub time: Option<f64>,
    }

    /// API response containing traversal result
    #[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
    pub struct ApiResponseTraversalResult {
        pub status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<TraversalResult>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub time: Option<f64>,
    }

    /// API response containing search results (scored points)
    #[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
    pub struct ApiResponseVecScoredPoint {
        pub status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<Vec<ScoredPoint>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub time: Option<f64>,
    }

    /// API response containing point records
    #[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
    pub struct ApiResponseVecPointRecord {
        pub status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<Vec<PointRecord>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub time: Option<f64>,
    }

    /// API response containing Cypher query result
    #[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
    pub struct ApiResponseCypherQueryResponse {
        pub status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<CypherQueryResponse>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub time: Option<f64>,
    }
}
