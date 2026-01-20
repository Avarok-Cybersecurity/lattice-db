//! Request DTOs for Qdrant-compatible API
//!
//! These structures match the Qdrant JSON schema for API compatibility.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Vector configuration for collection creation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct VectorParams {
    /// Vector dimensionality
    pub size: usize,
    /// Distance metric: "Cosine", "Euclid", or "Dot"
    pub distance: String,
}

/// HNSW configuration (PCND: all fields required)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct HnswConfigRequest {
    /// Max connections per node in upper layers
    pub m: usize,
    /// Max connections per node in layer 0
    #[serde(default)]
    pub m0: Option<usize>,
    /// Level multiplier
    #[serde(default)]
    pub ml: Option<f64>,
    /// Search queue size for queries
    pub ef_construct: usize,
}

/// Create collection request
///
/// Qdrant-compatible format:
/// ```json
/// {
///   "vectors": { "size": 128, "distance": "Cosine" },
///   "hnsw_config": { "m": 16, "ef_construct": 200 }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CreateCollectionRequest {
    /// Vector configuration
    pub vectors: VectorParams,
    /// HNSW index configuration
    #[serde(default)]
    pub hnsw_config: Option<HnswConfigRequest>,
}

/// Single point for upsert
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PointStruct {
    /// Point ID (u64)
    pub id: u64,
    /// Vector data
    pub vector: Vec<f32>,
    /// Optional payload (JSON object)
    #[serde(default)]
    pub payload: Option<HashMap<String, serde_json::Value>>,
}

/// Upsert points request
///
/// Qdrant-compatible format:
/// ```json
/// {
///   "points": [
///     { "id": 1, "vector": [0.1, 0.2, ...], "payload": {"key": "value"} }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct UpsertPointsRequest {
    /// Points to upsert
    pub points: Vec<PointStruct>,
}

/// Search request
///
/// Qdrant-compatible format:
/// ```json
/// {
///   "vector": [0.1, 0.2, ...],
///   "limit": 10,
///   "with_payload": true,
///   "with_vector": false,
///   "score_threshold": 0.5
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct SearchRequest {
    /// Query vector
    pub vector: Vec<f32>,
    /// Number of results to return
    pub limit: usize,
    /// Include payload in results
    #[serde(default = "default_true")]
    pub with_payload: bool,
    /// Include vector in results
    #[serde(default)]
    pub with_vector: bool,
    /// Override search ef parameter
    #[serde(default)]
    pub params: Option<SearchParams>,
    /// Filter results by score threshold
    #[serde(default)]
    pub score_threshold: Option<f32>,
}

/// Search parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct SearchParams {
    /// Override ef for this search
    #[serde(default)]
    pub ef: Option<usize>,
}

fn default_true() -> bool {
    true
}

/// Scroll request for paginated retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ScrollRequest {
    /// Number of points to return
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Offset for pagination (point ID to start after)
    #[serde(default)]
    pub offset: Option<u64>,
    /// Include payload in results
    #[serde(default = "default_true")]
    pub with_payload: bool,
    /// Include vector in results
    #[serde(default)]
    pub with_vector: bool,
}

fn default_limit() -> usize {
    10
}

/// Delete points request
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct DeletePointsRequest {
    /// Point IDs to delete
    pub points: Vec<u64>,
}

/// Get points request
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct GetPointsRequest {
    /// Point IDs to retrieve
    pub ids: Vec<u64>,
    /// Include payload in results
    #[serde(default = "default_true")]
    pub with_payload: bool,
    /// Include vector in results
    #[serde(default)]
    pub with_vector: bool,
}

// === Graph Extension DTOs ===

/// Add edge request (LatticeDB extension)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct AddEdgeRequest {
    /// Source point ID
    pub from_id: u64,
    /// Target point ID
    pub to_id: u64,
    /// Relation type name
    pub relation: String,
    /// Edge weight
    #[serde(default = "default_weight")]
    pub weight: f32,
}

fn default_weight() -> f32 {
    1.0
}

/// Traverse graph request (LatticeDB extension)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TraverseRequest {
    /// Starting point ID
    pub start_id: u64,
    /// Maximum traversal depth
    pub max_depth: usize,
    /// Filter by relation types (optional)
    #[serde(default)]
    pub relations: Option<Vec<String>>,
}

// === Qdrant v1.16+ Query API ===

/// Query request for /points/query endpoint (Qdrant v1.16+)
///
/// This is the newer unified search endpoint that supports multiple query types.
/// The `query` field can be a raw vector or a query object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct QueryRequest {
    /// Query - can be a raw vector or a query object with "nearest" field
    pub query: QueryVector,
    /// Number of results to return
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Include payload in results
    #[serde(default = "default_true")]
    pub with_payload: bool,
    /// Include vector in results
    #[serde(default)]
    pub with_vector: bool,
    /// Override search ef parameter
    #[serde(default)]
    pub params: Option<SearchParams>,
    /// Filter results by score threshold
    #[serde(default)]
    pub score_threshold: Option<f32>,
}

/// Query vector - supports multiple formats from qdrant-client
#[derive(Debug, Clone)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum QueryVector {
    /// Raw vector array
    Vector(Vec<f32>),
}

impl QueryVector {
    /// Extract the vector from any query format
    pub fn into_vector(self) -> Vec<f32> {
        match self {
            QueryVector::Vector(v) => v,
        }
    }
}

impl Serialize for QueryVector {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            QueryVector::Vector(v) => v.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for QueryVector {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, SeqAccess, Visitor};

        struct QueryVectorVisitor;

        impl<'de> Visitor<'de> for QueryVectorVisitor {
            type Value = QueryVector;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a vector array or a query object")
            }

            // Handle raw vector array: [0.1, 0.2, ...]
            fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
            where
                S: SeqAccess<'de>,
            {
                let mut vec = Vec::new();
                while let Some(val) = seq.next_element()? {
                    vec.push(val);
                }
                Ok(QueryVector::Vector(vec))
            }

            // Handle query object: {"nearest": [...]} or {"nearest": {"vector": [...]}}
            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut vector: Option<Vec<f32>> = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "nearest" => {
                            // Can be array or object with "vector" field
                            let value: serde_json::Value = map.next_value()?;
                            vector = Some(extract_vector_from_value(&value).map_err(de::Error::custom)?);
                        }
                        _ => {
                            // Skip unknown fields
                            let _ = map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }

                vector
                    .map(QueryVector::Vector)
                    .ok_or_else(|| de::Error::custom("expected 'nearest' field in query object"))
            }
        }

        deserializer.deserialize_any(QueryVectorVisitor)
    }
}

/// Extract vector from a JSON value (handles both array and object formats)
fn extract_vector_from_value(value: &serde_json::Value) -> Result<Vec<f32>, String> {
    match value {
        // Direct array: [0.1, 0.2, ...]
        serde_json::Value::Array(arr) => arr
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| "expected float in vector".to_string())
            })
            .collect(),
        // Object with vector field: {"vector": [0.1, 0.2, ...]}
        serde_json::Value::Object(obj) => {
            if let Some(vec_val) = obj.get("vector") {
                extract_vector_from_value(vec_val)
            } else {
                Err("expected 'vector' field in query object".to_string())
            }
        }
        _ => Err("expected array or object for vector".to_string()),
    }
}

impl From<QueryRequest> for SearchRequest {
    fn from(req: QueryRequest) -> Self {
        SearchRequest {
            vector: req.query.into_vector(),
            limit: req.limit,
            with_payload: req.with_payload,
            with_vector: req.with_vector,
            params: req.params,
            score_threshold: req.score_threshold,
        }
    }
}
