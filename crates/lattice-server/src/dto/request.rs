//! Request DTOs for Qdrant-compatible API
//!
//! These structures match the Qdrant JSON schema for API compatibility.

use lattice_core::{EmbeddingConfig, TextIndexConfig, VectorIndexKind, VectorStorage};
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
    /// Search structure (LatticeDB extension): `"hnsw"` (Qdrant's behaviour,
    /// used when absent) or `"flat"` (exact scan, no graph to build or load)
    #[serde(default)]
    #[cfg_attr(feature = "openapi", schema(value_type = Option<String>))]
    pub index: VectorIndexKind,
    /// Stored precision (LatticeDB extension): `"f32"` (used when absent),
    /// `"f16"` or `"int8"`; vectors are rounded to it on write
    #[serde(default)]
    #[cfg_attr(feature = "openapi", schema(value_type = Option<String>))]
    pub storage: VectorStorage,
}

impl VectorParams {
    /// HNSW, f32 vectors of `size` dimensions under `distance`
    pub fn new(size: usize, distance: &str) -> Self {
        Self {
            size,
            distance: distance.to_string(),
            index: VectorIndexKind::Hnsw,
            storage: VectorStorage::F32,
        }
    }
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
/// Qdrant-compatible format with LatticeDB extensions:
/// ```json
/// {
///   "vectors": { "size": 128, "distance": "Cosine" },
///   "hnsw_config": { "m": 16, "ef_construct": 200 },
///   "durability": "durable"
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CreateCollectionRequest {
    /// Dense vector configuration; omit it for a collection without dense
    /// vectors (BM25-only with `text_index`, or graph-only)
    #[serde(default)]
    pub vectors: Option<VectorParams>,
    /// HNSW index configuration
    #[serde(default)]
    pub hnsw_config: Option<HnswConfigRequest>,
    /// Durability mode: "ephemeral" (default) or "durable" (ACID with WAL)
    ///
    /// - `ephemeral`: Fast in-memory mode, no persistence
    /// - `durable`: ACID mode with WAL, crash recovery, fsync
    #[serde(default)]
    pub durability: Option<String>,
    /// BM25 lexical index (LatticeDB extension):
    /// `{"field": "text", "k1": 1.2, "b": 0.75}`
    #[serde(default)]
    #[cfg_attr(feature = "openapi", schema(value_type = Option<Object>))]
    pub text_index: Option<TextIndexConfig>,
    /// Embedding model that produced the vectors (LatticeDB extension):
    /// `{"model": "text-embedding-3-small"}`; its dimension is `vectors.size`
    #[serde(default)]
    #[cfg_attr(feature = "openapi", schema(value_type = Option<Object>))]
    pub embedding: Option<EmbeddingConfig>,
}

/// Single point for upsert
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PointStruct {
    /// Point ID (u64)
    pub id: u64,
    /// Vector data (omit for a collection without dense vectors)
    #[serde(default)]
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

/// Batch search request for multiple queries
///
/// More efficient than calling search multiple times:
/// ```json
/// {
///   "searches": [
///     { "vector": [0.1, 0.2, ...], "limit": 10 },
///     { "vector": [0.3, 0.4, ...], "limit": 10 }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct BatchSearchRequest {
    /// List of search requests to execute in parallel
    pub searches: Vec<SearchRequest>,
}

pub(crate) fn default_true() -> bool {
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

pub(crate) fn default_limit() -> usize {
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

// === Cypher Query API (LatticeDB extension) ===

/// Cypher query request
///
/// Execute openCypher queries against the graph:
/// ```json
/// {
///   "query": "MATCH (n:Person) WHERE n.age > 25 RETURN n.name",
///   "parameters": { "min_age": 25 }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CypherQueryRequest {
    /// The Cypher query to execute
    pub query: String,
    /// Optional query parameters (for parameterized queries)
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}
