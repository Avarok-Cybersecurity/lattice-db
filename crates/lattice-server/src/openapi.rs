//! OpenAPI documentation module
//!
//! Provides OpenAPI spec generation via utoipa.
//! Only compiled when the `openapi` feature is enabled.

use utoipa::OpenApi;

use crate::dto::{
    // Request DTOs
    AddEdgeRequest, CreateCollectionRequest, DeletePointsRequest, GetPointsRequest,
    HnswConfigRequest, PointStruct, ScrollRequest, SearchParams, SearchRequest,
    TraverseRequest, UpsertPointsRequest, VectorParams,
    // Response DTOs (Qdrant-compatible)
    AddEdgeResult, CollectionConfigResponse, CollectionDescription, CollectionInfo,
    CollectionParamsResponse, CollectionStatus, CollectionsResponse, EdgeRecord,
    HnswConfigResponse, OptimizersConfigResponse, OptimizersStatus, PayloadIndexInfo,
    PointRecord, ScoredPoint, ScrollResult, TraversalResult, UpdateResult, UpsertResult,
    VectorParamsResponse, WalConfigResponse,
    // OpenAPI response wrappers
    ApiResponseAddEdgeResult, ApiResponseBoolResult, ApiResponseCollectionInfo,
    ApiResponseCollectionsResponse, ApiResponseScrollResult, ApiResponseTraversalResult,
    ApiResponseUpdateResult, ApiResponseUpsertResult, ApiResponseVecPointRecord,
    ApiResponseVecScoredPoint,
};

/// OpenAPI documentation for LatticeDB API
#[derive(OpenApi)]
#[openapi(
    info(
        title = "LatticeDB API",
        version = "0.1.0",
        description = "High-performance Vector + Graph database REST API. Qdrant-compatible with graph extensions.",
        license(name = "MIT OR Apache-2.0")
    ),
    tags(
        (name = "Collections", description = "Collection management operations"),
        (name = "Points", description = "Point CRUD operations"),
        (name = "Search", description = "Vector search and scroll operations"),
        (name = "Graph", description = "Graph edge and traversal operations (LatticeDB extension)")
    ),
    paths(
        // Collections
        crate::handlers::collections::list_collections,
        crate::handlers::collections::create_collection,
        crate::handlers::collections::get_collection,
        crate::handlers::collections::delete_collection,
        // Points
        crate::handlers::points::upsert_points,
        crate::handlers::points::get_points,
        crate::handlers::points::delete_points,
        // Search
        crate::handlers::search::search_points,
        crate::handlers::search::scroll_points,
        // Graph
        crate::handlers::points::add_edge,
        crate::handlers::search::traverse_graph,
    ),
    components(schemas(
        // Request schemas
        VectorParams,
        HnswConfigRequest,
        CreateCollectionRequest,
        PointStruct,
        UpsertPointsRequest,
        SearchRequest,
        SearchParams,
        ScrollRequest,
        DeletePointsRequest,
        GetPointsRequest,
        AddEdgeRequest,
        TraverseRequest,
        // Response schemas (Qdrant-compatible)
        CollectionInfo,
        CollectionStatus,
        OptimizersStatus,
        CollectionConfigResponse,
        CollectionParamsResponse,
        VectorParamsResponse,
        HnswConfigResponse,
        OptimizersConfigResponse,
        WalConfigResponse,
        PayloadIndexInfo,
        ScoredPoint,
        PointRecord,
        ScrollResult,
        UpsertResult,
        UpdateResult,
        EdgeRecord,
        TraversalResult,
        AddEdgeResult,
        CollectionDescription,
        CollectionsResponse,
        // API response wrappers
        ApiResponseCollectionsResponse,
        ApiResponseCollectionInfo,
        ApiResponseBoolResult,
        ApiResponseUpsertResult,
        ApiResponseUpdateResult,
        ApiResponseScrollResult,
        ApiResponseAddEdgeResult,
        ApiResponseTraversalResult,
        ApiResponseVecScoredPoint,
        ApiResponseVecPointRecord,
    ))
)]
pub struct ApiDoc;

/// Get the OpenAPI specification
pub fn openapi_spec() -> utoipa::openapi::OpenApi {
    ApiDoc::openapi()
}
