//! `POST /collections/{name}/points/query` — the Qdrant Query API.
//!
//! Dense vectors, BM25 text, embedded text and Reciprocal Rank Fusion over
//! `prefetch` stages. Scores keep each source's own meaning: dense results
//! carry the engine's distance score (unchanged from `/points/search`), BM25
//! results their BM25 score, fused results their RRF score (both higher is
//! better). `score_threshold` keeps dense results by the engine's rule and
//! keeps BM25/RRF results whose score is at least the threshold.

pub mod plan;

use crate::dto::{ApiResponse, Fusion, QueryRequest, QueryResponse, ScoredPoint};
use crate::handlers::search::json_value_to_map;
use crate::router::{json_response, AppState};
use lattice_core::text::{check_embedder_dim, reciprocal_rank_fusion, validate_embeddings};
use lattice_core::{
    EmbeddingConfig, LatticeResponse, LatticeResult, SearchQuery, SearchResult, TextSearchQuery,
};
use plan::{build_plan, Stage, StageOpts};
use std::collections::HashMap;

#[cfg(feature = "native")]
type Engine = crate::engine_wrapper::AnyEngine;
#[cfg(not(feature = "native"))]
type Engine = lattice_core::CollectionEngine;

/// HTTP status for a configured-but-unavailable embedder.
const STATUS_SERVICE_UNAVAILABLE: u16 = 503;
/// HTTP status for a failed or invalid upstream embedding call.
const STATUS_BAD_GATEWAY: u16 = 502;

/// Execute a Query API request.
pub async fn query_points(
    state: &AppState,
    collection_name: &str,
    request: QueryRequest,
) -> LatticeResponse {
    let Some(handle) = state.get_collection(collection_name) else {
        return LatticeResponse::not_found(&format!("Collection '{collection_name}' not found"));
    };
    let (embedding, dim) = {
        let engine = handle.read().await;
        (
            engine.config().embedding.clone(),
            engine.config().vector_dim(),
        )
    };
    let (with_payload, with_vector) = (request.with_payload, request.with_vector);
    let plan = match build_plan(request, embedding.as_ref()) {
        Ok(plan) => plan,
        Err(e) => return LatticeResponse::bad_request(&e),
    };
    // Embed before locking the collection: the call may take a network round trip.
    let plan = match embed_plan(state, plan, embedding.as_ref(), dim).await {
        Ok(plan) => plan,
        Err(response) => return response,
    };
    let engine = handle.read().await;
    match execute(&engine, &plan, with_payload, with_vector) {
        Ok(results) => json_response(&ApiResponse::ok(QueryResponse {
            points: results.into_iter().map(to_scored_point).collect(),
        })),
        Err(e) => LatticeResponse::bad_request(&format!("Query failed: {e}")),
    }
}

async fn embed_plan(
    state: &AppState,
    plan: Stage,
    embedding: Option<&EmbeddingConfig>,
    dim: usize,
) -> Result<Stage, LatticeResponse> {
    let texts = plan.texts_to_embed();
    if texts.is_empty() {
        return Ok(plan);
    }
    // build_plan only emits Embed stages for the collection's own model.
    let model = embedding
        .map(|e| e.model.as_str())
        .ok_or_else(|| LatticeResponse::internal_error("embed stage without embedding model"))?;
    let embedder = state.embedders.get(model).ok_or_else(|| {
        LatticeResponse::error(
            STATUS_SERVICE_UNAVAILABLE,
            &format!("no embedder is registered for model '{model}'"),
        )
    })?;
    check_embedder_dim(embedder.as_ref(), dim)
        .map_err(|e| LatticeResponse::error(STATUS_SERVICE_UNAVAILABLE, &e.to_string()))?;
    let vectors = embedder
        .embed(&texts)
        .await
        .and_then(|v| validate_embeddings(texts.len(), dim, v))
        .map_err(|e| LatticeResponse::error(STATUS_BAD_GATEWAY, &e.to_string()))?;
    plan.resolve_embeddings(&mut vectors.into_iter())
        .map_err(|e| LatticeResponse::error(STATUS_BAD_GATEWAY, &e))
}

fn execute(
    engine: &Engine,
    stage: &Stage,
    with_payload: bool,
    with_vector: bool,
) -> LatticeResult<Vec<SearchResult>> {
    match stage {
        Stage::Dense { vector, opts } => {
            let mut query = SearchQuery::new(vector.clone(), opts.limit);
            if with_payload {
                query = query.include_payload();
            }
            if with_vector {
                query = query.include_vector();
            }
            if let Some(threshold) = opts.score_threshold {
                query = query.with_score_threshold(threshold);
            }
            if let Some(ef) = opts.ef {
                query = query.with_ef(ef);
            }
            engine.search(query)
        }
        Stage::Lexical { text, opts } => {
            let results = engine.search_text(&TextSearchQuery {
                text: text.clone(),
                limit: opts.limit,
                with_payload,
                with_vector,
            })?;
            Ok(keep_at_least(results, opts))
        }
        Stage::Fuse {
            fusion: Fusion::Rrf { k },
            children,
            opts,
        } => {
            let lists = children
                .iter()
                .map(|c| execute(engine, c, with_payload, with_vector))
                .collect::<LatticeResult<Vec<_>>>()?;
            let rankings: Vec<Vec<u64>> = lists
                .iter()
                .map(|l| l.iter().map(|r| r.id).collect())
                .collect();
            let mut records: HashMap<u64, SearchResult> = HashMap::new();
            for result in lists.into_iter().flatten() {
                records.entry(result.id).or_insert(result);
            }
            let fused = reciprocal_rank_fusion(&rankings, *k, opts.limit)
                .into_iter()
                .filter_map(|(id, score)| {
                    let mut record = records.remove(&id)?;
                    record.score = score;
                    Some(record)
                })
                .collect();
            Ok(keep_at_least(fused, opts))
        }
        Stage::Embed { .. } => Err(lattice_core::LatticeError::Internal {
            code: 50010,
            message: "embed stage reached execution unresolved".to_string(),
        }),
    }
}

/// Higher-is-better scores (BM25, RRF): keep results at or above the threshold.
fn keep_at_least(mut results: Vec<SearchResult>, opts: &StageOpts) -> Vec<SearchResult> {
    if let Some(threshold) = opts.score_threshold {
        results.retain(|r| r.score >= threshold);
    }
    results
}

fn to_scored_point(r: SearchResult) -> ScoredPoint {
    ScoredPoint {
        id: r.id,
        version: 0,
        score: r.score,
        payload: r.payload.map(json_value_to_map),
        vector: r.vector,
    }
}

#[cfg(test)]
mod tests;
