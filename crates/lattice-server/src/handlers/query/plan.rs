//! Turn a Query API request into a validated stage tree (pure, no I/O).

use crate::dto::{Fusion, Prefetch, QueryInput, QueryRequest, SearchParams};
use lattice_core::text::{route_text_model, TextModelRoute};
use lattice_core::EmbeddingConfig;

/// Largest `limit` for any stage (DoS protection; matches the scroll cap).
pub const MAX_QUERY_LIMIT: usize = 10_000;
/// Deepest allowed prefetch nesting.
pub const MAX_PREFETCH_DEPTH: usize = 4;
/// Most stages one request may run.
pub const MAX_STAGES: usize = 32;
/// Largest `ef` (CPU protection; same bound as the search endpoints).
pub const MAX_EF: usize = 10_000;
/// Longest accepted query text, in bytes.
pub const MAX_QUERY_TEXT_BYTES: usize = 8_192;

/// Shared per-stage knobs.
#[derive(Debug, Clone, PartialEq)]
pub struct StageOpts {
    pub limit: usize,
    pub ef: Option<usize>,
    pub score_threshold: Option<f32>,
}

/// One node of the execution tree.
#[derive(Debug, Clone, PartialEq)]
pub enum Stage {
    /// HNSW search with a dense vector.
    Dense { vector: Vec<f32>, opts: StageOpts },
    /// Text to be embedded by the collection's model, then searched densely.
    Embed { text: String, opts: StageOpts },
    /// BM25 search over the collection's text index.
    Lexical { text: String, opts: StageOpts },
    /// Fusion of child rankings.
    Fuse {
        fusion: Fusion,
        children: Vec<Stage>,
        opts: StageOpts,
    },
}

/// Build the stage tree, rejecting what cannot be answered meaningfully.
pub fn build_plan(req: QueryRequest, embedding: Option<&EmbeddingConfig>) -> Result<Stage, String> {
    let mut count = 0;
    build_stage(
        req.query,
        req.prefetch,
        opts(req.limit, req.params, req.score_threshold)?,
        embedding,
        0,
        &mut count,
    )
}

fn build_prefetch(
    p: Prefetch,
    embedding: Option<&EmbeddingConfig>,
    depth: usize,
    count: &mut usize,
) -> Result<Stage, String> {
    let opts = opts(p.limit, p.params, p.score_threshold)?;
    build_stage(p.query, p.prefetch, opts, embedding, depth, count)
}

fn build_stage(
    query: QueryInput,
    prefetch: Vec<Prefetch>,
    opts: StageOpts,
    embedding: Option<&EmbeddingConfig>,
    depth: usize,
    count: &mut usize,
) -> Result<Stage, String> {
    *count += 1;
    if *count > MAX_STAGES {
        return Err(format!("query has more than {MAX_STAGES} stages"));
    }
    match query {
        QueryInput::Fusion(fusion) => {
            if prefetch.is_empty() {
                return Err("fusion requires at least one 'prefetch' query".to_string());
            }
            if depth >= MAX_PREFETCH_DEPTH {
                return Err(format!("prefetch nesting exceeds {MAX_PREFETCH_DEPTH} levels"));
            }
            let children = prefetch
                .into_iter()
                .map(|p| build_prefetch(p, embedding, depth + 1, count))
                .collect::<Result<_, _>>()?;
            Ok(Stage::Fuse {
                fusion,
                children,
                opts,
            })
        }
        _ if !prefetch.is_empty() => Err(
            "re-scoring prefetch results with a vector or text query is not supported; use {\"fusion\": \"rrf\"}"
                .to_string(),
        ),
        QueryInput::Vector(vector) => Ok(Stage::Dense { vector, opts }),
        QueryInput::Text { text, model } => {
            if text.len() > MAX_QUERY_TEXT_BYTES {
                return Err(format!("query text exceeds {MAX_QUERY_TEXT_BYTES} bytes"));
            }
            match route_text_model(&text, &model, embedding).map_err(|e| e.to_string())? {
                TextModelRoute::Lexical => Ok(Stage::Lexical { text, opts }),
                TextModelRoute::Dense => Ok(Stage::Embed { text, opts }),
            }
        }
    }
}

fn opts(
    limit: usize,
    params: Option<SearchParams>,
    score_threshold: Option<f32>,
) -> Result<StageOpts, String> {
    if limit == 0 || limit > MAX_QUERY_LIMIT {
        return Err(format!(
            "limit must be between 1 and {MAX_QUERY_LIMIT}, got {limit}"
        ));
    }
    let ef = params.and_then(|p| p.ef);
    if let Some(ef) = ef {
        if ef == 0 || ef > MAX_EF {
            return Err(format!("ef must be between 1 and {MAX_EF}"));
        }
    }
    if score_threshold.is_some_and(|t| !t.is_finite()) {
        return Err("score_threshold must be a finite number (not NaN or Infinity)".to_string());
    }
    Ok(StageOpts {
        limit,
        ef,
        score_threshold,
    })
}

impl Stage {
    /// Texts awaiting embedding, in depth-first order.
    pub fn texts_to_embed(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_texts(&mut out);
        out
    }

    fn collect_texts(&self, out: &mut Vec<String>) {
        match self {
            Stage::Embed { text, .. } => out.push(text.clone()),
            Stage::Fuse { children, .. } => children.iter().for_each(|c| c.collect_texts(out)),
            Stage::Dense { .. } | Stage::Lexical { .. } => {}
        }
    }

    /// Replace every `Embed` stage, depth-first, with a `Dense` stage using
    /// the matching vector. `vectors` must come from [`Stage::texts_to_embed`].
    pub fn resolve_embeddings(
        self,
        vectors: &mut std::vec::IntoIter<Vec<f32>>,
    ) -> Result<Stage, String> {
        Ok(match self {
            Stage::Embed { opts, .. } => Stage::Dense {
                vector: vectors
                    .next()
                    .ok_or("embedder returned fewer vectors than texts")?,
                opts,
            },
            Stage::Fuse {
                fusion,
                children,
                opts,
            } => Stage::Fuse {
                fusion,
                children: children
                    .into_iter()
                    .map(|c| c.resolve_embeddings(vectors))
                    .collect::<Result<_, _>>()?,
                opts,
            },
            other => other,
        })
    }
}
