//! Text -> dense vector embedding, as a pure trait (SBIO).
//!
//! Core defines WHAT an embedder is and validates WHAT it returns; it never
//! performs the I/O. Implementations live at the edges: an OpenAI-compatible
//! HTTP client in `lattice-server` (feature `openai-embedder`) and a JS
//! callback bridge for wasm (`lattice-server/src/wasm_embedder.rs`).

use super::config::EmbeddingConfig;
use thiserror::Error;

/// `Send + Sync` on native (handlers run on a multi-threaded runtime); no
/// bound on wasm32, where JS callbacks and promises are single-threaded.
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSendSync: Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync + ?Sized> MaybeSendSync for T {}

/// `Send + Sync` on native; no bound on wasm32.
#[cfg(target_arch = "wasm32")]
pub trait MaybeSendSync {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> MaybeSendSync for T {}

/// Why an embedding could not be produced or was rejected.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum EmbedError {
    #[error("embedding transport failed: {0}")]
    Transport(String),
    #[error("embedding service returned an invalid response: {0}")]
    InvalidResponse(String),
    #[error("embedder returned {actual} embeddings for {expected} inputs")]
    CountMismatch { expected: usize, actual: usize },
    #[error("embedding {index} has dimension {actual}, expected {expected}")]
    DimensionMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("embedding {index} contains a non-finite value")]
    NonFinite { index: usize },
}

/// Turns texts into dense vectors of a fixed model and dimension.
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
pub trait Embedder: MaybeSendSync {
    /// Model name, matched against a collection's [`EmbeddingConfig::model`].
    fn model(&self) -> &str;
    /// Output dimension, matched against the collection's `vectors.size`.
    fn dim(&self) -> usize;
    /// Embed `texts`; one vector per input, in input order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError>;
}

/// Check an embedder's raw output: one finite vector of `dim` per input.
pub fn validate_embeddings(
    inputs: usize,
    dim: usize,
    vectors: Vec<Vec<f32>>,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    if vectors.len() != inputs {
        return Err(EmbedError::CountMismatch {
            expected: inputs,
            actual: vectors.len(),
        });
    }
    for (index, v) in vectors.iter().enumerate() {
        if v.len() != dim {
            return Err(EmbedError::DimensionMismatch {
                index,
                expected: dim,
                actual: v.len(),
            });
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(EmbedError::NonFinite { index });
        }
    }
    Ok(vectors)
}

/// Model names that select the built-in BM25 index instead of an embedder.
/// `qdrant/bm25` is accepted so Qdrant clients work unchanged.
pub const LEXICAL_MODELS: [&str; 2] = ["bm25", "qdrant/bm25"];

pub fn is_lexical_model(model: &str) -> bool {
    LEXICAL_MODELS.iter().any(|m| m.eq_ignore_ascii_case(model))
}

/// How a `{"text", "model"}` query must be answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextModelRoute {
    /// Score with the collection's BM25 index.
    Lexical,
    /// Embed with the named model and run a dense search.
    Dense,
}

/// A text query the collection cannot answer meaningfully.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TextModelError {
    #[error("query text must not be empty")]
    EmptyText,
    #[error("collection has no embedding model configured; text queries must use model 'bm25' (or configure \"embedding\": {{\"model\": \"{requested}\"}})")]
    NoEmbeddingModel { requested: String },
    #[error("collection vectors were produced by model '{configured}', but the query asks for '{requested}'; vectors from different models are not comparable")]
    ModelMismatch {
        requested: String,
        configured: String,
    },
    #[error("embedder for model '{model}' produces {embedder_dim}-dimensional vectors but the collection stores {collection_dim}")]
    DimensionMismatch {
        model: String,
        embedder_dim: usize,
        collection_dim: usize,
    },
}

/// Decide how to answer a text query, refusing a model the collection's
/// vectors were not built with.
pub fn route_text_model(
    text: &str,
    model: &str,
    collection: Option<&EmbeddingConfig>,
) -> Result<TextModelRoute, TextModelError> {
    if text.trim().is_empty() {
        return Err(TextModelError::EmptyText);
    }
    if is_lexical_model(model) {
        return Ok(TextModelRoute::Lexical);
    }
    match collection {
        None => Err(TextModelError::NoEmbeddingModel {
            requested: model.to_string(),
        }),
        Some(cfg) if cfg.model != model => Err(TextModelError::ModelMismatch {
            requested: model.to_string(),
            configured: cfg.model.clone(),
        }),
        Some(_) => Ok(TextModelRoute::Dense),
    }
}

/// An embedder may only serve a collection whose vectors have its dimension.
pub fn check_embedder_dim(
    embedder: &dyn Embedder,
    collection_dim: usize,
) -> Result<(), TextModelError> {
    if embedder.dim() != collection_dim {
        return Err(TextModelError::DimensionMismatch {
            model: embedder.model().to_string(),
            embedder_dim: embedder.dim(),
            collection_dim,
        });
    }
    Ok(())
}
