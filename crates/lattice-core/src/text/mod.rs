//! Text search: BM25 lexical index, sparse vectors, rank fusion and the
//! embedding seam.
//!
//! Everything here is pure computation (no I/O, wasm32-compatible):
//! - [`tokenizer`] — deterministic Unicode tokenizer
//! - [`bm25`] — inverted index and BM25 scoring
//! - [`sparse`] — Qdrant-shaped sparse vectors (the BM25 query representation)
//! - [`payload`] — keeps the index in step with a collection's payloads
//! - [`fusion`] — Reciprocal Rank Fusion
//! - [`embedder`] — the `Embedder` trait and model/dimension guards
//! - [`config`] — `text_index` / `embedding` collection settings

pub mod bm25;
pub mod config;
pub mod embedder;
pub mod fusion;
pub mod payload;
pub mod sparse;
pub mod tokenizer;

#[cfg(test)]
mod tests;

pub use bm25::{Bm25Params, LexicalHit, LexicalIndex};
pub use config::{EmbeddingConfig, TextIndexConfig, BM25_B, BM25_K1};
pub use embedder::{
    check_embedder_dim, is_lexical_model, route_text_model, validate_embeddings, EmbedError,
    Embedder, MaybeSendSync, TextModelError, TextModelRoute, LEXICAL_MODELS,
};
pub use fusion::{reciprocal_rank_fusion, RRF_K};
pub use payload::PayloadTextIndex;
pub use sparse::{SparseVector, SparseVectorError};

/// A lexical (BM25) query against a collection's text index.
#[derive(Debug, Clone)]
pub struct TextSearchQuery {
    pub text: String,
    pub limit: usize,
    pub with_payload: bool,
    pub with_vector: bool,
}

/// The error for a lexical query against a collection without `text_index`.
pub fn no_text_index_error(collection: &str) -> crate::error::LatticeError {
    crate::error::LatticeError::InvalidOperation {
        message: format!(
            "collection '{collection}' has no text_index; create it with \"text_index\": {{\"field\": \"<payload field>\"}} to enable BM25 queries"
        ),
    }
}
