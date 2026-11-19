//! Collection-level configuration for lexical search and text embedding.
//!
//! Both are opt-in: a collection without a `text_index` has no lexical index,
//! and a collection without an `embedding` rejects every text query that names
//! an embedding model.

use crate::error::ConfigError;
use serde::{Deserialize, Serialize};

/// BM25 term-frequency saturation parameter `k1`.
///
/// 1.2 is the value recommended by Robertson & Zaragoza, "The Probabilistic
/// Relevance Framework: BM25 and Beyond", Foundations and Trends in
/// Information Retrieval 3(4), 2009, §3.4 (k1 in [1.2, 2.0], b = 0.75), and is
/// the default of Lucene's `BM25Similarity`.
pub const BM25_K1: f32 = 1.2;

/// BM25 document-length normalisation parameter `b`.
///
/// 0.75 per Robertson & Zaragoza (2009), §3.4 — see [`BM25_K1`].
pub const BM25_B: f32 = 0.75;

fn bm25_k1() -> f32 {
    BM25_K1
}

fn bm25_b() -> f32 {
    BM25_B
}

/// BM25 inverted index over one payload text field.
///
/// JSON: `{"field": "text", "k1": 1.2, "b": 0.75}`. `field` is required;
/// `k1`/`b` fall back to the documented constants [`BM25_K1`] / [`BM25_B`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextIndexConfig {
    /// Top-level payload key whose value (a string, or an array of strings)
    /// is tokenized and indexed.
    pub field: String,
    /// Term-frequency saturation.
    #[serde(default = "bm25_k1")]
    pub k1: f32,
    /// Length normalisation, in `[0, 1]`.
    #[serde(default = "bm25_b")]
    pub b: f32,
}

impl TextIndexConfig {
    /// Index `field` with the standard BM25 constants.
    pub fn bm25(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            k1: BM25_K1,
            b: BM25_B,
        }
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.field.is_empty() {
            return Err(ConfigError::InvalidParameter {
                name: "text_index.field",
                message: "must not be empty".into(),
            });
        }
        if !self.k1.is_finite() || self.k1 < 0.0 {
            return Err(ConfigError::InvalidParameter {
                name: "text_index.k1",
                message: format!("must be a finite number >= 0, got {}", self.k1),
            });
        }
        if !self.b.is_finite() || !(0.0..=1.0).contains(&self.b) {
            return Err(ConfigError::InvalidParameter {
                name: "text_index.b",
                message: format!("must be within [0, 1], got {}", self.b),
            });
        }
        Ok(())
    }
}

/// The embedding model that produced the collection's dense vectors.
///
/// JSON: `{"model": "text-embedding-3-small"}`. The dimension is NOT repeated
/// here: it is the collection's `vectors.size` (single source of truth). A text
/// query naming any other model is rejected, because its vectors would live in
/// a different space and every score would be silently meaningless.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingConfig {
    pub model: String,
}

impl EmbeddingConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model.trim().is_empty() {
            return Err(ConfigError::InvalidParameter {
                name: "embedding.model",
                message: "must not be empty".into(),
            });
        }
        if super::is_lexical_model(&self.model) {
            return Err(ConfigError::InvalidParameter {
                name: "embedding.model",
                message: format!(
                    "'{}' is the reserved lexical model name, not an embedding model",
                    self.model
                ),
            });
        }
        Ok(())
    }
}
