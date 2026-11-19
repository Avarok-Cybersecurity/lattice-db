//! OpenAI-compatible `POST {base}/embeddings` client (feature `openai-embedder`).
//!
//! Configured only through explicit environment variables — there are no
//! defaults. Set all of them or none:
//!
//! | Variable | Meaning |
//! |---|---|
//! | `LATTICE_EMBEDDER_URL` | API base, e.g. `https://api.openai.com/v1` (`/embeddings` is appended) |
//! | `LATTICE_EMBEDDER_MODEL` | model name, matched against a collection's `embedding.model` |
//! | `LATTICE_EMBEDDER_API_KEY` | sent as `Authorization: Bearer …` |
//! | `LATTICE_EMBEDDER_DIM` | output dimension, matched against `vectors.size` |
//! | `LATTICE_EMBEDDER_TIMEOUT_MS` | per-request timeout |

use lattice_core::text::{is_lexical_model, validate_embeddings, EmbedError, Embedder};
use serde::Deserialize;
use std::time::Duration;
use thiserror::Error;

pub use super::{ENV_API_KEY, ENV_DIM, ENV_MODEL, ENV_TIMEOUT_MS, ENV_URL, ENV_VARS};

/// Longest slice of an error body echoed into an error message.
const MAX_ERROR_BODY_CHARS: usize = 512;

/// Invalid or partial embedder configuration.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EmbedderConfigError {
    #[error("embedder configuration is incomplete; missing: {}", .0.join(", "))]
    Missing(Vec<&'static str>),
    #[error("{var} is invalid: {message}")]
    Invalid { var: &'static str, message: String },
}

/// Validated client settings.
#[derive(Clone)]
pub struct OpenAiEmbedderConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
    pub dim: usize,
    pub timeout: Duration,
}

impl std::fmt::Debug for OpenAiEmbedderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiEmbedderConfig")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .field("dim", &self.dim)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl OpenAiEmbedderConfig {
    /// Read the settings through `lookup` (e.g. `|k| std::env::var(k).ok()`).
    /// `Ok(None)` when no variable is set; an error when only some are.
    pub fn from_lookup(
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Option<Self>, EmbedderConfigError> {
        let values: Vec<Option<String>> = ENV_VARS.iter().map(|v| lookup(v)).collect();
        if values.iter().all(Option::is_none) {
            return Ok(None);
        }
        let missing: Vec<&'static str> = ENV_VARS
            .iter()
            .zip(&values)
            .filter(|(_, v)| v.as_deref().map_or(true, |s| s.trim().is_empty()))
            .map(|(k, _)| *k)
            .collect();
        if !missing.is_empty() {
            return Err(EmbedderConfigError::Missing(missing));
        }
        let get = |i: usize| values[i].clone().unwrap_or_default().trim().to_string();
        let base_url = get(0).trim_end_matches('/').to_string();
        if !(base_url.starts_with("https://") || base_url.starts_with("http://")) {
            return Err(invalid(ENV_URL, "must start with http:// or https://"));
        }
        let model = get(1);
        if is_lexical_model(&model) {
            return Err(invalid(
                ENV_MODEL,
                "is reserved for the built-in BM25 index",
            ));
        }
        let dim = parse_positive(ENV_DIM, &get(3))?;
        let timeout_ms = parse_positive(ENV_TIMEOUT_MS, &get(4))?;
        Ok(Some(Self {
            base_url,
            model,
            api_key: get(2),
            dim,
            timeout: Duration::from_millis(timeout_ms as u64),
        }))
    }
}

fn invalid(var: &'static str, message: &str) -> EmbedderConfigError {
    EmbedderConfigError::Invalid {
        var,
        message: message.to_string(),
    }
}

fn parse_positive(var: &'static str, raw: &str) -> Result<usize, EmbedderConfigError> {
    raw.parse::<usize>()
        .ok()
        .filter(|&n| n >= 1)
        .ok_or_else(|| invalid(var, &format!("must be a positive integer, got {raw:?}")))
}

/// The HTTP embedder.
pub struct OpenAiEmbedder {
    client: reqwest::Client,
    endpoint: String,
    config: OpenAiEmbedderConfig,
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
    index: usize,
}

impl OpenAiEmbedder {
    pub fn new(config: OpenAiEmbedderConfig) -> Result<Self, EmbedError> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| EmbedError::Transport(e.to_string()))?;
        Ok(Self {
            client,
            endpoint: format!("{}/embeddings", config.base_url),
            config,
        })
    }
}

#[async_trait::async_trait]
impl Embedder for OpenAiEmbedder {
    fn model(&self) -> &str {
        &self.config.model
    }

    fn dim(&self) -> usize {
        self.config.dim
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": self.config.model,
            "input": texts,
        }))
        .map_err(|e| EmbedError::Transport(e.to_string()))?;
        let response = self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.config.api_key)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body)
            .send()
            .await
            .map_err(|e| EmbedError::Transport(e.to_string()))?;
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| EmbedError::Transport(e.to_string()))?;
        if !status.is_success() {
            let text: String = String::from_utf8_lossy(&bytes)
                .chars()
                .take(MAX_ERROR_BODY_CHARS)
                .collect();
            return Err(EmbedError::Transport(format!("HTTP {status}: {text}")));
        }
        let parsed: EmbeddingsResponse = serde_json::from_slice(&bytes)
            .map_err(|e| EmbedError::InvalidResponse(e.to_string()))?;
        let mut slots: Vec<Option<Vec<f32>>> = vec![None; parsed.data.len()];
        for item in parsed.data {
            let slot = slots.get_mut(item.index).ok_or_else(|| {
                EmbedError::InvalidResponse(format!("embedding index {} out of range", item.index))
            })?;
            if slot.replace(item.embedding).is_some() {
                return Err(EmbedError::InvalidResponse(format!(
                    "embedding index {} repeated",
                    item.index
                )));
            }
        }
        // Every slot is filled: n distinct indices in 0..n.
        let vectors = slots.into_iter().flatten().collect();
        validate_embeddings(texts.len(), self.config.dim, vectors)
    }
}

#[cfg(test)]
#[path = "openai_tests.rs"]
mod tests;
