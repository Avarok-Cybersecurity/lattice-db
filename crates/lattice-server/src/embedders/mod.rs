//! Registry of text embedders, keyed by model name.
//!
//! Embedders are the I/O edge of text search: `lattice_core::text::Embedder`
//! is the pure seam, and implementations are registered here at startup
//! (native: `openai`, feature `openai-embedder`) or from JavaScript
//! (wasm: `wasm_embedder`). The registry never invents an embedder — a text
//! query naming an unregistered model fails with a clear error.

#[cfg(feature = "openai-embedder")]
pub mod openai;

use lattice_core::text::{is_lexical_model, Embedder};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Environment variables configuring the OpenAI-compatible embedder. Defined
/// here, outside the feature gate, so a binary built without the feature can
/// refuse to start when they are set instead of silently ignoring them.
pub const ENV_URL: &str = "LATTICE_EMBEDDER_URL";
pub const ENV_MODEL: &str = "LATTICE_EMBEDDER_MODEL";
pub const ENV_API_KEY: &str = "LATTICE_EMBEDDER_API_KEY";
pub const ENV_DIM: &str = "LATTICE_EMBEDDER_DIM";
pub const ENV_TIMEOUT_MS: &str = "LATTICE_EMBEDDER_TIMEOUT_MS";
pub const ENV_VARS: [&str; 5] = [ENV_URL, ENV_MODEL, ENV_API_KEY, ENV_DIM, ENV_TIMEOUT_MS];

/// A registered embedder, shareable across requests.
pub type SharedEmbedder = Arc<dyn Embedder>;

/// Why an embedder could not be registered.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RegisterError {
    #[error("embedder model name must not be empty")]
    EmptyModel,
    #[error("'{0}' is reserved for the built-in BM25 index")]
    ReservedModel(String),
    #[error("embedder '{0}' must declare a dimension of at least 1")]
    ZeroDimension(String),
}

/// Model name -> embedder.
#[derive(Default)]
pub struct EmbedderRegistry {
    by_model: RwLock<HashMap<String, SharedEmbedder>>,
}

impl EmbedderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register (or replace) the embedder for its model name.
    pub fn register(&self, embedder: SharedEmbedder) -> Result<(), RegisterError> {
        let model = embedder.model().to_string();
        if model.trim().is_empty() {
            return Err(RegisterError::EmptyModel);
        }
        if is_lexical_model(&model) {
            return Err(RegisterError::ReservedModel(model));
        }
        if embedder.dim() == 0 {
            return Err(RegisterError::ZeroDimension(model));
        }
        self.by_model.write().insert(model, embedder);
        Ok(())
    }

    pub fn get(&self, model: &str) -> Option<SharedEmbedder> {
        self.by_model.read().get(model).cloned()
    }

    pub fn models(&self) -> Vec<String> {
        let mut models: Vec<String> = self.by_model.read().keys().cloned().collect();
        models.sort();
        models
    }
}
