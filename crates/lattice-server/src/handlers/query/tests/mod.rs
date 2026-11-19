//! Router-level tests for every `/points/query` form.
//!
//! The only stand-in is `ToyEmbedder`, a deterministic test embedder: the
//! production `Embedder` implementations call a network service or JavaScript,
//! neither of which exists in a unit test. Everything behind the trait —
//! routing, planning, model checks, fusion, the engines — is production code.

use crate::router::{new_app_state, route, AppState};
use lattice_core::text::{EmbedError, Embedder};
use lattice_core::LatticeRequest;
use serde_json::{json, Value};
use std::sync::Arc;

mod hybrid;
mod lexical;

/// Deliberately weak embedder: a 3-d vector from character statistics. It
/// carries almost no meaning, which is exactly what the hybrid test needs.
pub(super) struct ToyEmbedder {
    pub model: &'static str,
    pub dim: usize,
}

pub(super) fn toy_vector(text: &str) -> Vec<f32> {
    let len = text.chars().count() as f32;
    let vowels = text.chars().filter(|c| "aeiou".contains(*c)).count() as f32;
    let spaces = text.chars().filter(|c| c.is_whitespace()).count() as f32;
    vec![1.0 + len % 7.0, 1.0 + vowels % 5.0, 1.0 + spaces % 3.0]
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl Embedder for ToyEmbedder {
    fn model(&self) -> &str {
        self.model
    }
    fn dim(&self) -> usize {
        self.dim
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|t| toy_vector(t)).collect())
    }
}

pub(super) async fn call(state: &AppState, method: &str, path: &str, body: Value) -> (u16, Value) {
    let request = LatticeRequest::new(method, path).with_body(serde_json::to_vec(&body).unwrap());
    let response = route(state.clone(), request).await;
    let parsed = serde_json::from_slice(&response.body).expect("response body is JSON");
    (response.status, parsed)
}

pub(super) fn ids(body: &Value) -> Vec<u64> {
    body["result"]["points"]
        .as_array()
        .expect("result.points")
        .iter()
        .map(|p| p["id"].as_u64().unwrap())
        .collect()
}

pub(super) fn message(body: &Value) -> String {
    body["message"].as_str().unwrap_or_default().to_string()
}

/// A collection with a BM25 index on `text`, optionally recording a model.
pub(super) async fn seeded(model: Option<&str>, docs: &[(u64, &str)]) -> AppState {
    let state = new_app_state();
    let mut config = json!({
        "vectors": {"size": 3, "distance": "Cosine"},
        "text_index": {"field": "text"}
    });
    if let Some(model) = model {
        config["embedding"] = json!({"model": model});
    }
    let (status, body) = call(&state, "PUT", "/collections/docs", config).await;
    assert_eq!(status, 200, "{body}");
    let points: Vec<Value> = docs
        .iter()
        .map(|(id, text)| json!({"id": id, "vector": toy_vector(text), "payload": {"text": text}}))
        .collect();
    let (status, body) = call(
        &state,
        "PUT",
        "/collections/docs/points",
        json!({"points": points}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    state
}

pub(super) fn register_toy(state: &AppState, model: &'static str, dim: usize) {
    state
        .embedders
        .register(Arc::new(ToyEmbedder { model, dim }))
        .unwrap();
}
