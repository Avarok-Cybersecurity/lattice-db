//! JavaScript-provided text embedder and the async query binding (wasm).
//!
//! ```javascript
//! const db = new LatticeDB();
//! // Any async function returning one vector (number[] or Float32Array) per text.
//! db.setEmbedder("text-embedding-3-small", 1536, async (texts) => myEmbed(texts));
//! const res = await db.queryPoints("docs", {
//!   prefetch: [
//!     { query: { text: "free plan", model: "bm25" }, limit: 20 },
//!     { query: { text: "free plan", model: "text-embedding-3-small" }, limit: 20 },
//!   ],
//!   query: { fusion: "rrf" },
//!   limit: 10,
//! });
//! ```
//!
//! `queryPoints` returns a Promise because an embedder may await the network;
//! the synchronous bindings in `wasm_bindings` cannot drive such a future.

use crate::dto::QueryRequest;
use crate::handlers::query;
use crate::wasm_bindings::{response_to_js, LatticeDB};
use js_sys::{Array, Float32Array, Function, Promise};
use lattice_core::text::{EmbedError, Embedder};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::{future_to_promise, JsFuture};

/// An [`Embedder`] backed by a JS function `(texts: string[]) => vectors | Promise<vectors>`.
struct JsEmbedder {
    model: String,
    dim: usize,
    callback: Function,
}

#[async_trait::async_trait(?Send)]
impl Embedder for JsEmbedder {
    fn model(&self) -> &str {
        &self.model
    }

    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let input: Array = texts.iter().map(|t| JsValue::from_str(t)).collect();
        let returned = self
            .callback
            .call1(&JsValue::NULL, &input)
            .map_err(|e| EmbedError::Transport(format!("embedder callback threw: {e:?}")))?;
        let value = match returned.dyn_into::<Promise>() {
            Ok(promise) => JsFuture::from(promise)
                .await
                .map_err(|e| EmbedError::Transport(format!("embedder promise rejected: {e:?}")))?,
            Err(plain) => plain,
        };
        let outer: Array = value.dyn_into().map_err(|_| {
            EmbedError::InvalidResponse("embedder must return an array of vectors".to_string())
        })?;
        outer.iter().map(js_to_vector).collect()
    }
}

fn js_to_vector(value: JsValue) -> Result<Vec<f32>, EmbedError> {
    if let Some(typed) = value.dyn_ref::<Float32Array>() {
        return Ok(typed.to_vec());
    }
    if !Array::is_array(&value) {
        return Err(EmbedError::InvalidResponse(
            "each embedding must be a number[] or Float32Array".to_string(),
        ));
    }
    Array::from(&value)
        .iter()
        .map(|x| {
            x.as_f64().map(|f| f as f32).ok_or_else(|| {
                EmbedError::InvalidResponse("embedding values must be numbers".to_string())
            })
        })
        .collect()
}

#[wasm_bindgen]
impl LatticeDB {
    /// Register `embed` as the embedder for `model`, producing `dim`-sized
    /// vectors. Collections whose `embedding.model` is `model` can then answer
    /// `{"text": ..., "model": model}` queries through [`LatticeDB::query_points`].
    #[wasm_bindgen(js_name = setEmbedder)]
    pub fn set_embedder(&self, model: String, dim: usize, embed: Function) -> Result<(), JsValue> {
        // The browser runtime is single-threaded; the registry's Arc is shared
        // with no other thread.
        #[allow(clippy::arc_with_non_send_sync)]
        let embedder = Arc::new(JsEmbedder {
            model,
            dim,
            callback: embed,
        });
        self.state
            .embedders
            .register(embedder)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Qdrant Query API (`POST /collections/{name}/points/query`) as a Promise.
    #[wasm_bindgen(js_name = queryPoints)]
    pub fn query_points(&self, collection: String, request: JsValue) -> Promise {
        let state = self.state.clone();
        future_to_promise(async move {
            let request: QueryRequest = serde_wasm_bindgen::from_value(request)
                .map_err(|e| JsValue::from_str(&format!("Invalid query: {e}")))?;
            response_to_js(query::query_points(&state, &collection, request).await)
        })
    }
}
