//! Corpus loading and snapshot bindings for JavaScript hosts.
//!
//! These are the *host's* API — how a page, service worker or Cloudflare
//! Worker fills its database before serving — so read-only mode (which
//! governs the request surface) does not apply to them.

use crate::handlers::export_import::{export_collection, import_collection, import_mode_from_str};
use crate::handlers::import_jsonl::import_jsonl_collection;
use crate::wasm_bindings::{block_on, response_to_js, LatticeDB};
use crate::wasm_bytes::bytes_from_js;
use lattice_core::LatticeResponse;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl LatticeDB {
    /// Import a `lattice-jsonl` dump into the collection named by its header.
    ///
    /// * `data` - the dump as a string or bytes (any `ArrayBufferView`/`ArrayBuffer`)
    /// * `mode` - `"create"` | `"replace"` | `"merge"` (required)
    ///
    /// Returns `{ status, result: { points_imported, points_skipped, dimension, mode } }`.
    #[wasm_bindgen(js_name = importJsonl)]
    pub fn import_jsonl(&mut self, data: JsValue, mode: &str) -> Result<JsValue, JsValue> {
        let mode = import_mode_from_str(mode).map_err(error_to_js)?;
        let bytes = bytes_from_js(&data, true)?;
        response_to_js(block_on(import_jsonl_collection(&self.state, mode, &bytes)))
    }

    /// Import a binary (rkyv) snapshot produced by `exportSnapshot` or
    /// `GET /collections/{name}/export`.
    ///
    /// * `bytes` - snapshot bytes (any `ArrayBufferView`/`ArrayBuffer`)
    /// * `name` - collection name to install it under
    /// * `mode` - `"create"` | `"replace"` | `"merge"` (required)
    #[wasm_bindgen(js_name = importSnapshot)]
    pub fn import_snapshot(
        &mut self,
        bytes: JsValue,
        name: &str,
        mode: &str,
    ) -> Result<JsValue, JsValue> {
        let mode = import_mode_from_str(mode).map_err(error_to_js)?;
        let data = bytes_from_js(&bytes, false)?;
        response_to_js(block_on(import_collection(&self.state, name, mode, data)))
    }

    /// Export a collection as a binary (rkyv) snapshot.
    #[wasm_bindgen(js_name = exportSnapshot)]
    pub fn export_snapshot(&self, name: &str) -> Result<js_sys::Uint8Array, JsValue> {
        let response = block_on(export_collection(&self.state, name));
        if response.status == 200 {
            Ok(js_sys::Uint8Array::from(response.body.as_slice()))
        } else {
            Err(error_to_js(response))
        }
    }
}

fn error_to_js(response: LatticeResponse) -> JsValue {
    JsValue::from_str(&String::from_utf8_lossy(&response.body))
}
