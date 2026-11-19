//! `route()` for JavaScript: the native server's router, unchanged.
//!
//! A page, a service worker and a Cloudflare Worker all call this, so every
//! host exposes the identical Qdrant-compatible REST surface (including
//! read-only enforcement) without re-implementing any endpoint.

use crate::router::{route, AppState};
use crate::wasm_bindings::LatticeDB;
use crate::wasm_bytes::optional_bytes_from_js;
use js_sys::{Object, Promise, Reflect, Uint8Array};
use lattice_core::{LatticeRequest, LatticeResponse};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

#[wasm_bindgen(typescript_custom_section)]
const ROUTE_TYPES: &'static str = r#"
/** Result of `LatticeDB.route()`. */
export interface LatticeRouteResponse {
  status: number;
  headers: Record<string, string>;
  body: Uint8Array;
}
"#;

#[wasm_bindgen]
impl LatticeDB {
    /// Drive one request through the REST router.
    ///
    /// * `method` - HTTP method (case-insensitive)
    /// * `path` - path plus optional query string, e.g. `/collections/docs/points/query`
    /// * `body` - optional body: string, `ArrayBufferView` or `ArrayBuffer`
    ///
    /// Resolves to `{ status, headers, body: Uint8Array }`; HTTP errors are
    /// statuses, not rejections. Rejects only on an invalid argument.
    #[wasm_bindgen(js_name = route, unchecked_return_type = "Promise<LatticeRouteResponse>")]
    pub fn route_js(
        &self,
        method: &str,
        path: &str,
        body: Option<JsValue>,
    ) -> Result<Promise, JsValue> {
        let body = match body {
            Some(value) => optional_bytes_from_js(&value)?,
            None => Vec::new(),
        };
        let request = build_request(method, path, body)?;
        Ok(dispatch(self.state.clone(), request))
    }
}

/// Validate and assemble a router request from host-supplied parts.
pub(crate) fn build_request(
    method: &str,
    path: &str,
    body: Vec<u8>,
) -> Result<LatticeRequest, JsValue> {
    if method.is_empty() || !method.bytes().all(|b| b.is_ascii_alphabetic()) {
        return Err(JsValue::from_str(&format!(
            "Invalid HTTP method: {:?}",
            method
        )));
    }
    if !path.starts_with('/') {
        return Err(JsValue::from_str(&format!(
            "Path must start with '/': {:?}",
            path
        )));
    }
    Ok(LatticeRequest::new(method.to_ascii_uppercase(), path).with_body(body))
}

/// Run the router and resolve with the plain-object response.
pub(crate) fn dispatch(state: AppState, request: LatticeRequest) -> Promise {
    future_to_promise(async move { response_to_object(route(state, request).await) })
}

fn response_to_object(response: LatticeResponse) -> Result<JsValue, JsValue> {
    let headers = Object::new();
    for (key, value) in &response.headers {
        Reflect::set(&headers, &JsValue::from_str(key), &JsValue::from_str(value))?;
    }
    let out = Object::new();
    Reflect::set(&out, &"status".into(), &JsValue::from(response.status))?;
    Reflect::set(&out, &"headers".into(), &headers)?;
    let body = Uint8Array::from(response.body.as_slice());
    Reflect::set(&out, &"body".into(), &body)?;
    Ok(out.into())
}
