//! `LatticeFetchHandler`: answer `Request`s from a service worker or a
//! Cloudflare Worker with the native router.
//!
//! ```javascript
//! const db = new LatticeDB({ readOnly: true });
//! const api = db.fetchHandler({ prefix: '/api/lattice' });
//! self.addEventListener('fetch', (e) => {
//!   if (api.handles(e.request.url)) e.respondWith(api.handleFetch(e.request));
//! });
//! ```

use crate::fetch_mapping::FetchConfig;
use crate::router::{route, AppState};
use crate::service_worker::{handles_url, respond, to_web_response};
use crate::wasm_bindings::LatticeDB;
use js_sys::Promise;
use lattice_core::LatticeResponse;
use serde::Deserialize;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

/// `fetchHandler()` options.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FetchHandlerOptions {
    /// Mount path, e.g. `/api/lattice`, or `""` for every path (required)
    prefix: String,
    /// `Access-Control-Allow-Origin` value; absent means no CORS headers
    #[serde(rename = "corsOrigin", default)]
    cors_origin: Option<String>,
}

/// Fetch-event adapter over one database. Cheap to clone the underlying
/// state; the database and handler share collections.
#[wasm_bindgen]
pub struct LatticeFetchHandler {
    state: AppState,
    config: Rc<FetchConfig>,
}

#[wasm_bindgen]
impl LatticeDB {
    /// Create a fetch handler: `{ prefix: string, corsOrigin?: string }`.
    #[wasm_bindgen(js_name = fetchHandler)]
    pub fn fetch_handler(&self, options: JsValue) -> Result<LatticeFetchHandler, JsValue> {
        let opts: FetchHandlerOptions = serde_wasm_bindgen::from_value(options)
            .map_err(|e| JsValue::from_str(&format!("Invalid fetchHandler options: {}", e)))?;
        let config =
            FetchConfig::new(&opts.prefix, opts.cors_origin).map_err(|e| JsValue::from_str(&e))?;
        Ok(LatticeFetchHandler {
            state: self.state.clone(),
            config: Rc::new(config),
        })
    }
}

#[wasm_bindgen]
impl LatticeFetchHandler {
    /// The mount prefix this handler answers under.
    #[wasm_bindgen(getter)]
    pub fn prefix(&self) -> String {
        self.config.prefix().to_string()
    }

    /// Whether an absolute URL is under this handler's prefix.
    pub fn handles(&self, url: &str) -> bool {
        handles_url(&self.config, url)
    }

    /// Build an error `Response` with exactly the headers (security, CORS) a
    /// routed response would carry, for answers a JS host decides itself.
    #[wasm_bindgen(js_name = errorResponse)]
    pub fn error_response(&self, status: u16, message: &str) -> Result<web_sys::Response, JsValue> {
        if !(400..=599).contains(&status) {
            return Err(JsValue::from_str(&format!(
                "Not an error status: {}",
                status
            )));
        }
        // Encoded with serde_json (not `LatticeResponse::error`) so a message
        // carrying quotes or backslashes still yields valid JSON.
        let body = serde_json::json!({ "status": "error", "message": message }).to_string();
        let mut response = LatticeResponse::error(status, "");
        response.body = body.into_bytes();
        to_web_response(&self.config, response)
    }

    /// Answer a `Request`; resolves to a `Response` (HTTP errors are statuses).
    #[wasm_bindgen(js_name = handleFetch, unchecked_return_type = "Promise<Response>")]
    pub fn handle_fetch(&self, request: web_sys::Request) -> Promise {
        let state = self.state.clone();
        let config = Rc::clone(&self.config);
        future_to_promise(async move {
            let handler = |req| route(state.clone(), req);
            respond(&config, &handler, request).await.map(JsValue::from)
        })
    }
}
