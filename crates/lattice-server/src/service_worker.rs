//! Service Worker / fetch-event transport (WASM).
//!
//! Converts a platform `Request` into a [`LatticeRequest`], runs a handler
//! (normally [`crate::router::route`]), and converts the answer back into a
//! platform `Response`. The same conversion serves every fetch host:
//!
//! - [`ServiceWorkerTransport::serve`] registers its own `fetch` listener;
//! - `LatticeFetchHandler` (see `wasm_fetch.rs`) is called from a hand-written
//!   service worker or a Cloudflare Worker's `fetch(request, env)`.
//!
//! Prefix matching and headers live in [`crate::fetch_mapping`].

#[cfg(target_arch = "wasm32")]
pub use wasm::*;

#[cfg(target_arch = "wasm32")]
mod wasm {
    use crate::fetch_mapping::FetchConfig;
    use crate::wasm_route::build_request;
    use async_trait::async_trait;
    use js_sys::Uint8Array;
    use lattice_core::{LatticeRequest, LatticeResponse, LatticeTransport};
    use std::future::Future;
    use std::rc::Rc;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::{future_to_promise, JsFuture};
    use web_sys::{
        FetchEvent, Headers, Request, Response, ResponseInit, ServiceWorkerGlobalScope, Url,
    };

    /// Service Worker transport error
    #[derive(Debug)]
    pub struct ServiceWorkerError(String);

    impl std::fmt::Display for ServiceWorkerError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "ServiceWorker error: {}", self.0)
        }
    }

    impl std::error::Error for ServiceWorkerError {}

    /// Serves the router from a service worker's `fetch` events under a prefix.
    pub struct ServiceWorkerTransport {
        config: FetchConfig,
    }

    impl ServiceWorkerTransport {
        /// Create a transport that answers requests under `config`'s prefix.
        pub fn new(config: FetchConfig) -> Self {
            Self { config }
        }
    }

    #[async_trait(?Send)]
    impl LatticeTransport for ServiceWorkerTransport {
        type Error = ServiceWorkerError;

        /// Install the `fetch` listener and return. Requests outside the
        /// prefix are left to the network (no `respondWith`). The listener
        /// lives as long as the worker, so its closure is intentionally leaked.
        async fn serve<H, Fut>(self, handler: H) -> Result<(), Self::Error>
        where
            H: Fn(LatticeRequest) -> Fut + Clone + 'static,
            Fut: Future<Output = LatticeResponse> + 'static,
        {
            let scope: ServiceWorkerGlobalScope = js_sys::global().dyn_into().map_err(|_| {
                ServiceWorkerError("not running in a ServiceWorkerGlobalScope".into())
            })?;
            let config = Rc::new(self.config);
            let listener = Closure::<dyn Fn(FetchEvent)>::new(move |event: FetchEvent| {
                let request = event.request();
                if !handles_url(&config, &request.url()) {
                    return;
                }
                let config = Rc::clone(&config);
                let handler = handler.clone();
                let promise = future_to_promise(async move {
                    respond(&config, &handler, request).await.map(JsValue::from)
                });
                if let Err(e) = event.respond_with(&promise) {
                    wasm_bindgen::throw_val(e);
                }
            });
            scope
                .add_event_listener_with_callback("fetch", listener.as_ref().unchecked_ref())
                .map_err(|e| ServiceWorkerError(format!("addEventListener failed: {:?}", e)))?;
            listener.forget();
            Ok(())
        }
    }

    /// Whether an absolute URL falls under the configured prefix.
    pub(crate) fn handles_url(config: &FetchConfig, url: &str) -> bool {
        Url::new(url)
            .map(|u| config.router_path(&u.pathname()).is_some())
            .unwrap_or(false)
    }

    /// Answer one platform request through `handler`.
    ///
    /// A request outside the prefix gets 404 (callers normally check
    /// [`handles_url`] first); `OPTIONS` gets a 204 preflight when CORS is
    /// configured. Rejects only when the platform objects themselves fail.
    pub(crate) async fn respond<H, Fut>(
        config: &FetchConfig,
        handler: &H,
        request: Request,
    ) -> Result<Response, JsValue>
    where
        H: Fn(LatticeRequest) -> Fut,
        Fut: Future<Output = LatticeResponse>,
    {
        let url = Url::new(&request.url())?;
        let path = match config.router_path(&url.pathname()) {
            Some(p) => p,
            None => {
                let outside = LatticeResponse::not_found("Path is outside the LatticeDB prefix");
                return to_web_response(config, outside);
            }
        };
        let method = request.method();
        if method.eq_ignore_ascii_case("OPTIONS") && config.answers_preflight() {
            return web_response(204, config.preflight_headers(), None);
        }
        let body = if method.eq_ignore_ascii_case("GET") || method.eq_ignore_ascii_case("HEAD") {
            Vec::new()
        } else {
            let buffer = JsFuture::from(request.array_buffer()?).await?;
            Uint8Array::new(&buffer).to_vec()
        };
        let lattice_request = build_request(&method, &format!("{}{}", path, url.search()), body)?;
        to_web_response(config, handler(lattice_request).await)
    }

    pub(crate) fn to_web_response(
        config: &FetchConfig,
        response: LatticeResponse,
    ) -> Result<Response, JsValue> {
        let headers = config.response_headers(&response.headers);
        web_response(response.status, headers, Some(response.body))
    }

    fn web_response(
        status: u16,
        headers: Vec<(String, String)>,
        body: Option<Vec<u8>>,
    ) -> Result<Response, JsValue> {
        let platform_headers = Headers::new()?;
        for (key, value) in &headers {
            platform_headers.append(key, value)?;
        }
        let init = ResponseInit::new();
        init.set_status(status);
        init.set_headers(&platform_headers);
        match body {
            Some(mut bytes) => Response::new_with_opt_u8_array_and_init(Some(&mut bytes), &init),
            None => Response::new_with_opt_str_and_init(None, &init),
        }
    }
}

// Placeholder for non-wasm builds
#[cfg(not(target_arch = "wasm32"))]
#[deprecated(
    since = "0.1.0",
    note = "ServiceWorkerTransport requires WASM target (--target wasm32-unknown-unknown)"
)]
pub struct ServiceWorkerTransport;
