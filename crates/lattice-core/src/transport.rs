//! Transport abstraction layer (SBIO)
//!
//! This module defines the `LatticeTransport` trait that abstracts request/response handling.
//! The API layer uses this trait without knowing the underlying transport mechanism.
//!
//! # Implementations
//!
//! - `AxumTransport`: HTTP server using Axum (native)
//! - `ServiceWorkerTransport`: Fetch event interception (WASM/browser)

use async_trait::async_trait;
use std::collections::HashMap;
use std::future::Future;

/// Generic HTTP-like request
///
/// Abstracts the request format to work with both real HTTP and
/// Service Worker fetch events.
#[derive(Debug, Clone)]
pub struct LatticeRequest {
    /// HTTP method: GET, POST, PUT, DELETE
    pub method: String,
    /// Request path: /collections/{name}/points
    pub path: String,
    /// Raw request body (typically JSON)
    pub body: Vec<u8>,
    /// Request headers
    pub headers: HashMap<String, String>,
}

impl LatticeRequest {
    /// Create a new request
    pub fn new(method: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            path: path.into(),
            body: Vec::new(),
            headers: HashMap::new(),
        }
    }

    /// Set the request body
    pub fn with_body(mut self, body: impl Into<Vec<u8>>) -> Self {
        self.body = body.into();
        self
    }

    /// Add a header
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }
}

/// Generic HTTP-like response
#[derive(Debug, Clone)]
pub struct LatticeResponse {
    /// HTTP status code
    pub status: u16,
    /// Response body (typically JSON)
    pub body: Vec<u8>,
    /// Response headers
    pub headers: HashMap<String, String>,
}

impl LatticeResponse {
    /// Create a successful JSON response
    pub fn ok(body: impl Into<Vec<u8>>) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".into(), "application/json".into());
        Self {
            status: 200,
            body: body.into(),
            headers,
        }
    }

    /// Create an error response
    pub fn error(status: u16, message: &str) -> Self {
        let body = format!(r#"{{"status":"error","message":"{}"}}"#, message);
        let mut headers = HashMap::new();
        headers.insert("Content-Type".into(), "application/json".into());
        Self {
            status,
            body: body.into_bytes(),
            headers,
        }
    }

    /// Create a 404 Not Found response
    pub fn not_found(message: &str) -> Self {
        Self::error(404, message)
    }

    /// Create a 400 Bad Request response
    pub fn bad_request(message: &str) -> Self {
        Self::error(400, message)
    }

    /// Create a 500 Internal Server Error response
    pub fn internal_error(message: &str) -> Self {
        Self::error(500, message)
    }
}

/// Abstract transport interface (SBIO boundary)
///
/// This trait abstracts the concept of a "server". It allows the application
/// to serve requests via:
/// - TCP/HTTP (native server)
/// - Service Worker fetch interception (browser)
///
/// # Handler Pattern
///
/// The `serve` method takes a handler closure that processes requests.
/// This closure is called for each incoming request and returns a response.
///
/// # Platform Differences
///
/// - Native: Requires Send + Sync bounds for multi-threaded async runtimes
/// - WASM: Single-threaded, no Send bounds required
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait LatticeTransport: Send + Sync {
    /// Transport-specific error type
    type Error: std::error::Error + Send + Sync + 'static;

    /// Start serving requests
    ///
    /// The handler closure processes each request and returns a response.
    /// This method blocks until the server is shut down.
    async fn serve<H, Fut>(self, handler: H) -> Result<(), Self::Error>
    where
        H: Fn(LatticeRequest) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = LatticeResponse> + Send + 'static;
}

/// Abstract transport interface (SBIO boundary) - WASM version
///
/// WASM is single-threaded, so Send bounds are not required.
#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait LatticeTransport {
    /// Transport-specific error type
    type Error: std::error::Error + 'static;

    /// Start serving requests
    ///
    /// The handler closure processes each request and returns a response.
    async fn serve<H, Fut>(self, handler: H) -> Result<(), Self::Error>
    where
        H: Fn(LatticeRequest) -> Fut + Clone + 'static,
        Fut: Future<Output = LatticeResponse> + 'static;
}
