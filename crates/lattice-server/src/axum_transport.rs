//! Axum HTTP transport implementation
//!
//! Provides HTTP server functionality using the Axum web framework.

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    response::IntoResponse,
    routing::any,
    Router,
};
use lattice_core::{LatticeRequest, LatticeResponse, LatticeTransport};
use std::future::Future;
use std::sync::Arc;
use thiserror::Error;
use tokio::net::TcpListener;

#[cfg(feature = "openapi")]
use utoipa_swagger_ui::SwaggerUi;

/// Axum transport error
#[derive(Debug, Error)]
pub enum AxumError {
    #[error("Server error: {0}")]
    Server(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Axum HTTP transport
///
/// Wraps Axum to implement the LatticeTransport trait.
pub struct AxumTransport {
    addr: String,
}

impl AxumTransport {
    /// Create a new Axum transport
    ///
    /// # Arguments
    /// * `addr` - Address to bind to (e.g., "0.0.0.0:6333")
    pub fn new(addr: impl Into<String>) -> Self {
        Self { addr: addr.into() }
    }
}

#[async_trait]
impl LatticeTransport for AxumTransport {
    type Error = AxumError;

    async fn serve<H, Fut>(self, handler: H) -> Result<(), Self::Error>
    where
        H: Fn(LatticeRequest) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = LatticeResponse> + Send + 'static,
    {
        // Wrap handler in Arc for sharing
        let handler = Arc::new(handler);

        // Build API router - catch all routes and methods
        let api_router = Router::new()
            .route("/*path", any(handle_request::<H, Fut>))
            .route("/", any(handle_request::<H, Fut>))
            .with_state(handler);

        // Merge with OpenAPI/SwaggerUI routes if feature enabled
        #[cfg(feature = "openapi")]
        let app = {
            use crate::openapi::openapi_spec;
            Router::new()
                .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", openapi_spec()))
                .merge(api_router)
        };

        #[cfg(not(feature = "openapi"))]
        let app = api_router;

        // Bind and serve
        let listener = TcpListener::bind(&self.addr).await?;
        println!("LatticeDB server listening on {}", self.addr);

        #[cfg(feature = "openapi")]
        println!("OpenAPI docs available at http://{}/docs", self.addr);

        axum::serve(listener, app)
            .await
            .map_err(|e| AxumError::Server(e.to_string()))
    }
}

/// Handle incoming HTTP requests
async fn handle_request<H, Fut>(
    State(handler): State<Arc<H>>,
    request: Request<Body>,
) -> impl IntoResponse
where
    H: Fn(LatticeRequest) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = LatticeResponse> + Send,
{
    // Extract method and path - HTTP methods are already uppercase
    let method = request.method().to_string();
    let path = request.uri().path().to_owned();

    // Skip header parsing entirely - we don't use headers in the API
    // This saves ~20-40µs of allocations per request
    let headers = std::collections::HashMap::new();

    // Read body directly into Vec<u8> without extra copy
    let body = match axum::body::to_bytes(request.into_body(), 10 * 1024 * 1024).await {
        Ok(bytes) => bytes.into(),
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("Failed to read body: {}", e),
            )
                .into_response();
        }
    };

    // Build LatticeRequest
    let lattice_request = LatticeRequest {
        method,
        path,
        body,
        headers,
    };

    // Call handler
    let response = handler(lattice_request).await;

    // Convert LatticeResponse to Axum response
    let status = StatusCode::from_u16(response.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    // Build response with security headers
    let mut builder = axum::http::Response::builder()
        .status(status)
        // Security headers to prevent common web vulnerabilities
        .header("X-Content-Type-Options", "nosniff")
        .header("X-Frame-Options", "DENY")
        .header("Cache-Control", "no-store")
        .header("Content-Security-Policy", "default-src 'none'");

    // Add custom response headers
    for (key, value) in response.headers {
        builder = builder.header(key, value);
    }

    builder
        .body(Body::from(response.body))
        .unwrap()
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axum_transport_new() {
        let transport = AxumTransport::new("127.0.0.1:6333");
        assert_eq!(transport.addr, "127.0.0.1:6333");
    }
}
