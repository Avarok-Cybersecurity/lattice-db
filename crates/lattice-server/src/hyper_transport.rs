//! Hyper HTTP transport - raw performance implementation
//!
//! Uses Hyper directly for minimum overhead HTTP handling.

use async_trait::async_trait;
use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use lattice_core::{LatticeRequest, LatticeResponse, LatticeTransport};
use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tokio::net::TcpListener;

#[cfg(feature = "openapi")]
use utoipa_swagger_ui::SwaggerUi;

/// Hyper transport error
#[derive(Debug, Error)]
pub enum HyperError {
    #[error("Server error: {0}")]
    Server(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Hyper error: {0}")]
    Hyper(#[from] hyper::Error),
}

/// Hyper HTTP transport - maximum performance
///
/// Uses raw Hyper for direct HTTP handling with minimal overhead.
pub struct HyperTransport {
    addr: String,
}

impl HyperTransport {
    /// Create a new Hyper transport
    pub fn new(addr: impl Into<String>) -> Self {
        Self { addr: addr.into() }
    }
}

#[async_trait]
impl LatticeTransport for HyperTransport {
    type Error = HyperError;

    async fn serve<H, Fut>(self, handler: H) -> Result<(), Self::Error>
    where
        H: Fn(LatticeRequest) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = LatticeResponse> + Send + 'static,
    {
        let addr: SocketAddr = self.addr.parse().map_err(|e| {
            HyperError::Server(format!("Invalid address '{}': {}", self.addr, e))
        })?;

        let listener = TcpListener::bind(addr).await?;
        println!("LatticeDB server (Hyper) listening on {}", addr);

        #[cfg(feature = "openapi")]
        println!("OpenAPI docs available at http://{}/docs", addr);

        let handler = Arc::new(handler);

        loop {
            let (stream, _) = listener.accept().await?;

            // Enable TCP_NODELAY for lower latency
            let _ = stream.set_nodelay(true);

            let io = TokioIo::new(stream);
            let handler = handler.clone();

            tokio::task::spawn(async move {
                let service = service_fn(move |req| {
                    let handler = handler.clone();
                    async move { handle_request(req, handler).await }
                });

                // Use optimized HTTP/1.1 settings with pipelining
                if let Err(err) = http1::Builder::new()
                    .keep_alive(true)
                    .pipeline_flush(true) // Enable HTTP pipelining
                    .serve_connection(io, service)
                    .await
                {
                    eprintln!("Connection error: {}", err);
                }
            });
        }
    }
}

// Static error responses to avoid allocations
static METHOD_NOT_ALLOWED: &[u8] = b"Method not allowed";
static CONTENT_TYPE_JSON: &str = "application/json";

/// Handle incoming HTTP requests with minimal overhead
async fn handle_request<H, Fut>(
    req: Request<Incoming>,
    handler: Arc<H>,
) -> Result<Response<Full<Bytes>>, Infallible>
where
    H: Fn(LatticeRequest) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = LatticeResponse> + Send,
{
    // === TIMING: Request start ===
    let t_start = Instant::now();

    // Extract method directly - use static string slices (zero allocation)
    let method: &'static str = match req.method() {
        &Method::GET => "GET",
        &Method::POST => "POST",
        &Method::PUT => "PUT",
        &Method::DELETE => "DELETE",
        &Method::PATCH => "PATCH",
        &Method::HEAD => "HEAD",
        &Method::OPTIONS => "OPTIONS",
        _ => {
            return Ok(Response::builder()
                .status(StatusCode::METHOD_NOT_ALLOWED)
                .body(Full::new(Bytes::from_static(METHOD_NOT_ALLOWED)))
                .unwrap());
        }
    };

    // Extract path - single allocation
    let path = req.uri().path().to_owned();

    // Read body - zero-copy when Bytes has exclusive ownership
    let body: Vec<u8> = match req.collect().await {
        Ok(collected) => {
            let bytes = collected.to_bytes();
            // Bytes::into() is zero-copy when it has exclusive ownership
            bytes.into()
        }
        Err(e) => {
            return Ok(Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Full::new(Bytes::from(format!("Failed to read body: {}", e))))
                .unwrap());
        }
    };

    // === TIMING: Body collected ===
    let t_body = Instant::now();

    // Build minimal LatticeRequest - method uses String::from static str (optimized)
    let lattice_request = LatticeRequest {
        method: String::from(method),
        path,
        body,
        headers: std::collections::HashMap::new(),
    };

    // Call handler
    let response = handler(lattice_request).await;

    // === TIMING: Handler complete ===
    let t_handler = Instant::now();

    // Calculate timing durations in microseconds
    let body_us = t_body.duration_since(t_start).as_micros();
    let handler_us = t_handler.duration_since(t_body).as_micros();
    let total_us = t_handler.duration_since(t_start).as_micros();

    // Format Server-Timing header (RFC 6797 format)
    let server_timing = format!(
        "body;dur={}, handler;dur={}, total;dur={}",
        body_us, handler_us, total_us
    );

    // Build response - fast path for common case (no custom headers)
    let status = StatusCode::from_u16(response.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    if response.headers.is_empty() {
        // Fast path: no custom headers
        return Ok(Response::builder()
            .status(status)
            .header("Content-Type", CONTENT_TYPE_JSON)
            .header("Server-Timing", server_timing)
            .body(Full::new(Bytes::from(response.body)))
            .unwrap());
    }

    // Slow path: has custom headers
    let mut builder = Response::builder()
        .status(status)
        .header("Content-Type", CONTENT_TYPE_JSON)
        .header("Server-Timing", server_timing);

    for (key, value) in response.headers {
        builder = builder.header(key, value);
    }

    Ok(builder
        .body(Full::new(Bytes::from(response.body)))
        .unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyper_transport_new() {
        let transport = HyperTransport::new("127.0.0.1:6334");
        assert_eq!(transport.addr, "127.0.0.1:6334");
    }
}
