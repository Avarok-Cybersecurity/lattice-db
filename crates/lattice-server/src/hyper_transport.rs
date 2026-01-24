//! Hyper HTTP transport - raw performance implementation
//!
//! Uses Hyper directly for minimum overhead HTTP handling.
//! Supports optional TLS for HTTPS (requires `tls` feature).

use crate::auth::Authenticator;
use crate::rate_limit::{RateLimitConfig, RateLimiter};
#[cfg(feature = "tls")]
use crate::tls::TlsConfig;
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
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tokio::net::TcpListener;
use tracing::debug;

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

    #[cfg(feature = "tls")]
    #[error("TLS error: {0}")]
    Tls(#[from] crate::tls::TlsError),
}

/// Hyper HTTP transport - maximum performance
///
/// Uses raw Hyper for direct HTTP handling with minimal overhead.
/// Optionally includes rate limiting for DoS protection and authentication.
pub struct HyperTransport {
    addr: String,
    rate_limiter: Option<Arc<RateLimiter>>,
    authenticator: Option<Arc<Authenticator>>,
}

impl HyperTransport {
    /// Create a new Hyper transport without rate limiting or auth
    pub fn new(addr: impl Into<String>) -> Self {
        Self {
            addr: addr.into(),
            rate_limiter: None,
            authenticator: None,
        }
    }

    /// Create with rate limiting enabled (recommended for production)
    pub fn with_rate_limit(addr: impl Into<String>, config: RateLimitConfig) -> Self {
        Self {
            addr: addr.into(),
            rate_limiter: Some(Arc::new(RateLimiter::new(config))),
            authenticator: None,
        }
    }

    /// Create with default production rate limits (100 req/s, burst 200)
    pub fn with_default_rate_limit(addr: impl Into<String>) -> Self {
        Self::with_rate_limit(addr, RateLimitConfig::production())
    }

    /// Add authentication to the transport
    pub fn with_auth(mut self, authenticator: Authenticator) -> Self {
        self.authenticator = Some(Arc::new(authenticator));
        self
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
        let addr: SocketAddr = self
            .addr
            .parse()
            .map_err(|e| HyperError::Server(format!("Invalid address '{}': {}", self.addr, e)))?;

        let listener = TcpListener::bind(addr).await?;
        println!("LatticeDB server (Hyper) listening on {}", addr);

        if self.rate_limiter.is_some() {
            println!("Rate limiting enabled (100 req/s, burst 200)");
        }

        if self.authenticator.is_some() {
            println!("Authentication enabled");
        }

        #[cfg(feature = "openapi")]
        println!("OpenAPI docs available at http://{}/docs", addr);

        let handler = Arc::new(handler);
        let rate_limiter = self.rate_limiter;
        let authenticator = self.authenticator;

        loop {
            let (stream, client_addr) = listener.accept().await?;

            // Enable TCP_NODELAY for lower latency
            let _ = stream.set_nodelay(true);

            let io = TokioIo::new(stream);
            let handler = handler.clone();
            let rate_limiter = rate_limiter.clone();
            let authenticator = authenticator.clone();
            let client_ip = client_addr.ip();

            tokio::task::spawn(async move {
                let service = service_fn(move |req| {
                    let handler = handler.clone();
                    let rate_limiter = rate_limiter.clone();
                    let authenticator = authenticator.clone();
                    async move {
                        handle_request(req, handler, rate_limiter.as_ref(), authenticator.as_ref(), client_ip).await
                    }
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

/// Hyper HTTPS transport with TLS encryption
///
/// Provides enterprise-grade HTTPS support with certificate-based TLS.
/// Requires `tls` feature.
///
/// # Example
/// ```ignore
/// use lattice_server::tls::TlsConfig;
/// use lattice_server::hyper_transport::HyperTlsTransport;
///
/// let tls = TlsConfig::from_pem_files("cert.pem", "key.pem")?;
/// let transport = HyperTlsTransport::new("0.0.0.0:6334", tls);
/// ```
#[cfg(feature = "tls")]
pub struct HyperTlsTransport {
    addr: String,
    tls_config: TlsConfig,
    rate_limiter: Option<Arc<RateLimiter>>,
    authenticator: Option<Arc<Authenticator>>,
}

#[cfg(feature = "tls")]
impl HyperTlsTransport {
    /// Create a new HTTPS transport with TLS
    pub fn new(addr: impl Into<String>, tls_config: TlsConfig) -> Self {
        Self {
            addr: addr.into(),
            tls_config,
            rate_limiter: None,
            authenticator: None,
        }
    }

    /// Create with rate limiting enabled
    pub fn with_rate_limit(
        addr: impl Into<String>,
        tls_config: TlsConfig,
        rate_config: RateLimitConfig,
    ) -> Self {
        Self {
            addr: addr.into(),
            tls_config,
            rate_limiter: Some(Arc::new(RateLimiter::new(rate_config))),
            authenticator: None,
        }
    }

    /// Add authentication to the transport
    pub fn with_auth(mut self, authenticator: Authenticator) -> Self {
        self.authenticator = Some(Arc::new(authenticator));
        self
    }
}

#[cfg(feature = "tls")]
#[async_trait]
impl LatticeTransport for HyperTlsTransport {
    type Error = HyperError;

    async fn serve<H, Fut>(self, handler: H) -> Result<(), Self::Error>
    where
        H: Fn(LatticeRequest) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = LatticeResponse> + Send + 'static,
    {
        let addr: SocketAddr = self
            .addr
            .parse()
            .map_err(|e| HyperError::Server(format!("Invalid address '{}': {}", self.addr, e)))?;

        let listener = TcpListener::bind(addr).await?;
        println!("LatticeDB server (Hyper+TLS) listening on https://{}", addr);

        if self.rate_limiter.is_some() {
            println!("Rate limiting enabled (100 req/s, burst 200)");
        }

        if self.authenticator.is_some() {
            println!("Authentication enabled");
        }

        let handler = Arc::new(handler);
        let rate_limiter = self.rate_limiter;
        let authenticator = self.authenticator;
        let tls_acceptor = self.tls_config.acceptor().clone();

        loop {
            let (stream, client_addr) = listener.accept().await?;

            // Enable TCP_NODELAY for lower latency
            let _ = stream.set_nodelay(true);

            let handler = handler.clone();
            let rate_limiter = rate_limiter.clone();
            let authenticator = authenticator.clone();
            let client_ip = client_addr.ip();
            let acceptor = tls_acceptor.clone();

            tokio::task::spawn(async move {
                // Perform TLS handshake
                let tls_stream = match acceptor.accept(stream).await {
                    Ok(s) => s,
                    Err(e) => {
                        debug!(error = %e, ip = %client_ip, "TLS handshake failed");
                        return;
                    }
                };

                let io = TokioIo::new(tls_stream);
                let service = service_fn(move |req| {
                    let handler = handler.clone();
                    let rate_limiter = rate_limiter.clone();
                    let authenticator = authenticator.clone();
                    async move {
                        handle_request(req, handler, rate_limiter.as_ref(), authenticator.as_ref(), client_ip).await
                    }
                });

                // Use optimized HTTP/1.1 settings with pipelining
                if let Err(err) = http1::Builder::new()
                    .keep_alive(true)
                    .pipeline_flush(true)
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
static PAYLOAD_TOO_LARGE: &[u8] = b"{\"status\":\"error\",\"result\":\"Request body too large\"}";

/// Maximum request body size (16 MB)
///
/// This prevents memory exhaustion from oversized requests.
/// Adjust via environment variable LATTICE_MAX_BODY_SIZE if needed.
const DEFAULT_MAX_BODY_SIZE: usize = 16 * 1024 * 1024;

fn max_body_size() -> usize {
    std::env::var("LATTICE_MAX_BODY_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_BODY_SIZE)
}

/// Rate limit exceeded response (static to avoid allocation)
static RATE_LIMITED: &[u8] = b"{\"status\":\"error\",\"result\":\"Rate limit exceeded\"}";

/// Authentication failure responses
static AUTH_MISSING: &[u8] = b"{\"status\":\"error\",\"result\":\"Missing Authorization header\"}";
static AUTH_INVALID: &[u8] = b"{\"status\":\"error\",\"result\":\"Invalid credentials\"}";

/// Handle incoming HTTP requests with minimal overhead
async fn handle_request<H, Fut>(
    req: Request<Incoming>,
    handler: Arc<H>,
    rate_limiter: Option<&Arc<RateLimiter>>,
    authenticator: Option<&Arc<Authenticator>>,
    client_ip: IpAddr,
) -> Result<Response<Full<Bytes>>, Infallible>
where
    H: Fn(LatticeRequest) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = LatticeResponse> + Send,
{
    // === Rate Limiting Check ===
    if let Some(limiter) = rate_limiter {
        if !limiter.check(client_ip) {
            debug!(ip = %client_ip, "Rate limit exceeded");
            let headers = limiter.headers(client_ip);
            let mut builder = Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .header("Content-Type", CONTENT_TYPE_JSON)
                .header("Retry-After", "1");

            for (key, value) in headers {
                builder = builder.header(key, value);
            }

            return Ok(builder
                .body(Full::new(Bytes::from_static(RATE_LIMITED)))
                .unwrap());
        }
    }

    // === Authentication Check ===
    if let Some(auth) = authenticator {
        let path = req.uri().path();
        let auth_header = req
            .headers()
            .get(hyper::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok());

        if let Err(e) = auth.validate(path, auth_header) {
            debug!(ip = %client_ip, path = %path, error = %e, "Authentication failed");
            let (status, body) = match e {
                crate::auth::AuthError::MissingHeader => {
                    (StatusCode::UNAUTHORIZED, AUTH_MISSING)
                }
                _ => (StatusCode::UNAUTHORIZED, AUTH_INVALID),
            };

            return Ok(Response::builder()
                .status(status)
                .header("Content-Type", CONTENT_TYPE_JSON)
                .header("WWW-Authenticate", "ApiKey, Bearer")
                .body(Full::new(Bytes::from_static(body)))
                .unwrap());
        }
    }

    // === TIMING: Request start ===
    let t_start = Instant::now();

    // Extract method directly - use static string slices (zero allocation)
    let method: &'static str = match *req.method() {
        Method::GET => "GET",
        Method::POST => "POST",
        Method::PUT => "PUT",
        Method::DELETE => "DELETE",
        Method::PATCH => "PATCH",
        Method::HEAD => "HEAD",
        Method::OPTIONS => "OPTIONS",
        _ => {
            return Ok(Response::builder()
                .status(StatusCode::METHOD_NOT_ALLOWED)
                .body(Full::new(Bytes::from_static(METHOD_NOT_ALLOWED)))
                .unwrap());
        }
    };

    // Extract path - single allocation
    let path = req.uri().path().to_owned();

    // Check Content-Length header to reject oversized requests early
    let max_size = max_body_size();
    if let Some(content_length) = req.headers().get(hyper::header::CONTENT_LENGTH) {
        if let Ok(len_str) = content_length.to_str() {
            if let Ok(len) = len_str.parse::<usize>() {
                if len > max_size {
                    return Ok(Response::builder()
                        .status(StatusCode::PAYLOAD_TOO_LARGE)
                        .header("Content-Type", CONTENT_TYPE_JSON)
                        .body(Full::new(Bytes::from_static(PAYLOAD_TOO_LARGE)))
                        .unwrap());
                }
            }
        }
    }

    // Read body - zero-copy when Bytes has exclusive ownership
    let body: Vec<u8> = match req.collect().await {
        Ok(collected) => {
            let bytes = collected.to_bytes();
            // Check actual body size (in case Content-Length was missing or wrong)
            if bytes.len() > max_size {
                return Ok(Response::builder()
                    .status(StatusCode::PAYLOAD_TOO_LARGE)
                    .header("Content-Type", CONTENT_TYPE_JSON)
                    .body(Full::new(Bytes::from_static(PAYLOAD_TOO_LARGE)))
                    .unwrap());
            }
            // Bytes::into() is zero-copy when it has exclusive ownership
            bytes.into()
        }
        Err(e) => {
            return Ok(Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Full::new(Bytes::from(format!(
                    "Failed to read body: {}",
                    e
                ))))
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

    // Build response
    let status = StatusCode::from_u16(response.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let mut builder = Response::builder()
        .status(status)
        .header("Content-Type", CONTENT_TYPE_JSON)
        .header("Server-Timing", server_timing);

    // Add rate limit headers if rate limiting is enabled
    if let Some(limiter) = rate_limiter {
        for (key, value) in limiter.headers(client_ip) {
            builder = builder.header(key, value);
        }
    }

    // Add custom response headers
    for (key, value) in response.headers {
        builder = builder.header(key, value);
    }

    Ok(builder.body(Full::new(Bytes::from(response.body))).unwrap())
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
