//! LatticeDB Server binary
//!
//! Starts the HTTP server with Qdrant-compatible REST API.
//! Uses raw Hyper for maximum performance.
//!
//! # Environment Variables
//!
//! - `RUST_LOG` - Log level (default: info)
//! - `LATTICE_RATE_LIMIT` - Enable rate limiting (any value)
//! - `LATTICE_API_KEYS` - Comma-separated API keys for authentication
//! - `LATTICE_BEARER_TOKENS` - Comma-separated Bearer tokens for authentication
//! - `LATTICE_TLS_CERT` - Path to TLS certificate PEM (requires `tls` feature)
//! - `LATTICE_TLS_KEY` - Path to TLS private key PEM (requires `tls` feature)

use lattice_server::{
    auth::{AuthConfig, Authenticator},
    hyper_transport::HyperTransport,
    rate_limit::RateLimitConfig,
    router::{new_app_state, route},
    LatticeTransport,
};
use std::env;
use tracing::info;
use tracing_subscriber::{fmt, EnvFilter};

#[cfg(feature = "tls")]
use lattice_server::{hyper_transport::HyperTlsTransport, tls::TlsConfig};

#[tokio::main]
async fn main() {
    // Initialize structured logging
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_level(true)
        .with_thread_ids(true)
        .init();

    // Parse command-line arguments (default port 6334 for benchmarks)
    let addr = env::args()
        .nth(1)
        .unwrap_or_else(|| "0.0.0.0:6334".to_string());

    println!("╔═══════════════════════════════════════════╗");
    println!("║           LatticeDB Server v0.1           ║");
    println!("║   Vector + Graph Database in Rust/WASM   ║");
    println!("╚═══════════════════════════════════════════╝");
    println!();

    info!(address = %addr, "Starting LatticeDB server");

    // Create shared application state
    let state = new_app_state();

    // Load optional authentication config from env
    let authenticator = AuthConfig::from_env().map(|config| {
        info!("Authentication enabled");
        Authenticator::new(config)
    });

    // Check rate limiting
    let rate_limit = if env::var("LATTICE_RATE_LIMIT").is_ok() {
        info!("Rate limiting enabled (100 req/s, burst 200)");
        Some(RateLimitConfig::production())
    } else {
        None
    };

    // Check for TLS configuration
    #[cfg(feature = "tls")]
    {
        if let (Ok(cert_path), Ok(key_path)) =
            (env::var("LATTICE_TLS_CERT"), env::var("LATTICE_TLS_KEY"))
        {
            // Note: Only log cert path, not key path (sensitive)
            info!(cert = %cert_path, "TLS enabled");

            let tls_config = match TlsConfig::from_pem_files(&cert_path, &key_path) {
                Ok(cfg) => cfg,
                Err(e) => {
                    eprintln!("Failed to load TLS config: {}", e);
                    std::process::exit(1);
                }
            };

            let mut transport = if let Some(rate_config) = rate_limit {
                HyperTlsTransport::with_rate_limit(&addr, tls_config, rate_config)
            } else {
                HyperTlsTransport::new(&addr, tls_config)
            };

            if let Some(auth) = authenticator {
                transport = transport.with_auth(auth);
            }

            if let Err(e) = transport
                .serve(move |request| {
                    let state = state.clone();
                    async move { route(state, request).await }
                })
                .await
            {
                eprintln!("Server error: {}", e);
                std::process::exit(1);
            }
            return;
        }
    }

    // HTTP (non-TLS) transport
    let mut transport = if let Some(rate_config) = rate_limit {
        HyperTransport::with_rate_limit(&addr, rate_config)
    } else {
        HyperTransport::new(&addr)
    };

    if let Some(auth) = authenticator {
        transport = transport.with_auth(auth);
    }

    if let Err(e) = transport
        .serve(move |request| {
            let state = state.clone();
            async move { route(state, request).await }
        })
        .await
    {
        eprintln!("Server error: {}", e);
        std::process::exit(1);
    }
}
