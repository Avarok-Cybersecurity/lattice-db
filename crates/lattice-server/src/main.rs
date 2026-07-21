//! LatticeDB Server binary
//!
//! Starts the HTTP server with Qdrant-compatible REST API.
//! Uses raw Hyper for maximum performance.
//!
//! # Environment Variables
//!
//! - `RUST_LOG` - Log level (default: info)
//! - `LATTICE_DATA_DIR` - Data directory for durable collections (enables persistence)
//! - `LATTICE_RATE_LIMIT` - Enable rate limiting (any value)
//! - `LATTICE_API_KEYS` - Comma-separated API keys for authentication
//! - `LATTICE_BEARER_TOKENS` - Comma-separated Bearer tokens for authentication
//! - `LATTICE_TLS_CERT` - Path to TLS certificate PEM (requires `tls` feature)
//! - `LATTICE_TLS_KEY` - Path to TLS private key PEM (requires `tls` feature)

use lattice_server::{
    auth::{AuthConfig, Authenticator},
    engine_wrapper::AnyEngine,
    hyper_transport::HyperTransport,
    rate_limit::RateLimitConfig,
    router::{new_app_state_with_config, route, AppState, ServerConfig},
    LatticeTransport,
};
use std::env;
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, EnvFilter};

#[cfg(feature = "tls")]
use lattice_server::{hyper_transport::HyperTlsTransport, tls::TlsConfig};

/// Scan data_dir for subdirectories containing config.json and open durable engines.
///
/// Each collection is stored as:
/// ```text
/// data_dir/{collection_name}/
/// ├── config.json
/// ├── wal/   (WAL storage)
/// └── data/  (data storage)
/// ```
///
/// Returns the number of collections successfully loaded.
/// Logs warnings for collections that fail to load (does not abort startup).
async fn load_collections_from_disk(data_dir: &Path, state: &AppState) -> usize {
    let mut entries = match tokio::fs::read_dir(data_dir).await {
        Ok(e) => e,
        Err(e) => {
            error!(path = %data_dir.display(), error = %e, "Failed to read data directory");
            return 0;
        }
    };

    let mut loaded = 0usize;
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();

        // Skip non-directories
        let is_dir = entry.file_type().await.map(|ft| ft.is_dir()).unwrap_or(false);
        if !is_dir {
            continue;
        }

        let name = match entry.file_name().into_string() {
            Ok(n) => n,
            Err(_) => {
                warn!(path = %path.display(), "Skipping directory with non-UTF8 name");
                continue;
            }
        };

        let config_path = path.join("config.json");
        if !config_path.exists() {
            warn!(collection = %name, "Skipping directory without config.json");
            continue;
        }

        // Read and parse collection config
        let config_bytes = match tokio::fs::read(&config_path).await {
            Ok(b) => b,
            Err(e) => {
                error!(collection = %name, error = %e, "Failed to read config.json");
                continue;
            }
        };

        let config: lattice_core::CollectionConfig = match serde_json::from_slice(&config_bytes) {
            Ok(c) => c,
            Err(e) => {
                error!(collection = %name, error = %e, "Failed to parse config.json");
                continue;
            }
        };

        // Open durable engine (replays WAL for crash recovery)
        let wal_storage = lattice_storage::DiskStorage::new(path.join("wal"), 4096);
        let data_storage = lattice_storage::DiskStorage::new(path.join("data"), 4096);

        let engine = match AnyEngine::open_durable(config, wal_storage, data_storage).await {
            Ok(e) => e,
            Err(e) => {
                error!(collection = %name, error = %e, "Failed to open durable engine");
                continue;
            }
        };

        let point_count = engine.point_count();
        if let Err(e) = state.insert_collection(name.clone(), engine) {
            error!(collection = %name, error = ?e, "Failed to insert recovered collection");
            continue;
        }

        info!(collection = %name, points = point_count, "Recovered collection from disk");
        loaded += 1;
    }

    loaded
}

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

    // Parse data directory (PCND: explicit opt-in via env var)
    let data_dir: Option<PathBuf> = env::var("LATTICE_DATA_DIR").ok().map(PathBuf::from);

    println!("╔═══════════════════════════════════════════╗");
    println!("║           LatticeDB Server v0.1           ║");
    println!("║   Vector + Graph Database in Rust/WASM   ║");
    println!("╚═══════════════════════════════════════════╝");
    println!();

    info!(address = %addr, "Starting LatticeDB server");

    // Build server config
    let mut config = ServerConfig::production();
    if let Some(ref dir) = data_dir {
        config = config.with_data_dir(dir.clone());
        info!(data_dir = %dir.display(), "Durable mode enabled");
    } else {
        info!("Ephemeral mode (no LATTICE_DATA_DIR set)");
    }

    // Create shared application state
    let state = new_app_state_with_config(config);

    // Load existing collections from disk (WAL replay for crash recovery)
    if let Some(ref dir) = data_dir {
        // Ensure data directory exists
        if let Err(e) = tokio::fs::create_dir_all(dir).await {
            eprintln!("Failed to create data directory {}: {}", dir.display(), e);
            std::process::exit(1);
        }

        let loaded = load_collections_from_disk(dir, &state).await;
        if loaded > 0 {
            info!(count = loaded, "Collections recovered from disk");
        }
    }

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
