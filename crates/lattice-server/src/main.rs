//! LatticeDB Server binary
//!
//! Starts the HTTP server with Qdrant-compatible REST API.

use lattice_server::{
    axum_transport::AxumTransport,
    router::{new_app_state, route},
    LatticeTransport,
};
use std::env;

#[tokio::main]
async fn main() {
    // Parse command-line arguments
    let addr = env::args()
        .nth(1)
        .unwrap_or_else(|| "0.0.0.0:6333".to_string());

    println!("╔═══════════════════════════════════════════╗");
    println!("║           LatticeDB Server v0.1           ║");
    println!("║   Vector + Graph Database in Rust/WASM   ║");
    println!("╚═══════════════════════════════════════════╝");
    println!();
    println!("Starting server on {}...", addr);

    // Create shared application state
    let state = new_app_state();

    // Create transport
    let transport = AxumTransport::new(&addr);

    // Start serving
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
