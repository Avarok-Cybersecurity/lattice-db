// Allow some clippy lints
#![allow(clippy::derivable_impls)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_borrows_for_generic_args)]

//! LatticeDB Server - HTTP and Service Worker transport
//!
//! This crate provides the API layer for LatticeDB, including:
//! - Qdrant-compatible REST API
//! - Graph extension endpoints
//! - Transport implementations (Axum HTTP, Service Worker)
//!
//! # Quick Start (Native)
//!
//! ```ignore
//! use lattice_server::{
//!     axum_transport::AxumTransport,
//!     router::{new_app_state, route},
//!     LatticeTransport,
//! };
//!
//! #[tokio::main]
//! async fn main() {
//!     let state = new_app_state();
//!     let transport = AxumTransport::new("0.0.0.0:6333");
//!
//!     transport.serve(move |request| {
//!         let state = state.clone();
//!         async move { route(state, request).await }
//!     }).await.unwrap();
//! }
//! ```

// Configure WASM tests to run in browser
#[cfg(all(target_arch = "wasm32", test))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

pub mod auth;
pub mod dto;
pub mod handlers;
pub mod rate_limit;
pub mod router;

#[cfg(feature = "native")]
pub mod hyper_transport;

#[cfg(feature = "tls")]
pub mod tls;

#[cfg(feature = "axum-transport")]
pub mod axum_transport;

#[cfg(feature = "wasm")]
pub mod service_worker;

#[cfg(feature = "wasm")]
pub mod wasm_bindings;

#[cfg(feature = "openapi")]
pub mod openapi;

// Re-export transport trait
pub use lattice_core::{LatticeRequest, LatticeResponse, LatticeTransport};
