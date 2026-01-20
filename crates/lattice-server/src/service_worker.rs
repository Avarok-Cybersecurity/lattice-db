//! Service Worker transport implementation (WASM)
//!
//! TODO: Implement in Phase 3

#[cfg(target_arch = "wasm32")]
use async_trait::async_trait;
#[cfg(target_arch = "wasm32")]
use lattice_core::{LatticeRequest, LatticeResponse, LatticeTransport};
#[cfg(target_arch = "wasm32")]
use std::future::Future;

/// Service Worker transport error
#[cfg(target_arch = "wasm32")]
#[derive(Debug)]
pub struct ServiceWorkerError(String);

#[cfg(target_arch = "wasm32")]
impl std::fmt::Display for ServiceWorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ServiceWorker error: {}", self.0)
    }
}

#[cfg(target_arch = "wasm32")]
impl std::error::Error for ServiceWorkerError {}

/// Service Worker transport for browser environments
#[cfg(target_arch = "wasm32")]
pub struct ServiceWorkerTransport;

#[cfg(target_arch = "wasm32")]
impl ServiceWorkerTransport {
    /// Create a new Service Worker transport
    pub fn new() -> Self {
        Self
    }
}

#[cfg(target_arch = "wasm32")]
impl Default for ServiceWorkerTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
impl LatticeTransport for ServiceWorkerTransport {
    type Error = ServiceWorkerError;

    async fn serve<H, Fut>(self, _handler: H) -> Result<(), Self::Error>
    where
        H: Fn(LatticeRequest) -> Fut + Clone + 'static,
        Fut: Future<Output = LatticeResponse> + 'static,
    {
        todo!("ServiceWorkerTransport::serve - implement in Phase 3")
    }
}

// Placeholder for non-wasm builds
#[cfg(not(target_arch = "wasm32"))]
pub struct ServiceWorkerTransport;

#[cfg(not(target_arch = "wasm32"))]
impl ServiceWorkerTransport {
    pub fn new() -> Self {
        panic!("ServiceWorkerTransport is only available in WASM builds")
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for ServiceWorkerTransport {
    fn default() -> Self {
        Self::new()
    }
}
