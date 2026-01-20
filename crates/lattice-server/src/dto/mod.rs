//! Data Transfer Objects for Qdrant-compatible API
//!
//! These structures match the Qdrant JSON schema for API compatibility.

pub mod request;
pub mod response;

pub use request::*;
pub use response::*;

#[cfg(feature = "openapi")]
pub use response::openapi_types::*;
