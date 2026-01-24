//! Authentication middleware for enterprise-grade API security
//!
//! Provides API key and Bearer token authentication.
//! Designed for stateless validation with minimal overhead.

use std::collections::HashSet;
use std::sync::Arc;
use thiserror::Error;

/// Authentication error
#[derive(Debug, Error)]
pub enum AuthError {
    #[error("Missing Authorization header")]
    MissingHeader,

    #[error("Invalid Authorization header format")]
    InvalidFormat,

    #[error("Invalid API key")]
    InvalidApiKey,

    #[error("Invalid Bearer token")]
    InvalidToken,

    #[error("No valid credentials configured")]
    NoCredentials,
}

/// Authentication configuration
///
/// Supports multiple authentication schemes:
/// - API keys: Simple secret key validation
/// - Bearer tokens: JWT or opaque token validation
///
/// # Example
/// ```
/// use lattice_server::auth::AuthConfig;
///
/// // Single API key
/// let config = AuthConfig::api_key("my-secret-key");
///
/// // Multiple API keys
/// let config = AuthConfig::api_keys(vec!["key1", "key2", "key3"]);
/// ```
#[derive(Clone)]
pub struct AuthConfig {
    /// Valid API keys (if any)
    api_keys: HashSet<String>,
    /// Valid Bearer tokens (if any)
    bearer_tokens: HashSet<String>,
    /// Paths that bypass authentication (e.g., health checks)
    public_paths: HashSet<String>,
}

impl AuthConfig {
    /// Create config with a single API key
    pub fn api_key(key: impl Into<String>) -> Self {
        let mut keys = HashSet::new();
        keys.insert(key.into());
        Self {
            api_keys: keys,
            bearer_tokens: HashSet::new(),
            public_paths: default_public_paths(),
        }
    }

    /// Create config with multiple API keys
    pub fn api_keys(keys: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            api_keys: keys.into_iter().map(|k| k.into()).collect(),
            bearer_tokens: HashSet::new(),
            public_paths: default_public_paths(),
        }
    }

    /// Create config with Bearer tokens
    pub fn bearer_tokens(tokens: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            api_keys: HashSet::new(),
            bearer_tokens: tokens.into_iter().map(|t| t.into()).collect(),
            public_paths: default_public_paths(),
        }
    }

    /// Add additional public paths (no auth required)
    pub fn with_public_paths(
        mut self,
        paths: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        for path in paths {
            self.public_paths.insert(path.into());
        }
        self
    }

    /// Create from environment variables
    ///
    /// Reads from:
    /// - `LATTICE_API_KEYS`: Comma-separated list of API keys
    /// - `LATTICE_BEARER_TOKENS`: Comma-separated list of Bearer tokens
    ///
    /// Returns None if no credentials are configured.
    pub fn from_env() -> Option<Self> {
        let api_keys: HashSet<String> = std::env::var("LATTICE_API_KEYS")
            .ok()
            .map(|s| s.split(',').map(|k| k.trim().to_string()).collect())
            .unwrap_or_default();

        let bearer_tokens: HashSet<String> = std::env::var("LATTICE_BEARER_TOKENS")
            .ok()
            .map(|s| s.split(',').map(|t| t.trim().to_string()).collect())
            .unwrap_or_default();

        if api_keys.is_empty() && bearer_tokens.is_empty() {
            return None;
        }

        Some(Self {
            api_keys,
            bearer_tokens,
            public_paths: default_public_paths(),
        })
    }

    /// Check if a path is public (no auth required)
    pub fn is_public_path(&self, path: &str) -> bool {
        self.public_paths.contains(path)
    }

    /// Validate an API key
    pub fn validate_api_key(&self, key: &str) -> bool {
        self.api_keys.contains(key)
    }

    /// Validate a Bearer token
    pub fn validate_bearer_token(&self, token: &str) -> bool {
        self.bearer_tokens.contains(token)
    }

    /// Check if any credentials are configured
    pub fn has_credentials(&self) -> bool {
        !self.api_keys.is_empty() || !self.bearer_tokens.is_empty()
    }
}

/// Default paths that don't require authentication
fn default_public_paths() -> HashSet<String> {
    let mut paths = HashSet::new();
    paths.insert("/".to_string());
    paths.insert("/health".to_string());
    paths.insert("/healthz".to_string());
    paths.insert("/ready".to_string());
    paths.insert("/readyz".to_string());
    paths
}

/// Authentication validator
///
/// Thread-safe validator for checking request authentication.
#[derive(Clone)]
pub struct Authenticator {
    config: Arc<AuthConfig>,
}

impl Authenticator {
    /// Create a new authenticator
    pub fn new(config: AuthConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }

    /// Validate a request's authorization
    ///
    /// # Arguments
    /// * `path` - The request path
    /// * `auth_header` - The Authorization header value (if present)
    ///
    /// # Returns
    /// * `Ok(())` if authorized
    /// * `Err(AuthError)` if not authorized
    pub fn validate(&self, path: &str, auth_header: Option<&str>) -> Result<(), AuthError> {
        // Public paths bypass authentication
        if self.config.is_public_path(path) {
            return Ok(());
        }

        // Must have credentials configured
        if !self.config.has_credentials() {
            return Err(AuthError::NoCredentials);
        }

        // Must have Authorization header
        let header = auth_header.ok_or(AuthError::MissingHeader)?;

        // Try API key format: "ApiKey <key>" or "X-Api-Key: <key>"
        if let Some(key) = header.strip_prefix("ApiKey ") {
            if self.config.validate_api_key(key.trim()) {
                return Ok(());
            }
            return Err(AuthError::InvalidApiKey);
        }

        // Try Bearer token format: "Bearer <token>"
        if let Some(token) = header.strip_prefix("Bearer ") {
            if self.config.validate_bearer_token(token.trim()) {
                return Ok(());
            }
            return Err(AuthError::InvalidToken);
        }

        // Unknown format
        Err(AuthError::InvalidFormat)
    }

    /// Get the config
    pub fn config(&self) -> &AuthConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_auth() {
        let config = AuthConfig::api_key("secret-key-123");
        let auth = Authenticator::new(config);

        // Valid key
        assert!(auth.validate("/api/test", Some("ApiKey secret-key-123")).is_ok());

        // Invalid key
        assert!(auth.validate("/api/test", Some("ApiKey wrong-key")).is_err());

        // Missing header
        assert!(auth.validate("/api/test", None).is_err());
    }

    #[test]
    fn test_bearer_token_auth() {
        let config = AuthConfig::bearer_tokens(vec!["token-abc", "token-xyz"]);
        let auth = Authenticator::new(config);

        // Valid tokens
        assert!(auth.validate("/api/test", Some("Bearer token-abc")).is_ok());
        assert!(auth.validate("/api/test", Some("Bearer token-xyz")).is_ok());

        // Invalid token
        assert!(auth.validate("/api/test", Some("Bearer invalid")).is_err());
    }

    #[test]
    fn test_public_paths() {
        let config = AuthConfig::api_key("secret");
        let auth = Authenticator::new(config);

        // Public paths don't need auth
        assert!(auth.validate("/health", None).is_ok());
        assert!(auth.validate("/healthz", None).is_ok());
        assert!(auth.validate("/ready", None).is_ok());

        // Non-public paths need auth
        assert!(auth.validate("/collections", None).is_err());
    }

    #[test]
    fn test_multiple_api_keys() {
        let config = AuthConfig::api_keys(vec!["key1", "key2", "key3"]);
        let auth = Authenticator::new(config);

        assert!(auth.validate("/api", Some("ApiKey key1")).is_ok());
        assert!(auth.validate("/api", Some("ApiKey key2")).is_ok());
        assert!(auth.validate("/api", Some("ApiKey key3")).is_ok());
        assert!(auth.validate("/api", Some("ApiKey key4")).is_err());
    }

    #[test]
    fn test_custom_public_paths() {
        let config = AuthConfig::api_key("secret")
            .with_public_paths(vec!["/custom/public", "/another/public"]);
        let auth = Authenticator::new(config);

        assert!(auth.validate("/custom/public", None).is_ok());
        assert!(auth.validate("/another/public", None).is_ok());
        assert!(auth.validate("/private", None).is_err());
    }
}
