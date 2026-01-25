//! Authentication middleware for enterprise-grade API security
//!
//! Provides API key and Bearer token authentication.
//! Designed for stateless validation with minimal overhead.
//!
//! # Security
//! Uses constant-time comparison to prevent timing attacks on credentials.

use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use subtle::ConstantTimeEq;
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
    pub fn with_public_paths(mut self, paths: impl IntoIterator<Item = impl Into<String>>) -> Self {
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

    /// Validate an API key using constant-time comparison
    ///
    /// # Security
    /// Uses constant-time comparison to prevent timing attacks that could
    /// be used to brute-force valid API keys character-by-character.
    pub fn validate_api_key(&self, key: &str) -> bool {
        self.api_keys
            .iter()
            .any(|k| k.as_bytes().ct_eq(key.as_bytes()).into())
    }

    /// Validate a Bearer token using constant-time comparison
    ///
    /// # Security
    /// Uses constant-time comparison to prevent timing attacks.
    pub fn validate_bearer_token(&self, token: &str) -> bool {
        self.bearer_tokens
            .iter()
            .any(|t| t.as_bytes().ct_eq(token.as_bytes()).into())
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

/// Track auth failures per IP with exponential backoff
///
/// Prevents brute-force attacks by adding increasing delays after failed attempts.
/// After `max_failures` consecutive failures, the IP is blocked for `lockout_duration`.
///
/// # Security
/// - Exponential backoff: 1s, 2s, 4s, 8s, ... (capped at lockout_duration)
/// - Per-IP tracking prevents attackers from trying many API keys rapidly
/// - Automatic cleanup of stale entries to prevent memory exhaustion
pub struct AuthRateLimiter {
    /// Failed attempt tracking per IP
    failures: Mutex<HashMap<IpAddr, FailureRecord>>,
    /// Maximum consecutive failures before lockout
    max_failures: u32,
    /// Base delay after first failure (doubled for each subsequent failure)
    base_delay: Duration,
    /// Maximum delay/lockout duration
    max_delay: Duration,
    /// Cleanup interval for stale entries
    cleanup_interval: Duration,
    /// Last cleanup time
    last_cleanup: Mutex<Instant>,
}

/// Record of failed auth attempts for a single IP
struct FailureRecord {
    /// Number of consecutive failures
    count: u32,
    /// Time of last failure
    last_failure: Instant,
    /// Blocked until this time (if locked out)
    blocked_until: Option<Instant>,
}

impl Default for AuthRateLimiter {
    /// Default: 5 failures, 1s base delay, 5min max lockout
    fn default() -> Self {
        Self::new(5, Duration::from_secs(1), Duration::from_secs(300))
    }
}

impl AuthRateLimiter {
    /// Create a new auth rate limiter
    ///
    /// # Arguments
    /// * `max_failures` - Number of failures before lockout
    /// * `base_delay` - Initial delay after first failure (exponentially increases)
    /// * `max_delay` - Maximum delay/lockout duration
    pub fn new(max_failures: u32, base_delay: Duration, max_delay: Duration) -> Self {
        Self {
            failures: Mutex::new(HashMap::new()),
            max_failures,
            base_delay,
            max_delay,
            cleanup_interval: Duration::from_secs(60),
            last_cleanup: Mutex::new(Instant::now()),
        }
    }

    /// Check if an IP is currently blocked
    ///
    /// Returns `Some(remaining_duration)` if blocked, `None` if allowed
    pub fn is_blocked(&self, ip: IpAddr) -> Option<Duration> {
        self.maybe_cleanup();

        let failures = self.failures.lock();
        if let Some(record) = failures.get(&ip) {
            if let Some(blocked_until) = record.blocked_until {
                let now = Instant::now();
                if now < blocked_until {
                    return Some(blocked_until - now);
                }
            }
        }
        None
    }

    /// Get the current delay for an IP based on failure count
    ///
    /// Returns the delay duration that should be enforced before allowing another attempt
    pub fn get_delay(&self, ip: IpAddr) -> Option<Duration> {
        let failures = self.failures.lock();
        if let Some(record) = failures.get(&ip) {
            if record.count == 0 {
                return None;
            }
            // Exponential backoff: base_delay * 2^(count-1), capped at max_delay
            let multiplier = 2u64.saturating_pow(record.count.saturating_sub(1));
            let delay_ms = self.base_delay.as_millis() as u64 * multiplier;
            let delay = Duration::from_millis(delay_ms.min(self.max_delay.as_millis() as u64));

            // Check if enough time has passed since last failure
            let elapsed = record.last_failure.elapsed();
            if elapsed < delay {
                return Some(delay - elapsed);
            }
        }
        None
    }

    /// Record a failed authentication attempt
    ///
    /// Increments failure count and potentially triggers lockout
    pub fn record_failure(&self, ip: IpAddr) {
        let mut failures = self.failures.lock();
        let now = Instant::now();

        let record = failures.entry(ip).or_insert(FailureRecord {
            count: 0,
            last_failure: now,
            blocked_until: None,
        });

        record.count = record.count.saturating_add(1);
        record.last_failure = now;

        // Trigger lockout if max failures exceeded
        if record.count >= self.max_failures {
            record.blocked_until = Some(now + self.max_delay);
        }
    }

    /// Record a successful authentication (resets failure count)
    pub fn record_success(&self, ip: IpAddr) {
        let mut failures = self.failures.lock();
        failures.remove(&ip);
    }

    /// Get failure count for an IP
    pub fn failure_count(&self, ip: IpAddr) -> u32 {
        let failures = self.failures.lock();
        failures.get(&ip).map(|r| r.count).unwrap_or(0)
    }

    /// Cleanup stale entries periodically
    fn maybe_cleanup(&self) {
        let mut last_cleanup = self.last_cleanup.lock();
        let now = Instant::now();

        if now.duration_since(*last_cleanup) >= self.cleanup_interval {
            *last_cleanup = now;
            drop(last_cleanup);

            let mut failures = self.failures.lock();
            // Remove entries that have been successful (no record) or
            // whose lockout has expired and enough time has passed
            failures.retain(|_, record| {
                // Keep if still blocked
                if let Some(blocked_until) = record.blocked_until {
                    if now < blocked_until {
                        return true;
                    }
                }
                // Keep if recent failure (within cleanup_interval)
                now.duration_since(record.last_failure) < self.cleanup_interval
            });
        }
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
        assert!(auth
            .validate("/api/test", Some("ApiKey secret-key-123"))
            .is_ok());

        // Invalid key
        assert!(auth
            .validate("/api/test", Some("ApiKey wrong-key"))
            .is_err());

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

    #[test]
    fn test_auth_rate_limiter_tracks_failures() {
        let limiter =
            AuthRateLimiter::new(3, Duration::from_millis(10), Duration::from_millis(100));
        let ip = "127.0.0.1".parse().unwrap();

        assert_eq!(limiter.failure_count(ip), 0);
        assert!(limiter.is_blocked(ip).is_none());

        limiter.record_failure(ip);
        assert_eq!(limiter.failure_count(ip), 1);

        limiter.record_failure(ip);
        assert_eq!(limiter.failure_count(ip), 2);
    }

    #[test]
    fn test_auth_rate_limiter_lockout() {
        let limiter = AuthRateLimiter::new(3, Duration::from_millis(10), Duration::from_millis(50));
        let ip = "127.0.0.1".parse().unwrap();

        // 3 failures should trigger lockout
        limiter.record_failure(ip);
        limiter.record_failure(ip);
        assert!(limiter.is_blocked(ip).is_none()); // Not blocked yet

        limiter.record_failure(ip);
        assert!(limiter.is_blocked(ip).is_some()); // Now blocked
    }

    #[test]
    fn test_auth_rate_limiter_success_resets() {
        let limiter =
            AuthRateLimiter::new(5, Duration::from_millis(10), Duration::from_millis(100));
        let ip = "127.0.0.1".parse().unwrap();

        limiter.record_failure(ip);
        limiter.record_failure(ip);
        assert_eq!(limiter.failure_count(ip), 2);

        limiter.record_success(ip);
        assert_eq!(limiter.failure_count(ip), 0);
    }

    #[test]
    fn test_auth_rate_limiter_per_ip() {
        let limiter =
            AuthRateLimiter::new(2, Duration::from_millis(10), Duration::from_millis(100));
        let ip1: IpAddr = "127.0.0.1".parse().unwrap();
        let ip2: IpAddr = "127.0.0.2".parse().unwrap();

        // Failures are tracked per-IP
        limiter.record_failure(ip1);
        limiter.record_failure(ip1);
        assert!(limiter.is_blocked(ip1).is_some());
        assert!(limiter.is_blocked(ip2).is_none()); // ip2 not affected
    }
}
