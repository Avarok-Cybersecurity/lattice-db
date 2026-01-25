//! Rate limiting for DoS protection
//!
//! Implements a token bucket rate limiter that tracks requests per client IP.
//! This provides enterprise-grade protection against denial of service attacks.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::net::IpAddr;
use std::time::{Duration, Instant};

/// Rate limiter configuration (PCND: all fields explicit)
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per second per IP
    pub requests_per_second: u32,
    /// Burst capacity (max tokens that can accumulate)
    pub burst_size: u32,
    /// Cleanup interval for stale entries
    pub cleanup_interval: Duration,
    /// Maximum number of IP buckets to track (DoS protection)
    /// When exceeded, oldest entries are evicted
    pub max_buckets: usize,
}

impl RateLimitConfig {
    /// Production defaults - 100 req/s with burst of 200
    ///
    /// max_buckets set to 100,000 to limit memory usage (~10MB at ~100 bytes/bucket)
    /// while supporting legitimate high-cardinality client pools
    pub fn production() -> Self {
        Self {
            requests_per_second: 100,
            burst_size: 200,
            cleanup_interval: Duration::from_secs(60),
            max_buckets: 100_000,
        }
    }

    /// Disabled rate limiting (for testing)
    #[cfg(test)]
    pub fn disabled() -> Self {
        Self {
            requests_per_second: u32::MAX,
            burst_size: u32::MAX,
            cleanup_interval: Duration::from_secs(3600),
            max_buckets: usize::MAX,
        }
    }
}

/// Token bucket for a single client
struct TokenBucket {
    /// Available tokens
    tokens: f64,
    /// Last update time
    last_update: Instant,
}

impl TokenBucket {
    fn new(burst_size: u32) -> Self {
        Self {
            tokens: burst_size as f64,
            last_update: Instant::now(),
        }
    }

    /// Try to consume a token, returns true if allowed
    fn try_consume(&mut self, rate: f64, burst: f64) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        self.last_update = now;

        // Refill tokens based on elapsed time
        self.tokens = (self.tokens + elapsed * rate).min(burst);

        // Try to consume
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Global rate limiter tracking all clients
pub struct RateLimiter {
    config: RateLimitConfig,
    /// Per-IP token buckets
    buckets: Mutex<HashMap<IpAddr, TokenBucket>>,
    /// Last cleanup time
    last_cleanup: Mutex<Instant>,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            buckets: Mutex::new(HashMap::new()),
            last_cleanup: Mutex::new(Instant::now()),
        }
    }

    /// Check if a request from the given IP should be allowed
    ///
    /// Returns `true` if allowed, `false` if rate limited.
    pub fn check(&self, ip: IpAddr) -> bool {
        // Periodic cleanup of stale entries
        self.maybe_cleanup();

        let rate = self.config.requests_per_second as f64;
        let burst = self.config.burst_size as f64;

        let mut buckets = self.buckets.lock();

        // Evict oldest entry if at max capacity and this is a new IP
        if buckets.len() >= self.config.max_buckets && !buckets.contains_key(&ip) {
            // Find and remove the oldest bucket (LRU eviction)
            if let Some(oldest_ip) = buckets
                .iter()
                .min_by_key(|(_, bucket)| bucket.last_update)
                .map(|(ip, _)| *ip)
            {
                buckets.remove(&oldest_ip);
            }
        }

        let bucket = buckets
            .entry(ip)
            .or_insert_with(|| TokenBucket::new(self.config.burst_size));

        bucket.try_consume(rate, burst)
    }

    /// Get remaining tokens for an IP (for headers)
    pub fn remaining(&self, ip: IpAddr) -> u32 {
        let buckets = self.buckets.lock();
        buckets
            .get(&ip)
            .map(|b| b.tokens as u32)
            .unwrap_or(self.config.burst_size)
    }

    /// Cleanup stale entries periodically
    fn maybe_cleanup(&self) {
        let mut last_cleanup = self.last_cleanup.lock();
        let now = Instant::now();

        if now.duration_since(*last_cleanup) >= self.config.cleanup_interval {
            *last_cleanup = now;
            drop(last_cleanup); // Release lock before acquiring buckets lock

            let mut buckets = self.buckets.lock();
            // Remove entries that haven't been used recently
            // A bucket is stale if it's been full for cleanup_interval
            let stale_threshold = self.config.cleanup_interval;
            buckets.retain(|_, bucket| now.duration_since(bucket.last_update) < stale_threshold);
        }
    }

    /// Get rate limit headers for response
    pub fn headers(&self, ip: IpAddr) -> Vec<(&'static str, String)> {
        let remaining = self.remaining(ip);
        vec![
            (
                "X-RateLimit-Limit",
                self.config.requests_per_second.to_string(),
            ),
            ("X-RateLimit-Remaining", remaining.to_string()),
            ("X-RateLimit-Reset", "1".to_string()), // Simplified: 1 second window
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn test_rate_limiter_allows_burst() {
        let config = RateLimitConfig {
            requests_per_second: 10,
            burst_size: 5,
            cleanup_interval: Duration::from_secs(60),
            max_buckets: 1000,
        };
        let limiter = RateLimiter::new(config);
        let ip = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));

        // Should allow burst_size requests immediately
        for _ in 0..5 {
            assert!(limiter.check(ip), "Should allow burst");
        }

        // Should deny after burst exhausted
        assert!(!limiter.check(ip), "Should deny after burst");
    }

    #[test]
    fn test_rate_limiter_refills() {
        let config = RateLimitConfig {
            requests_per_second: 1000, // High rate for fast refill
            burst_size: 1,
            cleanup_interval: Duration::from_secs(60),
            max_buckets: 1000,
        };
        let limiter = RateLimiter::new(config);
        let ip = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));

        // Exhaust the bucket
        assert!(limiter.check(ip));
        assert!(!limiter.check(ip));

        // Wait a bit for refill (at 1000/s, 1ms = 1 token)
        std::thread::sleep(Duration::from_millis(2));

        // Should be allowed again
        assert!(limiter.check(ip));
    }

    #[test]
    fn test_rate_limiter_per_ip() {
        let config = RateLimitConfig {
            requests_per_second: 10,
            burst_size: 1,
            cleanup_interval: Duration::from_secs(60),
            max_buckets: 1000,
        };
        let limiter = RateLimiter::new(config);
        let ip1 = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
        let ip2 = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2));

        // Each IP gets its own bucket
        assert!(limiter.check(ip1));
        assert!(limiter.check(ip2));

        // Both should be rate limited now
        assert!(!limiter.check(ip1));
        assert!(!limiter.check(ip2));
    }

    #[test]
    fn test_rate_limiter_max_buckets_eviction() {
        let config = RateLimitConfig {
            requests_per_second: 10,
            burst_size: 5,
            cleanup_interval: Duration::from_secs(60),
            max_buckets: 2, // Very small for testing
        };
        let limiter = RateLimiter::new(config);
        let ip1 = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
        let ip2 = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2));
        let ip3 = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 3));

        // Add first two IPs
        limiter.check(ip1);
        std::thread::sleep(Duration::from_millis(1)); // Ensure different timestamps
        limiter.check(ip2);

        // Verify both are tracked
        {
            let buckets = limiter.buckets.lock();
            assert_eq!(buckets.len(), 2);
        }

        // Add third IP - should evict oldest (ip1)
        std::thread::sleep(Duration::from_millis(1));
        limiter.check(ip3);

        // Verify only 2 buckets and ip1 was evicted
        {
            let buckets = limiter.buckets.lock();
            assert_eq!(buckets.len(), 2);
            assert!(!buckets.contains_key(&ip1), "ip1 should be evicted");
            assert!(buckets.contains_key(&ip2), "ip2 should remain");
            assert!(buckets.contains_key(&ip3), "ip3 should be added");
        }
    }
}
