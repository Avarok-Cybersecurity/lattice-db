# Security & Code Quality Recommendations

Findings from repository audit. Items marked with checkboxes are pending.

## Critical (P0) - Security

- [x] **Authentication** - Added API key and Bearer token authentication
  - Location: `auth.rs`, `hyper_transport.rs`
  - Schemes: `ApiKey <key>` and `Bearer <token>` headers
  - Environment: `LATTICE_API_KEYS` and `LATTICE_BEARER_TOKENS` (comma-separated)
  - Public paths: `/`, `/health`, `/healthz`, `/ready`, `/readyz` (configurable)
  - Returns 401 Unauthorized with `WWW-Authenticate` header on failure

- [x] **TLS/HTTPS** - Added rustls-based TLS support
  - Location: `tls.rs`, `hyper_transport.rs` (HyperTlsTransport)
  - Feature: Enable with `--features tls`
  - Implementation: Certificate/key loading from PEM files
  - Usage: `HyperTlsTransport::new(addr, TlsConfig::from_pem_files(cert, key)?)`
  - Environment: `LATTICE_TLS_CERT` and `LATTICE_TLS_KEY` paths

- [x] **Rate Limiting** - Added token bucket rate limiter
  - Location: `rate_limit.rs`, `hyper_transport.rs`
  - Implementation: Per-IP token bucket with configurable rate/burst
  - Default: 100 req/s, burst 200 (production config)
  - Headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
  - Optional: `HyperTransport::with_rate_limit()` or `with_default_rate_limit()`

## Medium (P1)

- [x] **Unbounded batch sizes** - Added configurable limits
  - Added `ServerConfig` with explicit batch size limits (PCND compliant)
  - `max_upsert_batch_size`: 10,000 points per request
  - `max_delete_batch_size`: 10,000 IDs per request
  - `max_get_batch_size`: 10,000 IDs per request
  - `max_search_batch_size`: 100 queries per batch request
  - Validation in handlers returns 400 Bad Request if exceeded

- [x] **Minimal logging** - Added structured logging
  - Added `tracing` and `tracing-subscriber` dependencies
  - Router instrumented with `#[instrument]` for request tracing
  - Collection create/delete operations logged with context
  - Point upsert/delete operations logged with counts
  - Request failures logged at warn level
  - Configure via `RUST_LOG` env var (default: info)

## Code Quality

- [x] **`unwrap()` calls in lattice-core** - Replaced with proper error handling
  - Added `LockExt` trait for safe RwLock acquisition (error codes 50001-50003)
  - Added `MutexExt` trait for safe Mutex acquisition
  - Added `cmp_f32()` for NaN-safe float comparison
  - Replaced `unwrap()` in `collection.rs`, `async_indexer.rs`, `executor.rs`, `planner.rs`
  - Methods now return `LatticeResult<T>` for proper error propagation

- [x] **`panic!()` and `unreachable!()` calls in cypher/**
  - Replaced `unreachable!()` in `parser.rs` with `CypherError::internal()`
  - Remaining `panic!()` calls are in test code only (acceptable)

- [x] **89 `clone()` calls in hot paths** - Optimized critical paths
  - `executor.rs`: DISTINCT now uses index collection + swap_remove (avoids row cloning)
  - `executor.rs`: Expand multi-path already optimized (clone all but last)
  - `collection.rs`: Graph traversal paths documented (requires Arc for further optimization)
  - Remaining clones are in expression evaluation (would require API changes to fix)

- [x] **clippy allows in `lib.rs`** - Reviewed
  - 16 allows present (style preferences and compatibility)
  - Verified `dead_code`, `never_loop`, `if_same_then_else` don't hide issues
  - Remaining allows are intentional for code clarity and Rust version compatibility

## CI/CD Improvements

- [x] **Performance regression tests**
  - Added `ci_regression.rs` benchmark suite
  - Tests: vector search, vector upsert, Cypher parsing, Cypher execution
  - CI job: Runs on all PRs, caches baseline on main branch
  - Self-contained (no external services required)

- [x] **cargo-deny** for license and security checks
  - Added `deny.toml` with license allowlist
  - Added to CI security job
  - Checks: advisories, licenses, bans, sources

## Completed

- [x] **Lock unwrap() panics** - Replaced `std::sync::RwLock` with `parking_lot::RwLock`
- [x] **Codecov** - Added cargo-tarpaulin coverage reporting
- [x] **cargo audit** - Added security vulnerability scanning to CI
- [x] **Collection name validation** - Added path traversal protection
- [x] **Request size limits** - Added configurable body size limits
