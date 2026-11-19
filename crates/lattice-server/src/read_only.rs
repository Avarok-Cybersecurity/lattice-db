//! Read-only serving mode.
//!
//! When [`crate::router::ServerConfig::read_only`] is set, the request router
//! answers only an explicit allow-list of read endpoints. Everything else —
//! every mutation, and any endpoint this list does not know — is refused with
//! 403 before it reaches a handler. The list is an allow-list on purpose: a
//! write endpoint added later is refused until someone classifies it, rather
//! than silently mutating a corpus that was published as read-only.
//!
//! Read-only governs the *request surface* (native HTTP, `route()` from JS,
//! the service-worker and Cloudflare-Worker hosts). The embedding host itself
//! still loads its corpus through the direct API (snapshot/JSONL import).

use crate::dto::CypherQueryRequest;
use lattice_core::cypher::{CypherParser, Statement};
use lattice_core::LatticeResponse;

/// HTTP status returned for a refused request.
pub const FORBIDDEN: u16 = 403;

/// Environment variable that switches the native server into read-only mode.
pub const READ_ONLY_ENV: &str = "LATTICE_READ_ONLY";

/// Parse the `LATTICE_READ_ONLY` value.
///
/// Unset means writable, matching every other opt-in `LATTICE_*` variable.
/// When set it must be exactly `true` or `false`: a typo such as `yes` or `1`
/// fails startup instead of being read as "writable".
pub fn parse_read_only_flag(value: Option<&str>) -> Result<bool, String> {
    match value {
        None => Ok(false),
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        Some(other) => Err(format!(
            "{READ_ONLY_ENV} must be 'true' or 'false', got '{other}'"
        )),
    }
}

/// Refuse a request in read-only mode, or return `None` to let it through.
///
/// `segments` are the non-empty path segments with the query string removed,
/// exactly as the router matches them.
pub fn deny_unless_read(method: &str, segments: &[&str], body: &[u8]) -> Option<LatticeResponse> {
    match (method, segments) {
        ("GET", ["ping"])
        | ("GET", ["collections"])
        | ("GET", ["collections", _])
        | ("GET", ["collections", _, "export"])
        | ("POST", ["collections", _, "points"])
        | ("POST", ["collections", _, "points", "search"])
        | ("POST", ["collections", _, "points", "search", "batch"])
        | ("POST", ["collections", _, "points", "query"])
        | ("POST", ["collections", _, "points", "scroll"])
        | ("POST", ["collections", _, "graph", "traverse"]) => None,
        ("POST", ["collections", _, "graph", "query"]) => deny_cypher_write(body),
        _ => Some(forbidden(&format!(
            "Read-only mode: {} /{} is not permitted",
            method,
            segments.join("/")
        ))),
    }
}

/// Allow a Cypher request only when it parses to a pure read query.
///
/// A body that is not valid JSON is passed through: the router rejects it with
/// 400 before anything executes. A query that does not parse is refused here
/// rather than deferred, so a parser disagreement can never become a write.
fn deny_cypher_write(body: &[u8]) -> Option<LatticeResponse> {
    let request: CypherQueryRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(_) => return None,
    };
    match CypherParser::new().parse(&request.query) {
        Ok(Statement::Query(_)) => None,
        Ok(_) => Some(forbidden(
            "Read-only mode: Cypher write clauses (CREATE, DELETE, SET) are not permitted",
        )),
        Err(_) => Some(forbidden(
            "Read-only mode: the Cypher query could not be verified as read-only",
        )),
    }
}

fn forbidden(message: &str) -> LatticeResponse {
    LatticeResponse::error(FORBIDDEN, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_parses_only_explicit_values() {
        assert_eq!(parse_read_only_flag(None), Ok(false));
        assert_eq!(parse_read_only_flag(Some("true")), Ok(true));
        assert_eq!(parse_read_only_flag(Some("false")), Ok(false));
        for bad in ["", "1", "yes", "TRUE", " true"] {
            assert!(parse_read_only_flag(Some(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn unparseable_cypher_is_refused_not_deferred() {
        let body = br#"{"query":"THIS IS NOT CYPHER"}"#;
        let denied = deny_unless_read("POST", &["collections", "c", "graph", "query"], body);
        assert_eq!(denied.map(|r| r.status), Some(FORBIDDEN));
    }

    #[test]
    fn invalid_json_is_left_to_the_router() {
        let denied = deny_unless_read("POST", &["collections", "c", "graph", "query"], b"{");
        assert!(denied.is_none());
    }
}
