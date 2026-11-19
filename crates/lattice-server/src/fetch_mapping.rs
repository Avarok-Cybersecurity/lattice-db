//! Pure request/response mapping for fetch-event hosts (service worker,
//! Cloudflare Worker): prefix matching and response headers.
//!
//! Kept free of `web-sys` so it is unit-tested natively; the WASM glue in
//! `service_worker.rs` only moves bytes between this and the platform types.

/// Methods the REST surface answers, advertised to CORS preflights.
const ALLOWED_METHODS: &str = "GET, POST, PUT, DELETE, OPTIONS";
const ALLOWED_HEADERS: &str = "Content-Type, Authorization";
const PREFLIGHT_MAX_AGE_SECS: &str = "600";

/// Where a fetch host mounts the API and whom it lets read it cross-origin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchConfig {
    prefix: String,
    cors_origin: Option<String>,
}

impl FetchConfig {
    /// `prefix` is `""` (serve every path) or starts with `/` and has no
    /// trailing `/`, e.g. `/api/lattice`. `cors_origin` is the exact value for
    /// `Access-Control-Allow-Origin` (`"*"` or an origin); `None` sends no CORS
    /// headers, which is right for a same-origin service worker.
    pub fn new(prefix: &str, cors_origin: Option<String>) -> Result<Self, String> {
        let valid = prefix.is_empty() || (prefix.starts_with('/') && !prefix.ends_with('/'));
        if !valid || prefix.contains(['?', '#']) {
            return Err(format!(
                "Invalid prefix {:?}: use \"\" or a path like \"/api/lattice\" without a trailing slash",
                prefix
            ));
        }
        if let Some(origin) = &cors_origin {
            if origin.is_empty() || origin.contains(['\r', '\n', ',']) {
                return Err(format!("Invalid CORS origin {:?}", origin));
            }
        }
        Ok(Self {
            prefix: prefix.to_string(),
            cors_origin,
        })
    }

    /// The configured mount prefix.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Router path for a URL pathname under the prefix, or `None` when the
    /// pathname is outside it. Matches on a segment boundary, so `/api/lattice`
    /// does not claim `/api/latticex`.
    pub fn router_path(&self, pathname: &str) -> Option<String> {
        let rest = pathname.strip_prefix(self.prefix.as_str())?;
        if rest.is_empty() {
            Some("/".to_string())
        } else if rest.starts_with('/') {
            Some(rest.to_string())
        } else {
            None
        }
    }

    /// Whether `OPTIONS` should be answered as a CORS preflight.
    pub fn answers_preflight(&self) -> bool {
        self.cors_origin.is_some()
    }

    /// Headers for a router response: the router's own, the security headers
    /// the native server sends, and CORS when configured.
    pub fn response_headers<'a, I>(&self, router_headers: I) -> Vec<(String, String)>
    where
        I: IntoIterator<Item = (&'a String, &'a String)>,
    {
        let mut out: Vec<(String, String)> = router_headers
            .into_iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if !out
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("content-type"))
        {
            out.push(("Content-Type".into(), "application/json".into()));
        }
        out.push(("X-Content-Type-Options".into(), "nosniff".into()));
        out.push(("Cache-Control".into(), "no-store".into()));
        out.extend(self.cors_headers());
        out
    }

    /// Headers for a `204` preflight answer.
    pub fn preflight_headers(&self) -> Vec<(String, String)> {
        let mut out = self.cors_headers();
        out.push((
            "Access-Control-Allow-Methods".into(),
            ALLOWED_METHODS.into(),
        ));
        out.push((
            "Access-Control-Allow-Headers".into(),
            ALLOWED_HEADERS.into(),
        ));
        out.push((
            "Access-Control-Max-Age".into(),
            PREFLIGHT_MAX_AGE_SECS.into(),
        ));
        out
    }

    fn cors_headers(&self) -> Vec<(String, String)> {
        match &self.cors_origin {
            None => Vec::new(),
            Some(origin) => {
                let mut out = vec![("Access-Control-Allow-Origin".into(), origin.clone())];
                if origin != "*" {
                    out.push(("Vary".into(), "Origin".into()));
                }
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn prefix_validation() {
        assert!(FetchConfig::new("", None).is_ok());
        assert!(FetchConfig::new("/api/lattice", None).is_ok());
        for bad in ["/", "api", "/api/", "/a?b", "/a#b"] {
            assert!(FetchConfig::new(bad, None).is_err(), "{bad}");
        }
        assert!(FetchConfig::new("/a", Some("".into())).is_err());
        assert!(FetchConfig::new("/a", Some("x\r\ny".into())).is_err());
    }

    #[test]
    fn router_path_matches_on_segment_boundary() {
        let c = FetchConfig::new("/api/lattice", None).expect("valid");
        assert_eq!(c.router_path("/api/lattice").as_deref(), Some("/"));
        assert_eq!(
            c.router_path("/api/lattice/collections").as_deref(),
            Some("/collections")
        );
        assert_eq!(c.router_path("/api/latticex/collections"), None);
        assert_eq!(c.router_path("/other"), None);
        let root = FetchConfig::new("", None).expect("valid");
        assert_eq!(root.router_path("/ping").as_deref(), Some("/ping"));
    }

    #[test]
    fn headers_carry_security_and_optional_cors() {
        let mut h = HashMap::new();
        h.insert(
            "Content-Type".to_string(),
            "application/octet-stream".to_string(),
        );
        let plain = FetchConfig::new("/x", None).expect("valid");
        let out = plain.response_headers(&h);
        assert!(out.contains(&("Content-Type".into(), "application/octet-stream".into())));
        assert!(out.iter().all(|(k, _)| !k.starts_with("Access-Control")));
        assert!(!plain.answers_preflight());

        let cors = FetchConfig::new("/x", Some("https://a.example".into())).expect("valid");
        let out = cors.response_headers(&HashMap::new());
        assert!(out.contains(&("Content-Type".into(), "application/json".into())));
        assert!(out.contains(&(
            "Access-Control-Allow-Origin".into(),
            "https://a.example".into()
        )));
        assert!(out.contains(&("Vary".into(), "Origin".into())));
        assert!(cors
            .preflight_headers()
            .iter()
            .any(|(k, _)| k == "Access-Control-Allow-Methods"));
    }
}
