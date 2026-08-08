//! Axum HTTP transport implementation
//!
//! Provides HTTP server functionality using the Axum web framework.
//!
//! Entry points:
//!
//! * [`AxumTransport`] — runs LatticeDB as a standalone server.
//! * [`attach_to`] — serves the LatticeDB API from an application that already
//!   has its own Axum router, keeping LatticeDB's canonical paths.
//! * [`routes`] — the API as a standalone [`Router`], for mounting under a
//!   prefix with [`Router::nest`].

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    routing::any,
    Router,
};
use lattice_core::{LatticeRequest, LatticeResponse, LatticeTransport};
use std::future::Future;
use std::sync::Arc;
use thiserror::Error;
use tokio::net::TcpListener;

use crate::router::{route, AppState};

#[cfg(feature = "openapi")]
use utoipa_swagger_ui::SwaggerUi;

/// Axum transport error
#[derive(Debug, Error)]
pub enum AxumError {
    #[error("Server error: {0}")]
    Server(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Axum HTTP transport
///
/// Wraps Axum to implement the LatticeTransport trait.
pub struct AxumTransport {
    addr: String,
}

impl AxumTransport {
    /// Create a new Axum transport
    ///
    /// # Arguments
    /// * `addr` - Address to bind to (e.g., "0.0.0.0:6333")
    pub fn new(addr: impl Into<String>) -> Self {
        Self { addr: addr.into() }
    }
}

#[async_trait]
impl LatticeTransport for AxumTransport {
    type Error = AxumError;

    async fn serve<H, Fut>(self, handler: H) -> Result<(), Self::Error>
    where
        H: Fn(LatticeRequest) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = LatticeResponse> + Send + 'static,
    {
        // Wrap handler in Arc for sharing
        let handler = Arc::new(handler);

        // Build API router - catch all routes and methods
        let api_router = Router::new()
            .route("/{*path}", any(handle_request::<H, Fut>))
            .route("/", any(handle_request::<H, Fut>))
            .with_state(handler);

        // Merge with OpenAPI/SwaggerUI routes if feature enabled
        #[cfg(feature = "openapi")]
        let app = {
            use crate::openapi::openapi_spec;
            Router::new()
                .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", openapi_spec()))
                .merge(api_router)
        };

        #[cfg(not(feature = "openapi"))]
        let app = api_router;

        // Bind and serve
        let listener = TcpListener::bind(&self.addr).await?;
        println!("LatticeDB server listening on {}", self.addr);

        #[cfg(feature = "openapi")]
        println!("OpenAPI docs available at http://{}/docs", self.addr);

        axum::serve(listener, app)
            .await
            .map_err(|e| AxumError::Server(e.to_string()))
    }
}

/// Build an Axum [`Router`] serving the complete LatticeDB API.
///
/// The router forwards *every* path to [`crate::router::route`], which is the
/// single source of truth for the Qdrant- and Neo4j/Cypher-compatible surface.
/// Nothing here enumerates individual endpoints, so routes added to the
/// dispatcher are served automatically and this layer can never drift out of
/// sync with it.
///
/// Because it matches every path, mount it one of two ways:
///
/// * [`attach_to`] — serve the API at its canonical paths inside an existing
///   application (recommended; your own routes keep priority).
/// * [`Router::nest`] — serve it under a prefix, e.g.
///   `Router::nest("/vectordb", routes(state))`. Axum strips the prefix before
///   the request reaches LatticeDB.
///
/// Merging this router at the root with [`Router::merge`] is **not**
/// supported — it would shadow the host application's own routes. Use
/// [`attach_to`] instead.
///
/// To share one database between LatticeDB's API and your own handlers, clone
/// the [`AppState`] — it is an `Arc` — and keep a copy for your code.
pub fn routes(state: AppState) -> Router {
    Router::new()
        .route("/", any(dispatch))
        .route("/{*path}", any(dispatch))
        .with_state(state)
}

/// Serve the LatticeDB API from an existing Axum application, at LatticeDB's
/// canonical paths.
///
/// The host application's own routes are matched first; anything they don't
/// handle is passed to LatticeDB. That keeps `GET /collections`,
/// `POST /collections/{name}/graph/query`, and every other endpoint working
/// exactly as they do in the standalone server — including endpoints added in
/// future versions, since no path list is involved.
///
/// # Example
///
/// ```no_run
/// use axum::{routing::get, Router};
/// use lattice_server::{axum_transport, router::new_app_state};
///
/// # async fn example() {
/// let app = Router::new()
///     .route("/health", get(|| async { "ok" }))
///     .route("/api/users", get(|| async { "users" }));
///
/// // Qdrant + Neo4j/Cypher compatible API at its canonical paths
/// let app = axum_transport::attach_to(app, new_app_state());
///
/// let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
/// axum::serve(listener, app).await.unwrap();
/// # }
/// ```
///
/// `GET /health` is still served by the host application, while
/// `GET /collections` is served by LatticeDB.
///
/// # Trade-off
///
/// This installs LatticeDB as the application's fallback, so it replaces any
/// fallback already set, and unmatched paths return LatticeDB's JSON 404
/// rather than the host's. Use [`Router::nest`] with [`routes`] if you need to
/// keep your own fallback.
pub fn attach_to(app: Router, state: AppState) -> Router {
    app.fallback_service(routes(state))
}

/// Axum handler that dispatches straight into the LatticeDB router.
async fn dispatch(State(state): State<AppState>, request: Request<Body>) -> Response {
    let lattice_request = match to_lattice_request(request).await {
        Ok(req) => req,
        Err(response) => return response,
    };

    to_axum_response(route(state, lattice_request).await)
}

/// Handle incoming HTTP requests
async fn handle_request<H, Fut>(
    State(handler): State<Arc<H>>,
    request: Request<Body>,
) -> impl IntoResponse
where
    H: Fn(LatticeRequest) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = LatticeResponse> + Send,
{
    let lattice_request = match to_lattice_request(request).await {
        Ok(req) => req,
        Err(response) => return response,
    };

    to_axum_response(handler(lattice_request).await)
}

/// Convert an Axum request into a `LatticeRequest`.
///
/// Returns the error response to send if the body could not be read.
///
/// Note: when this router is mounted with [`Router::nest`], Axum strips the
/// mount prefix before the request reaches here, so LatticeDB always sees
/// paths relative to its mount point.
async fn to_lattice_request(request: Request<Body>) -> Result<LatticeRequest, Response> {
    // Extract method and path - HTTP methods are already uppercase
    let method = request.method().to_string();
    let path = request.uri().path().to_owned();

    // Skip header parsing entirely - we don't use headers in the API
    // This saves ~20-40µs of allocations per request
    let headers = std::collections::HashMap::new();

    // Read body directly into Vec<u8> without extra copy
    let body = match axum::body::to_bytes(request.into_body(), 10 * 1024 * 1024).await {
        Ok(bytes) => bytes.into(),
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("Failed to read body: {}", e),
            )
                .into_response());
        }
    };

    Ok(LatticeRequest {
        method,
        path,
        body,
        headers,
    })
}

/// Convert a `LatticeResponse` into an Axum response, applying security headers.
fn to_axum_response(response: LatticeResponse) -> Response {
    let status = StatusCode::from_u16(response.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    // Build response with security headers
    let mut builder = axum::http::Response::builder()
        .status(status)
        // Security headers to prevent common web vulnerabilities
        .header("X-Content-Type-Options", "nosniff")
        .header("X-Frame-Options", "DENY")
        .header("Cache-Control", "no-store")
        .header("Content-Security-Policy", "default-src 'none'");

    // Add custom response headers
    for (key, value) in response.headers {
        builder = builder.header(key, value);
    }

    builder
        .body(Body::from(response.body))
        .unwrap()
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::new_app_state;
    use axum::routing::get;
    use tower::ServiceExt; // for `oneshot`

    #[test]
    fn test_axum_transport_new() {
        let transport = AxumTransport::new("127.0.0.1:6333");
        assert_eq!(transport.addr, "127.0.0.1:6333");
    }

    /// Host application with its own routes plus the LatticeDB API attached at
    /// LatticeDB's canonical paths.
    fn host_app() -> Router {
        let app = Router::new().route("/health", get(|| async { "host-ok" }));
        attach_to(app, new_app_state())
    }

    async fn body_string(response: Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn host_routes_still_work_when_lattice_is_mounted() {
        let response = host_app()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_string(response).await, "host-ok");
    }

    #[tokio::test]
    async fn attached_routes_keep_their_canonical_paths() {
        let app = host_app();

        // Create a collection at the canonical (un-prefixed) Qdrant path
        let create = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/collections/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"vectors":{"size":4,"distance":"Cosine"}}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            create.status(),
            StatusCode::OK,
            "collection creation failed"
        );

        // ...and read it back, proving state persists across requests
        let list = app
            .oneshot(
                Request::builder()
                    .uri("/collections")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(list.status(), StatusCode::OK);
        assert!(
            body_string(list).await.contains("embeddings"),
            "collection created via the merged router should be listed"
        );
    }

    #[tokio::test]
    async fn host_routes_take_priority_over_lattice() {
        // The host app defines /health; attaching LatticeDB must not steal it.
        let response = host_app()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(body_string(response).await, "host-ok");
    }

    #[tokio::test]
    async fn unknown_paths_get_lattice_not_found() {
        let response = host_app()
            .oneshot(
                Request::builder()
                    .uri("/not-a-lattice-route")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn nesting_under_a_prefix_is_still_supported() {
        let app = Router::new().nest("/vectordb", routes(new_app_state()));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/vectordb/collections")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    /// Send a request through the attached app and assert it succeeded.
    async fn call(app: &Router, method: &str, uri: &str, body: &'static str) -> String {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK, "{method} {uri} failed");
        body_string(response).await
    }

    /// Exercises the Qdrant- and Neo4j/Cypher-compatible surface through an
    /// attached router at canonical paths.
    ///
    /// This is deliberately behavioural rather than a list of registered
    /// paths: `attach_to` forwards every path to `router::route`, so any
    /// endpoint that dispatcher gains is served automatically and this layer
    /// cannot drift out of sync with it.
    #[tokio::test]
    async fn qdrant_and_cypher_surfaces_work_through_an_attached_router() {
        let app = host_app();

        // --- Qdrant-compatible ---
        call(&app, "GET", "/ping", "").await;
        call(
            &app,
            "PUT",
            "/collections/graph",
            r#"{"vectors":{"size":2,"distance":"Cosine"}}"#,
        )
        .await;
        call(
            &app,
            "PUT",
            "/collections/graph/points",
            r#"{"points":[
                {"id":1,"vector":[1.0,0.0],"payload":{"_labels":["Person"],"name":"ada"}},
                {"id":2,"vector":[0.0,1.0],"payload":{"_labels":["Person"],"name":"grace"}}
            ]}"#,
        )
        .await;

        let search = call(
            &app,
            "POST",
            "/collections/graph/points/search",
            r#"{"vector":[1.0,0.0],"limit":1}"#,
        )
        .await;
        assert!(search.contains("ada"), "vector search returned: {search}");

        call(
            &app,
            "POST",
            "/collections/graph/points/scroll",
            r#"{"limit":10}"#,
        )
        .await;

        // --- Graph / Neo4j-compatible ---
        call(
            &app,
            "POST",
            "/collections/graph/graph/edges",
            r#"{"from_id":1,"to_id":2,"relation":"KNOWS","weight":1.0}"#,
        )
        .await;

        let traverse = call(
            &app,
            "POST",
            "/collections/graph/graph/traverse",
            r#"{"start_id":1,"max_depth":2}"#,
        )
        .await;
        assert!(traverse.contains('2'), "traverse returned: {traverse}");

        let cypher = call(
            &app,
            "POST",
            "/collections/graph/graph/query",
            r#"{"query":"MATCH (n:Person) RETURN n.name"}"#,
        )
        .await;
        assert!(
            cypher.contains("ada") && cypher.contains("grace"),
            "cypher query returned: {cypher}"
        );
    }

    #[tokio::test]
    async fn mounted_responses_carry_security_headers() {
        let response = host_app()
            .oneshot(
                Request::builder()
                    .uri("/collections")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.headers()["X-Content-Type-Options"], "nosniff");
        assert_eq!(response.headers()["X-Frame-Options"], "DENY");
    }
}
