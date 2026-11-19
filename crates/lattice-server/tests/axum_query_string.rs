//! The Axum transport hands the router the request's query string: an import
//! through a mounted router reaches the handler with its `mode`.
#![cfg(all(feature = "axum-transport", not(target_arch = "wasm32")))]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use lattice_server::axum_transport::attach_to;
use lattice_server::router::new_app_state;
use tower::ServiceExt;

async fn call(app: &Router, method: &str, uri: &str, body: Vec<u8>) -> (StatusCode, Vec<u8>) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .body(Body::from(body))
        .expect("request");
    let response = app.clone().oneshot(request).await.expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, bytes.to_vec())
}

#[tokio::test]
async fn import_mode_in_the_query_string_reaches_the_router() {
    let app = attach_to(Router::new(), new_app_state());

    let create = br#"{"vectors":{"size":2,"distance":"Cosine"}}"#.to_vec();
    assert_eq!(
        call(&app, "PUT", "/collections/src", create).await.0,
        StatusCode::OK
    );
    let points = br#"{"points":[{"id":1,"vector":[1.0,0.0],"payload":{"t":"a"}}]}"#.to_vec();
    let (status, _) = call(&app, "PUT", "/collections/src/points", points).await;
    assert_eq!(status, StatusCode::OK);
    let (status, snapshot) = call(&app, "GET", "/collections/src/export", Vec::new()).await;
    assert_eq!(status, StatusCode::OK);

    let uri = "/collections/copy/import?mode=create";
    let (status, body) = call(&app, "POST", uri, snapshot.clone()).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let (status, _) = call(&app, "GET", "/collections/copy", Vec::new()).await;
    assert_eq!(status, StatusCode::OK);

    let uri = "/collections/x/import?mode=bogus";
    let (status, body) = call(&app, "POST", uri, snapshot).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(!String::from_utf8_lossy(&body).contains("Missing required"));
}
