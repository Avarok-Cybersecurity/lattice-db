//! The Hyper transport hands the router the request's query string: an import
//! over real HTTP reaches the handler with its `mode`.
#![cfg(not(target_arch = "wasm32"))]

use lattice_server::hyper_transport::HyperTransport;
use lattice_server::router::{new_app_state, route};
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Send one HTTP/1.1 request and return (status, body).
async fn http(addr: SocketAddr, method: &str, target: &str, body: &[u8]) -> (u16, Vec<u8>) {
    let mut stream = TcpStream::connect(addr).await.expect("connect");
    let head = format!(
        "{method} {target} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await.expect("write head");
    stream.write_all(body).await.expect("write body");
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.expect("read");
    let split = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("header terminator");
    let status_line = String::from_utf8_lossy(&response[..split]).to_string();
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("status code");
    (status, response[split + 4..].to_vec())
}

#[tokio::test]
async fn import_mode_in_the_query_string_reaches_the_router() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let state = new_app_state();
    tokio::spawn(
        HyperTransport::new(addr.to_string()).serve_on(listener, move |request| {
            let state = state.clone();
            async move { route(state, request).await }
        }),
    );

    let create = br#"{"vectors":{"size":2,"distance":"Cosine"}}"#;
    assert_eq!(http(addr, "PUT", "/collections/src", create).await.0, 200);
    let points = br#"{"points":[{"id":1,"vector":[1.0,0.0],"payload":{"t":"a"}}]}"#;
    assert_eq!(
        http(addr, "PUT", "/collections/src/points", points).await.0,
        200
    );
    let (status, snapshot) = http(addr, "GET", "/collections/src/export", b"").await;
    assert_eq!(status, 200);

    let (status, body) = http(
        addr,
        "POST",
        "/collections/copy/import?mode=create",
        &snapshot,
    )
    .await;
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    let (status, body) = http(addr, "GET", "/collections/copy", b"").await;
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));

    // An invalid mode is now rejected for what it is, not as a missing one.
    let (status, body) = http(addr, "POST", "/collections/x/import?mode=bogus", &snapshot).await;
    assert_eq!(status, 400);
    assert!(!String::from_utf8_lossy(&body).contains("Missing required"));
}
