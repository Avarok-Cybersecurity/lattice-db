//! The client is exercised against a real local HTTP server (a scripted
//! TCP responder), not a mock: production request building, transport and
//! response parsing all run.

use super::*;
use std::collections::HashMap;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

fn env(pairs: &[(&'static str, &str)]) -> impl Fn(&str) -> Option<String> {
    let map: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    move |k| map.get(k).cloned()
}

fn full_env(url: &str) -> Vec<(&'static str, String)> {
    vec![
        (ENV_URL, url.to_string()),
        (ENV_MODEL, "toy-embed".to_string()),
        (ENV_API_KEY, "sk-test".to_string()),
        (ENV_DIM, "2".to_string()),
        (ENV_TIMEOUT_MS, "5000".to_string()),
    ]
}

fn config_for(url: &str) -> OpenAiEmbedderConfig {
    let pairs = full_env(url);
    let borrowed: Vec<(&'static str, &str)> = pairs.iter().map(|(k, v)| (*k, v.as_str())).collect();
    OpenAiEmbedderConfig::from_lookup(env(&borrowed))
        .unwrap()
        .unwrap()
}

/// Serve exactly one HTTP response; return the raw request it received.
async fn serve_once(
    status: &'static str,
    body: &'static str,
) -> (String, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let n = socket.read(&mut buf).await.unwrap();
            request.extend_from_slice(&buf[..n]);
            let text = String::from_utf8_lossy(&request).to_string();
            if let Some(split) = text.find("\r\n\r\n") {
                let len = text[..split]
                    .lines()
                    .find_map(|l| {
                        l.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                if request.len() >= split + 4 + len {
                    break;
                }
            }
        }
        let response = format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        String::from_utf8_lossy(&request).to_string()
    });
    (url, handle)
}

#[test]
fn config_requires_all_or_nothing() {
    assert!(OpenAiEmbedderConfig::from_lookup(env(&[]))
        .unwrap()
        .is_none());
    let err = OpenAiEmbedderConfig::from_lookup(env(&[(ENV_MODEL, "m")])).unwrap_err();
    assert_eq!(
        err,
        EmbedderConfigError::Missing(vec![ENV_URL, ENV_API_KEY, ENV_DIM, ENV_TIMEOUT_MS])
    );
}

#[test]
fn config_rejects_bad_values_and_redacts_the_key() {
    let mut pairs = full_env("ftp://x");
    let check = |pairs: &Vec<(&'static str, String)>| {
        let b: Vec<(&'static str, &str)> = pairs.iter().map(|(k, v)| (*k, v.as_str())).collect();
        OpenAiEmbedderConfig::from_lookup(env(&b))
    };
    assert!(matches!(
        check(&pairs),
        Err(EmbedderConfigError::Invalid { var: ENV_URL, .. })
    ));
    pairs[0].1 = "https://api.example.com/v1/".into();
    pairs[3].1 = "0".into();
    assert!(matches!(
        check(&pairs),
        Err(EmbedderConfigError::Invalid { var: ENV_DIM, .. })
    ));
    pairs[3].1 = "2".into();
    pairs[1].1 = "bm25".into();
    assert!(matches!(
        check(&pairs),
        Err(EmbedderConfigError::Invalid { var: ENV_MODEL, .. })
    ));
    pairs[1].1 = "m".into();
    let cfg = check(&pairs).unwrap().unwrap();
    assert_eq!(cfg.base_url, "https://api.example.com/v1");
    assert!(!format!("{cfg:?}").contains("sk-test"));
}

#[tokio::test]
async fn embeds_in_input_order_with_bearer_auth() {
    // Items deliberately out of order: the client must sort by `index`.
    let (url, server) = serve_once(
        "200 OK",
        r#"{"data":[{"embedding":[0.0,1.0],"index":1},{"embedding":[1.0,0.0],"index":0}],"model":"toy-embed"}"#,
    )
    .await;
    let embedder = OpenAiEmbedder::new(config_for(&url)).unwrap();
    let vectors = embedder
        .embed(&["first".to_string(), "second".to_string()])
        .await
        .unwrap();
    assert_eq!(vectors, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    let request = server.await.unwrap();
    assert!(request.starts_with("POST /v1/embeddings "), "{request}");
    assert!(request
        .to_ascii_lowercase()
        .contains("authorization: bearer sk-test"));
    assert!(request.contains(r#""input":["first","second"]"#));
    assert!(request.contains(r#""model":"toy-embed""#));
}

#[tokio::test]
async fn upstream_errors_and_bad_shapes_fail_fast() {
    let (url, _s) = serve_once("401 Unauthorized", r#"{"error":"bad key"}"#).await;
    let err = OpenAiEmbedder::new(config_for(&url))
        .unwrap()
        .embed(&["x".to_string()])
        .await
        .unwrap_err();
    assert!(matches!(&err, EmbedError::Transport(m) if m.contains("401") && m.contains("bad key")));

    let (url, _s) = serve_once(
        "200 OK",
        r#"{"data":[{"embedding":[1.0,2.0,3.0],"index":0}]}"#,
    )
    .await;
    let err = OpenAiEmbedder::new(config_for(&url))
        .unwrap()
        .embed(&["x".to_string()])
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        EmbedError::DimensionMismatch {
            expected: 2,
            actual: 3,
            ..
        }
    ));

    let (url, _s) = serve_once("200 OK", r#"{"data":[{"embedding":[1.0,2.0],"index":3}]}"#).await;
    let err = OpenAiEmbedder::new(config_for(&url))
        .unwrap()
        .embed(&["x".to_string()])
        .await
        .unwrap_err();
    assert!(matches!(err, EmbedError::InvalidResponse(_)));
}
