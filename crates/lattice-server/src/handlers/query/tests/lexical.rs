//! Lexical, raw-vector, embedded-text and error forms.

use super::{call, ids, message, register_toy, seeded, toy_vector};
use serde_json::json;

macro_rules! async_test {
    ($name:ident, $body:expr) => {
        #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
        async fn $name() {
            $body
        }
    };
}

const DOCS: &[(u64, &str)] = &[
    (1, "install the github action workflow"),
    (2, "run the database in a service worker"),
    (3, "deploy on a cloudflare worker free plan"),
];

const PATH: &str = "/collections/docs/points/query";

async_test!(bm25_text_query_ranks_and_returns_payload, {
    let state = seeded(None, DOCS).await;
    let (status, body) = call(
        &state,
        "POST",
        PATH,
        json!({
            "query": {"text": "cloudflare worker", "model": "bm25"}, "limit": 2
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["status"], "ok");
    assert_eq!(ids(&body), vec![3, 2]);
    let points = body["result"]["points"].as_array().unwrap();
    assert!(points[0]["score"].as_f64().unwrap() > points[1]["score"].as_f64().unwrap());
    assert_eq!(points[0]["payload"]["text"], DOCS[2].1);
});

async_test!(qdrant_bm25_alias_and_nearest_wrapper, {
    let state = seeded(None, DOCS).await;
    let (status, body) = call(
        &state,
        "POST",
        PATH,
        json!({
            "query": {"nearest": {"text": "github action", "model": "qdrant/bm25"}},
            "using": "bm25_sparse_vector", "with_payload": false
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(ids(&body)[0], 1);
    assert!(body["result"]["points"][0].get("payload").is_none());
});

async_test!(raw_vector_query_keeps_working, {
    let state = seeded(None, DOCS).await;
    let (status, body) = call(
        &state,
        "POST",
        PATH,
        json!({
            "query": toy_vector(DOCS[1].1), "limit": 1
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(ids(&body), vec![2]);
});

async_test!(text_query_embedded_by_the_collection_model, {
    let state = seeded(Some("toy"), DOCS).await;
    register_toy(&state, "toy", 3);
    let (status, body) = call(
        &state,
        "POST",
        PATH,
        json!({
            "query": {"text": DOCS[0].1, "model": "toy"}, "limit": 1
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(ids(&body), vec![1], "identical text embeds identically");
});

async_test!(a_different_embedding_model_is_rejected, {
    let state = seeded(Some("toy"), DOCS).await;
    register_toy(&state, "other", 3);
    let (status, body) = call(
        &state,
        "POST",
        PATH,
        json!({
            "query": {"text": "worker", "model": "other"}
        }),
    )
    .await;
    assert_eq!(status, 400);
    assert!(
        message(&body).contains("'toy'") && message(&body).contains("'other'"),
        "{body}"
    );

    let state = seeded(None, DOCS).await;
    let (status, body) = call(
        &state,
        "POST",
        PATH,
        json!({
            "query": {"text": "worker", "model": "toy"}
        }),
    )
    .await;
    assert_eq!(status, 400);
    assert!(message(&body).contains("no embedding model"), "{body}");
});

async_test!(missing_or_mismatched_embedder_is_unavailable, {
    let state = seeded(Some("toy"), DOCS).await;
    let query = json!({"query": {"text": "worker", "model": "toy"}});
    let (status, body) = call(&state, "POST", PATH, query.clone()).await;
    assert_eq!(status, 503);
    assert!(message(&body).contains("no embedder"), "{body}");
    register_toy(&state, "toy", 8);
    let (status, body) = call(&state, "POST", PATH, query).await;
    assert_eq!(status, 503);
    assert!(message(&body).contains("8-dimensional"), "{body}");
});

async_test!(malformed_queries_fail_with_clear_errors, {
    let state = seeded(None, DOCS).await;
    let cases = [
        (json!({"query": {"fusion": "rrf"}}), "prefetch"),
        (json!({"query": {"text": "x"}}), "model"),
        (json!({"query": {"text": "  ", "model": "bm25"}}), "empty"),
        (
            json!({"prefetch": [{"query": [1.0, 1.0, 1.0]}], "query": [1.0, 1.0, 1.0]}),
            "fusion",
        ),
        (
            json!({"query": {"text": "x", "model": "bm25"}, "limit": 0}),
            "limit",
        ),
        (json!({"query": {"fusion": "dbsf"}, "prefetch": []}), "dbsf"),
    ];
    for (query, needle) in cases {
        let (status, body) = call(&state, "POST", PATH, query.clone()).await;
        assert_eq!(status, 400, "{query}");
        assert!(message(&body).contains(needle), "{query} -> {body}");
    }
});

async_test!(bm25_without_text_index_is_refused, {
    let state = crate::router::new_app_state();
    call(
        &state,
        "PUT",
        "/collections/plain",
        json!({"vectors": {"size": 3, "distance": "Cosine"}}),
    )
    .await;
    let (status, body) = call(
        &state,
        "POST",
        "/collections/plain/points/query",
        json!({
            "query": {"text": "worker", "model": "bm25"}
        }),
    )
    .await;
    assert_eq!(status, 400);
    assert!(message(&body).contains("text_index"), "{body}");
});

async_test!(invalid_text_settings_are_rejected_at_creation, {
    let state = crate::router::new_app_state();
    for config in [
        json!({"vectors": {"size": 3, "distance": "Cosine"}, "text_index": {"field": "text", "b": 2.0}}),
        json!({"vectors": {"size": 3, "distance": "Cosine"}, "embedding": {"model": "bm25"}}),
        json!({"vectors": {"size": 3, "distance": "Cosine"}, "text_index": {"fields": "text"}}),
    ] {
        let (status, _) = call(&state, "PUT", "/collections/bad", config.clone()).await;
        assert_eq!(status, 400, "{config}");
    }
});
