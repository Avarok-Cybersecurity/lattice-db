//! Prefetch + RRF fusion, and hybrid vs. vector-only relevance.

use super::{call, ids, register_toy, seeded};
use serde_json::{json, Value};

macro_rules! async_test {
    ($name:ident, $body:expr) => {
        #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
        async fn $name() {
            $body
        }
    };
}

const PATH: &str = "/collections/docs/points/query";

const CORPUS: &[(u64, &str)] = &[
    (1, "installing the github action adds a workflow file"),
    (2, "pricing the free plan includes unlimited searches"),
    (3, "service worker mode runs the database in the browser"),
    (4, "cloudflare worker deployment fits the free plan limits"),
    (5, "hybrid search fuses keyword ranking with embeddings"),
    (6, "privacy no visitor query leaves the browser"),
    (7, "troubleshooting an empty index check the crawl step"),
    (8, "embedding models need an endpoint model name and key"),
    (9, "changelog version two adds sitemap crawling"),
    (10, "accessibility the dialog is keyboard navigable"),
];

const QUERIES: &[(&str, u64)] = &[
    ("github action workflow", 1),
    ("free plan pricing", 2),
    ("service worker browser", 3),
    ("cloudflare deployment", 4),
    ("keyword ranking fuses", 5),
    ("privacy visitor", 6),
    ("empty index troubleshooting", 7),
    ("embedding endpoint key", 8),
    ("sitemap changelog", 9),
    ("keyboard accessibility", 10),
];

fn hybrid(text: &str, limit: usize) -> Value {
    json!({
        "prefetch": [
            {"query": {"text": text, "model": "bm25"}, "limit": 10},
            {"query": {"text": text, "model": "toy-weak"}, "limit": 10}
        ],
        "query": {"fusion": "rrf"},
        "limit": limit
    })
}

/// Mean reciprocal rank of the expected document over all queries.
async fn mrr(state: &crate::router::AppState, request: impl Fn(&str) -> Value) -> f64 {
    let mut total = 0.0;
    for (text, expected) in QUERIES {
        let (status, body) = call(state, "POST", PATH, request(text)).await;
        assert_eq!(status, 200, "{body}");
        if let Some(pos) = ids(&body).iter().position(|id| id == expected) {
            total += 1.0 / (pos as f64 + 1.0);
        }
    }
    total / QUERIES.len() as f64
}

async_test!(rrf_fuses_bm25_and_dense_prefetches, {
    let state = seeded(Some("toy-weak"), CORPUS).await;
    register_toy(&state, "toy-weak", 3);
    let (status, body) = call(
        &state,
        "POST",
        PATH,
        hybrid("cloudflare worker free plan", 3),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["status"], "ok");
    let points = body["result"]["points"].as_array().unwrap();
    assert_eq!(points.len(), 3);
    assert_eq!(points[0]["id"], 4);
    // An RRF score is a sum of 1/(60 + rank) over at most two lists.
    let top = points[0]["score"].as_f64().unwrap();
    assert!(top > 1.0 / 71.0 && top <= 2.0 / 61.0 + 1e-6, "{top}");
    assert!(points[0]["payload"]["text"].is_string());
});

async_test!(rrf_k_parameter_and_nested_prefetch, {
    let state = seeded(None, CORPUS).await;
    let (status, body) = call(
        &state,
        "POST",
        PATH,
        json!({
            "prefetch": [{
                "prefetch": [{"query": {"text": "free plan", "model": "bm25"}}],
                "query": {"rrf": {}}
            }],
            "query": {"rrf": {"k": 1}},
            "limit": 1
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    // A single list, rank 1, k = 1: score = 1/(1 + 1).
    let score = body["result"]["points"][0]["score"].as_f64().unwrap();
    assert!((score - 0.5).abs() < 1e-6, "{body}");
});

async_test!(hybrid_beats_vector_only_when_dense_vectors_are_weak, {
    let state = seeded(Some("toy-weak"), CORPUS).await;
    register_toy(&state, "toy-weak", 3);
    let vector_only = mrr(
        &state,
        |text| json!({"query": {"text": text, "model": "toy-weak"}, "limit": 10}),
    )
    .await;
    let fused = mrr(&state, |text| hybrid(text, 10)).await;
    assert!(
        fused > vector_only + 0.2,
        "hybrid MRR {fused:.3} should clearly beat vector-only MRR {vector_only:.3}"
    );
});
