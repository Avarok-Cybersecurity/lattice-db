//! Lexical-only collections and flat/compact vector indexes, through the
//! real router (the same surface every host serves).
#![cfg(not(target_arch = "wasm32"))]

use lattice_server::router::{new_app_state, route, AppState};
use lattice_server::LatticeRequest;
use serde_json::{json, Value};

async fn call(state: &AppState, method: &str, path: &str, body: Value) -> (u16, Value) {
    let bytes = if body.is_null() {
        Vec::new()
    } else {
        body.to_string().into_bytes()
    };
    let response = route(
        state.clone(),
        LatticeRequest::new(method, path).with_body(bytes),
    )
    .await;
    let json = serde_json::from_slice(&response.body).unwrap_or(Value::Null);
    (response.status, json)
}

fn ids(body: &Value) -> Vec<u64> {
    let list = body["result"]["points"]
        .as_array()
        .or_else(|| body["result"].as_array())
        .cloned()
        .unwrap_or_default();
    list.iter().filter_map(|p| p["id"].as_u64()).collect()
}

fn message(body: &Value) -> String {
    body["message"]
        .as_str()
        .or_else(|| body["status"]["error"].as_str())
        .unwrap_or_default()
        .to_string()
}

async fn lexical_state() -> AppState {
    let state = new_app_state();
    let create = json!({ "text_index": { "field": "text" } });
    assert_eq!(
        call(&state, "PUT", "/collections/site", create).await.0,
        200
    );
    let points = json!({ "points": [
        { "id": 1, "payload": { "text": "free plan limits and pricing" } },
        { "id": 2, "payload": { "text": "deploying a worker" } },
        { "id": 3, "payload": { "text": "pricing for teams" } },
    ]});
    assert_eq!(
        call(&state, "PUT", "/collections/site/points", points)
            .await
            .0,
        200
    );
    state
}

#[tokio::test]
async fn a_collection_without_vectors_answers_bm25_queries() {
    let state = lexical_state().await;
    let (status, body) = call(
        &state,
        "POST",
        "/collections/site/points/query",
        json!({ "query": { "text": "pricing", "model": "bm25" }, "limit": 5 }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let mut found = ids(&body);
    found.sort_unstable();
    assert_eq!(found, vec![1, 3]);

    let (status, info) = call(&state, "GET", "/collections/site", Value::Null).await;
    assert_eq!(status, 200);
    assert!(
        info["result"]["config"]["params"]["vectors"].is_null(),
        "{info}"
    );

    let export = route(
        state.clone(),
        LatticeRequest::new("GET", "/collections/site/export"),
    )
    .await;
    assert_eq!(export.status, 200);
    let copy = new_app_state();
    let import =
        LatticeRequest::new("POST", "/collections/site/import?mode=create").with_body(export.body);
    assert_eq!(route(copy.clone(), import).await.status, 200);
    let query = json!({ "query": { "text": "worker", "model": "bm25" } });
    let (_, body) = call(&copy, "POST", "/collections/site/points/query", query).await;
    assert_eq!(ids(&body), vec![2]);
}

#[tokio::test]
async fn vector_operations_on_a_lexical_collection_are_clear_400s() {
    let state = lexical_state().await;
    let cases = [
        (
            "/collections/site/points/search",
            json!({ "vector": [0.1, 0.2], "limit": 3 }),
        ),
        (
            "/collections/site/points/query",
            json!({ "query": [0.1, 0.2] }),
        ),
        (
            "/collections/site/points/search/batch",
            json!({ "searches": [{ "vector": [0.1], "limit": 1 }] }),
        ),
    ];
    for (path, body) in cases {
        let (status, response) = call(&state, "POST", path, body).await;
        assert_eq!(status, 400, "{path}: {response}");
        assert!(
            message(&response).contains("no dense vectors"),
            "{path}: {response}"
        );
    }
    let with_vector = json!({ "points": [{ "id": 9, "vector": [1.0], "payload": {} }] });
    let (status, response) = call(&state, "PUT", "/collections/site/points", with_vector).await;
    assert_eq!(status, 400);
    assert!(
        message(&response).contains("no dense vectors"),
        "{response}"
    );
    let dense_text = json!({ "query": { "text": "pricing", "model": "some-model" } });
    let (status, _) = call(&state, "POST", "/collections/site/points/query", dense_text).await;
    assert_eq!(status, 400);
}

#[tokio::test]
async fn an_embedding_model_needs_vectors() {
    let state = new_app_state();
    let create = json!({ "text_index": { "field": "text" }, "embedding": { "model": "m" } });
    let (status, body) = call(&state, "PUT", "/collections/bad", create).await;
    assert_eq!(status, 400, "{body}");
    let zero = json!({ "vectors": { "size": 0, "distance": "Cosine" } });
    assert_eq!(call(&state, "PUT", "/collections/zero", zero).await.0, 400);
    let unknown = json!({ "vectors": { "size": 2, "distance": "Cosine", "storage": "f64" } });
    assert_eq!(
        call(&state, "PUT", "/collections/f64", unknown).await.0,
        400
    );
}

#[tokio::test]
async fn flat_f16_collections_search_exactly_and_report_their_config() {
    let state = new_app_state();
    let create =
        json!({ "vectors": { "size": 3, "distance": "Dot", "index": "flat", "storage": "f16" } });
    assert_eq!(call(&state, "PUT", "/collections/f", create).await.0, 200);
    let points = json!({ "points": [
        { "id": 1, "vector": [0.1, 0.0, 0.0] },
        { "id": 2, "vector": [0.0, 1.0, 0.0] },
        { "id": 3, "vector": [0.0, 0.5, 0.3] },
    ]});
    assert_eq!(
        call(&state, "PUT", "/collections/f/points", points).await.0,
        200
    );
    let search = json!({ "vector": [0.0, 1.0, 0.1], "limit": 3 });
    let (status, body) = call(
        &state,
        "POST",
        "/collections/f/points/search",
        search.clone(),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(ids(&body)[0], 2);

    let (_, info) = call(&state, "GET", "/collections/f", Value::Null).await;
    let vectors = &info["result"]["config"]["params"]["vectors"];
    assert_eq!(
        (vectors["index"].as_str(), vectors["storage"].as_str()),
        (Some("flat"), Some("f16"))
    );

    let (_, point) = call(
        &state,
        "POST",
        "/collections/f/points",
        json!({ "ids": [1], "with_vector": true }),
    )
    .await;
    let stored = point["result"][0]["vector"][0].as_f64().unwrap_or_default();
    assert_eq!(
        stored as f32,
        half_round(0.1),
        "vector rounded to f16 on write"
    );
}

fn half_round(x: f32) -> f32 {
    let mut v = [x];
    lattice_core::VectorStorage::F16.round_in_place(&mut v);
    v[0]
}
