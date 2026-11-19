//! Read-only serving mode, driven through the real router.
//!
//! Every mutating endpoint must answer 403 *and leave the data untouched*;
//! every read endpoint must keep answering 200.
#![cfg(not(target_arch = "wasm32"))]

use lattice_server::dto::{
    AddEdgeRequest, CreateCollectionRequest, PointStruct, UpsertPointsRequest, VectorParams,
};
use lattice_server::handlers::{collections, export_import, points};
use lattice_server::router::{new_app_state_with_config, route, AppState, ServerConfig};
use lattice_server::LatticeRequest;

const C: &str = "docs";

/// A read-only state holding one collection with two points and an edge,
/// seeded through the handlers (the host's own path, which read-only does not
/// govern).
async fn seeded(read_only: bool) -> AppState {
    let state = new_app_state_with_config(ServerConfig::production().with_read_only(read_only));
    let create = CreateCollectionRequest {
        vectors: VectorParams {
            size: 4,
            distance: "Cosine".to_string(),
        },
        hnsw_config: None,
        durability: None,
        text_index: None,
        embedding: None,
    };
    assert_eq!(
        collections::create_collection(&state, C, create)
            .await
            .status,
        200
    );
    let pts = (1..=2u64)
        .map(|id| PointStruct {
            id,
            vector: vec![id as f32, 0.5, 0.25, 1.0],
            payload: None,
        })
        .collect();
    let up = points::upsert_points(&state, C, UpsertPointsRequest { points: pts }).await;
    assert_eq!(up.status, 200);
    let edge = AddEdgeRequest {
        from_id: 1,
        to_id: 2,
        relation: "links".into(),
        weight: 1.0,
    };
    assert_eq!(points::add_edge(&state, C, edge).await.status, 200);
    state
}

async fn call(state: &AppState, method: &str, path: &str, body: &str) -> u16 {
    let req = LatticeRequest::new(method, path).with_body(body.as_bytes().to_vec());
    route(state.clone(), req).await.status
}

async fn point_count(state: &AppState) -> usize {
    let handle = state.get_collection(C).expect("collection survives");
    let count = handle.read().await.point_count();
    count
}

async fn snapshot(state: &AppState) -> Vec<u8> {
    export_import::export_collection(state, C).await.body
}

fn mutations(snapshot: &str) -> Vec<(&'static str, String, String)> {
    let base = format!("/collections/{C}");
    vec![
        (
            "PUT",
            "/collections/fresh".into(),
            r#"{"vectors":{"size":4,"distance":"Cosine"}}"#.into(),
        ),
        ("DELETE", base.clone(), String::new()),
        (
            "PUT",
            format!("{base}/points"),
            r#"{"points":[{"id":9,"vector":[1,0,0,0]}]}"#.into(),
        ),
        (
            "POST",
            format!("{base}/points/delete"),
            r#"{"points":[1]}"#.into(),
        ),
        (
            "POST",
            format!("{base}/graph/edges"),
            r#"{"from_id":2,"to_id":1,"relation":"links","weight":1.0}"#.into(),
        ),
        (
            "POST",
            format!("{base}/import?mode=replace"),
            snapshot.into(),
        ),
        (
            "POST",
            format!("{base}/graph/query"),
            r#"{"query":"CREATE (n:Doc {title: 'x'})"}"#.into(),
        ),
        (
            "POST",
            format!("{base}/graph/query"),
            r#"{"query":"MATCH (n) DELETE n"}"#.into(),
        ),
        ("GET", "/unknown/endpoint".into(), String::new()),
    ]
}

#[tokio::test]
async fn every_mutation_is_refused_and_changes_nothing() {
    let state = seeded(true).await;
    let before = snapshot(&state).await;
    // Import body is binary; a lossy string still exercises the 403 path.
    let snap = String::from_utf8_lossy(&before).into_owned();
    for (method, path, body) in mutations(&snap) {
        assert_eq!(
            call(&state, method, &path, &body).await,
            403,
            "{method} {path}"
        );
    }
    assert_eq!(state.list_collection_names(), vec![C.to_string()]);
    assert_eq!(point_count(&state).await, 2);
    assert_eq!(
        snapshot(&state).await,
        before,
        "data changed under read-only"
    );
}

#[tokio::test]
async fn every_read_still_answers() {
    let state = seeded(true).await;
    let base = format!("/collections/{C}");
    let reads: Vec<(&str, String, &str)> = vec![
        ("GET", "/ping".into(), ""),
        ("GET", "/collections".into(), ""),
        ("GET", base.clone(), ""),
        ("GET", format!("{base}/export"), ""),
        ("POST", format!("{base}/points"), r#"{"ids":[1,2]}"#),
        (
            "POST",
            format!("{base}/points/search"),
            r#"{"vector":[1,0.5,0.25,1],"limit":2}"#,
        ),
        (
            "POST",
            format!("{base}/points/search/batch"),
            r#"{"searches":[{"vector":[1,0.5,0.25,1],"limit":2}]}"#,
        ),
        (
            "POST",
            format!("{base}/points/query"),
            r#"{"query":[1,0.5,0.25,1],"limit":2}"#,
        ),
        ("POST", format!("{base}/points/scroll"), r#"{"limit":10}"#),
        (
            "POST",
            format!("{base}/graph/traverse"),
            r#"{"start_id":1,"max_depth":2}"#,
        ),
        (
            "POST",
            format!("{base}/graph/query"),
            r#"{"query":"MATCH (n) RETURN n.id"}"#,
        ),
    ];
    for (method, path, body) in reads {
        assert_eq!(
            call(&state, method, &path, body).await,
            200,
            "{method} {path}"
        );
    }
}

/// Negative control: the same mutations succeed when read-only is off, so the
/// 403s above come from the mode and not from malformed requests.
#[tokio::test]
async fn writable_state_accepts_the_same_mutations() {
    let state = seeded(false).await;
    let snap_bytes = snapshot(&state).await;
    let ok = |s: u16| (200..300).contains(&s);
    let base = format!("/collections/{C}");
    let fresh = r#"{"vectors":{"size":4,"distance":"Cosine"}}"#;
    assert!(ok(call(&state, "PUT", "/collections/fresh", fresh).await));
    let upsert = r#"{"points":[{"id":9,"vector":[1,0,0,0]}]}"#;
    assert!(ok(
        call(&state, "PUT", &format!("{base}/points"), upsert).await
    ));
    let edge = r#"{"from_id":2,"to_id":1,"relation":"links","weight":1.0}"#;
    assert!(ok(call(
        &state,
        "POST",
        &format!("{base}/graph/edges"),
        edge
    )
    .await));
    let del = r#"{"points":[9]}"#;
    assert!(ok(call(
        &state,
        "POST",
        &format!("{base}/points/delete"),
        del
    )
    .await));
    let import =
        LatticeRequest::new("POST", format!("{base}/import?mode=replace")).with_body(snap_bytes);
    assert!(ok(route(state.clone(), import).await.status));
    let create = r#"{"query":"CREATE (n:Doc {title: 'x'})"}"#;
    assert!(ok(call(
        &state,
        "POST",
        &format!("{base}/graph/query"),
        create
    )
    .await));
    assert!(ok(call(&state, "DELETE", &base, "").await));
}
