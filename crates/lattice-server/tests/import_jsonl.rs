//! JSONL (logical) import into server state, and the shared install path.
#![cfg(not(target_arch = "wasm32"))]

use lattice_core::{
    export_jsonl, CollectionConfig, CollectionEngine, Distance, ExportOptions, HnswConfig, Point,
    VectorConfig,
};
use lattice_server::dto::ImportMode;
use lattice_server::handlers::export_import::import_collection;
use lattice_server::handlers::import_jsonl::import_jsonl_collection;
use lattice_server::router::new_app_state;

fn config() -> CollectionConfig {
    CollectionConfig::new(
        "site",
        VectorConfig::new(3, Distance::Cosine),
        HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 64,
            ef_construction: 100,
        },
    )
}

fn dump(opts: &ExportOptions) -> Vec<u8> {
    let mut e = CollectionEngine::new(config()).expect("engine");
    e.upsert_points(vec![
        Point::new_vector(1, vec![1.0, 0.0, 0.0]).with_field("title", br#""a""#.to_vec()),
        Point::new_vector(2, vec![0.0, 1.0, 0.0]),
    ])
    .expect("upsert");
    e.add_edge(1, 2, "links", 0.5).expect("edge");
    let mut out = Vec::new();
    export_jsonl(&e, &config(), &mut out, opts).expect("export");
    out
}

#[tokio::test]
async fn jsonl_dump_installs_under_its_own_name() {
    let state = new_app_state();
    let resp =
        import_jsonl_collection(&state, ImportMode::Create, &dump(&ExportOptions::inline())).await;
    assert_eq!(resp.status, 200, "{}", String::from_utf8_lossy(&resp.body));
    let handle = state.get_collection("site").expect("named by the dump");
    let engine = handle.read().await;
    assert_eq!(engine.point_count(), 2);
    let p1 = engine.get_point(1).expect("read").expect("present");
    assert_eq!(p1.vector, vec![1.0, 0.0, 0.0]);
    assert_eq!(p1.outgoing_edges.as_ref().map(|e| e.len()), Some(1));
}

#[tokio::test]
async fn create_mode_conflicts_and_replace_overwrites() {
    let state = new_app_state();
    let bytes = dump(&ExportOptions::inline());
    assert_eq!(
        import_jsonl_collection(&state, ImportMode::Create, &bytes)
            .await
            .status,
        200
    );
    assert_eq!(
        import_jsonl_collection(&state, ImportMode::Create, &bytes)
            .await
            .status,
        409
    );
    assert_eq!(
        import_jsonl_collection(&state, ImportMode::Replace, &bytes)
            .await
            .status,
        200
    );
}

#[tokio::test]
async fn dump_without_vectors_is_refused() {
    let state = new_app_state();
    let resp = import_jsonl_collection(
        &state,
        ImportMode::Create,
        &dump(&ExportOptions::omit_vectors()),
    )
    .await;
    assert_eq!(resp.status, 400);
    assert!(state.get_collection("site").is_none());
}

#[tokio::test]
async fn truncated_dump_is_refused() {
    let state = new_app_state();
    let bytes = dump(&ExportOptions::inline());
    let text = String::from_utf8(bytes).expect("utf8");
    let cut: String = text.lines().take(2).collect::<Vec<_>>().join("\n");
    let resp = import_jsonl_collection(&state, ImportMode::Create, cut.as_bytes()).await;
    assert_eq!(resp.status, 400);
}

/// A malformed snapshot must not destroy the collection it was meant to replace.
#[tokio::test]
async fn replace_with_garbage_keeps_the_existing_collection() {
    let state = new_app_state();
    let bytes = dump(&ExportOptions::inline());
    assert_eq!(
        import_jsonl_collection(&state, ImportMode::Create, &bytes)
            .await
            .status,
        200
    );
    let resp = import_collection(&state, "site", ImportMode::Replace, vec![7; 64]).await;
    assert_eq!(resp.status, 400);
    assert!(
        state.get_collection("site").is_some(),
        "replace destroyed data"
    );
}
