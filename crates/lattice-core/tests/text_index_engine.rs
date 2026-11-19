//! The BM25 index is maintained by the engine and survives every load path.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

use lattice_core::engine::collection::CollectionEngine;
use lattice_core::text::{EmbeddingConfig, TextIndexConfig};
use lattice_core::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
use lattice_core::types::point::Point;
use lattice_core::{export_jsonl, import_jsonl, ExportOptions, LatticeError, TextSearchQuery};

fn config() -> CollectionConfig {
    CollectionConfig::new(
        "docs",
        VectorConfig::new(2, Distance::Cosine),
        HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 100,
            ef_construction: 200,
        },
    )
    .with_text_index(TextIndexConfig::bm25("text"))
}

fn doc(id: u64, text: &str) -> Point {
    let json = serde_json::to_vec(text).unwrap();
    Point::new_vector(id, vec![1.0, id as f32]).with_field("text", json)
}

fn query(text: &str) -> TextSearchQuery {
    TextSearchQuery {
        text: text.to_string(),
        limit: 10,
        with_payload: true,
        with_vector: false,
    }
}

fn ids(engine: &CollectionEngine, text: &str) -> Vec<u64> {
    engine
        .search_text(&query(text))
        .unwrap()
        .iter()
        .map(|r| r.id)
        .collect()
}

fn seeded() -> CollectionEngine {
    let mut e = CollectionEngine::new(config()).unwrap();
    e.upsert_points(vec![
        doc(1, "rust wasm database"),
        doc(2, "service worker cache"),
        doc(3, "bm25 keyword ranking in rust"),
    ])
    .unwrap();
    e
}

#[test]
fn upsert_update_and_delete_keep_the_index_current() {
    let mut e = seeded();
    assert_eq!(ids(&e, "rust"), vec![1, 3]);
    e.upsert_points(vec![doc(1, "python only")]).unwrap();
    assert_eq!(ids(&e, "rust"), vec![3]);
    e.delete_points(&[3]).unwrap();
    assert!(ids(&e, "rust").is_empty());
    assert_eq!(ids(&e, "python"), vec![1]);
}

#[test]
fn results_carry_payload_and_bm25_scores() {
    let e = seeded();
    let results = e.search_text(&query("service worker")).unwrap();
    assert_eq!(results[0].id, 2);
    assert!(results[0].score > 0.0);
    assert_eq!(
        results[0].payload.as_ref().unwrap()["text"],
        "service worker cache"
    );
}

#[test]
fn snapshot_restore_rebuilds_the_index() {
    let e = seeded();
    let restored = CollectionEngine::from_bytes(&e.to_bytes().unwrap()).unwrap();
    assert_eq!(
        restored.config().text_index,
        Some(TextIndexConfig::bm25("text"))
    );
    for q in ["rust", "worker", "keyword ranking"] {
        assert_eq!(ids(&restored, q), ids(&e, q), "{q}");
    }
}

#[test]
fn jsonl_import_rebuilds_the_index() {
    let e = seeded();
    let mut buf = Vec::new();
    export_jsonl(&e, e.config(), &mut buf, &ExportOptions::inline()).unwrap();
    let dump = import_jsonl(buf.as_slice()).unwrap();
    let mut loaded = CollectionEngine::new(dump.header.config.clone()).unwrap();
    loaded.upsert_points(dump.points).unwrap();
    assert_eq!(ids(&loaded, "bm25 rust"), ids(&e, "bm25 rust"));
}

#[test]
fn collection_without_text_index_refuses_text_queries() {
    let mut cfg = config();
    cfg.text_index = None;
    let e = CollectionEngine::new(cfg).unwrap();
    let err = e.search_text(&query("anything")).unwrap_err();
    assert!(
        matches!(err, LatticeError::InvalidOperation { .. }),
        "{err}"
    );
    assert!(err.to_string().contains("text_index"));
}

#[test]
fn invalid_text_and_embedding_config_is_rejected() {
    let mut bad_b = TextIndexConfig::bm25("text");
    bad_b.b = 1.5;
    assert!(CollectionEngine::new(config().with_text_index(bad_b)).is_err());
    assert!(CollectionEngine::new(config().with_text_index(TextIndexConfig::bm25(""))).is_err());
    let lexical = EmbeddingConfig {
        model: "bm25".into(),
    };
    assert!(CollectionEngine::new(config().with_embedding(lexical)).is_err());
}

#[test]
fn config_json_omits_absent_text_settings() {
    let mut cfg = config();
    cfg.text_index = None;
    let json = serde_json::to_string(&cfg).unwrap();
    assert!(!json.contains("text_index") && !json.contains("embedding"));
    let parsed: TextIndexConfig = serde_json::from_str(r#"{"field":"body"}"#).unwrap();
    assert_eq!(parsed, TextIndexConfig::bm25("body"));
    assert!(serde_json::from_str::<TextIndexConfig>(r#"{"field":"x","k":1}"#).is_err());
}
