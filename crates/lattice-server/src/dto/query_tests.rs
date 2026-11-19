use super::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

fn query(json: &str) -> Result<QueryInput, String> {
    serde_json::from_str::<QueryInput>(json).map_err(|e| e.to_string())
}

#[test]
fn dense_vector_forms() {
    let expected = QueryInput::Vector(vec![0.5, 1.0]);
    assert_eq!(query("[0.5, 1.0]").unwrap(), expected);
    assert_eq!(query(r#"{"nearest": [0.5, 1.0]}"#).unwrap(), expected);
    assert_eq!(
        query(r#"{"nearest": {"vector": [0.5, 1.0]}}"#).unwrap(),
        expected
    );
    assert!(query("[1e300]").unwrap_err().contains("finite"));
}

#[test]
fn text_document_forms() {
    let expected = QueryInput::Text {
        text: "hello".into(),
        model: "bm25".into(),
    };
    assert_eq!(
        query(r#"{"text": "hello", "model": "bm25"}"#).unwrap(),
        expected
    );
    assert_eq!(
        query(r#"{"nearest": {"text": "hello", "model": "bm25"}}"#).unwrap(),
        expected
    );
    assert!(query(r#"{"text": "hello"}"#).unwrap_err().contains("model"));
}

#[test]
fn fusion_forms() {
    let default = QueryInput::Fusion(Fusion::Rrf {
        k: lattice_core::text::RRF_K,
    });
    assert_eq!(query(r#"{"fusion": "rrf"}"#).unwrap(), default);
    assert_eq!(query(r#"{"rrf": {}}"#).unwrap(), default);
    assert_eq!(
        query(r#"{"rrf": {"k": 2}}"#).unwrap(),
        QueryInput::Fusion(Fusion::Rrf { k: 2 })
    );
    assert!(query(r#"{"rrf": {"k": 0}}"#).is_err());
    assert!(query(r#"{"fusion": "dbsf"}"#).unwrap_err().contains("dbsf"));
    assert!(query(r#"{"recommend": {}}"#).is_err());
}

#[test]
fn prefetch_accepts_one_object_or_a_list_and_nests() {
    let req: QueryRequest = serde_json::from_str(
        r#"{"prefetch": {"query": [1.0], "limit": 5}, "query": {"fusion": "rrf"}}"#,
    )
    .unwrap();
    assert_eq!(req.prefetch.len(), 1);
    assert_eq!(req.prefetch[0].limit, 5);
    assert_eq!(req.limit, 10);

    let req: QueryRequest = serde_json::from_str(
        r#"{"prefetch": [
              {"prefetch": [{"query": [1.0]}], "query": {"rrf": {}}},
              {"query": {"text": "x", "model": "bm25"}, "using": "sparse"}
           ],
           "query": {"fusion": "rrf"}, "limit": 3}"#,
    )
    .unwrap();
    assert_eq!(req.prefetch.len(), 2);
    assert_eq!(req.prefetch[0].prefetch.len(), 1);
}
