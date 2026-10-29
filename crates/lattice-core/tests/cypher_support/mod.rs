//! Shared fixture for the Cypher read-query tests: a small knowledge-graph shaped like the
//! metrale-kg store (Kernel, Measurement, Decision nodes; MEASURES, CITES, ABOUT edges).
//!
//! Everything goes through the public `DefaultCypherHandler::query` entry point, so these tests
//! exercise the parser, planner and executor together, exactly as la_query does.

#![allow(dead_code)]

use lattice_core::cypher::{CypherHandler, DefaultCypherHandler};
use lattice_core::engine::collection::CollectionEngine;
use lattice_core::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
use lattice_core::types::point::Point;
use lattice_core::types::value::CypherValue;
use std::collections::HashMap;

fn config() -> CollectionConfig {
    CollectionConfig::new(
        "cypher_kg",
        VectorConfig::new(2, Distance::Cosine),
        HnswConfig {
            m: 4,
            m0: 8,
            ml: HnswConfig::recommended_ml(4),
            ef: 16,
            ef_construction: 16,
        },
    )
}

fn node(id: u64, labels: &[&str], props: &[(&str, serde_json::Value)]) -> Point {
    let mut payload = HashMap::new();
    payload.insert(
        "_labels".to_string(),
        serde_json::to_vec(&labels).expect("labels serialise"),
    );
    for (k, v) in props {
        payload.insert(
            (*k).to_string(),
            serde_json::to_vec(v).expect("prop serialise"),
        );
    }
    Point::new_vector(id, vec![1.0, 0.0]).with_payload(payload)
}

/// The fixture graph. Node ids are chosen so `id` order and `name` order agree.
///
/// Kernels:      1 K-gemv {family: gemv}, 2 K-attn {family: attn}, 3 K-idle {no family}
/// Measurements: 10 M-1 {tok_s 10, verdict WIN}, 11 M-2 {tok_s 20, verdict LOSS},
///               12 M-3 {tok_s 30.5, verdict WIN}, 13 M-4 {no tok_s, no verdict}
/// Decision:     20 D-1 {tags [a, b]}
/// Edges:        M-1 -MEASURES-> K-gemv, M-1 -MEASURES-> K-attn, M-2 -MEASURES-> K-gemv,
///               M-3 -MEASURES-> K-attn, M-4 -MEASURES-> K-attn,
///               D-1 -CITES-> M-1, D-1 -CITES-> M-3, D-1 -ABOUT-> K-gemv
pub fn kg() -> CollectionEngine {
    use serde_json::json;
    let mut e = CollectionEngine::new(config()).expect("engine");
    e.upsert_points(vec![
        node(
            1,
            &["Kernel"],
            &[("name", json!("K-gemv")), ("family", json!("gemv"))],
        ),
        node(
            2,
            &["Kernel"],
            &[("name", json!("K-attn")), ("family", json!("attn"))],
        ),
        node(3, &["Kernel"], &[("name", json!("K-idle"))]),
        node(
            10,
            &["Measurement"],
            &[
                ("name", json!("M-1")),
                ("tok_s", json!(10)),
                ("verdict", json!("WIN")),
            ],
        ),
        node(
            11,
            &["Measurement"],
            &[
                ("name", json!("M-2")),
                ("tok_s", json!(20)),
                ("verdict", json!("LOSS")),
            ],
        ),
        node(
            12,
            &["Measurement"],
            &[
                ("name", json!("M-3")),
                ("tok_s", json!(30.5)),
                ("verdict", json!("WIN")),
            ],
        ),
        node(13, &["Measurement"], &[("name", json!("M-4"))]),
        node(
            20,
            &["Decision"],
            &[("name", json!("D-1")), ("tags", json!(["a", "b"]))],
        ),
    ])
    .expect("upsert");
    for (from, to, rel, w) in [
        (10, 1, "MEASURES", 1.0),
        (10, 2, "MEASURES", 0.5),
        (11, 1, "MEASURES", 1.0),
        (12, 2, "MEASURES", 1.0),
        (13, 2, "MEASURES", 1.0),
        (20, 10, "CITES", 1.0),
        (20, 12, "CITES", 1.0),
        (20, 1, "ABOUT", 1.0),
    ] {
        e.add_edge(from, to, rel, w).expect("edge");
    }
    e
}

/// Run a read query against `engine`; panic with the query text on error.
pub fn run(engine: &mut CollectionEngine, query: &str) -> (Vec<String>, Vec<Vec<CypherValue>>) {
    run_with(engine, query, HashMap::new())
}

/// Run a read query with parameters.
pub fn run_with(
    engine: &mut CollectionEngine,
    query: &str,
    params: HashMap<String, CypherValue>,
) -> (Vec<String>, Vec<Vec<CypherValue>>) {
    match DefaultCypherHandler::new().query(query, engine, params) {
        Ok(r) => (r.columns, r.rows),
        Err(e) => panic!("query failed: {query}\n  error: {e}"),
    }
}

/// Run a query that must fail; return the error text.
pub fn fail(engine: &mut CollectionEngine, query: &str) -> String {
    match DefaultCypherHandler::new().query(query, engine, HashMap::new()) {
        Ok(r) => panic!(
            "query unexpectedly succeeded: {query}\n  rows: {:?}",
            r.rows
        ),
        Err(e) => e.to_string(),
    }
}

/// Rows of a query with no graph access (`RETURN …` / `UNWIND …`).
pub fn eval(query: &str) -> Vec<Vec<CypherValue>> {
    run(&mut kg(), query).1
}

/// The single value of a one-row, one-column `RETURN`.
pub fn scalar(query: &str) -> CypherValue {
    let rows = eval(query);
    assert_eq!(rows.len(), 1, "expected one row from {query}, got {rows:?}");
    assert_eq!(
        rows[0].len(),
        1,
        "expected one column from {query}, got {rows:?}"
    );
    rows[0][0].clone()
}

pub fn s(v: &str) -> CypherValue {
    CypherValue::string(v)
}
pub fn i(v: i64) -> CypherValue {
    CypherValue::Int(v)
}
pub fn f(v: f64) -> CypherValue {
    CypherValue::Float(v)
}
pub fn b(v: bool) -> CypherValue {
    CypherValue::Bool(v)
}
pub const NULL: CypherValue = CypherValue::Null;
pub fn list(v: Vec<CypherValue>) -> CypherValue {
    CypherValue::List(v)
}
pub fn strs(v: &[&str]) -> CypherValue {
    CypherValue::List(v.iter().map(|x| s(x)).collect())
}

/// Assert the exact rows (in order) a query over the fixture returns.
pub fn rows(query: &str, expected: Vec<Vec<CypherValue>>) {
    let got = run(&mut kg(), query).1;
    assert_eq!(got, expected, "rows of: {query}");
}

/// Assert the exact single row a query returns.
pub fn row(query: &str, expected: Vec<CypherValue>) {
    rows(query, vec![expected]);
}

/// Assert that a query over the fixture fails and its error names every `needle`.
pub fn refused(query: &str, needles: &[&str]) {
    let err = fail(&mut kg(), query);
    for n in needles {
        assert!(err.contains(n), "error of `{query}` lacks `{n}`: {err}");
    }
}
