//! ORDER BY over enough rows to reach the executor's batch sort paths (>= 50 and >= 500 rows).
//! Those paths read the sort property straight from storage; they must agree with the generic
//! comparator: sort by the named variable (not whatever is in column 0), honour the special `id`
//! property, and put missing values last.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use lattice_core::cypher::{CypherHandler, DefaultCypherHandler};
use lattice_core::engine::collection::CollectionEngine;
use lattice_core::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
use lattice_core::types::point::Point;
use lattice_core::types::value::CypherValue;
use std::collections::HashMap;

fn engine() -> CollectionEngine {
    CollectionEngine::new(CollectionConfig::new(
        "sort",
        VectorConfig::new(2, Distance::Cosine),
        HnswConfig {
            m: 4,
            m0: 8,
            ml: HnswConfig::recommended_ml(4),
            ef: 16,
            ef_construction: 16,
        },
    ))
    .expect("engine")
}

fn point(id: u64, label: &str, v: Option<serde_json::Value>) -> Point {
    let mut p = HashMap::new();
    p.insert(
        "_labels".to_string(),
        serde_json::to_vec(&[label]).expect("labels"),
    );
    if let Some(v) = v {
        p.insert("v".to_string(), serde_json::to_vec(&v).expect("v"));
    }
    Point::new_vector(id, vec![1.0, 0.0]).with_payload(p)
}

/// One hub (v = 0) pointing at `n` leaves; leaf `i` has v = (7 * i) % n, the last `missing`
/// leaves have no `v`.
fn hub_and_leaves(n: u64, missing: u64) -> CollectionEngine {
    let mut e = engine();
    let mut pts = vec![point(1_000_000, "Hub", Some(serde_json::json!(0)))];
    for i in 0..n {
        let v = (i < n - missing).then(|| serde_json::json!((7 * i) % n));
        pts.push(point(i, "Leaf", v));
    }
    e.upsert_points(pts).expect("upsert");
    for i in 0..n {
        e.add_edge(1_000_000, i, "L", 1.0).expect("edge");
    }
    e
}

fn rows(e: &mut CollectionEngine, q: &str) -> Vec<Vec<CypherValue>> {
    DefaultCypherHandler::new()
        .query(q, e, HashMap::new())
        .expect(q)
        .rows
}

/// The `v` a leaf node carries, as the generic comparator sees it.
fn leaf_v(n: u64, missing: u64, node: &CypherValue) -> CypherValue {
    match node {
        CypherValue::NodeRef(id) if *id < n - missing => CypherValue::Int(((7 * id) % n) as i64),
        CypherValue::NodeRef(id) if *id < n => CypherValue::Null,
        other => panic!("not a leaf: {other:?}"),
    }
}

fn expected(n: u64, missing: u64, ascending: bool) -> Vec<CypherValue> {
    let mut vals: Vec<i64> = (0..n - missing).map(|i| ((7 * i) % n) as i64).collect();
    vals.sort_unstable();
    let mut out: Vec<CypherValue> = vals.into_iter().map(CypherValue::Int).collect();
    out.extend((0..missing).map(|_| CypherValue::Null));
    if !ascending {
        out.reverse();
    }
    out
}

fn sorted_v(
    e: &mut CollectionEngine,
    q: &str,
    col: usize,
    n: u64,
    missing: u64,
) -> Vec<CypherValue> {
    rows(e, q)
        .iter()
        .map(|r| leaf_v(n, missing, &r[col]))
        .collect()
}

#[test]
fn order_by_sorts_by_the_named_variable_not_column_zero() {
    for n in [60u64, 600] {
        let mut e = hub_and_leaves(n, 0);
        let q = "MATCH (h:Hub)-[:L]->(b) RETURN h, b ORDER BY b.v";
        assert_eq!(
            sorted_v(&mut e, q, 1, n, 0),
            expected(n, 0, true),
            "{q} n = {n}"
        );
        let q = "MATCH (h:Hub)-[:L]->(b) RETURN b.v ORDER BY b.v";
        let got: Vec<CypherValue> = rows(&mut e, q)
            .into_iter()
            .map(|mut r| r.remove(0))
            .collect();
        assert_eq!(got, expected(n, 0, true), "{q} n = {n}");
    }
}

#[test]
fn order_by_puts_missing_values_last_on_the_batch_paths() {
    for n in [60u64, 600] {
        let mut e = hub_and_leaves(n, 5);
        let q = "MATCH (b:Leaf) RETURN b ORDER BY b.v";
        assert_eq!(
            sorted_v(&mut e, q, 0, n, 5),
            expected(n, 5, true),
            "{q} n = {n}"
        );
        let q = "MATCH (b:Leaf) RETURN b ORDER BY b.v DESC";
        assert_eq!(
            sorted_v(&mut e, q, 0, n, 5),
            expected(n, 5, false),
            "{q} n = {n}"
        );
        let q = "MATCH (b:Leaf) RETURN b ORDER BY b.v LIMIT 10";
        let want: Vec<CypherValue> = expected(n, 5, true).into_iter().take(10).collect();
        assert_eq!(sorted_v(&mut e, q, 0, n, 5), want, "{q} n = {n}");
    }
}

#[test]
fn order_by_id_uses_the_point_id_on_the_batch_paths() {
    for n in [60u64, 600] {
        let mut e = hub_and_leaves(n, 0);
        let got = rows(&mut e, "MATCH (b:Leaf) RETURN b ORDER BY b.id DESC LIMIT 3");
        let want: Vec<Vec<CypherValue>> = (n - 3..n)
            .rev()
            .map(|i| vec![CypherValue::NodeRef(i)])
            .collect();
        assert_eq!(got, want, "n = {n}");
    }
}
