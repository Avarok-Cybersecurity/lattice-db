//! The la_query refusal the owner reported (2026-10-28), end to end.
//!
//! Root cause: the planner lowers the label of a pattern's *target* node (`(k:Kernel)`) and every
//! extra label of a scanned node to a synthetic `hasLabel(k, 'Kernel')` function-call filter, and
//! the evaluator had no arm for function calls, so it fell into the catch-all
//! `Unsupported feature: Complex expression evaluation`.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

mod cypher_support;
use cypher_support::*;

#[test]
fn owner_example_returns_every_measures_edge() {
    let mut e = kg();
    let (cols, mut got) = run(
        &mut e,
        "MATCH (m:Measurement)-[:MEASURES]->(k:Kernel) RETURN m.id, k.id LIMIT 5",
    );
    assert_eq!(cols, vec!["m.id", "k.id"]);
    got.sort_by_key(|r| (r[0].as_int(), r[1].as_int()));
    assert_eq!(
        got,
        vec![
            vec![i(10), i(1)],
            vec![i(10), i(2)],
            vec![i(11), i(1)],
            vec![i(12), i(2)],
            vec![i(13), i(2)],
        ]
    );
}

#[test]
fn target_label_filters_out_other_labels() {
    // D-1 has edges to a Kernel and to Measurements; the target label keeps only the Kernel.
    rows(
        "MATCH (d:Decision)-->(k:Kernel) RETURN d.name, k.name",
        vec![vec![s("D-1"), s("K-gemv")]],
    );
}

#[test]
fn multi_label_node_pattern_requires_every_label() {
    rows("MATCH (n:Kernel:Measurement) RETURN n.name", vec![]);
    rows(
        "MATCH (n:Kernel:Kernel) RETURN n.name ORDER BY n.name",
        vec![vec![s("K-attn")], vec![s("K-gemv")], vec![s("K-idle")]],
    );
}
