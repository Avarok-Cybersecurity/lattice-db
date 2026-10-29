//! Reading clauses and projection modifiers: WITH, ORDER BY, DISTINCT, SKIP/LIMIT,
//! OPTIONAL MATCH, UNWIND, relationship direction, multi-pattern joins.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

mod cypher_support;
use cypher_support::*;
use lattice_core::types::value::CypherValue;

#[test]
fn with_projects_and_filters() {
    rows(
        "MATCH (m:Measurement) WITH m.name AS name, m.tok_s AS t WHERE t > 15 RETURN name ORDER BY name",
        vec![vec![s("M-2")], vec![s("M-3")]],
    );
    rows(
        "MATCH (k:Kernel) WITH * WHERE k.family = 'gemv' RETURN k.name",
        vec![vec![s("K-gemv")]],
    );
}

#[test]
fn match_after_with_expands_from_the_projected_binding() {
    rows(
        "MATCH (d:Decision) WITH d MATCH (d)-[:CITES]->(m) RETURN m.name ORDER BY m.name",
        vec![vec![s("M-1")], vec![s("M-3")]],
    );
}

#[test]
fn with_order_by_limit_feeds_the_next_clause() {
    rows(
        "MATCH (m:Measurement) WITH m ORDER BY m.tok_s DESC LIMIT 2 RETURN m.name ORDER BY m.name",
        vec![vec![s("M-3")], vec![s("M-4")]],
    );
}

#[test]
fn order_by_puts_null_last_ascending_and_first_descending() {
    rows(
        "MATCH (m:Measurement) RETURN m.name ORDER BY m.tok_s",
        vec![
            vec![s("M-1")],
            vec![s("M-2")],
            vec![s("M-3")],
            vec![s("M-4")],
        ],
    );
    rows(
        "MATCH (m:Measurement) RETURN m.name ORDER BY m.tok_s DESC",
        vec![
            vec![s("M-4")],
            vec![s("M-3")],
            vec![s("M-2")],
            vec![s("M-1")],
        ],
    );
}

#[test]
fn order_by_an_unprojected_expression_or_an_alias() {
    rows(
        "MATCH (k:Kernel) RETURN k.name ORDER BY coalesce(k.family, 'zzz') DESC",
        vec![vec![s("K-idle")], vec![s("K-gemv")], vec![s("K-attn")]],
    );
    rows(
        "MATCH (m:Measurement) RETURN m.name AS n ORDER BY n DESC",
        vec![
            vec![s("M-4")],
            vec![s("M-3")],
            vec![s("M-2")],
            vec![s("M-1")],
        ],
    );
}

#[test]
fn distinct_applies_to_the_projected_columns() {
    rows(
        "MATCH (m:Measurement) RETURN DISTINCT m.verdict AS v ORDER BY v",
        vec![vec![s("LOSS")], vec![s("WIN")], vec![NULL]],
    );
    rows(
        "MATCH (m:Measurement)-[:MEASURES]->(k) RETURN DISTINCT k.name AS k ORDER BY k",
        vec![vec![s("K-attn")], vec![s("K-gemv")]],
    );
}

#[test]
fn limit_counts_distinct_rows() {
    rows(
        "MATCH (m:Measurement)-[:MEASURES]->(k) RETURN DISTINCT k.name AS k ORDER BY k LIMIT 2",
        vec![vec![s("K-attn")], vec![s("K-gemv")]],
    );
}

#[test]
fn skip_and_limit() {
    rows(
        "MATCH (m:Measurement) RETURN m.name ORDER BY m.name SKIP 1 LIMIT 2",
        vec![vec![s("M-2")], vec![s("M-3")]],
    );
    rows(
        "MATCH (m:Measurement) RETURN m.name ORDER BY m.name SKIP 10",
        vec![],
    );
    rows("MATCH (m:Measurement) RETURN m.name LIMIT 0", vec![]);
}

#[test]
fn optional_match_extends_unmatched_rows_with_null() {
    rows(
        "MATCH (k:Kernel) OPTIONAL MATCH (k)<-[:ABOUT]-(d:Decision) RETURN k.name, d.name ORDER BY k.name",
        vec![
            vec![s("K-attn"), NULL],
            vec![s("K-gemv"), s("D-1")],
            vec![s("K-idle"), NULL],
        ],
    );
}

#[test]
fn optional_match_where_belongs_to_the_optional_pattern() {
    rows(
        "MATCH (k:Kernel) OPTIONAL MATCH (m:Measurement)-[:MEASURES]->(k) WHERE m.verdict = 'LOSS' \
         RETURN k.name, m.name ORDER BY k.name",
        vec![
            vec![s("K-attn"), NULL],
            vec![s("K-gemv"), s("M-2")],
            vec![s("K-idle"), NULL],
        ],
    );
}

#[test]
fn leading_optional_match_without_a_match_yields_one_null_row() {
    rows("OPTIONAL MATCH (x:Nope) RETURN x", vec![vec![NULL]]);
}

#[test]
fn unwind() {
    rows(
        "UNWIND [3, 1, 2] AS x RETURN x ORDER BY x",
        vec![vec![i(1)], vec![i(2)], vec![i(3)]],
    );
    rows("UNWIND [] AS x RETURN x", vec![]);
    rows("UNWIND null AS x RETURN x", vec![]);
    rows(
        "MATCH (d:Decision) UNWIND d.tags AS t RETURN t ORDER BY t",
        vec![vec![s("a")], vec![s("b")]],
    );
    rows(
        "UNWIND [1, 2] AS x UNWIND ['a', 'b'] AS y RETURN x, y",
        vec![
            vec![i(1), s("a")],
            vec![i(1), s("b")],
            vec![i(2), s("a")],
            vec![i(2), s("b")],
        ],
    );
}

#[test]
fn incoming_and_undirected_relationships() {
    rows(
        "MATCH (k:Kernel {name: 'K-gemv'})<-[:MEASURES]-(m) RETURN m.name ORDER BY m.name",
        vec![vec![s("M-1")], vec![s("M-2")]],
    );
    rows(
        "MATCH (k:Kernel {name: 'K-gemv'})-[]-(x) RETURN x.name ORDER BY x.name",
        vec![vec![s("D-1")], vec![s("M-1")], vec![s("M-2")]],
    );
}

#[test]
fn comma_separated_patterns_join_on_shared_variables() {
    rows(
        "MATCH (d:Decision)-[:CITES]->(m), (m)-[:MEASURES]->(k:Kernel) RETURN m.name, k.name ORDER BY m.name, k.name",
        vec![
            vec![s("M-1"), s("K-attn")],
            vec![s("M-1"), s("K-gemv")],
            vec![s("M-3"), s("K-attn")],
        ],
    );
    rows(
        "MATCH (d:Decision)-[:ABOUT]->(k:Kernel)<-[:MEASURES]-(m) RETURN m.name ORDER BY m.name",
        vec![vec![s("M-1")], vec![s("M-2")]],
    );
    // Disconnected patterns: a cartesian product.
    row("MATCH (d:Decision), (k:Kernel) RETURN count(*)", vec![i(3)]);
}

#[test]
fn a_relationship_is_matched_at_most_once_per_pattern() {
    rows(
        "MATCH (a:Measurement)-[:MEASURES]->(k)<-[:MEASURES]-(b) WHERE a.name = 'M-2' RETURN b.name ORDER BY b.name",
        vec![vec![s("M-1")]],
    );
}

#[test]
fn return_star_returns_the_named_variables() {
    let (cols, got) = run(&mut kg(), "MATCH (k:Kernel {name: 'K-idle'}) RETURN *");
    assert_eq!(cols, vec!["k"]);
    assert_eq!(got, vec![vec![CypherValue::NodeRef(3)]]);
}

#[test]
fn variable_length_paths_keep_reachable_node_semantics() {
    rows(
        "MATCH (d:Decision)-[:CITES*1..2]->(x) RETURN x.name ORDER BY x.name",
        vec![vec![s("M-1")], vec![s("M-3")]],
    );
    rows(
        "MATCH (d:Decision)-[*1..2]->(x) RETURN x.name ORDER BY x.name",
        vec![
            vec![s("K-attn")],
            vec![s("K-gemv")],
            vec![s("M-1")],
            vec![s("M-3")],
        ],
    );
    rows(
        "MATCH (k:Kernel {name: 'K-attn'})<-[*1..2]-(x) RETURN x.name ORDER BY x.name",
        vec![
            vec![s("D-1")],
            vec![s("M-1")],
            vec![s("M-3")],
            vec![s("M-4")],
        ],
    );
}
