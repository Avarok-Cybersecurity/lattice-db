//! Aggregate functions and implicit grouping.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

mod cypher_support;
use cypher_support::*;

const MEASURES: &str = "MATCH (m:Measurement)-[:MEASURES]->(k:Kernel)";

#[test]
fn count_star_counts_rows() {
    row("MATCH (m:Measurement) RETURN count(*)", vec![i(4)]);
}

#[test]
fn count_expression_skips_nulls() {
    row(
        "MATCH (m:Measurement) RETURN count(m.tok_s), count(m)",
        vec![i(3), i(4)],
    );
}

#[test]
fn count_distinct() {
    row(
        "MATCH (m:Measurement) RETURN count(DISTINCT m.verdict), count(DISTINCT m)",
        vec![i(2), i(4)],
    );
    row(
        &format!("{MEASURES} RETURN count(DISTINCT k), count(k)"),
        vec![i(2), i(5)],
    );
}

#[test]
fn implicit_group_by_on_non_aggregate_columns() {
    rows(
        &format!("{MEASURES} RETURN k.name, count(*) AS n ORDER BY k.name"),
        vec![vec![s("K-attn"), i(3)], vec![s("K-gemv"), i(2)]],
    );
}

#[test]
fn collect_in_input_order_per_group() {
    rows(
        &format!(
            "{MEASURES} WITH k, m ORDER BY m.name RETURN k.name, collect(m.name) ORDER BY k.name"
        ),
        vec![
            vec![s("K-attn"), strs(&["M-1", "M-3", "M-4"])],
            vec![s("K-gemv"), strs(&["M-1", "M-2"])],
        ],
    );
}

#[test]
fn collect_skips_nulls_and_supports_distinct() {
    row(
        "MATCH (m:Measurement) WITH m ORDER BY m.name RETURN collect(m.tok_s), collect(DISTINCT m.verdict)",
        vec![list(vec![i(10), i(20), f(30.5)]), strs(&["WIN", "LOSS"])],
    );
}

#[test]
fn min_max_sum_avg() {
    row(
        "MATCH (m:Measurement) RETURN min(m.tok_s), max(m.tok_s), sum(m.tok_s), avg(m.tok_s)",
        vec![i(10), f(30.5), f(60.5), f(60.5 / 3.0)],
    );
    row(
        "MATCH (m:Measurement) WHERE m.tok_s < 25 RETURN sum(m.tok_s), avg(m.tok_s)",
        vec![i(30), f(15.0)],
    );
    row(
        "MATCH (m:Measurement) RETURN min(m.name), max(m.name), sum(DISTINCT 1)",
        vec![s("M-1"), s("M-4"), i(1)],
    );
}

#[test]
fn aggregates_over_an_empty_input_with_no_grouping_keys_give_one_row() {
    row(
        "MATCH (m:Measurement) WHERE m.tok_s > 100 RETURN count(*), count(m.tok_s), collect(m.name), \
         sum(m.tok_s), avg(m.tok_s), min(m.tok_s), max(m.tok_s)",
        vec![i(0), i(0), list(vec![]), i(0), NULL, NULL, NULL],
    );
    row("MATCH (x:Nope) RETURN count(x)", vec![i(0)]);
}

#[test]
fn aggregates_over_an_empty_input_with_grouping_keys_give_no_rows() {
    rows(
        "MATCH (m:Measurement) WHERE m.tok_s > 100 RETURN m.verdict, count(*)",
        vec![],
    );
}

#[test]
fn null_is_a_grouping_key_of_its_own() {
    rows(
        "MATCH (m:Measurement) RETURN m.verdict AS v, count(*) AS n ORDER BY v",
        vec![
            vec![s("LOSS"), i(1)],
            vec![s("WIN"), i(2)],
            vec![NULL, i(1)],
        ],
    );
    rows(
        "MATCH (m:Measurement) RETURN m.tok_s IS NULL AS missing, count(*) ORDER BY missing",
        vec![vec![b(false), i(3)], vec![b(true), i(1)]],
    );
}

#[test]
fn aggregates_inside_expressions() {
    row(
        "MATCH (m:Measurement) RETURN count(*) * 10 + 1",
        vec![i(41)],
    );
    rows(
        &format!("{MEASURES} RETURN k.name, size(collect(m.name)) + 0 AS n ORDER BY k.name"),
        vec![vec![s("K-attn"), i(3)], vec![s("K-gemv"), i(2)]],
    );
    // A non-aggregate part of an aggregating item must be a projected grouping key.
    rows(
        &format!(
            "{MEASURES} RETURN k.name, k.name + ':' + toString(count(*)) AS label ORDER BY label"
        ),
        vec![
            vec![s("K-attn"), s("K-attn:3")],
            vec![s("K-gemv"), s("K-gemv:2")],
        ],
    );
}

#[test]
fn order_by_an_aggregate_or_its_alias() {
    rows(
        &format!("{MEASURES} RETURN k.name, count(*) ORDER BY count(*) DESC, k.name"),
        vec![vec![s("K-attn"), i(3)], vec![s("K-gemv"), i(2)]],
    );
    rows(
        &format!("{MEASURES} RETURN k.name AS kernel, count(*) AS n ORDER BY n LIMIT 1"),
        vec![vec![s("K-gemv"), i(2)]],
    );
}

#[test]
fn with_aggregation_then_where() {
    rows(
        &format!("{MEASURES} WITH k, count(m) AS n WHERE n > 2 RETURN k.name, n"),
        vec![vec![s("K-attn"), i(3)]],
    );
}

#[test]
fn optional_match_counts_zero_for_unmatched_rows() {
    rows(
        "MATCH (k:Kernel) OPTIONAL MATCH (m:Measurement)-[:MEASURES]->(k) \
         RETURN k.name, count(m) ORDER BY k.name",
        vec![
            vec![s("K-attn"), i(3)],
            vec![s("K-gemv"), i(2)],
            vec![s("K-idle"), i(0)],
        ],
    );
}
