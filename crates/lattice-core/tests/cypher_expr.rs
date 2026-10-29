//! Expression evaluation: operators, null semantics, CASE, collections, parameters.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

mod cypher_support;
use cypher_support::*;
use lattice_core::types::value::CypherValue;
use std::collections::HashMap;

#[test]
fn arithmetic_operators_and_precedence() {
    row(
        "RETURN 7 - 2, 7 * 3, 7 / 2, 7 % 4, 2 ^ 3, -(3), 7.0 / 2, 1 + 2.5",
        vec![i(5), i(21), i(3), i(3), f(8.0), i(-3), f(3.5), f(3.5)],
    );
    row(
        "RETURN 1 + 2 * 3 - 4 / 2, (1 + 2) * 3, 10 - 3 - 2, 2 * 3 % 4",
        vec![i(5), i(9), i(5), i(2)],
    );
}

#[test]
fn arithmetic_on_properties() {
    rows(
        "MATCH (m:Measurement) WHERE m.tok_s IS NOT NULL RETURN m.tok_s - 1 AS x ORDER BY x",
        vec![vec![i(9)], vec![i(19)], vec![f(29.5)]],
    );
}

#[test]
fn plus_concatenates_strings_and_lists() {
    row(
        "RETURN 'a' + 'b', 'n' + 1, [1] + [2, 3], [1] + 4",
        vec![
            s("ab"),
            s("n1"),
            list(vec![i(1), i(2), i(3)]),
            list(vec![i(1), i(4)]),
        ],
    );
}

#[test]
fn null_propagates_through_arithmetic() {
    row(
        "RETURN null + 1, 1 - null, null * 2, 'a' + null",
        vec![NULL; 4],
    );
}

#[test]
fn comparison_operators() {
    row(
        "RETURN 1 < 2, 2 <= 1, 'a' < 'b', 1 = 1.0, 1 <> 2, [1, 2] = [1, 2], 2 >= 2, 3 > 4",
        vec![
            b(true),
            b(false),
            b(true),
            b(true),
            b(true),
            b(true),
            b(true),
            b(false),
        ],
    );
    row("RETURN 1 = 'a', 1 < 'a'", vec![b(false), NULL]);
    row("RETURN 1 < 2 < 3, 3 > 2 > 2", vec![b(true), b(false)]);
}

#[test]
fn comparison_with_null_is_null() {
    row(
        "RETURN null = null, 1 = null, null < 1, null <> 1",
        vec![NULL; 4],
    );
}

#[test]
fn three_valued_boolean_logic() {
    row(
        "RETURN true AND null, false AND null, true OR null, false OR null, NOT null, null XOR true",
        vec![NULL, b(false), b(true), NULL, NULL, NULL],
    );
    row(
        "RETURN true XOR false, true XOR true",
        vec![b(true), b(false)],
    );
}

#[test]
fn where_drops_rows_whose_predicate_is_null() {
    // M-4 has no tok_s: `null < 15` is null, so the row is not returned.
    rows(
        "MATCH (m:Measurement) WHERE m.tok_s < 15 RETURN m.name",
        vec![vec![s("M-1")]],
    );
    // ... and NOT null is still null.
    rows(
        "MATCH (m:Measurement) WHERE NOT m.tok_s > 15 RETURN m.name",
        vec![vec![s("M-1")]],
    );
}

#[test]
fn in_operator_with_nulls() {
    row(
        "RETURN 2 IN [1, 2, 3], 5 IN [1, 2], null IN [1], 1 IN [null, 1], 2 IN [null, 1], 1 IN [], null IN []",
        vec![b(true), b(false), NULL, b(true), NULL, b(false), b(false)],
    );
    rows(
        "MATCH (m:Measurement) WHERE m.name IN ['M-2', 'M-4'] RETURN m.name ORDER BY m.name",
        vec![vec![s("M-2")], vec![s("M-4")]],
    );
}

#[test]
fn string_predicates() {
    row(
        "RETURN 'kernel' STARTS WITH 'ker', 'kernel' ENDS WITH 'nel', 'kernel' CONTAINS 'rn', 'kernel' CONTAINS 'x'",
        vec![b(true), b(true), b(true), b(false)],
    );
    row(
        "RETURN null STARTS WITH 'a', 'a' CONTAINS null, 1 CONTAINS 'a'",
        vec![NULL; 3],
    );
    rows(
        "MATCH (k:Kernel) WHERE k.name CONTAINS 'att' OR k.name ENDS WITH 'dle' RETURN k.name ORDER BY k.name",
        vec![vec![s("K-attn")], vec![s("K-idle")]],
    );
}

#[test]
fn is_null_and_is_not_null() {
    row(
        "RETURN null IS NULL, 1 IS NULL, 1 IS NOT NULL",
        vec![b(true), b(false), b(true)],
    );
    rows(
        "MATCH (k:Kernel) WHERE k.family IS NULL AND k.name = 'K-idle' RETURN k.name",
        vec![vec![s("K-idle")]],
    );
    rows(
        "MATCH (k:Kernel) WHERE k.family IS NOT NULL RETURN k.name ORDER BY k.name",
        vec![vec![s("K-attn")], vec![s("K-gemv")]],
    );
}

#[test]
fn case_expressions() {
    row(
        "RETURN CASE WHEN 1 > 2 THEN 'a' WHEN 2 > 1 THEN 'b' ELSE 'c' END, \
                CASE 2 WHEN 1 THEN 'one' WHEN 2 THEN 'two' END, \
                CASE 3 WHEN 1 THEN 'x' END, \
                CASE null WHEN null THEN 'y' ELSE 'n' END, \
                CASE WHEN null THEN 1 ELSE 2 END",
        vec![s("b"), s("two"), NULL, s("n"), i(2)],
    );
    rows(
        "MATCH (m:Measurement) RETURN m.name, CASE WHEN m.tok_s >= 20 THEN 'fast' \
         WHEN m.tok_s < 20 THEN 'slow' ELSE 'unknown' END AS speed ORDER BY m.name",
        vec![
            vec![s("M-1"), s("slow")],
            vec![s("M-2"), s("fast")],
            vec![s("M-3"), s("fast")],
            vec![s("M-4"), s("unknown")],
        ],
    );
}

#[test]
fn list_and_map_literals_and_indexing() {
    row(
        "RETURN [1, 2, 3][0], [1, 2, 3][-1], [1, 2, 3][5], [1, 2, 3][1..], [1, 2, 3][..-1], \
                {a: 1, b: 'x'}.b, {a: 1}['a'], {a: 1}.missing, [[1, 2], [3]][0][1]",
        vec![
            i(1),
            i(3),
            NULL,
            list(vec![i(2), i(3)]),
            list(vec![i(1), i(2)]),
            s("x"),
            i(1),
            NULL,
            i(2),
        ],
    );
    row(
        "RETURN {a: 1, b: [true]}",
        vec![CypherValue::map_from([
            ("a", i(1)),
            ("b", list(vec![b(true)])),
        ])],
    );
    rows(
        "MATCH (d:Decision) RETURN d.tags[0], d['name']",
        vec![vec![s("a"), s("D-1")]],
    );
}

#[test]
fn parameters() {
    let mut p = HashMap::new();
    p.insert("x".to_string(), i(41));
    p.insert("v".to_string(), s("WIN"));
    let (_, got) = run_with(&mut kg(), "RETURN $x + 1, $v", p.clone());
    assert_eq!(got, vec![vec![i(42), s("WIN")]]);
    let (_, got) = run_with(
        &mut kg(),
        "MATCH (m:Measurement) WHERE m.verdict = $v RETURN m.name ORDER BY m.name",
        p,
    );
    assert_eq!(got, vec![vec![s("M-1")], vec![s("M-3")]]);
}

#[test]
fn property_access_on_relationship_and_map_bindings() {
    rows(
        "MATCH (m:Measurement)-[r:MEASURES]->(k:Kernel) WHERE m.name = 'M-1' \
         RETURN k.name, r.weight, type(r) ORDER BY k.name",
        vec![
            vec![s("K-attn"), f(0.5), s("MEASURES")],
            vec![s("K-gemv"), f(1.0), s("MEASURES")],
        ],
    );
    row("WITH {a: 1} AS m RETURN m.a, m.nope", vec![i(1), NULL]);
}

#[test]
fn label_predicate_in_where() {
    rows(
        "MATCH (n) WHERE n:Kernel RETURN n.name ORDER BY n.name",
        vec![vec![s("K-attn")], vec![s("K-gemv")], vec![s("K-idle")]],
    );
}

#[test]
fn unaliased_columns_are_named_by_their_expression() {
    let (cols, _) = run(
        &mut kg(),
        "MATCH (m:Measurement) RETURN m.name, count(*), m.tok_s + 1 AS t ORDER BY m.name",
    );
    assert_eq!(cols, vec!["m.name", "count(*)", "t"]);
}
