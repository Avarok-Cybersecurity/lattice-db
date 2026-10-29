//! Scalar functions.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

mod cypher_support;
use cypher_support::*;
use lattice_core::types::value::CypherValue;

#[test]
fn labels_type_and_id() {
    row(
        "MATCH (k:Kernel {name: 'K-gemv'}) RETURN labels(k), id(k)",
        vec![strs(&["Kernel"]), i(1)],
    );
    rows(
        "MATCH (d:Decision)-[r]->(x) RETURN type(r), x.name ORDER BY x.name",
        vec![
            vec![s("ABOUT"), s("K-gemv")],
            vec![s("CITES"), s("M-1")],
            vec![s("CITES"), s("M-3")],
        ],
    );
    row("RETURN labels(null), type(null), id(null)", vec![NULL; 3]);
}

#[test]
fn keys_and_properties_hide_internal_fields() {
    row(
        "MATCH (m:Measurement {name: 'M-1'}) RETURN keys(m)",
        vec![strs(&["name", "tok_s", "verdict"])],
    );
    row(
        "MATCH (k:Kernel {name: 'K-gemv'}) RETURN properties(k)",
        vec![CypherValue::map_from([
            ("family", s("gemv")),
            ("name", s("K-gemv")),
        ])],
    );
    row("RETURN keys({b: 1, a: 2})", vec![strs(&["b", "a"])]);
    row(
        "MATCH (:Measurement {name: 'M-1'})-[r]->(:Kernel {name: 'K-attn'}) RETURN keys(r), properties(r)",
        vec![strs(&["weight"]), CypherValue::map_from([("weight", f(0.5))])],
    );
}

#[test]
fn size_and_length() {
    row(
        "RETURN size([1, 2, 3]), size('abcd'), size(null)",
        vec![i(3), i(4), NULL],
    );
    row("MATCH (d:Decision) RETURN size(d.tags)", vec![i(2)]);
    row("RETURN length([1, 2]), length('abc')", vec![i(2), i(3)]);
}

#[test]
fn exists_tests_for_a_value() {
    rows(
        "MATCH (k:Kernel) RETURN k.name, exists(k.family) ORDER BY k.name",
        vec![
            vec![s("K-attn"), b(true)],
            vec![s("K-gemv"), b(true)],
            vec![s("K-idle"), b(false)],
        ],
    );
}

#[test]
fn coalesce_returns_first_non_null() {
    rows(
        "MATCH (k:Kernel) RETURN coalesce(k.family, 'none') ORDER BY k.name",
        vec![vec![s("attn")], vec![s("gemv")], vec![s("none")]],
    );
    row(
        "RETURN coalesce(null, null), coalesce(null, 2, 3)",
        vec![NULL, i(2)],
    );
}

#[test]
fn conversions() {
    row(
        "RETURN toString(1), toString(2.5), toString(3.0), toString(true), toString('x'), toString(null)",
        vec![s("1"), s("2.5"), s("3.0"), s("true"), s("x"), NULL],
    );
    row(
        "RETURN toInteger('42'), toInteger(3.9), toInteger('x'), toInteger(null), toInteger('2.7'), toInteger(-3.9)",
        vec![i(42), i(3), NULL, NULL, i(2), i(-3)],
    );
    row(
        "RETURN toFloat('1.5'), toFloat(2), toFloat('x'), toFloat(null)",
        vec![f(1.5), f(2.0), NULL, NULL],
    );
    row(
        "RETURN toBoolean('true'), toBoolean('FALSE'), toBoolean('x'), toBoolean(true)",
        vec![b(true), b(false), NULL, b(true)],
    );
}

#[test]
fn string_functions() {
    row(
        "RETURN toLower('AbC'), toUpper('AbC'), toUpper(null), TOUPPER('a')",
        vec![s("abc"), s("ABC"), NULL, s("A")],
    );
    row(
        "RETURN trim('  a '), ltrim(' a'), rtrim('a '), replace('aXbX', 'X', '-'), \
                substring('kernel', 1, 3), substring('kernel', 2), split('a,b', ','), reverse('abc')",
        vec![
            s("a"),
            s("a"),
            s("a"),
            s("a-b-"),
            s("ern"),
            s("rnel"),
            strs(&["a", "b"]),
            s("cba"),
        ],
    );
}

#[test]
fn list_functions() {
    row(
        "RETURN head([1, 2, 3]), last([1, 2, 3]), head([]), last(null), tail([1, 2, 3]), reverse([1, 2])",
        vec![
            i(1),
            i(3),
            NULL,
            NULL,
            list(vec![i(2), i(3)]),
            list(vec![i(2), i(1)]),
        ],
    );
    row(
        "RETURN range(1, 3), range(0, 10, 5), range(3, 1, -1), Size([1])",
        vec![
            list(vec![i(1), i(2), i(3)]),
            list(vec![i(0), i(5), i(10)]),
            list(vec![i(3), i(2), i(1)]),
            i(1),
        ],
    );
}

#[test]
fn numeric_functions() {
    row(
        "RETURN abs(-3), abs(-2.5), ceil(1.2), floor(1.8), round(2.5), round(-2.4), sqrt(16), sign(-5), sign(0.5)",
        vec![
            i(3),
            f(2.5),
            f(2.0),
            f(1.0),
            f(3.0),
            f(-2.0),
            f(4.0),
            i(-1),
            i(1),
        ],
    );
}

#[test]
fn relationship_endpoints() {
    row(
        "MATCH (m {name: 'M-2'})-[r]->() RETURN startNode(r).name, endNode(r).name",
        vec![s("M-2"), s("K-gemv")],
    );
}
