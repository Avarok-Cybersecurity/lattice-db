//! Unsupported or invalid constructs fail closed with a message naming the construct.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

mod cypher_support;
use cypher_support::*;

#[test]
fn write_and_unsupported_clauses_are_named() {
    refused(
        "MERGE (n:Kernel {name: 'x'}) RETURN n",
        &["Unsupported clause", "MERGE"],
    );
    refused(
        "MATCH (n) SET n.x = 1 RETURN n",
        &["Unsupported clause", "SET"],
    );
    refused(
        "MATCH (n) REMOVE n.x RETURN n",
        &["Unsupported clause", "REMOVE"],
    );
    refused("CALL db.labels()", &["Unsupported clause", "CALL"]);
    refused(
        "MATCH (n) RETURN n UNION MATCH (m) RETURN m",
        &["Unsupported clause", "UNION"],
    );
    refused(
        "MATCH p = shortestPath((a)-[*]-(b)) RETURN p",
        &["Unsupported", "shortestPath"],
    );
}

#[test]
fn unsupported_expressions_are_named() {
    refused("RETURN 'abc' =~ 'a.*'", &["Unsupported feature", "=~"]);
    refused(
        "RETURN [x IN [1, 2] | x * 2]",
        &["Unsupported", "list comprehension"],
    );
    refused(
        "MATCH (a:Kernel)-[r*1..2]->(b) RETURN r",
        &["Unsupported feature", "variable-length relationship", "'r'"],
    );
    refused(
        "MATCH (a)-[r]->(b) RETURN id(r)",
        &["Unsupported feature", "id() of a relationship"],
    );
    refused("RETURN frobnicate(1)", &["Unknown function", "frobnicate"]);
}

#[test]
fn unknown_variables_are_refused_even_when_no_row_matches() {
    refused("MATCH (x:Nope) RETURN y", &["Unknown variable", "'y'"]);
    refused(
        "MATCH (x:Nope) WHERE y.a = 1 RETURN x",
        &["Unknown variable", "'y'"],
    );
    refused(
        "MATCH (m:Measurement) WITH m.name AS name RETURN m.tok_s",
        &["Unknown variable", "'m'"],
    );
    refused("RETURN $nope", &["parameter", "nope"]);
}

#[test]
fn projection_rules() {
    refused(
        "MATCH (m:Measurement) WITH m.name RETURN 1",
        &["WITH", "aliased"],
    );
    refused("RETURN 1 AS a, 2 AS a", &["Multiple result columns", "'a'"]);
    refused(
        "MATCH (m:Measurement) RETURN m.name + count(*)",
        &["grouping key", "'m'"],
    );
    refused("RETURN count(count(*))", &["nested aggregate"]);
    refused(
        "MATCH (m:Measurement) WHERE count(*) > 1 RETURN m",
        &["aggregate", "count", "WHERE"],
    );
    refused(
        "MATCH (m:Measurement) RETURN DISTINCT m.name ORDER BY m.tok_s",
        &["Unknown variable", "'m'"],
    );
}

#[test]
fn type_errors_are_refused() {
    refused("RETURN 1 AND true", &["AND", "BOOLEAN", "INTEGER"]);
    refused(
        "MATCH (m:Measurement) WHERE m.name RETURN m",
        &["WHERE", "BOOLEAN", "STRING"],
    );
    refused("RETURN -'a'", &["negation", "STRING"]);
    refused("RETURN 'a' - 1", &["STRING", "INTEGER"]);
    refused("UNWIND ['a'] AS x RETURN sum(x)", &["sum", "STRING"]);
    refused("RETURN [1]['a']", &["index", "STRING"]);
    refused("RETURN (1).x", &["property access", "INTEGER"]);
    refused("RETURN size(1, 2)", &["size", "argument"]);
}

#[test]
fn arithmetic_faults_are_refused() {
    refused("RETURN 9223372036854775807 + 1", &["overflow"]);
    refused("RETURN 1 / 0", &["Division by zero"]);
    refused("RETURN 1 % 0", &["Division by zero"]);
}
