use super::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

#[test]
fn test_parse_simple_match() {
    let parser = CypherParser::new();
    let result = parser.parse("MATCH (n) RETURN n");
    assert!(result.is_ok(), "Failed: {:?}", result);

    if let Ok(Statement::Query(query)) = result {
        assert!(query.match_clause.is_some());
        assert_eq!(query.return_clause.items.len(), 1);
    }
}

#[test]
fn test_parse_match_with_label() {
    let parser = CypherParser::new();
    let result = parser.parse("MATCH (n:Person) RETURN n");
    assert!(result.is_ok(), "Failed: {:?}", result);

    if let Ok(Statement::Query(query)) = result {
        let match_clause = query.match_clause.unwrap();
        if let PatternElement::Node(node) = &match_clause.patterns[0].elements[0] {
            assert_eq!(node.labels[0], "Person");
        }
    }
}

#[test]
fn test_parse_match_with_properties() {
    let parser = CypherParser::new();
    let result = parser.parse("MATCH (n:Person {name: \"Alice\"}) RETURN n");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_parse_match_with_relationship() {
    let parser = CypherParser::new();
    let result = parser.parse("MATCH (a)-[r:KNOWS]->(b) RETURN a, b");
    assert!(result.is_ok(), "Failed: {:?}", result);

    if let Ok(Statement::Query(query)) = result {
        let match_clause = query.match_clause.unwrap();
        assert_eq!(match_clause.patterns[0].elements.len(), 3);
    }
}

#[test]
fn test_parse_where_clause() {
    let parser = CypherParser::new();
    let result = parser.parse("MATCH (n:Person) WHERE n.age > 25 RETURN n");
    assert!(result.is_ok(), "Failed: {:?}", result);

    if let Ok(Statement::Query(query)) = result {
        assert!(query.where_clause.is_some());
    }
}

#[test]
fn test_parse_order_by() {
    let parser = CypherParser::new();
    let result = parser.parse("MATCH (n) RETURN n ORDER BY n.name");
    assert!(result.is_ok(), "Failed: {:?}", result);

    if let Ok(Statement::Query(query)) = result {
        assert!(query.order_by.is_some());
    }
}

#[test]
fn test_parse_limit() {
    let parser = CypherParser::new();
    let result = parser.parse("MATCH (n) RETURN n LIMIT 10");
    assert!(result.is_ok(), "Failed: {:?}", result);

    if let Ok(Statement::Query(query)) = result {
        assert_eq!(query.limit, Some(10));
    }
}

#[test]
fn test_parse_create_node() {
    let parser = CypherParser::new();
    let result = parser.parse("CREATE (n:Person {name: \"Alice\"})");
    assert!(result.is_ok(), "Failed: {:?}", result);

    if let Ok(Statement::Create(create)) = result {
        assert_eq!(create.patterns.len(), 1);
    }
}

#[test]
fn test_parse_delete() {
    let parser = CypherParser::new();
    let result = parser.parse("DELETE n");
    assert!(result.is_ok(), "Failed: {:?}", result);

    if let Ok(Statement::Delete(delete)) = result {
        assert!(!delete.detach);
        assert_eq!(delete.variables.len(), 1);
    }
}

#[test]
fn test_parse_detach_delete() {
    let parser = CypherParser::new();
    let result = parser.parse("DETACH DELETE n");
    assert!(result.is_ok(), "Failed: {:?}", result);

    if let Ok(Statement::Delete(delete)) = result {
        assert!(delete.detach);
    }
}

#[test]
fn test_parse_return_alias() {
    let parser = CypherParser::new();
    let result = parser.parse("MATCH (n) RETURN n.name AS name");
    assert!(result.is_ok(), "Failed: {:?}", result);

    if let Ok(Statement::Query(query)) = result {
        assert!(query.return_clause.items[0].alias.is_some());
    }
}

#[test]
fn test_parse_complex_where() {
    let parser = CypherParser::new();
    let result = parser.parse("MATCH (n:Person) WHERE n.age > 25 AND n.name = \"Alice\" RETURN n");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_parse_function_call() {
    let parser = CypherParser::new();
    let result = parser.parse("MATCH (n) RETURN count(n)");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_parse_invalid_syntax() {
    let parser = CypherParser::new();
    let result = parser.parse("MATCH RETURN");
    assert!(result.is_err());
}
