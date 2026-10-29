use super::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

#[test]
fn test_node_pattern_builder() {
    let node = NodePattern::with_variable("n")
        .with_label("Person")
        .with_properties(MapLiteral::from_entries([("name", Expr::literal("Alice"))]));

    assert_eq!(node.variable, Some("n".to_string()));
    assert_eq!(node.labels.len(), 1);
    assert_eq!(node.labels[0], "Person");
    assert!(node.properties.is_some());
}

#[test]
fn test_rel_pattern_builder() {
    let rel = RelPattern::with_type("KNOWS", Direction::Outgoing)
        .with_variable("r")
        .with_length(PathLength::range(Some(1), Some(3)));

    assert_eq!(rel.variable, Some("r".to_string()));
    assert_eq!(rel.rel_types[0], "KNOWS");
    assert_eq!(rel.direction, Direction::Outgoing);
    assert!(rel.length.is_some());
}

#[test]
fn test_pattern_chain() {
    let pattern = Pattern::chain(
        NodePattern::with_variable("a").with_label("Person"),
        RelPattern::with_type("KNOWS", Direction::Outgoing),
        NodePattern::with_variable("b").with_label("Person"),
    );

    assert_eq!(pattern.elements.len(), 3);
    assert!(matches!(pattern.elements[0], PatternElement::Node(_)));
    assert!(matches!(
        pattern.elements[1],
        PatternElement::Relationship(_)
    ));
    assert!(matches!(pattern.elements[2], PatternElement::Node(_)));
}

#[test]
fn test_query_builder() {
    let query = Query::new(ReturnClause::new(vec![ProjectionItem::new(
        Expr::property(Expr::variable("n"), "name"),
    )
    .with_alias("name")]))
    .with_match(MatchClause::new(Pattern::node(
        NodePattern::with_variable("n").with_label("Person"),
    )))
    .with_where(WhereClause::new(
        Expr::property(Expr::variable("n"), "age").gt(Expr::literal(25i64)),
    ))
    .with_limit(10);

    assert!(query.match_clause.is_some());
    assert!(query.where_clause.is_some());
    assert_eq!(query.limit, Some(10));
}

#[test]
fn test_expr_shortcuts() {
    let expr = Expr::property(Expr::variable("n"), "age")
        .gt(Expr::literal(25i64))
        .and(Expr::property(Expr::variable("n"), "name").eq(Expr::literal("Alice")));

    assert!(matches!(
        expr,
        Expr::BinaryOp {
            op: BinaryOp::And,
            ..
        }
    ));
}

#[test]
fn test_direction() {
    assert_eq!(Direction::Outgoing, Direction::Outgoing);
    assert_ne!(Direction::Incoming, Direction::Outgoing);
}
