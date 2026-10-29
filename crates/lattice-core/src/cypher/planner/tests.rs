use super::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

#[test]
fn test_plan_all_nodes_scan() {
    let planner = QueryPlanner::new();
    let query = Query::new(ReturnClause::new(vec![ProjectionItem::new(
        Expr::variable("n"),
    )]))
    .with_match(MatchClause::new(Pattern::node(NodePattern::with_variable(
        "n",
    ))));

    let plan = planner.plan(&Statement::Query(query)).unwrap();

    // Should be Project(AllNodesScan)
    assert!(matches!(plan, LogicalOp::Project { .. }));
}

#[test]
fn test_plan_label_scan() {
    let planner = QueryPlanner::new();
    let query = Query::new(ReturnClause::new(vec![ProjectionItem::new(
        Expr::variable("n"),
    )]))
    .with_match(MatchClause::new(Pattern::node(
        NodePattern::with_variable("n").with_label("Person"),
    )));

    let plan = planner.plan(&Statement::Query(query)).unwrap();

    // Verify it's a label scan inside the project
    if let LogicalOp::Project { input, .. } = plan {
        assert!(matches!(*input, LogicalOp::NodeByLabelScan { .. }));
    } else {
        panic!("Expected Project");
    }
}

#[test]
fn test_plan_with_filter() {
    let planner = QueryPlanner::new();
    let query = Query::new(ReturnClause::new(vec![ProjectionItem::new(
        Expr::variable("n"),
    )]))
    .with_match(
        MatchClause::new(Pattern::node(
            NodePattern::with_variable("n").with_label("Person"),
        ))
        .with_where(WhereClause::new(
            Expr::property(Expr::variable("n"), "age").gt(Expr::literal(25i64)),
        )),
    );

    let plan = planner.plan(&Statement::Query(query)).unwrap();

    // Verify the filter is pushed down into NodeByLabelScan
    if let LogicalOp::Project { input, .. } = plan {
        // With predicate pushdown, Filter is merged into NodeByLabelScan
        assert!(matches!(
            *input,
            LogicalOp::NodeByLabelScan {
                predicate: Some(_),
                ..
            }
        ));
    } else {
        panic!("Expected Project");
    }
}

#[test]
fn test_plan_with_limit() {
    let planner = QueryPlanner::new();
    let query = Query::new(ReturnClause::new(vec![ProjectionItem::new(
        Expr::variable("n"),
    )]))
    .with_match(MatchClause::new(Pattern::node(NodePattern::with_variable(
        "n",
    ))))
    .with_limit(10);

    let plan = planner.plan(&Statement::Query(query)).unwrap();

    // LIMIT applies after the projection (openCypher order: project, DISTINCT, ORDER BY,
    // SKIP, LIMIT)
    if let LogicalOp::Limit { input, count } = plan {
        assert_eq!(count, 10);
        assert!(matches!(*input, LogicalOp::Project { .. }));
    } else {
        panic!("Expected Limit over Project, got {plan:?}");
    }
}

#[test]
fn test_plan_create_node() {
    let planner = QueryPlanner::new();
    let create = CreateClause::new(Pattern::node(
        NodePattern::with_variable("n")
            .with_label("Person")
            .with_properties(MapLiteral::from_entries([("name", Expr::literal("Alice"))])),
    ));

    let plan = planner.plan(&Statement::Create(create)).unwrap();

    if let LogicalOp::CreateNode {
        labels, variable, ..
    } = plan
    {
        assert_eq!(labels[0], "Person");
        assert_eq!(variable, Some(String::from("n")));
    } else {
        panic!("Expected CreateNode");
    }
}

#[test]
fn test_plan_expand() {
    let planner = QueryPlanner::new();
    let query = Query::new(ReturnClause::new(vec![
        ProjectionItem::new(Expr::variable("a")),
        ProjectionItem::new(Expr::variable("b")),
    ]))
    .with_match(MatchClause::new(Pattern::chain(
        NodePattern::with_variable("a").with_label("Person"),
        RelPattern::with_type("KNOWS", Direction::Outgoing),
        NodePattern::with_variable("b").with_label("Person"),
    )));

    let plan = planner.plan(&Statement::Query(query)).unwrap();

    // Verify there's an expand operation
    fn has_expand(op: &LogicalOp) -> bool {
        match op {
            LogicalOp::Expand { .. } => true,
            LogicalOp::Project { input, .. } => has_expand(input),
            LogicalOp::Filter { input, .. } => has_expand(input),
            LogicalOp::Limit { input, .. } => has_expand(input),
            _ => false,
        }
    }

    assert!(has_expand(&plan), "Expected Expand in plan");
}

#[test]
fn a_pattern_ending_at_a_bound_node_is_planned_from_that_node() {
    // `k` is bound by the first MATCH; expanding from it avoids joining every Measurement
    // against every Kernel row and filtering afterwards.
    let plan = crate::cypher::handler::DefaultCypherHandler::new()
        .explain("MATCH (k:Kernel) OPTIONAL MATCH (m:Measurement)-[:MEASURES]->(k) RETURN k, m")
        .unwrap();
    assert!(!plan.contains("Cartesian"), "{plan}");
    assert!(plan.contains("Expand((k)<-[MEASURES]<-(m))"), "{plan}");
}
