//! Query planner - converts Cypher AST to logical operations
//!
//! The planner transforms the parsed AST into a logical execution plan
//! that can be executed by the executor.

use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};
// compact_str removed - using standard String
use serde::{Deserialize, Serialize};
// smallvec removed - using standard Vec

/// Maximum allowed path traversal depth (DoS protection)
const MAX_PATH_LENGTH: u32 = 100;

/// Logical operation in the query plan
///
/// These operations form a tree that represents the logical execution plan.
/// The executor traverses this tree to produce results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LogicalOp {
    // === Scan Operators ===
    /// Scan all nodes in the collection
    AllNodesScan { variable: String },

    /// Scan nodes with a specific label (with optional predicate pushdown)
    NodeByLabelScan {
        variable: String,
        label: String,
        /// Optional predicate pushed down from Filter for early filtering
        predicate: Option<Box<Expr>>,
    },

    /// Seek nodes by ID
    NodeByIdSeek { variable: String, ids: Vec<u64> },

    // === Expand Operators ===
    /// Expand from a node along relationships
    Expand {
        input: Box<LogicalOp>,
        from: String,
        rel_variable: Option<String>,
        to: String,
        rel_types: Vec<String>,
        direction: Direction,
        min_hops: u32,
        max_hops: u32,
    },

    /// Optional expand (for OPTIONAL MATCH)
    OptionalExpand {
        input: Box<LogicalOp>,
        from: String,
        rel_variable: Option<String>,
        to: String,
        rel_types: Vec<String>,
        direction: Direction,
    },

    // === Filter/Transform ===
    /// Filter rows based on a predicate
    Filter {
        input: Box<LogicalOp>,
        predicate: Expr,
    },

    /// Project specific columns
    Project {
        input: Box<LogicalOp>,
        items: Vec<ProjectionItem>,
    },

    /// Sort rows (with optional limit for partial sort optimization)
    Sort {
        input: Box<LogicalOp>,
        items: Vec<OrderByItem>,
        /// Optional limit for partial sort optimization (O(n) instead of O(n log n))
        limit: Option<u64>,
    },

    /// Skip first N rows
    Skip { input: Box<LogicalOp>, count: u64 },

    /// Limit to N rows
    Limit { input: Box<LogicalOp>, count: u64 },

    /// Remove duplicate rows
    Distinct { input: Box<LogicalOp> },

    // === Mutation Operators ===
    /// Create a new node
    CreateNode {
        labels: Vec<String>,
        properties: MapLiteral,
        variable: Option<String>,
    },

    /// Create a new relationship
    CreateRelationship {
        from: String,
        to: String,
        rel_type: String,
        properties: MapLiteral,
        variable: Option<String>,
    },

    /// Delete a node
    DeleteNode {
        input: Box<LogicalOp>,
        variable: String,
        detach: bool,
    },

    /// Delete a relationship
    DeleteRelationship {
        input: Box<LogicalOp>,
        variable: String,
    },

    // === Utility ===
    /// Empty result set
    Empty,

    /// Produce a single row with no columns (for CREATE without MATCH)
    SingleRow,

    /// Apply property filter on a scan
    NodeByPropertySeek {
        variable: String,
        label: Option<String>,
        property: String,
        value: Expr,
    },
}

/// Query planner
///
/// Converts Cypher AST to a logical execution plan.
pub struct QueryPlanner;

impl QueryPlanner {
    /// Create a new planner
    pub fn new() -> Self {
        Self
    }

    /// Plan a statement
    pub fn plan(&self, stmt: &Statement) -> CypherResult<LogicalOp> {
        let plan = match stmt {
            Statement::Query(query) => self.plan_query(query),
            Statement::Create(create) => self.plan_create(create),
            Statement::Delete(delete) => self.plan_delete(delete, None),
            Statement::ReadWrite { query, mutations } => self.plan_read_write(query, mutations),
        }?;

        // Apply optimizations
        let optimized = self.optimize_predicate_pushdown(plan);
        Ok(optimized)
    }

    /// Plan a query
    fn plan_query(&self, query: &Query) -> CypherResult<LogicalOp> {
        // Start with the MATCH clause or a single row
        let mut plan = if let Some(match_clause) = &query.match_clause {
            self.plan_match(match_clause)?
        } else {
            LogicalOp::SingleRow
        };

        // Apply OPTIONAL MATCH clauses
        for optional in &query.optional_matches {
            plan = self.plan_optional_match(plan, optional)?;
        }

        // Apply WHERE clause
        if let Some(where_clause) = &query.where_clause {
            plan = LogicalOp::Filter {
                input: Box::new(plan),
                predicate: where_clause.predicate.clone(),
            };
        }

        // Apply ORDER BY (with optional limit for partial sort optimization)
        // When there's no SKIP, we can pass the LIMIT to Sort for O(n) partial sort
        let sort_limit = if query.skip.is_none() {
            query.limit
        } else {
            None
        };
        if let Some(order_by) = &query.order_by {
            plan = LogicalOp::Sort {
                input: Box::new(plan),
                items: order_by.items.clone(),
                limit: sort_limit,
            };
        }

        // Apply SKIP
        if let Some(skip) = query.skip {
            plan = LogicalOp::Skip {
                input: Box::new(plan),
                count: skip,
            };
        }

        // Apply LIMIT (even if Sort already handled it, keeps semantics correct)
        if let Some(limit) = query.limit {
            plan = LogicalOp::Limit {
                input: Box::new(plan),
                count: limit,
            };
        }

        // Apply RETURN clause (projection)
        if query.return_clause.distinct {
            plan = LogicalOp::Distinct {
                input: Box::new(plan),
            };
        }

        plan = LogicalOp::Project {
            input: Box::new(plan),
            items: query.return_clause.items.clone(),
        };

        Ok(plan)
    }

    /// Plan a MATCH clause
    fn plan_match(&self, match_clause: &MatchClause) -> CypherResult<LogicalOp> {
        if match_clause.patterns.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty MATCH clause"));
        }

        // Limitation: only single-pattern MATCH is supported.
        // Multi-pattern MATCH (Cartesian product) requires cross-join planning.
        if match_clause.patterns.len() > 1 {
            return Err(CypherError::unsupported(
                "Multiple MATCH patterns (Cartesian product) not yet supported",
            ));
        }
        let pattern = &match_clause.patterns[0];
        self.plan_pattern(pattern)
    }

    /// Plan an OPTIONAL MATCH clause
    fn plan_optional_match(
        &self,
        input: LogicalOp,
        match_clause: &MatchClause,
    ) -> CypherResult<LogicalOp> {
        // Limitation: OPTIONAL MATCH currently behaves like MATCH (inner join).
        // Proper left-outer-join semantics (NULL rows for non-matches) not yet implemented.

        if match_clause.patterns.is_empty() {
            return Ok(input);
        }

        let pattern = &match_clause.patterns[0];

        // Only support patterns that start with an existing variable
        if let Some(PatternElement::Node(node)) = pattern.elements.first() {
            if let Some(ref _var) = node.variable {
                // Expand from existing variable
                return self.plan_pattern_expansion(input, pattern);
            }
        }

        // For standalone patterns, just plan them
        self.plan_pattern(pattern)
    }

    /// Plan a single pattern
    fn plan_pattern(&self, pattern: &Pattern) -> CypherResult<LogicalOp> {
        if pattern.elements.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty pattern"));
        }

        // Get the first node
        let first_node = match &pattern.elements[0] {
            PatternElement::Node(n) => n,
            _ => return Err(CypherError::syntax(0, 0, "Pattern must start with a node")),
        };

        // Plan the initial scan
        let mut plan = self.plan_node_scan(first_node)?;

        // Process remaining elements (rel-node pairs)
        let mut i = 1;
        while i < pattern.elements.len() {
            if i + 1 >= pattern.elements.len() {
                return Err(CypherError::syntax(
                    0,
                    0,
                    "Incomplete pattern: relationship without target node",
                ));
            }

            let rel = match &pattern.elements[i] {
                PatternElement::Relationship(r) => r,
                _ => {
                    return Err(CypherError::syntax(
                        0,
                        0,
                        "Expected relationship after node",
                    ))
                }
            };

            let target_node = match &pattern.elements[i + 1] {
                PatternElement::Node(n) => n,
                _ => {
                    return Err(CypherError::syntax(
                        0,
                        0,
                        "Expected node after relationship",
                    ))
                }
            };

            // Get the source variable name
            let from_var = self.get_last_node_variable(&plan)?;

            // Get or generate target variable
            let to_var = target_node
                .variable
                .clone()
                .unwrap_or_else(|| String::from(format!("_anon_{}", i)));

            // Determine hop range (capped at MAX_PATH_LENGTH for DoS protection)
            let (min_hops, max_hops) = if let Some(len) = &rel.length {
                let min = len.min.unwrap_or(1);
                let max = len.max.unwrap_or(MAX_PATH_LENGTH).min(MAX_PATH_LENGTH);
                (min, max)
            } else {
                (1, 1)
            };

            // Create expand operation
            plan = LogicalOp::Expand {
                input: Box::new(plan),
                from: from_var,
                rel_variable: rel.variable.clone(),
                to: to_var.clone(),
                rel_types: rel.rel_types.clone(),
                direction: rel.direction,
                min_hops,
                max_hops,
            };

            // Apply target node constraints (labels, properties)
            if !target_node.labels.is_empty() || target_node.properties.is_some() {
                let filter = self.build_node_filter(&to_var, target_node)?;
                plan = LogicalOp::Filter {
                    input: Box::new(plan),
                    predicate: filter,
                };
            }

            i += 2;
        }

        Ok(plan)
    }

    /// Plan pattern expansion from existing input
    fn plan_pattern_expansion(
        &self,
        input: LogicalOp,
        pattern: &Pattern,
    ) -> CypherResult<LogicalOp> {
        // Similar to plan_pattern but starts from existing input
        // Skip the first node if it's a variable reference

        if pattern.elements.len() < 3 {
            return Ok(input);
        }

        let first_node = match &pattern.elements[0] {
            PatternElement::Node(n) => n,
            _ => return Err(CypherError::syntax(0, 0, "Pattern must start with a node")),
        };

        let from_var = first_node
            .variable
            .clone()
            .ok_or_else(|| CypherError::syntax(0, 0, "OPTIONAL MATCH requires bound variable"))?;

        let mut plan = input;
        let mut i = 1;

        while i < pattern.elements.len() {
            if i + 1 >= pattern.elements.len() {
                break;
            }

            let rel = match &pattern.elements[i] {
                PatternElement::Relationship(r) => r,
                _ => break,
            };

            let target_node = match &pattern.elements[i + 1] {
                PatternElement::Node(n) => n,
                _ => break,
            };

            let to_var = target_node
                .variable
                .clone()
                .unwrap_or_else(|| String::from(format!("_anon_{}", i)));

            // Compute from variable before moving plan
            let from = if i == 1 {
                from_var.clone()
            } else {
                self.get_last_node_variable(&plan)?
            };

            plan = LogicalOp::OptionalExpand {
                input: Box::new(plan),
                from,
                rel_variable: rel.variable.clone(),
                to: to_var.clone(),
                rel_types: rel.rel_types.clone(),
                direction: rel.direction,
            };

            i += 2;
        }

        Ok(plan)
    }

    /// Plan a node scan
    fn plan_node_scan(&self, node: &NodePattern) -> CypherResult<LogicalOp> {
        let variable = node
            .variable
            .clone()
            .unwrap_or_else(|| String::from("_anon_0"));

        // Choose scan type based on constraints
        let mut plan = if node.labels.is_empty() {
            LogicalOp::AllNodesScan {
                variable: variable.clone(),
            }
        } else {
            // Use label scan for first label
            LogicalOp::NodeByLabelScan {
                variable: variable.clone(),
                label: node.labels[0].clone(),
                predicate: None,
            }
        };

        // Add filter for additional labels
        if node.labels.len() > 1 {
            let mut predicates = Vec::new();
            for label in node.labels.iter().skip(1) {
                // Check if node has this label
                predicates.push(Expr::function(
                    "hasLabel",
                    vec![
                        Expr::variable(variable.clone()),
                        Expr::literal(label.as_str()),
                    ],
                ));
            }
            // SAFETY: predicates is non-empty because labels.len() > 1 and skip(1) yields at least 1
            if let Some(combined) = predicates.into_iter().reduce(|a, b| a.and(b)) {
                plan = LogicalOp::Filter {
                    input: Box::new(plan),
                    predicate: combined,
                };
            }
        }

        // Add property filter
        if let Some(props) = &node.properties {
            let filter = self.build_property_filter(&variable, props)?;
            plan = LogicalOp::Filter {
                input: Box::new(plan),
                predicate: filter,
            };
        }

        Ok(plan)
    }

    /// Build a filter expression for node constraints
    fn build_node_filter(&self, variable: &str, node: &NodePattern) -> CypherResult<Expr> {
        let mut predicates = Vec::new();

        // Label checks
        for label in &node.labels {
            predicates.push(Expr::function(
                "hasLabel",
                vec![Expr::variable(variable), Expr::literal(label.as_str())],
            ));
        }

        // Property checks
        if let Some(props) = &node.properties {
            for (key, value) in &props.entries {
                predicates.push(
                    Expr::property(Expr::variable(variable), key.as_str()).eq((**value).clone()),
                );
            }
        }

        // Combine predicates with AND
        predicates
            .into_iter()
            .reduce(|a, b| a.and(b))
            .ok_or_else(|| CypherError::internal("Empty filter"))
    }

    /// Build a filter expression for property constraints
    fn build_property_filter(&self, variable: &str, props: &MapLiteral) -> CypherResult<Expr> {
        let mut predicates = Vec::new();

        for (key, value) in &props.entries {
            predicates
                .push(Expr::property(Expr::variable(variable), key.as_str()).eq((**value).clone()));
        }

        predicates
            .into_iter()
            .reduce(|a, b| a.and(b))
            .ok_or_else(|| CypherError::internal("Empty property filter"))
    }

    /// Get the variable name of the last node in a plan
    fn get_last_node_variable(&self, plan: &LogicalOp) -> CypherResult<String> {
        match plan {
            LogicalOp::AllNodesScan { variable } => Ok(variable.clone()),
            LogicalOp::NodeByLabelScan { variable, .. } => Ok(variable.clone()),
            LogicalOp::NodeByIdSeek { variable, .. } => Ok(variable.clone()),
            LogicalOp::NodeByPropertySeek { variable, .. } => Ok(variable.clone()),
            LogicalOp::Expand { to, .. } => Ok(to.clone()),
            LogicalOp::OptionalExpand { to, .. } => Ok(to.clone()),
            LogicalOp::Filter { input, .. } => self.get_last_node_variable(input),
            LogicalOp::Project { input, .. } => self.get_last_node_variable(input),
            LogicalOp::Sort { input, .. } => self.get_last_node_variable(input),
            LogicalOp::Skip { input, .. } => self.get_last_node_variable(input),
            LogicalOp::Limit { input, .. } => self.get_last_node_variable(input),
            LogicalOp::Distinct { input } => self.get_last_node_variable(input),
            _ => Err(CypherError::internal("Cannot determine node variable")),
        }
    }

    /// Plan a CREATE statement
    fn plan_create(&self, create: &CreateClause) -> CypherResult<LogicalOp> {
        if create.patterns.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty CREATE clause"));
        }

        // For now, support creating a single node or node-relationship-node
        let pattern = &create.patterns[0];

        if pattern.elements.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty pattern in CREATE"));
        }

        // Create single node
        if pattern.elements.len() == 1 {
            if let PatternElement::Node(node) = &pattern.elements[0] {
                return Ok(LogicalOp::CreateNode {
                    labels: node.labels.clone(),
                    properties: node.properties.clone().unwrap_or_default(),
                    variable: node.variable.clone(),
                });
            }
        }

        // Create node-relationship-node
        if pattern.elements.len() == 3 {
            let start_node = match &pattern.elements[0] {
                PatternElement::Node(n) => n,
                _ => return Err(CypherError::syntax(0, 0, "Expected node")),
            };
            let rel = match &pattern.elements[1] {
                PatternElement::Relationship(r) => r,
                _ => return Err(CypherError::syntax(0, 0, "Expected relationship")),
            };
            let end_node = match &pattern.elements[2] {
                PatternElement::Node(n) => n,
                _ => return Err(CypherError::syntax(0, 0, "Expected node")),
            };

            // For relationship creation, we need both nodes to have variables
            let from_var = start_node.variable.clone().ok_or_else(|| {
                CypherError::syntax(
                    0,
                    0,
                    "Start node must have a variable for relationship creation",
                )
            })?;
            let to_var = end_node.variable.clone().ok_or_else(|| {
                CypherError::syntax(
                    0,
                    0,
                    "End node must have a variable for relationship creation",
                )
            })?;

            // Get relationship type
            let rel_type = rel
                .rel_types
                .first()
                .ok_or_else(|| CypherError::syntax(0, 0, "Relationship must have a type"))?
                .clone();

            return Ok(LogicalOp::CreateRelationship {
                from: from_var,
                to: to_var,
                rel_type,
                properties: rel.properties.clone().unwrap_or_default(),
                variable: rel.variable.clone(),
            });
        }

        Err(CypherError::unsupported("Complex CREATE patterns"))
    }

    /// Plan a DELETE statement
    fn plan_delete(
        &self,
        delete: &DeleteClause,
        input: Option<LogicalOp>,
    ) -> CypherResult<LogicalOp> {
        let input = input.unwrap_or(LogicalOp::SingleRow);

        // For each variable, create a delete operation
        // For simplicity, we'll delete the first variable as a node
        if let Some(var) = delete.variables.first() {
            Ok(LogicalOp::DeleteNode {
                input: Box::new(input),
                variable: var.clone(),
                detach: delete.detach,
            })
        } else {
            Err(CypherError::syntax(
                0,
                0,
                "DELETE requires at least one variable",
            ))
        }
    }

    /// Plan a query with mutations
    fn plan_read_write(&self, query: &Query, mutations: &[Mutation]) -> CypherResult<LogicalOp> {
        // First plan the query part
        let mut plan = self.plan_query(query)?;

        // Then apply mutations
        for mutation in mutations {
            plan = match mutation {
                Mutation::Create(create) => {
                    // Limitation: CREATE after MATCH currently replaces the plan
                    // rather than sequencing operations. Multi-clause mutations
                    // with data dependencies may produce incorrect results.
                    let create_plan = self.plan_create(create)?;
                    create_plan
                }
                Mutation::Delete(delete) => self.plan_delete(delete, Some(plan))?,
                Mutation::Set(_set) => {
                    return Err(CypherError::unsupported("SET clause"));
                }
            };
        }

        Ok(plan)
    }

    /// Optimize the plan by pushing predicates down into scans
    fn optimize_predicate_pushdown(&self, plan: LogicalOp) -> LogicalOp {
        match plan {
            // Filter over NodeByLabelScan - push predicate down
            LogicalOp::Filter { input, predicate } => {
                match *input {
                    LogicalOp::NodeByLabelScan {
                        variable,
                        label,
                        predicate: None,
                    } => {
                        // Push predicate into the scan
                        LogicalOp::NodeByLabelScan {
                            variable,
                            label,
                            predicate: Some(Box::new(predicate)),
                        }
                    }
                    // Recursively optimize the input
                    other => LogicalOp::Filter {
                        input: Box::new(self.optimize_predicate_pushdown(other)),
                        predicate,
                    },
                }
            }
            // Recursively optimize other operators
            LogicalOp::Project { input, items } => LogicalOp::Project {
                input: Box::new(self.optimize_predicate_pushdown(*input)),
                items,
            },
            LogicalOp::Sort {
                input,
                items,
                limit,
            } => LogicalOp::Sort {
                input: Box::new(self.optimize_predicate_pushdown(*input)),
                items,
                limit,
            },
            LogicalOp::Skip { input, count } => LogicalOp::Skip {
                input: Box::new(self.optimize_predicate_pushdown(*input)),
                count,
            },
            LogicalOp::Limit { input, count } => LogicalOp::Limit {
                input: Box::new(self.optimize_predicate_pushdown(*input)),
                count,
            },
            LogicalOp::Distinct { input } => LogicalOp::Distinct {
                input: Box::new(self.optimize_predicate_pushdown(*input)),
            },
            // Other operators pass through unchanged
            other => other,
        }
    }
}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
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
        .with_match(MatchClause::new(Pattern::node(
            NodePattern::with_variable("n").with_label("Person"),
        )))
        .with_where(WhereClause::new(
            Expr::property(Expr::variable("n"), "age").gt(Expr::literal(25i64)),
        ));

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

        // Verify there's a limit in the plan
        if let LogicalOp::Project { input, .. } = plan {
            if let LogicalOp::Limit { count, .. } = *input {
                assert_eq!(count, 10);
            } else {
                panic!("Expected Limit");
            }
        } else {
            panic!("Expected Project");
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
}
