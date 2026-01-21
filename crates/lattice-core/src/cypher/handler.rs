//! Cypher query handler - main entry point for query execution
//!
//! The handler provides a clean abstraction over the parsing, planning,
//! and execution pipeline.

use crate::cypher::ast::Statement;
use crate::cypher::error::CypherResult;
use crate::cypher::executor::{ExecutionContext, QueryExecutor, QueryResult};
use crate::cypher::parser::CypherParser;
use crate::cypher::planner::{LogicalOp, QueryPlanner};
use crate::engine::collection::CollectionEngine;
use crate::types::value::CypherValue;
use std::collections::HashMap;

/// Core query handler trait
///
/// Implement this trait for different backends or to customize
/// the query execution pipeline.
pub trait CypherHandler: Send + Sync {
    /// Parse a Cypher query string into an AST
    fn parse(&self, query: &str) -> CypherResult<Statement>;

    /// Plan a statement into a logical operation tree
    fn plan(&self, stmt: &Statement) -> CypherResult<LogicalOp>;

    /// Execute a logical plan against a collection
    fn execute(
        &self,
        plan: &LogicalOp,
        collection: &mut CollectionEngine,
        parameters: HashMap<String, CypherValue>,
    ) -> CypherResult<QueryResult>;

    /// Convenience method: parse, plan, and execute a query
    fn query(
        &self,
        query: &str,
        collection: &mut CollectionEngine,
        parameters: HashMap<String, CypherValue>,
    ) -> CypherResult<QueryResult> {
        let stmt = self.parse(query)?;
        let plan = self.plan(&stmt)?;
        self.execute(&plan, collection, parameters)
    }
}

/// Default Cypher handler implementation
///
/// Uses the pest parser, standard planner, and standard executor.
///
/// # Example
///
/// ```ignore
/// use lattice_core::cypher::{DefaultCypherHandler, CypherHandler};
///
/// let handler = DefaultCypherHandler::new();
///
/// // Parse and plan (no execution)
/// let stmt = handler.parse("MATCH (n:Person) RETURN n.name")?;
/// let plan = handler.plan(&stmt)?;
///
/// // Or execute directly
/// let result = handler.query(
///     "MATCH (n:Person) WHERE n.age > 25 RETURN n.name",
///     &mut collection,
///     HashMap::new(),
/// )?;
/// ```
pub struct DefaultCypherHandler {
    parser: CypherParser,
    planner: QueryPlanner,
    executor: QueryExecutor,
}

impl DefaultCypherHandler {
    /// Create a new default handler
    pub fn new() -> Self {
        Self {
            parser: CypherParser::new(),
            planner: QueryPlanner::new(),
            executor: QueryExecutor::new(),
        }
    }

    /// Get a reference to the parser
    pub fn parser(&self) -> &CypherParser {
        &self.parser
    }

    /// Get a reference to the planner
    pub fn planner(&self) -> &QueryPlanner {
        &self.planner
    }

    /// Get a reference to the executor
    pub fn executor(&self) -> &QueryExecutor {
        &self.executor
    }

    /// Validate a query without executing it
    ///
    /// Returns the logical plan if the query is valid.
    pub fn validate(&self, query: &str) -> CypherResult<LogicalOp> {
        let stmt = self.parser.parse(query)?;
        self.planner.plan(&stmt)
    }

    /// Explain a query plan
    ///
    /// Returns a human-readable description of the execution plan.
    pub fn explain(&self, query: &str) -> CypherResult<String> {
        let stmt = self.parser.parse(query)?;
        let plan = self.planner.plan(&stmt)?;
        Ok(self.format_plan(&plan, 0))
    }

    /// Format a logical plan as a string
    fn format_plan(&self, plan: &LogicalOp, indent: usize) -> String {
        let prefix = "  ".repeat(indent);
        match plan {
            LogicalOp::AllNodesScan { variable } => {
                format!("{}AllNodesScan({})\n", prefix, variable)
            }
            LogicalOp::NodeByLabelScan { variable, label, predicate } => {
                let filter_str = if predicate.is_some() { " [+filter]" } else { "" };
                format!("{}NodeByLabelScan({}, :{}){}\n", prefix, variable, label, filter_str)
            }
            LogicalOp::NodeByIdSeek { variable, ids } => {
                format!("{}NodeByIdSeek({}, {:?})\n", prefix, variable, ids)
            }
            LogicalOp::Expand {
                input,
                from,
                to,
                rel_types,
                direction,
                ..
            } => {
                let dir = match direction {
                    crate::cypher::ast::Direction::Outgoing => "->",
                    crate::cypher::ast::Direction::Incoming => "<-",
                    crate::cypher::ast::Direction::Both => "--",
                };
                let types = if rel_types.is_empty() {
                    "*".to_string()
                } else {
                    rel_types
                        .iter()
                        .map(|t| t.to_string())
                        .collect::<Vec<_>>()
                        .join("|")
                };
                format!(
                    "{}Expand(({}){}[{}]{}({}))\n{}",
                    prefix,
                    from,
                    dir,
                    types,
                    dir,
                    to,
                    self.format_plan(input, indent + 1)
                )
            }
            LogicalOp::Filter { input, predicate } => {
                format!(
                    "{}Filter({:?})\n{}",
                    prefix,
                    predicate,
                    self.format_plan(input, indent + 1)
                )
            }
            LogicalOp::Project { input, items } => {
                let cols: Vec<_> = items
                    .iter()
                    .map(|i| {
                        i.alias
                            .as_ref()
                            .map(|a| a.to_string())
                            .unwrap_or_else(|| format!("{:?}", i.expr))
                    })
                    .collect();
                format!(
                    "{}Project([{}])\n{}",
                    prefix,
                    cols.join(", "),
                    self.format_plan(input, indent + 1)
                )
            }
            LogicalOp::Sort { input, items, .. } => {
                let cols: Vec<_> = items
                    .iter()
                    .map(|i| {
                        format!(
                            "{:?} {}",
                            i.expr,
                            if i.ascending { "ASC" } else { "DESC" }
                        )
                    })
                    .collect();
                format!(
                    "{}Sort([{}])\n{}",
                    prefix,
                    cols.join(", "),
                    self.format_plan(input, indent + 1)
                )
            }
            LogicalOp::Skip { input, count } => {
                format!(
                    "{}Skip({})\n{}",
                    prefix,
                    count,
                    self.format_plan(input, indent + 1)
                )
            }
            LogicalOp::Limit { input, count } => {
                format!(
                    "{}Limit({})\n{}",
                    prefix,
                    count,
                    self.format_plan(input, indent + 1)
                )
            }
            LogicalOp::Distinct { input } => {
                format!("{}Distinct\n{}", prefix, self.format_plan(input, indent + 1))
            }
            LogicalOp::CreateNode { labels, variable, .. } => {
                let var = variable
                    .as_ref()
                    .map(|v| v.to_string())
                    .unwrap_or_default();
                let lbls: Vec<_> = labels.iter().map(|l| format!(":{}", l)).collect();
                format!("{}CreateNode({}{})\n", prefix, var, lbls.join(""))
            }
            LogicalOp::CreateRelationship {
                from,
                to,
                rel_type,
                ..
            } => {
                format!(
                    "{}CreateRelationship(({})-[:{}]->({}))\n",
                    prefix, from, rel_type, to
                )
            }
            LogicalOp::DeleteNode {
                input,
                variable,
                detach,
            } => {
                format!(
                    "{}{}Delete({})\n{}",
                    prefix,
                    if *detach { "Detach" } else { "" },
                    variable,
                    self.format_plan(input, indent + 1)
                )
            }
            LogicalOp::Empty => format!("{}Empty\n", prefix),
            LogicalOp::SingleRow => format!("{}SingleRow\n", prefix),
            _ => format!("{}Unknown\n", prefix),
        }
    }
}

impl Default for DefaultCypherHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CypherHandler for DefaultCypherHandler {
    fn parse(&self, query: &str) -> CypherResult<Statement> {
        self.parser.parse(query)
    }

    fn plan(&self, stmt: &Statement) -> CypherResult<LogicalOp> {
        self.planner.plan(stmt)
    }

    fn execute(
        &self,
        plan: &LogicalOp,
        collection: &mut CollectionEngine,
        parameters: HashMap<String, CypherValue>,
    ) -> CypherResult<QueryResult> {
        let mut ctx = ExecutionContext::with_parameters(collection, parameters);
        self.executor.execute(plan, &mut ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn test_config() -> CollectionConfig {
        CollectionConfig::new(
            "test_handler",
            VectorConfig::new(4, Distance::Cosine),
            HnswConfig {
                m: 16,
                m0: 32,
                ml: HnswConfig::recommended_ml(16),
                ef: 100,
                ef_construction: 200,
            },
        )
    }

    #[test]
    fn test_handler_creation() {
        let handler = DefaultCypherHandler::new();
        assert!(handler.validate("MATCH (n) RETURN n").is_ok());
    }

    #[test]
    fn test_handler_parse() {
        let handler = DefaultCypherHandler::new();
        let stmt = handler.parse("MATCH (n:Person) RETURN n.name").unwrap();
        assert!(matches!(stmt, Statement::Query(_)));
    }

    #[test]
    fn test_handler_plan() {
        let handler = DefaultCypherHandler::new();
        let stmt = handler.parse("MATCH (n:Person) RETURN n.name").unwrap();
        let plan = handler.plan(&stmt).unwrap();
        assert!(matches!(plan, LogicalOp::Project { .. }));
    }

    #[test]
    fn test_handler_explain() {
        let handler = DefaultCypherHandler::new();
        let explanation = handler
            .explain("MATCH (n:Person) WHERE n.age > 25 RETURN n.name LIMIT 10")
            .unwrap();

        assert!(explanation.contains("Project"));
        assert!(explanation.contains("Limit"));
        // Filter is pushed down into NodeByLabelScan, indicated by [+filter]
        assert!(explanation.contains("NodeByLabelScan"));
        assert!(explanation.contains("[+filter]"));
    }

    #[test]
    fn test_handler_query() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();
        let handler = DefaultCypherHandler::new();

        // Create a node
        let result = handler
            .query(
                "CREATE (n:Person {name: \"Alice\", age: 30})",
                &mut engine,
                HashMap::new(),
            )
            .unwrap();

        assert_eq!(result.stats.nodes_created, 1);

        // Query all nodes
        let result = handler
            .query("MATCH (n) RETURN n", &mut engine, HashMap::new())
            .unwrap();

        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_handler_query_with_filter() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();
        let handler = DefaultCypherHandler::new();

        // Create nodes
        handler
            .query(
                "CREATE (n:Person {name: \"Alice\", age: 30})",
                &mut engine,
                HashMap::new(),
            )
            .unwrap();
        handler
            .query(
                "CREATE (n:Person {name: \"Bob\", age: 20})",
                &mut engine,
                HashMap::new(),
            )
            .unwrap();

        // Query with filter
        let result = handler
            .query(
                "MATCH (n:Person) WHERE n.age > 25 RETURN n.name",
                &mut engine,
                HashMap::new(),
            )
            .unwrap();

        // Should only return Alice
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_handler_query_with_limit() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();
        let handler = DefaultCypherHandler::new();

        // Create multiple nodes
        for i in 0..10 {
            handler
                .query(
                    &format!("CREATE (n:Person {{name: \"Person{}\", age: {}}})", i, 20 + i),
                    &mut engine,
                    HashMap::new(),
                )
                .unwrap();
        }

        // Query with limit
        let result = handler
            .query(
                "MATCH (n:Person) RETURN n.name LIMIT 5",
                &mut engine,
                HashMap::new(),
            )
            .unwrap();

        assert_eq!(result.rows.len(), 5);
    }

    #[test]
    fn test_handler_validate_invalid() {
        let handler = DefaultCypherHandler::new();
        let result = handler.validate("INVALID QUERY SYNTAX");
        assert!(result.is_err());
    }

    #[test]
    fn test_handler_with_parameters() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();
        let handler = DefaultCypherHandler::new();

        // Create a node
        handler
            .query(
                "CREATE (n:Person {name: \"Alice\", age: 30})",
                &mut engine,
                HashMap::new(),
            )
            .unwrap();

        // Query with parameter
        let mut params = HashMap::new();
        params.insert("min_age".to_string(), CypherValue::Int(25));

        // Note: Parameter usage in WHERE would require enhanced expression evaluation
        // For now, just verify parameters are passed correctly
        let result = handler
            .query("MATCH (n:Person) RETURN n.name", &mut engine, params)
            .unwrap();

        assert!(!result.rows.is_empty());
    }
}
