//! Query executor - executes logical plans against storage
//!
//! The executor traverses the logical operation tree and produces results
//! by interacting with the CollectionEngine.

mod context;
mod evaluation;
mod hashing;
mod json;
mod operations;

pub use context::{ExecutionContext, LabelRegistry};

use crate::cypher::ast::Expr;
use crate::cypher::error::CypherResult;
use crate::cypher::planner::LogicalOp;
use crate::cypher::row::ExecutorRow;
use crate::types::value::CypherValue;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Sort key with inline storage for 1-3 keys (covers 95%+ of ORDER BY clauses)
/// SmallVec avoids heap allocation for common case, reducing memory pressure
pub(crate) type SortKey = SmallVec<[CypherValue; 3]>;

/// Internal row type for executor operations.
/// Uses SmallVec for inline storage of 1-2 elements (common case).
/// Convert to Vec<CypherValue> at API boundaries (QueryResult).
pub(crate) type InternalRows = Vec<ExecutorRow>;

/// Result of query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Column names
    pub columns: Vec<String>,
    /// Result rows
    pub rows: Vec<Vec<CypherValue>>,
    /// Execution statistics
    pub stats: QueryStats,
}

impl QueryResult {
    /// Create an empty result
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            stats: QueryStats::default(),
        }
    }

    /// Create a result with columns
    pub fn with_columns(columns: Vec<String>) -> Self {
        Self {
            columns,
            rows: Vec::new(),
            stats: QueryStats::default(),
        }
    }
}

/// Query execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryStats {
    pub nodes_created: u64,
    pub relationships_created: u64,
    pub nodes_deleted: u64,
    pub relationships_deleted: u64,
    pub properties_set: u64,
    pub labels_added: u64,
    pub execution_time_ms: u64,
}

/// Query executor
pub struct QueryExecutor;

impl QueryExecutor {
    /// Create a new executor
    pub fn new() -> Self {
        Self
    }

    /// Execute a logical plan
    pub fn execute(
        &self,
        plan: &LogicalOp,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<QueryResult> {
        #[cfg(not(target_arch = "wasm32"))]
        let start = std::time::Instant::now();

        #[allow(unused_mut)]
        let (internal_rows, mut stats) = self.execute_op(plan, ctx)?;

        // Extract column names from the projection
        let columns = self.extract_columns(plan);

        #[cfg(not(target_arch = "wasm32"))]
        {
            stats.execution_time_ms = start.elapsed().as_millis() as u64;
        }

        // Convert internal SmallVec rows to Vec for public API
        let rows: Vec<Vec<CypherValue>> =
            internal_rows.into_iter().map(|row| row.to_vec()).collect();

        Ok(QueryResult {
            columns,
            rows,
            stats,
        })
    }

    /// Extract column names from a plan
    fn extract_columns(&self, plan: &LogicalOp) -> Vec<String> {
        match plan {
            LogicalOp::Project { items, .. } => items
                .iter()
                .enumerate()
                .map(|(i, item)| {
                    item.alias
                        .as_ref()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| self.expr_to_column_name(&item.expr, i))
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Convert expression to column name
    fn expr_to_column_name(&self, expr: &Expr, index: usize) -> String {
        match expr {
            Expr::Variable(name) => name.to_string(),
            Expr::Property { expr, property } => {
                format!("{}.{}", self.expr_to_column_name(expr, index), property)
            }
            Expr::Star => "*".to_string(),
            _ => format!("column_{}", index),
        }
    }
}

impl Default for QueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cypher::ast::MapLiteral;
    use crate::cypher::planner::LogicalOp;
    use crate::engine::collection::CollectionEngine;
    use crate::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
    use crate::types::point::Point;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn test_config() -> CollectionConfig {
        CollectionConfig::new(
            "test_cypher",
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
    fn test_execute_create_node() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();
        let executor = QueryExecutor::new();

        let plan = LogicalOp::CreateNode {
            labels: vec![String::from("Person")],
            properties: MapLiteral::from_entries([
                ("name", Expr::literal("Alice")),
                ("age", Expr::literal(30i64)),
            ]),
            variable: Some(String::from("n")),
        };

        let mut ctx = ExecutionContext::new(&mut engine);
        let result = executor.execute(&plan, &mut ctx).unwrap();

        assert_eq!(result.stats.nodes_created, 1);
        assert_eq!(result.rows.len(), 1);

        // Verify node exists
        assert_eq!(engine.point_count(), 1);
    }

    #[test]
    fn test_execute_all_nodes_scan() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();
        let executor = QueryExecutor::new();

        // Create some nodes
        engine
            .upsert_points(vec![
                Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4]),
                Point::new_vector(2, vec![0.2, 0.3, 0.4, 0.5]),
            ])
            .unwrap();

        let plan = LogicalOp::AllNodesScan {
            variable: String::from("n"),
        };

        let mut ctx = ExecutionContext::new(&mut engine);
        let (rows, _) = executor.execute_op(&plan, &mut ctx).unwrap();

        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_execute_limit() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();
        let executor = QueryExecutor::new();

        // Create some nodes
        for i in 0..10 {
            engine
                .upsert_points(vec![Point::new_vector(i, vec![0.1, 0.2, 0.3, 0.4])])
                .unwrap();
        }

        let plan = LogicalOp::Limit {
            input: Box::new(LogicalOp::AllNodesScan {
                variable: String::from("n"),
            }),
            count: 5,
        };

        let mut ctx = ExecutionContext::new(&mut engine);
        let (rows, _) = executor.execute_op(&plan, &mut ctx).unwrap();

        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn test_evaluate_arithmetic() {
        let executor = QueryExecutor::new();

        // Addition
        let result = executor
            .add_values(&CypherValue::Int(5), &CypherValue::Int(3))
            .unwrap();
        assert_eq!(result, CypherValue::Int(8));

        // Subtraction
        let result = executor
            .sub_values(&CypherValue::Int(5), &CypherValue::Int(3))
            .unwrap();
        assert_eq!(result, CypherValue::Int(2));

        // Multiplication
        let result = executor
            .mul_values(&CypherValue::Int(5), &CypherValue::Int(3))
            .unwrap();
        assert_eq!(result, CypherValue::Int(15));

        // Division
        let result = executor
            .div_values(&CypherValue::Int(6), &CypherValue::Int(3))
            .unwrap();
        assert_eq!(result, CypherValue::Int(2));
    }

    #[test]
    fn test_evaluate_comparison() {
        let executor = QueryExecutor::new();

        assert!(executor
            .compare_values(&CypherValue::Int(5), &CypherValue::Int(3))
            .is_gt());
        assert!(executor
            .compare_values(&CypherValue::Int(3), &CypherValue::Int(5))
            .is_lt());
        assert!(executor
            .compare_values(&CypherValue::Int(5), &CypherValue::Int(5))
            .is_eq());
    }

    #[test]
    fn test_evaluate_logical() {
        use crate::cypher::ast::BinaryOp;

        let executor = QueryExecutor::new();

        let result = executor
            .evaluate_binary_op(
                BinaryOp::And,
                &CypherValue::Bool(true),
                &CypherValue::Bool(true),
            )
            .unwrap();
        assert_eq!(result, CypherValue::Bool(true));

        let result = executor
            .evaluate_binary_op(
                BinaryOp::And,
                &CypherValue::Bool(true),
                &CypherValue::Bool(false),
            )
            .unwrap();
        assert_eq!(result, CypherValue::Bool(false));

        let result = executor
            .evaluate_binary_op(
                BinaryOp::Or,
                &CypherValue::Bool(false),
                &CypherValue::Bool(true),
            )
            .unwrap();
        assert_eq!(result, CypherValue::Bool(true));
    }
}
