//! Query planner - converts Cypher AST to logical operations
//!
//! The planner transforms the parsed AST into a logical execution plan
//! that can be executed by the executor.

use crate::cypher::ast::*;
use crate::cypher::error::CypherResult;
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

mod mutation;
mod pattern;
mod query;
#[cfg(test)]
mod tests;
