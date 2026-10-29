//! Query planner - converts Cypher AST to logical operations
//!
//! The planner transforms the parsed AST into a logical execution plan
//! that can be executed by the executor. It also does the semantic checks openCypher requires
//! before execution: every variable must be in scope, every function must exist with the right
//! arity, aggregates may appear only in projections, and unaliased WITH expressions are refused.

use crate::cypher::ast::*;
use crate::cypher::error::CypherResult;
use serde::{Deserialize, Serialize};

/// Maximum allowed path traversal depth (DoS protection)
const MAX_PATH_LENGTH: u32 = 100;

/// One aggregate the `Aggregate` operator computes per group
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregateCall {
    pub kind: AggregateKind,
    /// `count(DISTINCT x)`, `collect(DISTINCT x)`, ...
    pub distinct: bool,
    /// The argument; `None` is `count(*)`
    pub arg: Option<Expr>,
}

/// Logical operation in the query plan
///
/// These operations form a tree that represents the logical execution plan.
/// The executor traverses this tree to produce results. Operators bind the variables they
/// introduce to row columns in execution order (inputs first), which the executor tracks.
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
    /// Expand from a bound node along relationships.
    ///
    /// When `to` (or `rel_variable`) is already bound, the expansion only keeps rows whose
    /// traversed node (relationship) equals the bound value.
    Expand {
        input: Box<LogicalOp>,
        from: String,
        /// Single-hop expansions always bind the relationship (anonymous ones under a hidden
        /// name) so relationship uniqueness can be enforced; variable-length ones never do.
        rel_variable: Option<String>,
        to: String,
        rel_types: Vec<String>,
        direction: Direction,
        min_hops: u32,
        max_hops: u32,
        /// Relationships bound earlier in the same MATCH that this one must differ from
        distinct_from: Vec<String>,
    },

    // === Joins ===
    /// Every combination of a left row and a right row (right is independent of left)
    Cartesian {
        left: Box<LogicalOp>,
        right: Box<LogicalOp>,
    },

    /// The rows of the enclosing `OptionalMatch` input (leaf of its `inner` plan)
    Argument,

    /// Left outer join: `inner` runs over the input rows (via `Argument`); input rows with no
    /// inner match are kept, with every variable `inner` introduced set to null
    OptionalMatch {
        input: Box<LogicalOp>,
        inner: Box<LogicalOp>,
    },

    /// One row per element of a list
    Unwind {
        input: Box<LogicalOp>,
        expr: Expr,
        variable: String,
    },

    // === Filter/Transform ===
    /// Filter rows based on a predicate
    Filter {
        input: Box<LogicalOp>,
        predicate: Expr,
    },

    /// Project columns. Every item has its output name as `alias`. With `carry`, the input
    /// columns are kept after the projected ones (still bound to their variables) so a
    /// following ORDER BY can use them; a later trimming `Project` drops them.
    Project {
        input: Box<LogicalOp>,
        items: Vec<ProjectionItem>,
        carry: bool,
    },

    /// Group rows by `keys` and compute `aggregates` per group. `items` (with output names as
    /// `alias`) are evaluated per group over `#key{i}` / `#agg{j}` variables.
    Aggregate {
        input: Box<LogicalOp>,
        keys: Vec<Expr>,
        aggregates: Vec<AggregateCall>,
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

impl LogicalOp {
    /// Rebuild this operator with every input operator replaced by `f(input)`
    fn map_inputs(self, f: &mut impl FnMut(LogicalOp) -> LogicalOp) -> LogicalOp {
        let mut b = |op: Box<LogicalOp>| Box::new(f(*op));
        match self {
            LogicalOp::Expand {
                input,
                from,
                rel_variable,
                to,
                rel_types,
                direction,
                min_hops,
                max_hops,
                distinct_from,
            } => LogicalOp::Expand {
                input: b(input),
                from,
                rel_variable,
                to,
                rel_types,
                direction,
                min_hops,
                max_hops,
                distinct_from,
            },
            LogicalOp::Cartesian { left, right } => LogicalOp::Cartesian {
                left: b(left),
                right: b(right),
            },
            LogicalOp::OptionalMatch { input, inner } => LogicalOp::OptionalMatch {
                input: b(input),
                inner: b(inner),
            },
            LogicalOp::Unwind {
                input,
                expr,
                variable,
            } => LogicalOp::Unwind {
                input: b(input),
                expr,
                variable,
            },
            LogicalOp::Filter { input, predicate } => LogicalOp::Filter {
                input: b(input),
                predicate,
            },
            LogicalOp::Project {
                input,
                items,
                carry,
            } => LogicalOp::Project {
                input: b(input),
                items,
                carry,
            },
            LogicalOp::Aggregate {
                input,
                keys,
                aggregates,
                items,
            } => LogicalOp::Aggregate {
                input: b(input),
                keys,
                aggregates,
                items,
            },
            LogicalOp::Sort {
                input,
                items,
                limit,
            } => LogicalOp::Sort {
                input: b(input),
                items,
                limit,
            },
            LogicalOp::Skip { input, count } => LogicalOp::Skip {
                input: b(input),
                count,
            },
            LogicalOp::Limit { input, count } => LogicalOp::Limit {
                input: b(input),
                count,
            },
            LogicalOp::Distinct { input } => LogicalOp::Distinct { input: b(input) },
            LogicalOp::DeleteNode {
                input,
                variable,
                detach,
            } => LogicalOp::DeleteNode {
                input: b(input),
                variable,
                detach,
            },
            LogicalOp::DeleteRelationship { input, variable } => LogicalOp::DeleteRelationship {
                input: b(input),
                variable,
            },
            leaf => leaf,
        }
    }
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
        Ok(Self::optimize_predicate_pushdown(plan))
    }

    /// Push a Filter directly over a label scan into the scan, everywhere in the plan
    fn optimize_predicate_pushdown(plan: LogicalOp) -> LogicalOp {
        match plan {
            LogicalOp::Filter { input, predicate } => match *input {
                LogicalOp::NodeByLabelScan {
                    variable,
                    label,
                    predicate: None,
                } => LogicalOp::NodeByLabelScan {
                    variable,
                    label,
                    predicate: Some(Box::new(predicate)),
                },
                other => LogicalOp::Filter {
                    input: Box::new(Self::optimize_predicate_pushdown(other)),
                    predicate,
                },
            },
            other => other.map_inputs(&mut Self::optimize_predicate_pushdown),
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
mod projection;
mod query;
mod resolve;
#[cfg(test)]
mod tests;
