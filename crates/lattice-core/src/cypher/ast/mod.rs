//! Cypher Abstract Syntax Tree (AST) types
//!
//! These types represent the parsed structure of Cypher queries.
//! They are produced by the parser and consumed by the planner.

use serde::{Deserialize, Serialize};

mod display;
mod expr;
mod function;
mod pattern;
#[cfg(test)]
mod tests;
pub(crate) mod walk;

pub use expr::*;
pub use function::{AggregateKind, ScalarFn};
pub use pattern::*;

/// Top-level Cypher statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    /// Read query (MATCH ... RETURN)
    Query(Query),
    /// Create nodes/relationships
    Create(CreateClause),
    /// Delete nodes/relationships
    Delete(DeleteClause),
    /// Combined query with mutations
    ReadWrite {
        query: Query,
        mutations: Vec<Mutation>,
    },
}

/// A mutation operation (CREATE, DELETE, SET)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Mutation {
    Create(CreateClause),
    Delete(DeleteClause),
    Set(SetClause),
}

/// Read query: reading and projection clauses in order, then the final RETURN.
///
/// openCypher evaluates clauses left to right; each clause consumes the rows (and the variables
/// in scope) the previous one produced. `WITH` replaces the scope with its projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Query {
    /// MATCH, OPTIONAL MATCH, UNWIND and WITH clauses before the RETURN
    pub clauses: Vec<Clause>,
    /// RETURN clause (required), with its ORDER BY / SKIP / LIMIT
    pub return_clause: ReturnClause,
}

impl Query {
    /// Create a new query with just a RETURN clause
    pub fn new(return_clause: ReturnClause) -> Self {
        Self {
            clauses: Vec::new(),
            return_clause,
        }
    }

    /// Append a MATCH (or OPTIONAL MATCH) clause
    pub fn with_match(mut self, match_clause: MatchClause) -> Self {
        self.clauses.push(Clause::Match(match_clause));
        self
    }

    /// Set the RETURN clause's LIMIT
    pub fn with_limit(mut self, limit: u64) -> Self {
        self.return_clause.limit = Some(limit);
        self
    }

    /// Set the RETURN clause's ORDER BY
    pub fn with_order_by(mut self, order_by: OrderByClause) -> Self {
        self.return_clause.order_by = Some(order_by);
        self
    }
}

/// A clause that precedes the final RETURN
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Clause {
    /// `MATCH` or `OPTIONAL MATCH`, with its own WHERE
    Match(MatchClause),
    /// `UNWIND expr AS variable`
    Unwind(UnwindClause),
    /// `WITH projection [WHERE predicate]`
    With(WithClause),
}

/// MATCH clause - pattern matching
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchClause {
    /// `OPTIONAL MATCH`: rows without a match are kept, extended with nulls
    pub optional: bool,
    /// Patterns to match (comma-separated patterns join on shared variables)
    pub patterns: Vec<Pattern>,
    /// The WHERE that belongs to this MATCH (for OPTIONAL MATCH it is part of the pattern)
    pub where_clause: Option<WhereClause>,
}

impl MatchClause {
    /// Create a new MATCH clause with a single pattern
    pub fn new(pattern: Pattern) -> Self {
        Self::with_patterns(vec![pattern])
    }

    /// Create a MATCH clause with multiple patterns
    pub fn with_patterns(patterns: Vec<Pattern>) -> Self {
        Self {
            optional: false,
            patterns,
            where_clause: None,
        }
    }

    /// Attach this MATCH's WHERE clause
    pub fn with_where(mut self, where_clause: WhereClause) -> Self {
        self.where_clause = Some(where_clause);
        self
    }

    /// Make this an OPTIONAL MATCH
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }
}

/// UNWIND clause: one row per element of a list
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnwindClause {
    pub expr: Expr,
    pub variable: String,
}

/// WHERE clause
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhereClause {
    pub predicate: Expr,
}

impl WhereClause {
    pub fn new(predicate: Expr) -> Self {
        Self { predicate }
    }
}

/// WITH clause: an intermediate projection, optionally filtered
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WithClause {
    /// Same projection body as RETURN (DISTINCT, items, ORDER BY, SKIP, LIMIT)
    pub projection: ReturnClause,
    /// WHERE over the projected variables
    pub where_clause: Option<WhereClause>,
}

/// Projection body of RETURN (and of WITH)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReturnClause {
    /// Whether DISTINCT was used
    pub distinct: bool,
    /// Projection items
    pub items: Vec<ProjectionItem>,
    /// ORDER BY, applied after projection (and aggregation)
    pub order_by: Option<OrderByClause>,
    /// SKIP, applied after ORDER BY
    pub skip: Option<u64>,
    /// LIMIT, applied after SKIP
    pub limit: Option<u64>,
}

impl ReturnClause {
    /// Create a RETURN clause with items
    pub fn new(items: Vec<ProjectionItem>) -> Self {
        Self {
            distinct: false,
            items,
            order_by: None,
            skip: None,
            limit: None,
        }
    }

    /// Return all columns (RETURN *)
    pub fn all() -> Self {
        Self::new(vec![ProjectionItem::all()])
    }

    /// Make this RETURN DISTINCT
    pub fn distinct(mut self) -> Self {
        self.distinct = true;
        self
    }
}

/// Projection item in RETURN or WITH
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectionItem {
    /// Expression to project
    pub expr: Expr,
    /// Alias (e.g., `AS name`)
    pub alias: Option<String>,
}

impl ProjectionItem {
    /// Create a projection item
    pub fn new(expr: Expr) -> Self {
        Self { expr, alias: None }
    }

    /// Create a projection for all columns (*)
    pub fn all() -> Self {
        Self {
            expr: Expr::Star,
            alias: None,
        }
    }

    /// Add an alias
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }
}

/// ORDER BY clause
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderByClause {
    pub items: Vec<OrderByItem>,
}

impl OrderByClause {
    /// Create an ORDER BY clause with a single item
    pub fn new(expr: Expr, ascending: bool) -> Self {
        Self {
            items: vec![OrderByItem { expr, ascending }],
        }
    }

    /// Add another ordering item
    pub fn then_by(mut self, expr: Expr, ascending: bool) -> Self {
        self.items.push(OrderByItem { expr, ascending });
        self
    }
}

/// ORDER BY item
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderByItem {
    pub expr: Expr,
    pub ascending: bool,
}

/// CREATE clause
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateClause {
    pub patterns: Vec<Pattern>,
}

impl CreateClause {
    pub fn new(pattern: Pattern) -> Self {
        Self {
            patterns: vec![pattern],
        }
    }
}

/// DELETE clause
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeleteClause {
    /// Whether DETACH DELETE was used
    pub detach: bool,
    /// Variables to delete
    pub variables: Vec<String>,
}

impl DeleteClause {
    pub fn new(variables: Vec<String>) -> Self {
        Self {
            detach: false,
            variables,
        }
    }

    pub fn detach(mut self) -> Self {
        self.detach = true;
        self
    }
}

/// SET clause for property updates
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetClause {
    pub items: Vec<SetItem>,
}

/// SET item
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SetItem {
    /// `SET n.prop = value`
    Property {
        variable: String,
        property: String,
        value: Expr,
    },
    /// `SET n = {props}`
    AllProperties {
        variable: String,
        properties: MapLiteral,
    },
    /// `SET n += {props}`
    MergeProperties {
        variable: String,
        properties: MapLiteral,
    },
    /// `SET n:Label`
    Label { variable: String, label: String },
}
