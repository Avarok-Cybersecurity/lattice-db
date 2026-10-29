//! Cypher Abstract Syntax Tree (AST) types
//!
//! These types represent the parsed structure of Cypher queries.
//! They are produced by the parser and consumed by the planner.

use serde::{Deserialize, Serialize};
mod display;
mod function;
pub use function::{AggregateKind, ScalarFn};

mod expr;
mod pattern;
#[cfg(test)]
mod tests;

pub use expr::*;
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

/// Read query structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Query {
    /// MATCH clause (optional for pure CREATE queries)
    pub match_clause: Option<MatchClause>,
    /// Optional MATCH (left outer join semantics)
    pub optional_matches: Vec<MatchClause>,
    /// WHERE clause for filtering
    pub where_clause: Option<WhereClause>,
    /// WITH clause for subquery chaining
    pub with_clause: Option<WithClause>,
    /// RETURN clause (required)
    pub return_clause: ReturnClause,
    /// ORDER BY clause
    pub order_by: Option<OrderByClause>,
    /// SKIP clause
    pub skip: Option<u64>,
    /// LIMIT clause
    pub limit: Option<u64>,
}

impl Query {
    /// Create a new query with just a RETURN clause
    pub fn new(return_clause: ReturnClause) -> Self {
        Self {
            match_clause: None,
            optional_matches: Vec::new(),
            where_clause: None,
            with_clause: None,
            return_clause,
            order_by: None,
            skip: None,
            limit: None,
        }
    }

    /// Add a MATCH clause
    pub fn with_match(mut self, match_clause: MatchClause) -> Self {
        self.match_clause = Some(match_clause);
        self
    }

    /// Add a WHERE clause
    pub fn with_where(mut self, where_clause: WhereClause) -> Self {
        self.where_clause = Some(where_clause);
        self
    }

    /// Add a LIMIT clause
    pub fn with_limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Add an ORDER BY clause
    pub fn with_order_by(mut self, order_by: OrderByClause) -> Self {
        self.order_by = Some(order_by);
        self
    }
}

/// MATCH clause - pattern matching
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchClause {
    /// Patterns to match
    pub patterns: Vec<Pattern>,
}

impl MatchClause {
    /// Create a new MATCH clause with a single pattern
    pub fn new(pattern: Pattern) -> Self {
        Self {
            patterns: vec![pattern],
        }
    }

    /// Create a MATCH clause with multiple patterns
    pub fn with_patterns(patterns: Vec<Pattern>) -> Self {
        Self { patterns }
    }
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

/// WITH clause for subquery chaining
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WithClause {
    pub items: Vec<ProjectionItem>,
    pub where_clause: Option<Box<WhereClause>>,
}

/// RETURN clause
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReturnClause {
    /// Whether RETURN DISTINCT was used
    pub distinct: bool,
    /// Projection items
    pub items: Vec<ProjectionItem>,
}

impl ReturnClause {
    /// Create a RETURN clause with items
    pub fn new(items: Vec<ProjectionItem>) -> Self {
        Self {
            distinct: false,
            items,
        }
    }

    /// Return all columns (RETURN *)
    pub fn all() -> Self {
        Self {
            distinct: false,
            items: vec![ProjectionItem::all()],
        }
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
