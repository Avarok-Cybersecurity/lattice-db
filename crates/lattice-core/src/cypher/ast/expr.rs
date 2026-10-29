//! Expression AST.

use super::{MapLiteral, Pattern, ScalarFn};
use crate::types::value::CypherValue;
use serde::{Deserialize, Serialize};

/// Expression in Cypher
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// Star/wildcard (*)
    Star,

    /// Literal value
    Literal(CypherValue),

    /// Variable reference
    Variable(String),

    /// Property access: `n.name`
    Property { expr: Box<Expr>, property: String },

    /// Parameter: `$param`
    Parameter(String),

    /// List literal: `[1, 2, 3]`
    List(Vec<Expr>),

    /// Map literal: `{a: 1, b: 2}`
    Map(MapLiteral),

    /// Binary operation: `a + b`, `a > b`, etc.
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },

    /// Unary operation: `NOT a`, `-x`
    UnaryOp { op: UnaryOp, expr: Box<Expr> },

    /// Function call: `count(*)`, `toUpper(s)`
    FunctionCall {
        name: String,
        args: Vec<Expr>,
        distinct: bool,
    },

    /// CASE expression
    Case {
        operand: Option<Box<Expr>>,
        when_clauses: Vec<(Expr, Expr)>,
        else_clause: Option<Box<Expr>>,
    },

    /// IS NULL / IS NOT NULL
    IsNull { expr: Box<Expr>, negated: bool },

    /// IN expression: `x IN [1, 2, 3]`
    In {
        expr: Box<Expr>,
        list: Box<Expr>,
        negated: bool,
    },

    /// STARTS WITH, ENDS WITH, CONTAINS
    StringPredicate {
        expr: Box<Expr>,
        predicate: StringPredicateType,
        pattern: Box<Expr>,
    },

    /// EXISTS subquery (Phase 2)
    Exists { pattern: Pattern },

    /// Label predicate: `n:Label1:Label2` (true when the node has every label)
    HasLabels {
        expr: Box<Expr>,
        labels: Vec<String>,
    },

    /// Subscript: `list[i]`, `map['key']`, `node['prop']`
    Index { expr: Box<Expr>, index: Box<Expr> },

    /// List slice: `list[from..to]` (either bound may be omitted)
    Slice {
        expr: Box<Expr>,
        from: Option<Box<Expr>>,
        to: Option<Box<Expr>>,
    },

    /// A scalar function call the planner resolved against the function catalogue.
    /// The parser never produces it; it replaces `FunctionCall` in planned expressions.
    Scalar { func: ScalarFn, args: Vec<Expr> },
}

impl Expr {
    /// Create a literal expression
    pub fn literal(value: impl Into<CypherValue>) -> Self {
        Self::Literal(value.into())
    }

    /// Create a variable reference
    pub fn variable(name: impl Into<String>) -> Self {
        Self::Variable(name.into())
    }

    /// Create a property access expression
    pub fn property(expr: Expr, property: impl Into<String>) -> Self {
        Self::Property {
            expr: Box::new(expr),
            property: property.into(),
        }
    }

    /// Create a parameter reference
    pub fn parameter(name: impl Into<String>) -> Self {
        Self::Parameter(name.into())
    }

    /// Create a binary operation
    pub fn binary(left: Expr, op: BinaryOp, right: Expr) -> Self {
        Self::BinaryOp {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    /// Create a unary operation
    pub fn unary(op: UnaryOp, expr: Expr) -> Self {
        Self::UnaryOp {
            op,
            expr: Box::new(expr),
        }
    }

    /// Create a function call
    pub fn function(name: impl Into<String>, args: Vec<Expr>) -> Self {
        Self::FunctionCall {
            name: name.into(),
            args,
            distinct: false,
        }
    }

    /// Comparison shortcuts
    pub fn eq(self, other: Expr) -> Self {
        Self::binary(self, BinaryOp::Eq, other)
    }

    pub fn neq(self, other: Expr) -> Self {
        Self::binary(self, BinaryOp::Neq, other)
    }

    pub fn lt(self, other: Expr) -> Self {
        Self::binary(self, BinaryOp::Lt, other)
    }

    pub fn lte(self, other: Expr) -> Self {
        Self::binary(self, BinaryOp::Lte, other)
    }

    pub fn gt(self, other: Expr) -> Self {
        Self::binary(self, BinaryOp::Gt, other)
    }

    pub fn gte(self, other: Expr) -> Self {
        Self::binary(self, BinaryOp::Gte, other)
    }

    /// Logical shortcuts
    pub fn and(self, other: Expr) -> Self {
        Self::binary(self, BinaryOp::And, other)
    }

    pub fn or(self, other: Expr) -> Self {
        Self::binary(self, BinaryOp::Or, other)
    }

    pub fn not(self) -> Self {
        Self::unary(UnaryOp::Not, self)
    }
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,

    // Comparison
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,

    // Logical
    And,
    Or,
    Xor,

    // String
    Concat,
    RegexMatch,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Not,
    Neg,
    Pos,
}

/// String predicate types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StringPredicateType {
    StartsWith,
    EndsWith,
    Contains,
}
