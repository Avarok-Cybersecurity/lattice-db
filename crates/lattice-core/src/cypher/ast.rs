//! Cypher Abstract Syntax Tree (AST) types
//!
//! These types represent the parsed structure of Cypher queries.
//! They are produced by the parser and consumed by the planner.

use crate::types::value::CypherValue;
use serde::{Deserialize, Serialize};
// smallvec removed - using standard Vec

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

/// A pattern in MATCH/CREATE
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pattern {
    /// Pattern elements (alternating nodes and relationships)
    pub elements: Vec<PatternElement>,
}

impl Pattern {
    /// Create a pattern with a single node
    pub fn node(node: NodePattern) -> Self {
        Self {
            elements: vec![PatternElement::Node(node)],
        }
    }

    /// Create a pattern with a node-relationship-node chain
    pub fn chain(
        start: NodePattern,
        rel: RelPattern,
        end: NodePattern,
    ) -> Self {
        Self {
            elements: vec![
                PatternElement::Node(start),
                PatternElement::Relationship(rel),
                PatternElement::Node(end),
            ],
        }
    }

    /// Add another hop to the pattern
    pub fn extend(mut self, rel: RelPattern, node: NodePattern) -> Self {
        self.elements.push(PatternElement::Relationship(rel));
        self.elements.push(PatternElement::Node(node));
        self
    }
}

/// Element in a pattern (node or relationship)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PatternElement {
    Node(NodePattern),
    Relationship(RelPattern),
}

/// Node pattern: `(variable:Label {props})`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodePattern {
    /// Variable name (e.g., `n` in `(n:Person)`)
    pub variable: Option<String>,
    /// Labels (e.g., `Person` in `(n:Person)`)
    pub labels: Vec<String>,
    /// Property constraints (e.g., `{name: "Alice"}`)
    pub properties: Option<MapLiteral>,
}

impl NodePattern {
    /// Create an anonymous node pattern
    pub fn anonymous() -> Self {
        Self {
            variable: None,
            labels: Vec::new(),
            properties: None,
        }
    }

    /// Create a node pattern with a variable
    pub fn with_variable(variable: impl Into<String>) -> Self {
        Self {
            variable: Some(variable.into()),
            labels: Vec::new(),
            properties: None,
        }
    }

    /// Add a label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.labels.push(label.into());
        self
    }

    /// Add labels
    pub fn with_labels(mut self, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.labels.extend(labels.into_iter().map(Into::into));
        self
    }

    /// Add properties
    pub fn with_properties(mut self, properties: MapLiteral) -> Self {
        self.properties = Some(properties);
        self
    }
}

/// Relationship pattern: `-[variable:TYPE {props}]->`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelPattern {
    /// Variable name
    pub variable: Option<String>,
    /// Relationship types (e.g., `KNOWS` in `-[:KNOWS]->`)
    pub rel_types: Vec<String>,
    /// Direction of the relationship
    pub direction: Direction,
    /// Property constraints
    pub properties: Option<MapLiteral>,
    /// Variable-length path range (e.g., `*1..3` for 1-3 hops)
    pub length: Option<PathLength>,
}

impl RelPattern {
    /// Create an anonymous relationship pattern
    pub fn anonymous(direction: Direction) -> Self {
        Self {
            variable: None,
            rel_types: Vec::new(),
            direction,
            properties: None,
            length: None,
        }
    }

    /// Create a relationship pattern with a type
    pub fn with_type(rel_type: impl Into<String>, direction: Direction) -> Self {
        let mut pattern = Self::anonymous(direction);
        pattern.rel_types.push(rel_type.into());
        pattern
    }

    /// Add a variable
    pub fn with_variable(mut self, variable: impl Into<String>) -> Self {
        self.variable = Some(variable.into());
        self
    }

    /// Add a variable-length path
    pub fn with_length(mut self, length: PathLength) -> Self {
        self.length = Some(length);
        self
    }
}

/// Relationship direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// Outgoing: `(a)-[r]->(b)`
    Outgoing,
    /// Incoming: `(a)<-[r]-(b)`
    Incoming,
    /// Both/undirected: `(a)-[r]-(b)`
    Both,
}

/// Variable-length path specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathLength {
    /// Minimum hops (default 1)
    pub min: Option<u32>,
    /// Maximum hops (None = unbounded)
    pub max: Option<u32>,
}

impl PathLength {
    /// Create a fixed-length path
    pub fn fixed(n: u32) -> Self {
        Self {
            min: Some(n),
            max: Some(n),
        }
    }

    /// Create a range path
    pub fn range(min: Option<u32>, max: Option<u32>) -> Self {
        Self { min, max }
    }
}

/// Map literal: `{key: value, ...}`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapLiteral {
    /// Key-value pairs (Boxed Expr to break recursive cycle)
    pub entries: Vec<(String, Box<Expr>)>,
}

impl MapLiteral {
    /// Create an empty map literal
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create a map literal from entries
    pub fn from_entries(entries: impl IntoIterator<Item = (impl Into<String>, Expr)>) -> Self {
        Self {
            entries: entries.into_iter().map(|(k, v)| (k.into(), Box::new(v))).collect(),
        }
    }

    /// Add an entry
    pub fn with_entry(mut self, key: impl Into<String>, value: Expr) -> Self {
        self.entries.push((key.into(), Box::new(value)));
        self
    }
}

impl Default for MapLiteral {
    fn default() -> Self {
        Self::new()
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
    Label {
        variable: String,
        label: String,
    },
}

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
    Property {
        expr: Box<Expr>,
        property: String,
    },

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
    UnaryOp {
        op: UnaryOp,
        expr: Box<Expr>,
    },

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
    IsNull {
        expr: Box<Expr>,
        negated: bool,
    },

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
    Exists {
        pattern: Pattern,
    },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn test_node_pattern_builder() {
        let node = NodePattern::with_variable("n")
            .with_label("Person")
            .with_properties(MapLiteral::from_entries([
                ("name", Expr::literal("Alice")),
            ]));

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
        assert!(matches!(pattern.elements[1], PatternElement::Relationship(_)));
        assert!(matches!(pattern.elements[2], PatternElement::Node(_)));
    }

    #[test]
    fn test_query_builder() {
        let query = Query::new(ReturnClause::new(vec![
            ProjectionItem::new(Expr::property(Expr::variable("n"), "name"))
                .with_alias("name"),
        ]))
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

        assert!(matches!(expr, Expr::BinaryOp { op: BinaryOp::And, .. }));
    }

    #[test]
    fn test_direction() {
        assert_eq!(Direction::Outgoing, Direction::Outgoing);
        assert_ne!(Direction::Incoming, Direction::Outgoing);
    }
}
