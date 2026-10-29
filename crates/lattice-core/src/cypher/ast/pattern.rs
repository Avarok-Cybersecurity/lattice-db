//! Pattern AST: node and relationship patterns, path lengths, map literals.

use super::Expr;
use serde::{Deserialize, Serialize};

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
    pub fn chain(start: NodePattern, rel: RelPattern, end: NodePattern) -> Self {
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
            entries: entries
                .into_iter()
                .map(|(k, v)| (k.into(), Box::new(v)))
                .collect(),
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
