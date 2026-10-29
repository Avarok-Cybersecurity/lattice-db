//! Cypher query language implementation for LatticeDB
//!
//! This module provides openCypher query support with:
//! - Memory-efficient data types (rkyv zero-copy serialization)
//! - WASM compatibility (no chrono, no std::time dependencies)
//! - Clean separation of parsing, planning, and execution
//!
//! # Architecture
//!
//! ```text
//! Query String
//!     │
//!     ▼
//! ┌─────────────┐
//! │   Parser    │  pest grammar → AST
//! └─────────────┘
//!     │
//!     ▼
//! ┌─────────────┐
//! │  Analyzer   │  Semantic validation, type checking
//! └─────────────┘
//!     │
//!     ▼
//! ┌─────────────┐
//! │  Planner    │  AST → Logical Plan
//! └─────────────┘
//!     │
//!     ▼
//! ┌─────────────┐
//! │  Executor   │  Execute against storage
//! └─────────────┘
//!     │
//!     ▼
//!   Results
//! ```
//!
//! # Supported Cypher Subset
//!
//! | Clause | Support | Example |
//! |--------|---------|---------|
//! | MATCH | Patterns in any direction, comma-joined, own WHERE | `MATCH (m)-[r:MEASURES]->(k:Kernel)` |
//! | OPTIONAL MATCH | Left outer join (nulls for no match) | `OPTIONAL MATCH (k)<-[:ABOUT]-(d)` |
//! | WHERE | Any expression, three-valued logic | `WHERE m.tok_s > 10 AND k:Kernel` |
//! | WITH | Projection, aggregation, DISTINCT, ORDER BY/SKIP/LIMIT, WHERE | `WITH k, count(m) AS n WHERE n > 2` |
//! | UNWIND | Lists (null: no rows) | `UNWIND d.tags AS t` |
//! | RETURN | Expressions, `*`, DISTINCT, implicit GROUP BY | `RETURN k.name, collect(m.name)` |
//! | ORDER BY / SKIP / LIMIT | Expressions, aliases, aggregates; nulls last ascending | `ORDER BY count(*) DESC` |
//! | CREATE | Nodes + relationships | `CREATE (n:Person {name: "Alice"})` |
//! | DELETE | Single element | `DELETE n` |
//!
//! Expressions: arithmetic, string/list `+`, comparisons (chained), AND/OR/XOR/NOT, IN,
//! STARTS WITH / ENDS WITH / CONTAINS, IS [NOT] NULL, label predicates, CASE, list and map
//! literals, indexing and slicing, parameters, the scalar functions of `ast::ScalarFn` and the
//! aggregates of `ast::AggregateKind`.
//!
//! Refused with an error naming the construct: MERGE, SET, REMOVE, CALL, FOREACH, UNION,
//! shortestPath, named paths, list/pattern comprehensions, subqueries, `=~`, and binding a
//! variable-length relationship to a variable. Variable-length patterns return each node
//! reachable within the hop range once (by shortest distance), not one row per path.
//!
//! # Example
//!
//! ```ignore
//! use lattice_core::cypher::{CypherHandler, DefaultCypherHandler, ExecutionContext};
//!
//! let handler = DefaultCypherHandler::new();
//! let ctx = ExecutionContext::new(&collection);
//!
//! let result = handler.query(
//!     "MATCH (n:Person) WHERE n.age > 25 RETURN n.name",
//!     ctx
//! )?;
//!
//! for row in result.rows {
//!     println!("{:?}", row);
//! }
//! ```

pub mod ast;
pub mod error;
pub mod executor;
mod explain;
pub mod handler;
pub mod parser;
pub mod planner;
pub mod row;

// Re-export commonly used types
pub use ast::{
    Direction, Expr, MatchClause, NodePattern, OrderByClause, Pattern, PatternElement, Query,
    RelPattern, ReturnClause, Statement, WhereClause,
};
pub use error::{CypherError, CypherResult};
pub use executor::{ExecutionContext, QueryResult, QueryStats};
pub use handler::{CypherHandler, DefaultCypherHandler};
pub use parser::CypherParser;
pub use planner::{LogicalOp, QueryPlanner};
#[cfg(feature = "simd")]
pub use row::{radix_partial_sort_i64_indexed, radix_sort_i64_indexed, SimdI64x4, SimdRowBatch};
pub use row::{
    ExecutorRow, PackedI64, ParallelRow, ParallelRowItem, Row, RowItem, SimdIntRow, SmallRow,
};
