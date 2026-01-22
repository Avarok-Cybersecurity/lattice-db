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
//! # Supported Cypher Subset (MVP)
//!
//! | Clause | Support | Example |
//! |--------|---------|---------|
//! | MATCH | Single pattern | `MATCH (n:Person)` |
//! | WHERE | Basic predicates | `WHERE n.age > 21` |
//! | RETURN | Properties, aliases | `RETURN n.name AS name` |
//! | CREATE | Nodes + relationships | `CREATE (n:Person {name: "Alice"})` |
//! | DELETE | Single element | `DELETE n` |
//! | LIMIT | Integer | `LIMIT 10` |
//! | ORDER BY | Single property | `ORDER BY n.name` |
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
