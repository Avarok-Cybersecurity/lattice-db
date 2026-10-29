//! Cypher query parser using pest grammar
//!
//! Converts Cypher query strings into AST structures.

use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};
// compact_str removed - using standard String for simplicity
use pest::iterators::{Pair, Pairs};
use pest::Parser;
use pest_derive::Parser;
// smallvec removed - AST now uses Vec for simplicity

/// Maximum allowed query string length (100KB) - prevents OOM from huge queries
const MAX_QUERY_LENGTH: usize = 100 * 1024;

/// Maximum allowed path traversal depth - prevents DoS via graph explosion
const MAX_PATH_LENGTH: u32 = 100;

#[derive(Parser)]
#[grammar = "cypher/cypher.pest"]
struct CypherPestParser;

/// Cypher query parser
///
/// Parses Cypher query strings into AST structures for further processing.
pub struct CypherParser;

impl CypherParser {
    /// Create a new parser
    pub fn new() -> Self {
        Self
    }

    /// Parse a Cypher statement
    ///
    /// # Security
    /// - Enforces maximum query length to prevent OOM attacks
    /// - Validates path lengths to prevent graph traversal DoS
    pub fn parse(&self, input: &str) -> CypherResult<Statement> {
        // Validate query size (DoS protection)
        if input.len() > MAX_QUERY_LENGTH {
            return Err(CypherError::syntax(
                0,
                0,
                format!(
                    "Query too large: {} bytes exceeds maximum of {} bytes",
                    input.len(),
                    MAX_QUERY_LENGTH
                ),
            ));
        }

        let pairs = CypherPestParser::parse(Rule::statement, input)
            .map_err(|e| self.pest_error_to_cypher(e))?;

        self.parse_statement(pairs)
    }

    /// Parse a statement from pest pairs
    pub(super) fn parse_statement(&self, pairs: Pairs<Rule>) -> CypherResult<Statement> {
        for pair in pairs {
            match pair.as_rule() {
                Rule::query => return Ok(Statement::Query(self.parse_query(pair)?)),
                Rule::create_statement => {
                    return Ok(Statement::Create(self.parse_create_statement(pair)?))
                }
                Rule::delete_statement => {
                    return Ok(Statement::Delete(self.parse_delete_statement(pair)?))
                }
                Rule::EOI => continue,
                _ => continue,
            }
        }
        Err(CypherError::UnexpectedEof)
    }

    /// Parse a query
    fn parse_query(&self, pair: Pair<Rule>) -> CypherResult<Query> {
        let mut match_clause = None;
        let mut optional_matches = Vec::new();
        let mut where_clause = None;
        let mut return_clause = None;
        let mut order_by = None;
        let mut skip = None;
        let mut limit = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::match_clause => {
                    match_clause = Some(self.parse_match_clause(inner)?);
                }
                Rule::optional_match => {
                    optional_matches.push(self.parse_optional_match(inner)?);
                }
                Rule::where_clause => {
                    where_clause = Some(self.parse_where_clause(inner)?);
                }
                Rule::return_clause => {
                    return_clause = Some(self.parse_return_clause(inner)?);
                }
                Rule::order_by_clause => {
                    order_by = Some(self.parse_order_by_clause(inner)?);
                }
                Rule::skip_clause => {
                    skip = Some(self.parse_skip_clause(inner)?);
                }
                Rule::limit_clause => {
                    limit = Some(self.parse_limit_clause(inner)?);
                }
                _ => {}
            }
        }

        let return_clause =
            return_clause.ok_or_else(|| CypherError::syntax(0, 0, "Missing RETURN clause"))?;

        Ok(Query {
            match_clause,
            optional_matches,
            where_clause,
            with_clause: None,
            return_clause,
            order_by,
            skip,
            limit,
        })
    }

    /// Parse MATCH clause
    fn parse_match_clause(&self, pair: Pair<Rule>) -> CypherResult<MatchClause> {
        let mut patterns = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::pattern_list {
                patterns = self.parse_pattern_list(inner)?;
            }
        }

        Ok(MatchClause { patterns })
    }

    /// Parse OPTIONAL MATCH
    fn parse_optional_match(&self, pair: Pair<Rule>) -> CypherResult<MatchClause> {
        let mut patterns = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::pattern_list {
                patterns = self.parse_pattern_list(inner)?;
            }
        }

        Ok(MatchClause { patterns })
    }

    /// Convert pest error to CypherError
    fn pest_error_to_cypher(&self, error: pest::error::Error<Rule>) -> CypherError {
        let (line, column) = match error.line_col {
            pest::error::LineColLocation::Pos((l, c)) => (l, c),
            pest::error::LineColLocation::Span((l, c), _) => (l, c),
        };

        CypherError::SyntaxError {
            line,
            column,
            message: error.to_string(),
        }
    }
}

impl Default for CypherParser {
    fn default() -> Self {
        Self::new()
    }
}

mod clause;
mod expr;
mod literal;
mod pattern;
#[cfg(test)]
mod tests;
