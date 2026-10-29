//! Cypher query parser using pest grammar
//!
//! Converts Cypher query strings into AST structures.

use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};
use pest::iterators::{Pair, Pairs};
use pest::Parser;
use pest_derive::Parser;

/// Maximum allowed query string length (100KB) - prevents OOM from huge queries
const MAX_QUERY_LENGTH: usize = 100 * 1024;

/// Maximum allowed path traversal depth - prevents DoS via graph explosion
const MAX_PATH_LENGTH: u32 = 100;

/// Maximum bracket nesting depth - bounds parser and evaluator recursion
const MAX_NESTING_DEPTH: usize = 128;

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
    /// - Enforces a maximum bracket nesting depth so recursive descent cannot exhaust the stack
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
        let depth = unsupported::nesting_depth(input);
        if depth > MAX_NESTING_DEPTH {
            return Err(CypherError::QueryTooComplex {
                message: format!(
                    "bracket nesting depth {depth} exceeds the maximum of {MAX_NESTING_DEPTH}"
                ),
            });
        }

        let pairs = match CypherPestParser::parse(Rule::statement, input) {
            Ok(pairs) => pairs,
            // A construct outside the grammar is named, rather than reported as a bare
            // syntax error at some column.
            Err(e) => {
                return Err(
                    unsupported::diagnose(input).unwrap_or_else(|| self.pest_error_to_cypher(e))
                )
            }
        };

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
                _ => continue,
            }
        }
        Err(CypherError::UnexpectedEof)
    }

    /// Parse a query: reading clauses in order, then RETURN
    fn parse_query(&self, pair: Pair<Rule>) -> CypherResult<Query> {
        let mut clauses = Vec::new();
        let mut return_clause = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::match_clause => clauses.push(Clause::Match(self.parse_match(inner, false)?)),
                Rule::optional_match => clauses.push(Clause::Match(self.parse_match(inner, true)?)),
                Rule::unwind_clause => clauses.push(Clause::Unwind(self.parse_unwind(inner)?)),
                Rule::with_clause => clauses.push(Clause::With(self.parse_with(inner)?)),
                Rule::return_clause => return_clause = Some(self.parse_return_clause(inner)?),
                _ => {}
            }
        }

        let return_clause =
            return_clause.ok_or_else(|| CypherError::syntax(0, 0, "Missing RETURN clause"))?;
        Ok(Query {
            clauses,
            return_clause,
        })
    }

    /// Parse MATCH / OPTIONAL MATCH with its WHERE
    fn parse_match(&self, pair: Pair<Rule>, optional: bool) -> CypherResult<MatchClause> {
        let mut patterns = Vec::new();
        let mut where_clause = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::pattern_list => patterns = self.parse_pattern_list(inner)?,
                Rule::where_clause => where_clause = Some(self.parse_where_clause(inner)?),
                _ => {}
            }
        }

        Ok(MatchClause {
            optional,
            patterns,
            where_clause,
        })
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
mod unsupported;

/// The name an identifier token denotes: backticks removed, doubled backticks unescaped.
pub(super) fn ident(pair: &Pair<Rule>) -> String {
    let s = pair.as_str();
    match s.strip_prefix('`').and_then(|x| x.strip_suffix('`')) {
        Some(quoted) => quoted.replace("``", "`"),
        None => s.to_string(),
    }
}
