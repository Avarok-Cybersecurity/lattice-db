//! Clause parsing: WHERE, WITH, UNWIND, RETURN projection bodies, CREATE, DELETE.

use super::{ident, CypherParser, Rule};
use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};
use pest::iterators::Pair;

impl CypherParser {
    /// Parse WHERE clause
    pub(super) fn parse_where_clause(&self, pair: Pair<Rule>) -> CypherResult<WhereClause> {
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::expression {
                return Ok(WhereClause::new(self.parse_expression(inner)?));
            }
        }
        Err(CypherError::syntax(0, 0, "Empty WHERE clause"))
    }

    /// Parse RETURN clause
    pub(super) fn parse_return_clause(&self, pair: Pair<Rule>) -> CypherResult<ReturnClause> {
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::projection_body {
                return self.parse_projection_body(inner);
            }
        }
        Err(CypherError::syntax(0, 0, "Empty RETURN clause"))
    }

    /// Parse WITH clause: a projection body and an optional WHERE
    pub(super) fn parse_with(&self, pair: Pair<Rule>) -> CypherResult<WithClause> {
        let mut projection = None;
        let mut where_clause = None;
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::projection_body => projection = Some(self.parse_projection_body(inner)?),
                Rule::where_clause => where_clause = Some(self.parse_where_clause(inner)?),
                _ => {}
            }
        }
        Ok(WithClause {
            projection: projection.ok_or_else(|| CypherError::syntax(0, 0, "Empty WITH clause"))?,
            where_clause,
        })
    }

    /// Parse UNWIND clause
    pub(super) fn parse_unwind(&self, pair: Pair<Rule>) -> CypherResult<UnwindClause> {
        let mut expr = None;
        let mut variable = None;
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::expression => expr = Some(self.parse_expression(inner)?),
                Rule::identifier => variable = Some(ident(&inner)),
                _ => {}
            }
        }
        match (expr, variable) {
            (Some(expr), Some(variable)) => Ok(UnwindClause { expr, variable }),
            _ => Err(CypherError::syntax(
                0,
                0,
                "UNWIND requires `expression AS variable`",
            )),
        }
    }

    /// Parse the projection body shared by RETURN and WITH
    fn parse_projection_body(&self, pair: Pair<Rule>) -> CypherResult<ReturnClause> {
        let mut body = ReturnClause::new(Vec::new());

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::distinct => body.distinct = true,
                Rule::return_items => {
                    for item in inner.into_inner() {
                        if item.as_rule() == Rule::return_item {
                            body.items.push(self.parse_return_item(item)?);
                        }
                    }
                }
                Rule::order_by_clause => body.order_by = Some(self.parse_order_by_clause(inner)?),
                Rule::skip_clause => body.skip = Some(self.parse_skip_clause(inner)?),
                Rule::limit_clause => body.limit = Some(self.parse_limit_clause(inner)?),
                _ => {}
            }
        }

        Ok(body)
    }

    /// Parse return item
    pub(super) fn parse_return_item(&self, pair: Pair<Rule>) -> CypherResult<ProjectionItem> {
        let mut expr = None;
        let mut alias = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::star => return Ok(ProjectionItem::all()),
                Rule::expression => expr = Some(self.parse_expression(inner)?),
                Rule::alias => {
                    for alias_inner in inner.into_inner() {
                        if alias_inner.as_rule() == Rule::identifier {
                            alias = Some(ident(&alias_inner));
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(ProjectionItem {
            expr: expr.ok_or_else(|| CypherError::syntax(0, 0, "Empty projection item"))?,
            alias,
        })
    }

    /// Parse ORDER BY clause
    pub(super) fn parse_order_by_clause(&self, pair: Pair<Rule>) -> CypherResult<OrderByClause> {
        let mut items = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::order_item {
                items.push(self.parse_order_item(inner)?);
            }
        }

        Ok(OrderByClause { items })
    }

    /// Parse order item
    pub(super) fn parse_order_item(&self, pair: Pair<Rule>) -> CypherResult<OrderByItem> {
        let mut expr = None;
        let mut ascending = true;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::expression => {
                    expr = Some(self.parse_expression(inner)?);
                }
                Rule::order_direction => {
                    let dir = inner.as_str().to_uppercase();
                    ascending = !dir.starts_with("DESC");
                }
                _ => {}
            }
        }

        Ok(OrderByItem {
            expr: expr.ok_or_else(|| CypherError::syntax(0, 0, "Missing ORDER BY expression"))?,
            ascending,
        })
    }

    /// Parse SKIP clause
    pub(super) fn parse_skip_clause(&self, pair: Pair<Rule>) -> CypherResult<u64> {
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::integer {
                return inner
                    .as_str()
                    .parse()
                    .map_err(|_| CypherError::syntax(0, 0, "Invalid SKIP value"));
            }
        }
        Err(CypherError::syntax(0, 0, "Missing SKIP value"))
    }

    /// Parse LIMIT clause
    pub(super) fn parse_limit_clause(&self, pair: Pair<Rule>) -> CypherResult<u64> {
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::integer {
                return inner
                    .as_str()
                    .parse()
                    .map_err(|_| CypherError::syntax(0, 0, "Invalid LIMIT value"));
            }
        }
        Err(CypherError::syntax(0, 0, "Missing LIMIT value"))
    }

    /// Parse CREATE statement
    pub(super) fn parse_create_statement(&self, pair: Pair<Rule>) -> CypherResult<CreateClause> {
        let mut patterns = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::pattern_list {
                patterns = self.parse_pattern_list(inner)?;
            }
        }

        Ok(CreateClause { patterns })
    }

    /// Parse DELETE statement
    pub(super) fn parse_delete_statement(&self, pair: Pair<Rule>) -> CypherResult<DeleteClause> {
        let mut detach = false;
        let mut variables = Vec::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::detach => {
                    detach = true;
                }
                Rule::variable_list => {
                    for var in inner.into_inner() {
                        if var.as_rule() == Rule::identifier {
                            variables.push(ident(&var));
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(DeleteClause { detach, variables })
    }
}
