//! Literal parsing: numbers, strings, parameters, calls, CASE, lists, maps.

use super::{ident, CypherParser, Rule};
use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};
use crate::types::value::CypherValue;
use pest::iterators::Pair;

impl CypherParser {
    /// Parse literal value
    pub(super) fn parse_literal(&self, pair: Pair<Rule>) -> CypherResult<CypherValue> {
        let inner = pair
            .into_inner()
            .next()
            .ok_or_else(|| CypherError::syntax(0, 0, "Invalid literal"))?;
        match inner.as_rule() {
            Rule::null_literal => Ok(CypherValue::Null),
            Rule::boolean_literal => Ok(CypherValue::Bool(
                inner.as_str().eq_ignore_ascii_case("true"),
            )),
            Rule::number_literal => self.parse_number(inner),
            Rule::string_literal => self.parse_string_literal(inner),
            other => Err(CypherError::internal(format!(
                "unexpected literal {other:?}"
            ))),
        }
    }

    /// Parse number literal
    pub(super) fn parse_number(&self, pair: Pair<Rule>) -> CypherResult<CypherValue> {
        let inner = pair
            .into_inner()
            .next()
            .ok_or_else(|| CypherError::syntax(0, 0, "Invalid number literal"))?;
        let text = inner.as_str();
        match inner.as_rule() {
            Rule::float => text
                .parse()
                .map(CypherValue::Float)
                .map_err(|_| CypherError::syntax(0, 0, format!("Invalid float literal: {text}"))),
            Rule::integer => text
                .parse()
                .map(CypherValue::Int)
                .map_err(|_| CypherError::syntax(0, 0, format!("Integer out of range: {text}"))),
            other => Err(CypherError::internal(format!(
                "unexpected number {other:?}"
            ))),
        }
    }

    /// Parse string literal: strip the quotes, then unescape in one left-to-right pass
    pub(super) fn parse_string_literal(&self, pair: Pair<Rule>) -> CypherResult<CypherValue> {
        let s = pair.as_str();
        let body = s
            .get(1..s.len().saturating_sub(1))
            .ok_or_else(|| CypherError::syntax(0, 0, "Unterminated string literal"))?;
        let mut out = String::with_capacity(body.len());
        let mut chars = body.chars();
        while let Some(c) = chars.next() {
            if c != '\\' {
                out.push(c);
                continue;
            }
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('b') => out.push('\u{8}'),
                Some('f') => out.push('\u{c}'),
                Some('u') => {
                    let hex: String = chars.by_ref().take(4).collect();
                    let ch = u32::from_str_radix(&hex, 16)
                        .ok()
                        .and_then(char::from_u32)
                        .ok_or_else(|| {
                            CypherError::syntax(0, 0, format!("Invalid escape \\u{hex}"))
                        })?;
                    out.push(ch);
                }
                Some(other @ ('\\' | '\'' | '"')) => out.push(other),
                Some(other) => {
                    return Err(CypherError::syntax(
                        0,
                        0,
                        format!("Invalid escape sequence \\{other}"),
                    ))
                }
                None => return Err(CypherError::syntax(0, 0, "Dangling escape in string")),
            }
        }
        Ok(CypherValue::string(out))
    }

    /// Parse parameter
    pub(super) fn parse_parameter(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        pair.into_inner()
            .next()
            .map(|name| Expr::Parameter(ident(&name)))
            .ok_or_else(|| CypherError::syntax(0, 0, "Invalid parameter"))
    }

    /// Parse function call; `count(*)` becomes `count` applied to `Expr::Star`
    pub(super) fn parse_function_call(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut name = None;
        let mut args = Vec::new();
        let mut distinct = false;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::function_name => name = Some(inner.as_str().to_string()),
                Rule::count_star => args.push(Expr::Star),
                Rule::distinct => distinct = true,
                Rule::expression => args.push(self.parse_expression(inner)?),
                _ => {}
            }
        }

        Ok(Expr::FunctionCall {
            name: name.ok_or_else(|| CypherError::syntax(0, 0, "Missing function name"))?,
            args,
            distinct,
        })
    }

    /// Parse case expression
    pub(super) fn parse_case_expr(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut operand = None;
        let mut when_clauses = Vec::new();
        let mut else_clause = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::case_operand => {
                    let e = inner
                        .into_inner()
                        .next()
                        .ok_or_else(|| CypherError::syntax(0, 0, "Empty CASE operand"))?;
                    operand = Some(Box::new(self.parse_expression(e)?));
                }
                Rule::when_clause => {
                    let mut parts = inner
                        .into_inner()
                        .filter(|p| p.as_rule() == Rule::expression);
                    match (parts.next(), parts.next()) {
                        (Some(w), Some(t)) => when_clauses
                            .push((self.parse_expression(w)?, self.parse_expression(t)?)),
                        _ => return Err(CypherError::syntax(0, 0, "WHEN requires THEN")),
                    }
                }
                Rule::else_clause => {
                    let e = inner
                        .into_inner()
                        .find(|p| p.as_rule() == Rule::expression)
                        .ok_or_else(|| CypherError::syntax(0, 0, "Empty ELSE"))?;
                    else_clause = Some(Box::new(self.parse_expression(e)?));
                }
                _ => {}
            }
        }

        Ok(Expr::Case {
            operand,
            when_clauses,
            else_clause,
        })
    }

    /// Parse list literal
    pub(super) fn parse_list_literal(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        pair.into_inner()
            .filter(|p| p.as_rule() == Rule::expression)
            .map(|p| self.parse_expression(p))
            .collect::<CypherResult<Vec<_>>>()
            .map(Expr::List)
    }

    /// Parse a map literal in an expression: `{key: expr, ...}`
    pub(super) fn parse_map_literal(&self, pair: Pair<Rule>) -> CypherResult<MapLiteral> {
        let mut entries = Vec::new();
        for entry in pair.into_inner() {
            if entry.as_rule() == Rule::map_entry {
                entries.push(self.parse_key_value(entry)?);
            }
        }
        Ok(MapLiteral { entries })
    }

    /// Parse inline pattern properties: `(n {key: expr, ...})`
    pub(super) fn parse_properties(&self, pair: Pair<Rule>) -> CypherResult<MapLiteral> {
        let mut entries = Vec::new();
        for list in pair.into_inner() {
            for property in list.into_inner() {
                entries.push(self.parse_key_value(property)?);
            }
        }
        Ok(MapLiteral { entries })
    }

    /// `identifier : expression`
    fn parse_key_value(&self, pair: Pair<Rule>) -> CypherResult<(String, Box<Expr>)> {
        let mut key = None;
        let mut value = None;
        for part in pair.into_inner() {
            match part.as_rule() {
                Rule::identifier => key = Some(ident(&part)),
                Rule::expression => value = Some(self.parse_expression(part)?),
                _ => {}
            }
        }
        match (key, value) {
            (Some(k), Some(v)) => Ok((k, Box::new(v))),
            _ => Err(CypherError::syntax(0, 0, "Map entry requires `key: value`")),
        }
    }
}
