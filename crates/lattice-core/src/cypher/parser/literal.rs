//! Literal parsing: numbers, strings, parameters, calls, CASE, lists, maps.

use super::{CypherParser, Rule};
use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};
use crate::types::value::CypherValue;
use pest::iterators::Pair;

impl CypherParser {
    /// Parse literal value
    pub(super) fn parse_literal(&self, pair: Pair<Rule>) -> CypherResult<CypherValue> {
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::null_literal => {
                    return Ok(CypherValue::Null);
                }
                Rule::boolean_literal => {
                    let val = inner.as_str().to_uppercase() == "TRUE";
                    return Ok(CypherValue::Bool(val));
                }
                Rule::number_literal => {
                    return self.parse_number(inner);
                }
                Rule::string_literal => {
                    return self.parse_string_literal(inner);
                }
                _ => {}
            }
        }

        Err(CypherError::syntax(0, 0, "Invalid literal"))
    }

    /// Parse number literal
    pub(super) fn parse_number(&self, pair: Pair<Rule>) -> CypherResult<CypherValue> {
        // Save string before consuming pair
        let s = pair.as_str().to_string();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::float => {
                    let val: f64 = inner
                        .as_str()
                        .parse()
                        .map_err(|_| CypherError::syntax(0, 0, "Invalid float literal"))?;
                    return Ok(CypherValue::Float(val));
                }
                Rule::integer => {
                    let val: i64 = inner
                        .as_str()
                        .parse()
                        .map_err(|_| CypherError::syntax(0, 0, "Invalid integer literal"))?;
                    return Ok(CypherValue::Int(val));
                }
                _ => {}
            }
        }

        // Try parsing the string directly
        let s = s.as_str();
        if s.contains('.') {
            let val: f64 = s
                .parse()
                .map_err(|_| CypherError::syntax(0, 0, "Invalid float literal"))?;
            Ok(CypherValue::Float(val))
        } else {
            let val: i64 = s
                .parse()
                .map_err(|_| CypherError::syntax(0, 0, "Invalid integer literal"))?;
            Ok(CypherValue::Int(val))
        }
    }

    /// Parse string literal
    pub(super) fn parse_string_literal(&self, pair: Pair<Rule>) -> CypherResult<CypherValue> {
        let s = pair.as_str();
        // Remove quotes and handle escape sequences
        let inner = if s.starts_with('"') {
            &s[1..s.len() - 1]
        } else if s.starts_with('\'') {
            &s[1..s.len() - 1]
        } else {
            s
        };

        // Unescape
        let unescaped = inner
            .replace("\\\"", "\"")
            .replace("\\'", "'")
            .replace("\\\\", "\\")
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t");

        Ok(CypherValue::string(unescaped))
    }

    /// Parse parameter
    pub(super) fn parse_parameter(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        // Save string before consuming pair
        let s = pair.as_str().to_string();
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::identifier {
                return Ok(Expr::Parameter(String::from(inner.as_str())));
            }
        }
        // Fallback: extract parameter name from the pair itself
        if s.starts_with('$') {
            Ok(Expr::Parameter(String::from(&s[1..])))
        } else {
            Err(CypherError::syntax(0, 0, "Invalid parameter"))
        }
    }

    /// Parse function call
    pub(super) fn parse_function_call(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut name = None;
        let mut args = Vec::new();
        let mut distinct = false;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::identifier => {
                    if name.is_none() {
                        name = Some(String::from(inner.as_str()));
                    }
                }
                Rule::distinct => {
                    distinct = true;
                }
                Rule::expression => {
                    args.push(self.parse_expression(inner)?);
                }
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

        let current_when: Option<Expr> = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::expression if operand.is_none() && current_when.is_none() => {
                    operand = Some(Box::new(self.parse_expression(inner)?));
                }
                Rule::when_clause => {
                    let mut when_expr = None;
                    let mut then_expr = None;
                    for when_inner in inner.into_inner() {
                        if when_inner.as_rule() == Rule::expression {
                            if when_expr.is_none() {
                                when_expr = Some(self.parse_expression(when_inner)?);
                            } else {
                                then_expr = Some(self.parse_expression(when_inner)?);
                            }
                        }
                    }
                    if let (Some(w), Some(t)) = (when_expr, then_expr) {
                        when_clauses.push((w, t));
                    }
                }
                Rule::else_clause => {
                    for else_inner in inner.into_inner() {
                        if else_inner.as_rule() == Rule::expression {
                            else_clause = Some(Box::new(self.parse_expression(else_inner)?));
                        }
                    }
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
        let mut items = Vec::new();

        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::expression {
                items.push(self.parse_expression(inner)?);
            }
        }

        Ok(Expr::List(items))
    }

    /// Parse map literal (also used for properties)
    pub(super) fn parse_map_literal(&self, pair: Pair<Rule>) -> CypherResult<MapLiteral> {
        let mut entries = Vec::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::property_list | Rule::map_entry | Rule::property => {
                    // Handle both property_list and direct map_entry
                    for entry in inner.into_inner() {
                        match entry.as_rule() {
                            Rule::property | Rule::map_entry => {
                                let mut key = None;
                                let mut value = None;
                                for e in entry.into_inner() {
                                    match e.as_rule() {
                                        Rule::identifier => {
                                            key = Some(String::from(e.as_str()));
                                        }
                                        Rule::expression => {
                                            value = Some(self.parse_expression(e)?);
                                        }
                                        _ => {}
                                    }
                                }
                                if let (Some(k), Some(v)) = (key, value) {
                                    entries.push((k, Box::new(v)));
                                }
                            }
                            Rule::identifier => {
                                // First identifier in a property
                                let _key = String::from(entry.as_str());
                                // Look for the next sibling (value)
                                // This branch handles direct identifier tokens
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(MapLiteral { entries })
    }
}
