//! Expression parsing: operators by precedence, atoms.

use super::{CypherParser, Rule};
use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};
use pest::iterators::Pair;

impl CypherParser {
    /// Parse expression
    pub(super) fn parse_expression(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        // expression = { or_expr }
        // Extract the or_expr child and parse it
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::or_expr {
                return self.parse_or_expr(inner);
            }
        }
        Err(CypherError::syntax(0, 0, "Empty expression"))
    }

    /// Parse OR expression
    pub(super) fn parse_or_expr(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        // or_expr = { xor_expr ~ (^"OR" ~ xor_expr)* }
        let inner: Vec<_> = pair.into_inner().collect();
        if inner.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty OR expression"));
        }

        // Find all xor_expr children
        let xor_exprs: Vec<_> = inner
            .iter()
            .filter(|p| p.as_rule() == Rule::xor_expr)
            .collect();

        if xor_exprs.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty OR expression"));
        }

        let mut result = self.parse_xor_expr(xor_exprs[0].clone())?;

        for xor_expr in &xor_exprs[1..] {
            let right = self.parse_xor_expr((*xor_expr).clone())?;
            result = Expr::binary(result, BinaryOp::Or, right);
        }

        Ok(result)
    }

    /// Parse XOR expression
    pub(super) fn parse_xor_expr(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        // xor_expr = { and_expr ~ (^"XOR" ~ and_expr)* }
        let inner: Vec<_> = pair.into_inner().collect();
        if inner.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty XOR expression"));
        }

        // Find all and_expr children
        let and_exprs: Vec<_> = inner
            .iter()
            .filter(|p| p.as_rule() == Rule::and_expr)
            .collect();

        if and_exprs.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty XOR expression"));
        }

        let mut result = self.parse_and_expr(and_exprs[0].clone())?;

        for and_expr in &and_exprs[1..] {
            let right = self.parse_and_expr((*and_expr).clone())?;
            result = Expr::binary(result, BinaryOp::Xor, right);
        }

        Ok(result)
    }

    /// Parse AND expression
    pub(super) fn parse_and_expr(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        // and_expr = { not_expr ~ (^"AND" ~ not_expr)* }
        let inner: Vec<_> = pair.into_inner().collect();
        if inner.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty AND expression"));
        }

        // Find all not_expr children
        let not_exprs: Vec<_> = inner
            .iter()
            .filter(|p| p.as_rule() == Rule::not_expr)
            .collect();

        if not_exprs.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty AND expression"));
        }

        let mut result = self.parse_not_expr(not_exprs[0].clone())?;

        for not_expr in &not_exprs[1..] {
            let right = self.parse_not_expr((*not_expr).clone())?;
            result = Expr::binary(result, BinaryOp::And, right);
        }

        Ok(result)
    }

    /// Parse NOT expression
    pub(super) fn parse_not_expr(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        // not_expr = { ^"NOT"* ~ comparison }
        let mut not_count = 0;
        let mut comparison = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::comparison => {
                    comparison = Some(self.parse_comparison(inner)?);
                }
                _ => {
                    // Count NOT keywords
                    if inner.as_str().to_uppercase() == "NOT" {
                        not_count += 1;
                    }
                }
            }
        }

        let mut result =
            comparison.ok_or_else(|| CypherError::syntax(0, 0, "Missing comparison"))?;

        // Apply NOT operators (odd number = NOT, even = no-op)
        if not_count % 2 == 1 {
            result = Expr::unary(UnaryOp::Not, result);
        }

        Ok(result)
    }

    /// Parse comparison expression
    pub(super) fn parse_comparison(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let inner: Vec<_> = pair.into_inner().collect();
        if inner.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty comparison"));
        }

        let mut result = self.parse_addition(inner[0].clone())?;

        let mut i = 1;
        while i < inner.len() {
            if inner[i].as_rule() == Rule::comparison_op {
                let op_str = inner[i].as_str().to_uppercase();
                let op = self.parse_comparison_op(&op_str)?;

                if i + 1 < inner.len() {
                    let right = self.parse_addition(inner[i + 1].clone())?;

                    // Handle special operators
                    match op_str.as_str() {
                        s if s.contains("NULL") => {
                            result = Expr::IsNull {
                                expr: Box::new(result),
                                negated: s.contains("NOT"),
                            };
                        }
                        s if s.contains("IN") => {
                            result = Expr::In {
                                expr: Box::new(result),
                                list: Box::new(right),
                                negated: false,
                            };
                        }
                        s if s.contains("STARTS") => {
                            result = Expr::StringPredicate {
                                expr: Box::new(result),
                                predicate: StringPredicateType::StartsWith,
                                pattern: Box::new(right),
                            };
                        }
                        s if s.contains("ENDS") => {
                            result = Expr::StringPredicate {
                                expr: Box::new(result),
                                predicate: StringPredicateType::EndsWith,
                                pattern: Box::new(right),
                            };
                        }
                        s if s.contains("CONTAINS") => {
                            result = Expr::StringPredicate {
                                expr: Box::new(result),
                                predicate: StringPredicateType::Contains,
                                pattern: Box::new(right),
                            };
                        }
                        _ => {
                            result = Expr::binary(result, op, right);
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        Ok(result)
    }

    /// Parse comparison operator
    pub(super) fn parse_comparison_op(&self, op: &str) -> CypherResult<BinaryOp> {
        match op {
            "=" => Ok(BinaryOp::Eq),
            "<>" | "!=" => Ok(BinaryOp::Neq),
            "<" => Ok(BinaryOp::Lt),
            "<=" => Ok(BinaryOp::Lte),
            ">" => Ok(BinaryOp::Gt),
            ">=" => Ok(BinaryOp::Gte),
            _ => Ok(BinaryOp::Eq), // Fallback for special operators handled separately
        }
    }

    /// Parse addition expression
    pub(super) fn parse_addition(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let inner: Vec<_> = pair.into_inner().collect();
        if inner.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty addition"));
        }

        let mut result = self.parse_multiplication(inner[0].clone())?;

        let mut i = 1;
        while i < inner.len() {
            let token = inner[i].as_str();
            if token == "+" || token == "-" {
                if i + 1 < inner.len() {
                    let right = self.parse_multiplication(inner[i + 1].clone())?;
                    let op = if token == "+" {
                        BinaryOp::Add
                    } else {
                        BinaryOp::Sub
                    };
                    result = Expr::binary(result, op, right);
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                // Try to parse as multiplication
                let right = self.parse_multiplication(inner[i].clone())?;
                result = Expr::binary(result, BinaryOp::Add, right);
                i += 1;
            }
        }

        Ok(result)
    }

    /// Parse multiplication expression
    pub(super) fn parse_multiplication(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let inner: Vec<_> = pair.into_inner().collect();
        if inner.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty multiplication"));
        }

        let mut result = self.parse_power(inner[0].clone())?;

        let mut i = 1;
        while i < inner.len() {
            let token = inner[i].as_str();
            if token == "*" || token == "/" || token == "%" {
                if i + 1 < inner.len() {
                    let right = self.parse_power(inner[i + 1].clone())?;
                    let op = match token {
                        "*" => BinaryOp::Mul,
                        "/" => BinaryOp::Div,
                        "%" => BinaryOp::Mod,
                        // Guard against impossible parse state - should never occur
                        other => {
                            return Err(CypherError::internal(format!(
                                "Unexpected operator in multiplication: {}",
                                other
                            )))
                        }
                    };
                    result = Expr::binary(result, op, right);
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                let right = self.parse_power(inner[i].clone())?;
                result = Expr::binary(result, BinaryOp::Mul, right);
                i += 1;
            }
        }

        Ok(result)
    }

    /// Parse power expression
    pub(super) fn parse_power(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let inner: Vec<_> = pair.into_inner().collect();
        if inner.is_empty() {
            return Err(CypherError::syntax(0, 0, "Empty power"));
        }

        let mut result = self.parse_unary(inner[0].clone())?;

        for item in inner[1..].iter() {
            if item.as_str() == "^" {
                continue;
            }
            let right = self.parse_unary(item.clone())?;
            result = Expr::binary(result, BinaryOp::Pow, right);
        }

        Ok(result)
    }

    /// Parse unary expression
    pub(super) fn parse_unary(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut neg = false;
        let mut pos = false;
        let mut expr_pair = None;

        for inner in pair.into_inner() {
            match inner.as_str() {
                "-" => neg = true,
                "+" => pos = true,
                _ => {
                    if inner.as_rule() == Rule::property_or_atom {
                        expr_pair = Some(inner);
                    }
                }
            }
        }

        let expr = if let Some(p) = expr_pair {
            self.parse_property_or_atom(p)?
        } else {
            return Err(CypherError::syntax(0, 0, "Missing unary operand"));
        };

        if neg {
            Ok(Expr::unary(UnaryOp::Neg, expr))
        } else if pos {
            Ok(Expr::unary(UnaryOp::Pos, expr))
        } else {
            Ok(expr)
        }
    }

    /// Parse property access or atom
    pub(super) fn parse_property_or_atom(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut result = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::atom => {
                    result = Some(self.parse_atom(inner)?);
                }
                Rule::property_access => {
                    for prop_inner in inner.into_inner() {
                        if prop_inner.as_rule() == Rule::identifier {
                            result = Some(Expr::property(
                                result.take().ok_or_else(|| {
                                    CypherError::syntax(0, 0, "Property access without base")
                                })?,
                                prop_inner.as_str(),
                            ));
                        }
                    }
                }
                _ => {}
            }
        }

        result.ok_or_else(|| CypherError::syntax(0, 0, "Empty property or atom"))
    }

    /// Parse atom
    pub(super) fn parse_atom(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::literal => {
                    return Ok(Expr::Literal(self.parse_literal(inner)?));
                }
                Rule::parameter => {
                    return self.parse_parameter(inner);
                }
                Rule::function_call => {
                    return self.parse_function_call(inner);
                }
                Rule::case_expr => {
                    return self.parse_case_expr(inner);
                }
                Rule::list_literal => {
                    return self.parse_list_literal(inner);
                }
                Rule::map_literal => {
                    return Ok(Expr::Map(self.parse_map_literal(inner)?));
                }
                Rule::expression => {
                    return self.parse_expression(inner);
                }
                Rule::identifier => {
                    return Ok(Expr::variable(inner.as_str()));
                }
                _ => {}
            }
        }

        Err(CypherError::syntax(0, 0, "Invalid atom"))
    }
}
