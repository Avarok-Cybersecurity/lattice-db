//! Expression parsing: operators by precedence, postfix access, atoms.
//!
//! Each grammar level yields its operands and its (named) operator tokens in order, so every
//! level is a left fold over `operand (op operand)*`.

use super::{ident, CypherParser, Rule};
use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};
use crate::types::value::CypherValue;
use pest::iterators::Pair;

fn missing(what: &str) -> CypherError {
    CypherError::syntax(0, 0, format!("Missing {what}"))
}

impl CypherParser {
    /// Parse expression
    pub(super) fn parse_expression(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let inner = pair
            .into_inner()
            .next()
            .ok_or_else(|| missing("expression"))?;
        self.parse_or_expr(inner)
    }

    /// Fold `operand (keyword operand)*` with one logical operator
    fn parse_logical(
        &self,
        pair: Pair<Rule>,
        op: BinaryOp,
        operand: fn(&Self, Pair<Rule>) -> CypherResult<Expr>,
    ) -> CypherResult<Expr> {
        let mut result: Option<Expr> = None;
        for inner in pair.into_inner() {
            if matches!(inner.as_rule(), Rule::kw_or | Rule::kw_xor | Rule::kw_and) {
                continue;
            }
            let right = operand(self, inner)?;
            result = Some(match result {
                None => right,
                Some(left) => Expr::binary(left, op, right),
            });
        }
        result.ok_or_else(|| missing("operand"))
    }

    fn parse_or_expr(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        self.parse_logical(pair, BinaryOp::Or, Self::parse_xor_expr)
    }

    fn parse_xor_expr(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        self.parse_logical(pair, BinaryOp::Xor, Self::parse_and_expr)
    }

    fn parse_and_expr(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        self.parse_logical(pair, BinaryOp::And, Self::parse_not_expr)
    }

    /// `NOT* comparison`; every NOT is kept (NOT NOT null is null, not a no-op on truthiness)
    fn parse_not_expr(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut nots = 0usize;
        let mut operand = None;
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::kw_not => nots += 1,
                Rule::comparison => operand = Some(self.parse_comparison(inner)?),
                _ => {}
            }
        }
        let mut result = operand.ok_or_else(|| missing("comparison"))?;
        for _ in 0..nots {
            result = Expr::unary(UnaryOp::Not, result);
        }
        Ok(result)
    }

    /// Comparisons chain: `a < b < c` is `a < b AND b < c`
    fn parse_comparison(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut inner = pair.into_inner();
        let mut left =
            self.parse_predicate_expr(inner.next().ok_or_else(|| missing("operand"))?)?;
        let mut chain: Option<Expr> = None;
        while let Some(op_pair) = inner.next() {
            let op = match op_pair.as_str() {
                "=" => BinaryOp::Eq,
                "<>" | "!=" => BinaryOp::Neq,
                "<" => BinaryOp::Lt,
                "<=" => BinaryOp::Lte,
                ">" => BinaryOp::Gt,
                ">=" => BinaryOp::Gte,
                "=~" => BinaryOp::RegexMatch,
                other => {
                    return Err(CypherError::internal(format!(
                        "unexpected comparison operator '{other}'"
                    )))
                }
            };
            let right =
                self.parse_predicate_expr(inner.next().ok_or_else(|| missing("operand"))?)?;
            let link = Expr::binary(left, op, right.clone());
            chain = Some(match chain {
                None => link,
                Some(c) => c.and(link),
            });
            left = right;
        }
        Ok(chain.unwrap_or(left))
    }

    /// `addition (IS [NOT] NULL | STARTS WITH x | ENDS WITH x | CONTAINS x | IN x)*`
    fn parse_predicate_expr(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut inner = pair.into_inner();
        let mut result = self.parse_addition(inner.next().ok_or_else(|| missing("operand"))?)?;
        while let Some(op) = inner.next() {
            let rule = op.as_rule();
            result = match rule {
                Rule::is_null | Rule::is_not_null => Expr::IsNull {
                    expr: Box::new(result),
                    negated: rule == Rule::is_not_null,
                },
                _ => {
                    let right =
                        self.parse_addition(inner.next().ok_or_else(|| missing("operand"))?)?;
                    let string = |predicate| Expr::StringPredicate {
                        expr: Box::new(result.clone()),
                        predicate,
                        pattern: Box::new(right.clone()),
                    };
                    match rule {
                        Rule::starts_with => string(StringPredicateType::StartsWith),
                        Rule::ends_with => string(StringPredicateType::EndsWith),
                        Rule::contains_op => string(StringPredicateType::Contains),
                        Rule::in_op => Expr::In {
                            expr: Box::new(result),
                            list: Box::new(right),
                            negated: false,
                        },
                        other => {
                            return Err(CypherError::internal(format!(
                                "unexpected predicate {other:?}"
                            )))
                        }
                    }
                }
            };
        }
        Ok(result)
    }

    /// Fold `operand (op operand)*` where each op token maps to a binary operator
    fn parse_arithmetic(
        &self,
        pair: Pair<Rule>,
        operand: fn(&Self, Pair<Rule>) -> CypherResult<Expr>,
    ) -> CypherResult<Expr> {
        let mut inner = pair.into_inner();
        let mut result = operand(self, inner.next().ok_or_else(|| missing("operand"))?)?;
        while let Some(op) = inner.next() {
            let op = match op.as_str() {
                "+" => BinaryOp::Add,
                "-" => BinaryOp::Sub,
                "*" => BinaryOp::Mul,
                "/" => BinaryOp::Div,
                "%" => BinaryOp::Mod,
                other => {
                    return Err(CypherError::internal(format!(
                        "unexpected arithmetic operator '{other}'"
                    )))
                }
            };
            let right = operand(self, inner.next().ok_or_else(|| missing("operand"))?)?;
            result = Expr::binary(result, op, right);
        }
        Ok(result)
    }

    fn parse_addition(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        self.parse_arithmetic(pair, Self::parse_multiplication)
    }

    fn parse_multiplication(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        self.parse_arithmetic(pair, Self::parse_power)
    }

    /// `^` is left-associative in openCypher
    fn parse_power(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut result: Option<Expr> = None;
        for inner in pair.into_inner() {
            let right = self.parse_unary(inner)?;
            result = Some(match result {
                None => right,
                Some(left) => Expr::binary(left, BinaryOp::Pow, right),
            });
        }
        result.ok_or_else(|| missing("operand"))
    }

    /// `(-|+)* postfix`; a negated integer literal is folded so i64::MIN is expressible
    fn parse_unary(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut negations = 0usize;
        let mut operand = None;
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::sign_op => {
                    if inner.as_str() == "-" {
                        negations += 1;
                    }
                }
                Rule::postfix => operand = Some(inner),
                _ => {}
            }
        }
        let operand = operand.ok_or_else(|| missing("unary operand"))?;
        if negations % 2 == 1 {
            if let Some(digits) = integer_literal_text(&operand) {
                let v: i64 = format!("-{digits}").parse().map_err(|_| {
                    CypherError::syntax(0, 0, format!("Integer out of range: -{digits}"))
                })?;
                return Ok(Expr::Literal(CypherValue::Int(v)));
            }
        }
        let mut result = self.parse_postfix(operand)?;
        for _ in 0..negations {
            result = Expr::unary(UnaryOp::Neg, result);
        }
        Ok(result)
    }

    /// `atom (.prop | [i] | [a..b])* (:Label)*`
    fn parse_postfix(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let mut result: Option<Expr> = None;
        for inner in pair.into_inner() {
            let base = |r: Option<Expr>| r.ok_or_else(|| missing("postfix base"));
            result = Some(match inner.as_rule() {
                Rule::atom => self.parse_atom(inner)?,
                Rule::property_access => {
                    let name = inner
                        .into_inner()
                        .next()
                        .map(|p| ident(&p))
                        .ok_or_else(|| missing("property name"))?;
                    Expr::property(base(result)?, name)
                }
                Rule::index => {
                    let index = inner
                        .into_inner()
                        .next()
                        .ok_or_else(|| missing("index expression"))?;
                    Expr::Index {
                        expr: Box::new(base(result)?),
                        index: Box::new(self.parse_expression(index)?),
                    }
                }
                Rule::slice => {
                    let (mut from, mut to) = (None, None);
                    for bound in inner.into_inner() {
                        let is_from = bound.as_rule() == Rule::slice_from;
                        let e = bound
                            .into_inner()
                            .next()
                            .ok_or_else(|| missing("slice bound"))?;
                        let parsed = Some(Box::new(self.parse_expression(e)?));
                        if is_from {
                            from = parsed;
                        } else {
                            to = parsed;
                        }
                    }
                    Expr::Slice {
                        expr: Box::new(base(result)?),
                        from,
                        to,
                    }
                }
                Rule::label_list => Expr::HasLabels {
                    expr: Box::new(base(result)?),
                    labels: inner.into_inner().map(|l| ident(&l)).collect(),
                },
                other => {
                    return Err(CypherError::internal(format!(
                        "unexpected postfix {other:?}"
                    )))
                }
            });
        }
        result.ok_or_else(|| missing("expression"))
    }

    /// Parse atom
    pub(super) fn parse_atom(&self, pair: Pair<Rule>) -> CypherResult<Expr> {
        let inner = pair.into_inner().next().ok_or_else(|| missing("atom"))?;
        match inner.as_rule() {
            Rule::literal => Ok(Expr::Literal(self.parse_literal(inner)?)),
            Rule::parameter => self.parse_parameter(inner),
            Rule::function_call => self.parse_function_call(inner),
            Rule::case_expr => self.parse_case_expr(inner),
            Rule::list_literal => self.parse_list_literal(inner),
            Rule::map_literal => Ok(Expr::Map(self.parse_map_literal(inner)?)),
            Rule::expression => self.parse_expression(inner),
            Rule::identifier => Ok(Expr::variable(ident(&inner))),
            other => Err(CypherError::internal(format!("unexpected atom {other:?}"))),
        }
    }
}

/// The digits of a postfix that is exactly an integer literal (no postfix operators)
fn integer_literal_text(postfix: &Pair<Rule>) -> Option<String> {
    let mut parts = postfix.clone().into_inner();
    let atom = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    let literal = atom.into_inner().next()?;
    if literal.as_rule() != Rule::literal {
        return None;
    }
    let number = literal.into_inner().next()?;
    if number.as_rule() != Rule::number_literal {
        return None;
    }
    let int = number.into_inner().next()?;
    (int.as_rule() == Rule::integer).then(|| int.as_str().to_string())
}
