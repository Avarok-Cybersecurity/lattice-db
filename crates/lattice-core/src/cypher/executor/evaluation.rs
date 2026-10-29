//! Expression evaluation for Cypher queries.
//!
//! Every `Expr` variant the planner can hand over is evaluated here or in `functions` /
//! `access`; nothing falls through to a catch-all. Semantics follow openCypher: null
//! propagates, comparisons and boolean operators are three-valued, and a WHERE keeps a row
//! only when its predicate is `true`.

use super::compare::{and3, compare, equals, or3, order_cmp, tri, xor3};
use super::{ExecutionContext, QueryExecutor};
use crate::cypher::ast::{BinaryOp, Expr, StringPredicateType, UnaryOp};
use crate::cypher::error::{CypherError, CypherResult};
use crate::types::value::CypherValue;
use crate::types::SharedStr;

/// A boolean operand: true/false/null, anything else is a type error naming the operator
pub(crate) fn as_bool(v: &CypherValue, operator: &str) -> CypherResult<Option<bool>> {
    match v {
        CypherValue::Bool(b) => Ok(Some(*b)),
        CypherValue::Null => Ok(None),
        other => Err(CypherError::type_mismatch(
            format!("BOOLEAN operands for {operator}"),
            other.type_name(),
        )),
    }
}

impl QueryExecutor {
    /// Evaluate a WHERE-style predicate: only `true` keeps the row
    pub(crate) fn evaluate_predicate(
        &self,
        predicate: &Expr,
        row: &[CypherValue],
        ctx: &ExecutionContext,
    ) -> CypherResult<bool> {
        match self.evaluate_expr(predicate, row, ctx)? {
            CypherValue::Bool(b) => Ok(b),
            CypherValue::Null => Ok(false),
            other => Err(CypherError::type_mismatch(
                "a BOOLEAN predicate in WHERE",
                other.type_name(),
            )),
        }
    }

    /// Evaluate an expression against one row
    pub(crate) fn evaluate_expr(
        &self,
        expr: &Expr,
        row: &[CypherValue],
        ctx: &ExecutionContext,
    ) -> CypherResult<CypherValue> {
        let eval = |e: &Expr| self.evaluate_expr(e, row, ctx);
        match expr {
            Expr::Literal(value) => Ok(value.clone()),

            Expr::Variable(name) => match ctx.get_variable_column(name.as_str()) {
                Some(column) => row.get(column).cloned().ok_or_else(|| {
                    CypherError::internal(format!(
                        "Variable '{name}' bound to column {column} but row has only {} columns",
                        row.len()
                    ))
                }),
                // CREATE property values may still name a parameter without `$`.
                None => ctx
                    .parameters
                    .get(name.as_str())
                    .cloned()
                    .ok_or_else(|| CypherError::unknown_variable(name.as_str())),
            },

            Expr::Parameter(name) => ctx.parameters.get(name.as_str()).cloned().ok_or_else(|| {
                CypherError::MissingParameter {
                    name: name.to_string(),
                }
            }),

            Expr::Property { expr, property } => self.property_of(&eval(expr)?, property, ctx),

            Expr::List(items) => Ok(CypherValue::List(
                items.iter().map(eval).collect::<CypherResult<_>>()?,
            )),

            Expr::Map(map) => Ok(CypherValue::Map(
                map.entries
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), eval(v)?)))
                    .collect::<CypherResult<_>>()?,
            )),

            Expr::BinaryOp { left, op, right } => match op {
                BinaryOp::And => {
                    let l = as_bool(&eval(left)?, "AND")?;
                    if l == Some(false) {
                        return Ok(CypherValue::Bool(false));
                    }
                    Ok(tri(and3(l, as_bool(&eval(right)?, "AND")?)))
                }
                BinaryOp::Or => {
                    let l = as_bool(&eval(left)?, "OR")?;
                    if l == Some(true) {
                        return Ok(CypherValue::Bool(true));
                    }
                    Ok(tri(or3(l, as_bool(&eval(right)?, "OR")?)))
                }
                _ => self.evaluate_binary_op(*op, &eval(left)?, &eval(right)?),
            },

            Expr::UnaryOp { op, expr } => self.evaluate_unary_op(*op, &eval(expr)?),

            Expr::IsNull { expr, negated } => {
                Ok(CypherValue::Bool(eval(expr)?.is_null() != *negated))
            }

            Expr::In {
                expr,
                list,
                negated,
            } => {
                let found = self.in_list(&eval(expr)?, &eval(list)?)?;
                Ok(tri(if *negated { found.map(|b| !b) } else { found }))
            }

            Expr::StringPredicate {
                expr,
                predicate,
                pattern,
            } => Ok(match (eval(expr)?, eval(pattern)?) {
                (CypherValue::String(s), CypherValue::String(p)) => {
                    CypherValue::Bool(match predicate {
                        StringPredicateType::StartsWith => s.starts_with(&*p),
                        StringPredicateType::EndsWith => s.ends_with(&*p),
                        StringPredicateType::Contains => s.contains(&*p),
                    })
                }
                // Null, or a non-string operand: the predicate is undefined.
                _ => CypherValue::Null,
            }),

            Expr::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                let subject = operand.as_deref().map(eval).transpose()?;
                for (when, then) in when_clauses {
                    let w = eval(when)?;
                    let hit = match &subject {
                        Some(s) => equals(s, &w) == Some(true),
                        None => as_bool(&w, "CASE WHEN")? == Some(true),
                    };
                    if hit {
                        return eval(then);
                    }
                }
                else_clause.as_deref().map_or(Ok(CypherValue::Null), eval)
            }

            Expr::HasLabels { expr, labels } => match eval(expr)? {
                CypherValue::Null => Ok(CypherValue::Null),
                CypherValue::NodeRef(id) => {
                    for label in labels {
                        if !ctx.node_has_label(id, label)? {
                            return Ok(CypherValue::Bool(false));
                        }
                    }
                    Ok(CypherValue::Bool(true))
                }
                other => Err(CypherError::invalid("label predicate", other.type_name())),
            },

            Expr::Index { expr, index } => self.index_value(&eval(expr)?, &eval(index)?, ctx),

            Expr::Slice { expr, from, to } => {
                let from = from.as_deref().map(eval).transpose()?;
                let to = to.as_deref().map(eval).transpose()?;
                self.slice_value(&eval(expr)?, from.as_ref(), to.as_ref())
            }

            Expr::Scalar { func, args } => {
                let args = args.iter().map(eval).collect::<CypherResult<Vec<_>>>()?;
                self.call_scalar(*func, &args, ctx)
            }

            Expr::FunctionCall { name, .. } => Err(CypherError::unsupported(format!(
                "function {name}() outside a planned read query"
            ))),

            Expr::Star => Err(CypherError::internal("`*` reached the evaluator")),

            Expr::Exists { .. } => Err(CypherError::unsupported("EXISTS { … } subquery")),
        }
    }

    /// `x IN list`: true if an element equals x, null if none does but some comparison was null
    fn in_list(&self, x: &CypherValue, list: &CypherValue) -> CypherResult<Option<bool>> {
        let items = match list {
            CypherValue::Null => return Ok(None),
            CypherValue::List(items) => items,
            other => {
                return Err(CypherError::type_mismatch(
                    "a LIST after IN",
                    other.type_name(),
                ))
            }
        };
        let mut saw_null = false;
        for item in items {
            match equals(x, item) {
                Some(true) => return Ok(Some(true)),
                None => saw_null = true,
                Some(false) => {}
            }
        }
        Ok(if saw_null { None } else { Some(false) })
    }

    /// Evaluate a binary operation (AND/OR here are the strict, non-short-circuit forms)
    pub(crate) fn evaluate_binary_op(
        &self,
        op: BinaryOp,
        left: &CypherValue,
        right: &CypherValue,
    ) -> CypherResult<CypherValue> {
        let cmp = |test: fn(std::cmp::Ordering) -> bool| match compare(left, right) {
            Ok(Some(ord)) => CypherValue::Bool(test(ord)),
            Ok(None) => CypherValue::Bool(false),
            Err(()) => CypherValue::Null,
        };
        match op {
            BinaryOp::And => Ok(tri(and3(as_bool(left, "AND")?, as_bool(right, "AND")?))),
            BinaryOp::Or => Ok(tri(or3(as_bool(left, "OR")?, as_bool(right, "OR")?))),
            BinaryOp::Xor => Ok(tri(xor3(as_bool(left, "XOR")?, as_bool(right, "XOR")?))),
            BinaryOp::Eq => Ok(tri(equals(left, right))),
            BinaryOp::Neq => Ok(tri(equals(left, right).map(|b| !b))),
            _ if left.is_null() || right.is_null() => Ok(CypherValue::Null),
            BinaryOp::Lt => Ok(cmp(|o| o.is_lt())),
            BinaryOp::Lte => Ok(cmp(|o| o.is_le())),
            BinaryOp::Gt => Ok(cmp(|o| o.is_gt())),
            BinaryOp::Gte => Ok(cmp(|o| o.is_ge())),
            BinaryOp::Add => self.add_values(left, right),
            BinaryOp::Sub => self.sub_values(left, right),
            BinaryOp::Mul => self.mul_values(left, right),
            BinaryOp::Div => self.div_values(left, right),
            BinaryOp::Mod => self.mod_values(left, right),
            BinaryOp::Pow => self.pow_values(left, right),
            BinaryOp::Concat => match (left, right) {
                (CypherValue::String(a), CypherValue::String(b)) => {
                    Ok(CypherValue::String(SharedStr::from(format!("{a}{b}"))))
                }
                _ => Err(CypherError::invalid(
                    "string concatenation",
                    format!("{} and {}", left.type_name(), right.type_name()),
                )),
            },
            BinaryOp::RegexMatch => Err(CypherError::unsupported("regular expression match (=~)")),
        }
    }

    /// Evaluate a unary operation
    pub(crate) fn evaluate_unary_op(
        &self,
        op: UnaryOp,
        value: &CypherValue,
    ) -> CypherResult<CypherValue> {
        match op {
            UnaryOp::Not => Ok(tri(as_bool(value, "NOT")?.map(|b| !b))),
            UnaryOp::Neg => self.neg_value(value),
            UnaryOp::Pos => match value {
                CypherValue::Null | CypherValue::Int(_) | CypherValue::Float(_) => {
                    Ok(value.clone())
                }
                other => Err(CypherError::invalid("unary plus", other.type_name())),
            },
        }
    }

    /// The ORDER BY total order (see `compare::order_cmp`)
    pub(crate) fn compare_values(
        &self,
        left: &CypherValue,
        right: &CypherValue,
    ) -> std::cmp::Ordering {
        order_cmp(left, right)
    }
}
