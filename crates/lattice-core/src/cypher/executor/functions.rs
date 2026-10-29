//! Scalar functions. Arity was checked by the planner; every function returns null for a null
//! argument (except `coalesce` and `exists`, which exist to inspect nulls) and refuses
//! arguments of the wrong type with an error naming the function.

use super::arith::scalar_text;
use super::compare::I64_RANGE;
use super::{ExecutionContext, QueryExecutor};
use crate::cypher::ast::ScalarFn;
use crate::cypher::error::{CypherError, CypherResult};
use crate::types::value::CypherValue;

/// Upper bound on the list `range()` may build (DoS protection)
const MAX_RANGE_LEN: i64 = 1_000_000;

fn wrong(func: ScalarFn, v: &CypherValue) -> CypherError {
    CypherError::invalid(format!("{}()", func.name()), v.type_name())
}

fn int_arg(func: ScalarFn, v: &CypherValue) -> CypherResult<i64> {
    match v {
        CypherValue::Int(i) => Ok(*i),
        other => Err(wrong(func, other)),
    }
}

fn str_arg(func: ScalarFn, v: &CypherValue) -> CypherResult<&str> {
    match v {
        CypherValue::String(s) => Ok(s),
        other => Err(wrong(func, other)),
    }
}

fn num_arg(func: ScalarFn, v: &CypherValue) -> CypherResult<f64> {
    match v {
        CypherValue::Int(i) => Ok(*i as f64),
        CypherValue::Float(f) => Ok(*f),
        other => Err(wrong(func, other)),
    }
}

/// Truncate a float to INTEGER (null when it does not fit)
fn truncate(f: f64) -> CypherValue {
    if I64_RANGE.contains(&f.trunc()) {
        CypherValue::Int(f.trunc() as i64)
    } else {
        CypherValue::Null
    }
}

fn string(s: impl AsRef<str>) -> CypherValue {
    CypherValue::string(s)
}

impl QueryExecutor {
    /// Evaluate a resolved scalar function
    pub(crate) fn call_scalar(
        &self,
        func: ScalarFn,
        args: &[CypherValue],
        ctx: &ExecutionContext,
    ) -> CypherResult<CypherValue> {
        use CypherValue as V;
        use ScalarFn as F;
        match func {
            F::Coalesce => {
                return Ok(args
                    .iter()
                    .find(|a| !a.is_null())
                    .cloned()
                    .unwrap_or(V::Null))
            }
            F::Exists => return Ok(V::Bool(!args[0].is_null())),
            _ => {}
        }
        if args.iter().any(CypherValue::is_null) {
            return Ok(V::Null);
        }
        let a = &args[0];
        Ok(match func {
            F::Coalesce | F::Exists => unreachable!("handled above"),
            F::Labels | F::Id | F::Keys | F::Properties | F::Type | F::StartNode | F::EndNode => {
                return self.call_graph(func, a, ctx)
            }
            F::Size | F::Length => match a {
                V::List(l) => V::Int(l.len() as i64),
                V::String(s) => V::Int(s.chars().count() as i64),
                other => return Err(wrong(func, other)),
            },
            F::Head | F::Last => match a {
                V::List(l) => {
                    let item = if func == F::Head { l.first() } else { l.last() };
                    item.cloned().unwrap_or(V::Null)
                }
                other => return Err(wrong(func, other)),
            },
            F::Tail => match a {
                V::List(l) => V::List(l.iter().skip(1).cloned().collect()),
                other => return Err(wrong(func, other)),
            },
            F::Reverse => match a {
                V::List(l) => V::List(l.iter().rev().cloned().collect()),
                V::String(s) => string(s.chars().rev().collect::<String>()),
                other => return Err(wrong(func, other)),
            },
            F::Range => {
                let start = int_arg(func, a)?;
                let end = int_arg(func, &args[1])?;
                let step = args.get(2).map_or(Ok(1), |s| int_arg(func, s))?;
                if step == 0 {
                    return Err(CypherError::invalid("range() step", "zero"));
                }
                let len = (end as i128 - start as i128) / step as i128 + 1;
                if len > MAX_RANGE_LEN as i128 {
                    return Err(CypherError::QueryTooComplex {
                        message: format!("range() of {len} elements exceeds {MAX_RANGE_LEN}"),
                    });
                }
                let mut out = Vec::new();
                let mut x = start;
                while (step > 0 && x <= end) || (step < 0 && x >= end) {
                    out.push(V::Int(x));
                    x = match x.checked_add(step) {
                        Some(n) => n,
                        None => break,
                    };
                }
                V::List(out)
            }
            F::ToString => match scalar_text(a) {
                Some(s) => string(s),
                None => return Err(wrong(func, a)),
            },
            F::ToInteger => match a {
                V::Int(i) => V::Int(*i),
                V::Float(f) => truncate(*f),
                V::Bool(b) => V::Int(i64::from(*b)),
                V::String(s) => {
                    let s = s.trim();
                    match s.parse::<i64>() {
                        Ok(i) => V::Int(i),
                        Err(_) => s.parse::<f64>().map_or(V::Null, truncate),
                    }
                }
                other => return Err(wrong(func, other)),
            },
            F::ToFloat => match a {
                V::Int(i) => V::Float(*i as f64),
                V::Float(f) => V::Float(*f),
                V::String(s) => s.trim().parse::<f64>().map_or(V::Null, V::Float),
                other => return Err(wrong(func, other)),
            },
            F::ToBoolean => match a {
                V::Bool(b) => V::Bool(*b),
                V::Int(i) => V::Bool(*i != 0),
                V::String(s) => match s.trim().to_ascii_lowercase().as_str() {
                    "true" => V::Bool(true),
                    "false" => V::Bool(false),
                    _ => V::Null,
                },
                other => return Err(wrong(func, other)),
            },
            F::ToLower => string(str_arg(func, a)?.to_lowercase()),
            F::ToUpper => string(str_arg(func, a)?.to_uppercase()),
            F::Trim => string(str_arg(func, a)?.trim()),
            F::LTrim => string(str_arg(func, a)?.trim_start()),
            F::RTrim => string(str_arg(func, a)?.trim_end()),
            F::Replace => {
                let (s, from, to) = (
                    str_arg(func, a)?,
                    str_arg(func, &args[1])?,
                    str_arg(func, &args[2])?,
                );
                if from.is_empty() {
                    return Err(CypherError::invalid(
                        "replace() search string",
                        "empty STRING",
                    ));
                }
                string(s.replace(from, to))
            }
            F::Substring => {
                let s = str_arg(func, a)?;
                let start = int_arg(func, &args[1])?;
                let len = args.get(2).map(|l| int_arg(func, l)).transpose()?;
                if start < 0 || len.is_some_and(|l| l < 0) {
                    return Err(CypherError::invalid(
                        "substring() negative index",
                        "INTEGER",
                    ));
                }
                let chars = s.chars().skip(start as usize);
                string(match len {
                    Some(l) => chars.take(l as usize).collect::<String>(),
                    None => chars.collect::<String>(),
                })
            }
            F::Split => {
                let (s, delim) = (str_arg(func, a)?, str_arg(func, &args[1])?);
                V::List(if delim.is_empty() {
                    s.chars().map(|c| string(c.to_string())).collect()
                } else {
                    s.split(delim).map(string).collect()
                })
            }
            F::Abs => match a {
                V::Int(i) => {
                    V::Int(
                        i.checked_abs()
                            .ok_or_else(|| CypherError::ArithmeticOverflow {
                                operation: "abs()".to_string(),
                            })?,
                    )
                }
                other => V::Float(num_arg(func, other)?.abs()),
            },
            F::Ceil => V::Float(num_arg(func, a)?.ceil()),
            F::Floor => V::Float(num_arg(func, a)?.floor()),
            F::Round => V::Float(num_arg(func, a)?.round()),
            F::Sqrt => V::Float(num_arg(func, a)?.sqrt()),
            F::Sign => {
                let x = num_arg(func, a)?;
                V::Int(if x > 0.0 {
                    1
                } else if x < 0.0 {
                    -1
                } else {
                    0
                })
            }
        })
    }

    /// Functions over nodes and relationships
    fn call_graph(
        &self,
        func: ScalarFn,
        a: &CypherValue,
        ctx: &ExecutionContext,
    ) -> CypherResult<CypherValue> {
        use CypherValue as V;
        use ScalarFn as F;
        let point = |id: u64| -> CypherResult<std::sync::Arc<crate::types::point::Point>> {
            ctx.get_point_cached(id)
                .ok_or(CypherError::NodeNotFound { id })
        };
        Ok(match (func, a) {
            (F::Id, V::NodeRef(id)) => V::Int(*id as i64),
            (F::Id, V::RelationshipRef(_)) => {
                return Err(CypherError::unsupported(
                    "id() of a relationship (LatticeDB relationships have no stable id)",
                ))
            }
            (F::Labels, V::NodeRef(id)) => V::List(self.point_labels(point(*id)?.as_ref())?),
            (F::Keys, V::NodeRef(id)) => V::List(
                self.point_properties(point(*id)?.as_ref())?
                    .into_iter()
                    .map(|(k, _)| string(k))
                    .collect(),
            ),
            (F::Properties, V::NodeRef(id)) => V::Map(self.point_properties(point(*id)?.as_ref())?),
            (F::Keys, V::Map(m)) => V::List(m.iter().map(|(k, _)| string(k)).collect()),
            (F::Properties, V::Map(m)) => V::Map(m.clone()),
            (F::Keys, V::RelationshipRef(_)) => V::List(vec![string("weight")]),
            (F::Properties, V::RelationshipRef(h)) => V::Map(vec![(
                "weight".to_string(),
                V::Float(f64::from(ctx.rel(*h)?.weight)),
            )]),
            (F::Type, V::RelationshipRef(h)) => V::String(ctx.rel(*h)?.rel_type.clone()),
            (F::StartNode, V::RelationshipRef(h)) => V::NodeRef(ctx.rel(*h)?.src),
            (F::EndNode, V::RelationshipRef(h)) => V::NodeRef(ctx.rel(*h)?.dst),
            (_, other) => return Err(wrong(func, other)),
        })
    }
}
