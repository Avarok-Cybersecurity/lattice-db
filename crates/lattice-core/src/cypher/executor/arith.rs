//! Arithmetic: `+ - * / % ^` and unary minus.
//!
//! Null propagates (callers return null before calling these). INTEGER arithmetic is checked
//! (overflow and integer division by zero are errors); mixing INTEGER and FLOAT gives FLOAT.
//! `+` also concatenates strings (with a number or boolean rendered as text) and lists.

use super::QueryExecutor;
use crate::cypher::error::{CypherError, CypherResult};
use crate::types::value::CypherValue;
use crate::types::SharedStr;

fn invalid(op: &str, a: &CypherValue, b: &CypherValue) -> CypherError {
    CypherError::invalid(op, format!("{} and {}", a.type_name(), b.type_name()))
}

fn overflow(op: &str) -> CypherError {
    CypherError::ArithmeticOverflow {
        operation: op.to_string(),
    }
}

/// Text of a scalar for string concatenation and toString()
pub(crate) fn scalar_text(v: &CypherValue) -> Option<String> {
    match v {
        CypherValue::String(s) => Some(s.to_string()),
        CypherValue::Int(i) => Some(i.to_string()),
        CypherValue::Float(f) => Some(format!("{f:?}")),
        CypherValue::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Both operands as floats, when both are numbers
fn floats(a: &CypherValue, b: &CypherValue) -> Option<(f64, f64)> {
    match (a, b) {
        (CypherValue::Int(x), CypherValue::Float(y)) => Some((*x as f64, *y)),
        (CypherValue::Float(x), CypherValue::Int(y)) => Some((*x, *y as f64)),
        (CypherValue::Float(x), CypherValue::Float(y)) => Some((*x, *y)),
        _ => None,
    }
}

impl QueryExecutor {
    /// `+`
    pub(crate) fn add_values(
        &self,
        left: &CypherValue,
        right: &CypherValue,
    ) -> CypherResult<CypherValue> {
        use CypherValue::*;
        match (left, right) {
            (Int(a), Int(b)) => a
                .checked_add(*b)
                .map(Int)
                .ok_or_else(|| overflow("addition")),
            (List(a), List(b)) => Ok(List(a.iter().chain(b).cloned().collect())),
            (List(a), b) => Ok(List(a.iter().cloned().chain([b.clone()]).collect())),
            (a, List(b)) => Ok(List(
                [a.clone()].into_iter().chain(b.iter().cloned()).collect(),
            )),
            (String(_), _) | (_, String(_)) => match (scalar_text(left), scalar_text(right)) {
                (Some(a), Some(b)) => Ok(String(SharedStr::from(a + &b))),
                _ => Err(invalid("addition", left, right)),
            },
            _ => floats(left, right)
                .map(|(a, b)| Float(a + b))
                .ok_or_else(|| invalid("addition", left, right)),
        }
    }

    /// `-`
    pub(crate) fn sub_values(
        &self,
        left: &CypherValue,
        right: &CypherValue,
    ) -> CypherResult<CypherValue> {
        match (left, right) {
            (CypherValue::Int(a), CypherValue::Int(b)) => a
                .checked_sub(*b)
                .map(CypherValue::Int)
                .ok_or_else(|| overflow("subtraction")),
            _ => floats(left, right)
                .map(|(a, b)| CypherValue::Float(a - b))
                .ok_or_else(|| invalid("subtraction", left, right)),
        }
    }

    /// `*`
    pub(crate) fn mul_values(
        &self,
        left: &CypherValue,
        right: &CypherValue,
    ) -> CypherResult<CypherValue> {
        match (left, right) {
            (CypherValue::Int(a), CypherValue::Int(b)) => a
                .checked_mul(*b)
                .map(CypherValue::Int)
                .ok_or_else(|| overflow("multiplication")),
            _ => floats(left, right)
                .map(|(a, b)| CypherValue::Float(a * b))
                .ok_or_else(|| invalid("multiplication", left, right)),
        }
    }

    /// `/` (integer division truncates; any division by zero is an error)
    pub(crate) fn div_values(
        &self,
        left: &CypherValue,
        right: &CypherValue,
    ) -> CypherResult<CypherValue> {
        match (left, right) {
            (CypherValue::Int(_), CypherValue::Int(0)) => Err(CypherError::DivisionByZero),
            (CypherValue::Int(a), CypherValue::Int(b)) => a
                .checked_div(*b)
                .map(CypherValue::Int)
                .ok_or_else(|| overflow("division")),
            _ => match floats(left, right) {
                Some((_, 0.0)) => Err(CypherError::DivisionByZero),
                Some((a, b)) => Ok(CypherValue::Float(a / b)),
                None => Err(invalid("division", left, right)),
            },
        }
    }

    /// `%` (sign follows the dividend, as in openCypher)
    pub(crate) fn mod_values(
        &self,
        left: &CypherValue,
        right: &CypherValue,
    ) -> CypherResult<CypherValue> {
        match (left, right) {
            (CypherValue::Int(_), CypherValue::Int(0)) => Err(CypherError::DivisionByZero),
            (CypherValue::Int(a), CypherValue::Int(b)) => a
                .checked_rem(*b)
                .map(CypherValue::Int)
                .ok_or_else(|| overflow("modulo")),
            _ => match floats(left, right) {
                Some((_, 0.0)) => Err(CypherError::DivisionByZero),
                Some((a, b)) => Ok(CypherValue::Float(a % b)),
                None => Err(invalid("modulo", left, right)),
            },
        }
    }

    /// `^` (always FLOAT)
    pub(crate) fn pow_values(
        &self,
        left: &CypherValue,
        right: &CypherValue,
    ) -> CypherResult<CypherValue> {
        let as_f = |v: &CypherValue| match v {
            CypherValue::Int(i) => Some(*i as f64),
            CypherValue::Float(f) => Some(*f),
            _ => None,
        };
        match (as_f(left), as_f(right)) {
            (Some(a), Some(b)) => Ok(CypherValue::Float(a.powf(b))),
            _ => Err(invalid("exponentiation", left, right)),
        }
    }

    /// Unary `-`
    pub(crate) fn neg_value(&self, value: &CypherValue) -> CypherResult<CypherValue> {
        match value {
            CypherValue::Null => Ok(CypherValue::Null),
            CypherValue::Int(i) => i
                .checked_neg()
                .map(CypherValue::Int)
                .ok_or_else(|| overflow("negation")),
            CypherValue::Float(f) => Ok(CypherValue::Float(-f)),
            _ => Err(CypherError::invalid("negation", value.type_name())),
        }
    }
}
