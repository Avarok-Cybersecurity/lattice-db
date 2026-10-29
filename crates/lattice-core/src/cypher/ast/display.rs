//! Cypher text for expressions.
//!
//! An unaliased RETURN column is named by its expression, as openCypher engines do
//! (`RETURN m.id, count(*)` has columns `m.id` and `count(*)`), and error messages quote
//! expressions in the same form. Operators are parenthesised only where precedence needs it.

use super::{BinaryOp, Expr, StringPredicateType, UnaryOp};
use crate::types::value::CypherValue;
use std::fmt;

/// Binding strength, loosest first (openCypher precedence)
fn precedence(e: &Expr) -> u8 {
    match e {
        Expr::BinaryOp { op, .. } => match op {
            BinaryOp::Or => 1,
            BinaryOp::Xor => 2,
            BinaryOp::And => 3,
            BinaryOp::Eq
            | BinaryOp::Neq
            | BinaryOp::Lt
            | BinaryOp::Lte
            | BinaryOp::Gt
            | BinaryOp::Gte
            | BinaryOp::RegexMatch => 5,
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Concat => 7,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => 8,
            BinaryOp::Pow => 9,
        },
        Expr::UnaryOp {
            op: UnaryOp::Not, ..
        } => 4,
        Expr::IsNull { .. } | Expr::In { .. } | Expr::StringPredicate { .. } => 6,
        Expr::UnaryOp { .. } => 10,
        _ => 11,
    }
}

fn op_text(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add | BinaryOp::Concat => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Mod => "%",
        BinaryOp::Pow => "^",
        BinaryOp::Eq => "=",
        BinaryOp::Neq => "<>",
        BinaryOp::Lt => "<",
        BinaryOp::Lte => "<=",
        BinaryOp::Gt => ">",
        BinaryOp::Gte => ">=",
        BinaryOp::And => "AND",
        BinaryOp::Or => "OR",
        BinaryOp::Xor => "XOR",
        BinaryOp::RegexMatch => "=~",
    }
}

/// Write `child`, parenthesised when it binds looser than `min`
fn operand(f: &mut fmt::Formatter<'_>, child: &Expr, min: u8) -> fmt::Result {
    if precedence(child) < min {
        write!(f, "({child})")
    } else {
        write!(f, "{child}")
    }
}

fn list(f: &mut fmt::Formatter<'_>, items: &[Expr]) -> fmt::Result {
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{item}")?;
    }
    Ok(())
}

fn literal(f: &mut fmt::Formatter<'_>, v: &CypherValue) -> fmt::Result {
    match v {
        CypherValue::Null => f.write_str("null"),
        CypherValue::Bool(b) => write!(f, "{b}"),
        CypherValue::Int(i) => write!(f, "{i}"),
        CypherValue::Float(x) => write!(f, "{x:?}"),
        CypherValue::String(s) => write!(f, "'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
        other => write!(f, "{other:?}"),
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = precedence(self);
        match self {
            Expr::Star => f.write_str("*"),
            Expr::Literal(v) => literal(f, v),
            Expr::Variable(name) => f.write_str(name),
            Expr::Parameter(name) => write!(f, "${name}"),
            Expr::Property { expr, property } => {
                operand(f, expr, 11)?;
                write!(f, ".{property}")
            }
            Expr::List(items) => {
                f.write_str("[")?;
                list(f, items)?;
                f.write_str("]")
            }
            Expr::Map(map) => {
                f.write_str("{")?;
                for (i, (k, v)) in map.entries.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                f.write_str("}")
            }
            Expr::BinaryOp { left, op, right } => {
                operand(f, left, p)?;
                write!(f, " {} ", op_text(*op))?;
                // Left-associative: a right operand of equal precedence needs parentheses.
                operand(f, right, p + 1)
            }
            Expr::UnaryOp { op, expr } => {
                match op {
                    UnaryOp::Not => f.write_str("NOT ")?,
                    UnaryOp::Neg => f.write_str("-")?,
                    UnaryOp::Pos => f.write_str("+")?,
                }
                operand(f, expr, p)
            }
            Expr::FunctionCall {
                name,
                args,
                distinct,
            } => {
                write!(f, "{name}(")?;
                if *distinct {
                    f.write_str("DISTINCT ")?;
                }
                list(f, args)?;
                f.write_str(")")
            }
            Expr::Scalar { func, args } => {
                write!(f, "{}(", func.name())?;
                list(f, args)?;
                f.write_str(")")
            }
            Expr::Case {
                operand: subject,
                when_clauses,
                else_clause,
            } => {
                f.write_str("CASE")?;
                if let Some(s) = subject {
                    write!(f, " {s}")?;
                }
                for (w, t) in when_clauses {
                    write!(f, " WHEN {w} THEN {t}")?;
                }
                if let Some(e) = else_clause {
                    write!(f, " ELSE {e}")?;
                }
                f.write_str(" END")
            }
            Expr::IsNull { expr, negated } => {
                operand(f, expr, p + 1)?;
                f.write_str(if *negated { " IS NOT NULL" } else { " IS NULL" })
            }
            Expr::In {
                expr,
                list: l,
                negated,
            } => {
                if *negated {
                    f.write_str("NOT ")?;
                }
                operand(f, expr, p + 1)?;
                f.write_str(" IN ")?;
                operand(f, l, p + 1)
            }
            Expr::StringPredicate {
                expr,
                predicate,
                pattern,
            } => {
                operand(f, expr, p + 1)?;
                f.write_str(match predicate {
                    StringPredicateType::StartsWith => " STARTS WITH ",
                    StringPredicateType::EndsWith => " ENDS WITH ",
                    StringPredicateType::Contains => " CONTAINS ",
                })?;
                operand(f, pattern, p + 1)
            }
            Expr::HasLabels { expr, labels } => {
                operand(f, expr, 11)?;
                for l in labels {
                    write!(f, ":{l}")?;
                }
                Ok(())
            }
            Expr::Index { expr, index } => {
                operand(f, expr, 11)?;
                write!(f, "[{index}]")
            }
            Expr::Slice { expr, from, to } => {
                operand(f, expr, 11)?;
                f.write_str("[")?;
                if let Some(a) = from {
                    write!(f, "{a}")?;
                }
                f.write_str("..")?;
                if let Some(b) = to {
                    write!(f, "{b}")?;
                }
                f.write_str("]")
            }
            Expr::Exists { .. } => f.write_str("EXISTS { … }"),
        }
    }
}
