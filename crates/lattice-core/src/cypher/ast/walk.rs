//! Generic traversal of expression trees.
//!
//! `map_children` rebuilds a node with each direct child replaced, so every rewrite (scope
//! checks, function resolution, aggregate extraction) is written once as "handle the nodes I
//! care about, recurse through the rest".

use super::{Expr, MapLiteral};

type Res<E> = Result<Expr, E>;

fn boxed<E>(e: &Expr, f: &mut impl FnMut(&Expr) -> Res<E>) -> Result<Box<Expr>, E> {
    f(e).map(Box::new)
}

fn opt<E>(
    e: &Option<Box<Expr>>,
    f: &mut impl FnMut(&Expr) -> Res<E>,
) -> Result<Option<Box<Expr>>, E> {
    e.as_ref().map(|x| boxed(x, f)).transpose()
}

fn all<E>(es: &[Expr], f: &mut impl FnMut(&Expr) -> Res<E>) -> Result<Vec<Expr>, E> {
    es.iter().map(f).collect()
}

/// Rebuild `expr` with every direct child expression replaced by `f(child)`
pub fn map_children<E>(expr: &Expr, f: &mut impl FnMut(&Expr) -> Res<E>) -> Res<E> {
    Ok(match expr {
        Expr::Star
        | Expr::Literal(_)
        | Expr::Variable(_)
        | Expr::Parameter(_)
        | Expr::Exists { .. } => expr.clone(),
        Expr::Property { expr, property } => Expr::Property {
            expr: boxed(expr, f)?,
            property: property.clone(),
        },
        Expr::List(items) => Expr::List(all(items, f)?),
        Expr::Map(map) => Expr::Map(MapLiteral {
            entries: map
                .entries
                .iter()
                .map(|(k, v)| Ok((k.clone(), boxed(v, f)?)))
                .collect::<Result<_, E>>()?,
        }),
        Expr::BinaryOp { left, op, right } => Expr::BinaryOp {
            left: boxed(left, f)?,
            op: *op,
            right: boxed(right, f)?,
        },
        Expr::UnaryOp { op, expr } => Expr::UnaryOp {
            op: *op,
            expr: boxed(expr, f)?,
        },
        Expr::FunctionCall {
            name,
            args,
            distinct,
        } => Expr::FunctionCall {
            name: name.clone(),
            args: all(args, f)?,
            distinct: *distinct,
        },
        Expr::Scalar { func, args } => Expr::Scalar {
            func: *func,
            args: all(args, f)?,
        },
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => Expr::Case {
            operand: opt(operand, f)?,
            when_clauses: when_clauses
                .iter()
                .map(|(w, t)| Ok((f(w)?, f(t)?)))
                .collect::<Result<_, E>>()?,
            else_clause: opt(else_clause, f)?,
        },
        Expr::IsNull { expr, negated } => Expr::IsNull {
            expr: boxed(expr, f)?,
            negated: *negated,
        },
        Expr::In {
            expr,
            list,
            negated,
        } => Expr::In {
            expr: boxed(expr, f)?,
            list: boxed(list, f)?,
            negated: *negated,
        },
        Expr::StringPredicate {
            expr,
            predicate,
            pattern,
        } => Expr::StringPredicate {
            expr: boxed(expr, f)?,
            predicate: *predicate,
            pattern: boxed(pattern, f)?,
        },
        Expr::HasLabels { expr, labels } => Expr::HasLabels {
            expr: boxed(expr, f)?,
            labels: labels.clone(),
        },
        Expr::Index { expr, index } => Expr::Index {
            expr: boxed(expr, f)?,
            index: boxed(index, f)?,
        },
        Expr::Slice { expr, from, to } => Expr::Slice {
            expr: boxed(expr, f)?,
            from: opt(from, f)?,
            to: opt(to, f)?,
        },
    })
}

/// True when `pred` holds for `expr` or any expression nested in it
pub fn any(expr: &Expr, pred: &impl Fn(&Expr) -> bool) -> bool {
    if pred(expr) {
        return true;
    }
    let mut found = false;
    // map_children visits every child; the rebuilt tree is discarded.
    let _ = map_children::<()>(expr, &mut |child| {
        found = found || any(child, pred);
        Ok(child.clone())
    });
    found
}
