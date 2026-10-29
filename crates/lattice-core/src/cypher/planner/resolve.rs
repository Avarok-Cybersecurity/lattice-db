//! Semantic resolution of expressions at plan time.
//!
//! - every variable must be in scope (refused even when no row would ever reach it);
//! - every function call is resolved once against the catalogue (`Expr::Scalar`), with arity
//!   checked, so the executor never looks names up per row;
//! - aggregates are only valid in projections; `Grouping` lifts them out into `AggregateCall`s
//!   and checks that whatever is left references grouping keys only.

use super::AggregateCall;
use crate::cypher::ast::{walk, AggregateKind, BinaryOp, Expr, ScalarFn};
use crate::cypher::error::{CypherError, CypherResult};

/// Where an expression appears, for error messages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Site {
    Where,
    With,
    Return,
    OrderBy,
    Unwind,
    Pattern,
}

impl Site {
    fn name(self) -> &'static str {
        match self {
            Site::Where => "WHERE",
            Site::With => "WITH",
            Site::Return => "RETURN",
            Site::OrderBy => "ORDER BY",
            Site::Unwind => "UNWIND",
            Site::Pattern => "a pattern",
        }
    }
}

/// The aggregate kind a call names, if it is an aggregate
pub(super) fn aggregate_kind(e: &Expr) -> Option<AggregateKind> {
    match e {
        Expr::FunctionCall { name, .. } => AggregateKind::from_name(name),
        _ => None,
    }
}

/// True when `e` contains an aggregate call anywhere
pub(super) fn contains_aggregate(e: &Expr) -> bool {
    walk::any(e, &|x| aggregate_kind(x).is_some())
}

/// Resolve `expr` against `scope` (see module docs). Aggregates are refused here.
pub(super) fn resolve(expr: &Expr, scope: &[String], site: Site) -> CypherResult<Expr> {
    match expr {
        Expr::Variable(v) => {
            if scope.iter().any(|s| s == v) {
                Ok(expr.clone())
            } else {
                Err(CypherError::unknown_variable(v.as_str()))
            }
        }
        Expr::Star => Err(CypherError::syntax(
            0,
            0,
            "`*` is only valid as RETURN *, WITH * or count(*)",
        )),
        Expr::FunctionCall {
            name,
            args,
            distinct,
        } => {
            if AggregateKind::from_name(name).is_some() {
                return Err(CypherError::aggregation(format!(
                    "aggregate function {name}() is not allowed in {}",
                    site.name()
                )));
            }
            let func = ScalarFn::from_name(name)
                .ok_or_else(|| CypherError::UnknownFunction { name: name.clone() })?;
            if *distinct {
                return Err(CypherError::syntax(
                    0,
                    0,
                    format!("DISTINCT is only valid in aggregate functions, not {name}()"),
                ));
            }
            let (min, max) = func.arity();
            if args.len() < min || args.len() > max {
                let expected = match (min, max) {
                    (a, b) if a == b => a.to_string(),
                    (a, usize::MAX) => format!("at least {a}"),
                    (a, b) => format!("{a} to {b}"),
                };
                return Err(CypherError::InvalidArgumentCount {
                    function: func.name().to_string(),
                    expected,
                    actual: args.len(),
                });
            }
            let args = args
                .iter()
                .map(|a| resolve(a, scope, site))
                .collect::<CypherResult<_>>()?;
            Ok(Expr::Scalar { func, args })
        }
        Expr::BinaryOp {
            op: BinaryOp::RegexMatch,
            ..
        } => Err(CypherError::unsupported("regular expression match (=~)")),
        Expr::Exists { .. } => Err(CypherError::unsupported("EXISTS { … } subquery")),
        _ => walk::map_children(expr, &mut |c| resolve(c, scope, site)),
    }
}

/// Hidden variable the `Aggregate` operator binds to grouping key `i`
pub(super) fn key_var(i: usize) -> String {
    format!("#key{i}")
}

/// Hidden variable the `Aggregate` operator binds to aggregate `j`
pub(super) fn agg_var(j: usize) -> String {
    format!("#agg{j}")
}

/// Lifts aggregates out of projection items (implicit GROUP BY on the other items)
pub(super) struct Grouping<'a> {
    /// The non-aggregate items, as written: the grouping keys
    pub keys: &'a [Expr],
    /// The input scope the aggregate arguments are resolved in
    pub scope: &'a [String],
    pub site: Site,
    /// The distinct aggregate calls found so far, as written, and as planned
    calls: Vec<Expr>,
    pub aggregates: Vec<AggregateCall>,
}

impl<'a> Grouping<'a> {
    pub fn new(keys: &'a [Expr], scope: &'a [String], site: Site) -> Self {
        Self {
            keys,
            scope,
            site,
            calls: Vec::new(),
            aggregates: Vec::new(),
        }
    }

    /// The hidden variables an item rewritten by `lift` may reference
    pub fn hidden_scope(&self) -> Vec<String> {
        (0..self.keys.len())
            .map(key_var)
            .chain((0..self.aggregates.len()).map(agg_var))
            .collect()
    }

    /// Rewrite `e`: grouping keys become `#key{i}`, aggregate calls become `#agg{j}`, and any
    /// other variable is refused (it would take an arbitrary row's value within the group)
    pub fn lift(&mut self, e: &Expr) -> CypherResult<Expr> {
        if let Some(i) = self.keys.iter().position(|k| k == e) {
            return Ok(Expr::Variable(key_var(i)));
        }
        if let (Some(kind), Expr::FunctionCall { args, distinct, .. }) = (aggregate_kind(e), e) {
            if let Some(j) = self.calls.iter().position(|c| c == e) {
                return Ok(Expr::Variable(agg_var(j)));
            }
            let arg = match (kind, args.as_slice()) {
                (AggregateKind::Count, [Expr::Star]) if !*distinct => None,
                (_, [a]) if !matches!(a, Expr::Star) => {
                    if contains_aggregate(a) {
                        return Err(CypherError::aggregation(format!(
                            "{e}: nested aggregate functions are not allowed"
                        )));
                    }
                    Some(resolve(a, self.scope, self.site)?)
                }
                _ => {
                    return Err(CypherError::InvalidArgumentCount {
                        function: kind.name().to_string(),
                        expected: "1".to_string(),
                        actual: args.len(),
                    })
                }
            };
            self.calls.push(e.clone());
            self.aggregates.push(AggregateCall {
                kind,
                distinct: *distinct,
                arg,
            });
            return Ok(Expr::Variable(agg_var(self.aggregates.len() - 1)));
        }
        if let Expr::Variable(v) = e {
            return Err(CypherError::aggregation(format!(
                "aggregation expression references '{v}', which is not a grouping key"
            )));
        }
        walk::map_children(e, &mut |c| self.lift(c))
    }
}

/// Replace every sub-expression of `e` that equals a projected expression by its column
pub(super) fn substitute(e: &Expr, projected: &[(Expr, String)]) -> Expr {
    if let Some((_, name)) = projected.iter().find(|(p, _)| p == e) {
        return Expr::Variable(name.clone());
    }
    walk::map_children::<()>(e, &mut |c| Ok(substitute(c, projected)))
        .unwrap_or_else(|()| e.clone())
}
