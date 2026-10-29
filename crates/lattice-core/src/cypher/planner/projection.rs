//! Projection planning for RETURN and WITH, in openCypher order:
//! project (or aggregate) → DISTINCT → ORDER BY → SKIP → LIMIT.
//!
//! ORDER BY may use the projected names, any projected expression, and — unless the projection
//! aggregates or is DISTINCT — the variables in scope before it. The latter are carried through
//! the projection as extra columns and trimmed after LIMIT.

use super::resolve::{contains_aggregate, resolve, substitute, Grouping, Site};
use super::{LogicalOp, QueryPlanner};
use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};

/// Which clause a projection body belongs to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProjectionKind {
    Return,
    With,
}

impl ProjectionKind {
    fn site(self) -> Site {
        match self {
            ProjectionKind::Return => Site::Return,
            ProjectionKind::With => Site::With,
        }
    }
}

/// Named variables in scope (hidden planner names start with `#`)
fn named(scope: &[String]) -> impl Iterator<Item = &String> {
    scope.iter().filter(|v| !v.starts_with('#'))
}

/// Expand `*` and give every item its output name
fn name_items(
    items: &[ProjectionItem],
    scope: &[String],
    kind: ProjectionKind,
) -> CypherResult<Vec<(Expr, String)>> {
    let mut out: Vec<(Expr, String)> = Vec::new();
    for item in items {
        if matches!(item.expr, Expr::Star) {
            let before = out.len();
            out.extend(named(scope).map(|v| (Expr::variable(v.as_str()), v.clone())));
            if out.len() == before {
                return Err(CypherError::syntax(
                    0,
                    0,
                    "RETURN * / WITH * requires at least one variable in scope",
                ));
            }
            continue;
        }
        let name = match (&item.alias, &item.expr, kind) {
            (Some(alias), _, _) => alias.clone(),
            (None, Expr::Variable(v), _) => v.clone(),
            (None, e, ProjectionKind::Return) => e.to_string(),
            (None, e, ProjectionKind::With) => {
                return Err(CypherError::syntax(
                    0,
                    0,
                    format!("Expression in WITH must be aliased (use AS): {e}"),
                ))
            }
        };
        out.push((item.expr.clone(), name));
    }
    for (i, (_, name)) in out.iter().enumerate() {
        if out[..i].iter().any(|(_, n)| n == name) {
            return Err(CypherError::syntax(
                0,
                0,
                format!("Multiple result columns with the same name '{name}' are not supported"),
            ));
        }
    }
    Ok(out)
}

impl QueryPlanner {
    /// Plan a RETURN or WITH projection body over `input`; `scope` becomes the projected names
    pub(super) fn plan_projection(
        &self,
        input: LogicalOp,
        scope: &mut Vec<String>,
        body: &ReturnClause,
        kind: ProjectionKind,
    ) -> CypherResult<LogicalOp> {
        let site = kind.site();
        let items = name_items(&body.items, scope, kind)?;
        let names: Vec<String> = items.iter().map(|(_, n)| n.clone()).collect();
        let aggregating = items.iter().any(|(e, _)| contains_aggregate(e));
        let carry = !aggregating && !body.distinct && body.order_by.is_some();

        let mut plan = if aggregating {
            let keys: Vec<Expr> = items
                .iter()
                .filter(|(e, _)| !contains_aggregate(e))
                .map(|(e, _)| e.clone())
                .collect();
            let mut grouping = Grouping::new(&keys, scope, site);
            let lifted = items
                .iter()
                .map(|(e, _)| grouping.lift(e))
                .collect::<CypherResult<Vec<_>>>()?;
            let hidden = grouping.hidden_scope();
            let aggregates = std::mem::take(&mut grouping.aggregates);
            let out_items = lifted
                .iter()
                .zip(&names)
                .map(|(e, n)| {
                    Ok(ProjectionItem::new(resolve(e, &hidden, site)?).with_alias(n.as_str()))
                })
                .collect::<CypherResult<_>>()?;
            LogicalOp::Aggregate {
                input: Box::new(input),
                keys: keys
                    .iter()
                    .map(|k| resolve(k, scope, site))
                    .collect::<CypherResult<_>>()?,
                aggregates,
                items: out_items,
            }
        } else {
            LogicalOp::Project {
                input: Box::new(input),
                items: items
                    .iter()
                    .map(|(e, n)| {
                        Ok(ProjectionItem::new(resolve(e, scope, site)?).with_alias(n.as_str()))
                    })
                    .collect::<CypherResult<_>>()?,
                carry,
            }
        };

        if body.distinct {
            plan = LogicalOp::Distinct {
                input: Box::new(plan),
            };
        }

        if let Some(order_by) = &body.order_by {
            let mut order_scope = names.clone();
            if carry {
                order_scope.extend(scope.iter().filter(|v| !names.contains(v)).cloned());
            }
            let sort_items = order_by
                .items
                .iter()
                .map(|item| {
                    let e = substitute(&item.expr, &items);
                    if aggregating && contains_aggregate(&e) {
                        return Err(CypherError::aggregation(format!(
                            "ORDER BY {e}: only aggregates that are projected can be ordered by"
                        )));
                    }
                    Ok(OrderByItem {
                        expr: resolve(&e, &order_scope, Site::OrderBy)?,
                        ascending: item.ascending,
                    })
                })
                .collect::<CypherResult<_>>()?;
            plan = LogicalOp::Sort {
                input: Box::new(plan),
                items: sort_items,
                // Rows past SKIP + LIMIT are never returned, so a partial sort suffices.
                limit: body.limit.map(|l| l.saturating_add(body.skip.unwrap_or(0))),
            };
        }

        if let Some(count) = body.skip {
            plan = LogicalOp::Skip {
                input: Box::new(plan),
                count,
            };
        }
        if let Some(count) = body.limit {
            plan = LogicalOp::Limit {
                input: Box::new(plan),
                count,
            };
        }

        if carry {
            plan = LogicalOp::Project {
                input: Box::new(plan),
                items: names
                    .iter()
                    .map(|n| ProjectionItem::new(Expr::variable(n.as_str())).with_alias(n.as_str()))
                    .collect(),
                carry: false,
            };
        }

        *scope = names;
        Ok(plan)
    }
}
