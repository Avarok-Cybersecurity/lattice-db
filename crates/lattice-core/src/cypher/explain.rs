//! Human-readable query plans (`DefaultCypherHandler::explain`).

use crate::cypher::planner::LogicalOp;

/// Format a logical plan as a string
pub(crate) fn format_plan(plan: &LogicalOp, indent: usize) -> String {
    let prefix = "  ".repeat(indent);
    match plan {
        LogicalOp::AllNodesScan { variable } => {
            format!("{}AllNodesScan({})\n", prefix, variable)
        }
        LogicalOp::NodeByLabelScan {
            variable,
            label,
            predicate,
        } => {
            let filter_str = if predicate.is_some() {
                " [+filter]"
            } else {
                ""
            };
            format!(
                "{}NodeByLabelScan({}, :{}){}\n",
                prefix, variable, label, filter_str
            )
        }
        LogicalOp::NodeByIdSeek { variable, ids } => {
            format!("{}NodeByIdSeek({}, {:?})\n", prefix, variable, ids)
        }
        LogicalOp::Expand {
            input,
            from,
            to,
            rel_types,
            direction,
            ..
        } => {
            let dir = match direction {
                crate::cypher::ast::Direction::Outgoing => "->",
                crate::cypher::ast::Direction::Incoming => "<-",
                crate::cypher::ast::Direction::Both => "--",
            };
            let types = if rel_types.is_empty() {
                "*".to_string()
            } else {
                rel_types
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join("|")
            };
            format!(
                "{}Expand(({}){}[{}]{}({}))\n{}",
                prefix,
                from,
                dir,
                types,
                dir,
                to,
                format_plan(input, indent + 1)
            )
        }
        LogicalOp::Filter { input, predicate } => {
            format!(
                "{}Filter({})\n{}",
                prefix,
                predicate,
                format_plan(input, indent + 1)
            )
        }
        LogicalOp::Project { input, items, .. } | LogicalOp::Aggregate { input, items, .. } => {
            let cols: Vec<_> = items
                .iter()
                .map(|i| match &i.alias {
                    Some(a) if *a != i.expr.to_string() => format!("{} AS {a}", i.expr),
                    _ => i.expr.to_string(),
                })
                .collect();
            let op = if matches!(plan, LogicalOp::Aggregate { .. }) {
                "Aggregate"
            } else {
                "Project"
            };
            format!(
                "{}{op}([{}])\n{}",
                prefix,
                cols.join(", "),
                format_plan(input, indent + 1)
            )
        }
        LogicalOp::Sort { input, items, .. } => {
            let cols: Vec<_> = items
                .iter()
                .map(|i| format!("{} {}", i.expr, if i.ascending { "ASC" } else { "DESC" }))
                .collect();
            format!(
                "{}Sort([{}])\n{}",
                prefix,
                cols.join(", "),
                format_plan(input, indent + 1)
            )
        }
        LogicalOp::Skip { input, count } => {
            format!(
                "{}Skip({})\n{}",
                prefix,
                count,
                format_plan(input, indent + 1)
            )
        }
        LogicalOp::Limit { input, count } => {
            format!(
                "{}Limit({})\n{}",
                prefix,
                count,
                format_plan(input, indent + 1)
            )
        }
        LogicalOp::Distinct { input } => {
            format!("{}Distinct\n{}", prefix, format_plan(input, indent + 1))
        }
        LogicalOp::CreateNode {
            labels, variable, ..
        } => {
            let var = variable.as_ref().map(|v| v.to_string()).unwrap_or_default();
            let lbls: Vec<_> = labels.iter().map(|l| format!(":{}", l)).collect();
            format!("{}CreateNode({}{})\n", prefix, var, lbls.join(""))
        }
        LogicalOp::CreateRelationship {
            from, to, rel_type, ..
        } => {
            format!(
                "{}CreateRelationship(({})-[:{}]->({}))\n",
                prefix, from, rel_type, to
            )
        }
        LogicalOp::DeleteNode {
            input,
            variable,
            detach,
        } => {
            format!(
                "{}{}Delete({})\n{}",
                prefix,
                if *detach { "Detach" } else { "" },
                variable,
                format_plan(input, indent + 1)
            )
        }
        LogicalOp::Cartesian { left, right } => format!(
            "{}Cartesian\n{}{}",
            prefix,
            format_plan(left, indent + 1),
            format_plan(right, indent + 1)
        ),
        LogicalOp::Argument => format!("{}Argument\n", prefix),
        LogicalOp::OptionalMatch { input, inner } => format!(
            "{}OptionalMatch\n{}{}",
            prefix,
            format_plan(input, indent + 1),
            format_plan(inner, indent + 1)
        ),
        LogicalOp::Unwind {
            input,
            expr,
            variable,
        } => format!(
            "{}Unwind({} AS {})\n{}",
            prefix,
            expr,
            variable,
            format_plan(input, indent + 1)
        ),
        LogicalOp::Empty => format!("{}Empty\n", prefix),
        LogicalOp::SingleRow => format!("{}SingleRow\n", prefix),
        _ => format!("{}Unknown\n", prefix),
    }
}
