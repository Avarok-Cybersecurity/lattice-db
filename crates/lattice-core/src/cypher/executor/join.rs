//! Row-combining operators: cartesian product, OPTIONAL MATCH (left outer join), UNWIND.

use super::{ExecutionContext, InternalRows, QueryExecutor, QueryStats};
use crate::cypher::ast::Expr;
use crate::cypher::error::{CypherError, CypherResult};
use crate::cypher::planner::LogicalOp;
use crate::cypher::row::{ExecutorRow, Row};
use crate::types::value::CypherValue;

/// Hidden column an OPTIONAL MATCH tags its argument rows with
const OPTIONAL_ROW: &str = "#optional_row";

fn merge(stats: &mut QueryStats, other: QueryStats) {
    stats.nodes_created += other.nodes_created;
    stats.relationships_created += other.relationships_created;
    stats.nodes_deleted += other.nodes_deleted;
    stats.relationships_deleted += other.relationships_deleted;
    stats.properties_set += other.properties_set;
    stats.labels_added += other.labels_added;
}

impl QueryExecutor {
    /// Every left row followed by every right row (right runs once; it does not read left)
    pub(crate) fn execute_cartesian(
        &self,
        left: &LogicalOp,
        right: &LogicalOp,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (left_rows, mut stats) = self.execute_op(left, ctx)?;
        let (right_rows, right_stats) = self.execute_op(right, ctx)?;
        merge(&mut stats, right_stats);
        let mut out = Vec::with_capacity(left_rows.len().saturating_mul(right_rows.len()));
        for l in &left_rows {
            for r in &right_rows {
                let mut row = l.clone();
                Row::extend_from_iter(&mut row, r.iter().cloned());
                out.push(row);
            }
        }
        Ok((out, stats))
    }

    /// OPTIONAL MATCH: run `inner` once over all input rows, each tagged with its position;
    /// input rows the inner plan dropped come back with nulls for the variables it binds
    pub(crate) fn execute_optional_match(
        &self,
        input: &LogicalOp,
        inner: &LogicalOp,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (rows, mut stats) = self.execute_op(input, ctx)?;
        let tag = ctx.column_count();
        ctx.bind_variable(OPTIONAL_ROW, tag);
        let tagged: InternalRows = rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut r = row.clone();
                Row::push(&mut r, CypherValue::Int(i as i64));
                r
            })
            .collect();
        ctx.set_argument(tagged);
        let (matched, inner_stats) = self.execute_op(inner, ctx)?;
        merge(&mut stats, inner_stats);
        let added = ctx.column_count() - tag - 1;

        let mut per_row: Vec<Vec<ExecutorRow>> = vec![Vec::new(); rows.len()];
        for m in matched {
            let i = match m.get(tag) {
                Some(CypherValue::Int(i)) => usize::try_from(*i).ok(),
                _ => None,
            }
            .ok_or_else(|| CypherError::internal("OPTIONAL MATCH row lost its tag"))?;
            let mut values = m.to_vec();
            values.remove(tag);
            per_row[i].push(ExecutorRow::from_vec(values));
        }

        let mut out = InternalRows::new();
        for (row, found) in rows.into_iter().zip(per_row) {
            if found.is_empty() {
                let mut r = row;
                Row::extend_from_iter(&mut r, std::iter::repeat(CypherValue::Null).take(added));
                out.push(r);
            } else {
                out.extend(found);
            }
        }
        ctx.unbind_column(tag);
        Ok((out, stats))
    }

    /// UNWIND: one row per list element; null gives no row, a non-list value one row
    pub(crate) fn execute_unwind(
        &self,
        input: &LogicalOp,
        expr: &Expr,
        variable: &str,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (rows, stats) = self.execute_op(input, ctx)?;
        let column = ctx.column_count();
        let mut out = InternalRows::new();
        for row in &rows {
            let items = match self.evaluate_expr(expr, row, ctx)? {
                CypherValue::Null => continue,
                CypherValue::List(items) => items,
                other => vec![other],
            };
            for item in items {
                let mut r = row.clone();
                Row::push(&mut r, item);
                out.push(r);
            }
        }
        ctx.bind_variable(variable, column);
        Ok((out, stats))
    }
}
