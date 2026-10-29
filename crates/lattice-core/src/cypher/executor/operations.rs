//! Query operation execution

use super::canon::Canon;
use super::{ExecutionContext, InternalRows, QueryExecutor, QueryStats};
use crate::cypher::ast::{Expr, ProjectionItem};
use crate::cypher::error::{CypherError, CypherResult};
use crate::cypher::planner::LogicalOp;
use crate::cypher::row::{ExecutorRow, Row};
use crate::types::value::CypherValue;
use std::collections::HashSet;

/// Safely convert u64 to usize with bounds checking.
///
/// On 64-bit systems this is a no-op. On 32-bit systems, caps at usize::MAX
/// to prevent truncation attacks (e.g., `LIMIT 2^63` becoming `LIMIT 0`).
///
/// For user-facing values like LIMIT/SKIP, capping at max is the correct behavior
/// since it effectively means "unlimited" which is a safe interpretation.
#[inline]
pub(super) fn u64_to_usize_saturating(value: u64) -> usize {
    // On 64-bit: usize::MAX == u64::MAX, so this is essentially a no-op
    // On 32-bit: usize::MAX == u32::MAX, so large values are capped
    value.min(usize::MAX as u64) as usize
}

impl QueryExecutor {
    /// Execute a single logical operation
    pub(crate) fn execute_op(
        &self,
        op: &LogicalOp,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        match op {
            LogicalOp::AllNodesScan { variable } => self.execute_all_nodes_scan(variable, ctx),
            LogicalOp::NodeByLabelScan {
                variable,
                label,
                predicate,
            } => self.execute_label_scan(variable, label, predicate.as_deref(), ctx),
            LogicalOp::NodeByIdSeek { variable, ids } => self.execute_id_seek(variable, ids, ctx),
            LogicalOp::Expand {
                input,
                from,
                rel_variable,
                to,
                rel_types,
                direction,
                min_hops,
                max_hops,
                distinct_from,
            } => self.execute_expand(
                input,
                from,
                rel_variable.as_ref(),
                to,
                rel_types,
                *direction,
                *min_hops,
                *max_hops,
                distinct_from,
                ctx,
            ),
            LogicalOp::Cartesian { left, right } => self.execute_cartesian(left, right, ctx),
            LogicalOp::Argument => Ok((ctx.take_argument()?, QueryStats::default())),
            LogicalOp::OptionalMatch { input, inner } => {
                self.execute_optional_match(input, inner, ctx)
            }
            LogicalOp::Unwind {
                input,
                expr,
                variable,
            } => self.execute_unwind(input, expr, variable, ctx),
            LogicalOp::Filter { input, predicate } => self.execute_filter(input, predicate, ctx),
            LogicalOp::Project {
                input,
                items,
                carry,
            } => self.execute_project(input, items, *carry, ctx),
            LogicalOp::Aggregate {
                input,
                keys,
                aggregates,
                items,
            } => self.execute_aggregate(input, keys, aggregates, items, ctx),
            LogicalOp::Sort {
                input,
                items,
                limit,
            } => self.execute_sort(input, items, *limit, ctx),
            LogicalOp::Skip { input, count } => self.execute_skip(input, *count, ctx),
            LogicalOp::Limit { input, count } => self.execute_limit(input, *count, ctx),
            LogicalOp::Distinct { input } => self.execute_distinct(input, ctx),
            LogicalOp::CreateNode {
                labels,
                properties,
                variable,
            } => self.execute_create_node(labels, properties, variable.as_ref(), ctx),
            LogicalOp::CreateRelationship {
                from,
                to,
                rel_type,
                properties,
                variable,
            } => self.execute_create_relationship(
                from,
                to,
                rel_type,
                properties,
                variable.as_ref(),
                ctx,
            ),
            LogicalOp::DeleteNode {
                input,
                variable,
                detach,
            } => self.execute_delete_node(input, variable, *detach, ctx),
            LogicalOp::Empty => Ok((Vec::new(), QueryStats::default())),
            LogicalOp::SingleRow => {
                // Return a single row with no columns
                Ok((vec![ExecutorRow::new()], QueryStats::default()))
            }
            LogicalOp::DeleteRelationship { .. } => {
                Err(CypherError::unsupported("deleting a relationship"))
            }
            LogicalOp::NodeByPropertySeek { .. } => {
                Err(CypherError::unsupported("NodeByPropertySeek operator"))
            }
        }
    }

    /// Execute AllNodesScan
    pub(crate) fn execute_all_nodes_scan(
        &self,
        variable: &String,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        // Register variable binding for this scan
        let column_index = ctx.column_count();
        ctx.bind_variable(variable, column_index);

        let ids = ctx.collection.point_ids()?;
        // Preallocate outer Vec to avoid reallocation during collect
        let mut rows = Vec::with_capacity(ids.len());
        for id in ids {
            // Single-element row - SmallVec stores inline (zero heap allocation)
            rows.push(ExecutorRow::single(CypherValue::NodeRef(id)));
        }
        Ok((rows, QueryStats::default()))
    }

    /// Execute NodeByLabelScan
    pub(crate) fn execute_label_scan(
        &self,
        variable: &String,
        label: &String,
        predicate: Option<&Expr>,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        // Register variable binding for this scan
        let column_index = ctx.column_count();
        ctx.bind_variable(variable, column_index);

        // Use label index for O(1) lookup instead of scanning all points
        let ids = ctx.collection.point_ids_by_label(label)?;

        // Pre-allocate caches based on expected row count (Phase 2 optimization)
        ctx.prepare_cache_for_rows(ids.len());

        // If predicate pushdown is enabled, filter during scan. The node sits at its bound
        // column (earlier columns are absent here; a pushed predicate only reads this node).
        let rows = if let Some(pred) = predicate {
            let mut filtered = Vec::with_capacity(ids.len() / 2); // estimate 50% match rate
            let mut probe: Vec<CypherValue> = vec![CypherValue::Null; column_index + 1];
            for id in ids {
                probe[column_index] = CypherValue::NodeRef(id);
                if self.evaluate_predicate(pred, &probe, ctx)? {
                    filtered.push(ExecutorRow::single(CypherValue::NodeRef(id)));
                }
            }
            filtered
        } else {
            // No predicate - return all IDs with the label
            ids.into_iter()
                .map(|id| ExecutorRow::single(CypherValue::NodeRef(id)))
                .collect()
        };

        Ok((rows, QueryStats::default()))
    }

    /// Execute NodeByIdSeek
    pub(crate) fn execute_id_seek(
        &self,
        variable: &String,
        ids: &[u64],
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        // Register variable binding for this seek
        let column_index = ctx.column_count();
        ctx.bind_variable(variable, column_index);

        let points = ctx.collection.get_points(ids)?;
        let rows: InternalRows = ids
            .iter()
            .zip(points.iter())
            .filter_map(|(&id, opt)| {
                opt.as_ref()
                    .map(|_| ExecutorRow::single(CypherValue::NodeRef(id)))
            })
            .collect();

        Ok((rows, QueryStats::default()))
    }

    /// Execute Filter operation
    ///
    /// Note: Parallelization opportunity exists here (parallel::filter_into_vec)
    /// but requires ExecutionContext to be Sync. Currently blocked by
    /// `collection: &'a mut CollectionEngine` field. Future work could split
    /// ctx into mutable/immutable parts for parallel predicate evaluation.
    pub(crate) fn execute_filter(
        &self,
        input: &LogicalOp,
        predicate: &Expr,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (input_rows, stats) = self.execute_op(input, ctx)?;

        // Pre-allocate assuming ~50% selectivity (reduces reallocations)
        let mut result_rows = Vec::with_capacity(input_rows.len() / 2);

        for row in input_rows {
            if self.evaluate_predicate(predicate, &row, ctx)? {
                result_rows.push(row);
            }
        }

        Ok((result_rows, stats))
    }

    /// Execute Project operation; every item carries its output name as `alias`
    pub(crate) fn execute_project(
        &self,
        input: &LogicalOp,
        items: &[ProjectionItem],
        carry: bool,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (input_rows, stats) = self.execute_op(input, ctx)?;
        let names: Vec<String> = items
            .iter()
            .map(|i| i.alias.clone().unwrap_or_else(|| i.expr.to_string()))
            .collect();

        let mut result_rows = Vec::with_capacity(input_rows.len());
        for row in input_rows {
            let mut new_row =
                ExecutorRow::with_capacity(items.len() + if carry { row.len() } else { 0 });
            for item in items {
                Row::push(&mut new_row, self.evaluate_expr(&item.expr, &row, ctx)?);
            }
            if carry {
                Row::extend_from_iter(&mut new_row, row.into_iter());
            }
            result_rows.push(new_row);
        }

        let carried = if carry {
            ctx.bindings_in_order()
        } else {
            Vec::new()
        };
        ctx.rebind(&names);
        for (name, column) in carried {
            if !names.contains(&name) {
                ctx.bind_variable(&name, names.len() + column);
            }
        }
        Ok((result_rows, stats))
    }

    /// Execute Skip operation
    pub(crate) fn execute_skip(
        &self,
        input: &LogicalOp,
        count: u64,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (input_rows, stats) = self.execute_op(input, ctx)?;
        // Use saturating conversion to prevent truncation on 32-bit systems
        let count_usize = u64_to_usize_saturating(count);
        let result_rows: InternalRows = input_rows.into_iter().skip(count_usize).collect();
        Ok((result_rows, stats))
    }

    /// Execute Limit operation
    pub(crate) fn execute_limit(
        &self,
        input: &LogicalOp,
        count: u64,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (input_rows, stats) = self.execute_op(input, ctx)?;
        // Use saturating conversion to prevent truncation on 32-bit systems
        let count_usize = u64_to_usize_saturating(count);
        let result_rows: InternalRows = input_rows.into_iter().take(count_usize).collect();
        Ok((result_rows, stats))
    }

    /// Execute Distinct operation (openCypher equivalence: null = null, 1 = 1.0)
    pub(crate) fn execute_distinct(
        &self,
        input: &LogicalOp,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (input_rows, stats) = self.execute_op(input, ctx)?;
        let mut seen: HashSet<Vec<Canon>> = HashSet::with_capacity(input_rows.len());
        let rows = input_rows
            .into_iter()
            .filter(|row| seen.insert(Canon::row(row)))
            .collect();
        Ok((rows, stats))
    }
}
