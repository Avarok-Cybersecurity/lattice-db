//! Expand execution: relationship traversal from bound nodes.
//!
//! Single hop: every matching relationship gives a row (parallel relationships and self-loops
//! included), in any direction; the relationship is bound so `type(r)`, `r.weight` and
//! relationship uniqueness work. Variable length keeps LatticeDB's reachability semantics:
//! each node reachable in `min..=max` hops (by its shortest distance) once, start excluded
//! unless `min` is 0.

use super::{ExecutionContext, InternalRows, QueryExecutor, QueryStats};
use crate::cypher::ast::Direction;
use crate::cypher::error::{CypherError, CypherResult};
use crate::cypher::planner::LogicalOp;
use crate::cypher::row::{ExecutorRow, Row};
use crate::types::value::CypherValue;
use crate::types::SharedStr;
use rustc_hash::FxHashSet;
use std::collections::VecDeque;

/// One expansion step: (relationship start, end, type, weight), seen from the source node
type Step = (u64, u64, SharedStr, f32);

/// Where the bound variables of an expansion live in the input row
struct Columns {
    from: usize,
    /// `to` was bound before this expansion: keep rows whose target equals it
    to_bound: Option<usize>,
    /// `rel_variable` was bound before: keep rows whose relationship equals it
    rel_bound: Option<usize>,
    push_to: bool,
    push_rel: bool,
    distinct_from: Vec<usize>,
}

impl QueryExecutor {
    /// Execute Expand operation
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_expand(
        &self,
        input: &LogicalOp,
        from: &str,
        rel_variable: Option<&String>,
        to: &str,
        rel_types: &[String],
        direction: Direction,
        min_hops: u32,
        max_hops: u32,
        distinct_from: &[String],
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (input_rows, stats) = self.execute_op(input, ctx)?;

        let from_col = ctx.get_variable_column(from).ok_or_else(|| {
            CypherError::internal(format!("Source variable '{from}' not bound in expand"))
        })?;
        let to_bound = ctx.get_variable_column(to);
        let rel_bound = rel_variable.and_then(|r| ctx.get_variable_column(r));
        if to_bound.is_none() {
            ctx.bind_variable(to, ctx.column_count());
        }
        let push_rel = rel_variable.is_some() && rel_bound.is_none();
        if let (Some(r), true) = (rel_variable, push_rel) {
            ctx.bind_variable(r, ctx.column_count());
        }
        let cols = Columns {
            from: from_col,
            to_bound,
            rel_bound,
            push_to: to_bound.is_none(),
            push_rel,
            distinct_from: distinct_from
                .iter()
                .filter_map(|r| ctx.get_variable_column(r))
                .collect(),
        };
        let single_hop = rel_variable.is_some();

        let mut out = InternalRows::new();
        for row in input_rows {
            let source = match row.get(cols.from) {
                Some(CypherValue::NodeRef(id)) => *id,
                // A null from OPTIONAL MATCH expands to nothing.
                Some(CypherValue::Null) => continue,
                Some(other) => {
                    return Err(CypherError::invalid(
                        "relationship expansion from",
                        other.type_name(),
                    ))
                }
                None => return Err(CypherError::internal("expand source column missing")),
            };
            if single_hop {
                for (src, dst, rel_type, weight) in self.steps(source, rel_types, direction, ctx)? {
                    let target = if src == source { dst } else { src };
                    let handle =
                        CypherValue::RelationshipRef(ctx.rel_handle(src, dst, &rel_type, weight));
                    if cols
                        .distinct_from
                        .iter()
                        .any(|c| row.get(*c) == Some(&handle))
                    {
                        continue;
                    }
                    self.emit(&mut out, &row, &cols, target, Some(handle));
                }
            } else {
                for target in
                    self.reachable(source, rel_types, direction, min_hops, max_hops, ctx)?
                {
                    self.emit(&mut out, &row, &cols, target, None);
                }
            }
        }

        Ok((out, stats))
    }

    /// Append `row` extended by the target (and relationship) unless a bound column disagrees
    fn emit(
        &self,
        out: &mut InternalRows,
        row: &ExecutorRow,
        cols: &Columns,
        target: u64,
        rel: Option<CypherValue>,
    ) {
        let node = CypherValue::NodeRef(target);
        if cols.to_bound.is_some_and(|c| row.get(c) != Some(&node)) {
            return;
        }
        if let (Some(c), Some(r)) = (cols.rel_bound, &rel) {
            if row.get(c) != Some(r) {
                return;
            }
        }
        let mut new_row = row.clone();
        if cols.push_to {
            Row::push(&mut new_row, node);
        }
        if let (true, Some(r)) = (cols.push_rel, rel) {
            Row::push(&mut new_row, r);
        }
        out.push(new_row);
    }

    /// The relationships one hop from `node` in `direction` with one of `rel_types` (any when
    /// empty), as (start, end, type, weight). An undirected self-loop is reported once.
    fn steps(
        &self,
        node: u64,
        rel_types: &[String],
        direction: Direction,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<Vec<Step>> {
        let wanted = |t: &str| rel_types.is_empty() || rel_types.iter().any(|r| r == t);
        let mut steps = Vec::new();
        if matches!(direction, Direction::Outgoing | Direction::Both) {
            for e in ctx.collection.get_edges(node)? {
                if wanted(&e.relation) {
                    steps.push((
                        node,
                        e.target_id,
                        SharedStr::from(e.relation.as_str()),
                        e.weight,
                    ));
                }
            }
        }
        if matches!(direction, Direction::Incoming | Direction::Both) {
            for (src, rel_type, weight) in ctx.incoming(node)? {
                let self_loop_seen = direction == Direction::Both && src == node;
                if wanted(&rel_type) && !self_loop_seen {
                    steps.push((src, node, rel_type, weight));
                }
            }
        }
        Ok(steps)
    }

    /// Nodes reachable from `start` in `min..=max` hops, each once at its shortest distance
    fn reachable(
        &self,
        start: u64,
        rel_types: &[String],
        direction: Direction,
        min: u32,
        max: u32,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<Vec<u64>> {
        let mut found = Vec::new();
        if min == 0 {
            found.push(start);
        }
        if direction == Direction::Outgoing {
            // The engine's own BFS (same semantics, one storage lock).
            let relations: Option<Vec<&str>> =
                (!rel_types.is_empty()).then(|| rel_types.iter().map(String::as_str).collect());
            let result = ctx
                .collection
                .traverse(start, max as usize, relations.as_deref())?;
            found.extend(
                result
                    .paths
                    .into_iter()
                    .filter(|p| p.depth >= min.max(1) as usize)
                    .map(|p| p.target_id),
            );
            return Ok(found);
        }
        let mut visited: FxHashSet<u64> = FxHashSet::default();
        visited.insert(start);
        let mut queue = VecDeque::from([(start, 0u32)]);
        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max {
                continue;
            }
            for (src, dst, _, _) in self.steps(node, rel_types, direction, ctx)? {
                let next = if src == node { dst } else { src };
                if visited.insert(next) {
                    if depth + 1 >= min {
                        found.push(next);
                    }
                    queue.push_back((next, depth + 1));
                }
            }
        }
        Ok(found)
    }
}
