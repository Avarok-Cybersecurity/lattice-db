//! Expand execution: relationship traversal from bound nodes.

//! Query operation execution

use super::{ExecutionContext, InternalRows, QueryExecutor, QueryStats};
use crate::cypher::ast::Direction;
use crate::cypher::error::{CypherError, CypherResult};
use crate::cypher::planner::LogicalOp;
use crate::cypher::row::Row;
use crate::types::value::CypherValue;
use smallvec::SmallVec;

impl QueryExecutor {
    /// Execute Expand operation
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_expand(
        &self,
        input: &LogicalOp,
        from: &String,
        rel_variable: Option<&String>,
        to: &String,
        rel_types: &Vec<String>,
        _direction: Direction,
        min_hops: u32,
        max_hops: u32,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (input_rows, stats) = self.execute_op(input, ctx)?;

        // Register target variable binding (adds new column)
        // Note: rel_variable would also add a column if we were returning relationship refs
        let to_column_index = ctx.column_count();
        ctx.bind_variable(to, to_column_index);

        // Note: We don't add rel_variable binding yet since we don't return RelationshipRef
        // in the current implementation. When we do, add:
        // if let Some(rel_var) = rel_variable {
        //     ctx.bind_variable(rel_var, ctx.column_count());
        // }
        let _ = rel_variable; // Suppress unused warning

        // Get the source column index from bindings
        let from_column = ctx
            .get_variable_column(from)
            .ok_or_else(|| CypherError::Internal {
                message: format!("Source variable '{}' not bound in expand", from),
            })?;

        let mut result_rows = Vec::new();

        for mut row in input_rows {
            // Get the source node ID from the correct column
            if let Some(CypherValue::NodeRef(source_id)) = row.get(from_column) {
                // Use collection traverse for multi-hop
                let relations: Option<Vec<&str>> = if rel_types.is_empty() {
                    None
                } else {
                    Some(rel_types.iter().map(|s| s.as_str()).collect())
                };

                let traverse_result = ctx
                    .collection
                    .traverse(*source_id, max_hops as usize, relations.as_deref())
                    .map_err(|e| CypherError::Internal {
                        message: e.to_string(),
                    })?;

                // Filter paths first to enable last-path optimization
                let valid_paths: SmallVec<[_; 8]> = traverse_result
                    .paths
                    .into_iter()
                    .filter(|path| path.depth >= min_hops as usize)
                    .collect();

                // Limitation: direction filtering assumes outgoing edges only.
                // Incoming and bidirectional edge traversal not yet implemented.

                // Optimization: fast paths for common cases
                let paths_len = valid_paths.len();
                if paths_len == 0 {
                    continue; // No valid paths, skip this row
                } else if paths_len == 1 {
                    // Single path - move row directly (no clone needed)
                    // SAFETY: We just verified paths_len == 1, so next() yields Some
                    if let Some(path) = valid_paths.into_iter().next() {
                        Row::push(&mut row, CypherValue::NodeRef(path.target_id));
                        result_rows.push(row);
                    }
                } else {
                    // Multiple paths - clone all but last
                    for (i, path) in valid_paths.into_iter().enumerate() {
                        let mut new_row = if i == paths_len - 1 {
                            Row::take(&mut row) // Move last one (avoids clone)
                        } else {
                            row.clone()
                        };
                        Row::push(&mut new_row, CypherValue::NodeRef(path.target_id));
                        result_rows.push(new_row);
                    }
                }
            }
        }

        Ok((result_rows, stats))
    }
}
