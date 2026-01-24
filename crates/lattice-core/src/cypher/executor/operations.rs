//! Query operation execution

use super::{ExecutionContext, InternalRows, QueryExecutor, QueryStats, SortKey};
use crate::cypher::ast::{Direction, Expr, MapLiteral, OrderByItem, ProjectionItem};
use crate::cypher::error::{CypherError, CypherResult};
use crate::cypher::planner::LogicalOp;
use crate::cypher::row::{ExecutorRow, Row};
use crate::parallel;
use crate::types::point::{Point, PointId};
use crate::types::value::CypherValue;
use smallvec::SmallVec;
use std::collections::HashMap;

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
            } => self.execute_expand(
                input,
                from,
                rel_variable.as_ref(),
                to,
                rel_types,
                *direction,
                *min_hops,
                *max_hops,
                ctx,
            ),
            LogicalOp::Filter { input, predicate } => self.execute_filter(input, predicate, ctx),
            LogicalOp::Project { input, items } => self.execute_project(input, items, ctx),
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
            _ => Err(CypherError::unsupported("Unsupported operation")),
        }
    }

    /// Execute AllNodesScan
    pub(crate) fn execute_all_nodes_scan(
        &self,
        _variable: &String,
        ctx: &ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
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
        _variable: &String,
        label: &String,
        predicate: Option<&Expr>,
        ctx: &ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        // Use label index for O(1) lookup instead of scanning all points
        let ids = ctx.collection.point_ids_by_label(label)?;

        // Pre-allocate caches based on expected row count (Phase 2 optimization)
        ctx.prepare_cache_for_rows(ids.len());

        // If predicate pushdown is enabled, filter during scan
        let rows = if let Some(pred) = predicate {
            // With predicate pushdown, evaluate predicate for each node
            let mut filtered = Vec::with_capacity(ids.len() / 2); // estimate 50% match rate

            for id in ids {
                let row = ExecutorRow::single(CypherValue::NodeRef(id));
                match self.evaluate_predicate(pred, &row, ctx) {
                    Ok(true) => filtered.push(row),
                    Ok(false) => continue,
                    Err(e) => {
                        return Err(CypherError::Internal {
                            message: format!("Predicate evaluation failed for node {}: {}", id, e),
                        })
                    }
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
        _variable: &String,
        ids: &[u64],
        ctx: &ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
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

    /// Execute Expand operation
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_expand(
        &self,
        input: &LogicalOp,
        _from: &String,
        _rel_variable: Option<&String>,
        _to: &String,
        rel_types: &Vec<String>,
        _direction: Direction,
        min_hops: u32,
        max_hops: u32,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (input_rows, stats) = self.execute_op(input, ctx)?;

        let mut result_rows = Vec::new();

        for mut row in input_rows {
            // Get the source node ID from the row
            // Assume the first column is the source node
            if let Some(CypherValue::NodeRef(source_id)) = row.first() {
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

                // Check direction (simplified - assume outgoing matches)
                // TODO: Proper direction handling

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

    /// Execute Project operation
    pub(crate) fn execute_project(
        &self,
        input: &LogicalOp,
        items: &[ProjectionItem],
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (input_rows, stats) = self.execute_op(input, ctx)?;

        let mut result_rows = Vec::new();

        for row in input_rows {
            let mut new_row: ExecutorRow = ExecutorRow::with_capacity(items.len());

            for item in items {
                if matches!(item.expr, Expr::Star) {
                    // Return all columns
                    Row::extend_from_iter(&mut new_row, row.iter().cloned());
                } else {
                    // Evaluate expression
                    let value = self.evaluate_expr(&item.expr, &row, ctx)?;
                    Row::push(&mut new_row, value);
                }
            }

            result_rows.push(new_row);
        }

        Ok((result_rows, stats))
    }

    /// Execute Sort operation using index-based Schwartzian transform
    /// Stores (SortKey, index) pairs instead of (keys, full_row) to eliminate row cloning
    /// With partial sort optimization when limit is specified
    pub(crate) fn execute_sort(
        &self,
        input: &LogicalOp,
        items: &[OrderByItem],
        limit: Option<u64>,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (mut input_rows, stats) = self.execute_op(input, ctx)?;

        if items.is_empty() {
            return Ok((input_rows, stats));
        }

        // FAST PATH: Single i64 sort key - avoids all CypherValue overhead
        // This is ~3x faster for ORDER BY integer_column queries
        if let Some((i64_keys, ascending)) = self.try_batch_i64_sort_keys(&input_rows, items, ctx) {
            return self.execute_sort_i64_fast(input_rows, i64_keys, ascending, limit, stats);
        }

        // OPTIMIZATION: Try batch prefetch for simple property access sort expressions
        // This enables parallel sort key construction by pre-fetching all needed data
        let prefetched = self.try_batch_prefetch_sort_keys(&input_rows, items, ctx);

        // Index-based Schwartzian transform: store (keys, index) not (keys, row)
        // SmallVec<[CypherValue; 3]> provides inline storage for 1-3 keys (common case)
        let mut keyed_indices: Vec<(SortKey, usize)> = if let Some(ref prefetch_data) = prefetched {
            // Parallel construction using prefetched data (no ctx needed)
            parallel::enumerate_map_collect(&input_rows, |idx, _row| {
                let keys: SortKey = prefetch_data[idx].iter().cloned().collect();
                (keys, idx)
            })
        } else {
            // Fallback: sequential extraction (requires ctx)
            input_rows
                .iter()
                .enumerate()
                .map(|(idx, row)| {
                    let keys: SortKey = items
                        .iter()
                        .map(|item| {
                            self.evaluate_expr(&item.expr, row, ctx)
                                .unwrap_or(CypherValue::Null)
                        })
                        .collect();
                    (keys, idx)
                })
                .collect()
        };

        // Check for homogeneous i64 keys - enables SIMD-accelerated comparison
        let use_simd = keyed_indices.len() >= 64 && self.keys_are_homogeneous_i64(&keyed_indices);

        // Comparator for sort keys
        let compare = |keys_a: &SortKey, keys_b: &SortKey| -> std::cmp::Ordering {
            for (i, item) in items.iter().enumerate() {
                let key_a = keys_a.get(i).unwrap_or(&CypherValue::Null);
                let key_b = keys_b.get(i).unwrap_or(&CypherValue::Null);
                let cmp = self.compare_values(key_a, key_b);
                let cmp = if item.ascending { cmp } else { cmp.reverse() };
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        };

        // SIMD-accelerated sort for homogeneous i64 keys (ORDER BY age, id, etc.)
        if use_simd && items.len() == 1 {
            self.sort_i64_simd(&mut keyed_indices, items[0].ascending);
        } else if let Some(k) = limit {
            // Optimization: Use partial sort when LIMIT is specified
            let k = k as usize;
            if k > 0 && k < keyed_indices.len() {
                // Partial sort: O(n) partition + O(k log k) sort of top k elements
                keyed_indices.select_nth_unstable_by(k - 1, |(keys_a, _), (keys_b, _)| {
                    compare(keys_a, keys_b)
                });
                keyed_indices.truncate(k);
                // Sort just the first k elements
                // Use expensive_cmp variant: CypherValue comparison has enum overhead
                parallel::sort_by_expensive_cmp(&mut keyed_indices, |(keys_a, _), (keys_b, _)| {
                    compare(keys_a, keys_b)
                });
            } else if k == 0 {
                return Ok((Vec::new(), stats));
            } else {
                // k >= keyed_indices.len(), full sort needed
                parallel::sort_by_expensive_cmp(&mut keyed_indices, |(keys_a, _), (keys_b, _)| {
                    compare(keys_a, keys_b)
                });
            }
        } else {
            // No limit: full sort O(n log n)
            // Use expensive_cmp: skips parallel in 30K-80K range where overhead > benefit
            parallel::sort_by_expensive_cmp(&mut keyed_indices, |(keys_a, _), (keys_b, _)| {
                compare(keys_a, keys_b)
            });
        }

        // Reconstruct rows in sorted order using indices
        // Use Row::take to move rows without cloning (zero-copy)
        let sorted_rows: InternalRows = keyed_indices
            .into_iter()
            .map(|(_, idx)| Row::take(&mut input_rows[idx]))
            .collect();

        Ok((sorted_rows, stats))
    }

    /// Fast path for sorting by a single i64 key.
    /// Avoids all CypherValue overhead by working directly with (i64, usize) pairs.
    ///
    /// With `simd` feature: Uses O(n) radix sort for large arrays.
    /// Without: Uses O(n log n) comparison-based parallel sort.
    pub(crate) fn execute_sort_i64_fast(
        &self,
        mut input_rows: InternalRows,
        keys: Vec<i64>,
        ascending: bool,
        limit: Option<u64>,
        stats: QueryStats,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        // Create (key, index) pairs for sorting
        let mut keyed_indices: Vec<(i64, usize)> = keys
            .into_iter()
            .enumerate()
            .map(|(idx, key)| (key, idx))
            .collect();

        #[cfg(feature = "simd")]
        {
            // Use O(n) radix sort for large arrays
            use crate::cypher::row::{radix_partial_sort_i64_indexed, radix_sort_i64_indexed};

            if let Some(k) = limit {
                let k = k as usize;
                if k == 0 {
                    return Ok((Vec::new(), stats));
                }
                // Radix partial sort: efficient for LIMIT queries
                radix_partial_sort_i64_indexed(&mut keyed_indices, k, ascending);
                keyed_indices.truncate(k);
            } else {
                // Full radix sort: O(n) for all elements
                radix_sort_i64_indexed(&mut keyed_indices, ascending);
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            // Comparator based on sort direction
            let compare = if ascending {
                |a: &(i64, usize), b: &(i64, usize)| a.0.cmp(&b.0)
            } else {
                |a: &(i64, usize), b: &(i64, usize)| b.0.cmp(&a.0)
            };

            if let Some(k) = limit {
                let k = k as usize;
                if k > 0 && k < keyed_indices.len() {
                    // Partial sort: O(n) partition + O(k log k) sort
                    keyed_indices.select_nth_unstable_by(k - 1, compare);
                    keyed_indices.truncate(k);
                    parallel::sort_by(&mut keyed_indices, compare);
                } else if k == 0 {
                    return Ok((Vec::new(), stats));
                } else {
                    parallel::sort_by(&mut keyed_indices, compare);
                }
            } else {
                parallel::sort_by(&mut keyed_indices, compare);
            }
        }

        // Reconstruct rows in sorted order using Row::take (zero-copy)
        let sorted_rows: InternalRows = keyed_indices
            .into_iter()
            .map(|(_, idx)| Row::take(&mut input_rows[idx]))
            .collect();

        Ok((sorted_rows, stats))
    }

    /// Check if all sort keys are homogeneous i64 (enables SIMD optimization)
    #[inline]
    pub(crate) fn keys_are_homogeneous_i64(&self, rows: &[(SortKey, usize)]) -> bool {
        rows.iter().all(|(keys, _)| {
            keys.iter()
                .all(|k| matches!(k, CypherValue::Int(_) | CypherValue::Null))
        })
    }

    /// Extract all NodeRef IDs from the first column of rows.
    /// Returns None if any row doesn't have a NodeRef in first position.
    /// Preallocates for efficiency.
    #[inline]
    pub(crate) fn extract_node_ids(input_rows: &InternalRows) -> Option<Vec<u64>> {
        let mut node_ids = Vec::with_capacity(input_rows.len());
        for row in input_rows {
            if let Some(CypherValue::NodeRef(id)) = Row::first(row) {
                node_ids.push(*id);
            } else {
                return None;
            }
        }
        Some(node_ids)
    }

    /// Try to extract sort keys using the highly optimized i64 path.
    /// Returns Some((keys, ascending)) if this is a single integer property ORDER BY.
    pub(crate) fn try_batch_i64_sort_keys(
        &self,
        input_rows: &InternalRows,
        items: &[OrderByItem],
        ctx: &ExecutionContext,
    ) -> Option<(Vec<i64>, bool)> {
        // Only optimize single-key integer sorts
        if items.len() != 1 {
            return None;
        }

        // Lower threshold: i64 fast-path is always better than CypherValue comparison
        // Even at 50 rows, avoiding enum overhead is worth it
        const I64_THRESHOLD: usize = 50;
        if input_rows.len() < I64_THRESHOLD {
            return None;
        }

        // Check if sort expression is simple property access
        let property = self.extract_simple_property_access(&items[0].expr)?;

        // Extract all NodeRef IDs from first column (preallocated)
        let node_ids = Self::extract_node_ids(input_rows)?;

        // Use the highly optimized i64 extraction (no cloning, no CypherValue overhead)
        let keys = ctx
            .collection
            .batch_extract_i64_property(&node_ids, property)
            .ok()?;
        Some((keys, items[0].ascending))
    }

    pub(crate) fn try_batch_prefetch_sort_keys(
        &self,
        input_rows: &InternalRows,
        items: &[OrderByItem],
        ctx: &ExecutionContext,
    ) -> Option<Vec<Vec<CypherValue>>> {
        // Optimize for moderately large row counts where batch extraction benefits
        // outweigh overhead. Lower threshold to enable optimization earlier.
        const BATCH_PREFETCH_THRESHOLD: usize = 500;
        if input_rows.len() < BATCH_PREFETCH_THRESHOLD {
            return None;
        }

        // Check if all sort expressions are simple property access on first column (variable)
        // Pattern: ORDER BY p.property where p is the first column (NodeRef)
        let property_names: Vec<&str> = items
            .iter()
            .filter_map(|item| self.extract_simple_property_access(&item.expr))
            .collect();

        // If not all expressions match, fall back to sequential
        if property_names.len() != items.len() {
            return None;
        }

        // Extract all NodeRef IDs from first column (preallocated)
        let node_ids = Self::extract_node_ids(input_rows)?;

        // Batch extract raw property bytes (single lock acquisition, no Point clone)
        let raw_properties = ctx
            .collection
            .batch_extract_properties(&node_ids, &property_names)
            .ok()?;

        // Parallel deserialize JSON bytes to CypherValues
        let prefetched: Vec<Vec<CypherValue>> = parallel::enumerate_map_collect(
            &raw_properties,
            |_idx, row_properties: &Vec<Option<Vec<u8>>>| {
                row_properties
                    .iter()
                    .map(|opt_bytes: &Option<Vec<u8>>| {
                        opt_bytes
                            .as_ref()
                            .and_then(|bytes| self.json_bytes_to_cypher_value(bytes).ok())
                            .unwrap_or(CypherValue::Null)
                    })
                    .collect()
            },
        );

        Some(prefetched)
    }

    /// Extract the property name if this expression is a simple property access on first column.
    /// Pattern: Expr::Property { expr: Expr::Variable(_), property }
    pub(crate) fn extract_simple_property_access<'a>(&self, expr: &'a Expr) -> Option<&'a str> {
        if let Expr::Property {
            expr: inner,
            property,
        } = expr
        {
            if matches!(inner.as_ref(), Expr::Variable(_)) {
                return Some(property.as_str());
            }
        }
        None
    }

    /// Optimized sort for single i64 key using unstable sort.
    /// The i64 comparison itself is cheap; this path avoids the generic CypherValue
    /// comparison path which has more overhead per comparison.
    #[inline]
    pub(crate) fn sort_i64_simd(&self, keyed_indices: &mut [(SortKey, usize)], ascending: bool) {
        // Direct comparison with inline .as_int() extraction
        // Cheaper than generic CypherValue comparison path
        if ascending {
            keyed_indices.sort_unstable_by(|(a, _), (b, _)| {
                let val_a = a.first().and_then(|v| v.as_int()).unwrap_or(i64::MAX);
                let val_b = b.first().and_then(|v| v.as_int()).unwrap_or(i64::MAX);
                val_a.cmp(&val_b)
            });
        } else {
            keyed_indices.sort_unstable_by(|(a, _), (b, _)| {
                let val_a = a.first().and_then(|v| v.as_int()).unwrap_or(i64::MIN);
                let val_b = b.first().and_then(|v| v.as_int()).unwrap_or(i64::MIN);
                val_b.cmp(&val_a)
            });
        }
    }

    /// Execute Skip operation
    pub(crate) fn execute_skip(
        &self,
        input: &LogicalOp,
        count: u64,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (input_rows, stats) = self.execute_op(input, ctx)?;
        let result_rows: InternalRows = input_rows.into_iter().skip(count as usize).collect();
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
        let result_rows: InternalRows = input_rows.into_iter().take(count as usize).collect();
        Ok((result_rows, stats))
    }

    /// Execute Distinct operation
    ///
    /// Uses index collection to avoid cloning rows during deduplication.
    /// Only the final unique rows are moved into the result.
    pub(crate) fn execute_distinct(
        &self,
        input: &LogicalOp,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (mut input_rows, stats) = self.execute_op(input, ctx)?;

        // Use HashMap with collision detection to avoid hash collision bugs
        // Key: hash, Value: indices of rows with that hash
        let mut seen: HashMap<u64, SmallVec<[usize; 1]>> = HashMap::with_capacity(input_rows.len());
        let mut unique_indices: Vec<usize> = Vec::with_capacity(input_rows.len());

        for (idx, row) in input_rows.iter().enumerate() {
            let hash = self.hash_row(row);
            let entry = seen.entry(hash).or_default();

            // Check for collision: verify row is actually different from all rows with same hash
            let is_duplicate = entry
                .iter()
                .any(|&prev_idx| self.rows_equal(&input_rows[prev_idx], row));

            if !is_duplicate {
                entry.push(idx);
                unique_indices.push(idx);
            }
        }

        // Extract unique rows by moving (not cloning) from input_rows
        // Process in reverse order to avoid index invalidation
        unique_indices.sort_unstable();
        let result_rows: Vec<_> = unique_indices
            .into_iter()
            .rev()
            .map(|idx| input_rows.swap_remove(idx))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        Ok((result_rows, stats))
    }

    /// Execute CreateNode operation
    pub(crate) fn execute_create_node(
        &self,
        labels: &Vec<String>,
        properties: &MapLiteral,
        _variable: Option<&String>,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        // Generate a new ID (hash of timestamp + random)
        let id = self.generate_point_id();

        // Build payload with labels and properties
        // Pre-allocate: 1 for _labels + number of properties
        let mut payload = HashMap::with_capacity(1 + properties.entries.len());

        // Compute label bitmap for O(1) label checks
        let label_bitmap = if !labels.is_empty() {
            // Store labels as JSON array (for backwards compatibility)
            let labels_json: Vec<&str> = labels.iter().map(|l| l.as_str()).collect();
            let labels_bytes =
                serde_json::to_vec(&labels_json).map_err(|e| CypherError::Internal {
                    message: e.to_string(),
                })?;
            payload.insert("_labels".to_string(), labels_bytes);

            // Compute bitmap for fast label checks
            ctx.labels_to_bitmap(labels)
        } else {
            0
        };

        // Store properties - use an empty slice as we're creating a new node
        let empty_row: ExecutorRow = ExecutorRow::new();
        for (key, value_expr) in &properties.entries {
            // Evaluate the expression to get the value
            let value = self.evaluate_expr(value_expr, &empty_row, ctx)?;
            let value_bytes = self.cypher_value_to_json_bytes(&value)?;
            payload.insert(key.to_string(), value_bytes);
        }

        // Create zero vector for graph-only node
        let dim = ctx.collection.vector_dim();
        let vector = vec![0.0f32; dim];

        let point = Point::new_vector(id, vector)
            .with_payload(payload)
            .with_label_bitmap(label_bitmap);

        ctx.collection
            .upsert_points(vec![point])
            .map_err(|e| CypherError::Internal {
                message: e.to_string(),
            })?;

        let mut stats = QueryStats::default();
        stats.nodes_created = 1;

        // Return the created node reference
        let row = ExecutorRow::single(CypherValue::NodeRef(id));

        Ok((vec![row], stats))
    }

    /// Execute CreateRelationship operation
    pub(crate) fn execute_create_relationship(
        &self,
        from: &String,
        to: &String,
        rel_type: &String,
        properties: &MapLiteral,
        _variable: Option<&String>,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        // For now, we need the from/to IDs to be provided via parameters
        // This is a simplified implementation
        let from_id = ctx
            .parameters
            .get(from.as_str())
            .and_then(|v| match v {
                CypherValue::NodeRef(id) => Some(*id),
                CypherValue::Int(id) => Some(*id as u64),
                _ => None,
            })
            .ok_or_else(|| CypherError::unknown_variable(from.as_str()))?;

        let to_id = ctx
            .parameters
            .get(to.as_str())
            .and_then(|v| match v {
                CypherValue::NodeRef(id) => Some(*id),
                CypherValue::Int(id) => Some(*id as u64),
                _ => None,
            })
            .ok_or_else(|| CypherError::unknown_variable(to.as_str()))?;

        // Get weight from properties or default to 1.0
        let weight = properties
            .entries
            .iter()
            .find(|(k, _)| k.as_str() == "weight")
            .and_then(|(_, v)| {
                if let Expr::Literal(CypherValue::Float(f)) = &**v {
                    Some(*f as f32)
                } else if let Expr::Literal(CypherValue::Int(i)) = &**v {
                    Some(*i as f32)
                } else {
                    None
                }
            })
            .unwrap_or(1.0);

        ctx.collection
            .add_edge(from_id, to_id, rel_type.as_str(), weight)
            .map_err(|e| CypherError::Internal {
                message: e.to_string(),
            })?;

        let mut stats = QueryStats::default();
        stats.relationships_created = 1;

        // Return the created relationship as a reference
        let rel_id = self.generate_point_id(); // Generate unique ID for the relationship
        let row = ExecutorRow::single(CypherValue::RelationshipRef(rel_id));

        Ok((vec![row], stats))
    }

    /// Execute DeleteNode operation
    pub(crate) fn execute_delete_node(
        &self,
        input: &LogicalOp,
        _variable: &String,
        _detach: bool,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (input_rows, _) = self.execute_op(input, ctx)?;

        let mut deleted = 0u64;

        for row in input_rows {
            // Find the node ID in the row
            for value in row.iter() {
                if let CypherValue::NodeRef(id) = value {
                    // TODO: If detach, remove all relationships first
                    deleted += ctx.collection.delete_points(&[*id])? as u64;
                }
            }
        }

        let mut stats = QueryStats::default();
        stats.nodes_deleted = deleted;

        Ok((Vec::new(), stats))
    }

    /// Generate a unique point ID
    pub(crate) fn generate_point_id(&self) -> PointId {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);

        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::time::{Duration, SystemTime, UNIX_EPOCH};

            // Use timestamp with atomic counter fallback for robustness
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos() as u64;

            // Combine timestamp with counter to ensure uniqueness even if time fails
            let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
            let mut hash = timestamp ^ counter;
            hash ^= hash >> 33;
            hash = hash.wrapping_mul(0xff51afd7ed558ccd);
            hash ^= hash >> 33;
            hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
            hash ^= hash >> 33;

            hash
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Use an incrementing counter with some entropy mixing
            let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
            let mut hash = counter;
            hash ^= hash >> 33;
            hash = hash.wrapping_mul(0xff51afd7ed558ccd);
            hash ^= hash >> 33;
            hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
            hash ^= hash >> 33;

            hash
        }
    }
}
