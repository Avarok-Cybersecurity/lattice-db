//! Sort execution: generic comparator plus batch fast paths for property keys.

//! Query operation execution

use super::{ExecutionContext, InternalRows, QueryExecutor, QueryStats, SortKey};
use crate::cypher::ast::{Expr, OrderByItem};
use crate::cypher::error::CypherResult;
use crate::cypher::planner::LogicalOp;
use crate::cypher::row::Row;
use crate::parallel;
use crate::types::value::CypherValue;

use super::operations::u64_to_usize_saturating;

impl QueryExecutor {
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
            // Use saturating conversion to prevent truncation on 32-bit systems
            let k = u64_to_usize_saturating(k);
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
                // Use saturating conversion to prevent truncation on 32-bit systems
                let k = u64_to_usize_saturating(k);
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
                // Use saturating conversion to prevent truncation on 32-bit systems
                let k = u64_to_usize_saturating(k);
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
}
