//! Query executor - executes logical plans against storage
//!
//! The executor traverses the logical operation tree and produces results
//! by interacting with the CollectionEngine.

use crate::cypher::ast::*;
use crate::cypher::error::{CypherError, CypherResult};
use crate::cypher::planner::LogicalOp;
use crate::cypher::row::{ExecutorRow, Row};
use crate::engine::collection::CollectionEngine;
use crate::parallel;
use crate::sync::SyncCell;
use crate::types::point::{Point, PointId};
use crate::types::value::CypherValue;
use crate::types::SharedStr;
use bumpalo::Bump;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::sync::Arc;

/// Sort key with inline storage for 1-3 keys (covers 95%+ of ORDER BY clauses)
/// SmallVec avoids heap allocation for common case, reducing memory pressure
type SortKey = SmallVec<[CypherValue; 3]>;

/// Registry for label -> bit position mapping (supports up to 64 labels)
///
/// Provides O(1) label membership checks via bitmap operations instead of
/// O(n) JSON deserialization and string comparison.
#[derive(Debug, Default)]
pub struct LabelRegistry {
    /// Map label string to bit position (0-63)
    label_to_id: HashMap<String, u8>,
    /// Reverse map for debugging/display
    id_to_label: Vec<String>,
}

impl LabelRegistry {
    /// Create a new empty label registry
    pub fn new() -> Self {
        Self {
            label_to_id: HashMap::new(),
            id_to_label: Vec::new(),
        }
    }

    /// Get or assign a bit position for a label
    /// Returns None if we've exceeded 64 labels
    pub fn get_or_create_id(&mut self, label: &str) -> Option<u8> {
        if let Some(&id) = self.label_to_id.get(label) {
            return Some(id);
        }

        // Assign new ID if we have room
        if self.id_to_label.len() >= 64 {
            return None; // Exceeded bitmap capacity
        }

        let id = self.id_to_label.len() as u8;
        self.label_to_id.insert(label.to_string(), id);
        self.id_to_label.push(label.to_string());
        Some(id)
    }

    /// Get the bit position for a label (if it exists)
    pub fn get_id(&self, label: &str) -> Option<u8> {
        self.label_to_id.get(label).copied()
    }

    /// Create a bitmap from a list of labels
    pub fn labels_to_bitmap(&mut self, labels: &[String]) -> u64 {
        let mut bitmap = 0u64;
        for label in labels {
            if let Some(id) = self.get_or_create_id(label) {
                bitmap |= 1u64 << id;
            }
        }
        bitmap
    }

    /// Check if a bitmap contains a specific label
    pub fn bitmap_has_label(&self, bitmap: u64, label: &str) -> Option<bool> {
        self.get_id(label).map(|id| (bitmap & (1u64 << id)) != 0)
    }
}

/// Context for query execution
/// Uses SyncCell for caches to enable parallel execution on native platforms.
/// SyncCell is RwLock on native (thread-safe) and RefCell on WASM (single-threaded).
pub struct ExecutionContext<'a> {
    /// The collection to execute against
    pub collection: &'a mut CollectionEngine,
    /// Query parameters
    pub parameters: HashMap<String, CypherValue>,
    /// Cache for point lookups using FxHashMap for faster integer hashing
    /// Uses Arc<Point> to avoid cloning entire Point structs on cache hit
    /// Uses SyncCell for thread-safe parallel access during sort key extraction
    point_cache: SyncCell<FxHashMap<u64, Option<Arc<Point>>>>,
    /// Two-level property cache: node_id -> (property_name -> CypherValue)
    /// Uses FxHashMap for O(1) lookups with fast integer hashing
    /// Inner map uses &str lookup to avoid String allocation
    /// Uses SyncCell for thread-safe parallel access during sort key extraction
    property_cache: SyncCell<FxHashMap<u64, FxHashMap<String, CypherValue>>>,
    /// Label registry for O(1) label checks via bitmap
    label_registry: SyncCell<LabelRegistry>,
    /// Cache for computed label bitmaps using FxHashMap for fast integer hashing
    label_bitmap_cache: SyncCell<FxHashMap<u64, u64>>,
    /// Query-scoped arena allocator for temporary allocations
    /// Reduces allocator pressure by reusing memory within a query
    #[allow(dead_code)]
    arena: Bump,
}

impl<'a> ExecutionContext<'a> {
    /// Create a new execution context
    pub fn new(collection: &'a mut CollectionEngine) -> Self {
        Self {
            collection,
            parameters: HashMap::new(),
            point_cache: SyncCell::new(FxHashMap::default()),
            property_cache: SyncCell::new(FxHashMap::default()),
            label_registry: SyncCell::new(LabelRegistry::new()),
            label_bitmap_cache: SyncCell::new(FxHashMap::default()),
            arena: Bump::with_capacity(64 * 1024), // 64KB initial capacity
        }
    }

    /// Create a context with parameters
    pub fn with_parameters(
        collection: &'a mut CollectionEngine,
        parameters: HashMap<String, CypherValue>,
    ) -> Self {
        Self {
            collection,
            parameters,
            point_cache: SyncCell::new(FxHashMap::default()),
            property_cache: SyncCell::new(FxHashMap::default()),
            label_registry: SyncCell::new(LabelRegistry::new()),
            label_bitmap_cache: SyncCell::new(FxHashMap::default()),
            arena: Bump::with_capacity(64 * 1024), // 64KB initial capacity
        }
    }

    /// Get a point by ID, using cache if available
    /// Returns Arc<Point> to avoid cloning - cheap reference count increment
    pub fn get_point_cached(&self, id: u64) -> Option<Arc<Point>> {
        // Check cache first - Arc::clone is cheap (just ref count increment)
        {
            let cache = self.point_cache.borrow();
            if let Some(cached) = cache.get(&id) {
                return cached.as_ref().map(Arc::clone);
            }
        }

        // Not in cache, fetch from collection
        // Native returns Option<Point>, WASM returns Option<&Point>
        #[cfg(not(target_arch = "wasm32"))]
        let point = self.collection.get_point(id);
        #[cfg(target_arch = "wasm32")]
        let point = self.collection.get_point(id).cloned();

        // Wrap in Arc and store in cache
        let arc_point = point.map(Arc::new);
        {
            let mut cache = self.point_cache.borrow_mut();
            cache.insert(id, arc_point.clone());
        }

        arc_point
    }

    /// Get a property value by node ID and property name, using cache if available
    /// Uses two-level map lookup to avoid String allocation on cache hits
    /// Returns cloned value for ownership transfer (O(1) with Rc<str>)
    #[inline]
    pub fn get_property_cached(&self, id: u64, property: &str) -> Option<CypherValue> {
        let cache = self.property_cache.borrow();
        // First lookup by node ID (u64), then by property name (&str)
        // No String allocation needed because HashMap::get accepts &str for String keys
        cache.get(&id)?.get(property).cloned()
    }

    /// Check if a property is cached (without cloning the value)
    #[inline]
    pub fn has_property_cached(&self, id: u64, property: &str) -> bool {
        let cache = self.property_cache.borrow();
        cache.get(&id).map_or(false, |m| m.contains_key(property))
    }

    /// Get a reference to a cached property value for zero-copy comparisons
    /// Returns a Ref guard that must be dropped before other cache operations
    /// Use this for read-only operations like comparisons to avoid any cloning
    #[inline]
    pub fn with_property_ref<F, R>(&self, id: u64, property: &str, f: F) -> Option<R>
    where
        F: FnOnce(&CypherValue) -> R,
    {
        let cache = self.property_cache.borrow();
        cache.get(&id)?.get(property).map(f)
    }

    /// Cache a property value
    /// Uses pre-allocated inner maps (8 slots) to avoid rehashing overhead
    #[inline]
    pub fn cache_property(&self, id: u64, property: &str, value: CypherValue) {
        let mut cache = self.property_cache.borrow_mut();
        cache
            .entry(id)
            .or_insert_with(|| FxHashMap::with_capacity_and_hasher(8, Default::default()))
            .insert(property.to_string(), value);
    }

    /// Pre-allocate cache capacity for expected row count
    /// Call at scan start to avoid rehashing during execution
    #[inline]
    pub fn prepare_cache_for_rows(&self, expected_rows: usize) {
        let mut point_cache = self.point_cache.borrow_mut();
        point_cache.reserve(expected_rows);

        let mut property_cache = self.property_cache.borrow_mut();
        property_cache.reserve(expected_rows);

        let mut label_cache = self.label_bitmap_cache.borrow_mut();
        label_cache.reserve(expected_rows);
    }

    /// Clear the point cache (call between row evaluations if needed)
    pub fn clear_point_cache(&self) {
        self.point_cache.borrow_mut().clear();
    }

    /// Clear the property cache
    pub fn clear_property_cache(&self) {
        self.property_cache.borrow_mut().clear();
    }

    /// Get or create a label ID for bitmap operations
    pub fn get_or_create_label_id(&self, label: &str) -> Option<u8> {
        self.label_registry.borrow_mut().get_or_create_id(label)
    }

    /// Create a bitmap from labels
    pub fn labels_to_bitmap(&self, labels: &[String]) -> u64 {
        self.label_registry.borrow_mut().labels_to_bitmap(labels)
    }

    /// Check if a bitmap contains a label
    pub fn bitmap_has_label(&self, bitmap: u64, label: &str) -> Option<bool> {
        self.label_registry.borrow().bitmap_has_label(bitmap, label)
    }

    /// Get or compute the label bitmap for a point
    /// First checks point.label_bitmap, then cache, then computes from JSON
    pub fn get_point_label_bitmap(&self, point: &Point) -> u64 {
        // If point already has a bitmap, use it
        if point.label_bitmap != 0 {
            return point.label_bitmap;
        }

        // Check cache
        {
            let cache = self.label_bitmap_cache.borrow();
            if let Some(&bitmap) = cache.get(&point.id) {
                return bitmap;
            }
        }

        // Compute from JSON labels
        let bitmap = if let Some(labels_bytes) = point.payload.get("_labels") {
            if let Ok(labels) = serde_json::from_slice::<Vec<String>>(labels_bytes) {
                self.labels_to_bitmap(&labels)
            } else {
                0
            }
        } else {
            0
        };

        // Cache the computed bitmap
        {
            let mut cache = self.label_bitmap_cache.borrow_mut();
            cache.insert(point.id, bitmap);
        }

        bitmap
    }

    /// Check if a point has a specific label
    /// Uses direct byte search to avoid JSON parsing overhead
    /// Zero-allocation implementation for maximum performance
    #[inline]
    pub fn point_has_label_fast(&self, point: &Point, label: &str) -> bool {
        if let Some(labels_bytes) = point.payload.get("_labels") {
            // Fast path: search for "label" pattern directly in bytes
            // This avoids full JSON deserialization (~1µs) and string allocation
            // JSON array format: ["Label1", "Label2", ...]
            let label_bytes = label.as_bytes();
            let quote = b'"';

            // Search for pattern: "label" in the bytes
            // We need to find a quote, followed by the label, followed by a quote
            let mut i = 0;
            while i < labels_bytes.len() {
                // Find the next quote
                if labels_bytes[i] == quote {
                    let start = i + 1;
                    // Check if we have enough space for the label + closing quote
                    if start + label_bytes.len() < labels_bytes.len() {
                        // Check if the label matches
                        let end = start + label_bytes.len();
                        if &labels_bytes[start..end] == label_bytes && labels_bytes[end] == quote {
                            return true;
                        }
                    }
                }
                i += 1;
            }
        }
        false
    }
}

/// Result of query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Column names
    pub columns: Vec<String>,
    /// Result rows
    pub rows: Vec<Vec<CypherValue>>,
    /// Execution statistics
    pub stats: QueryStats,
}

impl QueryResult {
    /// Create an empty result
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            stats: QueryStats::default(),
        }
    }

    /// Create a result with columns
    pub fn with_columns(columns: Vec<String>) -> Self {
        Self {
            columns,
            rows: Vec::new(),
            stats: QueryStats::default(),
        }
    }
}

/// Query execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryStats {
    pub nodes_created: u64,
    pub relationships_created: u64,
    pub nodes_deleted: u64,
    pub relationships_deleted: u64,
    pub properties_set: u64,
    pub labels_added: u64,
    pub execution_time_ms: u64,
}

/// Internal row type for executor operations.
/// Uses SmallVec for inline storage of 1-2 elements (common case).
/// Convert to Vec<CypherValue> at API boundaries (QueryResult).
type InternalRows = Vec<ExecutorRow>;

/// Query executor
pub struct QueryExecutor;

impl QueryExecutor {
    /// Create a new executor
    pub fn new() -> Self {
        Self
    }

    /// Execute a logical plan
    pub fn execute(
        &self,
        plan: &LogicalOp,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<QueryResult> {
        let start = std::time::Instant::now();

        let (internal_rows, mut stats) = self.execute_op(plan, ctx)?;

        // Extract column names from the projection
        let columns = self.extract_columns(plan);

        stats.execution_time_ms = start.elapsed().as_millis() as u64;

        // Convert internal SmallVec rows to Vec for public API
        let rows: Vec<Vec<CypherValue>> =
            internal_rows.into_iter().map(|row| row.to_vec()).collect();

        Ok(QueryResult {
            columns,
            rows,
            stats,
        })
    }

    /// Execute a single logical operation
    fn execute_op(
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
    fn execute_all_nodes_scan(
        &self,
        _variable: &String,
        ctx: &ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let ids = ctx.collection.point_ids();
        // Preallocate outer Vec to avoid reallocation during collect
        let mut rows = Vec::with_capacity(ids.len());
        for id in ids {
            // Single-element row - SmallVec stores inline (zero heap allocation)
            rows.push(ExecutorRow::single(CypherValue::NodeRef(id)));
        }
        Ok((rows, QueryStats::default()))
    }

    /// Execute NodeByLabelScan
    fn execute_label_scan(
        &self,
        _variable: &String,
        label: &String,
        predicate: Option<&Expr>,
        ctx: &ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        // Use label index for O(1) lookup instead of scanning all points
        let ids = ctx.collection.point_ids_by_label(label);

        // Pre-allocate caches based on expected row count (Phase 2 optimization)
        ctx.prepare_cache_for_rows(ids.len());

        // If no predicate, just return all IDs with the label
        if predicate.is_none() {
            let rows: InternalRows = ids
                .into_iter()
                .map(|id| ExecutorRow::single(CypherValue::NodeRef(id)))
                .collect();
            return Ok((rows, QueryStats::default()));
        }

        // With predicate pushdown, evaluate predicate for each node
        let pred = predicate.unwrap();
        let mut rows = Vec::with_capacity(ids.len() / 2); // estimate 50% match rate

        for id in ids {
            let row = ExecutorRow::single(CypherValue::NodeRef(id));
            match self.evaluate_predicate(pred, &row, ctx) {
                Ok(true) => rows.push(row),
                Ok(false) => continue,
                Err(_) => continue, // Skip on error
            }
        }

        Ok((rows, QueryStats::default()))
    }

    /// Execute NodeByIdSeek
    fn execute_id_seek(
        &self,
        _variable: &String,
        ids: &[u64],
        ctx: &ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let points = ctx.collection.get_points(ids);
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
    fn execute_expand(
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
                    let path = valid_paths.into_iter().next().unwrap();
                    Row::push(&mut row, CypherValue::NodeRef(path.target_id));
                    result_rows.push(row);
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
    fn execute_filter(
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
    fn execute_project(
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
    fn execute_sort(
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
    fn execute_sort_i64_fast(
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
    fn keys_are_homogeneous_i64(&self, rows: &[(SortKey, usize)]) -> bool {
        rows.iter().all(|(keys, _)| {
            keys.iter()
                .all(|k| matches!(k, CypherValue::Int(_) | CypherValue::Null))
        })
    }

    /// Extract all NodeRef IDs from the first column of rows.
    /// Returns None if any row doesn't have a NodeRef in first position.
    /// Preallocates for efficiency.
    #[inline]
    fn extract_node_ids(input_rows: &InternalRows) -> Option<Vec<u64>> {
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
    fn try_batch_i64_sort_keys(
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
            .batch_extract_i64_property(&node_ids, property);
        Some((keys, items[0].ascending))
    }

    fn try_batch_prefetch_sort_keys(
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
            .batch_extract_properties(&node_ids, &property_names);

        // Parallel deserialize JSON bytes to CypherValues
        let prefetched: Vec<Vec<CypherValue>> =
            parallel::enumerate_map_collect(&raw_properties, |_idx, row_properties| {
                row_properties
                    .iter()
                    .map(|opt_bytes| {
                        opt_bytes
                            .as_ref()
                            .and_then(|bytes| self.json_bytes_to_cypher_value(bytes).ok())
                            .unwrap_or(CypherValue::Null)
                    })
                    .collect()
            });

        Some(prefetched)
    }

    /// Extract the property name if this expression is a simple property access on first column.
    /// Pattern: Expr::Property { expr: Expr::Variable(_), property }
    fn extract_simple_property_access<'a>(&self, expr: &'a Expr) -> Option<&'a str> {
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
    fn sort_i64_simd(&self, keyed_indices: &mut [(SortKey, usize)], ascending: bool) {
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
    fn execute_skip(
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
    fn execute_limit(
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
    fn execute_distinct(
        &self,
        input: &LogicalOp,
        ctx: &mut ExecutionContext,
    ) -> CypherResult<(InternalRows, QueryStats)> {
        let (input_rows, stats) = self.execute_op(input, ctx)?;

        // Use HashMap with collision detection to avoid hash collision bugs
        // Key: hash, Value: indices of rows with that hash
        let mut seen: HashMap<u64, SmallVec<[usize; 1]>> = HashMap::with_capacity(input_rows.len());
        let mut result_rows = Vec::with_capacity(input_rows.len());

        for (idx, row) in input_rows.iter().enumerate() {
            let hash = self.hash_row(row);
            let entry = seen.entry(hash).or_default();

            // Check for collision: verify row is actually different from all rows with same hash
            let is_duplicate = entry
                .iter()
                .any(|&prev_idx| self.rows_equal(&input_rows[prev_idx], row));

            if !is_duplicate {
                entry.push(idx);
                result_rows.push(row.clone());
            }
        }

        Ok((result_rows, stats))
    }

    /// Compare two rows for equality
    #[inline]
    fn rows_equal(&self, a: &[CypherValue], b: &[CypherValue]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(va, vb)| va == vb)
    }

    /// Hash a row for deduplication using FNV-1a algorithm
    fn hash_row(&self, row: &[CypherValue]) -> u64 {
        // FNV-1a constants
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET;
        for value in row {
            hash = self.hash_value(value, hash, FNV_PRIME);
        }
        hash
    }

    /// Hash a single CypherValue
    fn hash_value(&self, value: &CypherValue, mut hash: u64, prime: u64) -> u64 {
        // Hash type discriminant using a portable approach
        // Discriminant size varies between platforms (32-bit on WASM, 64-bit on native)
        let type_tag: u8 = match value {
            CypherValue::Null => 0,
            CypherValue::Bool(_) => 1,
            CypherValue::Int(_) => 2,
            CypherValue::Float(_) => 3,
            CypherValue::String(_) => 4,
            CypherValue::Bytes(_) => 5,
            CypherValue::Date { .. } => 6,
            CypherValue::Time { .. } => 7,
            CypherValue::DateTime { .. } => 8,
            CypherValue::Duration { .. } => 9,
            CypherValue::Point2D { .. } => 10,
            CypherValue::Point3D { .. } => 11,
            CypherValue::List(_) => 12,
            CypherValue::Map(_) => 13,
            CypherValue::NodeRef(_) => 14,
            CypherValue::RelationshipRef(_) => 15,
            CypherValue::Path(_) => 16,
        };
        hash ^= type_tag as u64;
        hash = hash.wrapping_mul(prime);

        // Hash value content
        match value {
            CypherValue::Null => hash,
            CypherValue::Bool(b) => {
                hash ^= *b as u64;
                hash.wrapping_mul(prime)
            }
            CypherValue::Int(i) => {
                for byte in i.to_le_bytes() {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(prime);
                }
                hash
            }
            CypherValue::Float(f) => {
                for byte in f.to_le_bytes() {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(prime);
                }
                hash
            }
            CypherValue::String(s) => {
                for byte in s.as_bytes() {
                    hash ^= *byte as u64;
                    hash = hash.wrapping_mul(prime);
                }
                hash
            }
            CypherValue::NodeRef(id) | CypherValue::RelationshipRef(id) => {
                for byte in id.to_le_bytes() {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(prime);
                }
                hash
            }
            CypherValue::List(items) => {
                for item in items {
                    hash = self.hash_value(item, hash, prime);
                }
                hash
            }
            CypherValue::Map(entries) => {
                for (k, v) in entries {
                    for byte in k.as_bytes() {
                        hash ^= *byte as u64;
                        hash = hash.wrapping_mul(prime);
                    }
                    hash = self.hash_value(v, hash, prime);
                }
                hash
            }
            _ => hash, // Other types use just the discriminant
        }
    }

    /// Execute CreateNode operation
    fn execute_create_node(
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
    fn execute_create_relationship(
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
    fn execute_delete_node(
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
                    ctx.collection.delete_points(&[*id]);
                    deleted += 1;
                }
            }
        }

        let mut stats = QueryStats::default();
        stats.nodes_deleted = deleted;

        Ok((Vec::new(), stats))
    }

    /// Evaluate a predicate expression
    fn evaluate_predicate(
        &self,
        predicate: &Expr,
        row: &[CypherValue],
        ctx: &ExecutionContext,
    ) -> CypherResult<bool> {
        let value = self.evaluate_expr(predicate, row, ctx)?;
        Ok(value.is_truthy())
    }

    /// Evaluate an expression
    fn evaluate_expr(
        &self,
        expr: &Expr,
        row: &[CypherValue],
        ctx: &ExecutionContext,
    ) -> CypherResult<CypherValue> {
        match expr {
            Expr::Literal(value) => Ok(value.clone()),

            Expr::Variable(name) => {
                // Check parameters first
                if let Some(value) = ctx.parameters.get(name.as_str()) {
                    return Ok(value.clone());
                }

                // For now, assume first column is the variable
                // TODO: Proper variable binding
                row.first()
                    .cloned()
                    .ok_or_else(|| CypherError::unknown_variable(name.as_str()))
            }

            Expr::Property { expr, property } => {
                let base = self.evaluate_expr(expr, row, ctx)?;

                match base {
                    CypherValue::NodeRef(id) => {
                        // Check property cache first to avoid JSON deserialization
                        if let Some(cached) = ctx.get_property_cached(id, property) {
                            return Ok(cached);
                        }

                        // Get the point using point cache and extract property
                        if let Some(point) = ctx.get_point_cached(id) {
                            let value = self.get_point_property(&point, property)?;
                            // Cache the extracted property value
                            ctx.cache_property(id, property, value.clone());
                            Ok(value)
                        } else {
                            Ok(CypherValue::Null)
                        }
                    }
                    CypherValue::Map(entries) => {
                        // Find property in map
                        entries
                            .iter()
                            .find(|(k, _)| k.as_str() == property.as_str())
                            .map(|(_, v)| v.clone())
                            .ok_or_else(|| CypherError::UnknownProperty {
                                variable: "map".to_string(),
                                property: property.to_string(),
                            })
                    }
                    _ => Err(CypherError::InvalidOperation {
                        operation: "property access".to_string(),
                        value_type: base.type_name().to_string(),
                    }),
                }
            }

            Expr::Parameter(name) => ctx
                .parameters
                .get(name.as_str())
                .cloned()
                .ok_or_else(|| CypherError::unknown_variable(name.as_str())),

            Expr::BinaryOp { left, op, right } => {
                // Short-circuit evaluation for AND and OR
                match op {
                    BinaryOp::And => {
                        let left_val = self.evaluate_expr(left, row, ctx)?;
                        if !left_val.is_truthy() {
                            return Ok(CypherValue::Bool(false));
                        }
                        let right_val = self.evaluate_expr(right, row, ctx)?;
                        Ok(CypherValue::Bool(right_val.is_truthy()))
                    }
                    BinaryOp::Or => {
                        let left_val = self.evaluate_expr(left, row, ctx)?;
                        if left_val.is_truthy() {
                            return Ok(CypherValue::Bool(true));
                        }
                        let right_val = self.evaluate_expr(right, row, ctx)?;
                        Ok(CypherValue::Bool(right_val.is_truthy()))
                    }
                    _ => {
                        // Non-logical operators: evaluate both sides
                        let left_val = self.evaluate_expr(left, row, ctx)?;
                        let right_val = self.evaluate_expr(right, row, ctx)?;
                        self.evaluate_binary_op(*op, &left_val, &right_val)
                    }
                }
            }

            Expr::UnaryOp { op, expr } => {
                let val = self.evaluate_expr(expr, row, ctx)?;
                self.evaluate_unary_op(*op, &val)
            }

            Expr::IsNull { expr, negated } => {
                let val = self.evaluate_expr(expr, row, ctx)?;
                let is_null = val.is_null();
                Ok(CypherValue::Bool(if *negated { !is_null } else { is_null }))
            }

            Expr::List(items) => {
                let values: CypherResult<Vec<_>> = items
                    .iter()
                    .map(|item| self.evaluate_expr(item, row, ctx))
                    .collect();
                Ok(CypherValue::list(values?))
            }

            Expr::Star => {
                // Return all values as a list
                Ok(CypherValue::list(row.to_vec()))
            }

            _ => Err(CypherError::unsupported("Complex expression evaluation")),
        }
    }

    /// Evaluate a binary operation
    fn evaluate_binary_op(
        &self,
        op: BinaryOp,
        left: &CypherValue,
        right: &CypherValue,
    ) -> CypherResult<CypherValue> {
        match op {
            // Arithmetic
            BinaryOp::Add => self.add_values(left, right),
            BinaryOp::Sub => self.sub_values(left, right),
            BinaryOp::Mul => self.mul_values(left, right),
            BinaryOp::Div => self.div_values(left, right),
            BinaryOp::Mod => self.mod_values(left, right),

            // Comparison
            BinaryOp::Eq => Ok(CypherValue::Bool(left == right)),
            BinaryOp::Neq => Ok(CypherValue::Bool(left != right)),
            BinaryOp::Lt => Ok(CypherValue::Bool(self.compare_values(left, right).is_lt())),
            BinaryOp::Lte => Ok(CypherValue::Bool(self.compare_values(left, right).is_le())),
            BinaryOp::Gt => Ok(CypherValue::Bool(self.compare_values(left, right).is_gt())),
            BinaryOp::Gte => Ok(CypherValue::Bool(self.compare_values(left, right).is_ge())),

            // Logical
            BinaryOp::And => Ok(CypherValue::Bool(left.is_truthy() && right.is_truthy())),
            BinaryOp::Or => Ok(CypherValue::Bool(left.is_truthy() || right.is_truthy())),
            BinaryOp::Xor => Ok(CypherValue::Bool(left.is_truthy() ^ right.is_truthy())),

            // String concatenation - creates new Rc<str>
            BinaryOp::Concat => {
                let left_str = left.as_str().unwrap_or("");
                let right_str = right.as_str().unwrap_or("");
                let result = format!("{}{}", left_str, right_str);
                Ok(CypherValue::String(SharedStr::from(result)))
            }

            _ => Err(CypherError::unsupported("Unsupported binary operation")),
        }
    }

    /// Evaluate a unary operation
    fn evaluate_unary_op(&self, op: UnaryOp, value: &CypherValue) -> CypherResult<CypherValue> {
        match op {
            UnaryOp::Not => Ok(CypherValue::Bool(!value.is_truthy())),
            UnaryOp::Neg => match value {
                CypherValue::Int(i) => Ok(CypherValue::Int(-i)),
                CypherValue::Float(f) => Ok(CypherValue::Float(-f)),
                _ => Err(CypherError::InvalidOperation {
                    operation: "negation".to_string(),
                    value_type: value.type_name().to_string(),
                }),
            },
            UnaryOp::Pos => Ok(value.clone()),
        }
    }

    /// Add two values
    fn add_values(&self, left: &CypherValue, right: &CypherValue) -> CypherResult<CypherValue> {
        match (left, right) {
            (CypherValue::Int(a), CypherValue::Int(b)) => Ok(CypherValue::Int(a + b)),
            (CypherValue::Float(a), CypherValue::Float(b)) => Ok(CypherValue::Float(a + b)),
            (CypherValue::Int(a), CypherValue::Float(b)) => Ok(CypherValue::Float(*a as f64 + b)),
            (CypherValue::Float(a), CypherValue::Int(b)) => Ok(CypherValue::Float(a + *b as f64)),
            (CypherValue::String(a), CypherValue::String(b)) => {
                let result = format!("{}{}", a, b);
                Ok(CypherValue::String(SharedStr::from(result)))
            }
            _ => Err(CypherError::IncomparableTypes {
                left_type: left.type_name().to_string(),
                right_type: right.type_name().to_string(),
            }),
        }
    }

    /// Subtract two values
    fn sub_values(&self, left: &CypherValue, right: &CypherValue) -> CypherResult<CypherValue> {
        match (left, right) {
            (CypherValue::Int(a), CypherValue::Int(b)) => Ok(CypherValue::Int(a - b)),
            (CypherValue::Float(a), CypherValue::Float(b)) => Ok(CypherValue::Float(a - b)),
            (CypherValue::Int(a), CypherValue::Float(b)) => Ok(CypherValue::Float(*a as f64 - b)),
            (CypherValue::Float(a), CypherValue::Int(b)) => Ok(CypherValue::Float(a - *b as f64)),
            _ => Err(CypherError::IncomparableTypes {
                left_type: left.type_name().to_string(),
                right_type: right.type_name().to_string(),
            }),
        }
    }

    /// Multiply two values
    fn mul_values(&self, left: &CypherValue, right: &CypherValue) -> CypherResult<CypherValue> {
        match (left, right) {
            (CypherValue::Int(a), CypherValue::Int(b)) => Ok(CypherValue::Int(a * b)),
            (CypherValue::Float(a), CypherValue::Float(b)) => Ok(CypherValue::Float(a * b)),
            (CypherValue::Int(a), CypherValue::Float(b)) => Ok(CypherValue::Float(*a as f64 * b)),
            (CypherValue::Float(a), CypherValue::Int(b)) => Ok(CypherValue::Float(a * *b as f64)),
            _ => Err(CypherError::IncomparableTypes {
                left_type: left.type_name().to_string(),
                right_type: right.type_name().to_string(),
            }),
        }
    }

    /// Divide two values
    fn div_values(&self, left: &CypherValue, right: &CypherValue) -> CypherResult<CypherValue> {
        match (left, right) {
            (CypherValue::Int(a), CypherValue::Int(b)) => {
                if *b == 0 {
                    Err(CypherError::DivisionByZero)
                } else {
                    Ok(CypherValue::Int(a / b))
                }
            }
            (CypherValue::Float(a), CypherValue::Float(b)) => {
                if *b == 0.0 {
                    Err(CypherError::DivisionByZero)
                } else {
                    Ok(CypherValue::Float(a / b))
                }
            }
            (CypherValue::Int(a), CypherValue::Float(b)) => {
                if *b == 0.0 {
                    Err(CypherError::DivisionByZero)
                } else {
                    Ok(CypherValue::Float(*a as f64 / b))
                }
            }
            (CypherValue::Float(a), CypherValue::Int(b)) => {
                if *b == 0 {
                    Err(CypherError::DivisionByZero)
                } else {
                    Ok(CypherValue::Float(a / *b as f64))
                }
            }
            _ => Err(CypherError::IncomparableTypes {
                left_type: left.type_name().to_string(),
                right_type: right.type_name().to_string(),
            }),
        }
    }

    /// Modulo two values
    fn mod_values(&self, left: &CypherValue, right: &CypherValue) -> CypherResult<CypherValue> {
        match (left, right) {
            (CypherValue::Int(a), CypherValue::Int(b)) => {
                if *b == 0 {
                    Err(CypherError::DivisionByZero)
                } else {
                    Ok(CypherValue::Int(a % b))
                }
            }
            (CypherValue::Float(a), CypherValue::Float(b)) => {
                if *b == 0.0 {
                    Err(CypherError::DivisionByZero)
                } else {
                    Ok(CypherValue::Float(a % b))
                }
            }
            _ => Err(CypherError::IncomparableTypes {
                left_type: left.type_name().to_string(),
                right_type: right.type_name().to_string(),
            }),
        }
    }

    /// Compare two values
    fn compare_values(&self, left: &CypherValue, right: &CypherValue) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        match (left, right) {
            (CypherValue::Null, CypherValue::Null) => Ordering::Equal,
            (CypherValue::Null, _) => Ordering::Less,
            (_, CypherValue::Null) => Ordering::Greater,

            (CypherValue::Bool(a), CypherValue::Bool(b)) => a.cmp(b),
            (CypherValue::Int(a), CypherValue::Int(b)) => a.cmp(b),
            (CypherValue::Float(a), CypherValue::Float(b)) => {
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            (CypherValue::Int(a), CypherValue::Float(b)) => {
                (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            (CypherValue::Float(a), CypherValue::Int(b)) => {
                a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal)
            }
            (CypherValue::String(a), CypherValue::String(b)) => a.cmp(b),

            _ => Ordering::Equal, // Default for incompatible types
        }
    }

    /// Check if a point has a specific label
    fn point_has_label(&self, point: &Point, label: &str) -> bool {
        if let Some(labels_bytes) = point.payload.get("_labels") {
            if let Ok(labels) = serde_json::from_slice::<Vec<String>>(labels_bytes) {
                return labels.iter().any(|l| l == label);
            }
        }
        false
    }

    /// Get a property from a point
    fn get_point_property(&self, point: &Point, property: &str) -> CypherResult<CypherValue> {
        // Check for special properties
        match property {
            "id" => return Ok(CypherValue::Int(point.id as i64)),
            "_labels" => {
                if let Some(labels_bytes) = point.payload.get("_labels") {
                    if let Ok(labels) = serde_json::from_slice::<Vec<String>>(labels_bytes) {
                        return Ok(CypherValue::list(
                            labels
                                .into_iter()
                                .map(CypherValue::from)
                                .collect::<Vec<_>>(),
                        ));
                    }
                }
                return Ok(CypherValue::list(vec![]));
            }
            _ => {}
        }

        // Get from payload
        if let Some(value_bytes) = point.payload.get(property) {
            self.json_bytes_to_cypher_value(value_bytes)
        } else {
            Ok(CypherValue::Null)
        }
    }

    /// Convert JSON bytes to CypherValue
    /// Optimized with fast paths for simple types to avoid full JSON parsing
    #[inline]
    fn json_bytes_to_cypher_value(&self, bytes: &[u8]) -> CypherResult<CypherValue> {
        // Fast path for common simple types - avoids serde_json overhead
        if bytes.is_empty() {
            return Ok(CypherValue::Null);
        }

        match bytes[0] {
            // Null
            b'n' if bytes == b"null" => return Ok(CypherValue::Null),

            // Boolean
            b't' if bytes == b"true" => return Ok(CypherValue::Bool(true)),
            b'f' if bytes == b"false" => return Ok(CypherValue::Bool(false)),

            // Integer (fast path - direct parsing without iterator allocation)
            b'0'..=b'9' => {
                // Fast path for positive integers: parse directly
                if let Some(val) = self.fast_parse_positive_int(bytes) {
                    return Ok(CypherValue::Int(val));
                }
                // Fall through to serde_json for floats or complex numbers
            }
            b'-' if bytes.len() > 1 => {
                // Fast path for negative integers
                if let Some(val) = self.fast_parse_negative_int(bytes) {
                    return Ok(CypherValue::Int(val));
                }
                // Fall through to serde_json for floats or complex numbers
            }

            // String (fast path for simple strings without escapes)
            b'"' if bytes.len() >= 2 && bytes[bytes.len() - 1] == b'"' => {
                // Check if it's a simple string (no escape sequences)
                let inner = &bytes[1..bytes.len() - 1];
                if !inner.contains(&b'\\') {
                    if let Ok(s) = std::str::from_utf8(inner) {
                        return Ok(CypherValue::String(SharedStr::from(s)));
                    }
                }
                // Fall through to serde_json for strings with escapes
            }

            _ => {}
        }

        // Fall back to serde_json for complex types (arrays, objects, escaped strings)
        let json: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| CypherError::Internal {
                message: e.to_string(),
            })?;

        self.json_to_cypher_value(&json)
    }

    /// Fast parse a positive integer from bytes without any allocation
    #[inline]
    fn fast_parse_positive_int(&self, bytes: &[u8]) -> Option<i64> {
        let mut result: i64 = 0;
        for &b in bytes {
            if b.is_ascii_digit() {
                result = result.checked_mul(10)?.checked_add((b - b'0') as i64)?;
            } else {
                // Not a simple integer (float, exponent, etc.)
                return None;
            }
        }
        Some(result)
    }

    /// Fast parse a negative integer from bytes without any allocation
    #[inline]
    fn fast_parse_negative_int(&self, bytes: &[u8]) -> Option<i64> {
        if bytes.len() < 2 || bytes[0] != b'-' {
            return None;
        }
        let mut result: i64 = 0;
        for &b in &bytes[1..] {
            if b.is_ascii_digit() {
                result = result.checked_mul(10)?.checked_sub((b - b'0') as i64)?;
            } else {
                // Not a simple integer (float, exponent, etc.)
                return None;
            }
        }
        Some(result)
    }

    /// Convert JSON value to CypherValue
    fn json_to_cypher_value(&self, json: &serde_json::Value) -> CypherResult<CypherValue> {
        match json {
            serde_json::Value::Null => Ok(CypherValue::Null),
            serde_json::Value::Bool(b) => Ok(CypherValue::Bool(*b)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(CypherValue::Int(i))
                } else if let Some(f) = n.as_f64() {
                    Ok(CypherValue::Float(f))
                } else {
                    Ok(CypherValue::Null)
                }
            }
            serde_json::Value::String(s) => Ok(CypherValue::String(SharedStr::from(s.as_str()))),
            serde_json::Value::Array(arr) => {
                let values: CypherResult<Vec<_>> =
                    arr.iter().map(|v| self.json_to_cypher_value(v)).collect();
                Ok(CypherValue::list(values?))
            }
            serde_json::Value::Object(obj) => {
                let entries: CypherResult<Vec<_>> = obj
                    .iter()
                    .map(|(k, v)| Ok((k.as_str(), self.json_to_cypher_value(v)?)))
                    .collect();
                Ok(CypherValue::map_from(entries?))
            }
        }
    }

    /// Convert CypherValue to JSON bytes
    fn cypher_value_to_json_bytes(&self, value: &CypherValue) -> CypherResult<Vec<u8>> {
        let json = self.cypher_value_to_json(value)?;
        serde_json::to_vec(&json).map_err(|e| CypherError::Internal {
            message: e.to_string(),
        })
    }

    /// Convert CypherValue to JSON
    fn cypher_value_to_json(&self, value: &CypherValue) -> CypherResult<serde_json::Value> {
        match value {
            CypherValue::Null => Ok(serde_json::Value::Null),
            CypherValue::Bool(b) => Ok(serde_json::Value::Bool(*b)),
            CypherValue::Int(i) => Ok(serde_json::Value::Number((*i).into())),
            CypherValue::Float(f) => Ok(serde_json::Value::Number(
                serde_json::Number::from_f64(*f).unwrap_or_else(|| serde_json::Number::from(0)),
            )),
            CypherValue::String(s) => Ok(serde_json::Value::String((*s).to_string())),
            CypherValue::List(items) => {
                let json_items: CypherResult<Vec<_>> =
                    items.iter().map(|v| self.cypher_value_to_json(v)).collect();
                Ok(serde_json::Value::Array(json_items?))
            }
            CypherValue::Map(entries) => {
                let mut obj = serde_json::Map::new();
                for (k, v) in entries.iter() {
                    obj.insert(k.to_string(), self.cypher_value_to_json(v)?);
                }
                Ok(serde_json::Value::Object(obj))
            }
            _ => Ok(serde_json::Value::Null), // Simplify complex types to null
        }
    }

    /// Generate a unique point ID
    fn generate_point_id(&self) -> PointId {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Simple hash to distribute IDs
        let mut hash = timestamp;
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51afd7ed558ccd);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
        hash ^= hash >> 33;

        hash
    }

    /// Extract column names from a plan
    fn extract_columns(&self, plan: &LogicalOp) -> Vec<String> {
        match plan {
            LogicalOp::Project { items, .. } => items
                .iter()
                .enumerate()
                .map(|(i, item)| {
                    item.alias
                        .as_ref()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| self.expr_to_column_name(&item.expr, i))
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Convert expression to column name
    fn expr_to_column_name(&self, expr: &Expr, index: usize) -> String {
        match expr {
            Expr::Variable(name) => name.to_string(),
            Expr::Property { expr, property } => {
                format!("{}.{}", self.expr_to_column_name(expr, index), property)
            }
            Expr::Star => "*".to_string(),
            _ => format!("column_{}", index),
        }
    }
}

impl Default for QueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn test_config() -> CollectionConfig {
        CollectionConfig::new(
            "test_cypher",
            VectorConfig::new(4, Distance::Cosine),
            HnswConfig {
                m: 16,
                m0: 32,
                ml: HnswConfig::recommended_ml(16),
                ef: 100,
                ef_construction: 200,
            },
        )
    }

    #[test]
    fn test_execute_create_node() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();
        let executor = QueryExecutor::new();

        let plan = LogicalOp::CreateNode {
            labels: vec![String::from("Person")],
            properties: MapLiteral::from_entries([
                ("name", Expr::literal("Alice")),
                ("age", Expr::literal(30i64)),
            ]),
            variable: Some(String::from("n")),
        };

        let mut ctx = ExecutionContext::new(&mut engine);
        let result = executor.execute(&plan, &mut ctx).unwrap();

        assert_eq!(result.stats.nodes_created, 1);
        assert_eq!(result.rows.len(), 1);

        // Verify node exists
        assert_eq!(engine.point_count(), 1);
    }

    #[test]
    fn test_execute_all_nodes_scan() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();
        let executor = QueryExecutor::new();

        // Create some nodes
        engine
            .upsert_points(vec![
                Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4]),
                Point::new_vector(2, vec![0.2, 0.3, 0.4, 0.5]),
            ])
            .unwrap();

        let plan = LogicalOp::AllNodesScan {
            variable: String::from("n"),
        };

        let mut ctx = ExecutionContext::new(&mut engine);
        let (rows, _) = executor.execute_op(&plan, &mut ctx).unwrap();

        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_execute_limit() {
        let mut engine = CollectionEngine::new(test_config()).unwrap();
        let executor = QueryExecutor::new();

        // Create some nodes
        for i in 0..10 {
            engine
                .upsert_points(vec![Point::new_vector(i, vec![0.1, 0.2, 0.3, 0.4])])
                .unwrap();
        }

        let plan = LogicalOp::Limit {
            input: Box::new(LogicalOp::AllNodesScan {
                variable: String::from("n"),
            }),
            count: 5,
        };

        let mut ctx = ExecutionContext::new(&mut engine);
        let (rows, _) = executor.execute_op(&plan, &mut ctx).unwrap();

        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn test_evaluate_arithmetic() {
        let executor = QueryExecutor::new();

        // Addition
        let result = executor
            .add_values(&CypherValue::Int(5), &CypherValue::Int(3))
            .unwrap();
        assert_eq!(result, CypherValue::Int(8));

        // Subtraction
        let result = executor
            .sub_values(&CypherValue::Int(5), &CypherValue::Int(3))
            .unwrap();
        assert_eq!(result, CypherValue::Int(2));

        // Multiplication
        let result = executor
            .mul_values(&CypherValue::Int(5), &CypherValue::Int(3))
            .unwrap();
        assert_eq!(result, CypherValue::Int(15));

        // Division
        let result = executor
            .div_values(&CypherValue::Int(6), &CypherValue::Int(3))
            .unwrap();
        assert_eq!(result, CypherValue::Int(2));
    }

    #[test]
    fn test_evaluate_comparison() {
        let executor = QueryExecutor::new();

        assert!(executor
            .compare_values(&CypherValue::Int(5), &CypherValue::Int(3))
            .is_gt());
        assert!(executor
            .compare_values(&CypherValue::Int(3), &CypherValue::Int(5))
            .is_lt());
        assert!(executor
            .compare_values(&CypherValue::Int(5), &CypherValue::Int(5))
            .is_eq());
    }

    #[test]
    fn test_evaluate_logical() {
        let executor = QueryExecutor::new();

        let result = executor
            .evaluate_binary_op(
                BinaryOp::And,
                &CypherValue::Bool(true),
                &CypherValue::Bool(true),
            )
            .unwrap();
        assert_eq!(result, CypherValue::Bool(true));

        let result = executor
            .evaluate_binary_op(
                BinaryOp::And,
                &CypherValue::Bool(true),
                &CypherValue::Bool(false),
            )
            .unwrap();
        assert_eq!(result, CypherValue::Bool(false));

        let result = executor
            .evaluate_binary_op(
                BinaryOp::Or,
                &CypherValue::Bool(false),
                &CypherValue::Bool(true),
            )
            .unwrap();
        assert_eq!(result, CypherValue::Bool(true));
    }
}
