//! Native (non-WASM) CollectionEngine implementation with async indexing
//!
//! On native builds, the engine uses async indexing for improved insert latency:
//! - Points are stored immediately in `points` storage
//! - Index updates happen in background via `AsyncIndexer`
//! - Search merges indexed results with brute-force on pending points

use crate::engine::async_indexer::{AsyncIndexerHandle, BackpressureSignal};
use crate::error::{LatticeError, LatticeResult};
use crate::index::hnsw::HnswIndex;
use crate::sync::{LockExt, MutexExt};
use crate::types::collection::CollectionConfig;
use crate::types::point::{Edge, Point, PointId};
use crate::types::query::{ScrollPoint, ScrollQuery, ScrollResult, SearchQuery, SearchResult};
use rayon::prelude::*;
use rkyv::rancor::Error as RkyvError;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ops::Bound;
use std::sync::{Arc, Condvar, Mutex, RwLock};

use super::types::{EdgeInfo, TraversalPath, TraversalResult, UpsertResult};

/// Starting page ID for point storage (reserved for Phase 2)
const PAGE_POINTS_START: u64 = 1_000_000;

/// Max pending points before blocking upsert to let indexer catch up
/// Higher = faster upserts, slower search on pending buffer
/// Lower = slower upserts (blocks more), faster search
/// Set high enough that typical benchmark batches don't trigger blocking
const PENDING_THRESHOLD: usize = 10000;

/// Collection engine - manages a single collection (Native implementation)
///
/// Thread-safe with Arc<RwLock<>> for concurrent reads.
/// Uses async indexing for improved insert latency.
pub struct CollectionEngine {
    /// Collection configuration
    config: CollectionConfig,
    /// HNSW index for vector search
    index: Arc<RwLock<HnswIndex>>,
    /// Point storage: id -> Point
    points: Arc<RwLock<BTreeMap<PointId, Point>>>,
    /// Points pending indexing (not yet in HNSW)
    pending_index: Arc<RwLock<FxHashSet<PointId>>>,
    /// Background indexer handle
    indexer: AsyncIndexerHandle,
    /// Label index: label -> set of point IDs (for O(1) label lookups)
    label_index: Arc<RwLock<HashMap<String, FxHashSet<PointId>>>>,
    /// Signal for backpressure coordination with async indexer
    backpressure_signal: BackpressureSignal,
    /// Next available page ID for storage (reserved for Phase 2)
    #[allow(dead_code)]
    next_page_id: u64,
}

impl CollectionEngine {
    /// Create a new collection engine
    pub fn new(config: CollectionConfig) -> LatticeResult<Self> {
        config.validate()?;

        let index = Arc::new(RwLock::new(HnswIndex::new(
            config.hnsw.clone(),
            config.vectors.distance,
        )));
        let points = Arc::new(RwLock::new(BTreeMap::new()));
        let pending_index = Arc::new(RwLock::new(FxHashSet::default()));
        // Pre-allocate for typical number of unique labels in a graph
        let label_index = Arc::new(RwLock::new(HashMap::with_capacity(64)));
        let backpressure_signal = Arc::new((Mutex::new(()), Condvar::new()));

        // Spawn background indexer
        let indexer = AsyncIndexerHandle::spawn(
            Arc::clone(&index),
            Arc::clone(&points),
            Arc::clone(&pending_index),
            Arc::clone(&backpressure_signal),
        );

        Ok(Self {
            config,
            index,
            points,
            pending_index,
            indexer,
            label_index,
            backpressure_signal,
            next_page_id: PAGE_POINTS_START,
        })
    }

    /// Get collection configuration
    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    /// Get collection name
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get number of points in the collection
    pub fn point_count(&self) -> usize {
        self.points.read().map(|g| g.len()).unwrap_or(0)
    }

    /// Check if collection is empty
    pub fn is_empty(&self) -> bool {
        self.points.read().map(|g| g.is_empty()).unwrap_or(true)
    }

    /// Get vector dimension
    pub fn vector_dim(&self) -> usize {
        self.config.vectors.size
    }

    /// Get number of points pending indexing
    pub fn pending_count(&self) -> usize {
        self.pending_index.read().map(|g| g.len()).unwrap_or(0)
    }

    // --- Point Operations ---

    /// Upsert points (insert or update)
    ///
    /// Points are stored immediately and indexed asynchronously in background.
    /// If pending buffer exceeds threshold, blocks until indexer catches up.
    pub fn upsert_points(&mut self, points: Vec<Point>) -> LatticeResult<UpsertResult> {
        let mut inserted = 0;
        let mut updated = 0;

        // Validate all points first
        for point in &points {
            if point.vector.len() != self.config.vectors.size {
                return Err(LatticeError::DimensionMismatch {
                    expected: self.config.vectors.size,
                    actual: point.vector.len(),
                });
            }
        }

        // Check pending threshold - block if too many pending
        {
            let pending = self.pending_index.read_safe()?;
            if pending.len() > PENDING_THRESHOLD {
                drop(pending); // Release read lock

                // Wait for indexer to catch up using condvar (efficient, no CPU spin)
                // Use exponential backoff with max retries to prevent infinite loops
                const MAX_BACKPRESSURE_RETRIES: u32 = 100;
                const INITIAL_WAIT_MS: u64 = 10;
                const MAX_WAIT_MS: u64 = 1000;

                let (lock, cv) = &*self.backpressure_signal;
                let mut guard = lock.lock_safe()?;
                let mut retries = 0u32;
                let mut wait_ms = INITIAL_WAIT_MS;

                while self.pending_index.read_safe()?.len() > PENDING_THRESHOLD / 2 {
                    if retries >= MAX_BACKPRESSURE_RETRIES {
                        // Indexer may be stuck or stopped - proceed anyway to prevent deadlock
                        break;
                    }

                    // Wait with exponential backoff
                    let result = cv
                        .wait_timeout(guard, std::time::Duration::from_millis(wait_ms))
                        .map_err(|e| LatticeError::Internal {
                            code: 50004,
                            message: format!("Condvar wait failed: {}", e),
                        })?;
                    guard = result.0;

                    retries += 1;
                    wait_ms = (wait_ms * 2).min(MAX_WAIT_MS); // Exponential backoff capped at 1s
                }
            }
        }

        // Pre-parse labels BEFORE acquiring locks (JSON parsing is expensive)
        let points_with_labels: Vec<(Point, Option<Vec<String>>)> = points
            .into_iter()
            .map(|point| {
                let labels = point
                    .payload
                    .get("_labels")
                    .and_then(|bytes| serde_json::from_slice::<Vec<String>>(bytes).ok());
                (point, labels)
            })
            .collect();

        // Store points immediately and queue for background indexing
        {
            let mut pts = self.points.write_safe()?;
            let mut pending = self.pending_index.write_safe()?;
            let mut label_idx = self.label_index.write_safe()?;

            for (point, labels) in points_with_labels {
                let is_update = pts.contains_key(&point.id);

                if is_update {
                    // For updates, remove old labels from label index
                    if let Some(old_point) = pts.get(&point.id) {
                        Self::remove_point_labels_from_index(old_point, &mut label_idx);
                    }
                    // Delete from HNSW index first (sync to ensure consistency)
                    self.indexer.queue_delete(point.id);
                    updated += 1;
                } else {
                    inserted += 1;
                }

                // Update label index with pre-parsed labels (no JSON parsing here)
                if let Some(labels) = labels {
                    for label in labels {
                        label_idx
                            .entry(label)
                            .or_insert_with(FxHashSet::default)
                            .insert(point.id);
                    }
                }

                // Store point
                pts.insert(point.id, point.clone());

                // Add to pending and queue for indexing
                pending.insert(point.id);
                self.indexer.queue_insert(point.id);
            }
        }

        Ok(UpsertResult { inserted, updated })
    }

    /// Extract labels from a point's _labels payload and add to label index
    #[allow(dead_code)]
    fn add_point_labels_to_index(
        point: &Point,
        label_idx: &mut HashMap<String, FxHashSet<PointId>>,
    ) {
        if let Some(labels_bytes) = point.payload.get("_labels") {
            if let Ok(labels) = serde_json::from_slice::<Vec<String>>(labels_bytes) {
                for label in labels {
                    label_idx
                        .entry(label)
                        .or_insert_with(FxHashSet::default)
                        .insert(point.id);
                }
            }
        }
    }

    /// Remove a point's labels from the label index
    fn remove_point_labels_from_index(
        point: &Point,
        label_idx: &mut HashMap<String, FxHashSet<PointId>>,
    ) {
        if let Some(labels_bytes) = point.payload.get("_labels") {
            if let Ok(labels) = serde_json::from_slice::<Vec<String>>(labels_bytes) {
                for label in labels {
                    if let Some(ids) = label_idx.get_mut(&label) {
                        ids.remove(&point.id);
                    }
                }
            }
        }
    }

    /// Get points by IDs (returns cloned points for thread safety)
    pub fn get_points(&self, ids: &[PointId]) -> LatticeResult<Vec<Option<Point>>> {
        let pts = self.points.read_safe()?;
        Ok(ids.iter().map(|id| pts.get(id).cloned()).collect())
    }

    /// Get a single point by ID (returns cloned point for thread safety)
    pub fn get_point(&self, id: PointId) -> LatticeResult<Option<Point>> {
        Ok(self.points.read_safe()?.get(&id).cloned())
    }

    /// Batch extract specific properties from points without cloning entire Points.
    ///
    /// This is optimized for ORDER BY queries where we only need sort key properties.
    /// Returns `Vec<Vec<Option<Vec<u8>>>>` - outer: per point, inner: per property.
    /// The values are the raw JSON bytes from the payload (caller deserializes).
    ///
    /// Returns None if point doesn't exist, Some(None) if property doesn't exist.
    pub fn batch_extract_properties(
        &self,
        ids: &[PointId],
        properties: &[&str],
    ) -> LatticeResult<Vec<Vec<Option<Vec<u8>>>>> {
        let pts = self.points.read_safe()?;
        Ok(ids
            .iter()
            .map(|id| {
                if let Some(point) = pts.get(id) {
                    properties
                        .iter()
                        .map(|prop| point.payload.get(*prop).cloned())
                        .collect()
                } else {
                    // Point not found - return None for all properties
                    vec![None; properties.len()]
                }
            })
            .collect())
    }

    /// Batch extract a single numeric property as i64, optimized for ORDER BY on integer fields.
    ///
    /// This is highly optimized:
    /// - No byte cloning (parses in-place)
    /// - No CypherValue allocation
    /// - Single lock acquisition
    /// - Fast integer parsing without serde overhead
    ///
    /// Returns i64::MIN for missing points/properties (sorts to bottom for DESC).
    pub fn batch_extract_i64_property(
        &self,
        ids: &[PointId],
        property: &str,
    ) -> LatticeResult<Vec<i64>> {
        // For large N, pre-extract ALL values in BTreeMap order (sequential access),
        // then use O(1) Vec lookup (IDs are typically sequential starting from 0)
        const LARGE_THRESHOLD: usize = 5000;

        if ids.len() >= LARGE_THRESHOLD && !ids.is_empty() {
            // Check if IDs are roughly sequential (common case after AllNodesScan)
            // Safe: ids is non-empty so min/max will succeed
            let min_id = *ids.iter().min().unwrap_or(&0);
            let max_id = *ids.iter().max().unwrap_or(&0);
            let id_range = max_id - min_id + 1;

            // Only use dense Vec if IDs are ~dense (not sparse)
            if id_range <= ids.len() as u64 * 2 {
                // Pre-extract all properties in BTreeMap order (sequential, cache-friendly)
                let pts = self.points.read_safe()?;
                let mut values: Vec<i64> = vec![i64::MIN; id_range as usize];

                // Single sequential iteration over BTreeMap
                for (&id, point) in pts.range(min_id..=max_id) {
                    let idx = (id - min_id) as usize;
                    if let Some(bytes) = point.payload.get(property) {
                        if let Some(val) = Self::fast_parse_i64(bytes) {
                            values[idx] = val;
                        }
                    }
                }

                // O(1) lookup for each id
                return Ok(ids
                    .iter()
                    .map(|id| values[(id - min_id) as usize])
                    .collect());
            }
        }

        // Fallback: direct BTreeMap lookups
        let pts = self.points.read_safe()?;
        Ok(ids
            .iter()
            .map(|id| {
                pts.get(id)
                    .and_then(|point| point.payload.get(property))
                    .and_then(|bytes| Self::fast_parse_i64(bytes))
                    .unwrap_or(i64::MIN)
            })
            .collect())
    }

    /// Fast parse i64 from JSON bytes without allocation.
    /// Returns None for non-integer values (floats, strings, etc).
    #[inline]
    fn fast_parse_i64(bytes: &[u8]) -> Option<i64> {
        if bytes.is_empty() {
            return None;
        }

        let (start, negative) = if bytes[0] == b'-' {
            if bytes.len() < 2 {
                return None;
            }
            (1, true)
        } else {
            (0, false)
        };

        let mut result: i64 = 0;
        for &b in &bytes[start..] {
            if b.is_ascii_digit() {
                result = result.checked_mul(10)?;
                let digit = (b - b'0') as i64;
                result = if negative {
                    result.checked_sub(digit)?
                } else {
                    result.checked_add(digit)?
                };
            } else {
                // Not a simple integer (float, string, etc.)
                return None;
            }
        }
        Some(result)
    }

    /// Delete points by IDs
    ///
    /// Returns the number of points actually deleted.
    pub fn delete_points(&mut self, ids: &[PointId]) -> LatticeResult<usize> {
        let mut deleted = 0;

        let mut pts = self.points.write_safe()?;
        let mut pending = self.pending_index.write_safe()?;
        let mut label_idx = self.label_index.write_safe()?;

        for &id in ids {
            if let Some(point) = pts.remove(&id) {
                // Remove from label index
                Self::remove_point_labels_from_index(&point, &mut label_idx);
                // Queue deletion from HNSW index (background)
                self.indexer.queue_delete(id);
                // Remove from pending if it was there
                pending.remove(&id);
                deleted += 1;
            }
        }

        Ok(deleted)
    }

    /// Check if a point exists
    pub fn point_exists(&self, id: PointId) -> bool {
        self.points
            .read()
            .map(|g| g.contains_key(&id))
            .unwrap_or(false)
    }

    /// Get all point IDs
    pub fn point_ids(&self) -> LatticeResult<Vec<PointId>> {
        Ok(self.points.read_safe()?.keys().copied().collect())
    }

    /// Get point IDs that have a specific label (O(1) lookup via label index)
    pub fn point_ids_by_label(&self, label: &str) -> LatticeResult<Vec<PointId>> {
        let label_idx = self.label_index.read_safe()?;
        Ok(label_idx
            .get(label)
            .map(|ids| ids.iter().copied().collect())
            .unwrap_or_default())
    }

    // --- Search Operations ---

    /// Search for nearest neighbors
    ///
    /// Searches both the HNSW index and pending (unindexed) points.
    /// Pending points are searched with parallel brute-force and merged with HNSW results.
    pub fn search(&self, query: SearchQuery) -> LatticeResult<Vec<SearchResult>> {
        use crate::index::distance::DistanceCalculator;

        // Validate query vector dimension
        if query.vector.len() != self.config.vectors.size {
            return Err(LatticeError::DimensionMismatch {
                expected: self.config.vectors.size,
                actual: query.vector.len(),
            });
        }

        // Use query ef or default from config
        let ef = query.ef.unwrap_or(self.config.hnsw.ef);

        // Search HNSW index
        let mut results = {
            let index = self.index.read_safe()?;
            index.search(&query.vector, query.limit, ef)
        };

        // Search pending points with parallel brute-force
        // Acquire both locks atomically to prevent TOCTOU race where points could be
        // deleted or replaced between reading pending_index and accessing points
        let pending_results: Vec<SearchResult> = {
            let pending = self.pending_index.read_safe()?;
            let pending_ids: Vec<PointId> = pending.iter().copied().collect();

            if pending_ids.is_empty() {
                Vec::new()
            } else {
                let pts = self.points.read_safe()?;
                let distance_calc = DistanceCalculator::new(self.config.vectors.distance);

                // Parallel brute-force on pending points
                pending_ids
                    .par_iter()
                    .filter_map(|&id| {
                        pts.get(&id).map(|point| {
                            let score = distance_calc.calculate(&query.vector, &point.vector);
                            SearchResult {
                                id,
                                score,
                                vector: None,
                                payload: None,
                            }
                        })
                    })
                    .collect()
            }
        };

        if !pending_results.is_empty() {
            // Merge results
            results.extend(pending_results);
            results.sort_by(|a, b| crate::sync::cmp_f32(a.score, b.score));
            results.truncate(query.limit);
        }

        // Apply score threshold if specified
        if let Some(threshold) = query.score_threshold {
            results.retain(|r| r.score <= threshold);
        }

        // Enrich results with vector/payload if requested
        {
            let pts = self.points.read_safe()?;
            for result in &mut results {
                if let Some(point) = pts.get(&result.id) {
                    if query.with_vector {
                        result.vector = Some(point.vector.clone());
                    }
                    if query.with_payload {
                        result.payload = Some(self.payload_to_json(&point.payload));
                    }
                }
            }
        }

        Ok(results)
    }

    /// Batch search for multiple queries (parallel processing)
    ///
    /// More efficient than calling `search` multiple times for many queries.
    /// Uses rayon for parallel query processing on native platforms.
    pub fn search_batch(&self, queries: Vec<SearchQuery>) -> LatticeResult<Vec<Vec<SearchResult>>> {
        if queries.is_empty() {
            return Ok(vec![]);
        }

        // Validate all query dimensions upfront
        for query in &queries {
            if query.vector.len() != self.config.vectors.size {
                return Err(LatticeError::DimensionMismatch {
                    expected: self.config.vectors.size,
                    actual: query.vector.len(),
                });
            }
        }

        // Extract vector references for batch search (avoids cloning 4MB+ for 1000 queries × 512-dim)
        let query_vectors: Vec<&[f32]> = queries.iter().map(|q| q.vector.as_slice()).collect();
        let first_query = &queries[0];
        let ef = first_query.ef.unwrap_or(self.config.hnsw.ef);
        let limit = first_query.limit;

        // Batch search on HNSW index
        let batch_results = {
            let index = self.index.read_safe()?;
            index.search_batch(&query_vectors, limit, ef)
        };

        // Handle pending points for each query (parallel)
        // Acquire both locks atomically to prevent TOCTOU race
        let all_results: Vec<Vec<SearchResult>> = {
            let pending = self.pending_index.read_safe()?;
            let pending_ids: Vec<PointId> = pending.iter().copied().collect();

            if pending_ids.is_empty() {
                batch_results
            } else {
                use crate::index::distance::DistanceCalculator;
                let pts = self.points.read_safe()?;
                let distance_calc = DistanceCalculator::new(self.config.vectors.distance);

                // Process each query with pending points
                let mut results: Vec<Vec<SearchResult>> = batch_results;
                for (i, query) in queries.iter().enumerate() {
                    let pending_results: Vec<SearchResult> = pending_ids
                        .par_iter()
                        .filter_map(|&id| {
                            pts.get(&id).map(|point| {
                                let score = distance_calc.calculate(&query.vector, &point.vector);
                                SearchResult {
                                    id,
                                    score,
                                    vector: None,
                                    payload: None,
                                }
                            })
                        })
                        .collect();

                    results[i].extend(pending_results);
                    results[i].sort_by(|a, b| crate::sync::cmp_f32(a.score, b.score));
                    results[i].truncate(query.limit);
                }
                results
            }
        };

        Ok(all_results)
    }

    /// Scroll through points (paginated retrieval)
    pub fn scroll(&self, query: ScrollQuery) -> LatticeResult<ScrollResult> {
        let pts = self.points.read_safe()?;

        // Use BTreeMap's range for efficient iteration from offset
        let iter: Box<dyn Iterator<Item = (&PointId, &Point)>> = match query.offset {
            Some(offset) => Box::new(pts.range((Bound::Excluded(offset), Bound::Unbounded))),
            None => Box::new(pts.iter()),
        };

        // Take limit + 1 to check if there are more results
        let page: Vec<_> = iter.take(query.limit + 1).collect();
        let has_more = page.len() > query.limit;

        // Build scroll points (only up to limit)
        let result_count = if has_more { query.limit } else { page.len() };
        let points: Vec<ScrollPoint> = page[..result_count]
            .iter()
            .map(|(&id, point)| ScrollPoint {
                id,
                vector: if query.with_vector {
                    Some(point.vector.clone())
                } else {
                    None
                },
                payload: if query.with_payload {
                    Some(self.payload_to_json(&point.payload))
                } else {
                    None
                },
            })
            .collect();

        let next_offset = if has_more {
            points.last().map(|p| p.id)
        } else {
            None
        };

        Ok(ScrollResult {
            points,
            next_offset,
        })
    }

    // --- Graph Operations ---

    /// Add an edge between two points
    pub fn add_edge(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: &str,
        weight: f32,
    ) -> LatticeResult<()> {
        // Get or create relation ID first (before taking points lock)
        let relation_id = self.get_or_create_relation_id(relation);

        let mut pts = self.points.write_safe()?;

        // Validate both points exist
        if !pts.contains_key(&from_id) {
            return Err(LatticeError::PointNotFound { id: from_id });
        }
        if !pts.contains_key(&to_id) {
            return Err(LatticeError::PointNotFound { id: to_id });
        }

        // Add edge to source point (safe: we validated from_id exists above)
        let point = pts
            .get_mut(&from_id)
            .ok_or(LatticeError::PointNotFound { id: from_id })?;
        let edge = Edge::new(to_id, weight, relation_id);

        match &mut point.outgoing_edges {
            Some(edges) => {
                // Check for duplicate
                if !edges
                    .iter()
                    .any(|e| e.target_id == to_id && e.relation_id == relation_id)
                {
                    edges.push(edge);
                }
            }
            None => {
                point.outgoing_edges = Some(SmallVec::from_vec(vec![edge]));
            }
        }

        Ok(())
    }

    /// Remove an edge between two points
    pub fn remove_edge(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: Option<&str>,
    ) -> LatticeResult<bool> {
        let mut pts = self.points.write_safe()?;

        let point = pts
            .get_mut(&from_id)
            .ok_or(LatticeError::PointNotFound { id: from_id })?;

        let removed = if let Some(edges) = &mut point.outgoing_edges {
            let original_len = edges.len();

            if let Some(rel) = relation {
                if let Some(rel_id) = self.config.relation_id(rel) {
                    edges.retain(|e| !(e.target_id == to_id && e.relation_id == rel_id));
                }
            } else {
                edges.retain(|e| e.target_id != to_id);
            }

            edges.len() < original_len
        } else {
            false
        };

        Ok(removed)
    }

    /// Get outgoing edges from a point
    pub fn get_edges(&self, point_id: PointId) -> LatticeResult<Vec<EdgeInfo>> {
        let pts = self.points.read_safe()?;

        let point = pts
            .get(&point_id)
            .ok_or(LatticeError::PointNotFound { id: point_id })?;

        let edges = point
            .outgoing_edges
            .as_ref()
            .map(|edges| {
                edges
                    .iter()
                    .map(|e| EdgeInfo {
                        target_id: e.target_id,
                        weight: e.weight,
                        relation: self
                            .config
                            .relation_name(e.relation_id)
                            .unwrap_or("unknown")
                            .to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(edges)
    }

    /// Traverse the graph from a starting point
    pub fn traverse(
        &self,
        start_id: PointId,
        max_depth: usize,
        relations: Option<&[&str]>,
    ) -> LatticeResult<TraversalResult> {
        let pts = self.points.read_safe()?;

        if !pts.contains_key(&start_id) {
            return Err(LatticeError::PointNotFound { id: start_id });
        }

        // Convert relation names to IDs
        let relation_ids: Option<Vec<u16>> = relations.map(|rels| {
            rels.iter()
                .filter_map(|r| self.config.relation_id(r))
                .collect()
        });

        let mut visited: HashMap<PointId, usize> = HashMap::new(); // id -> depth
        let mut paths: Vec<TraversalPath> = Vec::new();
        // Use VecDeque for proper BFS traversal (pop_front instead of pop)
        let mut queue: VecDeque<(PointId, usize, Vec<PointId>)> =
            VecDeque::from([(start_id, 0, vec![start_id])]);

        visited.insert(start_id, 0);

        while let Some((current_id, depth, path)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            if let Some(point) = pts.get(&current_id) {
                if let Some(edges) = &point.outgoing_edges {
                    for edge in edges.iter() {
                        // Filter by relation if specified
                        if let Some(ref rel_ids) = relation_ids {
                            if !rel_ids.contains(&edge.relation_id) {
                                continue;
                            }
                        }

                        let target_id = edge.target_id;
                        let new_depth = depth + 1;

                        // Single lookup: only visit if not visited or found at shorter depth
                        let should_visit = visited
                            .get(&target_id)
                            .map_or(true, |&prev_depth| prev_depth > new_depth);

                        if should_visit {
                            visited.insert(target_id, new_depth);

                            // Build path: requires two clones since both storage and queue need ownership
                            let mut new_path = path.clone();
                            new_path.push(target_id);

                            paths.push(TraversalPath {
                                target_id,
                                depth: new_depth,
                                path: new_path.clone(),
                                weight: edge.weight,
                                relation_id: edge.relation_id,
                            });

                            queue.push_back((target_id, new_depth, new_path));
                        }
                    }
                }
            }
        }

        Ok(TraversalResult {
            start_id,
            max_depth,
            nodes_visited: visited.len(),
            paths,
        })
    }

    // --- Helper Methods ---

    /// Convert payload HashMap to JSON Value
    fn payload_to_json(&self, payload: &HashMap<String, Vec<u8>>) -> serde_json::Value {
        let map: serde_json::Map<String, serde_json::Value> = payload
            .iter()
            .filter_map(|(k, v)| serde_json::from_slice(v).ok().map(|val| (k.clone(), val)))
            .collect();
        serde_json::Value::Object(map)
    }

    /// Get or create a relation ID
    fn get_or_create_relation_id(&mut self, relation: &str) -> u16 {
        if let Some(id) = self.config.relation_id(relation) {
            id
        } else {
            let new_id = self.config.relations.len() as u16;
            self.config.relations.insert(relation.to_string(), new_id);
            new_id
        }
    }

    /// Serialize the collection to bytes (for persistence)
    ///
    /// Format: `[config_len:u32][config:JSON][padding][points:rkyv]`
    /// Padding ensures rkyv data starts at 16-byte alignment.
    pub fn to_bytes(&self) -> LatticeResult<Vec<u8>> {
        // Serialize config as JSON (small, schema-flexible)
        let config_bytes =
            serde_json::to_vec(&self.config).map_err(|e| LatticeError::Serialization {
                message: e.to_string(),
            })?;

        // Serialize points with rkyv (zero-copy, fast)
        let pts = self.points.read_safe()?;
        let points_vec: Vec<Point> = pts.values().cloned().collect();
        drop(pts); // Release lock before serialization

        let points_bytes =
            rkyv::to_bytes::<RkyvError>(&points_vec).map_err(|e| LatticeError::Serialization {
                message: e.to_string(),
            })?;

        // Calculate padding to align rkyv data to 16 bytes
        let header_size = 4 + config_bytes.len() + 4; // config_len + config + points_len
        let padding = (16 - (header_size % 16)) % 16;

        // Build result: [config_len][config][points_len][padding][points]
        let mut result = Vec::with_capacity(header_size + padding + points_bytes.len());
        result.extend_from_slice(&(config_bytes.len() as u32).to_le_bytes());
        result.extend_from_slice(&config_bytes);
        result.extend_from_slice(&(points_bytes.len() as u32).to_le_bytes());
        result.extend(std::iter::repeat(0u8).take(padding)); // Alignment padding
        result.extend_from_slice(&points_bytes);

        Ok(result)
    }

    /// Deserialize a collection from bytes
    pub fn from_bytes(bytes: &[u8]) -> LatticeResult<Self> {
        if bytes.len() < 8 {
            return Err(LatticeError::Serialization {
                message: "Invalid data: too short".to_string(),
            });
        }

        // Read config length
        let config_len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let config_end = 4 + config_len;

        if bytes.len() < config_end + 4 {
            return Err(LatticeError::Serialization {
                message: "Invalid data: config truncated".to_string(),
            });
        }

        // Deserialize config (JSON)
        let config: CollectionConfig =
            serde_json::from_slice(&bytes[4..config_end]).map_err(|e| {
                LatticeError::Serialization {
                    message: e.to_string(),
                }
            })?;

        // Read points length
        let points_len = u32::from_le_bytes([
            bytes[config_end],
            bytes[config_end + 1],
            bytes[config_end + 2],
            bytes[config_end + 3],
        ]) as usize;

        // Calculate padding (same formula as serialization)
        let header_size = 4 + config_len + 4;
        let padding = (16 - (header_size % 16)) % 16;
        let points_start = config_end + 4 + padding;

        if bytes.len() < points_start + points_len {
            return Err(LatticeError::Serialization {
                message: "Invalid data: points truncated".to_string(),
            });
        }

        // Copy to aligned buffer for rkyv deserialization
        let mut aligned_bytes = rkyv::util::AlignedVec::<16>::new();
        aligned_bytes.extend_from_slice(&bytes[points_start..points_start + points_len]);

        // Deserialize points with rkyv
        let points_vec: Vec<Point> = rkyv::from_bytes::<Vec<Point>, RkyvError>(&aligned_bytes)
            .map_err(|e| LatticeError::Serialization {
                message: e.to_string(),
            })?;

        // Build engine
        let mut engine = Self::new(config)?;
        engine.upsert_points(points_vec)?;

        Ok(engine)
    }

    /// Wait for all pending indexing to complete
    ///
    /// This is useful for tests or when you need consistent search results.
    pub fn flush_pending(&self) -> LatticeResult<()> {
        // Use condvar to wait for background indexer instead of busy-waiting
        let (lock, cv) = &*self.backpressure_signal;

        loop {
            if self.pending_index.read_safe()?.is_empty() {
                return Ok(());
            }

            // Wait for background indexer to signal progress
            // Use timeout to handle edge cases where notification is missed
            let guard = lock.lock().map_err(|_| LatticeError::Internal {
                code: 50001,
                message: "Condvar lock poisoned in flush_pending".to_string(),
            })?;
            let _ = cv
                .wait_timeout(guard, std::time::Duration::from_millis(100))
                .map_err(|_| LatticeError::Internal {
                    code: 50002,
                    message: "Condvar wait failed in flush_pending".to_string(),
                })?;
        }
    }
}
