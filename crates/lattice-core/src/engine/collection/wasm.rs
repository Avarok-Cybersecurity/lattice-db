//! WASM CollectionEngine implementation with synchronous indexing
//!
//! On WASM builds, indexing is synchronous since:
//! - No multi-threading available in standard WASM
//! - Direct ownership without Arc<RwLock<>>
//! - Simpler memory model

use crate::error::{LatticeError, LatticeResult};
use crate::index::hnsw::HnswIndex;
use crate::types::collection::CollectionConfig;
use crate::types::point::{Edge, Point, PointId};
use crate::types::query::{ScrollPoint, ScrollQuery, ScrollResult, SearchQuery, SearchResult};
use rkyv::rancor::Error as RkyvError;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ops::Bound;

use super::types::{EdgeInfo, TraversalPath, TraversalResult, UpsertResult};

/// Starting page ID for point storage (reserved for Phase 2)
const PAGE_POINTS_START: u64 = 1_000_000;

/// Collection engine - manages a single collection (WASM implementation)
///
/// Synchronous indexing with direct ownership (no Arc/RwLock needed).
pub struct CollectionEngine {
    /// Collection configuration
    config: CollectionConfig,
    /// HNSW index for vector search
    index: HnswIndex,
    /// Point storage: id -> Point (BTreeMap for O(log n) sorted iteration)
    points: BTreeMap<PointId, Point>,
    /// Label index: label -> set of point IDs (for O(1) label lookups)
    label_index: HashMap<String, FxHashSet<PointId>>,
    /// Next available page ID for storage (reserved for Phase 2)
    #[allow(dead_code)]
    next_page_id: u64,
}

impl CollectionEngine {
    /// Create a new collection engine
    pub fn new(config: CollectionConfig) -> LatticeResult<Self> {
        config.validate()?;

        let index = HnswIndex::new(config.hnsw.clone(), config.vectors.distance);

        Ok(Self {
            config,
            index,
            points: BTreeMap::new(),
            // Pre-allocate for typical number of unique labels in a graph
            label_index: HashMap::with_capacity(64),
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
        self.points.len()
    }

    /// Check if collection is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get vector dimension
    pub fn vector_dim(&self) -> usize {
        self.config.vectors.size
    }

    // --- Point Operations ---

    /// Upsert points (insert or update)
    ///
    /// Points with existing IDs are updated, new IDs are inserted.
    pub fn upsert_points(&mut self, points: Vec<Point>) -> LatticeResult<UpsertResult> {
        let mut inserted = 0;
        let mut updated = 0;

        for point in points {
            // Validate vector dimension
            if point.vector.len() != self.config.vectors.size {
                return Err(LatticeError::DimensionMismatch {
                    expected: self.config.vectors.size,
                    actual: point.vector.len(),
                });
            }

            let is_update = self.points.contains_key(&point.id);

            // Update index
            if is_update {
                // Remove old labels from label index
                if let Some(old_point) = self.points.get(&point.id) {
                    Self::remove_point_labels_from_index(old_point, &mut self.label_index);
                }
                // Remove old entry from HNSW index
                self.index.delete(point.id);
                updated += 1;
            } else {
                inserted += 1;
            }

            // Update label index with new labels
            Self::add_point_labels_to_index(&point, &mut self.label_index);

            // Insert into HNSW index
            self.index.insert(&point)?;

            // Store point
            self.points.insert(point.id, point);
        }

        Ok(UpsertResult { inserted, updated })
    }

    /// Extract labels from a point's _labels payload and add to label index
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

    /// Get points by IDs
    pub fn get_points(&self, ids: &[PointId]) -> LatticeResult<Vec<Option<Point>>> {
        Ok(ids.iter().map(|id| self.points.get(id).cloned()).collect())
    }

    /// Get a single point by ID
    pub fn get_point(&self, id: PointId) -> LatticeResult<Option<Point>> {
        Ok(self.points.get(&id).cloned())
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
        Ok(ids
            .iter()
            .map(|id| {
                if let Some(point) = self.points.get(id) {
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
    /// - Fast integer parsing without serde overhead
    ///
    /// Returns i64::MIN for missing points/properties (sorts to bottom for DESC).
    pub fn batch_extract_i64_property(
        &self,
        ids: &[PointId],
        property: &str,
    ) -> LatticeResult<Vec<i64>> {
        Ok(ids
            .iter()
            .map(|id| {
                self.points
                    .get(id)
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

        for &id in ids {
            if let Some(point) = self.points.remove(&id) {
                // Remove from label index
                Self::remove_point_labels_from_index(&point, &mut self.label_index);
                self.index.delete(id);
                deleted += 1;
            }
        }

        Ok(deleted)
    }

    /// Check if a point exists
    pub fn point_exists(&self, id: PointId) -> bool {
        self.points.contains_key(&id)
    }

    /// Get all point IDs
    pub fn point_ids(&self) -> LatticeResult<Vec<PointId>> {
        Ok(self.points.keys().copied().collect())
    }

    /// Get point IDs that have a specific label (O(1) lookup via label index)
    pub fn point_ids_by_label(&self, label: &str) -> LatticeResult<Vec<PointId>> {
        Ok(self
            .label_index
            .get(label)
            .map(|ids| ids.iter().copied().collect())
            .unwrap_or_default())
    }

    // --- Search Operations ---

    /// Search for nearest neighbors
    pub fn search(&self, query: SearchQuery) -> LatticeResult<Vec<SearchResult>> {
        // Validate query vector dimension
        if query.vector.len() != self.config.vectors.size {
            return Err(LatticeError::DimensionMismatch {
                expected: self.config.vectors.size,
                actual: query.vector.len(),
            });
        }

        // Use query ef or default from config
        let ef = query.ef.unwrap_or(self.config.hnsw.ef);

        // Perform HNSW search
        let mut results = self.index.search(&query.vector, query.limit, ef);

        // Apply score threshold if specified
        if let Some(threshold) = query.score_threshold {
            results.retain(|r| r.score <= threshold);
        }

        // Enrich results with vector/payload if requested
        for result in &mut results {
            if let Some(point) = self.points.get(&result.id) {
                if query.with_vector {
                    result.vector = Some(point.vector.clone());
                }
                if query.with_payload {
                    result.payload = Some(self.payload_to_json(&point.payload));
                }
            }
        }

        Ok(results)
    }

    /// Batch search for multiple queries (sequential processing on WASM)
    ///
    /// WASM version processes queries sequentially since rayon is not available.
    pub fn search_batch(&self, queries: Vec<SearchQuery>) -> LatticeResult<Vec<Vec<SearchResult>>> {
        queries.into_iter().map(|q| self.search(q)).collect()
    }

    /// Scroll through points (paginated retrieval)
    ///
    /// Uses BTreeMap range iteration for O(log n + limit) instead of O(n log n).
    pub fn scroll(&self, query: ScrollQuery) -> LatticeResult<ScrollResult> {
        // Use BTreeMap's range for efficient iteration from offset
        // This is O(log n) to find start position, then O(limit) to iterate
        let iter: Box<dyn Iterator<Item = (&PointId, &Point)>> = match query.offset {
            Some(offset) => {
                // Start after the offset ID
                Box::new(
                    self.points
                        .range((Bound::Excluded(offset), Bound::Unbounded)),
                )
            }
            None => {
                // Start from the beginning
                Box::new(self.points.iter())
            }
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
        // Validate both points exist
        if !self.points.contains_key(&from_id) {
            return Err(LatticeError::PointNotFound { id: from_id });
        }
        if !self.points.contains_key(&to_id) {
            return Err(LatticeError::PointNotFound { id: to_id });
        }

        // Get or create relation ID
        let relation_id = self.get_or_create_relation_id(relation);

        // Add edge to source point
        let point = self.points.get_mut(&from_id).unwrap();
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
        let point = self
            .points
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
        let point = self
            .points
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
        if !self.points.contains_key(&start_id) {
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

            if let Some(point) = self.points.get(&current_id) {
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
        let points_vec: Vec<Point> = self.points.values().cloned().collect();
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

        // Read config length with overflow check
        let config_len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let config_end =
            4usize
                .checked_add(config_len)
                .ok_or_else(|| LatticeError::Serialization {
                    message: "Invalid data: config length overflow".to_string(),
                })?;

        let config_end_plus_4 =
            config_end
                .checked_add(4)
                .ok_or_else(|| LatticeError::Serialization {
                    message: "Invalid data: offset overflow".to_string(),
                })?;

        if bytes.len() < config_end_plus_4 {
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

        // Calculate padding (same formula as serialization) with overflow checks
        let header_size = 4usize
            .checked_add(config_len)
            .and_then(|v| v.checked_add(4))
            .ok_or_else(|| LatticeError::Serialization {
                message: "Invalid data: header size overflow".to_string(),
            })?;
        let padding = (16 - (header_size % 16)) % 16;
        let points_start = config_end
            .checked_add(4)
            .and_then(|v| v.checked_add(padding))
            .ok_or_else(|| LatticeError::Serialization {
                message: "Invalid data: points start offset overflow".to_string(),
            })?;

        let required_len =
            points_start
                .checked_add(points_len)
                .ok_or_else(|| LatticeError::Serialization {
                    message: "Invalid data: total size overflow".to_string(),
                })?;

        if bytes.len() < required_len {
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
}
