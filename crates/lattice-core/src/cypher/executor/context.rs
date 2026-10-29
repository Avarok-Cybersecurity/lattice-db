//! Execution context and label registry for query execution

use super::InternalRows;
use crate::cypher::error::{CypherError, CypherResult};
use crate::engine::EngineOps;
use crate::sync::SyncCell;
use crate::types::point::Point;
use crate::types::value::CypherValue;
use crate::types::SharedStr;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;
use std::sync::Arc;

/// A relationship a query matched. LatticeDB edges have no identity of their own, so a query
/// refers to one by a handle into the context's table, keyed by (start, end, type).
#[derive(Debug, Clone, PartialEq)]
pub struct RelRecord {
    pub src: u64,
    pub dst: u64,
    pub rel_type: SharedStr,
    pub weight: f32,
}

/// Incoming edges per node: target -> (source, type, weight)
type ReverseAdjacency = FxHashMap<u64, Vec<(u64, SharedStr, f32)>>;

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
    /// The collection to execute against (any storage backend via dynamic dispatch)
    pub collection: &'a mut dyn EngineOps,
    /// Query parameters
    pub parameters: HashMap<String, CypherValue>,
    /// Variable name → column index mapping for proper variable binding
    /// Updated by scan/expand operations as they add columns to rows
    variable_bindings: HashMap<String, usize>,
    /// Cache for point lookups using FxHashMap for faster integer hashing
    /// Uses `Arc<Point>` to avoid cloning entire Point structs on cache hit
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
    /// Members of each label a label predicate asked about (one index read per label)
    label_members: SyncCell<FxHashMap<String, Arc<FxHashSet<u64>>>>,
    /// Relationships matched so far; `RelationshipRef(h)` is `rels[h]`
    rels: Vec<RelRecord>,
    rel_handles: FxHashMap<(u64, u64, SharedStr), u64>,
    /// Built on the first incoming/undirected expansion (edges are stored outgoing only)
    incoming: Option<ReverseAdjacency>,
    /// Rows an `OptionalMatch` hands to the `Argument` leaf of its inner plan
    argument: Option<InternalRows>,
}

impl<'a> ExecutionContext<'a> {
    /// Create a new execution context
    pub fn new(collection: &'a mut dyn EngineOps) -> Self {
        Self {
            collection,
            parameters: HashMap::new(),
            variable_bindings: HashMap::new(),
            point_cache: SyncCell::new(FxHashMap::default()),
            property_cache: SyncCell::new(FxHashMap::default()),
            label_registry: SyncCell::new(LabelRegistry::new()),
            label_bitmap_cache: SyncCell::new(FxHashMap::default()),
            label_members: SyncCell::new(FxHashMap::default()),
            rels: Vec::new(),
            rel_handles: FxHashMap::default(),
            incoming: None,
            argument: None,
        }
    }

    /// Create a context with parameters
    pub fn with_parameters(
        collection: &'a mut dyn EngineOps,
        parameters: HashMap<String, CypherValue>,
    ) -> Self {
        Self {
            collection,
            parameters,
            variable_bindings: HashMap::new(),
            point_cache: SyncCell::new(FxHashMap::default()),
            property_cache: SyncCell::new(FxHashMap::default()),
            label_registry: SyncCell::new(LabelRegistry::new()),
            label_bitmap_cache: SyncCell::new(FxHashMap::default()),
            label_members: SyncCell::new(FxHashMap::default()),
            rels: Vec::new(),
            rel_handles: FxHashMap::default(),
            incoming: None,
            argument: None,
        }
    }

    /// Register a variable with its column index
    /// Called by scan/expand operations when they add columns to rows
    #[inline]
    pub fn bind_variable(&mut self, name: &str, column_index: usize) {
        self.variable_bindings
            .insert(name.to_string(), column_index);
    }

    /// Get the column index for a variable
    /// Returns None if the variable is not bound
    #[inline]
    pub fn get_variable_column(&self, name: &str) -> Option<usize> {
        self.variable_bindings.get(name).copied()
    }

    /// Get current number of bound variables (next column index)
    #[inline]
    pub fn column_count(&self) -> usize {
        self.variable_bindings.len()
    }

    /// Clear variable bindings (for subquery isolation)
    #[inline]
    pub fn clear_bindings(&mut self) {
        self.variable_bindings.clear();
    }

    /// Bindings sorted by column
    pub(crate) fn bindings_in_order(&self) -> Vec<(String, usize)> {
        let mut b: Vec<(String, usize)> = self
            .variable_bindings
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        b.sort_by_key(|(_, c)| *c);
        b
    }

    /// Replace all bindings: `names[i]` is column `i`
    pub(crate) fn rebind(&mut self, names: impl IntoIterator<Item = impl AsRef<str>>) {
        self.variable_bindings.clear();
        for (i, n) in names.into_iter().enumerate() {
            self.bind_variable(n.as_ref(), i);
        }
    }

    /// Drop the binding of column `col` and shift every later column down by one
    pub(crate) fn unbind_column(&mut self, col: usize) {
        self.variable_bindings.retain(|_, c| *c != col);
        for c in self.variable_bindings.values_mut() {
            if *c > col {
                *c -= 1;
            }
        }
    }

    /// Hand rows to the next `Argument` leaf
    pub(crate) fn set_argument(&mut self, rows: InternalRows) {
        self.argument = Some(rows);
    }

    /// Take the rows an enclosing `OptionalMatch` provided
    pub(crate) fn take_argument(&mut self) -> CypherResult<InternalRows> {
        self.argument
            .take()
            .ok_or_else(|| CypherError::internal("Argument evaluated outside OPTIONAL MATCH"))
    }

    /// The handle of relationship (src)-[rel_type]->(dst), registering it on first sight
    pub(crate) fn rel_handle(&mut self, src: u64, dst: u64, rel_type: &str, weight: f32) -> u64 {
        let key = (src, dst, SharedStr::from(rel_type));
        if let Some(h) = self.rel_handles.get(&key) {
            return *h;
        }
        let h = self.rels.len() as u64;
        self.rels.push(RelRecord {
            src,
            dst,
            rel_type: key.2.clone(),
            weight,
        });
        self.rel_handles.insert(key, h);
        h
    }

    /// The relationship behind a handle
    pub(crate) fn rel(&self, handle: u64) -> CypherResult<&RelRecord> {
        usize::try_from(handle)
            .ok()
            .and_then(|i| self.rels.get(i))
            .ok_or(CypherError::RelationshipNotFound { id: handle })
    }

    /// Incoming edges of `node`, building the reverse adjacency on first use (one pass over
    /// every node's outgoing edges)
    pub(crate) fn incoming(&mut self, node: u64) -> CypherResult<Vec<(u64, SharedStr, f32)>> {
        if self.incoming.is_none() {
            let mut rev: ReverseAdjacency = FxHashMap::default();
            for src in self.collection.point_ids()? {
                for e in self.collection.get_edges(src)? {
                    rev.entry(e.target_id).or_default().push((
                        src,
                        SharedStr::from(e.relation.as_str()),
                        e.weight,
                    ));
                }
            }
            self.incoming = Some(rev);
        }
        Ok(self
            .incoming
            .as_ref()
            .and_then(|rev| rev.get(&node).cloned())
            .unwrap_or_default())
    }

    /// Whether `node` carries `label` (answered from the label index)
    pub(crate) fn node_has_label(&self, node: u64, label: &str) -> CypherResult<bool> {
        if let Some(members) = self.label_members.borrow().get(label) {
            return Ok(members.contains(&node));
        }
        let members: Arc<FxHashSet<u64>> = Arc::new(
            self.collection
                .point_ids_by_label(label)?
                .into_iter()
                .collect(),
        );
        let has = members.contains(&node);
        self.label_members
            .borrow_mut()
            .insert(label.to_string(), members);
        Ok(has)
    }

    /// Get a point by ID, using cache if available
    /// Returns `Arc<Point>` to avoid cloning - cheap reference count increment
    pub fn get_point_cached(&self, id: u64) -> Option<Arc<Point>> {
        // Check cache first - Arc::clone is cheap (just ref count increment)
        {
            let cache = self.point_cache.borrow();
            if let Some(cached) = cache.get(&id) {
                return cached.as_ref().map(Arc::clone);
            }
        }

        // Not in cache, fetch from collection
        // Both native and WASM return LatticeResult<Option<Point>>
        let point = self.collection.get_point(id).ok().flatten();

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
    /// Returns cloned value for ownership transfer (O(1) with `Rc<str>`)
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
