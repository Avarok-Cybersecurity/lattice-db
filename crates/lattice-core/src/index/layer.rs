//! HNSW layer management
//!
//! Handles the multi-layer graph structure where each layer has
//! fewer nodes than the layer below.

use crate::types::point::PointId;
use rustc_hash::FxHashMap;

/// Node in the HNSW graph
///
/// Each node exists in layer 0 and possibly higher layers.
/// Connections are maintained per-layer.
#[derive(Debug, Clone)]
pub struct HnswNode {
    /// Point ID this node represents
    pub point_id: PointId,
    /// Maximum layer this node exists in
    pub max_layer: u16,
    /// Neighbors per layer: neighbors[layer] = vec of neighbor IDs
    pub neighbors: Vec<Vec<PointId>>,
}

impl HnswNode {
    /// Create a new node at a given layer
    ///
    /// Pre-allocates neighbor vectors with typical capacity to reduce reallocations.
    pub fn new(point_id: PointId, max_layer: u16) -> Self {
        // Pre-allocate with typical M/M0 capacity to avoid reallocations
        // Layer 0 typically has M0=32 neighbors, higher layers have M=16
        let neighbors = (0..=max_layer as usize)
            .map(|layer| {
                if layer == 0 {
                    Vec::with_capacity(32) // M0
                } else {
                    Vec::with_capacity(16) // M
                }
            })
            .collect();
        Self {
            point_id,
            max_layer,
            neighbors,
        }
    }

    /// Get neighbors at a specific layer
    pub fn neighbors_at(&self, layer: u16) -> &[PointId] {
        self.neighbors.get(layer as usize).map_or(&[], |v| v.as_slice())
    }

    /// Get mutable neighbors at a specific layer
    pub fn neighbors_at_mut(&mut self, layer: u16) -> Option<&mut Vec<PointId>> {
        self.neighbors.get_mut(layer as usize)
    }

    /// Add a neighbor at a specific layer
    ///
    /// Uses binary search for O(log n) duplicate check instead of O(n) linear scan.
    /// Maintains sorted order for efficient iteration and lookup.
    pub fn add_neighbor(&mut self, layer: u16, neighbor: PointId) {
        if let Some(neighbors) = self.neighbors.get_mut(layer as usize) {
            match neighbors.binary_search(&neighbor) {
                Ok(_) => {} // Already exists
                Err(pos) => neighbors.insert(pos, neighbor), // Insert at sorted position
            }
        }
    }

    /// Remove a neighbor at a specific layer
    ///
    /// Uses binary search for O(log n) lookup.
    pub fn remove_neighbor(&mut self, layer: u16, neighbor: PointId) {
        if let Some(neighbors) = self.neighbors.get_mut(layer as usize) {
            if let Ok(pos) = neighbors.binary_search(&neighbor) {
                neighbors.remove(pos);
            }
        }
    }

    /// Check if this node exists at a given layer
    pub fn exists_at(&self, layer: u16) -> bool {
        layer <= self.max_layer
    }
}

/// Layer manager for HNSW graph
///
/// Tracks which nodes exist at each layer and provides efficient lookups.
#[derive(Debug, Default)]
pub struct LayerManager {
    /// All nodes indexed by point ID (FxHashMap for faster integer hashing)
    nodes: FxHashMap<PointId, HnswNode>,
    /// Entry point (highest layer node)
    entry_point: Option<PointId>,
    /// Current maximum layer in the graph
    max_layer: u16,
}

impl LayerManager {
    /// Create a new layer manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the entry point
    pub fn entry_point(&self) -> Option<PointId> {
        self.entry_point
    }

    /// Get the current maximum layer
    pub fn max_layer(&self) -> u16 {
        self.max_layer
    }

    /// Get a node by ID
    pub fn get_node(&self, id: PointId) -> Option<&HnswNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, id: PointId) -> Option<&mut HnswNode> {
        self.nodes.get_mut(&id)
    }

    /// Insert a new node
    ///
    /// Updates entry point if the new node has a higher layer.
    pub fn insert_node(&mut self, node: HnswNode) {
        let id = node.point_id;
        let node_layer = node.max_layer;

        // Update entry point if this node has a higher layer
        if self.entry_point.is_none() || node_layer > self.max_layer {
            self.entry_point = Some(id);
            self.max_layer = node_layer;
        }

        self.nodes.insert(id, node);
    }

    /// Remove a node
    ///
    /// Also removes it from all neighbor lists using swap_remove for O(1) per list.
    /// Re-sorts affected lists to maintain binary_search invariant.
    pub fn remove_node(&mut self, id: PointId) -> Option<HnswNode> {
        let node = self.nodes.remove(&id)?;

        // Remove from all neighbor lists using swap_remove (O(1)) + sort
        // For small M (16-32), this is faster than element-by-element shifting
        for other_node in self.nodes.values_mut() {
            for neighbors in &mut other_node.neighbors {
                if let Ok(pos) = neighbors.binary_search(&id) {
                    neighbors.swap_remove(pos); // O(1) instead of O(M) shift
                    // Re-sort to maintain binary_search invariant
                    // For nearly-sorted arrays, this is very fast (adaptive algorithm)
                    neighbors.sort_unstable();
                }
            }
        }

        // Update entry point if needed
        if self.entry_point == Some(id) {
            self.update_entry_point();
        }

        Some(node)
    }

    /// Update entry point to highest layer node
    fn update_entry_point(&mut self) {
        self.entry_point = None;
        self.max_layer = 0;

        for (id, node) in &self.nodes {
            if node.max_layer > self.max_layer || self.entry_point.is_none() {
                self.entry_point = Some(*id);
                self.max_layer = node.max_layer;
            }
        }
    }

    /// Get all node IDs at a specific layer
    pub fn nodes_at_layer(&self, layer: u16) -> Vec<PointId> {
        self.nodes
            .values()
            .filter(|n| n.exists_at(layer))
            .map(|n| n.point_id)
            .collect()
    }

    /// Get count of nodes at each layer
    pub fn layer_counts(&self) -> Vec<usize> {
        if self.nodes.is_empty() {
            return vec![];
        }

        let mut counts = vec![0; self.max_layer as usize + 1];
        for node in self.nodes.values() {
            for layer in 0..=node.max_layer as usize {
                counts[layer] += 1;
            }
        }
        counts
    }

    /// Total number of nodes
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get all node IDs
    pub fn node_ids(&self) -> impl Iterator<Item = PointId> + '_ {
        self.nodes.keys().copied()
    }

    /// Check if a node exists
    pub fn contains(&self, id: PointId) -> bool {
        self.nodes.contains_key(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn test_node_creation() {
        let node = HnswNode::new(42, 3);
        assert_eq!(node.point_id, 42);
        assert_eq!(node.max_layer, 3);
        assert_eq!(node.neighbors.len(), 4); // layers 0, 1, 2, 3
    }

    #[test]
    fn test_node_neighbors() {
        let mut node = HnswNode::new(1, 2);

        node.add_neighbor(0, 10);
        node.add_neighbor(0, 20);
        node.add_neighbor(1, 30);

        assert_eq!(node.neighbors_at(0), &[10, 20]);
        assert_eq!(node.neighbors_at(1), &[30]);
        assert!(node.neighbors_at(2).is_empty());
    }

    #[test]
    fn test_node_no_duplicate_neighbors() {
        let mut node = HnswNode::new(1, 0);

        node.add_neighbor(0, 10);
        node.add_neighbor(0, 10); // Duplicate

        assert_eq!(node.neighbors_at(0).len(), 1);
    }

    #[test]
    fn test_node_remove_neighbor() {
        let mut node = HnswNode::new(1, 0);

        node.add_neighbor(0, 10);
        node.add_neighbor(0, 20);
        node.remove_neighbor(0, 10);

        assert_eq!(node.neighbors_at(0), &[20]);
    }

    #[test]
    fn test_layer_manager_empty() {
        let manager = LayerManager::new();
        assert!(manager.is_empty());
        assert_eq!(manager.entry_point(), None);
        assert_eq!(manager.max_layer(), 0);
    }

    #[test]
    fn test_layer_manager_insert() {
        let mut manager = LayerManager::new();

        let node1 = HnswNode::new(1, 0);
        let node2 = HnswNode::new(2, 2);
        let node3 = HnswNode::new(3, 1);

        manager.insert_node(node1);
        assert_eq!(manager.entry_point(), Some(1));
        assert_eq!(manager.max_layer(), 0);

        manager.insert_node(node2);
        assert_eq!(manager.entry_point(), Some(2)); // Higher layer
        assert_eq!(manager.max_layer(), 2);

        manager.insert_node(node3);
        assert_eq!(manager.entry_point(), Some(2)); // Still node 2
        assert_eq!(manager.len(), 3);
    }

    #[test]
    fn test_layer_manager_remove() {
        let mut manager = LayerManager::new();

        let mut node1 = HnswNode::new(1, 2);
        let mut node2 = HnswNode::new(2, 0);

        node1.add_neighbor(0, 2);
        node2.add_neighbor(0, 1);

        manager.insert_node(node1);
        manager.insert_node(node2);

        // Remove entry point
        manager.remove_node(1);

        assert_eq!(manager.entry_point(), Some(2));
        assert_eq!(manager.max_layer(), 0);
        assert!(manager.get_node(2).unwrap().neighbors_at(0).is_empty());
    }

    #[test]
    fn test_layer_counts() {
        let mut manager = LayerManager::new();

        manager.insert_node(HnswNode::new(1, 0)); // In layer 0
        manager.insert_node(HnswNode::new(2, 1)); // In layers 0, 1
        manager.insert_node(HnswNode::new(3, 2)); // In layers 0, 1, 2

        let counts = manager.layer_counts();
        assert_eq!(counts, vec![3, 2, 1]); // 3 in L0, 2 in L1, 1 in L2
    }

    #[test]
    fn test_nodes_at_layer() {
        let mut manager = LayerManager::new();

        manager.insert_node(HnswNode::new(1, 0));
        manager.insert_node(HnswNode::new(2, 1));
        manager.insert_node(HnswNode::new(3, 1));

        let l0_nodes = manager.nodes_at_layer(0);
        assert_eq!(l0_nodes.len(), 3);

        let l1_nodes = manager.nodes_at_layer(1);
        assert_eq!(l1_nodes.len(), 2);
        assert!(l1_nodes.contains(&2));
        assert!(l1_nodes.contains(&3));
    }
}
