//! Graph traversal utilities
//!
//! Provides iterators for BFS and DFS traversal of the graph.

use crate::types::point::PointId;
use rustc_hash::FxHashSet;
use std::collections::VecDeque;

/// A path through the graph
#[derive(Debug, Clone)]
pub struct GraphPath {
    /// Nodes in the path (in order)
    pub nodes: Vec<PointId>,
    /// Total weight of the path
    pub total_weight: f32,
}

impl GraphPath {
    /// Create a new path starting from a single node
    pub fn new(start: PointId) -> Self {
        Self {
            nodes: vec![start],
            total_weight: 0.0,
        }
    }

    /// Extend the path with a new node
    pub fn extend(&self, node: PointId, edge_weight: f32) -> Self {
        let mut nodes = self.nodes.clone();
        nodes.push(node);
        Self {
            nodes,
            total_weight: self.total_weight + edge_weight,
        }
    }

    /// Get the length of the path (number of edges)
    pub fn len(&self) -> usize {
        self.nodes.len().saturating_sub(1)
    }

    /// Check if the path is empty (single node, no edges)
    pub fn is_empty(&self) -> bool {
        self.nodes.len() <= 1
    }

    /// Get the start node
    pub fn start(&self) -> Option<PointId> {
        self.nodes.first().copied()
    }

    /// Get the end node
    pub fn end(&self) -> Option<PointId> {
        self.nodes.last().copied()
    }
}

/// Breadth-first search iterator
///
/// Visits nodes level by level, finding shortest paths first.
pub struct BfsIterator<F> {
    queue: VecDeque<(PointId, usize)>, // (node, depth)
    visited: FxHashSet<PointId>,
    max_depth: usize,
    get_neighbors: F,
}

impl<F> BfsIterator<F>
where
    F: FnMut(PointId) -> Vec<PointId>,
{
    /// Create a new BFS iterator
    pub fn new(start: PointId, max_depth: usize, get_neighbors: F) -> Self {
        let mut queue = VecDeque::new();
        queue.push_back((start, 0));
        let mut visited = FxHashSet::default();
        visited.insert(start);

        Self {
            queue,
            visited,
            max_depth,
            get_neighbors,
        }
    }
}

impl<F> Iterator for BfsIterator<F>
where
    F: FnMut(PointId) -> Vec<PointId>,
{
    type Item = (PointId, usize); // (node_id, depth)

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, depth)) = self.queue.pop_front() {
            // Expand neighbors if within depth limit
            if depth < self.max_depth {
                let neighbors = (self.get_neighbors)(node);
                for neighbor in neighbors {
                    if self.visited.insert(neighbor) {
                        self.queue.push_back((neighbor, depth + 1));
                    }
                }
            }

            return Some((node, depth));
        }

        None
    }
}

/// Depth-first search iterator
///
/// Explores as far as possible along each branch before backtracking.
pub struct DfsIterator<F> {
    stack: Vec<(PointId, usize)>, // (node, depth)
    visited: FxHashSet<PointId>,
    max_depth: usize,
    get_neighbors: F,
}

impl<F> DfsIterator<F>
where
    F: FnMut(PointId) -> Vec<PointId>,
{
    /// Create a new DFS iterator
    pub fn new(start: PointId, max_depth: usize, get_neighbors: F) -> Self {
        let mut stack = Vec::new();
        stack.push((start, 0));
        let mut visited = FxHashSet::default();
        visited.insert(start);

        Self {
            stack,
            visited,
            max_depth,
            get_neighbors,
        }
    }
}

impl<F> Iterator for DfsIterator<F>
where
    F: FnMut(PointId) -> Vec<PointId>,
{
    type Item = (PointId, usize); // (node_id, depth)

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, depth)) = self.stack.pop() {
            // Expand neighbors if within depth limit
            if depth < self.max_depth {
                let neighbors = (self.get_neighbors)(node);
                for neighbor in neighbors.into_iter().rev() {
                    // Reverse to maintain natural order
                    if self.visited.insert(neighbor) {
                        self.stack.push((neighbor, depth + 1));
                    }
                }
            }

            return Some((node, depth));
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn simple_graph() -> impl FnMut(PointId) -> Vec<PointId> {
        // Graph: 1 -> [2, 3], 2 -> [4], 3 -> [4, 5], 4 -> [], 5 -> []
        move |node| match node {
            1 => vec![2, 3],
            2 => vec![4],
            3 => vec![4, 5],
            _ => vec![],
        }
    }

    #[test]
    fn test_graph_path() {
        let path = GraphPath::new(1);
        assert_eq!(path.len(), 0);
        assert!(path.is_empty());
        assert_eq!(path.start(), Some(1));
        assert_eq!(path.end(), Some(1));

        let path = path.extend(2, 0.5);
        assert_eq!(path.len(), 1);
        assert!(!path.is_empty());
        assert_eq!(path.nodes, vec![1, 2]);
        assert_eq!(path.total_weight, 0.5);

        let path = path.extend(3, 0.3);
        assert_eq!(path.len(), 2);
        assert_eq!(path.nodes, vec![1, 2, 3]);
        assert_eq!(path.total_weight, 0.8);
    }

    #[test]
    fn test_bfs_iterator() {
        let bfs: Vec<(PointId, usize)> = BfsIterator::new(1, 3, simple_graph()).collect();

        // BFS should visit level by level
        assert_eq!(bfs[0], (1, 0)); // Start
                                    // Level 1: 2, 3 (order may vary)
        assert!(bfs[1..3].contains(&(2, 1)));
        assert!(bfs[1..3].contains(&(3, 1)));
        // Level 2: 4, 5 (order may vary)
        assert!(bfs[3..5].contains(&(4, 2)));
        assert!(bfs[3..5].contains(&(5, 2)));
    }

    #[test]
    fn test_bfs_max_depth() {
        let bfs: Vec<(PointId, usize)> = BfsIterator::new(1, 1, simple_graph()).collect();

        // Should only visit start and depth 1
        assert_eq!(bfs.len(), 3); // 1, 2, 3
        assert!(bfs.iter().all(|(_, d)| *d <= 1));
    }

    #[test]
    fn test_dfs_iterator() {
        let dfs: Vec<(PointId, usize)> = DfsIterator::new(1, 3, simple_graph()).collect();

        // DFS should visit all reachable nodes
        assert_eq!(dfs.len(), 5);
        assert_eq!(dfs[0], (1, 0)); // Always starts with root

        // Verify all nodes visited
        let visited: HashSet<PointId> = dfs.iter().map(|(n, _)| *n).collect();
        assert!(visited.contains(&1));
        assert!(visited.contains(&2));
        assert!(visited.contains(&3));
        assert!(visited.contains(&4));
        assert!(visited.contains(&5));
    }

    #[test]
    fn test_dfs_max_depth() {
        let dfs: Vec<(PointId, usize)> = DfsIterator::new(1, 1, simple_graph()).collect();

        // Should only visit start and depth 1
        assert_eq!(dfs.len(), 3); // 1, 2, 3
        assert!(dfs.iter().all(|(_, d)| *d <= 1));
    }

    #[test]
    fn test_empty_graph() {
        let empty_graph = |_: PointId| vec![];
        let bfs: Vec<_> = BfsIterator::new(1, 10, empty_graph).collect();
        assert_eq!(bfs, vec![(1, 0)]);

        let dfs: Vec<_> = DfsIterator::new(1, 10, |_| vec![]).collect();
        assert_eq!(dfs, vec![(1, 0)]);
    }
}
