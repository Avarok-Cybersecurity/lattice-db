//! Persisting an HNSW graph so a snapshot loads without re-inserting points.
//!
//! Rebuilding the graph costs `O(n · ef_construction · log n)` distance
//! computations (seconds for a few thousand 768-d vectors on wasm); restoring
//! it from [`HnswGraphParts`] is a linear copy. The parts come from untrusted
//! snapshot bytes, so [`HnswIndex::from_parts`] validates every reference
//! before building the index.

use super::HnswIndex;
use crate::error::{LatticeError, LatticeResult};
use crate::index::dense_vectors::DenseVectorStore;
use crate::index::distance::DistanceCalculator;
use crate::index::layer::{HnswNode, LayerManager};
use crate::types::collection::{Distance, HnswConfig};
use crate::types::point::PointId;
use rkyv::{Archive, Deserialize, Serialize};
use rustc_hash::FxHashSet;

/// One graph node: its id and sorted neighbour ids per layer (`0..=max_layer`).
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HnswNodeParts {
    pub id: PointId,
    pub layers: Vec<Vec<PointId>>,
}

/// The graph of an [`HnswIndex`], without its vectors (those travel once, in
/// the snapshot's vector section).
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HnswGraphParts {
    pub entry_point: Option<PointId>,
    pub max_layer: u16,
    /// Level-selection RNG state, so later inserts continue the same sequence.
    pub rng_state: u64,
    pub nodes: Vec<HnswNodeParts>,
}

impl HnswIndex {
    /// Capture the graph for persistence.
    pub fn to_parts(&self) -> HnswGraphParts {
        let mut nodes: Vec<HnswNodeParts> = self
            .layers
            .nodes()
            .map(|n| HnswNodeParts {
                id: n.point_id,
                layers: n.neighbors.clone(),
            })
            .collect();
        nodes.sort_unstable_by_key(|n| n.id);
        HnswGraphParts {
            entry_point: self.layers.entry_point(),
            max_layer: self.layers.max_layer(),
            rng_state: self.rng_state,
            nodes,
        }
    }

    /// Rebuild an index from persisted parts and the vectors of its points.
    ///
    /// Fails if the graph and the vectors disagree on which points exist, or
    /// if any neighbour, layer count or entry point is inconsistent.
    pub fn from_parts<'a, I>(
        config: HnswConfig,
        metric: Distance,
        dim: usize,
        parts: HnswGraphParts,
        vectors: I,
    ) -> LatticeResult<Self>
    where
        I: ExactSizeIterator<Item = (PointId, &'a [f32])>,
    {
        let ids: FxHashSet<PointId> = parts.nodes.iter().map(|n| n.id).collect();
        if ids.len() != parts.nodes.len() {
            return Err(corrupt("duplicate graph node"));
        }
        if vectors.len() != ids.len() {
            return Err(corrupt("graph and vectors hold different point counts"));
        }
        let mut store = DenseVectorStore::with_capacity(dim, vectors.len());
        for (id, vector) in vectors {
            if !ids.contains(&id) {
                return Err(corrupt("a point is missing from the graph"));
            }
            store
                .insert(id, vector.to_vec())
                .ok_or_else(|| corrupt("vector dimension does not match the collection"))?;
        }
        let mut nodes = Vec::with_capacity(parts.nodes.len());
        for node in parts.nodes {
            nodes.push(restore_node(node, &ids, parts.max_layer)?);
        }
        match parts.entry_point {
            None if nodes.is_empty() => {}
            Some(ep)
                if nodes
                    .iter()
                    .any(|n| n.point_id == ep && n.max_layer == parts.max_layer) => {}
            _ => return Err(corrupt("entry point is not a top-layer node")),
        }
        Ok(Self {
            config,
            distance: DistanceCalculator::new(metric),
            layers: LayerManager::restore(nodes, parts.entry_point, parts.max_layer),
            vectors: store,
            rng_state: parts.rng_state,
        })
    }
}

fn restore_node(
    node: HnswNodeParts,
    ids: &FxHashSet<PointId>,
    max_layer: u16,
) -> LatticeResult<HnswNode> {
    let levels = node.layers.len();
    if levels == 0 || levels > usize::from(max_layer) + 1 {
        return Err(corrupt("node layer count out of range"));
    }
    for layer in &node.layers {
        let sorted = layer.windows(2).all(|w| w[0] < w[1]);
        let known = layer.iter().all(|n| *n != node.id && ids.contains(n));
        if !sorted || !known {
            return Err(corrupt(
                "neighbour list is unsorted or names an unknown node",
            ));
        }
    }
    Ok(HnswNode {
        point_id: node.id,
        max_layer: (levels - 1) as u16,
        neighbors: node.layers,
    })
}

fn corrupt(message: &str) -> LatticeError {
    LatticeError::Serialization {
        message: format!("Invalid snapshot HNSW graph: {message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::point::Point;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn config() -> HnswConfig {
        HnswConfig {
            m: 8,
            m0: 16,
            ml: HnswConfig::recommended_ml(8),
            ef: 50,
            ef_construction: 50,
        }
    }

    fn built(n: u64) -> (HnswIndex, Vec<(PointId, Vec<f32>)>) {
        let mut index = HnswIndex::new(config(), Distance::Cosine);
        let vectors: Vec<(PointId, Vec<f32>)> = (0..n)
            .map(|i| {
                (
                    i,
                    vec![(i % 7) as f32 + 0.5, (i % 11) as f32, (i % 3) as f32 + 1.0],
                )
            })
            .collect();
        for (id, v) in &vectors {
            index.insert(&Point::new_vector(*id, v.clone())).unwrap();
        }
        (index, vectors)
    }

    fn restore(parts: HnswGraphParts, v: &[(PointId, Vec<f32>)]) -> LatticeResult<HnswIndex> {
        HnswIndex::from_parts(
            config(),
            Distance::Cosine,
            3,
            parts,
            v.iter().map(|(i, x)| (*i, x.as_slice())),
        )
    }

    #[test]
    fn restored_index_answers_exactly_like_the_original() {
        let (index, vectors) = built(300);
        let restored = restore(index.to_parts(), &vectors).unwrap();
        assert_eq!(restored.to_parts(), index.to_parts());
        for q in [[1.0, 2.0, 3.0], [6.5, 0.0, 1.0], [0.1, 9.0, 2.0]] {
            let a: Vec<PointId> = index.search(&q, 10, 50).iter().map(|r| r.id).collect();
            let b: Vec<PointId> = restored.search(&q, 10, 50).iter().map(|r| r.id).collect();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn inconsistent_parts_are_rejected() {
        let (index, vectors) = built(20);
        let mut dangling = index.to_parts();
        dangling.nodes[0].layers[0] = vec![999];
        assert!(restore(dangling, &vectors).is_err());
        let mut bad_entry = index.to_parts();
        bad_entry.entry_point = Some(12345);
        assert!(restore(bad_entry, &vectors).is_err());
        assert!(restore(index.to_parts(), &vectors[1..]).is_err());
        let empty = HnswIndex::new(config(), Distance::Cosine).to_parts();
        assert!(restore(empty, &[]).unwrap().is_empty());
    }
}
