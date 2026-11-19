//! The canonical form a snapshot is written in, so equal collections export
//! equal bytes.
//!
//! Relation ids are assigned in the order relations are first used, which is
//! an accident of history rather than content. A snapshot renumbers them in
//! relation-name order (rewriting every edge to match), so two collections
//! holding the same named edges write the same bytes. Loading needs nothing
//! special: the snapshot's configuration and edges agree on the new ids.

use crate::types::collection::CollectionConfig;
use crate::types::point::Point;
use rustc_hash::FxHashMap;

/// Old relation id -> canonical relation id.
pub(super) struct RelationRemap(FxHashMap<u16, u16>);

/// `config` with its relations renumbered `0..n` in ascending name order,
/// and the mapping applied to get there.
pub(super) fn canonical_relations(config: &CollectionConfig) -> (CollectionConfig, RelationRemap) {
    let mut by_name: Vec<(&String, u16)> =
        config.relations.iter().map(|(n, &id)| (n, id)).collect();
    by_name.sort_unstable();
    let mut canonical = config.clone();
    canonical.relations = by_name
        .iter()
        .enumerate()
        .map(|(new, (name, _))| ((*name).clone(), new as u16))
        .collect();
    let remap = by_name
        .iter()
        .enumerate()
        .map(|(new, (_, old))| (*old, new as u16))
        .collect();
    (canonical, RelationRemap(remap))
}

/// The point as a snapshot stores it: no vector (vectors are written once,
/// in their own section) and edges on canonical relation ids. An edge whose
/// relation is not in the configuration keeps its id.
pub(super) fn stored_point(point: &Point, remap: &RelationRemap) -> Point {
    let outgoing_edges = point.outgoing_edges.as_ref().map(|edges| {
        edges
            .iter()
            .map(|edge| {
                let mut edge = edge.clone();
                if let Some(&id) = remap.0.get(&edge.relation_id) {
                    edge.relation_id = id;
                }
                edge
            })
            .collect()
    });
    Point {
        id: point.id,
        vector: Vec::new(),
        payload: point.payload.clone(),
        outgoing_edges,
        label_bitmap: point.label_bitmap,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::collection::{Distance, HnswConfig, VectorConfig};
    use crate::types::point::Edge;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn relations_are_renumbered_by_name_and_edges_follow() {
        let mut config = CollectionConfig::new(
            "c",
            VectorConfig::new(2, Distance::Cosine),
            HnswConfig {
                m: 4,
                m0: 8,
                ml: HnswConfig::recommended_ml(4),
                ef: 16,
                ef_construction: 16,
            },
        );
        config.relations.insert("zeta".into(), 0);
        config.relations.insert("alpha".into(), 1);
        let (canonical, remap) = canonical_relations(&config);
        assert_eq!(canonical.relation_id("alpha"), Some(0));
        assert_eq!(canonical.relation_id("zeta"), Some(1));

        let mut point = Point::new_vector(1, vec![1.0, 0.0]);
        point.outgoing_edges = Some(
            [
                Edge::new(2, 0.5, 0),
                Edge::new(3, 0.5, 1),
                Edge::new(4, 0.5, 9),
            ]
            .into_iter()
            .collect(),
        );
        let stored = stored_point(&point, &remap);
        assert!(stored.vector.is_empty());
        let ids: Vec<(u64, u16)> = stored
            .outgoing_edges
            .iter()
            .flatten()
            .map(|e| (e.target_id, e.relation_id))
            .collect();
        assert_eq!(ids, vec![(2, 1), (3, 0), (4, 9)]);
    }
}
