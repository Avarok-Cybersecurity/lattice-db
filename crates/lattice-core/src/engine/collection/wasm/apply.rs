//! Replay target for the wasm engine: WAL entries map onto its unlogged
//! mutations.

use super::CollectionEngine;
use crate::engine::collection::UpsertResult;
use crate::error::LatticeResult;
use crate::types::point::{Edge, Point, PointId};
use crate::wal::relations::define_relation;
use crate::wal::replay::WalApply;

impl WalApply for CollectionEngine {
    fn apply_upsert(&mut self, points: Vec<Point>) -> LatticeResult<UpsertResult> {
        self.upsert_points(points)
    }

    fn apply_delete(&mut self, ids: &[PointId]) -> LatticeResult<usize> {
        self.delete_points(ids)
    }

    fn apply_add_edge(&mut self, from_id: PointId, edge: Edge) -> LatticeResult<()> {
        self.insert_edge(from_id, edge)
    }

    fn apply_remove_edge(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation_id: u16,
    ) -> LatticeResult<bool> {
        self.remove_edge_by_id(from_id, to_id, relation_id)
    }

    fn apply_define_relation(&mut self, name: &str, relation_id: u16) -> LatticeResult<()> {
        define_relation(&mut self.config, name, relation_id)
    }
}
