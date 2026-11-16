//! Replay target for the native engine: WAL entries map onto the unlogged
//! `apply_*_internal` mutations.

use super::CollectionEngine;
use crate::engine::collection::UpsertResult;
use crate::error::LatticeResult;
use crate::storage::LatticeStorage;
use crate::types::point::{Edge, Point, PointId};
use crate::wal::relations::define_relation;
use crate::wal::replay::WalApply;

impl<W: LatticeStorage, D: LatticeStorage> WalApply for CollectionEngine<W, D> {
    fn apply_upsert(&mut self, points: Vec<Point>) -> LatticeResult<UpsertResult> {
        self.apply_upsert_internal(points)
    }

    fn apply_delete(&mut self, ids: &[PointId]) -> LatticeResult<usize> {
        self.apply_delete_internal(ids)
    }

    fn apply_add_edge(&mut self, from_id: PointId, edge: Edge) -> LatticeResult<()> {
        self.apply_add_edge_internal(from_id, edge)
    }

    fn apply_remove_edge(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation_id: u16,
    ) -> LatticeResult<bool> {
        self.apply_remove_edge_internal(from_id, to_id, relation_id)
    }

    fn apply_define_relation(&mut self, name: &str, relation_id: u16) -> LatticeResult<()> {
        define_relation(&mut self.config, name, relation_id)
    }
}
