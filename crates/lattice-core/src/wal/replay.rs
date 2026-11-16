//! WAL replay: the one place that turns logged entries back into state.
//!
//! Both the native and the wasm engines implement [`WalApply`]; recovery on
//! either platform goes through [`replay`].

use super::{Lsn, WalEntry};
use crate::engine::collection::UpsertResult;
use crate::error::LatticeResult;
use crate::types::point::{Edge, Point, PointId};
use std::collections::HashSet;

/// Unlogged mutations an engine applies, for replay and after a commit
pub trait WalApply {
    /// Insert or update points
    fn apply_upsert(&mut self, points: Vec<Point>) -> LatticeResult<UpsertResult>;
    /// Delete points, returning how many existed
    fn apply_delete(&mut self, ids: &[PointId]) -> LatticeResult<usize>;
    /// Add an edge whose relation id is already registered
    fn apply_add_edge(&mut self, from_id: PointId, edge: Edge) -> LatticeResult<()>;
    /// Remove the edge `from_id -> to_id` with `relation_id`, returning whether one existed
    fn apply_remove_edge(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation_id: u16,
    ) -> LatticeResult<bool>;
    /// Register a relation name under the id edges store
    fn apply_define_relation(&mut self, name: &str, relation_id: u16) -> LatticeResult<()>;
}

/// Apply logged entries in order, skipping any an `Abort` entry cancels
///
/// Returns the number of state-changing entries applied. Any apply error
/// stops replay and is returned: a log that cannot be replayed is not
/// silently half-loaded.
pub fn replay<A: WalApply + ?Sized>(
    target: &mut A,
    entries: Vec<(Lsn, WalEntry)>,
) -> LatticeResult<usize> {
    // Collect aborted LSNs first
    let aborted_lsns: HashSet<Lsn> = entries
        .iter()
        .filter_map(|(_, entry)| match entry {
            WalEntry::Abort { aborted_lsn } => Some(*aborted_lsn),
            _ => None,
        })
        .collect();

    let mut replayed = 0;
    for (lsn, entry) in entries {
        if aborted_lsns.contains(&lsn) {
            continue;
        }

        match entry {
            WalEntry::Upsert { points } => {
                target.apply_upsert(points)?;
            }
            WalEntry::Delete { ids } => {
                target.apply_delete(&ids)?;
            }
            WalEntry::AddEdge { from_id, edge } => {
                target.apply_add_edge(from_id, edge)?;
            }
            WalEntry::RemoveEdge {
                from_id,
                to_id,
                relation_id,
            } => {
                target.apply_remove_edge(from_id, to_id, relation_id)?;
            }
            WalEntry::DefineRelation { name, relation_id } => {
                target.apply_define_relation(&name, relation_id)?;
            }
            WalEntry::Checkpoint { .. } | WalEntry::Abort { .. } => continue,
        }
        replayed += 1;
    }

    Ok(replayed)
}
