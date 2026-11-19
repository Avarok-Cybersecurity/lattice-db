//! Logged mutations: commit to the WAL, then apply in memory.

use super::DurableEngine;
use crate::engine::collection::UpsertResult;
use crate::error::{LatticeError, LatticeResult};
use crate::storage::LatticeStorage;
use crate::types::point::{Edge, Point, PointId};
use crate::wal::relations::plan_relation;
use crate::wal::replay::WalApply;
use crate::wal::{Lsn, WalEntry};

impl<W: LatticeStorage, D: LatticeStorage> DurableEngine<W, D> {
    /// Upsert points; durable once this returns `Ok`
    pub async fn upsert_points_async(&mut self, points: Vec<Point>) -> LatticeResult<UpsertResult> {
        self.check_dimensions(&points)?;
        let entry = WalEntry::Upsert {
            points: points.clone(),
        };
        let lsns = self.txn.log_entries(&[entry]).await?;
        let applied = self.engine.apply_upsert(points);
        self.settle(applied, &lsns).await
    }

    /// Upsert several batches as one commit (one sync), one WAL entry each
    pub async fn upsert_batch_async(
        &mut self,
        points_batch: Vec<Vec<Point>>,
    ) -> LatticeResult<Vec<UpsertResult>> {
        for points in &points_batch {
            self.check_dimensions(points)?;
        }
        let entries: Vec<WalEntry> = points_batch
            .iter()
            .map(|points| WalEntry::Upsert {
                points: points.clone(),
            })
            .collect();
        let lsns = self.txn.log_entries(&entries).await?;
        let mut results = Vec::with_capacity(points_batch.len());
        for points in points_batch {
            match self.engine.apply_upsert(points) {
                Ok(result) => results.push(result),
                Err(e) => return self.settle(Err(e), &lsns[results.len()..]).await,
            }
        }
        Ok(results)
    }

    /// Delete points, returning how many existed; durable once this returns `Ok`
    pub async fn delete_points_async(&mut self, ids: &[PointId]) -> LatticeResult<usize> {
        let lsns = self
            .txn
            .log_entries(&[WalEntry::Delete { ids: ids.to_vec() }])
            .await?;
        let applied = self.engine.apply_delete(ids);
        self.settle(applied, &lsns).await
    }

    /// Add a directed edge; durable once this returns `Ok`
    ///
    /// A relation the collection has not seen is registered in the same
    /// commit as the edge.
    pub async fn add_edge_async(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: &str,
        weight: f32,
    ) -> LatticeResult<()> {
        for id in [from_id, to_id] {
            if !self.engine.point_exists(id) {
                return Err(LatticeError::PointNotFound { id });
            }
        }

        let (relation_id, is_new) = plan_relation(self.engine.config(), relation)?;
        let edge = Edge::new(to_id, weight, relation_id);
        let mut entries = Vec::with_capacity(2);
        if is_new {
            entries.push(WalEntry::DefineRelation {
                name: relation.to_string(),
                relation_id,
            });
        }
        entries.push(WalEntry::AddEdge {
            from_id,
            edge: edge.clone(),
        });
        let lsns = self.txn.log_entries(&entries).await?;

        if is_new {
            // A registered relation is harmless if the edge later aborts, so
            // only the edge's LSN is ever aborted.
            let defined = self.engine.apply_define_relation(relation, relation_id);
            self.settle(defined, &[]).await?;
        }
        let applied = self.engine.apply_add_edge(from_id, edge);
        self.settle(applied, &lsns[lsns.len() - 1..]).await
    }

    /// Remove edges `from_id -> to_id`, returning whether any existed
    ///
    /// With a relation, removes that edge; without one, removes the edges of
    /// every relation (the wasm engine's semantics). Durable once this
    /// returns `Ok`; nothing is logged when there is nothing to remove.
    pub async fn remove_edge_async(
        &mut self,
        from_id: PointId,
        to_id: PointId,
        relation: Option<&str>,
    ) -> LatticeResult<bool> {
        let point = self
            .engine
            .get_point(from_id)?
            .ok_or(LatticeError::PointNotFound { id: from_id })?;
        let wanted = relation.map(|rel| self.engine.config().relation_id(rel));
        let relation_ids: Vec<u16> = point
            .outgoing_edges
            .iter()
            .flatten()
            .filter(|e| e.target_id == to_id)
            .map(|e| e.relation_id)
            .filter(|id| wanted.map_or(true, |w| w == Some(*id)))
            .collect();
        if relation_ids.is_empty() {
            return Ok(false);
        }

        let entries: Vec<WalEntry> = relation_ids
            .iter()
            .map(|&relation_id| WalEntry::RemoveEdge {
                from_id,
                to_id,
                relation_id,
            })
            .collect();
        let lsns = self.txn.log_entries(&entries).await?;
        for (i, relation_id) in relation_ids.into_iter().enumerate() {
            let applied = self.engine.apply_remove_edge(from_id, to_id, relation_id);
            self.settle(applied, &lsns[i..]).await?;
        }
        Ok(true)
    }

    fn check_dimensions(&self, points: &[Point]) -> LatticeResult<()> {
        let expected = self.engine.config().vector_dim();
        match points.iter().find(|p| p.vector.len() != expected) {
            Some(p) => Err(LatticeError::DimensionMismatch {
                expected,
                actual: p.vector.len(),
            }),
            None => Ok(()),
        }
    }

    /// Pass an in-memory apply result through, aborting `lsns` if it failed
    ///
    /// The entries are already durable, so a failed apply must be cancelled
    /// in the log too or replay would re-run it. If the abort itself cannot
    /// be logged, both failures are reported.
    async fn settle<T>(&mut self, applied: LatticeResult<T>, lsns: &[Lsn]) -> LatticeResult<T> {
        let error = match applied {
            Ok(value) => return Ok(value),
            Err(error) => error,
        };
        if lsns.is_empty() {
            return Err(error);
        }
        let aborts: Vec<WalEntry> = lsns
            .iter()
            .map(|&aborted_lsn| WalEntry::Abort { aborted_lsn })
            .collect();
        match self.txn.log_entries(&aborts).await {
            Ok(_) => Err(error),
            Err(abort_error) => Err(LatticeError::Internal {
                code: 50020,
                message: format!(
                    "apply failed ({}) and logging its abort failed ({}); \
                     reopening will retry the logged write",
                    error, abort_error
                ),
            }),
        }
    }
}
