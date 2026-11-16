//! WAL entry types and the log sequence number.

use crate::types::point::{Edge, Point, PointId};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

/// Log Sequence Number - monotonically increasing identifier for each WAL entry
pub type Lsn = u64;

/// WAL entry types
///
/// Each entry represents an atomic operation that can be replayed during recovery.
#[derive(Archive, RkyvDeserialize, RkyvSerialize, Debug, Clone)]
pub enum WalEntry {
    /// Insert or update points
    Upsert {
        /// Points to upsert (can be single or batch)
        points: Vec<Point>,
    },

    /// Delete points by ID
    Delete {
        /// Point IDs to delete
        ids: Vec<PointId>,
    },

    /// Add a graph edge
    AddEdge {
        /// Source point ID
        from_id: PointId,
        /// Edge to add
        edge: Edge,
    },

    /// Remove a graph edge
    RemoveEdge {
        /// Source point ID
        from_id: PointId,
        /// Target point ID
        to_id: PointId,
        /// Relation ID (from collection config)
        relation_id: u16,
    },

    /// Abort marker
    ///
    /// Indicates that the operation at the given LSN failed to apply
    /// and should be skipped during recovery replay.
    Abort {
        /// LSN of the failed operation
        aborted_lsn: Lsn,
    },

    /// Checkpoint marker
    ///
    /// Indicates that a snapshot was taken at this LSN.
    /// WAL entries before this can be safely truncated.
    Checkpoint {
        /// LSN at which snapshot was taken
        snapshot_lsn: Lsn,
    },

    /// Registers a relation name under the id that edges store.
    ///
    /// Logged, in the same commit, before the first edge that uses a relation
    /// the collection config did not already know, so replay can name the
    /// edge. Appended last: earlier variants keep their tags and the archived
    /// size is unchanged, so logs written before it still read.
    DefineRelation {
        /// Relation name
        name: String,
        /// Id stored in `Edge::relation_id`
        relation_id: u16,
    },
}
