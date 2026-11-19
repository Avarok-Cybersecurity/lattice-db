//! Durable collection engine for wasm32 (e.g. a Cloudflare Durable Object)
//!
//! Wraps the in-memory wasm [`CollectionEngine`] with the same write-ahead
//! log the native durable engine uses, over any [`LatticeStorage`]. There is
//! no background task, thread, runtime or clock: every write commits
//! synchronously inside the call that makes it.
//!
//! # Durability
//!
//! - Each `*_async` write is appended to the WAL and synced before it is
//!   applied in memory; the call returns `Ok` only once that sync succeeded.
//!   A failed sync leaves neither the WAL nor memory changed.
//! - [`DurableEngine::checkpoint`] writes a snapshot to data storage and
//!   records it in the WAL, so recovery loads the snapshot and replays only
//!   what came after. Checkpoints are taken only when the caller asks (for
//!   example from an alarm); [`DurableEngine::should_checkpoint`] says when
//!   the log has grown past the threshold.
//! - Reads go through `Deref` to the in-memory engine. Mutations are only
//!   available through this type, so none can bypass the log.

mod writes;

use super::wasm::CollectionEngine;
use crate::error::{ConfigError, LatticeError, LatticeResult};
use crate::storage::LatticeStorage;
use crate::transaction::TransactionManager;
use crate::types::collection::{CollectionConfig, DurabilityMode};
use crate::wal::replay::{replay, WalApply};
use crate::wal::snapshot::{read_snapshot, write_snapshot};
use crate::wal::Lsn;
use std::ops::Deref;

/// WAL-backed collection engine for wasm32
///
/// - `W`: storage for the WAL (pages below `PAGE_POINTS_START`)
/// - `D`: storage for checkpoint snapshots (pages from `PAGE_POINTS_START`);
///   the same storage as `W` is fine, see [`DurableEngine::open_durable_unified`]
pub struct DurableEngine<W: LatticeStorage, D: LatticeStorage = W> {
    engine: CollectionEngine,
    txn: TransactionManager<W>,
    data: D,
}

impl<W: LatticeStorage, D: LatticeStorage> DurableEngine<W, D> {
    /// Open a durable collection, recovering whatever the storage holds
    ///
    /// Loads the latest checkpoint snapshot from `data_storage`, if any, then
    /// replays the WAL from that snapshot's LSN.
    pub async fn open_durable(
        config: CollectionConfig,
        wal_storage: W,
        data_storage: D,
    ) -> LatticeResult<Self> {
        if config.durability != DurabilityMode::Durable {
            return Err(LatticeError::Config(ConfigError::InvalidParameter {
                name: "durability",
                message: "Expected DurabilityMode::Durable for open_durable".to_string(),
            }));
        }
        config.validate()?;

        let txn = TransactionManager::open(wal_storage).await?;
        let wal_checkpoint = txn.last_checkpoint_lsn();
        let (mut engine, start_lsn) = match read_snapshot(&data_storage).await? {
            Some(snapshot) if snapshot.lsn >= wal_checkpoint => (
                engine_from_snapshot(&config, &snapshot.bytes)?,
                snapshot.lsn,
            ),
            Some(snapshot) => {
                return Err(missing_snapshot(format!(
                    "snapshot is at LSN {} but the WAL is checkpointed at {}",
                    snapshot.lsn, wal_checkpoint
                )))
            }
            None if wal_checkpoint > 0 => {
                return Err(missing_snapshot(format!(
                    "the WAL is checkpointed at LSN {} but data storage has no snapshot",
                    wal_checkpoint
                )))
            }
            None => (CollectionEngine::new(config)?, 0),
        };

        let entries = txn.read_entries_from(start_lsn).await?;
        replay(&mut engine, entries)?;

        Ok(Self {
            engine,
            txn,
            data: data_storage,
        })
    }

    /// Snapshot the collection and record the checkpoint in the WAL
    ///
    /// Returns the checkpoint LSN. If it fails, nothing acknowledged is lost:
    /// recovery uses the previous snapshot, or the new one if its manifest
    /// landed, and replays the WAL from whichever it loads.
    pub async fn checkpoint(&mut self) -> LatticeResult<Lsn> {
        let lsn = self.txn.next_lsn();
        let bytes = self.engine.to_bytes()?;
        write_snapshot(&self.data, lsn, &bytes).await?;
        self.txn.checkpoint(lsn).await?;
        self.txn.truncate_before(lsn).await?;
        Ok(lsn)
    }

    /// Whether the WAL has grown enough since the last checkpoint to take one
    pub fn should_checkpoint(&self) -> bool {
        self.txn.should_checkpoint()
    }

    /// The in-memory engine, for reads
    pub fn engine(&self) -> &CollectionEngine {
        &self.engine
    }
}

impl<S: LatticeStorage + Clone> DurableEngine<S, S> {
    /// Open a durable collection whose WAL and snapshots share one storage
    pub async fn open_durable_unified(config: CollectionConfig, storage: S) -> LatticeResult<Self> {
        Self::open_durable(config, storage.clone(), storage).await
    }
}

impl<W: LatticeStorage, D: LatticeStorage> Deref for DurableEngine<W, D> {
    type Target = CollectionEngine;

    fn deref(&self) -> &CollectionEngine {
        &self.engine
    }
}

/// Rebuild the engine from a snapshot, refusing one of another collection
///
/// The snapshot's config is used, since it carries the relations registered
/// up to the checkpoint; relations named only in `config` are added to it.
fn engine_from_snapshot(
    config: &CollectionConfig,
    bytes: &[u8],
) -> LatticeResult<CollectionEngine> {
    let mut engine = CollectionEngine::from_bytes(bytes)?;
    let stored = engine.config();
    if stored.name != config.name || stored.vector_dim() != config.vector_dim() {
        return Err(LatticeError::InvalidOperation {
            message: format!(
                "storage holds collection '{}' ({} dims), not '{}' ({} dims)",
                stored.name,
                stored.vector_dim(),
                config.name,
                config.vector_dim()
            ),
        });
    }
    for (name, id) in &config.relations {
        engine.apply_define_relation(name, *id)?;
    }
    Ok(engine)
}

fn missing_snapshot(context: String) -> LatticeError {
    LatticeError::DataCorruption {
        code: 60030,
        context,
    }
}
