//! Builder pattern for CollectionEngine configuration
//!
//! Provides a fluent API for creating both ephemeral and durable engines.
//!
//! # Examples
//!
//! ```rust,ignore
//! // Ephemeral mode (fast, no persistence)
//! let engine = CollectionEngineBuilder::new(config).build()?;
//!
//! // Durable mode with unified storage
//! let engine = CollectionEngineBuilder::new(config)
//!     .with_storage(storage)
//!     .open()
//!     .await?;
//!
//! // Durable mode with separate WAL and data storage
//! let engine = CollectionEngineBuilder::new(config)
//!     .with_wal(wal_storage)
//!     .with_data(data_storage)
//!     .open()
//!     .await?;
//! ```

use super::native::{CollectionEngine, NoStorage};
use crate::error::LatticeResult;
use crate::storage::LatticeStorage;
use crate::types::collection::{CollectionConfig, DurabilityMode};

/// Builder for creating a CollectionEngine with optional durability
///
/// Start with `CollectionEngineBuilder::new(config)` and configure as needed.
pub struct CollectionEngineBuilder<W: LatticeStorage = NoStorage, D: LatticeStorage = NoStorage> {
    config: CollectionConfig,
    wal_storage: Option<W>,
    data_storage: Option<D>,
}

impl CollectionEngineBuilder<NoStorage, NoStorage> {
    /// Start building an engine with the given configuration
    ///
    /// By default, this creates an ephemeral engine. Use `with_storage()` or
    /// `with_wal().with_data()` to enable durability.
    pub fn new(config: CollectionConfig) -> Self {
        Self {
            config,
            wal_storage: None,
            data_storage: None,
        }
    }

    /// Build an ephemeral engine (no durability)
    ///
    /// This is the fast path with no WAL overhead. Data is lost on restart.
    pub fn build(self) -> LatticeResult<CollectionEngine<NoStorage, NoStorage>> {
        CollectionEngine::new(self.config)
    }

    /// Enable ACID mode with unified storage for both WAL and data
    ///
    /// Use this when you want a single storage backend for everything.
    /// The storage is cloned internally for WAL and data separation.
    pub fn with_storage<S: LatticeStorage + Clone + Send + Sync + 'static>(
        mut self,
        storage: S,
    ) -> CollectionEngineBuilder<S, S> {
        self.config.durability = DurabilityMode::Durable;
        CollectionEngineBuilder {
            config: self.config,
            wal_storage: Some(storage.clone()),
            data_storage: Some(storage),
        }
    }

    /// Set WAL storage backend (requires `with_data()` before building)
    ///
    /// Use this when you want separate storage backends for WAL and data.
    /// WAL storage should be optimized for fast sequential writes.
    pub fn with_wal<W: LatticeStorage + Send + Sync + 'static>(
        mut self,
        storage: W,
    ) -> CollectionEngineBuilder<W, NoStorage> {
        self.config.durability = DurabilityMode::Durable;
        CollectionEngineBuilder {
            config: self.config,
            wal_storage: Some(storage),
            data_storage: None,
        }
    }
}

impl<W: LatticeStorage + Send + Sync + 'static> CollectionEngineBuilder<W, NoStorage> {
    /// Set data/checkpoint storage backend
    ///
    /// This completes the configuration for separate WAL and data storage.
    /// Data storage is used for checkpoints and snapshots.
    pub fn with_data<D: LatticeStorage + Send + Sync + 'static>(
        self,
        storage: D,
    ) -> CollectionEngineBuilder<W, D> {
        CollectionEngineBuilder {
            config: self.config,
            wal_storage: self.wal_storage,
            data_storage: Some(storage),
        }
    }
}

impl<W, D> CollectionEngineBuilder<W, D>
where
    W: LatticeStorage + Send + Sync + 'static,
    D: LatticeStorage + Send + Sync + 'static,
{
    /// Open a durable engine with WAL recovery
    ///
    /// On startup, the engine replays the WAL from the last checkpoint
    /// to restore state after crashes.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let engine = CollectionEngineBuilder::new(config)
    ///     .with_storage(storage)
    ///     .open()
    ///     .await?;
    /// ```
    pub async fn open(self) -> LatticeResult<CollectionEngine<W, D>> {
        let wal = self
            .wal_storage
            .expect("WAL storage required for durable mode (use with_storage or with_wal)");
        let data = self
            .data_storage
            .expect("Data storage required for durable mode (use with_storage or with_data)");
        CollectionEngine::open_durable(self.config, wal, data).await
    }
}

impl CollectionConfig {
    /// Create a builder for this configuration
    ///
    /// This is a convenience method for the fluent API.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let engine = config.builder().build()?;
    /// ```
    pub fn builder(self) -> CollectionEngineBuilder<NoStorage, NoStorage> {
        CollectionEngineBuilder::new(self)
    }
}
