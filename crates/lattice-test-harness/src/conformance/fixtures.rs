//! Fixtures for backends in this workspace

use super::StorageFixture;
use crate::shared_state::{MockStorage, SharedState};
use lattice_core::storage::{Page, StorageResult};
use lattice_core::LatticeStorage;
use std::sync::Arc;

/// `SharedState` outlives each `MockStorage` it hands out, so it is its own
/// fixture
#[async_trait::async_trait(?Send)]
impl StorageFixture for SharedState {
    type Storage = MockStorage;

    async fn open(&self) -> StorageResult<MockStorage> {
        Ok(self.mock())
    }
}

/// A fixture for a backend that is its own durable state, such as
/// `MemStorage`: every `open` is a new handle onto the one instance
pub struct Reopen<S>(Arc<S>);

impl<S> Reopen<S> {
    /// Hand out handles onto `storage`
    pub fn new(storage: S) -> Self {
        Self(Arc::new(storage))
    }
}

#[async_trait::async_trait(?Send)]
impl<S: LatticeStorage> StorageFixture for Reopen<S> {
    type Storage = Handle<S>;

    async fn open(&self) -> StorageResult<Handle<S>> {
        Ok(Handle(self.0.clone()))
    }
}

/// One handle from a [`Reopen`] fixture
pub struct Handle<S>(Arc<S>);

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl<S: LatticeStorage> LatticeStorage for Handle<S> {
    async fn get_meta(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        self.0.get_meta(key).await
    }
    async fn set_meta(&self, key: &str, value: &[u8]) -> StorageResult<()> {
        self.0.set_meta(key, value).await
    }
    async fn delete_meta(&self, key: &str) -> StorageResult<()> {
        self.0.delete_meta(key).await
    }
    async fn read_page(&self, page_id: u64) -> StorageResult<Page> {
        self.0.read_page(page_id).await
    }
    async fn write_page(&self, page_id: u64, data: &[u8]) -> StorageResult<()> {
        self.0.write_page(page_id, data).await
    }
    async fn page_exists(&self, page_id: u64) -> StorageResult<bool> {
        self.0.page_exists(page_id).await
    }
    async fn delete_page(&self, page_id: u64) -> StorageResult<()> {
        self.0.delete_page(page_id).await
    }
    async fn sync(&self) -> StorageResult<()> {
        self.0.sync().await
    }
}
