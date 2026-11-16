//! Broken backends the suite must reject
//!
//! These exist to prove the checks discriminate: each models one defect a
//! real adapter could have, and the suite's own tests assert that the check
//! aimed at that defect fails against it.

use super::StorageFixture;
use lattice_core::storage::{Page, StorageResult};
use lattice_core::LatticeStorage;
use std::sync::Mutex;

/// Wraps a fixture so that each `sync` loses the page written last before it
///
/// This is a write-back cache that forgets its newest entry on flush: the
/// call reports success, but an acknowledged commit's page is not durable.
pub struct DropsLastPageOnSync<F>(pub F);

#[async_trait::async_trait(?Send)]
impl<F: StorageFixture> StorageFixture for DropsLastPageOnSync<F> {
    type Storage = DroppingStorage<F::Storage>;

    async fn open(&self) -> StorageResult<Self::Storage> {
        Ok(DroppingStorage {
            inner: self.0.open().await?,
            last_written: Mutex::new(None),
        })
    }
}

/// The storage handle of [`DropsLastPageOnSync`]
pub struct DroppingStorage<S> {
    inner: S,
    last_written: Mutex<Option<u64>>,
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl<S: LatticeStorage> LatticeStorage for DroppingStorage<S> {
    async fn get_meta(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        self.inner.get_meta(key).await
    }
    async fn set_meta(&self, key: &str, value: &[u8]) -> StorageResult<()> {
        self.inner.set_meta(key, value).await
    }
    async fn delete_meta(&self, key: &str) -> StorageResult<()> {
        self.inner.delete_meta(key).await
    }
    async fn read_page(&self, page_id: u64) -> StorageResult<Page> {
        self.inner.read_page(page_id).await
    }
    async fn write_page(&self, page_id: u64, data: &[u8]) -> StorageResult<()> {
        self.inner.write_page(page_id, data).await?;
        *self.last_written.lock().unwrap() = Some(page_id);
        Ok(())
    }
    async fn page_exists(&self, page_id: u64) -> StorageResult<bool> {
        self.inner.page_exists(page_id).await
    }
    async fn delete_page(&self, page_id: u64) -> StorageResult<()> {
        self.inner.delete_page(page_id).await
    }
    async fn sync(&self) -> StorageResult<()> {
        let dropped = self.last_written.lock().unwrap().take();
        if let Some(page_id) = dropped {
            self.inner.delete_page(page_id).await?;
        }
        self.inner.sync().await
    }
}
