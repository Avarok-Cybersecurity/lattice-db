//! Storage with configurable failure injection for ACID testing

use crate::shared_state::{MockStorage, SharedState};
use lattice_core::storage::{StorageError, StorageResult};
use lattice_core::LatticeStorage;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Storage that delegates to another backend but can inject failures
///
/// Set `*_fail_at` to a call count to trigger an I/O error on that call.
/// Set to 0 to disable failure (default). Use `clear_failures()` to reset.
///
/// An injected failure returns before the call reaches the wrapped backend,
/// so a failed call never takes effect.
pub struct FailingStorage<S = MockStorage> {
    inner: S,
    /// Fail write_page on this call number (0 = never)
    pub write_page_fail_at: Arc<AtomicU64>,
    /// Fail set_meta on this call number (0 = never)
    pub set_meta_fail_at: Arc<AtomicU64>,
    /// Fail sync on this call number (0 = never)
    pub sync_fail_at: Arc<AtomicU64>,
    write_page_count: Arc<AtomicU64>,
    set_meta_count: Arc<AtomicU64>,
    sync_count: Arc<AtomicU64>,
}

impl FailingStorage<MockStorage> {
    pub fn new(state: &SharedState) -> Self {
        Self::wrap(state.mock())
    }
}

impl<S> FailingStorage<S> {
    /// Inject failures into any backend
    pub fn wrap(inner: S) -> Self {
        Self {
            inner,
            write_page_fail_at: Arc::new(AtomicU64::new(0)),
            set_meta_fail_at: Arc::new(AtomicU64::new(0)),
            sync_fail_at: Arc::new(AtomicU64::new(0)),
            write_page_count: Arc::new(AtomicU64::new(0)),
            set_meta_count: Arc::new(AtomicU64::new(0)),
            sync_count: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn clear_failures(&self) {
        self.write_page_fail_at.store(0, Ordering::SeqCst);
        self.set_meta_fail_at.store(0, Ordering::SeqCst);
        self.sync_fail_at.store(0, Ordering::SeqCst);
    }

    fn should_fail(counter: &AtomicU64, fail_at: &AtomicU64) -> bool {
        let count = counter.fetch_add(1, Ordering::SeqCst) + 1;
        let target = fail_at.load(Ordering::SeqCst);
        target != 0 && count >= target
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl<S: LatticeStorage> LatticeStorage for FailingStorage<S> {
    async fn get_meta(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        self.inner.get_meta(key).await
    }

    async fn set_meta(&self, key: &str, value: &[u8]) -> StorageResult<()> {
        if Self::should_fail(&self.set_meta_count, &self.set_meta_fail_at) {
            return Err(StorageError::Io {
                message: "Injected set_meta failure".into(),
            });
        }
        self.inner.set_meta(key, value).await
    }

    async fn delete_meta(&self, key: &str) -> StorageResult<()> {
        self.inner.delete_meta(key).await
    }

    async fn read_page(&self, page_id: u64) -> StorageResult<Vec<u8>> {
        self.inner.read_page(page_id).await
    }

    async fn write_page(&self, page_id: u64, data: &[u8]) -> StorageResult<()> {
        if Self::should_fail(&self.write_page_count, &self.write_page_fail_at) {
            return Err(StorageError::Io {
                message: "Injected write_page failure".into(),
            });
        }
        self.inner.write_page(page_id, data).await
    }

    async fn page_exists(&self, page_id: u64) -> StorageResult<bool> {
        self.inner.page_exists(page_id).await
    }

    async fn delete_page(&self, page_id: u64) -> StorageResult<()> {
        self.inner.delete_page(page_id).await
    }

    async fn sync(&self) -> StorageResult<()> {
        if Self::should_fail(&self.sync_count, &self.sync_fail_at) {
            return Err(StorageError::Io {
                message: "Injected sync failure".into(),
            });
        }
        self.inner.sync().await
    }
}
