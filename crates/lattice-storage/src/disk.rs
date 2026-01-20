//! Disk-based storage implementation (stub)
//!
//! TODO: Implement in Phase 2/3

use async_trait::async_trait;
use lattice_core::{LatticeStorage, Page, StorageError, StorageResult};
use std::path::PathBuf;

/// Disk-based storage backend
///
/// Uses tokio::fs for async file operations.
/// Data is persisted to disk and survives process restarts.
pub struct DiskStorage {
    #[allow(dead_code)]
    path: PathBuf,
    #[allow(dead_code)]
    page_size: usize,
}

impl DiskStorage {
    /// Create a new disk storage at the given path
    ///
    /// # Arguments
    /// * `path` - Directory to store data files
    /// * `page_size` - Size of each page in bytes (PCND: required)
    pub fn new(path: PathBuf, page_size: usize) -> Self {
        Self { path, page_size }
    }
}

#[async_trait]
impl LatticeStorage for DiskStorage {
    async fn get_meta(&self, _key: &str) -> StorageResult<Option<Vec<u8>>> {
        todo!("DiskStorage::get_meta - implement in Phase 2")
    }

    async fn set_meta(&self, _key: &str, _value: &[u8]) -> StorageResult<()> {
        todo!("DiskStorage::set_meta - implement in Phase 2")
    }

    async fn delete_meta(&self, _key: &str) -> StorageResult<()> {
        todo!("DiskStorage::delete_meta - implement in Phase 2")
    }

    async fn read_page(&self, page_id: u64) -> StorageResult<Page> {
        Err(StorageError::PageNotFound { page_id })
    }

    async fn write_page(&self, _page_id: u64, _data: &[u8]) -> StorageResult<()> {
        todo!("DiskStorage::write_page - implement in Phase 2")
    }

    async fn page_exists(&self, _page_id: u64) -> StorageResult<bool> {
        todo!("DiskStorage::page_exists - implement in Phase 2")
    }

    async fn delete_page(&self, _page_id: u64) -> StorageResult<()> {
        todo!("DiskStorage::delete_page - implement in Phase 2")
    }

    async fn sync(&self) -> StorageResult<()> {
        todo!("DiskStorage::sync - implement in Phase 2")
    }
}
