//! In-memory storage implementation
//!
//! `MemStorage` provides a simple HashMap-based storage backend.
//! Perfect for testing and ephemeral use cases.

use async_trait::async_trait;
use lattice_core::{LatticeStorage, Page, StorageError, StorageResult};
use std::collections::HashMap;
use std::sync::RwLock;

/// In-memory storage backend
///
/// Uses `RwLock<HashMap>` for thread-safe access.
/// All data is lost when the instance is dropped.
///
/// # Usage
///
/// ```ignore
/// use lattice_storage::MemStorage;
/// use lattice_core::LatticeStorage;
///
/// let storage = MemStorage::new();
/// storage.write_page(0, b"hello").await.unwrap();
/// let page = storage.read_page(0).await.unwrap();
/// assert_eq!(page, b"hello");
/// ```
#[derive(Debug, Default)]
pub struct MemStorage {
    pages: RwLock<HashMap<u64, Vec<u8>>>,
    meta: RwLock<HashMap<String, Vec<u8>>>,
}

impl MemStorage {
    /// Create a new in-memory storage
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the number of stored pages
    pub fn page_count(&self) -> usize {
        self.pages.read().unwrap().len()
    }

    /// Get the number of stored metadata keys
    pub fn meta_count(&self) -> usize {
        self.meta.read().unwrap().len()
    }

    /// Clear all data
    pub fn clear(&self) {
        self.pages.write().unwrap().clear();
        self.meta.write().unwrap().clear();
    }

    /// Get total bytes stored (pages only)
    pub fn total_bytes(&self) -> usize {
        self.pages.read().unwrap().values().map(|v| v.len()).sum()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl LatticeStorage for MemStorage {
    async fn get_meta(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        let meta = self.meta.read().map_err(|e| StorageError::Io {
            message: format!("Lock poisoned: {}", e),
        })?;
        Ok(meta.get(key).cloned())
    }

    async fn set_meta(&self, key: &str, value: &[u8]) -> StorageResult<()> {
        let mut meta = self.meta.write().map_err(|e| StorageError::Io {
            message: format!("Lock poisoned: {}", e),
        })?;
        meta.insert(key.to_string(), value.to_vec());
        Ok(())
    }

    async fn delete_meta(&self, key: &str) -> StorageResult<()> {
        let mut meta = self.meta.write().map_err(|e| StorageError::Io {
            message: format!("Lock poisoned: {}", e),
        })?;
        meta.remove(key);
        Ok(())
    }

    async fn read_page(&self, page_id: u64) -> StorageResult<Page> {
        let pages = self.pages.read().map_err(|e| StorageError::Io {
            message: format!("Lock poisoned: {}", e),
        })?;

        pages
            .get(&page_id)
            .cloned()
            .ok_or(StorageError::PageNotFound { page_id })
    }

    async fn write_page(&self, page_id: u64, data: &[u8]) -> StorageResult<()> {
        let mut pages = self.pages.write().map_err(|e| StorageError::Io {
            message: format!("Lock poisoned: {}", e),
        })?;
        pages.insert(page_id, data.to_vec());
        Ok(())
    }

    async fn page_exists(&self, page_id: u64) -> StorageResult<bool> {
        let pages = self.pages.read().map_err(|e| StorageError::Io {
            message: format!("Lock poisoned: {}", e),
        })?;
        Ok(pages.contains_key(&page_id))
    }

    async fn delete_page(&self, page_id: u64) -> StorageResult<()> {
        let mut pages = self.pages.write().map_err(|e| StorageError::Io {
            message: format!("Lock poisoned: {}", e),
        })?;
        pages.remove(&page_id);
        Ok(())
    }

    async fn sync(&self) -> StorageResult<()> {
        // No-op for in-memory storage
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Async test macro for both native (tokio) and WASM (wasm_bindgen_test)
    macro_rules! async_test {
        ($name:ident, $body:expr) => {
            #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
            async fn $name() {
                $body
            }
        };
    }

    async_test!(test_new_storage_is_empty, {
        let storage = MemStorage::new();
        assert_eq!(storage.page_count(), 0);
        assert_eq!(storage.meta_count(), 0);
    });

    async_test!(test_write_then_read_page, {
        let storage = MemStorage::new();

        storage.write_page(1, b"hello world").await.unwrap();
        let page = storage.read_page(1).await.unwrap();

        assert_eq!(page, b"hello world");
    });

    async_test!(test_page_not_found_error, {
        let storage = MemStorage::new();

        let result = storage.read_page(999).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            StorageError::PageNotFound { page_id } => {
                assert_eq!(page_id, 999);
            }
            e => panic!("Expected PageNotFound, got {:?}", e),
        }
    });

    async_test!(test_page_exists, {
        let storage = MemStorage::new();

        assert!(!storage.page_exists(1).await.unwrap());

        storage.write_page(1, b"data").await.unwrap();

        assert!(storage.page_exists(1).await.unwrap());
    });

    async_test!(test_delete_page, {
        let storage = MemStorage::new();

        storage.write_page(1, b"data").await.unwrap();
        assert!(storage.page_exists(1).await.unwrap());

        storage.delete_page(1).await.unwrap();
        assert!(!storage.page_exists(1).await.unwrap());
    });

    async_test!(test_delete_nonexistent_page_ok, {
        let storage = MemStorage::new();

        // Should not error
        storage.delete_page(999).await.unwrap();
    });

    async_test!(test_overwrite_page, {
        let storage = MemStorage::new();

        storage.write_page(1, b"first").await.unwrap();
        storage.write_page(1, b"second").await.unwrap();

        let page = storage.read_page(1).await.unwrap();
        assert_eq!(page, b"second");
    });

    async_test!(test_meta_crud, {
        let storage = MemStorage::new();

        // Initially empty
        assert!(storage.get_meta("key").await.unwrap().is_none());

        // Set
        storage.set_meta("key", b"value").await.unwrap();
        assert_eq!(
            storage.get_meta("key").await.unwrap(),
            Some(b"value".to_vec())
        );

        // Overwrite
        storage.set_meta("key", b"new_value").await.unwrap();
        assert_eq!(
            storage.get_meta("key").await.unwrap(),
            Some(b"new_value".to_vec())
        );

        // Delete
        storage.delete_meta("key").await.unwrap();
        assert!(storage.get_meta("key").await.unwrap().is_none());
    });

    async_test!(test_delete_nonexistent_meta_ok, {
        let storage = MemStorage::new();

        // Should not error
        storage.delete_meta("nonexistent").await.unwrap();
    });

    async_test!(test_clear, {
        let storage = MemStorage::new();

        storage.write_page(1, b"page").await.unwrap();
        storage.set_meta("key", b"meta").await.unwrap();

        storage.clear();

        assert_eq!(storage.page_count(), 0);
        assert_eq!(storage.meta_count(), 0);
    });

    async_test!(test_total_bytes, {
        let storage = MemStorage::new();

        storage.write_page(1, b"hello").await.unwrap(); // 5 bytes
        storage.write_page(2, b"world!").await.unwrap(); // 6 bytes

        assert_eq!(storage.total_bytes(), 11);
    });

    async_test!(test_sync_is_noop, {
        let storage = MemStorage::new();

        // Should not error
        storage.sync().await.unwrap();
    });

    async_test!(test_multiple_pages, {
        let storage = MemStorage::new();

        for i in 0..100 {
            storage
                .write_page(i, format!("page_{}", i).as_bytes())
                .await
                .unwrap();
        }

        assert_eq!(storage.page_count(), 100);

        for i in 0..100 {
            let page = storage.read_page(i).await.unwrap();
            assert_eq!(page, format!("page_{}", i).as_bytes());
        }
    });

    async_test!(test_binary_data, {
        let storage = MemStorage::new();

        // Binary data with all byte values
        let data: Vec<u8> = (0..=255).collect();
        storage.write_page(1, &data).await.unwrap();

        let page = storage.read_page(1).await.unwrap();
        assert_eq!(page, data);
    });

    async_test!(test_empty_page, {
        let storage = MemStorage::new();

        storage.write_page(1, b"").await.unwrap();
        let page = storage.read_page(1).await.unwrap();

        assert!(page.is_empty());
        assert!(storage.page_exists(1).await.unwrap());
    });
}
