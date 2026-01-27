//! Disk-based storage implementation
//!
//! Provides durable storage using the filesystem.
//! Used for ACID mode with WAL and crash recovery.

use async_trait::async_trait;
use lattice_core::{LatticeStorage, Page, StorageError, StorageResult};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

#[cfg(feature = "native")]
use tokio::fs::{self, File, OpenOptions};
#[cfg(feature = "native")]
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, SeekFrom};

/// Default page size (4KB, matching typical filesystem block size)
const DEFAULT_PAGE_SIZE: usize = 4096;

/// Disk-based storage backend
///
/// Uses tokio::fs for async file operations.
/// Data is persisted to disk and survives process restarts.
///
/// # File Layout
///
/// ```text
/// base_path/
/// ├── meta.json       # Metadata key-value store
/// ├── pages.dat       # Page data (offset = page_id * page_size)
/// └── wal.dat         # Write-ahead log (appended entries)
/// ```
pub struct DiskStorage {
    /// Base directory for all files
    path: PathBuf,
    /// Page size in bytes
    page_size: usize,
    /// In-memory cache of metadata (flushed on sync)
    #[cfg(feature = "native")]
    meta_cache: RwLock<HashMap<String, Vec<u8>>>,
    /// Flag indicating if metadata was modified
    #[cfg(feature = "native")]
    meta_dirty: RwLock<bool>,
}

impl DiskStorage {
    /// Create a new disk storage at the given path
    ///
    /// # Arguments
    /// * `path` - Directory to store data files
    /// * `page_size` - Size of each page in bytes (PCND: required)
    pub fn new(path: PathBuf, page_size: usize) -> Self {
        Self {
            path,
            page_size,
            #[cfg(feature = "native")]
            meta_cache: RwLock::new(HashMap::new()),
            #[cfg(feature = "native")]
            meta_dirty: RwLock::new(false),
        }
    }

    /// Create with default page size (4KB)
    pub fn with_defaults(path: PathBuf) -> Self {
        Self::new(path, DEFAULT_PAGE_SIZE)
    }

    /// Initialize storage directory and load existing metadata
    #[cfg(feature = "native")]
    pub async fn init(&self) -> StorageResult<()> {
        // Create directory if it doesn't exist
        fs::create_dir_all(&self.path)
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to create storage directory: {}", e),
            })?;

        // Load existing metadata if present
        let meta_path = self.path.join("meta.json");
        if meta_path.exists() {
            let content = fs::read_to_string(&meta_path)
                .await
                .map_err(|e| StorageError::Io {
                    message: format!("Failed to read metadata: {}", e),
                })?;

            let meta: HashMap<String, Vec<u8>> =
                serde_json::from_str(&content).map_err(|e| StorageError::Serialization {
                    message: format!("Failed to parse metadata: {}", e),
                })?;

            *self.meta_cache.write().unwrap() = meta;
        }

        Ok(())
    }

    /// Get the path to the pages file
    fn pages_path(&self) -> PathBuf {
        self.path.join("pages.dat")
    }

    /// Get the path to the metadata file
    fn meta_path(&self) -> PathBuf {
        self.path.join("meta.json")
    }
}

#[cfg(feature = "native")]
#[async_trait]
impl LatticeStorage for DiskStorage {
    async fn get_meta(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        Ok(self.meta_cache.read().unwrap().get(key).cloned())
    }

    async fn set_meta(&self, key: &str, value: &[u8]) -> StorageResult<()> {
        self.meta_cache
            .write()
            .unwrap()
            .insert(key.to_string(), value.to_vec());
        *self.meta_dirty.write().unwrap() = true;
        Ok(())
    }

    async fn delete_meta(&self, key: &str) -> StorageResult<()> {
        self.meta_cache.write().unwrap().remove(key);
        *self.meta_dirty.write().unwrap() = true;
        Ok(())
    }

    async fn read_page(&self, page_id: u64) -> StorageResult<Page> {
        let pages_path = self.pages_path();

        // If file doesn't exist, return empty page
        if !pages_path.exists() {
            return Err(StorageError::PageNotFound { page_id });
        }

        let mut file = File::open(&pages_path)
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to open pages file: {}", e),
            })?;

        // Calculate offset
        let offset = page_id * self.page_size as u64;

        // Check if page exists within file
        let file_len = file
            .metadata()
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to get file metadata: {}", e),
            })?
            .len();

        if offset >= file_len {
            return Err(StorageError::PageNotFound { page_id });
        }

        // Seek to page offset
        file.seek(SeekFrom::Start(offset))
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to seek to page: {}", e),
            })?;

        // Read page — expect exactly page_size bytes for a valid page
        let mut buffer = vec![0u8; self.page_size];
        let bytes_read = file.read(&mut buffer).await.map_err(|e| StorageError::Io {
            message: format!("Failed to read page: {}", e),
        })?;

        if bytes_read < self.page_size {
            // Short read indicates a truncated or corrupted page file.
            // Return the partial data rather than silently padding with zeros,
            // so callers (e.g. WAL recovery) can detect and handle corruption.
            buffer.truncate(bytes_read);
        }

        Ok(buffer)
    }

    async fn write_page(&self, page_id: u64, data: &[u8]) -> StorageResult<()> {
        let pages_path = self.pages_path();

        // Open file for reading and writing, create if doesn't exist
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&pages_path)
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to open pages file: {}", e),
            })?;

        // Calculate offset
        let offset = page_id * self.page_size as u64;

        // Seek to page offset
        file.seek(SeekFrom::Start(offset))
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to seek to page: {}", e),
            })?;

        // Write data (pad to page size if smaller)
        let mut padded = data.to_vec();
        if padded.len() < self.page_size {
            padded.resize(self.page_size, 0);
        }

        file.write_all(&padded[..self.page_size])
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to write page: {}", e),
            })?;

        Ok(())
    }

    async fn page_exists(&self, page_id: u64) -> StorageResult<bool> {
        let pages_path = self.pages_path();

        if !pages_path.exists() {
            return Ok(false);
        }

        let file = File::open(&pages_path)
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to open pages file: {}", e),
            })?;

        let offset = page_id * self.page_size as u64;
        let file_len = file
            .metadata()
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to get file metadata: {}", e),
            })?
            .len();

        Ok(offset < file_len)
    }

    async fn delete_page(&self, page_id: u64) -> StorageResult<()> {
        // Known limitation: zero-fills the page rather than releasing disk space.
        // The file will not shrink. A future compaction pass could reclaim space
        // by tracking free pages in a bitmap and rewriting the file.
        let zeros = vec![0u8; self.page_size];
        self.write_page(page_id, &zeros).await
    }

    async fn sync(&self) -> StorageResult<()> {
        // Sync metadata if dirty
        {
            let is_dirty = *self.meta_dirty.read().unwrap();
            if is_dirty {
                let meta_content = {
                    let cache = self.meta_cache.read().unwrap();
                    serde_json::to_string_pretty(&*cache).map_err(|e| {
                        StorageError::Serialization {
                            message: format!("Failed to serialize metadata: {}", e),
                        }
                    })?
                };

                fs::write(self.meta_path(), meta_content)
                    .await
                    .map_err(|e| StorageError::Io {
                        message: format!("Failed to write metadata: {}", e),
                    })?;

                *self.meta_dirty.write().unwrap() = false;
            }
        }

        // Sync pages file if it exists
        // Open with write access — Windows requires it for sync_all()
        let pages_path = self.pages_path();
        if pages_path.exists() {
            let file = OpenOptions::new()
                .write(true)
                .open(&pages_path)
                .await
                .map_err(|e| StorageError::Io {
                    message: format!("Failed to open pages file for sync: {}", e),
                })?;

            file.sync_all().await.map_err(|e| StorageError::Io {
                message: format!("Failed to sync pages file: {}", e),
            })?;
        }

        Ok(())
    }
}

// Stub implementation for non-native targets
#[cfg(not(feature = "native"))]
#[async_trait]
impl LatticeStorage for DiskStorage {
    async fn get_meta(&self, _key: &str) -> StorageResult<Option<Vec<u8>>> {
        Err(StorageError::NotImplemented {
            feature: "DiskStorage requires 'native' feature",
        })
    }

    async fn set_meta(&self, _key: &str, _value: &[u8]) -> StorageResult<()> {
        Err(StorageError::NotImplemented {
            feature: "DiskStorage requires 'native' feature",
        })
    }

    async fn delete_meta(&self, _key: &str) -> StorageResult<()> {
        Err(StorageError::NotImplemented {
            feature: "DiskStorage requires 'native' feature",
        })
    }

    async fn read_page(&self, page_id: u64) -> StorageResult<Page> {
        Err(StorageError::PageNotFound { page_id })
    }

    async fn write_page(&self, _page_id: u64, _data: &[u8]) -> StorageResult<()> {
        Err(StorageError::NotImplemented {
            feature: "DiskStorage requires 'native' feature",
        })
    }

    async fn page_exists(&self, _page_id: u64) -> StorageResult<bool> {
        Err(StorageError::NotImplemented {
            feature: "DiskStorage requires 'native' feature",
        })
    }

    async fn delete_page(&self, _page_id: u64) -> StorageResult<()> {
        Err(StorageError::NotImplemented {
            feature: "DiskStorage requires 'native' feature",
        })
    }

    async fn sync(&self) -> StorageResult<()> {
        Err(StorageError::NotImplemented {
            feature: "DiskStorage requires 'native' feature",
        })
    }
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;
    use std::env::temp_dir;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_storage_path() -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        temp_dir().join(format!("lattice-test-{}", timestamp))
    }

    #[tokio::test]
    async fn test_disk_storage_init() {
        let path = temp_storage_path();
        let storage = DiskStorage::with_defaults(path.clone());
        storage.init().await.unwrap();

        assert!(path.exists());

        // Cleanup
        fs::remove_dir_all(&path).await.ok();
    }

    #[tokio::test]
    async fn test_disk_storage_metadata() {
        let path = temp_storage_path();
        let storage = DiskStorage::with_defaults(path.clone());
        storage.init().await.unwrap();

        // Set metadata
        storage.set_meta("key1", b"value1").await.unwrap();
        storage.set_meta("key2", b"value2").await.unwrap();

        // Get metadata (from cache)
        let val1 = storage.get_meta("key1").await.unwrap();
        assert_eq!(val1, Some(b"value1".to_vec()));

        // Sync to disk
        storage.sync().await.unwrap();

        // Delete metadata
        storage.delete_meta("key1").await.unwrap();
        let val1 = storage.get_meta("key1").await.unwrap();
        assert_eq!(val1, None);

        // Cleanup
        fs::remove_dir_all(&path).await.ok();
    }

    #[tokio::test]
    async fn test_disk_storage_pages() {
        let path = temp_storage_path();
        let storage = DiskStorage::with_defaults(path.clone());
        storage.init().await.unwrap();

        // Write pages
        storage.write_page(0, b"page0").await.unwrap();
        storage.write_page(1, b"page1").await.unwrap();
        storage.write_page(5, b"page5").await.unwrap();

        // Read pages
        let page0 = storage.read_page(0).await.unwrap();
        assert!(page0.starts_with(b"page0"));

        let page1 = storage.read_page(1).await.unwrap();
        assert!(page1.starts_with(b"page1"));

        let page5 = storage.read_page(5).await.unwrap();
        assert!(page5.starts_with(b"page5"));

        // Non-existent page
        let result = storage.read_page(10).await;
        assert!(matches!(result, Err(StorageError::PageNotFound { .. })));

        // Sync
        storage.sync().await.unwrap();

        // Cleanup
        fs::remove_dir_all(&path).await.ok();
    }

    #[tokio::test]
    async fn test_disk_storage_persistence() {
        let path = temp_storage_path();

        // Write data
        {
            let storage = DiskStorage::with_defaults(path.clone());
            storage.init().await.unwrap();
            storage.set_meta("key", b"value").await.unwrap();
            storage.write_page(0, b"persistent").await.unwrap();
            storage.sync().await.unwrap();
        }

        // Read data in new instance
        {
            let storage = DiskStorage::with_defaults(path.clone());
            storage.init().await.unwrap();

            let meta = storage.get_meta("key").await.unwrap();
            assert_eq!(meta, Some(b"value".to_vec()));

            let page = storage.read_page(0).await.unwrap();
            assert!(page.starts_with(b"persistent"));
        }

        // Cleanup
        fs::remove_dir_all(&path).await.ok();
    }
}
