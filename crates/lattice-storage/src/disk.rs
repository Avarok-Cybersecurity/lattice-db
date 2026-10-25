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
use tokio::fs::{self, File};
#[cfg(feature = "native")]
use tokio::io::AsyncWriteExt;

/// Default largest page a `DiskStorage` holds (1 GiB).
///
/// Pages have no fixed size: a WAL page grows to 256 KiB, a single synced batch
/// can exceed that, and a checkpoint snapshot holds the whole collection. The
/// bound only refuses a runaway page loudly instead of writing it.
pub const DEFAULT_MAX_PAGE_BYTES: usize = 1 << 30;

/// The page layout of releases up to 0.3.4: one `pages.dat` holding page `n` at
/// byte `n * 4096`. A page larger than the stride overwrote its neighbour, so a
/// directory holding one is refused, never read.
const LEGACY_PAGES_FILE: &str = "pages.dat";

/// Disk-based storage backend
///
/// Uses tokio::fs for async file operations.
/// Data is persisted to disk and survives process restarts.
///
/// # File Layout
///
/// ```text
/// base_path/
/// ├── meta.json                         # Metadata key-value store
/// └── pages/
///     └── 00000000000000000000.page     # One file per page, exactly the bytes written
/// ```
///
/// Each page is its own file, so a page of any length never touches another,
/// and a read returns exactly the bytes of the last write.
pub struct DiskStorage {
    /// Base directory for all files
    path: PathBuf,
    /// Largest page in bytes; a larger write fails with `CapacityExceeded`
    max_page_bytes: usize,
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
    /// * `max_page_bytes` - Largest page this storage writes (PCND: required);
    ///   a larger page fails with `StorageError::CapacityExceeded`
    pub fn new(path: PathBuf, max_page_bytes: usize) -> Self {
        Self {
            path,
            max_page_bytes,
            #[cfg(feature = "native")]
            meta_cache: RwLock::new(HashMap::new()),
            #[cfg(feature = "native")]
            meta_dirty: RwLock::new(false),
        }
    }

    /// Create with the default page bound ([`DEFAULT_MAX_PAGE_BYTES`])
    pub fn with_defaults(path: PathBuf) -> Self {
        Self::new(path, DEFAULT_MAX_PAGE_BYTES)
    }

    /// Initialize storage directory and load existing metadata
    ///
    /// Fails with `StorageError::InvalidFormat` on a directory written in the
    /// fixed-stride layout of 0.3.4 and earlier (`pages.dat`): its pages may
    /// already overlap, so it must be rebuilt, not read.
    #[cfg(feature = "native")]
    pub async fn init(&self) -> StorageResult<()> {
        if self.path.join(LEGACY_PAGES_FILE).exists() {
            return Err(StorageError::InvalidFormat {
                expected: "pages/ (one file per page)".to_string(),
                found: format!(
                    "{} (the fixed-stride layout of 0.3.4 and earlier, whose pages overlap; rebuild this store)",
                    self.path.join(LEGACY_PAGES_FILE).display()
                ),
            });
        }

        fs::create_dir_all(self.pages_dir())
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to create storage directory: {}", e),
            })?;

        // Load existing metadata if present
        let meta_path = self.meta_path();
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

    /// Get the directory holding one file per page
    fn pages_dir(&self) -> PathBuf {
        self.path.join("pages")
    }

    /// Get the path of one page's file
    fn page_path(&self, page_id: u64) -> PathBuf {
        self.pages_dir().join(format!("{:020}.page", page_id))
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
        match fs::read(self.page_path(page_id)).await {
            Ok(page) => Ok(page),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(StorageError::PageNotFound { page_id })
            }
            Err(e) => Err(StorageError::Io {
                message: format!("Failed to read page {}: {}", page_id, e),
            }),
        }
    }

    async fn write_page(&self, page_id: u64, data: &[u8]) -> StorageResult<()> {
        if data.len() > self.max_page_bytes {
            return Err(StorageError::CapacityExceeded);
        }

        // Write a temporary file, make it durable, then rename it over the page:
        // a crash leaves either the old page or the new one, never a torn page.
        fs::create_dir_all(self.pages_dir())
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to create pages directory: {}", e),
            })?;
        let final_path = self.page_path(page_id);
        let tmp_path = final_path.with_extension("page.tmp");
        let mut file = File::create(&tmp_path)
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to create page {}: {}", page_id, e),
            })?;
        file.write_all(data).await.map_err(|e| StorageError::Io {
            message: format!("Failed to write page {}: {}", page_id, e),
        })?;
        file.sync_all().await.map_err(|e| StorageError::Io {
            message: format!("Failed to sync page {}: {}", page_id, e),
        })?;
        drop(file);
        fs::rename(&tmp_path, &final_path)
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to install page {}: {}", page_id, e),
            })?;

        Ok(())
    }

    async fn page_exists(&self, page_id: u64) -> StorageResult<bool> {
        fs::try_exists(self.page_path(page_id))
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to stat page {}: {}", page_id, e),
            })
    }

    async fn delete_page(&self, page_id: u64) -> StorageResult<()> {
        match fs::remove_file(self.page_path(page_id)).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(StorageError::Io {
                message: format!("Failed to delete page {}: {}", page_id, e),
            }),
        }
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

        // Page contents are made durable by write_page; make their directory
        // entries durable too (Unix only: Windows cannot open a directory).
        #[cfg(unix)]
        {
            let pages_dir = self.pages_dir();
            if pages_dir.exists() {
                let dir = File::open(&pages_dir).await.map_err(|e| StorageError::Io {
                    message: format!("Failed to open pages directory for sync: {}", e),
                })?;
                dir.sync_all().await.map_err(|e| StorageError::Io {
                    message: format!("Failed to sync pages directory: {}", e),
                })?;
            }
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
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    // Unique per call: cargo runs tests in parallel and the system clock can
    // hand out identical timestamps, so an atomic counter (plus the pid)
    // guarantees each test gets its own directory and they never race on
    // creation/cleanup of a shared path.
    fn temp_storage_path() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        temp_dir().join(format!(
            "lattice-test-{}-{}-{}",
            std::process::id(),
            timestamp,
            unique
        ))
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

    fn pattern(seed: u8, len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| seed.wrapping_add((i % 251) as u8))
            .collect()
    }

    /// Pages larger than 4 KiB (a WAL page grows to 256 KiB) keep their own
    /// bytes: writing page 1 must not overwrite page 0, and a read returns the
    /// exact bytes written, before and after a reopen.
    #[tokio::test]
    async fn test_disk_storage_large_pages_do_not_overlap() {
        let path = temp_storage_path();
        let pages: Vec<Vec<u8>> = (0..4u8)
            .map(|i| pattern(i * 7, 300_000 + usize::from(i)))
            .collect();
        {
            let storage = DiskStorage::with_defaults(path.clone());
            storage.init().await.unwrap();
            for (id, p) in pages.iter().enumerate() {
                storage.write_page(id as u64, p).await.unwrap();
            }
            storage.sync().await.unwrap();
            for (id, p) in pages.iter().enumerate() {
                assert_eq!(&storage.read_page(id as u64).await.unwrap(), p, "page {id}");
            }
        }
        let storage = DiskStorage::with_defaults(path.clone());
        storage.init().await.unwrap();
        for (id, p) in pages.iter().enumerate() {
            assert_eq!(
                &storage.read_page(id as u64).await.unwrap(),
                p,
                "page {id} after reopen"
            );
        }
        fs::remove_dir_all(&path).await.ok();
    }

    /// A page rewritten shorter reads back as the shorter bytes, with no tail
    /// left over from the longer write.
    #[tokio::test]
    async fn test_disk_storage_rewrite_shorter_page_has_no_stale_tail() {
        let path = temp_storage_path();
        let storage = DiskStorage::with_defaults(path.clone());
        storage.init().await.unwrap();
        storage.write_page(3, &pattern(1, 10_000)).await.unwrap();
        storage.write_page(3, b"short").await.unwrap();
        assert_eq!(storage.read_page(3).await.unwrap(), b"short".to_vec());
        storage.delete_page(3).await.unwrap();
        assert!(!storage.page_exists(3).await.unwrap());
        assert!(matches!(
            storage.read_page(3).await,
            Err(StorageError::PageNotFound { page_id: 3 })
        ));
        fs::remove_dir_all(&path).await.ok();
    }

    /// A page larger than the configured bound fails loudly and writes nothing;
    /// a page of exactly the bound is accepted.
    #[tokio::test]
    async fn test_disk_storage_refuses_a_page_over_its_bound() {
        let path = temp_storage_path();
        let storage = DiskStorage::new(path.clone(), 1024);
        storage.init().await.unwrap();
        assert!(matches!(
            storage.write_page(0, &pattern(0, 1025)).await,
            Err(StorageError::CapacityExceeded)
        ));
        assert!(!storage.page_exists(0).await.unwrap());
        storage.write_page(0, &pattern(0, 1024)).await.unwrap();
        assert_eq!(storage.read_page(0).await.unwrap(), pattern(0, 1024));
        fs::remove_dir_all(&path).await.ok();
    }

    /// A directory in the 0.3.4 fixed-stride layout is refused, not misread.
    #[tokio::test]
    async fn test_disk_storage_refuses_the_legacy_layout() {
        let path = temp_storage_path();
        fs::create_dir_all(&path).await.unwrap();
        fs::write(path.join("pages.dat"), vec![0u8; 8192])
            .await
            .unwrap();
        let storage = DiskStorage::with_defaults(path.clone());
        assert!(matches!(
            storage.init().await,
            Err(StorageError::InvalidFormat { .. })
        ));
        fs::remove_dir_all(&path).await.ok();
    }
}
