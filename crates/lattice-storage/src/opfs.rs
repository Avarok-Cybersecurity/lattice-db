//! OPFS (Origin Private File System) storage implementation
//!
//! This storage backend uses the browser's Origin Private File System API
//! for persistent storage in WASM environments.
//!
//! # Requirements
//!
//! - Must run in a Dedicated Worker (for sync API access)
//! - Requires `StorageManager.getDirectory()` permission
//!
//! # Architecture
//!
//! ```text
//! OPFS Root/
//! └── lattice-db/
//!     └── {collection_name}/
//!         ├── meta.json       # Metadata key-value store
//!         └── pages.bin       # Page data (fixed-size pages)
//! ```

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm_impl {
    use async_trait::async_trait;
    use js_sys::Uint8Array;
    use lattice_core::{LatticeStorage, Page, StorageError, StorageResult};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{
        File, FileSystemDirectoryHandle, FileSystemFileHandle, FileSystemGetDirectoryOptions,
        FileSystemGetFileOptions, FileSystemReadWriteOptions, FileSystemSyncAccessHandle,
        FileSystemWritableFileStream,
    };

    /// Maximum allowed collection name length
    const MAX_COLLECTION_NAME_LENGTH: usize = 128;

    /// Validate a collection name to prevent path traversal attacks
    ///
    /// # Security
    ///
    /// Collection names are used as directory names. Allowing arbitrary
    /// user input could enable path traversal attacks (e.g., "../../../etc").
    ///
    /// Valid names contain only:
    /// - ASCII letters (a-z, A-Z)
    /// - Digits (0-9)
    /// - Underscores (_)
    /// - Hyphens (-)
    ///
    /// Names must be 1-128 characters and cannot be "." or "..".
    fn validate_collection_name(name: &str) -> StorageResult<()> {
        // Check length
        if name.is_empty() {
            return Err(StorageError::Io {
                message: "Collection name cannot be empty".to_string(),
            });
        }
        if name.len() > MAX_COLLECTION_NAME_LENGTH {
            return Err(StorageError::Io {
                message: format!(
                    "Collection name too long: {} > {} characters",
                    name.len(),
                    MAX_COLLECTION_NAME_LENGTH
                ),
            });
        }

        // Reject . and .. explicitly
        if name == "." || name == ".." {
            return Err(StorageError::Io {
                message: "Collection name cannot be '.' or '..'".to_string(),
            });
        }

        // Validate characters
        for c in name.chars() {
            if !c.is_ascii_alphanumeric() && c != '_' && c != '-' {
                return Err(StorageError::Io {
                    message: format!(
                        "Invalid character '{}' in collection name. Only a-z, A-Z, 0-9, _, - allowed",
                        c
                    ),
                });
            }
        }

        Ok(())
    }

    /// OPFS-based storage backend for browser environments
    ///
    /// Uses `FileSystemSyncAccessHandle` for efficient synchronous I/O.
    /// Must run in a Dedicated Worker (not main thread).
    ///
    /// # PCND Compliance
    ///
    /// All configuration parameters are required - no defaults.
    pub struct OpfsStorage {
        /// Size of each page in bytes
        page_size: usize,
        /// Directory handle for this collection
        dir_handle: FileSystemDirectoryHandle,
        /// Metadata cache (persisted to meta.json)
        meta_cache: RefCell<HashMap<String, Vec<u8>>>,
        /// Meta file handle
        meta_handle: FileSystemFileHandle,
    }

    impl OpfsStorage {
        /// Create a new OPFS storage for a collection
        ///
        /// # Arguments
        /// * `collection_name` - Name of the collection (used as directory name).
        ///   Must contain only alphanumeric characters, underscores, and hyphens.
        /// * `page_size` - Size of each page in bytes (PCND: required)
        ///
        /// # Errors
        /// Returns error if:
        /// - Collection name is invalid (path traversal attempt, invalid characters)
        /// - OPFS is not available
        /// - Directory creation fails
        pub async fn new(collection_name: &str, page_size: usize) -> StorageResult<Self> {
            // Validate collection name to prevent path traversal
            validate_collection_name(collection_name)?;

            // Get OPFS root directory
            let root = get_opfs_root().await?;

            // Create/get lattice-db directory
            let lattice_dir = get_or_create_dir(&root, "lattice-db").await?;

            // Create/get collection directory (safe after validation)
            let dir_handle = get_or_create_dir(&lattice_dir, collection_name).await?;

            // Get or create meta file
            let meta_handle = get_or_create_file(&dir_handle, "meta.json").await?;

            // Load existing metadata
            let meta_cache = load_meta(&meta_handle).await?;

            Ok(Self {
                page_size,
                dir_handle,
                meta_cache: RefCell::new(meta_cache),
                meta_handle,
            })
        }

        /// Get sync access handle for pages file
        async fn get_pages_handle(&self) -> StorageResult<FileSystemSyncAccessHandle> {
            // Get or create pages file
            let pages_file = get_or_create_file(&self.dir_handle, "pages.bin").await?;

            // Get sync access handle (requires worker context)
            get_sync_access_handle(&pages_file).await
        }

        /// Calculate byte offset for a page
        fn page_offset(&self, page_id: u64) -> f64 {
            (page_id * self.page_size as u64) as f64
        }

        /// Persist metadata to file
        async fn persist_meta(&self) -> StorageResult<()> {
            // Clone the data to avoid holding RefCell borrow across await points
            let json = {
                let meta = self.meta_cache.borrow();
                serde_json::to_vec(&*meta).map_err(|e| StorageError::Serialization {
                    message: format!("Failed to serialize metadata: {}", e),
                })?
            };

            // Write to meta file
            let writable = get_writable_stream(&self.meta_handle).await?;
            write_to_stream(&writable, &json).await?;
            close_stream(&writable).await?;

            Ok(())
        }
    }

    #[async_trait(?Send)]
    impl LatticeStorage for OpfsStorage {
        async fn get_meta(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
            let meta = self.meta_cache.borrow();
            Ok(meta.get(key).cloned())
        }

        async fn set_meta(&self, key: &str, value: &[u8]) -> StorageResult<()> {
            {
                let mut meta = self.meta_cache.borrow_mut();
                meta.insert(key.to_string(), value.to_vec());
            }
            self.persist_meta().await
        }

        async fn delete_meta(&self, key: &str) -> StorageResult<()> {
            {
                let mut meta = self.meta_cache.borrow_mut();
                meta.remove(key);
            }
            self.persist_meta().await
        }

        async fn read_page(&self, page_id: u64) -> StorageResult<Page> {
            let handle = self.get_pages_handle().await?;
            let offset = self.page_offset(page_id);

            // Check file size
            let file_size = handle.get_size().map_err(|e| StorageError::Io {
                message: format!("Failed to get file size: {:?}", e),
            })?;

            if offset >= file_size {
                return Err(StorageError::PageNotFound { page_id });
            }

            // Read page data
            let buffer = Uint8Array::new_with_length(self.page_size as u32);
            let options = FileSystemReadWriteOptions::new();
            options.set_at(offset);

            let bytes_read = handle
                .read_with_buffer_source_and_options(&buffer, &options)
                .map_err(|e| StorageError::Io {
                    message: format!("Sync read failed: {:?}", e),
                })?;

            if bytes_read == 0.0 {
                return Err(StorageError::PageNotFound { page_id });
            }

            // Convert to Vec<u8>
            let mut data = vec![0u8; bytes_read as usize];
            buffer.slice(0, bytes_read as u32).copy_to(&mut data);

            // Close handle to release lock
            handle.close();

            Ok(data)
        }

        async fn write_page(&self, page_id: u64, data: &[u8]) -> StorageResult<()> {
            if data.len() > self.page_size {
                return Err(StorageError::Io {
                    message: format!(
                        "Data size {} exceeds page size {}",
                        data.len(),
                        self.page_size
                    ),
                });
            }

            let handle = self.get_pages_handle().await?;
            let offset = self.page_offset(page_id);

            // Pad data to page size
            let mut padded = data.to_vec();
            padded.resize(self.page_size, 0);

            // Convert to Uint8Array
            let buffer = Uint8Array::new_with_length(padded.len() as u32);
            buffer.copy_from(&padded);

            // Write with offset
            let options = FileSystemReadWriteOptions::new();
            options.set_at(offset);

            handle
                .write_with_buffer_source_and_options(&buffer, &options)
                .map_err(|e| StorageError::Io {
                    message: format!("Sync write failed: {:?}", e),
                })?;

            // Flush and close
            let _ = handle.flush();
            handle.close();

            Ok(())
        }

        async fn page_exists(&self, page_id: u64) -> StorageResult<bool> {
            let handle = self.get_pages_handle().await?;
            let offset = self.page_offset(page_id);

            let file_size = handle.get_size().map_err(|e| StorageError::Io {
                message: format!("Failed to get file size: {:?}", e),
            })?;

            handle.close();

            Ok(offset < file_size)
        }

        async fn delete_page(&self, page_id: u64) -> StorageResult<()> {
            // For fixed-size page storage, "deleting" means zeroing the page
            let handle = self.get_pages_handle().await?;
            let offset = self.page_offset(page_id);

            // Check if page exists
            let file_size = handle.get_size().map_err(|e| StorageError::Io {
                message: format!("Failed to get file size: {:?}", e),
            })?;

            if offset >= file_size {
                handle.close();
                return Ok(()); // Already doesn't exist
            }

            // Zero out the page
            let zeros = Uint8Array::new_with_length(self.page_size as u32);
            let options = FileSystemReadWriteOptions::new();
            options.set_at(offset);

            handle
                .write_with_buffer_source_and_options(&zeros, &options)
                .map_err(|e| StorageError::Io {
                    message: format!("Failed to zero page: {:?}", e),
                })?;

            let _ = handle.flush();
            handle.close();

            Ok(())
        }

        async fn sync(&self) -> StorageResult<()> {
            // Meta is persisted on each write
            // Pages are flushed after each write
            Ok(())
        }
    }

    // === Helper functions for OPFS API ===

    /// Get the OPFS root directory
    async fn get_opfs_root() -> StorageResult<FileSystemDirectoryHandle> {
        let window = web_sys::window().ok_or(StorageError::Io {
            message: "No window object available".to_string(),
        })?;

        let navigator = window.navigator();
        let storage = navigator.storage();

        let promise = storage.get_directory();
        let result = JsFuture::from(promise)
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to get OPFS root: {:?}", e),
            })?;

        Ok(result.unchecked_into())
    }

    /// Get or create a subdirectory
    async fn get_or_create_dir(
        parent: &FileSystemDirectoryHandle,
        name: &str,
    ) -> StorageResult<FileSystemDirectoryHandle> {
        let options = FileSystemGetDirectoryOptions::new();
        options.set_create(true);

        let promise = parent.get_directory_handle_with_options(name, &options);
        let result = JsFuture::from(promise)
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to get/create directory '{}': {:?}", name, e),
            })?;

        Ok(result.unchecked_into())
    }

    /// Get or create a file
    async fn get_or_create_file(
        dir: &FileSystemDirectoryHandle,
        name: &str,
    ) -> StorageResult<FileSystemFileHandle> {
        let options = FileSystemGetFileOptions::new();
        options.set_create(true);

        let promise = dir.get_file_handle_with_options(name, &options);
        let result = JsFuture::from(promise)
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to get/create file '{}': {:?}", name, e),
            })?;

        Ok(result.unchecked_into())
    }

    /// Get sync access handle for a file (requires worker context)
    async fn get_sync_access_handle(
        file: &FileSystemFileHandle,
    ) -> StorageResult<FileSystemSyncAccessHandle> {
        let promise = file.create_sync_access_handle();
        let result = JsFuture::from(promise)
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to create sync access handle: {:?}", e),
            })?;

        Ok(result.unchecked_into())
    }

    /// Get writable stream for a file
    async fn get_writable_stream(
        file: &FileSystemFileHandle,
    ) -> StorageResult<FileSystemWritableFileStream> {
        let promise = file.create_writable();
        let result = JsFuture::from(promise)
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to create writable stream: {:?}", e),
            })?;

        Ok(result.unchecked_into())
    }

    /// Write data to a writable stream
    async fn write_to_stream(
        stream: &FileSystemWritableFileStream,
        data: &[u8],
    ) -> StorageResult<()> {
        let len = u32::try_from(data.len()).map_err(|_| StorageError::Io {
            message: format!(
                "Data too large for OPFS write: {} bytes (max: 4GB)",
                data.len()
            ),
        })?;
        let array = Uint8Array::new_with_length(len);
        array.copy_from(data);

        let promise = stream
            .write_with_buffer_source(&array)
            .map_err(|e| StorageError::Io {
                message: format!("Failed to start write: {:?}", e),
            })?;
        JsFuture::from(promise)
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to write to stream: {:?}", e),
            })?;

        Ok(())
    }

    /// Close a writable stream
    async fn close_stream(stream: &FileSystemWritableFileStream) -> StorageResult<()> {
        let promise = stream.close();
        JsFuture::from(promise)
            .await
            .map_err(|e| StorageError::Io {
                message: format!("Failed to close stream: {:?}", e),
            })?;

        Ok(())
    }

    /// Load metadata from file
    async fn load_meta(file: &FileSystemFileHandle) -> StorageResult<HashMap<String, Vec<u8>>> {
        // Get file as blob
        let promise = file.get_file();
        let blob = match JsFuture::from(promise).await {
            Ok(b) => b,
            Err(_) => return Ok(HashMap::new()), // File doesn't exist or is empty
        };

        let file_obj: File = blob.unchecked_into();
        let array_buffer_promise = file_obj.array_buffer();
        let array_buffer =
            JsFuture::from(array_buffer_promise)
                .await
                .map_err(|e| StorageError::Io {
                    message: format!("Failed to read meta file: {:?}", e),
                })?;

        let uint8_array = Uint8Array::new(&array_buffer);
        let bytes = uint8_array.to_vec();

        if bytes.is_empty() {
            return Ok(HashMap::new());
        }

        serde_json::from_slice(&bytes).map_err(|e| StorageError::Serialization {
            message: format!("Failed to parse metadata: {}", e),
        })
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use wasm_impl::OpfsStorage;

// Placeholder for non-wasm builds (allows the code to compile on native)
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub struct OpfsStorage {
    _private: (),
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
impl OpfsStorage {
    /// OpfsStorage is only available in WASM builds with the `wasm` feature
    pub fn new(_collection_name: &str, _page_size: usize) -> Result<Self, &'static str> {
        Err("OpfsStorage is only available in WASM builds with the `wasm` feature")
    }
}
