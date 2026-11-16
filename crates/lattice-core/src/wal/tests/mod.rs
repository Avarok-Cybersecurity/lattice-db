#![cfg(not(target_arch = "wasm32"))]
//! Unit tests for the WAL (native).

use super::*;
use crate::storage::StorageResult;
use crate::types::point::Point;

// Mock storage for testing
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

struct MockStorage {
    pages: Arc<RwLock<HashMap<u64, Vec<u8>>>>,
    meta: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl MockStorage {
    fn new() -> Self {
        Self {
            pages: Arc::new(RwLock::new(HashMap::new())),
            meta: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
impl LatticeStorage for MockStorage {
    async fn get_meta(&self, key: &str) -> StorageResult<Option<Vec<u8>>> {
        Ok(self.meta.read().unwrap().get(key).cloned())
    }

    async fn set_meta(&self, key: &str, value: &[u8]) -> StorageResult<()> {
        self.meta
            .write()
            .unwrap()
            .insert(key.to_string(), value.to_vec());
        Ok(())
    }

    async fn delete_meta(&self, key: &str) -> StorageResult<()> {
        self.meta.write().unwrap().remove(key);
        Ok(())
    }

    async fn read_page(&self, page_id: u64) -> StorageResult<Vec<u8>> {
        Ok(self
            .pages
            .read()
            .unwrap()
            .get(&page_id)
            .cloned()
            .unwrap_or_default())
    }

    async fn write_page(&self, page_id: u64, data: &[u8]) -> StorageResult<()> {
        self.pages.write().unwrap().insert(page_id, data.to_vec());
        Ok(())
    }

    async fn page_exists(&self, page_id: u64) -> StorageResult<bool> {
        Ok(self.pages.read().unwrap().contains_key(&page_id))
    }

    async fn delete_page(&self, page_id: u64) -> StorageResult<()> {
        self.pages.write().unwrap().remove(&page_id);
        Ok(())
    }

    async fn sync(&self) -> StorageResult<()> {
        Ok(())
    }
}

mod basic;
mod pages;
