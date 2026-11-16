//! The storage backends in this crate pass the `LatticeStorage` conformance
//! suite: `MemStorage` on every target, `DiskStorage` natively.

use lattice_storage::MemStorage;
use lattice_test_harness::conformance::Reopen;
use lattice_test_harness::run_conformance;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn mem_storage_conforms() {
    run_conformance(|| Reopen::new(MemStorage::new()))
        .await
        .assert_conformant();
}

#[cfg(feature = "native")]
mod disk {
    use lattice_core::storage::StorageResult;
    use lattice_storage::disk::DEFAULT_MAX_PAGE_BYTES;
    use lattice_storage::DiskStorage;
    use lattice_test_harness::{run_conformance, StorageFixture};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A directory of its own; each `open` is a process starting over it
    struct DiskFixture {
        path: PathBuf,
    }

    impl DiskFixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "lattice-conformance-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&path);
            Self { path }
        }
    }

    impl Drop for DiskFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[async_trait::async_trait(?Send)]
    impl StorageFixture for DiskFixture {
        type Storage = DiskStorage;

        async fn open(&self) -> StorageResult<DiskStorage> {
            let storage = DiskStorage::new(self.path.clone(), DEFAULT_MAX_PAGE_BYTES);
            storage.init().await?;
            Ok(storage)
        }
    }

    #[tokio::test]
    async fn disk_storage_conforms() {
        run_conformance(DiskFixture::new).await.assert_conformant();
    }
}
