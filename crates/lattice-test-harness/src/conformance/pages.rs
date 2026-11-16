//! Pages: exact bytes at WAL size, under any `u64` id
//!
//! The WAL tells a page that was never written from an empty one by
//! `StorageError::PageNotFound`, and a snapshot uses it to detect a missing
//! chunk, so an absent page must not read as empty bytes. Snapshot chunks
//! live above `1 << 32`; an adapter that keys pages by a JavaScript number
//! merges ids above `2^53`.

use super::{ensure, CheckResult, Context, StorageFixture};
use lattice_core::storage::StorageError;
use lattice_core::wal::WAL_PAGE_MAX_BYTES;
use lattice_core::LatticeStorage;

/// Distinct, position-dependent bytes, so a truncated or shifted page differs
pub(crate) fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

pub(crate) async fn expect_page<S: LatticeStorage>(
    storage: &S,
    page_id: u64,
    expected: &[u8],
) -> CheckResult {
    let page = storage.read_page(page_id).await.ctx("read_page")?;
    ensure(page == expected, || {
        format!(
            "page {} read back {} bytes that differ from the {} written",
            page_id,
            page.len(),
            expected.len()
        )
    })
}

pub(crate) async fn expect_absent<S: LatticeStorage>(storage: &S, page_id: u64) -> CheckResult {
    match storage.read_page(page_id).await {
        Err(StorageError::PageNotFound { .. }) => {}
        Ok(page) => {
            return Err(format!(
                "absent page {} read as {} bytes instead of PageNotFound",
                page_id,
                page.len()
            ))
        }
        Err(e) => return Err(format!("absent page {} failed to read: {}", page_id, e)),
    }
    let exists = storage.page_exists(page_id).await.ctx("page_exists")?;
    ensure(!exists, || {
        format!("page_exists({}) is true for an absent page", page_id)
    })
}

/// A page never written is `PageNotFound` and does not exist
pub async fn missing_page_is_not_found<F: StorageFixture>(fixture: &F) -> CheckResult {
    let storage = fixture.open().await.ctx("open")?;
    expect_absent(&storage, 0).await
}

/// A page reads back exactly the bytes written, with no padding
pub async fn page_round_trips_exactly<F: StorageFixture>(fixture: &F) -> CheckResult {
    let storage = fixture.open().await.ctx("open")?;
    let data = pattern(4099, 1);
    storage.write_page(3, &data).await.ctx("write_page")?;
    expect_page(&storage, 3, &data).await?;
    let exists = storage.page_exists(3).await.ctx("page_exists")?;
    ensure(exists, || "page_exists is false for a written page".into())
}

/// Overwriting replaces the whole page, whether shorter or longer
pub async fn overwrite_replaces_the_whole_page<F: StorageFixture>(fixture: &F) -> CheckResult {
    let storage = fixture.open().await.ctx("open")?;
    for (len, seed) in [(10_000, 1), (17, 2), (20_000, 3)] {
        let data = pattern(len, seed);
        storage.write_page(0, &data).await.ctx("write_page")?;
        expect_page(&storage, 0, &data).await?;
    }
    Ok(())
}

/// A page as large as a full WAL page round-trips
///
/// Key-value stores with a 128 KiB value limit fail here.
pub async fn wal_sized_page_round_trips<F: StorageFixture>(fixture: &F) -> CheckResult {
    let storage = fixture.open().await.ctx("open")?;
    let data = pattern(WAL_PAGE_MAX_BYTES as usize, 4);
    storage
        .write_page(1, &data)
        .await
        .ctx("write_page of a full WAL page")?;
    expect_page(&storage, 1, &data).await
}

/// Ids anywhere in `u64` name distinct pages
pub async fn page_ids_are_independent<F: StorageFixture>(fixture: &F) -> CheckResult {
    let storage = fixture.open().await.ctx("open")?;
    let ids = [
        0,
        1,
        1 << 32,
        (1 << 33) + 5,
        1 << 53,
        (1 << 53) + 1,
        u64::MAX - 1,
        u64::MAX,
    ];
    for (i, id) in ids.iter().enumerate() {
        storage
            .write_page(*id, &pattern(64 + i, i as u8))
            .await
            .ctx("write_page")?;
    }
    for (i, id) in ids.iter().enumerate() {
        expect_page(&storage, *id, &pattern(64 + i, i as u8)).await?;
    }
    storage.delete_page(1 << 53).await.ctx("delete_page")?;
    expect_absent(&storage, 1 << 53).await?;
    expect_page(&storage, (1 << 53) + 1, &pattern(69, 5)).await
}

/// A deleted page is absent; deleting an absent page is `Ok` or
/// `PageNotFound`, the two outcomes the WAL accepts
pub async fn delete_removes_the_page<F: StorageFixture>(fixture: &F) -> CheckResult {
    let storage = fixture.open().await.ctx("open")?;
    storage
        .write_page(2, &pattern(32, 9))
        .await
        .ctx("write_page")?;
    storage.delete_page(2).await.ctx("delete_page")?;
    expect_absent(&storage, 2).await?;
    match storage.delete_page(2).await {
        Ok(()) | Err(StorageError::PageNotFound { .. }) => Ok(()),
        Err(e) => Err(format!("deleting an absent page failed: {}", e)),
    }
}
