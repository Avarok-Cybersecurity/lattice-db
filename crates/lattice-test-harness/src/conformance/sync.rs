//! Sync: what was written and synced is there after a restart
//!
//! The WAL writes a page, then the header key, then calls `sync`; an `Ok`
//! from `sync` is what turns a commit into an acknowledged one. A backend may
//! make writes durable earlier (a Durable Object's storage does), but never
//! later.

use super::pages::{expect_absent, expect_page, pattern};
use super::{ensure, CheckResult, Context, StorageFixture};
use lattice_core::LatticeStorage;

/// `sync` with nothing written succeeds, and so does a second one
pub async fn sync_on_an_empty_store_succeeds<F: StorageFixture>(fixture: &F) -> CheckResult {
    let storage = fixture.open().await.ctx("open")?;
    storage.sync().await.ctx("sync")?;
    storage.sync().await.ctx("second sync")
}

/// Pages and metadata written then synced read back after a reopen
pub async fn synced_writes_survive_reopen<F: StorageFixture>(fixture: &F) -> CheckResult {
    {
        let storage = fixture.open().await.ctx("open")?;
        storage
            .write_page(0, &pattern(5000, 1))
            .await
            .ctx("write_page")?;
        storage
            .write_page(1 << 32, &pattern(70, 2))
            .await
            .ctx("write_page")?;
        storage.set_meta("kept", b"value").await.ctx("set_meta")?;
        storage.sync().await.ctx("sync")?;
    }

    let storage = fixture.open().await.ctx("reopen")?;
    expect_page(&storage, 0, &pattern(5000, 1)).await?;
    expect_page(&storage, 1 << 32, &pattern(70, 2)).await?;
    let kept = storage.get_meta("kept").await.ctx("get_meta")?;
    ensure(kept.as_deref() == Some(&b"value"[..]), || {
        format!("synced metadata read back as {:?} after reopen", kept)
    })
}

/// Deletes and overwrites synced before a restart are not undone by it
pub async fn synced_deletes_survive_reopen<F: StorageFixture>(fixture: &F) -> CheckResult {
    {
        let storage = fixture.open().await.ctx("open")?;
        storage
            .write_page(0, &pattern(100, 1))
            .await
            .ctx("write_page")?;
        storage
            .write_page(1, &pattern(100, 2))
            .await
            .ctx("write_page")?;
        storage.set_meta("gone", b"x").await.ctx("set_meta")?;
        storage.set_meta("changed", b"old").await.ctx("set_meta")?;
        storage.sync().await.ctx("sync")?;
    }
    {
        let storage = fixture.open().await.ctx("reopen")?;
        storage.delete_page(0).await.ctx("delete_page")?;
        storage
            .write_page(1, &pattern(30, 3))
            .await
            .ctx("write_page")?;
        storage.delete_meta("gone").await.ctx("delete_meta")?;
        storage.set_meta("changed", b"new").await.ctx("set_meta")?;
        storage.sync().await.ctx("sync")?;
    }

    let storage = fixture.open().await.ctx("second reopen")?;
    expect_absent(&storage, 0).await?;
    expect_page(&storage, 1, &pattern(30, 3)).await?;
    let gone = storage.get_meta("gone").await.ctx("get_meta")?;
    let changed = storage.get_meta("changed").await.ctx("get_meta")?;
    ensure(
        gone.is_none() && changed.as_deref() == Some(&b"new"[..]),
        || {
            format!(
                "after reopen a deleted key reads {:?} and an overwritten one {:?}",
                gone, changed
            )
        },
    )
}
