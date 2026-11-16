//! Crash consistency with the WAL on top
//!
//! These run the real `WriteAheadLog` over the backend. Dropping the log and
//! opening a new one over a reopened handle is a crash and restart. Two
//! properties must hold whatever the backend:
//!
//! - every commit whose `sync` returned `Ok` is read back, at its LSN;
//! - a commit whose page or header write failed is never read back, and the
//!   next commit takes its LSN.
//!
//! The failures are injected by `FailingStorage` in front of the backend and
//! return before reaching it (see the module docs of `conformance`).

use super::{ensure, CheckResult, Context, StorageFixture};
use crate::FailingStorage;
use lattice_core::types::point::Point;
use lattice_core::wal::{WalEntry, WriteAheadLog, WAL_PAGE_MAX_BYTES};
use lattice_core::LatticeStorage;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Vector length of a small entry: a few dozen bytes on the page
const SMALL: usize = 1;

/// Vector length of an entry about a sixth of a WAL page
const LARGE: usize = (WAL_PAGE_MAX_BYTES as usize / 6) / 4;

fn upsert(id: u64, dim: usize) -> WalEntry {
    WalEntry::Upsert {
        points: vec![Point::new_vector(id, vec![id as f32; dim])],
    }
}

/// `(lsn, point id)` of every upsert the log holds from LSN 0
async fn logged<S: LatticeStorage>(wal: &WriteAheadLog<S>) -> Result<Vec<(u64, u64)>, String> {
    let entries = wal.read_from(0).await.ctx("WAL read_from")?;
    Ok(entries
        .iter()
        .filter_map(|(lsn, entry)| match entry {
            WalEntry::Upsert { points } => Some((*lsn, points[0].id)),
            _ => None,
        })
        .collect())
}

async fn reopen_log<F: StorageFixture>(fixture: &F) -> Result<WriteAheadLog<F::Storage>, String> {
    let storage = fixture.open().await.ctx("reopen")?;
    WriteAheadLog::open(storage)
        .await
        .ctx("WAL open after reopen")
}

async fn expect_logged<F: StorageFixture>(fixture: &F, expected: &[(u64, u64)]) -> CheckResult {
    let wal = reopen_log(fixture).await?;
    let found = logged(&wal).await?;
    ensure(found == expected, || {
        format!(
            "after reopen the log holds (lsn, id) {:?}, expected {:?}",
            found, expected
        )
    })
}

/// Commit `ids` over a fresh handle, one sync each
async fn commit<F: StorageFixture>(fixture: &F, ids: &[u64], dim: usize) -> CheckResult {
    let mut wal = reopen_log(fixture).await?;
    for &id in ids {
        wal.append(&upsert(id, dim)).await.ctx("WAL append")?;
        wal.sync().await.ctx("WAL sync")?;
    }
    Ok(())
}

/// Acknowledged commits, across several WAL pages and one multi-entry batch,
/// are all read back after a restart at the LSNs they were given
pub async fn acknowledged_commits_survive_reopen<F: StorageFixture>(fixture: &F) -> CheckResult {
    let singles: Vec<u64> = (0..14).collect();
    commit(fixture, &singles, LARGE).await?;
    {
        let mut wal = reopen_log(fixture).await?;
        for id in 100..104 {
            wal.append(&upsert(id, LARGE)).await.ctx("WAL append")?;
        }
        wal.sync().await.ctx("WAL sync of a batch")?;
    }

    let expected: Vec<(u64, u64)> = singles
        .iter()
        .chain(&[100, 101, 102, 103])
        .enumerate()
        .map(|(lsn, &id)| (lsn as u64, id))
        .collect();
    expect_logged(fixture, &expected).await?;

    let storage = fixture.open().await.ctx("reopen")?;
    let rotated = storage.page_exists(2).await.ctx("page_exists")?;
    ensure(rotated, || {
        "the commits never reached a third WAL page, so rotation went untested".into()
    })
}

/// Which storage call of the failing commit is made to fail
#[derive(Clone, Copy)]
enum Failure {
    PageWrite,
    HeaderWrite,
}

/// Commit a baseline of `baseline_dim` entries, fail one commit of a
/// `failing_dim` entry, restart, and check that only the baseline is there
/// and that the next commit of that size takes the failed one's LSN
async fn failed_commit_is_invisible<F: StorageFixture>(
    fixture: &F,
    failure: Failure,
    baseline_dim: usize,
    failing_dim: usize,
) -> CheckResult {
    let baseline: Vec<u64> = (0..3).collect();
    commit(fixture, &baseline, baseline_dim).await?;
    let mut expected: Vec<(u64, u64)> = baseline.iter().map(|&id| (id, id)).collect();

    {
        let failing = FailingStorage::wrap(fixture.open().await.ctx("reopen")?);
        let trigger: Arc<AtomicU64> = match failure {
            Failure::PageWrite => failing.write_page_fail_at.clone(),
            Failure::HeaderWrite => failing.set_meta_fail_at.clone(),
        };
        trigger.store(1, Ordering::SeqCst);
        let mut wal = WriteAheadLog::open(failing).await.ctx("WAL open")?;
        wal.append(&upsert(99, failing_dim))
            .await
            .ctx("WAL append")?;
        ensure(wal.sync().await.is_err(), || {
            "the injected storage failure did not fail the commit".into()
        })?;
    }
    expect_logged(fixture, &expected).await?;

    commit(fixture, &[7], failing_dim).await?;
    expected.push((3, 7));
    expect_logged(fixture, &expected).await?;

    let storage = fixture.open().await.ctx("reopen")?;
    let rotated = storage.page_exists(1).await.ctx("page_exists")?;
    let expect_rotation = failing_dim == ROTATING;
    ensure(rotated == expect_rotation, || {
        format!(
            "WAL page 1 exists: {}, but this check {} a rotation",
            rotated,
            if expect_rotation {
                "needs"
            } else {
                "must not make"
            }
        )
    })
}

/// Vector length of an entry that cannot share a page with a `LARGE`
/// baseline: three sixths of a page plus six tenths overflow it
const ROTATING: usize = (WAL_PAGE_MAX_BYTES as usize * 6 / 10) / 4;

/// A commit whose page write failed is not there after a restart
pub async fn failed_page_write_is_not_visible_after_reopen<F: StorageFixture>(
    fixture: &F,
) -> CheckResult {
    failed_commit_is_invisible(fixture, Failure::PageWrite, SMALL, SMALL).await
}

/// A commit whose page landed but whose header write failed is not there
/// after a restart: the page's bytes past the header's length are ignored
pub async fn failed_header_write_is_not_visible_after_reopen<F: StorageFixture>(
    fixture: &F,
) -> CheckResult {
    failed_commit_is_invisible(fixture, Failure::HeaderWrite, SMALL, SMALL).await
}

/// A commit that rotates to a new WAL page and fails writing that page is
/// not there after a restart
pub async fn failed_rotating_page_write_is_not_visible_after_reopen<F: StorageFixture>(
    fixture: &F,
) -> CheckResult {
    failed_commit_is_invisible(fixture, Failure::PageWrite, LARGE, ROTATING).await
}

/// A commit that wrote its new WAL page but not the header naming it is not
/// there after a restart, and the orphaned page is reused, not replayed
pub async fn failed_rotating_header_write_is_not_visible_after_reopen<F: StorageFixture>(
    fixture: &F,
) -> CheckResult {
    failed_commit_is_invisible(fixture, Failure::HeaderWrite, LARGE, ROTATING).await
}
