//! Metadata blobs: exact bytes, one key at a time
//!
//! The WAL's commit point is one metadata key (`WAL_HEADER_KEY`), and the
//! checkpoint snapshot's is another (`SNAPSHOT_MANIFEST_KEY`), so a backend
//! that mixes keys up or keeps part of an old value breaks recovery.

use super::{ensure, CheckResult, Context, StorageFixture};
use lattice_core::wal::snapshot::SNAPSHOT_MANIFEST_KEY;
use lattice_core::wal::WAL_HEADER_KEY;
use lattice_core::LatticeStorage;

/// Bytes a text-based encoding could mangle: NUL, high bytes, invalid UTF-8
fn awkward_bytes() -> Vec<u8> {
    let mut bytes: Vec<u8> = (0..=255u8).collect();
    bytes.extend_from_slice(&[0xC3, 0x28, 0xFF, 0xFE, 0x00, 0x00]);
    bytes
}

async fn expect_value<S: LatticeStorage>(
    storage: &S,
    key: &str,
    expected: Option<&[u8]>,
) -> CheckResult {
    let found = storage.get_meta(key).await.ctx("get_meta")?;
    ensure(found.as_deref() == expected, || {
        format!(
            "get_meta({:?}) returned {:?} bytes, expected {:?}",
            key,
            found.as_ref().map(Vec::len),
            expected.map(<[u8]>::len)
        )
    })
}

/// A key never written reads as `None`, not as an empty value or an error
pub async fn missing_key_is_none<F: StorageFixture>(fixture: &F) -> CheckResult {
    let storage = fixture.open().await.ctx("open")?;
    expect_value(&storage, WAL_HEADER_KEY, None).await
}

/// A value reads back byte for byte, including an empty one
pub async fn value_round_trips_exactly<F: StorageFixture>(fixture: &F) -> CheckResult {
    let storage = fixture.open().await.ctx("open")?;
    let value = awkward_bytes();
    storage.set_meta("binary", &value).await.ctx("set_meta")?;
    storage.set_meta("empty", &[]).await.ctx("set_meta")?;
    expect_value(&storage, "binary", Some(&value)).await?;
    expect_value(&storage, "empty", Some(&[])).await
}

/// Setting a key replaces its value; no tail of a longer old value remains
pub async fn set_replaces_the_whole_value<F: StorageFixture>(fixture: &F) -> CheckResult {
    let storage = fixture.open().await.ctx("open")?;
    storage
        .set_meta(WAL_HEADER_KEY, &[7u8; 512])
        .await
        .ctx("set_meta")?;
    storage
        .set_meta(WAL_HEADER_KEY, &[9u8; 3])
        .await
        .ctx("set_meta")?;
    expect_value(&storage, WAL_HEADER_KEY, Some(&[9u8; 3])).await
}

/// Writing or deleting one key leaves every other key as it was, including
/// keys that share a prefix with it
pub async fn keys_are_independent<F: StorageFixture>(fixture: &F) -> CheckResult {
    let storage = fixture.open().await.ctx("open")?;
    let keys = [
        WAL_HEADER_KEY,
        SNAPSHOT_MANIFEST_KEY,
        "wal",
        "wal_header_",
        "WAL_HEADER",
    ];
    for (i, key) in keys.iter().enumerate() {
        storage.set_meta(key, &[i as u8; 8]).await.ctx("set_meta")?;
    }
    storage
        .set_meta(WAL_HEADER_KEY, b"replaced")
        .await
        .ctx("set_meta")?;
    storage.delete_meta("wal").await.ctx("delete_meta")?;

    expect_value(&storage, WAL_HEADER_KEY, Some(b"replaced")).await?;
    expect_value(&storage, "wal", None).await?;
    for (i, key) in keys.iter().enumerate().skip(1) {
        if *key != "wal" {
            expect_value(&storage, key, Some(&[i as u8; 8])).await?;
        }
    }
    Ok(())
}

/// A deleted key reads as `None`; deleting a key that is absent succeeds
pub async fn delete_removes_the_key<F: StorageFixture>(fixture: &F) -> CheckResult {
    let storage = fixture.open().await.ctx("open")?;
    storage.set_meta("doomed", b"x").await.ctx("set_meta")?;
    storage.delete_meta("doomed").await.ctx("delete_meta")?;
    expect_value(&storage, "doomed", None).await?;
    storage
        .delete_meta("never-written")
        .await
        .ctx("delete_meta of an absent key")
}
