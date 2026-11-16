//! Append, read-back, checkpoint and recovery on a single page.

use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
async fn test_wal_append_and_read() {
    let storage = MockStorage::new();
    let mut wal = WriteAheadLog::open(storage).await.unwrap();

    // Append some entries
    let point1 = Point::new_vector(1, vec![0.1, 0.2, 0.3]);
    let point2 = Point::new_vector(2, vec![0.4, 0.5, 0.6]);

    let lsn1 = wal
        .append(&WalEntry::Upsert {
            points: vec![point1.clone()],
        })
        .await
        .unwrap();
    let lsn2 = wal
        .append(&WalEntry::Upsert {
            points: vec![point2.clone()],
        })
        .await
        .unwrap();
    let lsn3 = wal
        .append(&WalEntry::Delete { ids: vec![1] })
        .await
        .unwrap();

    assert_eq!(lsn1, 0);
    assert_eq!(lsn2, 1);
    assert_eq!(lsn3, 2);

    // Sync to storage
    wal.sync().await.unwrap();

    // Read back
    let entries = wal.read_from(0).await.unwrap();
    assert_eq!(entries.len(), 3);

    match &entries[0].1 {
        WalEntry::Upsert { points } => {
            assert_eq!(points.len(), 1);
            assert_eq!(points[0].id, 1);
        }
        _ => panic!("Expected Upsert"),
    }

    match &entries[2].1 {
        WalEntry::Delete { ids } => {
            assert_eq!(ids, &vec![1]);
        }
        _ => panic!("Expected Delete"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
async fn test_wal_checkpoint() {
    let storage = MockStorage::new();
    let mut wal = WriteAheadLog::open(storage).await.unwrap();

    // Append entry
    wal.append(&WalEntry::Upsert {
        points: vec![Point::new_vector(1, vec![0.1])],
    })
    .await
    .unwrap();
    wal.sync().await.unwrap();

    // Checkpoint
    wal.checkpoint(0).await.unwrap();

    assert_eq!(wal.last_checkpoint_lsn(), 0);
    assert!(!wal.should_checkpoint());
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
async fn test_wal_recovery() {
    let storage = MockStorage::new();

    // Write some entries
    {
        let mut wal = WriteAheadLog::open(MockStorage {
            pages: storage.pages.clone(),
            meta: storage.meta.clone(),
        })
        .await
        .unwrap();

        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(1, vec![0.1])],
        })
        .await
        .unwrap();
        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(2, vec![0.2])],
        })
        .await
        .unwrap();
        wal.sync().await.unwrap();
    }

    // Reopen and read back (simulates recovery)
    {
        let wal = WriteAheadLog::open(MockStorage {
            pages: storage.pages.clone(),
            meta: storage.meta.clone(),
        })
        .await
        .unwrap();

        let entries = wal.read_from(0).await.unwrap();
        assert_eq!(entries.len(), 2);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
async fn test_wal_read_from_specific_lsn() {
    let storage = MockStorage::new();
    let mut wal: WriteAheadLog<MockStorage> = WriteAheadLog::open(storage).await.unwrap();

    // Append 10 entries
    for i in 0..10 {
        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(i, vec![i as f32])],
        })
        .await
        .unwrap();
    }
    wal.sync().await.unwrap();

    // Read from LSN 5 onwards
    let entries = wal.read_from(5).await.unwrap();
    assert_eq!(entries.len(), 5);
    assert_eq!(entries[0].0, 5);
    assert_eq!(entries[4].0, 9);
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
async fn test_wal_entries_since_checkpoint_tracking() {
    let storage = MockStorage::new();
    let mut wal: WriteAheadLog<MockStorage> = WriteAheadLog::open(storage).await.unwrap();

    // Append entries
    for i in 0..5 {
        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(i, vec![i as f32])],
        })
        .await
        .unwrap();
    }
    wal.sync().await.unwrap();

    // Should have 5 entries since last checkpoint
    assert_eq!(wal.header.entries_since_checkpoint, 5);

    // Checkpoint
    wal.checkpoint(4).await.unwrap();

    // Should reset to 1 (the checkpoint entry itself)
    assert_eq!(wal.header.entries_since_checkpoint, 0);

    // Add more entries
    for i in 5..8 {
        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(i, vec![i as f32])],
        })
        .await
        .unwrap();
    }
    wal.sync().await.unwrap();

    // Should have 3 new entries since checkpoint
    assert_eq!(wal.header.entries_since_checkpoint, 3);
}
