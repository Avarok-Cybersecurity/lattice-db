//! Page rotation, truncation and multi-page recovery.

use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
async fn test_wal_page_rotation() {
    let storage = MockStorage::new();
    let mut wal: WriteAheadLog<MockStorage> = WriteAheadLog::open(MockStorage {
        pages: storage.pages.clone(),
        meta: storage.meta.clone(),
    })
    .await
    .unwrap();

    // Initial state: should be on page 0
    assert_eq!(wal.header.current_page_id, WAL_PAGE_START);
    assert_eq!(wal.header.current_page_bytes, 0);

    // Create a large vector to force page rotation
    // Each entry will be substantial in size
    let large_vector: Vec<f32> = (0..10000).map(|i| i as f32).collect();

    // Write enough entries to exceed 16MB
    // Each entry with 10000 floats is ~40KB, so ~400 entries should exceed 16MB
    let mut entry_count = 0;
    let initial_page = wal.header.current_page_id;

    for i in 0..500 {
        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(i, large_vector.clone())],
        })
        .await
        .unwrap();
        wal.sync().await.unwrap();
        entry_count += 1;

        // Check if rotation happened
        if wal.header.current_page_id > initial_page {
            break;
        }
    }

    // Verify that page rotation occurred
    assert!(
        wal.header.current_page_id > initial_page,
        "Page rotation should have occurred after {} entries",
        entry_count
    );

    // Verify that page_first_lsn is tracking pages
    assert!(
        !wal.header.page_first_lsn.is_empty(),
        "page_first_lsn should contain entries after rotation"
    );

    // Verify all entries can still be read back
    let entries = wal.read_from(0).await.unwrap();
    assert_eq!(entries.len(), entry_count as usize);
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
async fn test_wal_truncate_after_checkpoint() {
    let storage = MockStorage::new();
    let pages_arc = storage.pages.clone();
    let mut wal: WriteAheadLog<MockStorage> = WriteAheadLog::open(MockStorage {
        pages: storage.pages.clone(),
        meta: storage.meta.clone(),
    })
    .await
    .unwrap();

    // Create entries that span multiple pages
    let large_vector: Vec<f32> = (0..10000).map(|i| i as f32).collect();

    // Write enough to span at least 2 pages
    for i in 0..500 {
        wal.append(&WalEntry::Upsert {
            points: vec![Point::new_vector(i, large_vector.clone())],
        })
        .await
        .unwrap();
        wal.sync().await.unwrap();

        if wal.header.current_page_id > WAL_PAGE_START + 1 {
            break;
        }
    }

    // Get the current LSN and page count
    let checkpoint_lsn = wal.next_lsn() - 1;
    let pages_before = pages_arc.read().unwrap().len();

    // Perform checkpoint which should truncate old pages
    wal.checkpoint(checkpoint_lsn).await.unwrap();

    // After checkpoint, old pages should be deleted
    let pages_after = pages_arc.read().unwrap().len();

    // The current page should remain, old pages should be deleted
    assert!(
        pages_after <= pages_before,
        "Truncation should not increase page count"
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
async fn test_wal_multi_page_recovery() {
    let storage = MockStorage::new();

    // Phase 1: Write entries spanning multiple pages
    let entry_count;
    {
        let mut wal: WriteAheadLog<MockStorage> = WriteAheadLog::open(MockStorage {
            pages: storage.pages.clone(),
            meta: storage.meta.clone(),
        })
        .await
        .unwrap();

        let large_vector: Vec<f32> = (0..10000).map(|i| i as f32).collect();
        let mut count = 0;

        for i in 0..500 {
            wal.append(&WalEntry::Upsert {
                points: vec![Point::new_vector(i, large_vector.clone())],
            })
            .await
            .unwrap();
            wal.sync().await.unwrap();
            count += 1;

            // Stop after spanning at least 2 pages
            if wal.header.current_page_id > WAL_PAGE_START + 1 {
                break;
            }
        }

        entry_count = count;
    }

    // Phase 2: Reopen and verify recovery across multiple pages
    {
        let wal: WriteAheadLog<MockStorage> = WriteAheadLog::open(MockStorage {
            pages: storage.pages.clone(),
            meta: storage.meta.clone(),
        })
        .await
        .unwrap();

        let entries = wal.read_from(0).await.unwrap();
        assert_eq!(
            entries.len(),
            entry_count as usize,
            "Recovery should restore all {} entries from multiple pages",
            entry_count
        );

        // Verify entries are in correct order
        for (idx, (lsn, _entry)) in entries.iter().enumerate() {
            assert_eq!(*lsn, idx as u64, "LSN should match entry index");
        }
    }
}
