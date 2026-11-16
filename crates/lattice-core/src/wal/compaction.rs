//! Checkpoint markers and truncation of the log they make redundant.

use super::{header, Lsn, WalEntry, WriteAheadLog, WAL_HEADER_KEY};
use crate::error::{LatticeError, LatticeResult};
use crate::storage::{LatticeStorage, StorageError};

impl<S: LatticeStorage> WriteAheadLog<S> {
    /// Record a checkpoint
    ///
    /// This should be called after successfully saving a snapshot.
    pub async fn checkpoint(&mut self, snapshot_lsn: Lsn) -> LatticeResult<()> {
        // Write checkpoint entry
        self.append(&WalEntry::Checkpoint { snapshot_lsn }).await?;

        // Update header
        self.header.last_checkpoint_lsn = snapshot_lsn;
        self.header.entries_since_checkpoint = 0;

        // Sync to ensure checkpoint is durable
        self.sync().await?;

        Ok(())
    }

    /// Truncate WAL entries before the given LSN
    ///
    /// This reclaims disk space after a successful checkpoint by deleting
    /// pages that contain only entries before the given LSN.
    ///
    /// Safety: Only entries before `last_checkpoint_lsn` can be truncated.
    pub async fn truncate_before(&mut self, lsn: Lsn) -> LatticeResult<()> {
        // Safety: only truncate entries before checkpoint
        if lsn > self.header.last_checkpoint_lsn {
            return Err(LatticeError::InvalidOperation {
                message: format!(
                    "Cannot truncate LSN {} beyond checkpoint {}",
                    lsn, self.header.last_checkpoint_lsn
                ),
            });
        }

        // Find pages entirely before the LSN
        let mut pages_to_delete = Vec::new();
        let mut new_page_first_lsn = Vec::new();

        for i in 0..self.header.page_first_lsn.len() {
            let (page_id, first_lsn) = self.header.page_first_lsn[i];

            // Check if this page's first LSN is before our target
            if first_lsn < lsn {
                // Check if the ENTIRE page is before LSN by looking at next page's first LSN
                let next_page_first_lsn = if i + 1 < self.header.page_first_lsn.len() {
                    Some(self.header.page_first_lsn[i + 1].1)
                } else {
                    // This is the current/last page, don't delete it
                    None
                };

                if let Some(next_lsn) = next_page_first_lsn {
                    if next_lsn <= lsn {
                        // Entire page is before target LSN, safe to delete
                        pages_to_delete.push(page_id);
                        continue;
                    }
                }
            }
            // Keep this page
            new_page_first_lsn.push((page_id, first_lsn));
        }

        if pages_to_delete.is_empty() {
            return Ok(());
        }

        // Persist the shortened page list before deleting anything: a crash in
        // between leaves orphaned pages, never a header naming deleted ones.
        let mut durable = self.synced_header.clone();
        durable.page_first_lsn = new_page_first_lsn;
        let header_bytes = header::serialize_header(&durable);
        self.storage.set_meta(WAL_HEADER_KEY, &header_bytes).await?;
        self.storage.sync().await?;
        self.header.page_first_lsn = durable.page_first_lsn.clone();
        self.synced_header = durable;

        // Delete old pages
        for page_id in pages_to_delete {
            match self.storage.delete_page(page_id).await {
                Ok(()) | Err(StorageError::PageNotFound { .. }) => {}
                Err(e) => return Err(e.into()),
            }
            #[cfg(not(target_arch = "wasm32"))]
            tracing::info!(page_id, "Deleted WAL page after truncation");
        }

        Ok(())
    }

    /// Get the current page ID (for testing/debugging)
    #[cfg(test)]
    pub fn current_page_id(&self) -> u64 {
        self.header.current_page_id
    }

    /// Get the current page bytes (for testing/debugging)
    #[cfg(test)]
    pub fn current_page_bytes(&self) -> u64 {
        self.header.current_page_bytes
    }
}
