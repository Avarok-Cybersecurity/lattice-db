//! Reading the log back for recovery.

use super::{record, Lsn, WalEntry, WriteAheadLog};
use crate::error::{LatticeError, LatticeResult};
use crate::storage::{LatticeStorage, StorageError};

/// Read a page, treating a page that was never written as empty
pub(super) async fn read_page_or_empty<S: LatticeStorage>(
    storage: &S,
    page_id: u64,
) -> LatticeResult<Vec<u8>> {
    match storage.read_page(page_id).await {
        Ok(page) => Ok(page),
        Err(StorageError::PageNotFound { .. }) => Ok(Vec::new()),
        Err(e) => Err(e.into()),
    }
}

/// Keep only the committed prefix of the current page
///
/// Bytes past `committed` were written by a sync whose header never landed,
/// so they were never acknowledged.
pub(super) fn committed_prefix(
    mut page: Vec<u8>,
    page_id: u64,
    committed: u64,
) -> LatticeResult<Vec<u8>> {
    if (page.len() as u64) < committed {
        return Err(LatticeError::DataCorruption {
            code: 60001,
            context: format!(
                "WAL page {} holds {} bytes but the header records {} committed",
                page_id,
                page.len(),
                committed
            ),
        });
    }
    page.truncate(committed as usize);
    Ok(page)
}

impl<S: LatticeStorage> WriteAheadLog<S> {
    /// Read all entries from a given LSN
    ///
    /// Used during recovery to replay the WAL. Walks the pages the durable
    /// header lists, numbering each page's entries from the first LSN the
    /// header records for it, so pages removed by truncation do not shift the
    /// numbering. Pages holding only entries before `start_lsn` are not read.
    /// The current page is bounded by the header's committed length.
    pub async fn read_from(&self, start_lsn: Lsn) -> LatticeResult<Vec<(Lsn, WalEntry)>> {
        let durable = &self.synced_header;
        let pages = &durable.page_first_lsn;
        let mut entries = Vec::new();

        for (i, &(page_id, first_lsn)) in pages.iter().enumerate() {
            if let Some(&(_, next_first_lsn)) = pages.get(i + 1) {
                if next_first_lsn <= start_lsn || next_first_lsn == first_lsn {
                    continue;
                }
            }

            let page = if page_id == durable.current_page_id {
                if durable.current_page_bytes == 0 {
                    continue;
                }
                let page = read_page_or_empty(&self.storage, page_id).await?;
                committed_prefix(page, page_id, durable.current_page_bytes)?
            } else {
                self.read_listed_page(page_id, first_lsn).await?
            };

            let mut current_lsn = first_lsn;
            let page_entries = record::parse_page_entries(&page, start_lsn, &mut current_lsn)?;
            entries.extend(page_entries);
        }

        Ok(entries)
    }

    /// Read a full page the header lists; its absence means lost entries
    async fn read_listed_page(&self, page_id: u64, first_lsn: Lsn) -> LatticeResult<Vec<u8>> {
        match self.storage.read_page(page_id).await {
            Ok(page) => Ok(page),
            Err(StorageError::PageNotFound { .. }) => Err(LatticeError::DataCorruption {
                code: 60002,
                context: format!(
                    "WAL page {} holding entries from LSN {} is missing",
                    page_id, first_lsn
                ),
            }),
            Err(e) => Err(e.into()),
        }
    }
}
