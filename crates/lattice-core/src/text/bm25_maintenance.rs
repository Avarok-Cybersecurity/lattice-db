//! Bookkeeping for [`LexicalIndex`]: reclaiming dead document slots and
//! deriving per-document term lists (which a restored index does not carry).

use super::LexicalIndex;
use crate::error::LatticeError;
use crate::types::point::PointId;

impl LexicalIndex {
    /// Drop dead slots and renumber live documents by ascending point id.
    pub(super) fn compact(&mut self) {
        let mut live: Vec<(PointId, u32)> = self.by_id.iter().map(|(&id, &o)| (id, o)).collect();
        live.sort_unstable();
        let mut remap = vec![u32::MAX; self.docs.len()];
        for (new, (_, old)) in live.iter().enumerate() {
            remap[*old as usize] = new as u32;
        }
        for list in &mut self.postings {
            list.retain_mut(|p| {
                p.ord = remap[p.ord as usize];
                p.ord != u32::MAX
            });
            list.sort_unstable_by_key(|p| p.ord);
        }
        let docs = std::mem::take(&mut self.docs);
        let mut terms = self.doc_terms.take();
        self.docs = live
            .iter()
            .map(|(_, o)| docs[*o as usize].clone())
            .collect();
        self.doc_terms = terms.as_mut().map(|t| {
            live.iter()
                .map(|(_, o)| std::mem::take(&mut t[*o as usize]))
                .collect()
        });
        self.by_id = live
            .iter()
            .enumerate()
            .map(|(new, (id, _))| (*id, new as u32))
            .collect();
    }

    /// Per-ordinal term lists, derived from the postings on first use after a restore.
    pub(super) fn terms_by_doc(&mut self) -> &mut Vec<Vec<u32>> {
        let docs = self.docs.len();
        let postings = &self.postings;
        self.doc_terms.get_or_insert_with(|| {
            let mut terms = vec![Vec::new(); docs];
            for (term, list) in postings.iter().enumerate() {
                for p in list {
                    terms[p.ord as usize].push(term as u32);
                }
            }
            terms
        })
    }
}

pub(super) fn too_large(what: &str) -> LatticeError {
    LatticeError::InvalidOperation {
        message: format!("text {what} exceeds the BM25 index's 2^32 limit"),
    }
}
