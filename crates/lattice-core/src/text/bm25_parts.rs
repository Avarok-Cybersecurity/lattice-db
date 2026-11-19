//! Persisting a BM25 index so a snapshot loads without re-tokenizing.
//!
//! The stored form is columnar and compact: postings are flat `u32` arrays
//! (document ordinal, term frequency) sliced per term by `offsets`, and
//! documents are numbered by ascending point id. Restoring copies arrays; no
//! text is tokenized and no hash map of postings is built. The parts come
//! from untrusted snapshot bytes, so [`LexicalIndex::from_parts`] validates
//! every reference before any of it is used.

use super::{Bm25Params, Doc, LexicalIndex, Posting};
use crate::error::{LatticeError, LatticeResult};
use crate::types::point::PointId;
use rkyv::{Archive, Deserialize, Serialize};
use rustc_hash::FxHashMap;
use std::collections::HashMap;

/// A [`LexicalIndex`] without its BM25 parameters (those come from the
/// collection's `text_index` configuration).
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LexicalParts {
    /// Term strings; the position is the term id.
    pub terms: Vec<String>,
    /// `offsets[t]..offsets[t + 1]` slices `ords`/`tfs` for term `t`.
    pub offsets: Vec<u32>,
    /// Document ordinals (indexes into `doc_ids`), ascending within a term.
    pub ords: Vec<u32>,
    pub tfs: Vec<u32>,
    /// Point id of each ordinal, ascending.
    pub doc_ids: Vec<PointId>,
    /// Token count of each ordinal.
    pub doc_lens: Vec<u32>,
}

impl LexicalIndex {
    /// Capture the live documents in canonical form: documents by ascending
    /// point id, terms in ascending order, and only terms some live document
    /// still holds. Term ids are assigned in first-seen order and survive
    /// deletions, so this keeps the parts a function of the indexed content,
    /// not of the order it was indexed in.
    pub fn to_parts(&self) -> LexicalParts {
        let mut live: Vec<(PointId, u32)> = self.by_id.iter().map(|(&id, &o)| (id, o)).collect();
        live.sort_unstable();
        let mut remap = vec![u32::MAX; self.docs.len()];
        for (new, (_, old)) in live.iter().enumerate() {
            remap[*old as usize] = new as u32;
        }
        let mut by_term: Vec<(&str, Vec<(u32, u32)>)> = self
            .vocab
            .iter()
            .filter_map(|(term, &id)| {
                let mut entries: Vec<(u32, u32)> = self.postings[id as usize]
                    .iter()
                    .map(|p| (remap[p.ord as usize], p.tf))
                    .filter(|(ord, _)| *ord != u32::MAX)
                    .collect();
                entries.sort_unstable();
                (!entries.is_empty()).then_some((term.as_str(), entries))
            })
            .collect();
        by_term.sort_unstable_by(|a, b| a.0.cmp(b.0));
        let (mut offsets, mut ords, mut tfs) = (vec![0u32], Vec::new(), Vec::new());
        for (_, entries) in &by_term {
            ords.extend(entries.iter().map(|e| e.0));
            tfs.extend(entries.iter().map(|e| e.1));
            offsets.push(ords.len() as u32);
        }
        LexicalParts {
            terms: by_term.iter().map(|(t, _)| (*t).to_string()).collect(),
            offsets,
            ords,
            tfs,
            doc_ids: live.iter().map(|(id, _)| *id).collect(),
            doc_lens: live
                .iter()
                .map(|(_, o)| self.docs[*o as usize].len)
                .collect(),
        }
    }

    /// Rebuild an index from persisted parts.
    pub fn from_parts(params: Bm25Params, parts: LexicalParts) -> LatticeResult<Self> {
        let n_docs = parts.doc_ids.len();
        if parts.doc_lens.len() != n_docs
            || parts.ords.len() != parts.tfs.len()
            || parts.offsets.len() != parts.terms.len() + 1
            || parts.offsets.first() != Some(&0)
            || parts.offsets.last().map(|&o| o as usize) != Some(parts.ords.len())
            || parts.offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err(corrupt("section lengths disagree"));
        }
        if parts.doc_ids.windows(2).any(|w| w[0] >= w[1]) {
            return Err(corrupt("document ids are not strictly ascending"));
        }
        // Every document's term frequencies must add up to its length.
        let mut token_sums = vec![0u64; n_docs];
        let mut postings = Vec::with_capacity(parts.terms.len());
        let mut df = Vec::with_capacity(parts.terms.len());
        for w in parts.offsets.windows(2) {
            let range = w[0] as usize..w[1] as usize;
            let (ords, tfs) = (&parts.ords[range.clone()], &parts.tfs[range]);
            if ords.windows(2).any(|p| p[0] >= p[1]) {
                return Err(corrupt("posting list is not strictly ascending"));
            }
            let mut list = Vec::with_capacity(ords.len());
            for (&ord, &tf) in ords.iter().zip(tfs) {
                let sum = token_sums
                    .get_mut(ord as usize)
                    .ok_or_else(|| corrupt("posting names an unknown document"))?;
                if tf == 0 {
                    return Err(corrupt("zero term frequency"));
                }
                *sum += u64::from(tf);
                list.push(Posting { ord, tf });
            }
            df.push(list.len() as u32);
            postings.push(list);
        }
        if token_sums
            .iter()
            .zip(&parts.doc_lens)
            .any(|(sum, len)| *sum != u64::from(*len) || *len == 0)
        {
            return Err(corrupt("document length disagrees with its postings"));
        }
        let mut vocab = HashMap::with_capacity(parts.terms.len());
        for (term_id, term) in parts.terms.into_iter().enumerate() {
            if vocab.insert(term, term_id as u32).is_some() {
                return Err(corrupt("duplicate term"));
            }
        }
        let by_id: FxHashMap<PointId, u32> = parts
            .doc_ids
            .iter()
            .enumerate()
            .map(|(ord, id)| (*id, ord as u32))
            .collect();
        let docs = parts
            .doc_ids
            .iter()
            .zip(&parts.doc_lens)
            .map(|(&id, &len)| Doc {
                id,
                len,
                live: true,
            })
            .collect();
        Ok(Self {
            params,
            vocab,
            postings,
            df,
            docs,
            by_id,
            doc_terms: None,
            live_len: parts.doc_lens.iter().map(|l| u64::from(*l)).sum(),
        })
    }
}

fn corrupt(message: &str) -> LatticeError {
    LatticeError::Serialization {
        message: format!("Invalid snapshot text index: {message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    const PARAMS: Bm25Params = Bm25Params { k1: 1.2, b: 0.75 };

    fn sample() -> LexicalIndex {
        let mut index = LexicalIndex::new(PARAMS);
        for (id, text) in [
            (3, "the quick brown fox"),
            (1, "a lazy dog and a quick cat"),
            (2, "fox fox fox"),
            (4, "temporary"),
        ] {
            index.index_document(id, text).unwrap();
        }
        index.remove_document(4);
        index
            .index_document(1, "a lazy dog and a quick quick cat")
            .unwrap();
        index
    }

    #[test]
    fn restored_index_scores_identically_and_stays_mutable() {
        let index = sample();
        let mut restored = LexicalIndex::from_parts(PARAMS, index.to_parts()).unwrap();
        assert_eq!(restored.to_parts(), index.to_parts());
        for q in ["fox", "quick dog", "temporary", "absent", "cat lazy"] {
            assert_eq!(restored.search(q, 10), index.search(q, 10), "{q}");
        }
        restored.remove_document(2);
        restored.index_document(5, "fox").unwrap();
        let ids: Vec<PointId> = restored.search("fox", 10).iter().map(|h| h.id).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&5) && ids.contains(&3));
        assert_eq!(restored.doc_count(), 3);
    }

    #[test]
    fn inconsistent_parts_are_rejected() {
        let good = sample().to_parts();
        let reject = |f: &dyn Fn(&mut LexicalParts)| {
            let mut parts = good.clone();
            f(&mut parts);
            assert!(LexicalIndex::from_parts(PARAMS, parts).is_err());
        };
        reject(&|p| p.ords[0] = 99);
        reject(&|p| {
            p.doc_lens.pop();
        });
        reject(&|p| p.terms[1] = p.terms[0].clone());
        reject(&|p| p.tfs[0] = 0);
        reject(&|p| p.doc_lens[0] += 1);
        reject(&|p| p.doc_ids.swap(0, 1));
        reject(&|p| *p.offsets.last_mut().unwrap() += 1);
        assert!(LexicalIndex::from_parts(PARAMS, good).is_ok());
    }
}
