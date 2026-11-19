//! Persisting a BM25 index so a snapshot loads without re-tokenizing.
//!
//! The stored form is columnar (terms, postings, document lengths); the
//! per-document term lists and the corpus length are derived on restore
//! rather than stored twice. Parts come from untrusted snapshot bytes, so
//! [`LexicalIndex::from_parts`] validates every reference.

use super::{Bm25Params, LexicalIndex};
use crate::error::{LatticeError, LatticeResult};
use crate::types::point::PointId;
use rkyv::{Archive, Deserialize, Serialize};
use rustc_hash::FxHashMap;
use std::collections::HashMap;

/// Documents containing one term, with the term's frequency in each.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PostingList {
    pub ids: Vec<PointId>,
    pub tfs: Vec<u32>,
}

/// A [`LexicalIndex`] without its BM25 parameters (those come from the
/// collection's `text_index` configuration).
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LexicalParts {
    /// Term strings; the position is the term id.
    pub terms: Vec<String>,
    /// `postings[term id]`, ids ascending.
    pub postings: Vec<PostingList>,
    /// Indexed documents, ids ascending, with their token counts.
    pub doc_ids: Vec<PointId>,
    pub doc_lens: Vec<u32>,
}

impl LexicalIndex {
    /// Capture the index for persistence (deterministic order).
    pub fn to_parts(&self) -> LexicalParts {
        let mut terms = vec![String::new(); self.vocab.len()];
        for (term, &id) in &self.vocab {
            terms[id as usize] = term.clone();
        }
        let postings = self
            .postings
            .iter()
            .map(|p| {
                let mut entries: Vec<(PointId, u32)> = p.iter().map(|(&i, &tf)| (i, tf)).collect();
                entries.sort_unstable_by_key(|e| e.0);
                PostingList {
                    ids: entries.iter().map(|e| e.0).collect(),
                    tfs: entries.iter().map(|e| e.1).collect(),
                }
            })
            .collect();
        let mut docs: Vec<(PointId, u32)> =
            self.docs.iter().map(|(&i, (len, _))| (i, *len)).collect();
        docs.sort_unstable_by_key(|d| d.0);
        LexicalParts {
            terms,
            postings,
            doc_ids: docs.iter().map(|d| d.0).collect(),
            doc_lens: docs.iter().map(|d| d.1).collect(),
        }
    }

    /// Rebuild an index from persisted parts.
    pub fn from_parts(params: Bm25Params, parts: LexicalParts) -> LatticeResult<Self> {
        if parts.terms.len() != parts.postings.len() || parts.doc_ids.len() != parts.doc_lens.len()
        {
            return Err(corrupt("section lengths disagree"));
        }
        let mut docs: FxHashMap<PointId, (u32, Vec<u32>)> =
            FxHashMap::with_capacity_and_hasher(parts.doc_ids.len(), Default::default());
        let mut total_len = 0u64;
        for (&id, &len) in parts.doc_ids.iter().zip(&parts.doc_lens) {
            if docs.insert(id, (len, Vec::new())).is_some() {
                return Err(corrupt("duplicate document"));
            }
            total_len += u64::from(len);
        }
        let mut vocab = HashMap::with_capacity(parts.terms.len());
        for (term_id, term) in parts.terms.into_iter().enumerate() {
            if vocab.insert(term, term_id as u32).is_some() {
                return Err(corrupt("duplicate term"));
            }
        }
        let mut postings = Vec::with_capacity(parts.postings.len());
        for (term_id, list) in parts.postings.into_iter().enumerate() {
            if list.ids.len() != list.tfs.len() {
                return Err(corrupt("posting list lengths disagree"));
            }
            let mut map = FxHashMap::with_capacity_and_hasher(list.ids.len(), Default::default());
            for (id, tf) in list.ids.into_iter().zip(list.tfs) {
                let doc = docs
                    .get_mut(&id)
                    .ok_or_else(|| corrupt("posting names an unknown document"))?;
                if tf == 0 || tf > doc.0 || map.insert(id, tf).is_some() {
                    return Err(corrupt("posting frequency is inconsistent"));
                }
                doc.1.push(term_id as u32);
            }
            postings.push(map);
        }
        Ok(Self {
            params,
            vocab,
            postings,
            docs,
            total_len,
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
        index.index_document(3, "the quick brown fox");
        index.index_document(1, "a lazy dog and a quick cat");
        index.index_document(2, "fox fox fox");
        index.index_document(4, "temporary");
        index.remove_document(4);
        index
    }

    #[test]
    fn restored_index_scores_identically_and_stays_mutable() {
        let index = sample();
        let mut restored = LexicalIndex::from_parts(PARAMS, index.to_parts()).unwrap();
        assert_eq!(restored.to_parts(), index.to_parts());
        for q in ["fox", "quick dog", "temporary", "absent"] {
            assert_eq!(restored.search(q, 10), index.search(q, 10), "{q}");
        }
        restored.remove_document(2);
        restored.index_document(5, "fox");
        assert_eq!(restored.search("fox", 10).len(), 2);
    }

    #[test]
    fn inconsistent_parts_are_rejected() {
        let good = sample().to_parts();
        let mut unknown = good.clone();
        unknown.postings[0].ids[0] = 99;
        assert!(LexicalIndex::from_parts(PARAMS, unknown).is_err());
        let mut lens = good.clone();
        lens.doc_lens.pop();
        assert!(LexicalIndex::from_parts(PARAMS, lens).is_err());
        let mut dup = good.clone();
        dup.terms[1] = dup.terms[0].clone();
        assert!(LexicalIndex::from_parts(PARAMS, dup).is_err());
        let mut tf = good;
        tf.postings[0].tfs[0] = 0;
        assert!(LexicalIndex::from_parts(PARAMS, tf).is_err());
    }
}
