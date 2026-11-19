//! BM25 inverted index (Okapi BM25 with the Lucene non-negative IDF).
//!
//! For a query with term weights `q_t` (how often `t` occurs in the query):
//!
//! ```text
//! score(D) = Σ_t  q_t · idf(t) · tf(t,D)·(k1+1) / (tf(t,D) + k1·(1 − b + b·|D|/avgdl))
//! idf(t)   = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))
//! ```
//!
//! The `1 +` inside the logarithm (as in Lucene's `BM25Similarity`) keeps IDF
//! positive for terms present in more than half the corpus, so adding a common
//! word to a query can never lower a document's score.
//!
//! Results are ordered by score descending, then point id ascending, so equal
//! scores never produce run-to-run differences.

#[path = "bm25_maintenance.rs"]
mod maintenance;
#[path = "bm25_parts.rs"]
mod parts;

use super::sparse::SparseVector;
use super::tokenizer::tokenize;
use crate::error::LatticeResult;
use crate::types::point::PointId;
use maintenance::too_large;
pub use parts::LexicalParts;
use rustc_hash::FxHashMap;
use std::collections::HashMap;

/// BM25 parameters. See [`super::config::BM25_K1`] / [`super::config::BM25_B`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
}

/// One scored document.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LexicalHit {
    pub id: PointId,
    pub score: f32,
}

/// One posting: a document ordinal and the term's frequency in it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Posting {
    ord: u32,
    tf: u32,
}

/// A document slot; removal tombstones it until compaction reclaims it.
#[derive(Debug, Clone)]
struct Doc {
    id: PointId,
    len: u32,
    live: bool,
}

/// Dead slots tolerated before compaction is considered.
const COMPACT_MIN_DEAD: usize = 1024;

/// Inverted index over documents addressed by dense ordinals.
///
/// Postings are append-only vectors in ordinal order, so indexing a document
/// is `O(distinct terms)`, removing one is `O(1)` plus a df update, and a
/// snapshot restores by copying arrays instead of rebuilding hash maps (see
/// `bm25_parts.rs`).
#[derive(Debug, Clone)]
pub struct LexicalIndex {
    params: Bm25Params,
    /// Term -> stable term id. Ids are never reused, so a sparse query vector
    /// built from this vocabulary stays valid while documents come and go.
    vocab: HashMap<String, u32>,
    /// Postings by term id, ordinals ascending (may include dead slots).
    postings: Vec<Vec<Posting>>,
    /// Live documents containing each term.
    df: Vec<u32>,
    /// Document slots by ordinal.
    docs: Vec<Doc>,
    /// Live point id -> ordinal.
    by_id: FxHashMap<PointId, u32>,
    /// Distinct term ids per ordinal, needed to update `df` on removal.
    /// `None` after a restore until the first removal derives them.
    doc_terms: Option<Vec<Vec<u32>>>,
    /// Token count of all live documents.
    live_len: u64,
}

impl LexicalIndex {
    pub fn new(params: Bm25Params) -> Self {
        Self {
            params,
            vocab: HashMap::new(),
            postings: Vec::new(),
            df: Vec::new(),
            docs: Vec::new(),
            by_id: FxHashMap::default(),
            doc_terms: Some(Vec::new()),
            live_len: 0,
        }
    }

    /// Number of indexed (non-empty) documents.
    pub fn doc_count(&self) -> usize {
        self.by_id.len()
    }

    /// Number of distinct terms ever seen.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn contains(&self, id: PointId) -> bool {
        self.by_id.contains_key(&id)
    }

    /// Index (or re-index) one document. Text without tokens leaves the
    /// document out of the index entirely.
    pub fn index_document(&mut self, id: PointId, text: &str) -> LatticeResult<()> {
        self.remove_document(id);
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return Ok(());
        }
        let len = u32::try_from(tokens.len()).map_err(|_| too_large("document"))?;
        if self.docs.len() >= u32::MAX as usize {
            self.compact();
        }
        let ord = u32::try_from(self.docs.len()).map_err(|_| too_large("index"))?;
        let mut tf: FxHashMap<u32, u32> = FxHashMap::default();
        for token in &tokens {
            *tf.entry(self.term_id_or_insert(token)).or_insert(0) += 1;
        }
        let mut term_ids: Vec<u32> = tf.keys().copied().collect();
        term_ids.sort_unstable();
        for &term in &term_ids {
            self.postings[term as usize].push(Posting { ord, tf: tf[&term] });
            self.df[term as usize] += 1;
        }
        self.docs.push(Doc {
            id,
            len,
            live: true,
        });
        self.terms_by_doc().push(term_ids);
        self.by_id.insert(id, ord);
        self.live_len += u64::from(len);
        Ok(())
    }

    /// Remove a document. Returns whether it was indexed.
    pub fn remove_document(&mut self, id: PointId) -> bool {
        let Some(ord) = self.by_id.remove(&id) else {
            return false;
        };
        let ord = ord as usize;
        self.docs[ord].live = false;
        self.live_len -= u64::from(self.docs[ord].len);
        let terms = std::mem::take(&mut self.terms_by_doc()[ord]);
        for term in terms {
            self.df[term as usize] -= 1;
        }
        let dead = self.docs.len() - self.by_id.len();
        if dead >= COMPACT_MIN_DEAD && dead > self.by_id.len() {
            self.compact();
        }
        true
    }

    /// The sparse query vector for `text`: term id -> occurrences in the query.
    /// Terms absent from the vocabulary cannot match anything and are dropped.
    pub fn query_vector(&self, text: &str) -> SparseVector {
        let mut weights: FxHashMap<u32, f32> = FxHashMap::default();
        for token in tokenize(text) {
            if let Some(&term) = self.vocab.get(&token) {
                *weights.entry(term).or_insert(0.0) += 1.0;
            }
        }
        SparseVector::from_unique_counts(weights.into_iter().collect())
    }

    /// Inverse document frequency of a term id.
    pub fn idf(&self, term: u32) -> f32 {
        self.idf_f64(term) as f32
    }

    fn idf_f64(&self, term: u32) -> f64 {
        let n = self.by_id.len() as f64;
        let df = f64::from(self.df.get(term as usize).copied().unwrap_or(0));
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    /// Score `text` and return the best `limit` documents.
    pub fn search(&self, text: &str, limit: usize) -> Vec<LexicalHit> {
        self.search_sparse(&self.query_vector(text), limit)
    }

    /// Score a sparse query over this index's vocabulary.
    pub fn search_sparse(&self, query: &SparseVector, limit: usize) -> Vec<LexicalHit> {
        if limit == 0 || self.by_id.is_empty() {
            return Vec::new();
        }
        let avgdl = self.live_len as f64 / self.by_id.len() as f64;
        let (k1, b) = (f64::from(self.params.k1), f64::from(self.params.b));
        let mut scores = vec![0.0f64; self.docs.len()];
        let mut touched: Vec<u32> = Vec::new();
        for (term, weight) in query.iter() {
            let Some(postings) = self.postings.get(term as usize) else {
                continue;
            };
            let idf = self.idf_f64(term);
            let weight = f64::from(weight);
            for p in postings {
                let doc = &self.docs[p.ord as usize];
                if !doc.live {
                    continue;
                }
                let (dl, tf) = (f64::from(doc.len), f64::from(p.tf));
                let norm = tf + k1 * (1.0 - b + b * dl / avgdl);
                let slot = &mut scores[p.ord as usize];
                if *slot == 0.0 {
                    touched.push(p.ord);
                }
                *slot += weight * idf * tf * (k1 + 1.0) / norm;
            }
        }
        let mut hits: Vec<LexicalHit> = touched
            .into_iter()
            .map(|ord| LexicalHit {
                id: self.docs[ord as usize].id,
                score: scores[ord as usize] as f32,
            })
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
        hits.truncate(limit);
        hits
    }

    fn term_id_or_insert(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.vocab.get(token) {
            return id;
        }
        let id = self.postings.len() as u32;
        self.vocab.insert(token.to_string(), id);
        self.postings.push(Vec::new());
        self.df.push(0);
        id
    }
}
