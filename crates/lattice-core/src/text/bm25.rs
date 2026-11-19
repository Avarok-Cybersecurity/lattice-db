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

use super::sparse::SparseVector;
use super::tokenizer::tokenize;
use crate::types::point::PointId;
use rustc_hash::FxHashMap;
use std::collections::HashMap;

/// BM25 parameters. See [`super::config::BM25_K1`] / [`super::config::BM25_B`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
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

/// Inverted index: term id -> (point id -> term frequency).
///
/// Not persisted: every load path (snapshot, WAL replay, JSONL import) rebuilds
/// it from the payloads. Measured on 10K docs (`text/tests/perf.rs`): rebuild
/// ~75 ms vs ~28 ms to deserialize a JSON copy — but that copy is ~7.7 MB, more
/// than twice the indexed text, and would change the snapshot format. For
/// download-bound browser/edge loads, rebuilding is the better trade. The serde
/// derive exists only for that measurement.
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub struct LexicalIndex {
    params: Bm25Params,
    /// Term -> stable term id. Ids are never reused, so a sparse query vector
    /// built from this vocabulary stays valid while documents come and go.
    vocab: HashMap<String, u32>,
    /// Postings, indexed by term id.
    postings: Vec<FxHashMap<PointId, u32>>,
    /// Per-document length and distinct term ids (needed to remove it again).
    docs: FxHashMap<PointId, (u32, Vec<u32>)>,
    total_len: u64,
}

impl LexicalIndex {
    pub fn new(params: Bm25Params) -> Self {
        Self {
            params,
            vocab: HashMap::new(),
            postings: Vec::new(),
            docs: FxHashMap::default(),
            total_len: 0,
        }
    }

    /// Number of indexed (non-empty) documents.
    pub fn doc_count(&self) -> usize {
        self.docs.len()
    }

    /// Number of distinct terms ever seen.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn contains(&self, id: PointId) -> bool {
        self.docs.contains_key(&id)
    }

    /// Index (or re-index) one document. Text without tokens leaves the
    /// document out of the index entirely.
    pub fn index_document(&mut self, id: PointId, text: &str) {
        self.remove_document(id);
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return;
        }
        let mut tf: FxHashMap<u32, u32> = FxHashMap::default();
        for token in &tokens {
            *tf.entry(self.term_id_or_insert(token)).or_insert(0) += 1;
        }
        let mut term_ids: Vec<u32> = tf.keys().copied().collect();
        term_ids.sort_unstable();
        for &term in &term_ids {
            self.postings[term as usize].insert(id, tf[&term]);
        }
        let len = tokens.len() as u32;
        self.total_len += u64::from(len);
        self.docs.insert(id, (len, term_ids));
    }

    /// Remove a document. Returns whether it was indexed.
    pub fn remove_document(&mut self, id: PointId) -> bool {
        let Some((len, term_ids)) = self.docs.remove(&id) else {
            return false;
        };
        for term in term_ids {
            self.postings[term as usize].remove(&id);
        }
        self.total_len -= u64::from(len);
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
        let n = self.docs.len() as f64;
        let df = self.postings.get(term as usize).map_or(0, |p| p.len()) as f64;
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    /// Score `text` and return the best `limit` documents.
    pub fn search(&self, text: &str, limit: usize) -> Vec<LexicalHit> {
        self.search_sparse(&self.query_vector(text), limit)
    }

    /// Score a sparse query over this index's vocabulary.
    pub fn search_sparse(&self, query: &SparseVector, limit: usize) -> Vec<LexicalHit> {
        if limit == 0 || self.docs.is_empty() {
            return Vec::new();
        }
        let avgdl = self.total_len as f64 / self.docs.len() as f64;
        let (k1, b) = (f64::from(self.params.k1), f64::from(self.params.b));
        let mut scores: FxHashMap<PointId, f64> = FxHashMap::default();
        for (term, weight) in query.iter() {
            let Some(postings) = self.postings.get(term as usize) else {
                continue;
            };
            let idf = self.idf_f64(term);
            let weight = f64::from(weight);
            for (&id, &tf) in postings {
                let dl = f64::from(self.docs.get(&id).map_or(0, |(len, _)| *len));
                let tf = f64::from(tf);
                let norm = tf + k1 * (1.0 - b + b * dl / avgdl);
                *scores.entry(id).or_insert(0.0) += weight * idf * tf * (k1 + 1.0) / norm;
            }
        }
        let mut hits: Vec<LexicalHit> = scores
            .into_iter()
            .map(|(id, score)| LexicalHit {
                id,
                score: score as f32,
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
        self.postings.push(FxHashMap::default());
        id
    }
}
