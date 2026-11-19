//! BM25 scores checked against values computed by hand.
//!
//! Corpus (k1 = 1.2, b = 0.75):
//!   d1 = "the cat sat"  (|d| = 3)
//!   d2 = "the dog"      (|d| = 2)
//!   d3 = "cat cat dog"  (|d| = 3)
//! N = 3, avgdl = 8/3.
//!
//! "cat": df = 2, idf = ln(1 + (3 − 2 + 0.5)/(2 + 0.5)) = ln 1.6 = 0.470004
//!   d1: tf = 1, norm = 1 + 1.2·(0.25 + 0.75·3/(8/3)) = 2.3125
//!       0.470004 · 1·2.2 / 2.3125 = 0.447139
//!   d3: tf = 2, norm = 2 + 1.3125 = 3.3125
//!       0.470004 · 2·2.2 / 3.3125 = 0.624307
//! "dog": d2: |d| = 2, norm = 1 + 1.2·(0.25 + 0.75·2/(8/3)) = 1.975
//!       0.470004 · 2.2 / 1.975 = 0.523548

use crate::text::{Bm25Params, LexicalIndex, BM25_B, BM25_K1};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

const EPS: f32 = 1e-5;

fn fixture() -> LexicalIndex {
    let mut idx = LexicalIndex::new(Bm25Params {
        k1: BM25_K1,
        b: BM25_B,
    });
    idx.index_document(1, "the cat sat");
    idx.index_document(2, "the dog");
    idx.index_document(3, "cat cat dog");
    idx
}

fn score(idx: &LexicalIndex, query: &str, id: u64) -> f32 {
    idx.search(query, 10)
        .into_iter()
        .find(|h| h.id == id)
        .map_or(0.0, |h| h.score)
}

#[test]
fn idf_matches_hand_computation() {
    let idx = fixture();
    let q = idx.query_vector("cat");
    assert!((idx.idf(q.indices()[0]) - 0.470_004).abs() < EPS);
}

#[test]
fn single_term_scores_match_hand_computation() {
    let idx = fixture();
    assert!((score(&idx, "cat", 1) - 0.447_139).abs() < EPS);
    assert!((score(&idx, "cat", 3) - 0.624_307).abs() < EPS);
    assert!((score(&idx, "dog", 2) - 0.523_548).abs() < EPS);
    assert_eq!(score(&idx, "cat", 2), 0.0);
}

#[test]
fn multi_term_query_sums_and_orders() {
    let idx = fixture();
    let hits = idx.search("Cat, DOG!", 10);
    let ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
    assert_eq!(ids, vec![3, 2, 1]);
    assert!((hits[0].score - (0.624_307 + 0.447_139)).abs() < EPS);
}

#[test]
fn repeated_query_terms_weigh_double() {
    let idx = fixture();
    assert!((score(&idx, "cat cat", 1) - 2.0 * 0.447_139).abs() < EPS);
    let q = idx.query_vector("cat cat unknown");
    assert_eq!(q.len(), 1, "unknown terms are dropped");
    assert_eq!(q.values(), &[2.0]);
}

#[test]
fn equal_scores_tie_break_by_id_and_limit_applies() {
    let idx = fixture();
    // "the" scores d1 and d2; d2 is shorter so it ranks first.
    let ids: Vec<u64> = idx.search("the", 10).iter().map(|h| h.id).collect();
    assert_eq!(ids, vec![2, 1]);
    let mut same = LexicalIndex::new(Bm25Params {
        k1: BM25_K1,
        b: BM25_B,
    });
    same.index_document(9, "alpha");
    same.index_document(4, "alpha");
    let ids: Vec<u64> = same.search("alpha", 1).iter().map(|h| h.id).collect();
    assert_eq!(ids, vec![4]);
}

#[test]
fn removal_restores_corpus_statistics() {
    let mut idx = fixture();
    idx.index_document(4, "zebra zebra zebra");
    assert!(idx.remove_document(4));
    assert!(!idx.remove_document(4));
    assert_eq!(idx.doc_count(), 3);
    assert!((score(&idx, "cat", 3) - 0.624_307).abs() < EPS);
    assert!(idx.search("zebra", 10).is_empty());
}

#[test]
fn reindexing_replaces_the_old_document() {
    let mut idx = fixture();
    idx.index_document(1, "a completely different text");
    assert_eq!(score(&idx, "cat", 1), 0.0);
    assert_eq!(idx.doc_count(), 3);
}

#[test]
fn empty_index_and_empty_query_return_nothing() {
    let idx = LexicalIndex::new(Bm25Params {
        k1: BM25_K1,
        b: BM25_B,
    });
    assert!(idx.search("cat", 10).is_empty());
    assert!(fixture().search("   ", 10).is_empty());
    assert!(fixture().search("cat", 0).is_empty());
}
