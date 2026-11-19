//! A small site-search corpus where the right page must rank first.

use crate::text::{Bm25Params, LexicalIndex, BM25_B, BM25_K1};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

const CORPUS: &[(u64, &str)] = &[
    (1, "Installing the GitHub Action: add the workflow file to .github/workflows and commit it."),
    (2, "Pricing: the free plan includes unlimited searches; the team plan adds analytics."),
    (3, "Service worker mode runs the vector database inside the browser cache scope."),
    (4, "Cloudflare Worker deployment fits within the free plan CPU limits for small sites."),
    (5, "Hybrid search fuses BM25 keyword ranking with embedding similarity using reciprocal rank fusion."),
    (6, "Privacy: no visitor queries leave the browser when the index runs client side."),
    (7, "Troubleshooting: if the index is empty, check that the crawl step produced pages."),
    (8, "Embedding models: configure an OpenAI compatible endpoint, model name and API key."),
    (9, "Changelog: version 2 adds sitemap crawling and incremental index updates."),
    (10, "Accessibility: the search dialog is keyboard navigable and announces results."),
];

const QUERIES: &[(&str, u64)] = &[
    ("how do I install the github action workflow", 1),
    ("is there a free plan pricing", 2),
    ("run inside a service worker", 3),
    ("deploy to cloudflare worker", 4),
    ("what is reciprocal rank fusion hybrid", 5),
    ("does my query leave the browser privacy", 6),
    ("index is empty troubleshooting", 7),
    ("which embedding model and api key", 8),
    ("sitemap crawling changelog", 9),
    ("keyboard accessibility", 10),
];

fn corpus_index() -> LexicalIndex {
    let mut idx = LexicalIndex::new(Bm25Params {
        k1: BM25_K1,
        b: BM25_B,
    });
    for (id, text) in CORPUS {
        idx.index_document(*id, text).unwrap();
    }
    idx
}

#[test]
fn the_right_page_ranks_first_for_every_query() {
    let idx = corpus_index();
    for (query, expected) in QUERIES {
        let hits = idx.search(query, 3);
        assert_eq!(
            hits.first().map(|h| h.id),
            Some(*expected),
            "query {query:?} ranked {hits:?}"
        );
    }
}

#[test]
fn insertion_order_does_not_change_results() {
    let forward = corpus_index();
    let mut reverse = LexicalIndex::new(Bm25Params {
        k1: BM25_K1,
        b: BM25_B,
    });
    for (id, text) in CORPUS.iter().rev() {
        reverse.index_document(*id, text).unwrap();
    }
    // Term ids depend on insertion order, so the summation order of a
    // multi-term score can differ in the last ulp; rankings must not.
    for (query, _) in QUERIES {
        let (a, b) = (forward.search(query, 10), reverse.search(query, 10));
        let ids = |h: &[crate::text::LexicalHit]| h.iter().map(|x| x.id).collect::<Vec<_>>();
        assert_eq!(ids(&a), ids(&b), "{query}");
        for (x, y) in a.iter().zip(&b) {
            assert!((x.score - y.score).abs() < 1e-6, "{query}");
        }
    }
}
