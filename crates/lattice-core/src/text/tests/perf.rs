//! Build/query timings and rebuild-vs-deserialize, for the readiness report.
//!
//! Ignored by default (timing, not correctness). Run with:
//! `cargo test -p latticedb-core --release --lib text::tests::perf -- --ignored --nocapture`

use crate::text::{Bm25Params, LexicalIndex, LexicalParts, BM25_B, BM25_K1};
use std::time::Instant;

const VOCAB: usize = 5_000;
const WORDS_PER_DOC: usize = 60;
const QUERIES: usize = 1_000;

/// Deterministic LCG so every run indexes the same corpus.
struct Lcg(u64);
impl Lcg {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Skewed word choice: low ids are common, like natural language.
    fn word(&mut self) -> String {
        format!("w{}", (VOCAB as f64 * self.next_unit().powi(3)) as usize)
    }
}

fn corpus(n: usize) -> Vec<String> {
    let mut rng = Lcg(42);
    (0..n)
        .map(|_| {
            (0..WORDS_PER_DOC)
                .map(|_| rng.word())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

fn idx_params() -> Bm25Params {
    Bm25Params {
        k1: BM25_K1,
        b: BM25_B,
    }
}

fn build(docs: &[String]) -> LexicalIndex {
    let mut idx = LexicalIndex::new(idx_params());
    for (id, text) in docs.iter().enumerate() {
        idx.index_document(id as u64, text).unwrap();
    }
    idx
}

#[test]
#[ignore = "timing report, run explicitly"]
fn bm25_perf_report() {
    for n in [1_000usize, 10_000] {
        let docs = corpus(n);
        let t = Instant::now();
        let idx = build(&docs);
        let build_ms = t.elapsed().as_secs_f64() * 1e3;

        let mut rng = Lcg(7);
        let queries: Vec<String> = (0..QUERIES)
            .map(|_| format!("{} {} {}", rng.word(), rng.word(), rng.word()))
            .collect();
        let t = Instant::now();
        let mut found = 0usize;
        for q in &queries {
            found += idx.search(q, 10).len();
        }
        let query_us = t.elapsed().as_secs_f64() * 1e6 / QUERIES as f64;

        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&idx.to_parts()).unwrap();
        let t = Instant::now();
        let parts: LexicalParts =
            rkyv::from_bytes::<LexicalParts, rkyv::rancor::Error>(&bytes).unwrap();
        let restored = LexicalIndex::from_parts(idx_params(), parts).unwrap();
        let deser_ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(restored.doc_count(), n);
        assert!(found > 0);

        eprintln!(
            "BM25 n={n}: build {build_ms:.1} ms, query {query_us:.1} µs avg (top-10, 3 terms), \
             snapshot section {:.1} MB, restore {deser_ms:.1} ms",
            bytes.len() as f64 / 1e6
        );
    }
}

#[test]
#[ignore = "timing report, run explicitly"]
fn snapshot_load_with_and_without_text_index() {
    use crate::engine::collection::CollectionEngine;
    use crate::text::TextIndexConfig;
    use crate::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
    use crate::types::point::Point;

    let docs = corpus(10_000);
    let mut rng = Lcg(3);
    let points: Vec<Point> = docs
        .iter()
        .enumerate()
        .map(|(id, text)| {
            let vector = (0..32).map(|_| rng.next_unit() as f32).collect();
            Point::new_vector(id as u64, vector)
                .with_field("text", serde_json::to_vec(text).unwrap())
        })
        .collect();
    for with_index in [false, true] {
        let mut config = CollectionConfig::new(
            "perf",
            VectorConfig::new(32, Distance::Cosine),
            HnswConfig {
                m: 16,
                m0: 32,
                ml: HnswConfig::recommended_ml(16),
                ef: 100,
                ef_construction: 100,
            },
        );
        if with_index {
            config = config.with_text_index(TextIndexConfig::bm25("text"));
        }
        let mut engine = CollectionEngine::new(config).unwrap();
        engine.upsert_points(points.clone()).unwrap();
        let bytes = engine.to_bytes().unwrap();
        let t = Instant::now();
        let restored = CollectionEngine::from_bytes(&bytes).unwrap();
        let load_ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(restored.point_count(), 10_000);
        eprintln!(
            "snapshot load 10K points (HNSW graph and postings restored), text_index={with_index}: {load_ms:.1} ms"
        );
    }
}
