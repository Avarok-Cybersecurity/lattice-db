//! Snapshot bytes are a function of the collection's logical content: not of
//! hash-map seeds (which differ per map and per process), nor of the order
//! payload fields, relations or (for a collection without an HNSW graph)
//! points were inserted in.
//!
//! An HNSW graph is itself a product of insertion order, so permuting the
//! points of a vector collection builds a different graph and, rightly, a
//! different snapshot; building it in the same order reproduces it exactly.

use super::CollectionEngine;
use crate::text::TextIndexConfig;
use crate::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
use crate::types::point::{Point, PointId};
use sha2::{Digest, Sha256};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

/// sha256 of `snapshot(&Build::canonical(true))`. It changes only when the
/// snapshot encoding does: on native and wasm32, on every platform, in every
/// process. Updating it is a snapshot format decision, not a test fix.
const GOLDEN_SHA256: &str = "3c60c94474c183ffcc3cedf9b62f998b06988c83822fe2cb43cf8537ddfeeae4";

const POINTS: u64 = 12;
/// Enough fields that a payload's hash table has probe collisions, whose
/// placement depends on the order entries are archived in.
const FIELDS: usize = 24;
const RELATIONS: [&str; 6] = ["cites", "links", "owns", "precedes", "mentions", "imports"];
const WORDS: [&str; 7] = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta"];

/// How to build the fixture; every variant holds the same logical content.
struct Build {
    vectors: bool,
    points_reversed: bool,
    fields_reversed: bool,
    edges_reversed: bool,
}

impl Build {
    fn canonical(vectors: bool) -> Self {
        Self {
            vectors,
            points_reversed: false,
            fields_reversed: false,
            edges_reversed: false,
        }
    }
}

fn config(vectors: bool) -> CollectionConfig {
    let hnsw = HnswConfig {
        m: 4,
        m0: 8,
        ml: HnswConfig::recommended_ml(4),
        ef: 16,
        ef_construction: 16,
    };
    // Euclid over small integers: every distance is exact on every platform.
    let vectors = vectors.then(|| VectorConfig::new(4, Distance::Euclid));
    CollectionConfig::with_optional_vectors("golden", vectors, hnsw)
        .with_text_index(TextIndexConfig::bm25("body"))
}

fn point(id: PointId, build: &Build) -> Point {
    let vector = if build.vectors {
        vec![id as f32, (id % 3) as f32, (id % 5) as f32, 1.0]
    } else {
        Vec::new()
    };
    let w = |i: u64| WORDS[(i % WORDS.len() as u64) as usize];
    let body = format!("\"{} {} {}\"", w(id), w(id * 3), w(id + id / 2));
    let mut fields: Vec<(String, Vec<u8>)> = (0..FIELDS)
        .map(|f| {
            let value = format!(r#"{{"n":{},"tag":"t{}"}}"#, id * 100 + f as u64, f % 4);
            (format!("f{f:02}"), value.into_bytes())
        })
        .collect();
    fields.push(("body".to_string(), body.into_bytes()));
    if build.fields_reversed {
        fields.reverse();
    }
    fields
        .into_iter()
        .fold(Point::new_vector(id, vector), |p, (k, v)| {
            p.with_field(k, v)
        })
}

/// Two edges per point, in a fixed order per source (adjacency order is
/// observable, e.g. in traversal results, so it is content, not noise).
fn edges() -> Vec<(PointId, PointId, &'static str)> {
    (1..=POINTS)
        .flat_map(|id| {
            [
                (id, id % POINTS + 1, RELATIONS[(id % 6) as usize]),
                (
                    id,
                    (id + 4) % POINTS + 1,
                    RELATIONS[((id + 3) % 6) as usize],
                ),
            ]
        })
        .collect()
}

fn snapshot(build: &Build) -> Vec<u8> {
    let mut engine = CollectionEngine::new(config(build.vectors)).expect("engine");
    let mut ids: Vec<PointId> = (1..=POINTS).collect();
    if build.points_reversed {
        ids.reverse();
    }
    // One upsert per point so the order reaches every index.
    for id in ids {
        engine
            .upsert_points(vec![point(id, build)])
            .expect("upsert");
    }
    let mut edges = edges();
    if build.edges_reversed {
        // Reverse sources (so relations are first used in another order) but
        // keep each source's own edges in order.
        edges.sort_by_key(|&(from, _, _)| std::cmp::Reverse(from));
    }
    for (from, to, relation) in edges {
        engine.add_edge(from, to, relation, 0.5).expect("edge");
    }
    engine.to_bytes().expect("snapshot")
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn the_same_collection_exports_identical_bytes() {
    for vectors in [true, false] {
        let first = snapshot(&Build::canonical(vectors));
        for _ in 0..4 {
            assert!(
                snapshot(&Build::canonical(vectors)) == first,
                "vectors={vectors}: a rebuild exported different bytes"
            );
        }
    }
}

#[test]
fn payload_field_and_relation_order_do_not_change_the_bytes() {
    for vectors in [true, false] {
        let reordered = Build {
            fields_reversed: true,
            edges_reversed: true,
            ..Build::canonical(vectors)
        };
        assert!(
            snapshot(&reordered) == snapshot(&Build::canonical(vectors)),
            "vectors={vectors}"
        );
    }
}

#[test]
fn point_order_does_not_change_a_graphless_snapshot() {
    let permuted = Build {
        points_reversed: true,
        fields_reversed: true,
        edges_reversed: true,
        ..Build::canonical(false)
    };
    assert!(snapshot(&permuted) == snapshot(&Build::canonical(false)));
}

#[test]
fn the_fixture_snapshot_matches_its_golden_hash() {
    let bytes = snapshot(&Build::canonical(true));
    assert_eq!(sha256_hex(&bytes), GOLDEN_SHA256, "{} bytes", bytes.len());
}

#[test]
fn the_fixture_snapshot_round_trips() {
    let bytes = snapshot(&Build::canonical(true));
    let restored = CollectionEngine::from_bytes(&bytes).expect("load");
    assert_eq!(restored.point_count(), POINTS as usize);
    assert!(restored.to_bytes().expect("re-export") == bytes);
}
