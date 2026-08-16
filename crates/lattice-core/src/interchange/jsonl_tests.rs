//! Tests for the JSONL logical interchange format.

use super::*;
use crate::engine::collection::CollectionEngine;
use crate::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
use crate::types::point::Point;

fn config() -> CollectionConfig {
    CollectionConfig::new(
        "journey",
        VectorConfig::new(4, Distance::Cosine),
        HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 100,
            ef_construction: 200,
        },
    )
    .with_relation("invalidates", 0)
    .with_relation("attests", 1)
}

/// A small graph exercising every field the format carries: vectors, JSON
/// payload, labels, and two relation types in a deliberately unsorted order.
fn seeded() -> CollectionEngine {
    let mut e = CollectionEngine::new(config()).unwrap();
    e.upsert_points(vec![
        Point::new_vector(3, vec![0.3, 0.0, 0.0, 1.0]).with_field("kind", br#""gate""#.to_vec()),
        Point::new_vector(1, vec![0.1, 0.5, 0.0, 1.0])
            .with_field("kind", br#""commit""#.to_vec())
            .with_field("sha", br#""78271f26""#.to_vec()),
        Point::new_vector(2, vec![0.2, 0.0, 0.5, 1.0]),
    ])
    .unwrap();
    e.add_edge(3, 1, "attests", 1.0).unwrap();
    e.add_edge(1, 2, "invalidates", 0.5).unwrap();
    e.add_edge(1, 3, "invalidates", 0.25).unwrap();
    e
}

fn dump(e: &CollectionEngine, opts: &ExportOptions) -> String {
    let mut buf = Vec::new();
    export_jsonl(e, &config(), &mut buf, opts).unwrap();
    String::from_utf8(buf).unwrap()
}

/// ★ The property the whole format exists for. Two exports of the same data
/// must be byte-identical, or every diff is noise and the file is useless in
/// version control — which is the only reason to have a text format at all.
/// Repeated rather than compared once: the bug this pins was `HashMap`
/// iteration order, and `RandomState` seeds each map instance separately, so a
/// single comparison catches it only by luck. Each `dump()` builds a fresh
/// config — i.e. a fresh `relations` map — which is exactly the condition that
/// made the header non-deterministic.
#[test]
fn export_is_byte_deterministic() {
    let e = seeded();
    let first = dump(&e, &ExportOptions::inline());
    for i in 1..32 {
        assert_eq!(
            first,
            dump(&e, &ExportOptions::inline()),
            "export {i} differed"
        );
    }
}

/// Points ascend by id and edges by (from, relation, to), regardless of the
/// order they were inserted in. Insertion order above is deliberately scrambled.
#[test]
fn output_is_sorted_not_insertion_ordered() {
    let text = dump(&seeded(), &ExportOptions::inline());
    let ids: Vec<&str> = text
        .lines()
        .filter(|l| l.contains(r#""t":"point""#))
        .map(|l| {
            l.split(r#""id":"#)
                .nth(1)
                .unwrap()
                .split(',')
                .next()
                .unwrap()
        })
        .collect();
    assert_eq!(ids, ["1", "2", "3"], "points must ascend by id");

    let edges: Vec<String> = text
        .lines()
        .filter(|l| l.contains(r#""t":"edge""#))
        .map(|l| l.to_string())
        .collect();
    assert!(edges[0].contains(r#""from":1"#) && edges[0].contains(r#""rel":"invalidates""#));
    assert!(
        edges[0].contains(r#""to":2"#),
        "ties break on `to`: {}",
        edges[0]
    );
    assert!(edges[1].contains(r#""to":3"#));
    assert!(edges[2].contains(r#""from":3"#));
}

/// Roundtrip: everything that went in comes back, including payload values and
/// edges resolved by name.
#[test]
fn roundtrip_preserves_points_payload_and_edges() {
    let text = dump(&seeded(), &ExportOptions::inline());
    let back = import_jsonl(text.as_bytes()).unwrap();

    assert_eq!(back.header.points, 3);
    assert_eq!(back.header.edges, 3);
    assert_eq!(back.header.dim, 4);
    assert_eq!(back.points.len(), 3);

    let p1 = back.points.iter().find(|p| p.id == 1).unwrap();
    assert_eq!(p1.vector, vec![0.1, 0.5, 0.0, 1.0]);
    assert_eq!(p1.payload.get("sha").unwrap().as_slice(), br#""78271f26""#);
    assert_eq!(p1.payload.get("kind").unwrap().as_slice(), br#""commit""#);

    assert!(back
        .edges
        .contains(&(3u64, 1u64, "attests".to_string(), 1.0)));
    assert!(back
        .edges
        .contains(&(1u64, 2u64, "invalidates".to_string(), 0.5)));
}

/// ★ Edges carry the relation NAME, never the collection-local `relation_id`.
///
/// A dump holding the numeric id would silently mean something different when
/// imported into a collection whose `relations` map assigns ids in another
/// order — a corruption that reports success. This asserts the name is on the
/// wire and the id is not.
#[test]
fn edges_are_exported_by_relation_name_not_id() {
    let text = dump(&seeded(), &ExportOptions::inline());
    let edge = text.lines().find(|l| l.contains(r#""t":"edge""#)).unwrap();
    assert!(edge.contains(r#""rel":"invalidates""#), "{edge}");
    assert!(
        !edge.contains(r#""relation_id""#),
        "the collection-local id must not reach the wire: {edge}"
    );
}

/// Vectors are omitted on request, and the header still records the dimension
/// so a re-embedding importer can check its embedder agrees.
#[test]
fn omit_vectors_drops_them_but_keeps_the_dimension() {
    let text = dump(&seeded(), &ExportOptions::omit_vectors());
    assert!(!text.contains(r#""vector""#), "vectors must be absent");
    let back = import_jsonl(text.as_bytes()).unwrap();
    assert_eq!(
        back.header.dim, 4,
        "dimension survives even without vectors"
    );
    assert_eq!(back.header.vectors, VectorMode::Omit);
    assert!(back.points.iter().all(|p| p.vector.is_empty()));
    // Payload and graph structure are untouched by the vector choice.
    assert_eq!(back.edges.len(), 3);
    let p1 = back.points.iter().find(|p| p.id == 1).unwrap();
    assert_eq!(p1.payload.get("sha").unwrap().as_slice(), br#""78271f26""#);
}

/// Omitting vectors is what makes the dump reviewable — assert it actually is
/// dramatically smaller, since that is the entire justification for the mode.
#[test]
fn omitting_vectors_is_substantially_smaller() {
    let e = seeded();
    let inline = dump(&e, &ExportOptions::inline()).len();
    let omitted = dump(&e, &ExportOptions::omit_vectors()).len();
    assert!(omitted < inline, "{omitted} !< {inline}");
}

/// Payload bytes that are not valid JSON are preserved exactly rather than
/// dropped or lossily stringified. A dump that silently discards what it did
/// not understand is worse than one that is ugly.
#[test]
fn non_json_payload_bytes_survive_the_roundtrip() {
    let mut e = CollectionEngine::new(config()).unwrap();
    let raw = vec![0xff, 0x00, 0x7f, 0x80];
    e.upsert_points(vec![
        Point::new_vector(1, vec![1.0, 0.0, 0.0, 0.0]).with_field("blob", raw.clone())
    ])
    .unwrap();

    let text = dump(&e, &ExportOptions::inline());
    assert!(text.contains("$bytes"), "expected the escape hatch: {text}");
    let back = import_jsonl(text.as_bytes()).unwrap();
    assert_eq!(back.points[0].payload.get("blob").unwrap(), &raw);
}

/// f32 values survive the text trip exactly — Rust's float formatting is
/// shortest-roundtrip, so a decimal dump is lossless, not approximate.
#[test]
fn vectors_roundtrip_exactly() {
    let mut e = CollectionEngine::new(config()).unwrap();
    let awkward = vec![0.1, 1.0 / 3.0, f32::MIN_POSITIVE, 1e-7];
    e.upsert_points(vec![Point::new_vector(1, awkward.clone())])
        .unwrap();
    let back = import_jsonl(dump(&e, &ExportOptions::inline()).as_bytes()).unwrap();
    assert_eq!(back.points[0].vector, awkward);
}

/// ★ export ∘ import ∘ export is the identity for f64 config fields.
///
/// The header carries the collection config, whose `hnsw.ml` is an f64. The
/// serializer already emits the shortest round-trippable representation (ryu),
/// but without serde_json's `float_roundtrip` feature the *parser* reads that
/// string back up to 1 ulp off — e.g. `0.36067376022224085` returns as
/// `0.3606737602222409`. The re-exported header then differs by one digit and
/// the dumps are no longer byte-identical, breaking the format's whole reason
/// to exist. Vectors are f32 and unaffected; this only bites f64 config fields.
///
/// `ml` is set to `1.0/(16f64).ln()`, an irrational value that has no short
/// decimal and so exercises the full-precision parse.
#[test]
fn f64_config_fields_survive_import_byte_identically() {
    let ml = 1.0 / (16f64).ln();
    let cfg = CollectionConfig::new(
        "roundtrip",
        VectorConfig::new(4, Distance::Cosine),
        HnswConfig {
            m: 16,
            m0: 32,
            ml,
            ef: 100,
            ef_construction: 200,
        },
    );

    let mut e = CollectionEngine::new(cfg.clone()).unwrap();
    e.upsert_points(vec![Point::new_vector(1, vec![0.1, 0.2, 0.3, 0.4])])
        .unwrap();

    // First export.
    let mut buf = Vec::new();
    export_jsonl(&e, &cfg, &mut buf, &ExportOptions::inline()).unwrap();
    let first = String::from_utf8(buf).unwrap();

    // Import, rebuild, and re-export using the config as it came back.
    let back = import_jsonl(first.as_bytes()).unwrap();
    let mut e2 = CollectionEngine::new(back.header.config.clone()).unwrap();
    e2.upsert_points(back.points.clone()).unwrap();
    for (from, to, rel, w) in &back.edges {
        e2.add_edge(*from, *to, rel, *w).unwrap();
    }
    let mut buf = Vec::new();
    export_jsonl(&e2, &back.header.config, &mut buf, &ExportOptions::inline()).unwrap();
    let second = String::from_utf8(buf).unwrap();

    assert_eq!(
        first, second,
        "export∘import∘export must be byte-identical for f64 config fields"
    );
}

/// ★ A truncated dump is an error, not a short collection.
///
/// The header's counts are a checksum on the body. Without honouring them, a
/// writer that died mid-dump produces a file that imports "successfully" with
/// silently missing rows.
#[test]
fn a_truncated_dump_is_refused() {
    let text = dump(&seeded(), &ExportOptions::inline());
    let cut: String = text.lines().take(2).map(|l| format!("{l}\n")).collect();
    let err = import_jsonl(cut.as_bytes()).unwrap_err().to_string();
    assert!(err.contains("truncated or corrupt"), "{err}");
}

/// A future format version is refused rather than partially parsed — an unknown
/// line kind quietly skipped is data loss that reports success.
#[test]
fn a_newer_format_version_is_refused() {
    let text =
        dump(&seeded(), &ExportOptions::inline()).replace(r#""version":1"#, r#""version":9999"#);
    let err = import_jsonl(text.as_bytes()).unwrap_err().to_string();
    assert!(err.contains("9999"), "{err}");
}

#[test]
fn a_foreign_format_is_refused() {
    let text = dump(&seeded(), &ExportOptions::inline()).replace(
        r#""format":"lattice-jsonl""#,
        r#""format":"something-else""#,
    );
    let err = import_jsonl(text.as_bytes()).unwrap_err().to_string();
    assert!(err.contains("something-else"), "{err}");
}

#[test]
fn a_body_without_a_header_is_refused() {
    let text = dump(&seeded(), &ExportOptions::inline());
    let body: String = text.lines().skip(1).map(|l| format!("{l}\n")).collect();
    let err = import_jsonl(body.as_bytes()).unwrap_err().to_string();
    assert!(err.contains("precedes the header"), "{err}");
}

/// An empty collection is a legitimate dump, not an edge case that panics.
#[test]
fn an_empty_collection_roundtrips() {
    let e = CollectionEngine::new(config()).unwrap();
    let back = import_jsonl(dump(&e, &ExportOptions::inline()).as_bytes()).unwrap();
    assert_eq!(back.points.len(), 0);
    assert_eq!(back.edges.len(), 0);
    assert_eq!(back.header.config.name, "journey");
}

/// The header carries the collection config, so an importer can recreate the
/// collection — including the relation map the edge names resolve against.
#[test]
fn the_header_carries_enough_config_to_rebuild() {
    let back = import_jsonl(dump(&seeded(), &ExportOptions::inline()).as_bytes()).unwrap();
    assert_eq!(back.header.config.vectors.size, 4);
    assert_eq!(back.header.config.relations.get("invalidates"), Some(&0));
    assert_eq!(back.header.config.relations.get("attests"), Some(&1));
}

/// Blank lines are tolerated — concatenating dumps or a trailing newline from a
/// shell pipeline must not break a reader.
#[test]
fn blank_lines_are_ignored() {
    let text = dump(&seeded(), &ExportOptions::inline()).replace('\n', "\n\n");
    assert_eq!(import_jsonl(text.as_bytes()).unwrap().points.len(), 3);
}

/// ★ Determinism must not depend on a feature a CONSUMER controls.
///
/// The first version of this relied on `serde_json::Value` being
/// `BTreeMap`-backed, which holds only while nobody enables serde_json's
/// `preserve_order`. Cargo unifies features across the whole build graph, so a
/// downstream crate can turn it on for us — Atlas, the first real consumer,
/// does. Under `preserve_order` a `Value` faithfully preserves whatever
/// arbitrary order it was handed, silently restoring the bug in a build this
/// repo's own CI would never see.
///
/// This asserts the output is sorted as *text*, which is true under either
/// backing and cannot be quietly undone by a feature flag.
#[test]
fn header_keys_are_sorted_regardless_of_the_serde_json_backing() {
    let header = dump(&seeded(), &ExportOptions::inline())
        .lines()
        .next()
        .unwrap()
        .to_string();

    // Depth-aware: sibling keys must ascend WITHIN each object. A flat scan
    // would mix levels and report a false failure on correctly sorted output.
    assert!(
        keys_are_sorted_per_level(&header),
        "some object's keys are not in sorted order: {header}"
    );

    // The nested relations map is the one that actually carried the bug.
    let rel = header.find("\"relations\"").expect("relations in header");
    let after = &header[rel..];
    let a = after.find("\"attests\"").expect("attests present");
    let i = after.find("\"invalidates\"").expect("invalidates present");
    assert!(a < i, "nested relations map is not sorted: {after}");
}

/// Scan raw JSON text and check that every object's own keys ascend.
///
/// Deliberately textual: parsing with `serde_json` would re-order the keys
/// through whichever map backs `Value`, which is exactly the thing under test.
fn keys_are_sorted_per_level(json: &str) -> bool {
    let bytes: Vec<char> = json.chars().collect();
    let mut stack: Vec<Option<String>> = Vec::new();
    let mut i = 0;
    let mut in_string = false;
    let mut current = String::new();
    let mut escaped = false;

    while i < bytes.len() {
        let c = bytes[i];
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
                // A string immediately followed by ':' is a key.
                if bytes.get(i + 1) == Some(&':') {
                    if let Some(slot) = stack.last_mut() {
                        if let Some(prev) = slot {
                            if current.as_str() < prev.as_str() {
                                return false;
                            }
                        }
                        *slot = Some(current.clone());
                    }
                }
                current.clear();
            } else {
                current.push(c);
            }
        } else {
            match c {
                '"' => in_string = true,
                '{' => stack.push(None),
                '}' => {
                    stack.pop();
                }
                // Arrays do not introduce keys, but they must not be mistaken
                // for the enclosing object's scope either.
                '[' => stack.push(Some(String::new())),
                ']' => {
                    stack.pop();
                }
                _ => {}
            }
        }
        i += 1;
    }
    true
}

/// A negative control for the scanner above.
///
/// A checker that always returned `true` would make the determinism test pass
/// vacuously, so the checker itself is tested against known-unsorted input —
/// including the nested and post-array cases, which are where a naive
/// implementation gets it wrong.
#[test]
fn the_sortedness_scanner_actually_detects_disorder() {
    assert!(keys_are_sorted_per_level(r#"{"a":1,"b":2}"#));
    assert!(!keys_are_sorted_per_level(r#"{"b":1,"a":2}"#));
    assert!(keys_are_sorted_per_level(r#"{"a":{"x":1,"y":2},"b":3}"#));
    assert!(
        !keys_are_sorted_per_level(r#"{"a":{"y":1,"x":2},"b":3}"#),
        "disorder nested one level deep must be caught"
    );
    // A nested object must not leak its last key into the parent's comparison.
    assert!(
        keys_are_sorted_per_level(r#"{"a":{"z":1},"b":2}"#),
        "a child's key must not be compared against a parent's next key"
    );
    // Arrays introduce no keys and must not disturb the enclosing scope.
    assert!(keys_are_sorted_per_level(r#"{"a":[1,2,3],"b":2}"#));
    assert!(!keys_are_sorted_per_level(r#"{"b":[1],"a":2}"#));
    // A key containing a quote or colon must not desynchronise the scan.
    assert!(keys_are_sorted_per_level(r#"{"a:b":1,"c\"d":2}"#));
}
