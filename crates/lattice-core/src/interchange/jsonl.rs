//! JSONL logical interchange: one self-describing JSON object per line.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};

use serde::{Deserialize, Serialize};

use crate::engine::EngineOps;
use crate::error::{LatticeError, LatticeResult};
use crate::types::collection::CollectionConfig;
use crate::types::point::{Point, PointId};

/// Format marker written into the header line.
pub const FORMAT: &str = "lattice-jsonl";

/// Version of the *interchange* format, independent of the crate version and
/// of the on-disk page format. A reader must refuse a version it does not
/// understand rather than guess at unknown fields.
pub const FORMAT_VERSION: u32 = 1;

/// What to do with vectors on export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VectorMode {
    /// Write every vector as a JSON array of `f32`.
    ///
    /// Exact — Rust's shortest-roundtrip float formatting means the values
    /// survive the text trip unchanged — but roughly 3x the size of the binary
    /// form, and a 1024-dimension vector is one very long line that no reviewer
    /// can read. Correct when the dump is the only copy of the data.
    Inline,
    /// Omit vectors entirely; write points with their payload and labels only.
    ///
    /// The right choice when vectors are *derived* — recomputable by embedding
    /// the payload again. The dump stays small and reviewable, and the index is
    /// rebuilt on import. Re-embedding is not bit-reproducible across model
    /// versions, so a consumer that needs identical vectors must use
    /// [`VectorMode::Inline`].
    Omit,
}

/// Export knobs.
#[derive(Debug, Clone)]
pub struct ExportOptions {
    pub vectors: VectorMode,
}

impl ExportOptions {
    /// Vectors inline. Named rather than derived so that `Default` never
    /// quietly decides something this consequential for a caller.
    pub fn inline() -> Self {
        Self {
            vectors: VectorMode::Inline,
        }
    }

    /// Vectors omitted.
    pub fn omit_vectors() -> Self {
        Self {
            vectors: VectorMode::Omit,
        }
    }
}

/// The first line of every dump.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub format: String,
    pub version: u32,
    pub config: CollectionConfig,
    pub vectors: VectorMode,
    /// Vector dimensionality the dump was taken at. Present even when vectors
    /// are omitted: an importer that re-embeds must be able to check that its
    /// embedder agrees with the collection it is filling.
    pub dim: usize,
    pub points: usize,
    pub edges: usize,
}

/// A decoded dump: everything needed to rebuild a collection, and nothing about
/// how to store it.
#[derive(Debug, Clone)]
pub struct Dump {
    pub header: Header,
    pub points: Vec<Point>,
    /// `(from, to, relation name, weight)` — names, not ids. See the module doc.
    pub edges: Vec<(PointId, PointId, String, f32)>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "lowercase")]
enum Line {
    Header(Header),
    Point {
        id: PointId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        vector: Option<Vec<f32>>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        payload: BTreeMap<String, serde_json::Value>,
        #[serde(default, skip_serializing_if = "is_zero")]
        labels: u64,
    },
    Edge {
        from: PointId,
        to: PointId,
        rel: String,
        w: f32,
    },
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}

/// Payload values are stored as raw bytes that are *usually* encoded JSON.
///
/// Decoding them makes the dump readable and diffable — a changed payload shows
/// up as a changed field rather than a changed blob. Bytes that are not valid
/// JSON are preserved exactly as `{"$bytes":[…]}` rather than being lossily
/// stringified or dropped; a dump that silently discards data it did not
/// understand is worse than one that is ugly.
fn payload_to_json(bytes: &[u8]) -> serde_json::Value {
    match serde_json::from_slice::<serde_json::Value>(bytes) {
        Ok(v) => v,
        Err(_) => serde_json::json!({ "$bytes": bytes }),
    }
}

fn json_to_payload(v: &serde_json::Value) -> LatticeResult<Vec<u8>> {
    if let Some(obj) = v.as_object() {
        if obj.len() == 1 {
            if let Some(raw) = obj.get("$bytes").and_then(|b| b.as_array()) {
                return raw
                    .iter()
                    .map(|n| {
                        n.as_u64()
                            .and_then(|n| u8::try_from(n).ok())
                            .ok_or_else(|| LatticeError::Serialization {
                                message: "$bytes must contain only integers 0..=255".into(),
                            })
                    })
                    .collect();
            }
        }
    }
    serde_json::to_vec(v).map_err(|e| LatticeError::Serialization {
        message: format!("re-encoding payload value: {e}"),
    })
}

/// Rebuild a value with every nested object's keys in sorted order.
///
/// ★ Why this is explicit rather than "just serialize through `Value`".
///
/// Serializing a `HashMap` directly emits it in *iteration* order, and Rust's
/// `RandomState` seeds each map instance separately, so two exports of
/// identical data inside one process can order the same map differently.
/// [`CollectionConfig::relations`] is exactly such a map and rides in the
/// header.
///
/// The tempting fix is to round-trip through [`serde_json::Value`] and rely on
/// it being `BTreeMap`-backed. That is true **only while nobody enables
/// serde_json's `preserve_order` feature** — and cargo unifies features across
/// the whole build graph, so a *downstream consumer* can turn it on for us.
/// Atlas, the first such consumer, does exactly that. Under `preserve_order`
/// `Value` is `IndexMap`-backed and faithfully preserves the arbitrary order it
/// was given, silently restoring the bug in a build we do not control.
///
/// So the ordering is imposed here, where it cannot be switched off. Inserting
/// into a fresh map in sorted key order produces sorted output under both
/// backings.
fn sort_keys(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(String, serde_json::Value)> = map.into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            serde_json::Value::Object(
                entries
                    .into_iter()
                    .map(|(k, v)| (k, sort_keys(v)))
                    .collect(),
            )
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(sort_keys).collect())
        }
        other => other,
    }
}

/// Serialize one line with every map in sorted key order.
fn write_line<W: Write>(w: &mut W, line: &Line) -> LatticeResult<()> {
    let normalized = serde_json::to_value(line).map_err(|e| LatticeError::Serialization {
        message: format!("encoding dump line: {e}"),
    })?;
    let s =
        serde_json::to_string(&sort_keys(normalized)).map_err(|e| LatticeError::Serialization {
            message: format!("encoding dump line: {e}"),
        })?;
    writeln!(w, "{s}").map_err(|e| LatticeError::Serialization {
        message: format!("writing dump line: {e}"),
    })
}

/// Write a collection as JSONL.
///
/// Takes the engine by reference and the config alongside it: `EngineOps` is
/// deliberately about points and edges, not about collection metadata, and
/// threading the config through a trait method purely for export would widen
/// that interface for one caller's benefit.
///
/// Output is deterministic — see the module documentation.
pub fn export_jsonl<E: EngineOps + ?Sized, W: Write>(
    engine: &E,
    config: &CollectionConfig,
    out: &mut W,
    opts: &ExportOptions,
) -> LatticeResult<Header> {
    let mut ids = engine.point_ids()?;
    ids.sort_unstable();

    // Collected before the header is written because the header states the
    // counts, and a header whose counts disagree with the body would make the
    // file unverifiable — the reader could not tell truncation from a lie.
    let mut edges: Vec<(PointId, PointId, String, f32)> = Vec::new();
    for &id in &ids {
        for e in engine.get_edges(id)? {
            edges.push((id, e.target_id, e.relation, e.weight));
        }
    }
    edges.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.1.cmp(&b.1))
    });

    let header = Header {
        format: FORMAT.to_string(),
        version: FORMAT_VERSION,
        config: config.clone(),
        vectors: opts.vectors,
        dim: engine.vector_dim(),
        points: ids.len(),
        edges: edges.len(),
    };
    write_line(out, &Line::Header(header.clone()))?;

    for &id in &ids {
        let Some(p) = engine.get_point(id)? else {
            // A point that vanished between `point_ids` and `get_point` means
            // the collection is being mutated underneath the export. Refusing
            // is the honest answer; a dump missing rows it counted in the
            // header is a corrupt artifact that looks valid.
            return Err(LatticeError::InvalidOperation {
                message: format!(
                    "point {id} disappeared during export — the collection was mutated concurrently"
                ),
            });
        };
        let payload: BTreeMap<String, serde_json::Value> = p
            .payload
            .iter()
            .map(|(k, v)| (k.clone(), payload_to_json(v)))
            .collect();
        write_line(
            out,
            &Line::Point {
                id: p.id,
                vector: match opts.vectors {
                    VectorMode::Inline => Some(p.vector.clone()),
                    VectorMode::Omit => None,
                },
                payload,
                labels: p.label_bitmap,
            },
        )?;
    }

    for (from, to, rel, w) in &edges {
        write_line(
            out,
            &Line::Edge {
                from: *from,
                to: *to,
                rel: rel.clone(),
                w: *w,
            },
        )?;
    }

    Ok(header)
}

/// Read a JSONL dump.
///
/// Rebuilds points and edges but constructs nothing: the caller decides what
/// engine and storage to fill, which keeps this side of the interchange as free
/// of I/O policy as the export side.
pub fn import_jsonl<R: BufRead>(reader: R) -> LatticeResult<Dump> {
    let mut header: Option<Header> = None;
    let mut points = Vec::new();
    let mut edges = Vec::new();

    for (n, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| LatticeError::Serialization {
            message: format!("reading dump line {}: {e}", n + 1),
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Line =
            serde_json::from_str(&line).map_err(|e| LatticeError::Serialization {
                message: format!("parsing dump line {}: {e}", n + 1),
            })?;
        match parsed {
            Line::Header(h) => {
                if header.is_some() {
                    return Err(LatticeError::Serialization {
                        message: format!("second header at line {}", n + 1),
                    });
                }
                if h.format != FORMAT {
                    return Err(LatticeError::Serialization {
                        message: format!("not a {FORMAT} dump: format is {:?}", h.format),
                    });
                }
                if h.version > FORMAT_VERSION {
                    // Refusing a newer format beats parsing it partially: an
                    // unknown line kind silently skipped is data loss that
                    // reports success.
                    return Err(LatticeError::Serialization {
                        message: format!(
                            "dump is format version {} but this build understands at most {FORMAT_VERSION}",
                            h.version
                        ),
                    });
                }
                header = Some(h);
            }
            Line::Point {
                id,
                vector,
                payload,
                labels,
            } => {
                if header.is_none() {
                    return Err(LatticeError::Serialization {
                        message: format!("point at line {} precedes the header", n + 1),
                    });
                }
                let mut p = Point::new_vector(id, vector.unwrap_or_default());
                for (k, v) in &payload {
                    p.payload.insert(k.clone(), json_to_payload(v)?);
                }
                p.label_bitmap = labels;
                points.push(p);
            }
            Line::Edge { from, to, rel, w } => {
                if header.is_none() {
                    return Err(LatticeError::Serialization {
                        message: format!("edge at line {} precedes the header", n + 1),
                    });
                }
                edges.push((from, to, rel, w));
            }
        }
    }

    let header = header.ok_or_else(|| LatticeError::Serialization {
        message: "dump has no header line".into(),
    })?;

    // The header's counts are a checksum on the body. Honouring them turns a
    // truncated file — the common failure when a writer dies mid-dump — into an
    // error instead of a silently short collection.
    if points.len() != header.points || edges.len() != header.edges {
        return Err(LatticeError::Serialization {
            message: format!(
                "dump is truncated or corrupt: header declares {} point(s) and {} edge(s), body has {} and {}",
                header.points,
                header.edges,
                points.len(),
                edges.len()
            ),
        });
    }

    Ok(Dump {
        header,
        points,
        edges,
    })
}
