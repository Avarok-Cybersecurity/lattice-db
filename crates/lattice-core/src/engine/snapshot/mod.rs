//! Binary collection snapshots (`GET /collections/{name}/export`).
//!
//! # Format 2 (current)
//!
//! ```text
//! "LTDB" | version: u32 LE = 2 | config_len: u32 LE | config: JSON
//!        | body_len: u64 LE | zero padding to a 16-byte boundary | body: rkyv
//! ```
//!
//! The body ([`SnapshotBody`]) holds the points without their vectors, the
//! vectors once, contiguous and at the collection's storage precision, and —
//! when the collection has them — the HNSW graph and the BM25 postings, so a
//! load restores indexes instead of rebuilding them.
//!
//! # Format 1 (legacy, still loaded)
//!
//! `config_len: u32 | config | points_len: u32 | padding | rkyv Vec<Point>`,
//! with no index sections: loading one rebuilds the indexes. A format-1 file
//! can never start with `LTDB`: that prefix read as `config_len` is over 1 GiB,
//! larger than any accepted import.

mod legacy;

use crate::error::{LatticeError, LatticeResult};
use crate::index::hnsw::HnswGraphParts;
use crate::index::vector_codec::EncodedVectors;
use crate::text::LexicalParts;
use crate::types::collection::CollectionConfig;
use crate::types::point::Point;
use rkyv::rancor::Error as RkyvError;
use rkyv::{Archive, Deserialize, Serialize};

/// First four bytes of a format-2 (or later) snapshot.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"LTDB";
/// The format this build writes.
pub const SNAPSHOT_FORMAT: u32 = 2;
/// The headerless format written before 0.6.
pub const LEGACY_SNAPSHOT_FORMAT: u32 = 1;

const ALIGN: usize = 16;

/// Everything after the configuration.
#[derive(Archive, Serialize, Deserialize, Debug)]
pub struct SnapshotBody {
    /// Points with empty `vector`s; their vectors are in `vectors`, in order.
    pub points: Vec<Point>,
    pub vectors: EncodedVectors,
    pub hnsw: Option<HnswGraphParts>,
    pub text: Option<LexicalParts>,
}

/// A decoded snapshot, ready to install into an engine.
#[derive(Debug)]
pub struct Snapshot {
    pub config: CollectionConfig,
    /// Points with their (decoded) vectors.
    pub points: Vec<Point>,
    /// Persisted HNSW graph; `None` means "build it from the points".
    pub hnsw: Option<HnswGraphParts>,
    /// Persisted BM25 postings; `None` means "index the payloads".
    pub text: Option<LexicalParts>,
    /// Format the bytes were written in.
    pub format: u32,
}

/// Write a format-2 snapshot.
pub fn encode<'a, I>(
    config: &CollectionConfig,
    points: I,
    hnsw: Option<HnswGraphParts>,
    text: Option<LexicalParts>,
) -> LatticeResult<Vec<u8>>
where
    I: ExactSizeIterator<Item = &'a Point>,
{
    let points: Vec<&Point> = points.collect();
    let vectors = EncodedVectors::encode(
        config.vectors.as_ref().map(|v| v.storage),
        config.vector_dim(),
        points.iter().map(|p| p.vector.as_slice()),
    );
    let body = SnapshotBody {
        points: points.iter().map(|p| without_vector(p)).collect(),
        vectors,
        hnsw,
        text,
    };
    let body = rkyv::to_bytes::<RkyvError>(&body).map_err(serialization)?;
    let config = serde_json::to_vec(config).map_err(serialization)?;
    let config_len = u32::try_from(config.len()).map_err(|_| too_large("configuration"))?;

    let header = 4 + 4 + 4 + config.len() + 8;
    let padding = (ALIGN - header % ALIGN) % ALIGN;
    let mut out = Vec::with_capacity(header + padding + body.len());
    out.extend_from_slice(&SNAPSHOT_MAGIC);
    out.extend_from_slice(&SNAPSHOT_FORMAT.to_le_bytes());
    out.extend_from_slice(&config_len.to_le_bytes());
    out.extend_from_slice(&config);
    out.extend_from_slice(&(body.len() as u64).to_le_bytes());
    out.resize(out.len() + padding, 0);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Read a snapshot of any supported format.
pub fn decode(bytes: &[u8]) -> LatticeResult<Snapshot> {
    if bytes.len() < 8 {
        return Err(invalid("too short"));
    }
    if bytes[..4] != SNAPSHOT_MAGIC {
        return legacy::decode(bytes);
    }
    let format = read_u32(bytes, 4)?;
    if format != SNAPSHOT_FORMAT {
        return Err(LatticeError::Serialization {
            message: format!(
                "Unsupported snapshot format {format}: this build reads formats {LEGACY_SNAPSHOT_FORMAT} and {SNAPSHOT_FORMAT}{}",
                if format > SNAPSHOT_FORMAT {
                    "; the snapshot was written by a newer LatticeDB, upgrade to load it"
                } else {
                    ""
                }
            ),
        });
    }
    let config_len = read_u32(bytes, 8)? as usize;
    let config_end = 12usize
        .checked_add(config_len)
        .ok_or_else(|| invalid("config length overflow"))?;
    let config: CollectionConfig = serde_json::from_slice(
        bytes
            .get(12..config_end)
            .ok_or_else(|| invalid("config truncated"))?,
    )
    .map_err(serialization)?;
    let body_len = usize::try_from(read_u64(bytes, config_end)?)
        .map_err(|_| invalid("body length overflow"))?;
    let header = config_end + 8;
    let start = header + (ALIGN - header % ALIGN) % ALIGN;
    let end = start
        .checked_add(body_len)
        .ok_or_else(|| invalid("body length overflow"))?;
    let body = bytes
        .get(start..end)
        .ok_or_else(|| invalid("body truncated"))?;
    let body = deserialize_aligned::<SnapshotBody>(body)?;
    let points = attach_vectors(&config, body.points, &body.vectors)?;
    Ok(Snapshot {
        config,
        points,
        hnsw: body.hnsw,
        text: body.text,
        format,
    })
}

fn attach_vectors(
    config: &CollectionConfig,
    mut points: Vec<Point>,
    vectors: &EncodedVectors,
) -> LatticeResult<Vec<Point>> {
    let expected = config.vectors.as_ref().map(|v| v.storage);
    if vectors.storage() != expected {
        return Err(invalid(
            "vector encoding does not match the collection's storage",
        ));
    }
    let decoded = vectors.decode(config.vector_dim(), points.len())?;
    for (point, vector) in points.iter_mut().zip(decoded) {
        point.vector = vector;
    }
    Ok(points)
}

fn without_vector(point: &Point) -> Point {
    Point {
        id: point.id,
        vector: Vec::new(),
        payload: point.payload.clone(),
        outgoing_edges: point.outgoing_edges.clone(),
        label_bitmap: point.label_bitmap,
    }
}

/// Validate and deserialize rkyv bytes that may not be 16-byte aligned.
pub(crate) fn deserialize_aligned<T>(bytes: &[u8]) -> LatticeResult<T>
where
    T: Archive,
    T::Archived: for<'a> rkyv::bytecheck::CheckBytes<rkyv::api::high::HighValidator<'a, RkyvError>>
        + Deserialize<T, rkyv::api::high::HighDeserializer<RkyvError>>,
{
    if bytes.as_ptr() as usize % ALIGN == 0 {
        return rkyv::from_bytes::<T, RkyvError>(bytes).map_err(serialization);
    }
    let mut aligned = rkyv::util::AlignedVec::<ALIGN>::with_capacity(bytes.len());
    aligned.extend_from_slice(bytes);
    rkyv::from_bytes::<T, RkyvError>(&aligned).map_err(serialization)
}

fn read_u32(bytes: &[u8], at: usize) -> LatticeResult<u32> {
    let raw = bytes
        .get(at..at + 4)
        .ok_or_else(|| invalid("header truncated"))?;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

fn read_u64(bytes: &[u8], at: usize) -> LatticeResult<u64> {
    let raw = bytes
        .get(at..at + 8)
        .ok_or_else(|| invalid("header truncated"))?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(raw);
    Ok(u64::from_le_bytes(buf))
}

fn invalid(message: &str) -> LatticeError {
    LatticeError::Serialization {
        message: format!("Invalid data: {message}"),
    }
}

fn too_large(what: &str) -> LatticeError {
    LatticeError::Serialization {
        message: format!("{what} is too large for a snapshot"),
    }
}

fn serialization(e: impl std::fmt::Display) -> LatticeError {
    LatticeError::Serialization {
        message: e.to_string(),
    }
}

/// Write a format-1 snapshot from raw config JSON (tests of the legacy path).
#[cfg(test)]
pub(crate) fn legacy_for_tests(config: &[u8], points: &[Point]) -> Vec<u8> {
    legacy::encode(config, points)
}

#[cfg(test)]
mod tests;
