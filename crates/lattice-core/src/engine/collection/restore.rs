//! Shared by both engines: what a collection's vector config means for
//! writes and searches, and how a decoded snapshot becomes indexes.

use crate::engine::snapshot::Snapshot;
use crate::error::{LatticeError, LatticeResult};
use crate::index::distance::DistanceCalculator;
use crate::index::flat::exact_search;
use crate::index::hnsw::HnswIndex;
use crate::text::PayloadTextIndex;
use crate::types::collection::{CollectionConfig, VectorIndexKind};
use crate::types::point::{Point, PointId};
use crate::types::query::SearchResult;
use std::collections::BTreeMap;

/// Validate a point's vector against the collection.
pub(crate) fn check_vector(config: &CollectionConfig, point: &Point) -> LatticeResult<()> {
    match &config.vectors {
        Some(vectors) if point.vector.len() != vectors.size => {
            Err(LatticeError::DimensionMismatch {
                expected: vectors.size,
                actual: point.vector.len(),
            })
        }
        None if !point.vector.is_empty() => Err(LatticeError::InvalidOperation {
            message: format!(
                "collection '{}' has no dense vectors; send points without a vector",
                config.name
            ),
        }),
        _ => Ok(()),
    }
}

/// Validate a point's vector and round it to the collection's storage
/// precision (idempotent, so WAL replay reproduces it exactly).
pub(crate) fn prepare_vector(config: &CollectionConfig, point: &mut Point) -> LatticeResult<()> {
    check_vector(config, point)?;
    if let Some(vectors) = &config.vectors {
        vectors.storage.round_in_place(&mut point.vector);
    }
    Ok(())
}

/// Validate a query vector; the error names the problem (no dense vectors,
/// or a dimension mismatch).
pub(crate) fn check_query(config: &CollectionConfig, query: &[f32]) -> LatticeResult<()> {
    let vectors = config.dense()?;
    if query.len() != vectors.size {
        return Err(LatticeError::DimensionMismatch {
            expected: vectors.size,
            actual: query.len(),
        });
    }
    Ok(())
}

/// Exact search over `points` (for `index: flat`).
pub(crate) fn flat_search(
    config: &CollectionConfig,
    points: &BTreeMap<PointId, Point>,
    query: &[f32],
    limit: usize,
) -> LatticeResult<Vec<SearchResult>> {
    let calc = DistanceCalculator::new(config.dense()?.distance);
    Ok(exact_search(
        &calc,
        query,
        points.iter().map(|(id, p)| (*id, p.vector.as_slice())),
        limit,
    ))
}

/// A snapshot's contents as an engine holds them.
pub(crate) struct Restored {
    pub config: CollectionConfig,
    pub points: BTreeMap<PointId, Point>,
    /// `Some` exactly when the collection uses an HNSW index.
    pub index: Option<HnswIndex>,
    pub text_index: Option<PayloadTextIndex>,
}

/// Turn a decoded snapshot into points and indexes: persisted sections are
/// restored as-is, missing ones (format-1 snapshots) are rebuilt.
pub(crate) fn restore(snapshot: Snapshot) -> LatticeResult<Restored> {
    let config = snapshot.config;
    config.validate()?;
    let count = snapshot.points.len();
    let mut points = BTreeMap::new();
    for mut point in snapshot.points {
        prepare_vector(&config, &mut point)?;
        points.insert(point.id, point);
    }
    if points.len() != count {
        return Err(corrupt("duplicate point id"));
    }
    let index = restore_hnsw(&config, &points, snapshot.hnsw)?;
    let text_index = match (&config.text_index, snapshot.text) {
        (Some(text), Some(parts)) => Some(PayloadTextIndex::from_parts(text, parts)?),
        (Some(text), None) => {
            let mut built = PayloadTextIndex::new(text);
            points.values().for_each(|p| built.upsert_point(p));
            Some(built)
        }
        (None, Some(_)) => return Err(corrupt("text section without a text_index")),
        (None, None) => None,
    };
    Ok(Restored {
        config,
        points,
        index,
        text_index,
    })
}

/// A new, empty HNSW index when the collection uses one.
pub(crate) fn new_hnsw(config: &CollectionConfig) -> Option<HnswIndex> {
    match &config.vectors {
        Some(v) if v.index == VectorIndexKind::Hnsw => {
            Some(HnswIndex::new(config.hnsw.clone(), v.distance))
        }
        _ => None,
    }
}

fn restore_hnsw(
    config: &CollectionConfig,
    points: &BTreeMap<PointId, Point>,
    parts: Option<crate::index::hnsw::HnswGraphParts>,
) -> LatticeResult<Option<HnswIndex>> {
    let Some(mut index) = new_hnsw(config) else {
        return match parts {
            Some(_) => Err(corrupt(
                "HNSW section for a collection without an HNSW index",
            )),
            None => Ok(None),
        };
    };
    match parts {
        Some(parts) => {
            let distance = config.dense()?.distance;
            HnswIndex::from_parts(
                config.hnsw.clone(),
                distance,
                config.vector_dim(),
                parts,
                points.iter().map(|(id, p)| (*id, p.vector.as_slice())),
            )
            .map(Some)
        }
        None => {
            for point in points.values() {
                index.insert(point)?;
            }
            Ok(Some(index))
        }
    }
}

fn corrupt(message: &str) -> LatticeError {
    LatticeError::Serialization {
        message: format!("Invalid snapshot: {message}"),
    }
}
