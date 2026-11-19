//! Format 1: `[config_len: u32][config JSON][points_len: u32][padding][rkyv Vec<Point>]`.
//!
//! Written by LatticeDB before 0.6. It carries no index sections, so the
//! caller rebuilds the HNSW graph and BM25 index from the points.

use super::LEGACY_SNAPSHOT_FORMAT;
use super::{deserialize_aligned, invalid, read_u32, serialization, Snapshot, ALIGN};
use crate::error::LatticeResult;
use crate::types::collection::CollectionConfig;
use crate::types::point::Point;

pub(super) fn decode(bytes: &[u8]) -> LatticeResult<Snapshot> {
    let config_len = read_u32(bytes, 0)? as usize;
    let config_end = 4usize
        .checked_add(config_len)
        .ok_or_else(|| invalid("config length overflow"))?;
    let config: CollectionConfig = serde_json::from_slice(
        bytes
            .get(4..config_end)
            .ok_or_else(|| invalid("config truncated"))?,
    )
    .map_err(serialization)?;
    let points_len = read_u32(bytes, config_end)? as usize;
    let header = config_end + 4;
    let start = header + (ALIGN - header % ALIGN) % ALIGN;
    let end = start
        .checked_add(points_len)
        .ok_or_else(|| invalid("total size overflow"))?;
    let raw = bytes
        .get(start..end)
        .ok_or_else(|| invalid("points truncated"))?;
    let points = deserialize_aligned::<Vec<Point>>(raw)?;
    Ok(Snapshot {
        config,
        points,
        hnsw: None,
        text: None,
        format: LEGACY_SNAPSHOT_FORMAT,
    })
}

/// Write format 1 (tests only: proves old snapshots still load).
#[cfg(test)]
pub(super) fn encode(config: &[u8], points: &[Point]) -> Vec<u8> {
    let body = rkyv::to_bytes::<rkyv::rancor::Error>(&points.to_vec()).expect("points serialize");
    let header = 4 + config.len() + 4;
    let padding = (ALIGN - header % ALIGN) % ALIGN;
    let mut out = Vec::new();
    out.extend_from_slice(&(config.len() as u32).to_le_bytes());
    out.extend_from_slice(config);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.resize(out.len() + padding, 0);
    out.extend_from_slice(&body);
    out
}
