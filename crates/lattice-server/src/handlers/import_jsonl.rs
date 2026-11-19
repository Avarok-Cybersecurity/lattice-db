//! Logical (lattice-jsonl) import: rebuild a collection from a JSONL dump.
//!
//! The dump names its own collection (`header.config.name`), so the caller
//! supplies only the import mode. Installation is shared with the binary
//! snapshot path via [`install_collection`].

use crate::dto::ImportMode;
use crate::handlers::export_import::install_collection;
use crate::router::AppState;
use lattice_core::{import_jsonl, CollectionEngine, Dump, LatticeResponse, VectorMode};

/// Maximum JSONL payload accepted for import (1 GiB, matching binary import).
const MAX_JSONL_SIZE: usize = 1024 * 1024 * 1024;

/// Decode a JSONL dump into a ready-to-install engine and its collection name.
///
/// Fails (400) on a malformed or truncated dump, and on a dump written with
/// vectors omitted: such a dump needs re-embedding, which this path does not
/// do, and installing zero-length vectors would produce a collection that
/// silently answers every search wrongly.
pub fn engine_from_jsonl(data: &[u8]) -> Result<(String, CollectionEngine), LatticeResponse> {
    if data.len() > MAX_JSONL_SIZE {
        return Err(LatticeResponse::bad_request(&format!(
            "JSONL payload size {} exceeds maximum of {} bytes",
            data.len(),
            MAX_JSONL_SIZE
        )));
    }
    let dump = import_jsonl(data)
        .map_err(|e| LatticeResponse::bad_request(&format!("Invalid JSONL dump: {}", e)))?;
    build_engine(dump)
}

fn build_engine(dump: Dump) -> Result<(String, CollectionEngine), LatticeResponse> {
    let Dump {
        header,
        points,
        edges,
    } = dump;
    if header.vectors == VectorMode::Omit {
        return Err(LatticeResponse::bad_request(
            "JSONL dump omits vectors; re-embed it before import or export with vectors inline",
        ));
    }
    let name = header.config.name.clone();
    let mut engine = CollectionEngine::new(header.config)
        .map_err(|e| LatticeResponse::bad_request(&format!("Invalid collection config: {}", e)))?;
    engine
        .upsert_points(points)
        .map_err(|e| LatticeResponse::bad_request(&format!("Invalid point in dump: {}", e)))?;
    for (from, to, relation, weight) in edges {
        engine
            .add_edge(from, to, &relation, weight)
            .map_err(|e| LatticeResponse::bad_request(&format!("Invalid edge in dump: {}", e)))?;
    }
    Ok((name, engine))
}

/// Import a JSONL dump into the collection it names, according to `mode`.
pub async fn import_jsonl_collection(
    state: &AppState,
    mode: ImportMode,
    data: &[u8],
) -> LatticeResponse {
    match engine_from_jsonl(data) {
        Ok((name, engine)) => install_collection(state, &name, mode, move || Ok(engine)).await,
        Err(response) => response,
    }
}
