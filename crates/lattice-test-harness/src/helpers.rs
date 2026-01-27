//! Helper functions for ACID tests

use crate::shared_state::SharedState;
use lattice_core::engine::collection::CollectionEngineBuilder;
use lattice_core::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
use lattice_core::types::point::Point;
use lattice_core::CollectionEngine;

const DEFAULT_DIM: usize = 4;

/// Create a test point with a deterministic vector
pub fn make_point(id: u64, dim: usize) -> Point {
    let vector: Vec<f32> = (0..dim)
        .map(|i| (id as f32 * 0.1) + (i as f32 * 0.01))
        .collect();
    Point::new_vector(id, vector)
}

/// Create a test collection config
pub fn make_config(name: &str, dim: usize) -> CollectionConfig {
    CollectionConfig::new(
        name,
        VectorConfig::new(dim, Distance::Cosine),
        HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 50,
            ef_construction: 100,
        },
    )
}

/// Open a durable engine from shared state
pub async fn open_engine(
    state: &SharedState,
) -> lattice_core::LatticeResult<CollectionEngine<crate::MockStorage, crate::MockStorage>> {
    let config = make_config("test_acid", DEFAULT_DIM);
    CollectionEngineBuilder::new(config)
        .with_wal(state.mock())
        .with_data(state.mock())
        .open()
        .await
}
