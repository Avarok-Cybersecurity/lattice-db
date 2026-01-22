//! LatticeDB vector operation benchmark runner
//!
//! Direct vector operations (not Cypher) for fair comparison with Qdrant.

use crate::Timer;
use lattice_core::engine::collection::CollectionEngine;
use lattice_core::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
use lattice_core::types::point::Point;
use lattice_core::types::query::{ScrollQuery, SearchQuery};
use rand::prelude::*;
use std::collections::HashMap;
use std::time::Duration;

/// LatticeDB vector operation benchmark runner
pub struct LatticeVectorRunner {
    engine: CollectionEngine,
    vector_dim: usize,
}

impl LatticeVectorRunner {
    /// Create a new LatticeDB vector runner
    pub fn new(name: &str, vector_dim: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let config = CollectionConfig::new(
            name,
            VectorConfig::new(vector_dim, Distance::Cosine),
            HnswConfig {
                m: 16,
                m0: 32,
                ml: HnswConfig::recommended_ml(16),
                ef: 100,
                ef_construction: 200,
            },
        );

        let engine = CollectionEngine::new(config)?;

        Ok(Self { engine, vector_dim })
    }

    /// Generate random vector
    fn random_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
        (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    /// Load test data into LatticeDB
    pub fn load_data(&mut self, count: usize, seed: u64) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let batch_size = 100;

        for batch_start in (0..count).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(count);
            let points: Vec<Point> = (batch_start..batch_end)
                .map(|i| {
                    let mut payload = HashMap::new();
                    payload.insert(
                        "index".to_string(),
                        serde_json::to_vec(&i).unwrap(),
                    );
                    payload.insert(
                        "name".to_string(),
                        serde_json::to_vec(&format!("point_{}", i)).unwrap(),
                    );
                    payload.insert(
                        "category".to_string(),
                        serde_json::to_vec(&["A", "B", "C"][i % 3]).unwrap(),
                    );

                    Point {
                        id: i as u64,
                        vector: Self::random_vector(self.vector_dim, &mut rng),
                        payload,
                        outgoing_edges: None,
                        label_bitmap: 0,
                    }
                })
                .collect();

            self.engine.upsert_points(points)?;
        }

        Ok(())
    }

    /// Get point count
    pub fn point_count(&self) -> usize {
        self.engine.point_count()
    }

    /// Flush pending points to HNSW index
    /// Call this after load_data to ensure fair benchmark comparison
    pub fn flush_pending(&self) {
        self.engine.flush_pending();
    }

    // === Benchmark methods ===

    /// Benchmark single point upsert
    pub fn bench_upsert(&mut self, seed: u64) -> Result<Duration, Box<dyn std::error::Error>> {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut payload = HashMap::new();
        payload.insert("bench".to_string(), serde_json::to_vec(&true).unwrap());

        let point = Point {
            id: 999_999,
            vector: Self::random_vector(self.vector_dim, &mut rng),
            payload,
            outgoing_edges: None,
            label_bitmap: 0,
        };

        let timer = Timer::start();
        self.engine.upsert_points(vec![point])?;
        Ok(timer.elapsed())
    }

    /// Benchmark vector search
    pub fn bench_search(
        &self,
        k: usize,
        seed: u64,
    ) -> Result<Duration, Box<dyn std::error::Error>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let query_vector = Self::random_vector(self.vector_dim, &mut rng);

        let query = SearchQuery {
            vector: query_vector,
            limit: k,
            ef: Some(100),
            score_threshold: None,
            with_vector: false,
            with_payload: false, // Disable payload retrieval for fair benchmark
        };

        let timer = Timer::start();
        let _results = self.engine.search(query)?;
        Ok(timer.elapsed())
    }

    /// Benchmark point retrieval by ID
    pub fn bench_retrieve(&self, ids: &[u64]) -> Result<Duration, Box<dyn std::error::Error>> {
        let timer = Timer::start();
        for &id in ids {
            let _point = self.engine.get_point(id);
        }
        Ok(timer.elapsed())
    }

    /// Benchmark scroll pagination
    pub fn bench_scroll(&self, limit: usize) -> Result<Duration, Box<dyn std::error::Error>> {
        let query = ScrollQuery {
            limit,
            offset: None,
            with_vector: false,
            with_payload: true,
        };

        let timer = Timer::start();
        let _results = self.engine.scroll(query);
        Ok(timer.elapsed())
    }

    /// Benchmark batch search
    pub fn bench_batch_search(
        &self,
        queries: usize,
        k: usize,
        seed: u64,
    ) -> Result<Duration, Box<dyn std::error::Error>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut total = Duration::ZERO;

        for _ in 0..queries {
            let query_vector = Self::random_vector(self.vector_dim, &mut rng);
            let query = SearchQuery {
                vector: query_vector,
                limit: k,
                ef: Some(100),
                score_threshold: None,
                with_vector: false,
                with_payload: false,
            };

            let timer = Timer::start();
            let _results = self.engine.search(query)?;
            total += timer.elapsed();
        }

        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_vector_runner() {
        let mut runner = LatticeVectorRunner::new("test_vector", 128).unwrap();

        runner.load_data(100, 42).unwrap();
        assert_eq!(runner.point_count(), 100);

        let search_time = runner.bench_search(10, 42).unwrap();
        println!("Search time: {:?}", search_time);

        let retrieve_time = runner.bench_retrieve(&[0, 1, 2]).unwrap();
        println!("Retrieve time: {:?}", retrieve_time);
    }
}
