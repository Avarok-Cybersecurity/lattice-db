//! LatticeDB durable vector operation benchmark runner
//!
//! Direct vector operations with ACID mode enabled for measuring WAL overhead.
//! Uses in-memory storage to isolate WAL overhead from disk I/O.

use crate::Timer;
use lattice_core::engine::collection::{CollectionEngine, CollectionEngineBuilder};
use lattice_core::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
use lattice_core::types::point::Point;
use lattice_core::types::query::{ScrollQuery, SearchQuery};
use lattice_storage::MemStorage;
use rand::prelude::*;
use std::collections::HashMap;
use std::time::Duration;

/// LatticeDB durable vector operation benchmark runner
///
/// Uses in-memory storage with WAL enabled to measure ACID overhead
/// without disk I/O latency.
pub struct LatticeDurableRunner {
    engine: CollectionEngine<MemStorage, MemStorage>,
    vector_dim: usize,
}

impl LatticeDurableRunner {
    /// Create a new LatticeDB durable runner with in-memory WAL
    pub async fn new(name: &str, vector_dim: usize) -> Result<Self, Box<dyn std::error::Error>> {
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

        // Create separate in-memory storage for WAL and data
        let wal_storage = MemStorage::new();
        let data_storage = MemStorage::new();

        let engine = CollectionEngineBuilder::new(config)
            .with_wal(wal_storage)
            .with_data(data_storage)
            .open()
            .await?;

        Ok(Self { engine, vector_dim })
    }

    /// Generate random vector
    fn random_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
        (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    /// Load test data into LatticeDB
    pub async fn load_data(
        &mut self,
        count: usize,
        seed: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let batch_size = 100;

        for batch_start in (0..count).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(count);
            let points: Vec<Point> = (batch_start..batch_end)
                .map(|i| {
                    let mut payload = HashMap::new();
                    payload.insert("index".to_string(), serde_json::to_vec(&i).unwrap());
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

            self.engine.upsert_points_async(points).await?;
        }

        Ok(())
    }

    /// Get point count
    pub fn point_count(&self) -> usize {
        self.engine.point_count()
    }

    /// Flush pending points to HNSW index
    pub fn flush_pending(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.engine.flush_pending()?;
        Ok(())
    }

    // === Benchmark methods ===

    /// Benchmark single point upsert with WAL
    pub async fn bench_upsert(&mut self, seed: u64) -> Result<Duration, Box<dyn std::error::Error>> {
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
        self.engine.upsert_points_async(vec![point]).await?;
        Ok(timer.elapsed())
    }

    /// Benchmark batch upsert with WAL (single sync for N points)
    pub async fn bench_upsert_batch(
        &mut self,
        count: usize,
        seed: u64,
    ) -> Result<Duration, Box<dyn std::error::Error>> {
        let mut rng = StdRng::seed_from_u64(seed);

        let batch: Vec<Vec<Point>> = (0..count)
            .map(|i| {
                let mut payload = HashMap::new();
                payload.insert("bench".to_string(), serde_json::to_vec(&true).unwrap());

                vec![Point {
                    id: 900_000 + i as u64,
                    vector: Self::random_vector(self.vector_dim, &mut rng),
                    payload,
                    outgoing_edges: None,
                    label_bitmap: 0,
                }]
            })
            .collect();

        let timer = Timer::start();
        self.engine.upsert_batch_async(batch).await?;
        Ok(timer.elapsed())
    }

    /// Benchmark vector search (same as ephemeral - reads don't use WAL)
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
            with_payload: false,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lattice_durable_runner() {
        let mut runner = LatticeDurableRunner::new("test_durable", 128)
            .await
            .unwrap();

        runner.load_data(100, 42).await.unwrap();
        assert_eq!(runner.point_count(), 100);

        let search_time = runner.bench_search(10, 42).unwrap();
        println!("Durable Search time: {:?}", search_time);

        let retrieve_time = runner.bench_retrieve(&[0, 1, 2]).unwrap();
        println!("Durable Retrieve time: {:?}", retrieve_time);

        let upsert_time = runner.bench_upsert(42).await.unwrap();
        println!("Durable Upsert time: {:?}", upsert_time);
    }

    #[tokio::test]
    async fn test_acid_upsert_overhead() {
        // Test with 5000 points to match benchmark
        let mut runner = LatticeDurableRunner::new("test_acid_overhead", 128)
            .await
            .unwrap();

        runner.load_data(5000, 42).await.unwrap();
        assert_eq!(runner.point_count(), 5000);

        // Run 100 upserts and measure time
        let iters = 100;
        let start = std::time::Instant::now();
        for i in 0..iters {
            let _ = runner.bench_upsert(42 + i).await.unwrap();
        }
        let elapsed = start.elapsed();

        println!("ACID upserts with 5000 points:");
        println!("  Total time for {} upserts: {:?}", iters, elapsed);
        println!("  Average time per upsert: {:?}", elapsed / iters as u32);
    }

    #[tokio::test]
    async fn test_acid_upsert_many_iterations() {
        // Test with 5000 points and many iterations
        let mut runner = LatticeDurableRunner::new("test_acid_many", 128)
            .await
            .unwrap();

        runner.load_data(5000, 42).await.unwrap();
        assert_eq!(runner.point_count(), 5000);

        // Run 10000 upserts (matching criterion benchmark)
        let iters = 10000;
        let start = std::time::Instant::now();
        for i in 0..iters {
            let _ = runner.bench_upsert(42 + i).await.unwrap();
        }
        let elapsed = start.elapsed();

        println!("ACID upserts with 5000 points and {} iterations:", iters);
        println!("  Total time: {:?}", elapsed);
        println!("  Average time per upsert: {:?}", elapsed / iters as u32);
    }

    #[test]
    fn test_acid_upsert_with_block_on() {
        // Test the exact pattern used in criterion benchmark
        let rt = tokio::runtime::Runtime::new().unwrap();

        for size in [1000, 5000] {
            println!("Testing size {}", size);

            let mut runner = rt.block_on(async {
                let mut r = LatticeDurableRunner::new(&format!("test_block_on_{}", size), 128)
                    .await
                    .unwrap();
                r.load_data(size, 42).await.unwrap();
                r
            });
            println!("  Loaded {} points", runner.point_count());

            // Run 100 upserts with block_on pattern
            let iters = 100;
            let start = std::time::Instant::now();
            for i in 0..iters {
                rt.block_on(async {
                    runner.bench_upsert(42 + i as u64).await.unwrap()
                });
            }
            let elapsed = start.elapsed();
            println!("  {} upserts in {:?} (avg {:?})", iters, elapsed, elapsed / iters);
        }
    }

    #[tokio::test]
    async fn test_batch_vs_single_upsert() {
        println!("\n=== Single vs Batch ACID Upsert Comparison ===\n");

        let count = 100;

        for size in [1000, 5000] {
            println!("Collection size: {}", size);

            // Single upserts (100x sync)
            let mut runner = LatticeDurableRunner::new(&format!("test_single_{}", size), 128)
                .await
                .unwrap();
            runner.load_data(size, 42).await.unwrap();

            let start = std::time::Instant::now();
            for i in 0..count {
                runner.bench_upsert(42 + i as u64).await.unwrap();
            }
            let single_time = start.elapsed();
            let single_avg = single_time / count;
            println!("  Single:  {:?} total, {:?} avg", single_time, single_avg);

            // Batch upsert (1x sync)
            let mut runner2 = LatticeDurableRunner::new(&format!("test_batch_{}", size), 128)
                .await
                .unwrap();
            runner2.load_data(size, 42).await.unwrap();

            let start = std::time::Instant::now();
            runner2.bench_upsert_batch(count as usize, 42).await.unwrap();
            let batch_time = start.elapsed();
            let batch_avg = batch_time / count;
            println!("  Batch:   {:?} total, {:?} avg", batch_time, batch_avg);

            let speedup = single_time.as_nanos() as f64 / batch_time.as_nanos() as f64;
            println!("  Speedup: {:.1}x\n", speedup);
        }
    }

    #[test]
    fn test_ephemeral_vs_acid_comparison() {
        use crate::lattice_vector_runner::LatticeVectorRunner;

        let rt = tokio::runtime::Runtime::new().unwrap();
        let iters = 1000;

        println!("\n=== Ephemeral vs ACID Upsert Comparison ===\n");

        for size in [1000, 5000] {
            println!("Collection size: {}", size);

            // Ephemeral
            let mut ephemeral =
                LatticeVectorRunner::new(&format!("test_ephemeral_{}", size), 128).unwrap();
            ephemeral.load_data(size, 42).unwrap();

            let start = std::time::Instant::now();
            for i in 0..iters {
                let _ = ephemeral.bench_upsert(42 + i as u64).unwrap();
            }
            let ephemeral_time = start.elapsed();
            let ephemeral_avg = ephemeral_time / iters;
            println!("  Ephemeral: {:?} total, {:?} avg", ephemeral_time, ephemeral_avg);

            // ACID
            let mut acid = rt.block_on(async {
                let mut r = LatticeDurableRunner::new(&format!("test_acid_{}", size), 128)
                    .await
                    .unwrap();
                r.load_data(size, 42).await.unwrap();
                r
            });

            let start = std::time::Instant::now();
            for i in 0..iters {
                rt.block_on(async { acid.bench_upsert(42 + i as u64).await.unwrap() });
            }
            let acid_time = start.elapsed();
            let acid_avg = acid_time / iters;
            println!("  ACID:      {:?} total, {:?} avg", acid_time, acid_avg);

            let overhead = acid_avg.as_nanos() as f64 / ephemeral_avg.as_nanos() as f64;
            println!("  Overhead:  {:.1}x\n", overhead);

            // Prevent drop from blocking
            std::mem::forget(ephemeral);
        }
    }
}
