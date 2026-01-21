//! LatticeDB vs Neo4j Cypher Benchmark Suite
//!
//! This crate provides benchmarks comparing LatticeDB's Cypher implementation
//! against Neo4j for common graph database operations.
//!
//! ## Running Benchmarks
//!
//! ### Prerequisites
//!
//! 1. Start a Neo4j instance (Docker recommended):
//!    ```bash
//!    docker run -d \
//!      --name neo4j-bench \
//!      -p 7687:7687 \
//!      -p 7474:7474 \
//!      -e NEO4J_AUTH=neo4j/benchmarkpassword \
//!      -e NEO4J_PLUGINS='["apoc"]' \
//!      neo4j:5
//!    ```
//!
//! 2. Run the benchmarks:
//!    ```bash
//!    cargo bench -p lattice-bench
//!    ```
//!
//! ## Benchmark Categories
//!
//! - **Node Creation**: Single and batch node creation
//! - **Node Lookup**: Label scan, property filter, ID lookup
//! - **Pattern Matching**: Simple patterns, multi-hop traversals
//! - **Aggregation**: Count, filtering with LIMIT/SKIP

pub mod datasets;
pub mod http_runner;
pub mod lattice_runner;
pub mod neo4j_runner;
pub mod results;

use std::time::{Duration, Instant};

/// Result of a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub lattice_duration: Duration,
    pub neo4j_duration: Duration,
    pub iterations: u32,
    pub data_size: usize,
}

impl BenchmarkResult {
    pub fn speedup(&self) -> f64 {
        self.neo4j_duration.as_secs_f64() / self.lattice_duration.as_secs_f64()
    }

    pub fn print_summary(&self) {
        println!("\n=== {} ===", self.name);
        println!("Data size: {} nodes", self.data_size);
        println!("Iterations: {}", self.iterations);
        println!(
            "LatticeDB: {:>10.3}ms (avg: {:>8.3}µs/iter)",
            self.lattice_duration.as_secs_f64() * 1000.0,
            self.lattice_duration.as_secs_f64() * 1_000_000.0 / self.iterations as f64
        );
        println!(
            "Neo4j:     {:>10.3}ms (avg: {:>8.3}µs/iter)",
            self.neo4j_duration.as_secs_f64() * 1000.0,
            self.neo4j_duration.as_secs_f64() * 1_000_000.0 / self.iterations as f64
        );
        let speedup = self.speedup();
        if speedup >= 1.0 {
            println!("LatticeDB is {:.2}x faster", speedup);
        } else {
            println!("Neo4j is {:.2}x faster", 1.0 / speedup);
        }
    }
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub neo4j_uri: String,
    pub neo4j_user: String,
    pub neo4j_password: String,
    pub warmup_iterations: u32,
    pub benchmark_iterations: u32,
    pub data_sizes: Vec<usize>,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            neo4j_uri: "bolt://localhost:7687".to_string(),
            neo4j_user: "neo4j".to_string(),
            neo4j_password: "benchmarkpassword".to_string(),
            warmup_iterations: 5,
            benchmark_iterations: 100,
            data_sizes: vec![100, 1000, 10000],
        }
    }
}

/// Timer utility for benchmarking
pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}
