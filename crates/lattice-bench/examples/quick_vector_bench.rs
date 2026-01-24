//! Quick vector benchmark comparison
//!
//! Run: cargo run -p lattice-bench --release --example quick_vector_bench

use lattice_bench::lattice_vector_runner::LatticeVectorRunner;
use lattice_bench::qdrant_runner::{qdrant_available, QdrantRunner};
use std::time::{Duration, Instant};

const VECTOR_DIM: usize = 128;
const SEED: u64 = 42;
const DATASET_SIZE: usize = 1000;
const SEARCH_K: usize = 10;
const ITERATIONS: usize = 100;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LatticeDB vs Qdrant Vector Benchmark ===\n");
    println!("Dataset: {} points, {}D vectors", DATASET_SIZE, VECTOR_DIM);
    println!("Search: k={}", SEARCH_K);
    println!("Iterations: {}\n", ITERATIONS);

    // Setup LatticeDB
    println!("Setting up LatticeDB...");
    let setup_start = Instant::now();
    let mut lattice = LatticeVectorRunner::new("bench_quick", VECTOR_DIM)?;
    lattice.load_data(DATASET_SIZE, SEED)?;
    lattice.flush_pending()?; // Ensure all points are indexed before benchmark
    let lattice_setup = setup_start.elapsed();
    println!("LatticeDB setup: {:?}\n", lattice_setup);

    // Setup Qdrant if available (try port 6334 first, then 6333)
    let qdrant = if qdrant_available() {
        println!("Setting up Qdrant...");
        let setup_start = Instant::now();
        // Try port 6334 first (common Docker mapping), then 6333
        let url = if reqwest::blocking::Client::new()
            .get("http://localhost:6334/collections")
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
        {
            "http://localhost:6334"
        } else {
            "http://localhost:6333"
        };
        let runner = QdrantRunner::new(url, "bench_quick", VECTOR_DIM)?;
        runner.create_collection()?;
        runner.load_data(DATASET_SIZE, SEED)?;
        let qdrant_setup = setup_start.elapsed();
        println!("Qdrant setup: {:?}\n", qdrant_setup);
        Some(runner)
    } else {
        println!("Qdrant not available, skipping\n");
        None
    };

    // Benchmark results
    let mut results = Vec::new();

    // 1. Upsert benchmark
    println!("--- Upsert Benchmark ---");
    let lattice_upsert = bench_iterations(ITERATIONS, || lattice.bench_upsert(SEED + 1).unwrap());
    println!(
        "LatticeDB: {:>12.2} µs/op",
        lattice_upsert.as_secs_f64() * 1_000_000.0
    );

    if let Some(ref q) = qdrant {
        let qdrant_upsert = bench_iterations(ITERATIONS, || q.bench_upsert(SEED + 1).unwrap());
        println!(
            "Qdrant:    {:>12.2} µs/op",
            qdrant_upsert.as_secs_f64() * 1_000_000.0
        );
        println!(
            "Speedup:   {:>12.1}x",
            qdrant_upsert.as_secs_f64() / lattice_upsert.as_secs_f64()
        );
        results.push(("upsert", lattice_upsert, qdrant_upsert));
    }
    println!();

    // 2. Search benchmark
    println!("--- Search Benchmark (k={}) ---", SEARCH_K);
    let lattice_search =
        bench_iterations(ITERATIONS, || lattice.bench_search(SEARCH_K, SEED).unwrap());
    println!(
        "LatticeDB: {:>12.2} µs/op",
        lattice_search.as_secs_f64() * 1_000_000.0
    );

    if let Some(ref q) = qdrant {
        let qdrant_search =
            bench_iterations(ITERATIONS, || q.bench_search(SEARCH_K, SEED).unwrap());
        println!(
            "Qdrant:    {:>12.2} µs/op",
            qdrant_search.as_secs_f64() * 1_000_000.0
        );
        println!(
            "Speedup:   {:>12.1}x",
            qdrant_search.as_secs_f64() / lattice_search.as_secs_f64()
        );
        results.push(("search", lattice_search, qdrant_search));
    }
    println!();

    // 3. Retrieve benchmark
    println!("--- Retrieve Benchmark (10 IDs) ---");
    let ids: Vec<u64> = (0..10).collect();
    let lattice_retrieve = bench_iterations(ITERATIONS, || lattice.bench_retrieve(&ids).unwrap());
    println!(
        "LatticeDB: {:>12.2} µs/op",
        lattice_retrieve.as_secs_f64() * 1_000_000.0
    );

    if let Some(ref q) = qdrant {
        let qdrant_retrieve = bench_iterations(ITERATIONS, || q.bench_retrieve(&ids).unwrap());
        println!(
            "Qdrant:    {:>12.2} µs/op",
            qdrant_retrieve.as_secs_f64() * 1_000_000.0
        );
        println!(
            "Speedup:   {:>12.1}x",
            qdrant_retrieve.as_secs_f64() / lattice_retrieve.as_secs_f64()
        );
        results.push(("retrieve", lattice_retrieve, qdrant_retrieve));
    }
    println!();

    // 4. Scroll benchmark
    println!("--- Scroll Benchmark (limit=100) ---");
    let lattice_scroll = bench_iterations(ITERATIONS, || lattice.bench_scroll(100).unwrap());
    println!(
        "LatticeDB: {:>12.2} µs/op",
        lattice_scroll.as_secs_f64() * 1_000_000.0
    );

    if let Some(ref q) = qdrant {
        let qdrant_scroll = bench_iterations(ITERATIONS, || q.bench_scroll(100).unwrap());
        println!(
            "Qdrant:    {:>12.2} µs/op",
            qdrant_scroll.as_secs_f64() * 1_000_000.0
        );
        println!(
            "Speedup:   {:>12.1}x",
            qdrant_scroll.as_secs_f64() / lattice_scroll.as_secs_f64()
        );
        results.push(("scroll", lattice_scroll, qdrant_scroll));
    }
    println!();

    // Summary
    if !results.is_empty() {
        println!("=== SUMMARY ===");
        println!(
            "{:<12} {:>15} {:>15} {:>10}",
            "Operation", "LatticeDB (µs)", "Qdrant (µs)", "Speedup"
        );
        println!("{}", "-".repeat(55));
        for (op, lattice_time, qdrant_time) in &results {
            let speedup = qdrant_time.as_secs_f64() / lattice_time.as_secs_f64();
            println!(
                "{:<12} {:>15.2} {:>15.2} {:>9.1}x",
                op,
                lattice_time.as_secs_f64() * 1_000_000.0,
                qdrant_time.as_secs_f64() * 1_000_000.0,
                speedup
            );
        }

        // Output JSON for chart generation
        println!("\n=== JSON for Charts ===");
        println!("{{");
        println!("  \"dataset_size\": {},", DATASET_SIZE);
        println!("  \"vector_dim\": {},", VECTOR_DIM);
        println!("  \"results\": [");
        for (i, (op, lattice_time, qdrant_time)) in results.iter().enumerate() {
            let comma = if i < results.len() - 1 { "," } else { "" };
            println!(
                "    {{\"operation\": \"{}\", \"lattice_us\": {:.2}, \"qdrant_us\": {:.2}}}{}",
                op,
                lattice_time.as_secs_f64() * 1_000_000.0,
                qdrant_time.as_secs_f64() * 1_000_000.0,
                comma
            );
        }
        println!("  ]");
        println!("}}");
    }

    Ok(())
}

fn bench_iterations<F>(iterations: usize, mut f: F) -> Duration
where
    F: FnMut() -> Duration,
{
    let mut total = Duration::ZERO;
    for _ in 0..iterations {
        total += f();
    }
    total / iterations as u32
}
