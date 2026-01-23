//! Graph Performance Profiler (In-Memory)
//!
//! Measures Cypher query performance: LatticeDB in-memory (no network) vs Neo4j (Bolt protocol)
//!
//! This benchmark shows the RAW query engine performance without HTTP overhead.
//! For fair HTTP-vs-protocol comparison, use http_graph_profiler instead.
//!
//! Run: cargo run -p lattice-bench --release --example graph_profiler
//!
//! Prerequisites:
//! - Neo4j: docker run -d -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/benchmarkpassword neo4j:5

use lattice_bench::datasets::generate_people;
use lattice_bench::lattice_runner::LatticeRunner;
use lattice_bench::neo4j_runner::Neo4jRunner;
use std::time::Duration;
use tokio::runtime::Runtime;

const DATASET_SIZE: usize = 1000;
const SEED: u64 = 42;
const ITERATIONS: usize = 100;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Graph Performance Profiler (In-Memory) ===\n");
    println!("Dataset: {} nodes", DATASET_SIZE);
    println!("Iterations: {}\n", ITERATIONS);

    let rt = Runtime::new()?;

    // Check Neo4j availability
    let neo4j_available = rt.block_on(async {
        match Neo4jRunner::connect("bolt://localhost:7687", "neo4j", "benchmarkpassword").await {
            Ok(r) => r.query("RETURN 1").await.is_ok(),
            Err(_) => false,
        }
    });

    println!("Server Status:");
    println!("  LatticeDB: ✓ In-Memory (no HTTP)");
    println!(
        "  Neo4j:     {}",
        if neo4j_available {
            "✓ Available"
        } else {
            "✗ Not running"
        }
    );
    println!();

    // Setup LatticeDB
    println!("Setting up LatticeDB...");
    let mut lattice = LatticeRunner::new("bench_graph")?;
    let people = generate_people(DATASET_SIZE, SEED);
    lattice.load_people(&people)?;
    println!("LatticeDB ready: {} nodes\n", lattice.node_count());

    // Setup Neo4j if available
    let neo4j = if neo4j_available {
        println!("Setting up Neo4j...");
        let runner = rt
            .block_on(async {
                let r = Neo4jRunner::connect("bolt://localhost:7687", "neo4j", "benchmarkpassword")
                    .await?;
                r.clear().await?;
                r.load_people(&people).await?;
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>(r)
            })
            .ok();
        if runner.is_some() {
            println!("Neo4j ready\n");
        }
        runner
    } else {
        println!("Neo4j not available, skipping\n");
        None
    };

    // Benchmark results
    let mut results = Vec::new();

    // 1. MATCH (n) RETURN n LIMIT 100
    println!("--- match_all: MATCH (n) RETURN n LIMIT 100 ---");
    let lattice_time = bench_lattice_match_all(&mut lattice)?;
    println!(
        "LatticeDB: {:>10.2} µs/op",
        lattice_time.as_secs_f64() * 1_000_000.0
    );

    if let Some(ref neo) = neo4j {
        let neo4j_time = bench_neo4j_match_all(&rt, neo)?;
        println!(
            "Neo4j:     {:>10.2} µs/op",
            neo4j_time.as_secs_f64() * 1_000_000.0
        );
        let speedup = neo4j_time.as_secs_f64() / lattice_time.as_secs_f64();
        println!("Speedup:   {:>10.1}x", speedup);
        results.push(("match_all", lattice_time, neo4j_time));
    }
    println!();

    // 2. MATCH (n:Person) RETURN n LIMIT 100
    println!("--- match_by_label: MATCH (n:Person) RETURN n LIMIT 100 ---");
    let lattice_time = bench_lattice_match_label(&mut lattice)?;
    println!(
        "LatticeDB: {:>10.2} µs/op",
        lattice_time.as_secs_f64() * 1_000_000.0
    );

    if let Some(ref neo) = neo4j {
        let neo4j_time = bench_neo4j_match_label(&rt, neo)?;
        println!(
            "Neo4j:     {:>10.2} µs/op",
            neo4j_time.as_secs_f64() * 1_000_000.0
        );
        let speedup = neo4j_time.as_secs_f64() / lattice_time.as_secs_f64();
        println!("Speedup:   {:>10.1}x", speedup);
        results.push(("match_by_label", lattice_time, neo4j_time));
    }
    println!();

    // 3. MATCH (n:Person) RETURN n LIMIT 10
    println!("--- match_with_limit: MATCH (n:Person) RETURN n LIMIT 10 ---");
    let lattice_time = bench_lattice_match_limit(&mut lattice)?;
    println!(
        "LatticeDB: {:>10.2} µs/op",
        lattice_time.as_secs_f64() * 1_000_000.0
    );

    if let Some(ref neo) = neo4j {
        let neo4j_time = bench_neo4j_match_limit(&rt, neo)?;
        println!(
            "Neo4j:     {:>10.2} µs/op",
            neo4j_time.as_secs_f64() * 1_000_000.0
        );
        let speedup = neo4j_time.as_secs_f64() / lattice_time.as_secs_f64();
        println!("Speedup:   {:>10.1}x", speedup);
        results.push(("match_with_limit", lattice_time, neo4j_time));
    }
    println!();

    // 4. WHERE filter
    println!("--- where_property: MATCH (n:Person) WHERE n.age > 30 RETURN n ---");
    let lattice_time = bench_lattice_filter(&mut lattice)?;
    println!(
        "LatticeDB: {:>10.2} µs/op",
        lattice_time.as_secs_f64() * 1_000_000.0
    );

    if let Some(ref neo) = neo4j {
        let neo4j_time = bench_neo4j_filter(&rt, neo)?;
        println!(
            "Neo4j:     {:>10.2} µs/op",
            neo4j_time.as_secs_f64() * 1_000_000.0
        );
        let speedup = neo4j_time.as_secs_f64() / lattice_time.as_secs_f64();
        println!("Speedup:   {:>10.1}x", speedup);
        results.push(("where_property", lattice_time, neo4j_time));
    }
    println!();

    // 5. ORDER BY
    println!("--- order_by: MATCH (n:Person) RETURN n.name ORDER BY n.name LIMIT 50 ---");
    let lattice_time = bench_lattice_order(&mut lattice)?;
    println!(
        "LatticeDB: {:>10.2} µs/op",
        lattice_time.as_secs_f64() * 1_000_000.0
    );

    if let Some(ref neo) = neo4j {
        let neo4j_time = bench_neo4j_order(&rt, neo)?;
        println!(
            "Neo4j:     {:>10.2} µs/op",
            neo4j_time.as_secs_f64() * 1_000_000.0
        );
        let speedup = neo4j_time.as_secs_f64() / lattice_time.as_secs_f64();
        println!("Speedup:   {:>10.1}x", speedup);
        results.push(("order_by", lattice_time, neo4j_time));
    }
    println!();

    // Summary
    if !results.is_empty() {
        println!("=== SUMMARY ===");
        println!(
            "{:<20} {:>15} {:>15} {:>10}",
            "Operation", "LatticeDB (µs)", "Neo4j (µs)", "Speedup"
        );
        println!("{}", "-".repeat(65));
        for (op, lattice_t, neo4j_t) in &results {
            let speedup = neo4j_t.as_secs_f64() / lattice_t.as_secs_f64();
            println!(
                "{:<20} {:>15.2} {:>15.2} {:>9.1}x",
                op,
                lattice_t.as_secs_f64() * 1_000_000.0,
                neo4j_t.as_secs_f64() * 1_000_000.0,
                speedup
            );
        }
    }

    Ok(())
}

// LatticeDB benchmark helpers
fn bench_lattice_match_all(
    runner: &mut LatticeRunner,
) -> Result<Duration, Box<dyn std::error::Error>> {
    for _ in 0..10 {
        let _ = runner.bench_match_all();
    }
    let mut total = Duration::ZERO;
    for _ in 0..ITERATIONS {
        total += runner.bench_match_all()?;
    }
    Ok(total / ITERATIONS as u32)
}

fn bench_lattice_match_label(
    runner: &mut LatticeRunner,
) -> Result<Duration, Box<dyn std::error::Error>> {
    for _ in 0..10 {
        let _ = runner.bench_match_by_label();
    }
    let mut total = Duration::ZERO;
    for _ in 0..ITERATIONS {
        total += runner.bench_match_by_label()?;
    }
    Ok(total / ITERATIONS as u32)
}

fn bench_lattice_match_limit(
    runner: &mut LatticeRunner,
) -> Result<Duration, Box<dyn std::error::Error>> {
    for _ in 0..10 {
        let _ = runner.bench_match_by_label_limit();
    }
    let mut total = Duration::ZERO;
    for _ in 0..ITERATIONS {
        total += runner.bench_match_by_label_limit()?;
    }
    Ok(total / ITERATIONS as u32)
}

fn bench_lattice_filter(
    runner: &mut LatticeRunner,
) -> Result<Duration, Box<dyn std::error::Error>> {
    for _ in 0..10 {
        let _ = runner.bench_match_with_filter(30);
    }
    let mut total = Duration::ZERO;
    for _ in 0..ITERATIONS {
        total += runner.bench_match_with_filter(30)?;
    }
    Ok(total / ITERATIONS as u32)
}

fn bench_lattice_order(runner: &mut LatticeRunner) -> Result<Duration, Box<dyn std::error::Error>> {
    for _ in 0..10 {
        let _ = runner.bench_order_by();
    }
    let mut total = Duration::ZERO;
    for _ in 0..ITERATIONS {
        total += runner.bench_order_by()?;
    }
    Ok(total / ITERATIONS as u32)
}

// Neo4j benchmark helpers
fn bench_neo4j_match_all(
    rt: &Runtime,
    runner: &Neo4jRunner,
) -> Result<Duration, Box<dyn std::error::Error>> {
    for _ in 0..10 {
        let _ = rt.block_on(runner.bench_match_all());
    }
    let mut total = Duration::ZERO;
    for _ in 0..ITERATIONS {
        total += rt
            .block_on(runner.bench_match_all())
            .map_err(|e| e.to_string())?;
    }
    Ok(total / ITERATIONS as u32)
}

fn bench_neo4j_match_label(
    rt: &Runtime,
    runner: &Neo4jRunner,
) -> Result<Duration, Box<dyn std::error::Error>> {
    for _ in 0..10 {
        let _ = rt.block_on(runner.bench_match_by_label());
    }
    let mut total = Duration::ZERO;
    for _ in 0..ITERATIONS {
        total += rt
            .block_on(runner.bench_match_by_label())
            .map_err(|e| e.to_string())?;
    }
    Ok(total / ITERATIONS as u32)
}

fn bench_neo4j_match_limit(
    rt: &Runtime,
    runner: &Neo4jRunner,
) -> Result<Duration, Box<dyn std::error::Error>> {
    for _ in 0..10 {
        let _ = rt.block_on(runner.bench_match_by_label_limit());
    }
    let mut total = Duration::ZERO;
    for _ in 0..ITERATIONS {
        total += rt
            .block_on(runner.bench_match_by_label_limit())
            .map_err(|e| e.to_string())?;
    }
    Ok(total / ITERATIONS as u32)
}

fn bench_neo4j_filter(
    rt: &Runtime,
    runner: &Neo4jRunner,
) -> Result<Duration, Box<dyn std::error::Error>> {
    for _ in 0..10 {
        let _ = rt.block_on(runner.bench_match_with_filter(30));
    }
    let mut total = Duration::ZERO;
    for _ in 0..ITERATIONS {
        total += rt
            .block_on(runner.bench_match_with_filter(30))
            .map_err(|e| e.to_string())?;
    }
    Ok(total / ITERATIONS as u32)
}

fn bench_neo4j_order(
    rt: &Runtime,
    runner: &Neo4jRunner,
) -> Result<Duration, Box<dyn std::error::Error>> {
    for _ in 0..10 {
        let _ = rt.block_on(runner.bench_order_by());
    }
    let mut total = Duration::ZERO;
    for _ in 0..ITERATIONS {
        total += rt
            .block_on(runner.bench_order_by())
            .map_err(|e| e.to_string())?;
    }
    Ok(total / ITERATIONS as u32)
}
