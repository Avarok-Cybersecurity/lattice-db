//! HTTP Graph Performance Profiler
//!
//! Compares LatticeDB HTTP Cypher queries vs Neo4j Bolt protocol.
//! This is the fair HTTP-vs-protocol comparison for graph operations.
//!
//! Run: cargo run -p lattice-bench --release --example http_graph_profiler
//!
//! Prerequisites:
//! - LatticeDB server: cargo run --release -p lattice-server -- --port 6335
//! - Neo4j: docker run -d -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/benchmarkpassword neo4j:5

use lattice_bench::datasets::generate_people;
use lattice_bench::http_runner::HttpRunner;
use lattice_bench::neo4j_runner::Neo4jRunner;
use std::time::Duration;
use tokio::runtime::Runtime;

const LATTICE_URL: &str = "http://localhost:6335";
const COLLECTION_NAME: &str = "bench_graph_http";
const DATASET_SIZE: usize = 1000;
const SEED: u64 = 42;
const ITERATIONS: usize = 100;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== HTTP Graph Performance Profiler ===\n");
    println!("Comparing: LatticeDB HTTP vs Neo4j Bolt\n");
    println!("Dataset: {} nodes", DATASET_SIZE);
    println!("Iterations: {}\n", ITERATIONS);

    let rt = Runtime::new()?;

    // Check server availability and setup
    let lattice = rt.block_on(async {
        match HttpRunner::new(LATTICE_URL, COLLECTION_NAME).await {
            Ok(r) if r.is_available().await => Some(r),
            _ => None,
        }
    });

    let neo4j_available = rt.block_on(async {
        match Neo4jRunner::connect("bolt://localhost:7687", "neo4j", "benchmarkpassword").await {
            Ok(r) => r.query("RETURN 1").await.is_ok(),
            Err(_) => false,
        }
    });

    println!("Server Status:");
    println!(
        "  LatticeDB HTTP: {}",
        if lattice.is_some() {
            "✓ Available"
        } else {
            "✗ Not running"
        }
    );
    println!(
        "  Neo4j Bolt:     {}",
        if neo4j_available {
            "✓ Available"
        } else {
            "✗ Not running"
        }
    );
    println!();

    let lattice = match lattice {
        Some(l) => l,
        None => {
            eprintln!("ERROR: LatticeDB server not running!");
            eprintln!("Start it with: cargo run --release -p lattice-server -- --port 6335");
            return Ok(());
        }
    };

    // Generate and load test data
    let people = generate_people(DATASET_SIZE, SEED);

    println!("Setting up LatticeDB HTTP collection...");
    rt.block_on(async {
        lattice.clear().await?;
        lattice.load_people(&people).await?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
    })
    .map_err(|e| -> Box<dyn std::error::Error> {
        Box::new(std::io::Error::other(e.to_string()))
    })?;
    println!("LatticeDB ready: {} nodes via HTTP\n", people.len());

    // Setup Neo4j if available
    let neo4j = if neo4j_available {
        println!("Setting up Neo4j...");
        let runner = rt.block_on(async {
            let r =
                Neo4jRunner::connect("bolt://localhost:7687", "neo4j", "benchmarkpassword").await?;
            r.clear().await?;
            r.load_people(&people).await?;
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(r)
        });
        match runner {
            Ok(r) => {
                println!("Neo4j ready\n");
                Some(r)
            }
            Err(e) => {
                println!("Neo4j setup failed: {}\n", e);
                None
            }
        }
    } else {
        println!("Neo4j not available, skipping\n");
        None
    };

    // Benchmark results
    let mut results: Vec<(&str, Duration, Option<Duration>)> = Vec::new();

    // 1. MATCH (n) RETURN n LIMIT 100
    println!("--- match_all: MATCH (n) RETURN n LIMIT 100 ---");
    let lattice_time = bench_lattice_http(&rt, &lattice, "MATCH (n) RETURN n LIMIT 100")?;
    println!(
        "LatticeDB HTTP: {:>10.2} µs/op",
        lattice_time.as_secs_f64() * 1_000_000.0
    );

    if let Some(ref neo) = neo4j {
        let neo4j_time = bench_neo4j_query(&rt, neo, "MATCH (n) RETURN n LIMIT 100")?;
        println!(
            "Neo4j Bolt:     {:>10.2} µs/op",
            neo4j_time.as_secs_f64() * 1_000_000.0
        );
        let speedup = neo4j_time.as_secs_f64() / lattice_time.as_secs_f64();
        println!("Speedup:        {:>10.1}x", speedup);
        results.push(("match_all", lattice_time, Some(neo4j_time)));
    } else {
        results.push(("match_all", lattice_time, None));
    }
    println!();

    // 2. MATCH (n:Person) RETURN n LIMIT 100
    println!("--- match_by_label: MATCH (n:Person) RETURN n LIMIT 100 ---");
    let lattice_time = bench_lattice_http(&rt, &lattice, "MATCH (n:Person) RETURN n LIMIT 100")?;
    println!(
        "LatticeDB HTTP: {:>10.2} µs/op",
        lattice_time.as_secs_f64() * 1_000_000.0
    );

    if let Some(ref neo) = neo4j {
        let neo4j_time = bench_neo4j_query(&rt, neo, "MATCH (n:Person) RETURN n LIMIT 100")?;
        println!(
            "Neo4j Bolt:     {:>10.2} µs/op",
            neo4j_time.as_secs_f64() * 1_000_000.0
        );
        let speedup = neo4j_time.as_secs_f64() / lattice_time.as_secs_f64();
        println!("Speedup:        {:>10.1}x", speedup);
        results.push(("match_by_label", lattice_time, Some(neo4j_time)));
    } else {
        results.push(("match_by_label", lattice_time, None));
    }
    println!();

    // 3. MATCH (n:Person) RETURN n LIMIT 10
    println!("--- match_with_limit: MATCH (n:Person) RETURN n LIMIT 10 ---");
    let lattice_time = bench_lattice_http(&rt, &lattice, "MATCH (n:Person) RETURN n LIMIT 10")?;
    println!(
        "LatticeDB HTTP: {:>10.2} µs/op",
        lattice_time.as_secs_f64() * 1_000_000.0
    );

    if let Some(ref neo) = neo4j {
        let neo4j_time = bench_neo4j_query(&rt, neo, "MATCH (n:Person) RETURN n LIMIT 10")?;
        println!(
            "Neo4j Bolt:     {:>10.2} µs/op",
            neo4j_time.as_secs_f64() * 1_000_000.0
        );
        let speedup = neo4j_time.as_secs_f64() / lattice_time.as_secs_f64();
        println!("Speedup:        {:>10.1}x", speedup);
        results.push(("match_with_limit", lattice_time, Some(neo4j_time)));
    } else {
        results.push(("match_with_limit", lattice_time, None));
    }
    println!();

    // 4. WHERE filter
    println!("--- where_property: MATCH (n:Person) WHERE n.age > 30 RETURN n ---");
    let lattice_time =
        bench_lattice_http(&rt, &lattice, "MATCH (n:Person) WHERE n.age > 30 RETURN n")?;
    println!(
        "LatticeDB HTTP: {:>10.2} µs/op",
        lattice_time.as_secs_f64() * 1_000_000.0
    );

    if let Some(ref neo) = neo4j {
        let neo4j_time = bench_neo4j_query(&rt, neo, "MATCH (n:Person) WHERE n.age > 30 RETURN n")?;
        println!(
            "Neo4j Bolt:     {:>10.2} µs/op",
            neo4j_time.as_secs_f64() * 1_000_000.0
        );
        let speedup = neo4j_time.as_secs_f64() / lattice_time.as_secs_f64();
        println!("Speedup:        {:>10.1}x", speedup);
        results.push(("where_property", lattice_time, Some(neo4j_time)));
    } else {
        results.push(("where_property", lattice_time, None));
    }
    println!();

    // 5. ORDER BY
    println!("--- order_by: MATCH (n:Person) RETURN n.name ORDER BY n.name LIMIT 50 ---");
    let lattice_time = bench_lattice_http(
        &rt,
        &lattice,
        "MATCH (n:Person) RETURN n.name ORDER BY n.name LIMIT 50",
    )?;
    println!(
        "LatticeDB HTTP: {:>10.2} µs/op",
        lattice_time.as_secs_f64() * 1_000_000.0
    );

    if let Some(ref neo) = neo4j {
        let neo4j_time = bench_neo4j_query(
            &rt,
            neo,
            "MATCH (n:Person) RETURN n.name ORDER BY n.name LIMIT 50",
        )?;
        println!(
            "Neo4j Bolt:     {:>10.2} µs/op",
            neo4j_time.as_secs_f64() * 1_000_000.0
        );
        let speedup = neo4j_time.as_secs_f64() / lattice_time.as_secs_f64();
        println!("Speedup:        {:>10.1}x", speedup);
        results.push(("order_by", lattice_time, Some(neo4j_time)));
    } else {
        results.push(("order_by", lattice_time, None));
    }
    println!();

    // Summary
    if results.iter().any(|(_, _, neo)| neo.is_some()) {
        println!("=== SUMMARY: LatticeDB HTTP vs Neo4j Bolt ===");
        println!(
            "{:<20} {:>18} {:>18} {:>10}",
            "Operation", "LatticeDB HTTP", "Neo4j Bolt", "Speedup"
        );
        println!("{}", "-".repeat(70));
        for (op, lattice_t, neo4j_t) in &results {
            if let Some(neo_t) = neo4j_t {
                let speedup = neo_t.as_secs_f64() / lattice_t.as_secs_f64();
                println!(
                    "{:<20} {:>15.0} µs {:>15.0} µs {:>9.1}x",
                    op,
                    lattice_t.as_secs_f64() * 1_000_000.0,
                    neo_t.as_secs_f64() * 1_000_000.0,
                    speedup
                );
            }
        }
        println!();
        println!("Note: LatticeDB HTTP uses POST /collections/{{name}}/graph/query endpoint");
        println!("      Neo4j Bolt uses the official Bolt protocol (binary, port 7687)");
    }

    Ok(())
}

fn bench_lattice_http(
    rt: &Runtime,
    runner: &HttpRunner,
    query: &str,
) -> Result<Duration, Box<dyn std::error::Error>> {
    // Warmup
    for _ in 0..10 {
        let _ = rt.block_on(runner.query(query));
    }

    // Benchmark
    let mut total = Duration::ZERO;
    for _ in 0..ITERATIONS {
        let start = std::time::Instant::now();
        rt.block_on(runner.query(query))
            .map_err(|e| e.to_string())?;
        total += start.elapsed();
    }

    Ok(total / ITERATIONS as u32)
}

fn bench_neo4j_query(
    rt: &Runtime,
    runner: &Neo4jRunner,
    query: &str,
) -> Result<Duration, Box<dyn std::error::Error>> {
    // Warmup
    for _ in 0..10 {
        let _ = rt.block_on(runner.query(query));
    }

    // Benchmark
    let mut total = Duration::ZERO;
    for _ in 0..ITERATIONS {
        let start = std::time::Instant::now();
        rt.block_on(runner.query(query))
            .map_err(|e| e.to_string())?;
        total += start.elapsed();
    }

    Ok(total / ITERATIONS as u32)
}
