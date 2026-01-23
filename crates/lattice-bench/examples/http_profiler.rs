//! HTTP Performance Profiler
//!
//! Measures where time is spent in HTTP requests to diagnose the ~220µs overhead.
//!
//! Run: cargo run -p lattice-bench --release --example http_profiler
//!
//! Prerequisites:
//! - LatticeDB server: cargo run --release -p lattice-server
//! - Qdrant (optional): docker run -p 6333:6333 qdrant/qdrant

use reqwest::blocking::Client;
use serde::Serialize;
use std::collections::HashMap;
use std::time::{Duration, Instant};

const LATTICE_URL: &str = "http://localhost:6335";
const QDRANT_URL: &str = "http://localhost:6334";
const VECTOR_DIM: usize = 128;
const ITERATIONS: usize = 100;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== HTTP Performance Profiler ===\n");
    println!("Measuring HTTP overhead components...\n");

    let client = Client::builder().timeout(Duration::from_secs(30)).build()?;

    // Check server availability
    let lattice_available = check_server(&client, LATTICE_URL);
    let qdrant_available = check_server(&client, QDRANT_URL);

    println!("Server Status:");
    println!(
        "  LatticeDB: {}",
        if lattice_available {
            "✓ Available"
        } else {
            "✗ Not running"
        }
    );
    println!(
        "  Qdrant:    {}",
        if qdrant_available {
            "✓ Available"
        } else {
            "✗ Not running"
        }
    );
    println!();

    if !lattice_available {
        eprintln!("ERROR: LatticeDB server not running!");
        eprintln!("Start it with: cargo run --release -p lattice-server");
        return Ok(());
    }

    // === Phase 1: Baseline HTTP Overhead (ping endpoint) ===
    println!("=== Phase 1: Baseline HTTP Overhead ===\n");

    let ping_times = bench_ping(&client, ITERATIONS)?;
    let ping_avg = ping_times.iter().sum::<Duration>() / ITERATIONS as u32;
    let ping_min = ping_times.iter().min().unwrap();
    let ping_max = ping_times.iter().max().unwrap();

    println!("/ping endpoint (minimal processing):");
    println!("  Avg: {:>10.2} µs", ping_avg.as_secs_f64() * 1_000_000.0);
    println!("  Min: {:>10.2} µs", ping_min.as_secs_f64() * 1_000_000.0);
    println!("  Max: {:>10.2} µs", ping_max.as_secs_f64() * 1_000_000.0);
    println!();
    println!(
        "Baseline HTTP round-trip: {:.2} µs",
        ping_avg.as_secs_f64() * 1_000_000.0
    );
    println!();

    // === Phase 2: Server-Side Timing Breakdown ===
    println!("=== Phase 2: Server-Side Timing Breakdown ===\n");

    // Setup collection
    setup_collection(&client, LATTICE_URL)?;

    // Test upsert with timing
    let (upsert_total, upsert_server_timing) = bench_with_timing(&client, "upsert", ITERATIONS)?;
    println!("Upsert:");
    println!(
        "  Total RTT:     {:>10.2} µs",
        upsert_total.as_secs_f64() * 1_000_000.0
    );
    if let Some(timing) = &upsert_server_timing {
        println!("  Server-Timing: {}", timing);
        if let Some(breakdown) = parse_server_timing(timing) {
            println!("  Breakdown:");
            println!("    Body read:   {:>10.2} µs", breakdown.body_us as f64);
            println!("    Handler:     {:>10.2} µs", breakdown.handler_us as f64);
            println!("    Total:       {:>10.2} µs", breakdown.total_us as f64);
            let network = upsert_total.as_micros() as i64 - breakdown.total_us as i64;
            println!(
                "    Network:     {:>10.2} µs (RTT - server)",
                network as f64
            );
        }
    }
    println!();

    // Test search with timing
    let (search_total, search_server_timing) = bench_search_with_timing(&client, ITERATIONS)?;
    println!("Search:");
    println!(
        "  Total RTT:     {:>10.2} µs",
        search_total.as_secs_f64() * 1_000_000.0
    );
    if let Some(timing) = &search_server_timing {
        println!("  Server-Timing: {}", timing);
        if let Some(breakdown) = parse_server_timing(timing) {
            println!("  Breakdown:");
            println!("    Body read:   {:>10.2} µs", breakdown.body_us as f64);
            println!("    Handler:     {:>10.2} µs", breakdown.handler_us as f64);
            println!("    Total:       {:>10.2} µs", breakdown.total_us as f64);
            let network = search_total.as_micros() as i64 - breakdown.total_us as i64;
            println!(
                "    Network:     {:>10.2} µs (RTT - server)",
                network as f64
            );
        }
    }
    println!();

    // === Phase 3: Compare with Qdrant ===
    if qdrant_available {
        println!("=== Phase 3: Qdrant Comparison ===\n");

        // Setup Qdrant collection
        setup_qdrant_collection(&client)?;

        let qdrant_ping = bench_qdrant_health(&client, ITERATIONS)?;
        let qdrant_upsert = bench_qdrant_upsert(&client, ITERATIONS)?;
        let qdrant_search = bench_qdrant_search(&client, ITERATIONS)?;

        println!("| Operation | LatticeDB | Qdrant   | Diff     |");
        println!("|-----------|-----------|----------|----------|");
        println!(
            "| Ping      | {:>7.2} µs | {:>6.2} µs | {:>+7.2} µs |",
            ping_avg.as_secs_f64() * 1_000_000.0,
            qdrant_ping.as_secs_f64() * 1_000_000.0,
            (ping_avg.as_secs_f64() - qdrant_ping.as_secs_f64()) * 1_000_000.0
        );
        println!(
            "| Upsert    | {:>7.2} µs | {:>6.2} µs | {:>+7.2} µs |",
            upsert_total.as_secs_f64() * 1_000_000.0,
            qdrant_upsert.as_secs_f64() * 1_000_000.0,
            (upsert_total.as_secs_f64() - qdrant_upsert.as_secs_f64()) * 1_000_000.0
        );
        println!(
            "| Search    | {:>7.2} µs | {:>6.2} µs | {:>+7.2} µs |",
            search_total.as_secs_f64() * 1_000_000.0,
            qdrant_search.as_secs_f64() * 1_000_000.0,
            (search_total.as_secs_f64() - qdrant_search.as_secs_f64()) * 1_000_000.0
        );
        println!();
    }

    // === Summary ===
    println!("=== Summary ===\n");
    println!("HTTP Overhead Analysis:");
    println!(
        "  Baseline (ping): {:.2} µs",
        ping_avg.as_secs_f64() * 1_000_000.0
    );
    if let Some(timing) = &upsert_server_timing {
        if let Some(breakdown) = parse_server_timing(timing) {
            let server_overhead = breakdown.total_us - breakdown.handler_us;
            println!(
                "  Server overhead (body+serialize): {:.2} µs",
                server_overhead as f64
            );
            let client_overhead = upsert_total.as_micros() as i64
                - breakdown.total_us as i64
                - ping_avg.as_micros() as i64;
            println!(
                "  Client JSON overhead: ~{:.2} µs",
                client_overhead.max(0) as f64
            );
        }
    }

    Ok(())
}

fn check_server(client: &Client, url: &str) -> bool {
    client
        .get(format!("{}/collections", url))
        .send()
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

fn bench_ping(
    client: &Client,
    iterations: usize,
) -> Result<Vec<Duration>, Box<dyn std::error::Error>> {
    let mut times = Vec::with_capacity(iterations);

    // Warmup
    for _ in 0..10 {
        client.get(format!("{}/ping", LATTICE_URL)).send()?;
    }

    for _ in 0..iterations {
        let start = Instant::now();
        let resp = client.get(format!("{}/ping", LATTICE_URL)).send()?;
        let elapsed = start.elapsed();

        if !resp.status().is_success() {
            return Err("Ping failed".into());
        }
        times.push(elapsed);
    }

    Ok(times)
}

fn setup_collection(client: &Client, base_url: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Delete if exists
    let _ = client
        .delete(format!("{}/collections/bench_profile", base_url))
        .send();

    // Create collection
    #[derive(Serialize)]
    struct CreateReq {
        vectors: VectorConfig,
    }
    #[derive(Serialize)]
    struct VectorConfig {
        size: usize,
        distance: String,
    }

    let body = CreateReq {
        vectors: VectorConfig {
            size: VECTOR_DIM,
            distance: "Cosine".to_string(),
        },
    };

    client
        .put(format!("{}/collections/bench_profile", base_url))
        .json(&body)
        .send()?;

    // Load some data
    #[derive(Serialize)]
    struct UpsertReq {
        points: Vec<Point>,
    }
    #[derive(Serialize)]
    struct Point {
        id: u64,
        vector: Vec<f32>,
    }

    let points: Vec<Point> = (0..1000)
        .map(|i| Point {
            id: i,
            vector: (0..VECTOR_DIM)
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect(),
        })
        .collect();

    client
        .put(format!("{}/collections/bench_profile/points", base_url))
        .json(&UpsertReq { points })
        .send()?;

    std::thread::sleep(Duration::from_millis(50));

    Ok(())
}

fn bench_with_timing(
    client: &Client,
    _op: &str,
    iterations: usize,
) -> Result<(Duration, Option<String>), Box<dyn std::error::Error>> {
    #[derive(Serialize)]
    struct UpsertReq {
        points: Vec<Point>,
    }
    #[derive(Serialize)]
    struct Point {
        id: u64,
        vector: Vec<f32>,
        payload: HashMap<String, serde_json::Value>,
    }

    let mut total = Duration::ZERO;
    let mut last_timing = None;

    // Warmup
    for i in 0..10 {
        let point = Point {
            id: 999_000 + i,
            vector: (0..VECTOR_DIM)
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect(),
            payload: HashMap::new(),
        };
        client
            .put(format!("{}/collections/bench_profile/points", LATTICE_URL))
            .json(&UpsertReq {
                points: vec![point],
            })
            .send()?;
    }

    for i in 0..iterations {
        let point = Point {
            id: 1_000_000 + i as u64,
            vector: (0..VECTOR_DIM)
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect(),
            payload: HashMap::new(),
        };

        let start = Instant::now();
        let resp = client
            .put(format!("{}/collections/bench_profile/points", LATTICE_URL))
            .json(&UpsertReq {
                points: vec![point],
            })
            .send()?;
        total += start.elapsed();

        if let Some(timing) = resp.headers().get("server-timing") {
            last_timing = Some(timing.to_str().unwrap_or("").to_string());
        }
    }

    Ok((total / iterations as u32, last_timing))
}

fn bench_search_with_timing(
    client: &Client,
    iterations: usize,
) -> Result<(Duration, Option<String>), Box<dyn std::error::Error>> {
    #[derive(Serialize)]
    struct SearchReq {
        vector: Vec<f32>,
        limit: usize,
        with_payload: bool,
        with_vector: bool,
    }

    let mut total = Duration::ZERO;
    let mut last_timing = None;

    // Warmup
    for _ in 0..10 {
        let query: Vec<f32> = (0..VECTOR_DIM)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();
        client
            .post(format!(
                "{}/collections/bench_profile/points/search",
                LATTICE_URL
            ))
            .json(&SearchReq {
                vector: query,
                limit: 10,
                with_payload: false,
                with_vector: false,
            })
            .send()?;
    }

    for _ in 0..iterations {
        let query: Vec<f32> = (0..VECTOR_DIM)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        let start = Instant::now();
        let resp = client
            .post(format!(
                "{}/collections/bench_profile/points/search",
                LATTICE_URL
            ))
            .json(&SearchReq {
                vector: query,
                limit: 10,
                with_payload: false,
                with_vector: false,
            })
            .send()?;
        total += start.elapsed();

        if let Some(timing) = resp.headers().get("server-timing") {
            last_timing = Some(timing.to_str().unwrap_or("").to_string());
        }
    }

    Ok((total / iterations as u32, last_timing))
}

#[derive(Debug)]
struct ServerTiming {
    body_us: u128,
    handler_us: u128,
    total_us: u128,
}

fn parse_server_timing(header: &str) -> Option<ServerTiming> {
    // Parse "body;dur=X, handler;dur=Y, total;dur=Z"
    let mut body_us = 0;
    let mut handler_us = 0;
    let mut total_us = 0;

    for part in header.split(',') {
        let part = part.trim();
        if let Some(dur_idx) = part.find(";dur=") {
            let name = &part[..dur_idx];
            let value: u128 = part[dur_idx + 5..].parse().unwrap_or(0);

            match name {
                "body" => body_us = value,
                "handler" => handler_us = value,
                "total" => total_us = value,
                _ => {}
            }
        }
    }

    Some(ServerTiming {
        body_us,
        handler_us,
        total_us,
    })
}

// === Qdrant helpers ===

fn setup_qdrant_collection(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    // Delete if exists
    let _ = client
        .delete(format!("{}/collections/bench_profile", QDRANT_URL))
        .send();
    std::thread::sleep(Duration::from_millis(100));

    // Create
    #[derive(Serialize)]
    struct CreateReq {
        vectors: VectorConfig,
    }
    #[derive(Serialize)]
    struct VectorConfig {
        size: usize,
        distance: String,
    }

    client
        .put(format!("{}/collections/bench_profile", QDRANT_URL))
        .json(&CreateReq {
            vectors: VectorConfig {
                size: VECTOR_DIM,
                distance: "Cosine".to_string(),
            },
        })
        .send()?;

    // Load data
    #[derive(Serialize)]
    struct UpsertReq {
        points: Vec<Point>,
    }
    #[derive(Serialize)]
    struct Point {
        id: u64,
        vector: Vec<f32>,
    }

    let points: Vec<Point> = (0..1000)
        .map(|i| Point {
            id: i,
            vector: (0..VECTOR_DIM)
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect(),
        })
        .collect();

    client
        .put(format!("{}/collections/bench_profile/points", QDRANT_URL))
        .json(&UpsertReq { points })
        .send()?;

    std::thread::sleep(Duration::from_millis(100));
    Ok(())
}

fn bench_qdrant_health(
    client: &Client,
    iterations: usize,
) -> Result<Duration, Box<dyn std::error::Error>> {
    // Warmup
    for _ in 0..10 {
        client.get(format!("{}/", QDRANT_URL)).send()?;
    }

    let mut total = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        client.get(format!("{}/", QDRANT_URL)).send()?;
        total += start.elapsed();
    }
    Ok(total / iterations as u32)
}

fn bench_qdrant_upsert(
    client: &Client,
    iterations: usize,
) -> Result<Duration, Box<dyn std::error::Error>> {
    #[derive(Serialize)]
    struct UpsertReq {
        points: Vec<Point>,
    }
    #[derive(Serialize)]
    struct Point {
        id: u64,
        vector: Vec<f32>,
    }

    // Warmup
    for i in 0..10 {
        let point = Point {
            id: 999_000 + i,
            vector: (0..VECTOR_DIM)
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect(),
        };
        client
            .put(format!("{}/collections/bench_profile/points", QDRANT_URL))
            .json(&UpsertReq {
                points: vec![point],
            })
            .send()?;
    }

    let mut total = Duration::ZERO;
    for i in 0..iterations {
        let point = Point {
            id: 1_000_000 + i as u64,
            vector: (0..VECTOR_DIM)
                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                .collect(),
        };

        let start = Instant::now();
        client
            .put(format!("{}/collections/bench_profile/points", QDRANT_URL))
            .json(&UpsertReq {
                points: vec![point],
            })
            .send()?;
        total += start.elapsed();
    }
    Ok(total / iterations as u32)
}

fn bench_qdrant_search(
    client: &Client,
    iterations: usize,
) -> Result<Duration, Box<dyn std::error::Error>> {
    #[derive(Serialize)]
    struct SearchReq {
        vector: Vec<f32>,
        limit: usize,
        with_payload: bool,
        with_vector: bool,
    }

    // Warmup
    for _ in 0..10 {
        let query: Vec<f32> = (0..VECTOR_DIM)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();
        client
            .post(format!(
                "{}/collections/bench_profile/points/search",
                QDRANT_URL
            ))
            .json(&SearchReq {
                vector: query,
                limit: 10,
                with_payload: false,
                with_vector: false,
            })
            .send()?;
    }

    let mut total = Duration::ZERO;
    for _ in 0..iterations {
        let query: Vec<f32> = (0..VECTOR_DIM)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        let start = Instant::now();
        client
            .post(format!(
                "{}/collections/bench_profile/points/search",
                QDRANT_URL
            ))
            .json(&SearchReq {
                vector: query,
                limit: 10,
                with_payload: false,
                with_vector: false,
            })
            .send()?;
        total += start.elapsed();
    }
    Ok(total / iterations as u32)
}
