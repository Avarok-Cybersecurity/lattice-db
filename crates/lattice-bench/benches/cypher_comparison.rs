//! Cypher Query Benchmark: LatticeDB vs Neo4j
//!
//! This benchmark compares the performance of Cypher query execution
//! between LatticeDB and Neo4j across various query patterns.
//!
//! ## Prerequisites
//!
//! Start Neo4j with Docker:
//! ```bash
//! docker run -d \
//!   --name neo4j-bench \
//!   -p 7687:7687 \
//!   -p 7474:7474 \
//!   -e NEO4J_AUTH=neo4j/benchmarkpassword \
//!   neo4j:5
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo bench -p lattice-bench
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lattice_bench::datasets::{generate_people, generate_relationships};
use lattice_bench::lattice_runner::LatticeRunner;
use lattice_bench::neo4j_runner::Neo4jRunner;
use std::time::Duration;
use tokio::runtime::Runtime;

const SEED: u64 = 42;

/// Check if Neo4j is available by attempting to connect and run a simple query
fn neo4j_available(rt: &Runtime) -> bool {
    rt.block_on(async {
        match Neo4jRunner::connect("bolt://localhost:7687", "neo4j", "benchmarkpassword").await {
            Ok(runner) => {
                // Try a simple query to verify connection works
                runner.query("RETURN 1").await.is_ok()
            }
            Err(_) => false,
        }
    })
}

/// Setup LatticeDB with test data
fn setup_lattice(size: usize) -> LatticeRunner {
    let mut runner = LatticeRunner::new("bench_collection").expect("Failed to create LatticeDB");
    let people = generate_people(size, SEED);
    runner.load_people(&people).expect("Failed to load people");
    runner
}

/// Setup Neo4j with test data
async fn setup_neo4j(size: usize) -> Result<Neo4jRunner, Box<dyn std::error::Error + Send + Sync>> {
    let runner =
        Neo4jRunner::connect("bolt://localhost:7687", "neo4j", "benchmarkpassword").await?;
    runner.clear().await?;
    let people = generate_people(size, SEED);
    runner.load_people(&people).await?;
    Ok(runner)
}

// === Benchmark Functions ===

fn bench_match_all(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let has_neo4j = neo4j_available(&rt);

    let mut group = c.benchmark_group("match_all");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for size in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // LatticeDB
        let mut lattice = setup_lattice(*size);
        group.bench_with_input(BenchmarkId::new("LatticeDB", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    lattice
                        .bench_match_all()
                        .expect("LatticeDB match_all failed"),
                )
            })
        });
        drop(lattice);

        // Neo4j (if available)
        if has_neo4j {
            let neo4j = rt
                .block_on(setup_neo4j(*size))
                .expect("Failed to setup Neo4j");
            group.bench_with_input(BenchmarkId::new("Neo4j", size), size, |b, _| {
                b.to_async(&rt)
                    .iter(|| async { black_box(neo4j.bench_match_all().await.unwrap()) })
            });
        }
    }

    group.finish();
}

fn bench_match_by_label(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let has_neo4j = neo4j_available(&rt);

    let mut group = c.benchmark_group("match_by_label");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for size in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // LatticeDB
        let mut lattice = setup_lattice(*size);
        group.bench_with_input(BenchmarkId::new("LatticeDB", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    lattice
                        .bench_match_by_label()
                        .expect("LatticeDB match_by_label failed"),
                )
            })
        });
        drop(lattice);

        // Neo4j (if available)
        if has_neo4j {
            let neo4j = rt
                .block_on(setup_neo4j(*size))
                .expect("Failed to setup Neo4j");
            group.bench_with_input(BenchmarkId::new("Neo4j", size), size, |b, _| {
                b.to_async(&rt)
                    .iter(|| async { black_box(neo4j.bench_match_by_label().await.unwrap()) })
            });
        }
    }

    group.finish();
}

fn bench_match_with_limit(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let has_neo4j = neo4j_available(&rt);

    let mut group = c.benchmark_group("match_with_limit");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for size in [100, 500, 1000].iter() {
        // LatticeDB
        let mut lattice = setup_lattice(*size);
        group.bench_with_input(BenchmarkId::new("LatticeDB", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    lattice
                        .bench_match_by_label_limit()
                        .expect("LatticeDB match_with_limit failed"),
                )
            })
        });
        drop(lattice);

        // Neo4j (if available)
        if has_neo4j {
            let neo4j = rt
                .block_on(setup_neo4j(*size))
                .expect("Failed to setup Neo4j");
            group.bench_with_input(BenchmarkId::new("Neo4j", size), size, |b, _| {
                b.to_async(&rt)
                    .iter(|| async { black_box(neo4j.bench_match_by_label_limit().await.unwrap()) })
            });
        }
    }

    group.finish();
}

fn bench_match_with_filter(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let has_neo4j = neo4j_available(&rt);

    let mut group = c.benchmark_group("match_with_filter");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for size in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // LatticeDB
        let mut lattice = setup_lattice(*size);
        group.bench_with_input(BenchmarkId::new("LatticeDB", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    lattice
                        .bench_match_with_filter(30)
                        .expect("LatticeDB match_with_filter failed"),
                )
            })
        });
        drop(lattice);

        // Neo4j (if available)
        if has_neo4j {
            let neo4j = rt
                .block_on(setup_neo4j(*size))
                .expect("Failed to setup Neo4j");
            group.bench_with_input(BenchmarkId::new("Neo4j", size), size, |b, _| {
                b.to_async(&rt)
                    .iter(|| async { black_box(neo4j.bench_match_with_filter(30).await.unwrap()) })
            });
        }
    }

    group.finish();
}

fn bench_complex_filter(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let has_neo4j = neo4j_available(&rt);

    let mut group = c.benchmark_group("complex_filter");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for size in [100, 500, 1000].iter() {
        // LatticeDB
        let mut lattice = setup_lattice(*size);
        group.bench_with_input(BenchmarkId::new("LatticeDB", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    lattice
                        .bench_match_complex_filter(30, "New York")
                        .expect("LatticeDB complex_filter failed"),
                )
            })
        });
        drop(lattice);

        // Neo4j (if available)
        if has_neo4j {
            let neo4j = rt
                .block_on(setup_neo4j(*size))
                .expect("Failed to setup Neo4j");
            group.bench_with_input(BenchmarkId::new("Neo4j", size), size, |b, _| {
                b.to_async(&rt).iter(|| async {
                    black_box(
                        neo4j
                            .bench_match_complex_filter(30, "New York")
                            .await
                            .unwrap(),
                    )
                })
            });
        }
    }

    group.finish();
}

fn bench_projection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let has_neo4j = neo4j_available(&rt);

    let mut group = c.benchmark_group("projection");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for size in [100, 500, 1000].iter() {
        // LatticeDB
        let mut lattice = setup_lattice(*size);
        group.bench_with_input(BenchmarkId::new("LatticeDB", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    lattice
                        .bench_projection()
                        .expect("LatticeDB projection failed"),
                )
            })
        });
        drop(lattice);

        // Neo4j (if available)
        if has_neo4j {
            let neo4j = rt
                .block_on(setup_neo4j(*size))
                .expect("Failed to setup Neo4j");
            group.bench_with_input(BenchmarkId::new("Neo4j", size), size, |b, _| {
                b.to_async(&rt)
                    .iter(|| async { black_box(neo4j.bench_projection().await.unwrap()) })
            });
        }
    }

    group.finish();
}

fn bench_order_by(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let has_neo4j = neo4j_available(&rt);

    let mut group = c.benchmark_group("order_by");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for size in [100, 500, 1000, 5000, 10000, 20000, 50000].iter() {
        // LatticeDB
        let mut lattice = setup_lattice(*size);
        group.bench_with_input(BenchmarkId::new("LatticeDB", size), size, |b, _| {
            b.iter(|| black_box(lattice.bench_order_by().expect("LatticeDB order_by failed")))
        });
        drop(lattice);

        // Neo4j (if available)
        if has_neo4j {
            let neo4j = rt
                .block_on(setup_neo4j(*size))
                .expect("Failed to setup Neo4j");
            group.bench_with_input(BenchmarkId::new("Neo4j", size), size, |b, _| {
                b.to_async(&rt)
                    .iter(|| async { black_box(neo4j.bench_order_by().await.unwrap()) })
            });
        }
    }

    group.finish();
}

fn bench_skip_limit(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let has_neo4j = neo4j_available(&rt);

    let mut group = c.benchmark_group("skip_limit");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for size in [200, 500, 1000].iter() {
        // LatticeDB
        let mut lattice = setup_lattice(*size);
        group.bench_with_input(BenchmarkId::new("LatticeDB", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    lattice
                        .bench_skip_limit()
                        .expect("LatticeDB skip_limit failed"),
                )
            })
        });
        drop(lattice);

        // Neo4j (if available)
        if has_neo4j {
            let neo4j = rt
                .block_on(setup_neo4j(*size))
                .expect("Failed to setup Neo4j");
            group.bench_with_input(BenchmarkId::new("Neo4j", size), size, |b, _| {
                b.to_async(&rt)
                    .iter(|| async { black_box(neo4j.bench_skip_limit().await.unwrap()) })
            });
        }
    }

    group.finish();
}

fn bench_node_creation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let has_neo4j = neo4j_available(&rt);
    let people = generate_people(100, SEED);

    let mut group = c.benchmark_group("node_creation");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    // LatticeDB single node creation
    let mut lattice = LatticeRunner::new("bench_create").expect("Failed to create LatticeDB");
    group.bench_function("LatticeDB/single", |b| {
        let mut idx = 0;
        b.iter(|| {
            let person = &people[idx % people.len()];
            idx += 1;
            black_box(
                lattice
                    .bench_create_single(person)
                    .expect("LatticeDB create failed"),
            )
        })
    });
    drop(lattice);

    // Neo4j single node creation
    if has_neo4j {
        if let Ok(neo4j) = rt.block_on(Neo4jRunner::connect(
            "bolt://localhost:7687",
            "neo4j",
            "benchmarkpassword",
        )) {
            if rt.block_on(neo4j.clear()).is_ok() {
                group.bench_function("Neo4j/single", |b| {
                    let mut idx = 0;
                    b.iter(|| {
                        let person = &people[idx % people.len()];
                        idx += 1;
                        // Run async code synchronously for this benchmark
                        black_box(rt.block_on(neo4j.bench_create_single(person)).unwrap())
                    })
                });
            }
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_node_creation,
    bench_match_all,
    bench_match_by_label,
    bench_match_with_limit,
    bench_match_with_filter,
    bench_complex_filter,
    bench_projection,
    bench_order_by,
    bench_skip_limit,
);

criterion_main!(benches);
