//! Vector operation benchmarks: LatticeDB vs Qdrant
//!
//! ## Prerequisites
//!
//! For HTTP benchmarks, start LatticeDB server:
//! ```bash
//! cargo run --release -p lattice-server
//! ```
//!
//! Start Qdrant:
//! ```bash
//! docker run -d --name qdrant-bench -p 6333:6333 qdrant/qdrant
//! ```
//!
//! ## Running
//!
//! ```bash
//! # Run all vector benchmarks (in-memory + HTTP)
//! cargo bench -p lattice-bench -- vector
//!
//! # Run only in-memory benchmarks
//! cargo bench -p lattice-bench -- "vector_(upsert|search|retrieve|scroll)/"
//!
//! # Run only HTTP benchmarks (requires LatticeDB server)
//! cargo bench -p lattice-bench -- "http_vector_"
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lattice_bench::http_vector_runner::{lattice_http_available, HttpVectorRunner};
use lattice_bench::lattice_vector_runner::LatticeVectorRunner;
use lattice_bench::qdrant_runner::{qdrant_available, QdrantRunner};
use std::time::Duration;

const VECTOR_DIM: usize = 128;
const SEED: u64 = 42;

fn setup_lattice(size: usize) -> LatticeVectorRunner {
    let mut runner =
        LatticeVectorRunner::new(&format!("bench_vector_{}", size), VECTOR_DIM).unwrap();
    runner.load_data(size, SEED).unwrap();
    runner
}

fn setup_qdrant(size: usize) -> Result<QdrantRunner, Box<dyn std::error::Error>> {
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
    let runner = QdrantRunner::new(
        url,
        &format!("bench_vector_{}", size),
        VECTOR_DIM,
    )?;
    runner.create_collection()?;
    runner.load_data(size, SEED)?;
    Ok(runner)
}

fn setup_http_lattice(size: usize) -> Result<HttpVectorRunner, Box<dyn std::error::Error>> {
    // Use port 6335 when Qdrant is on 6334, otherwise 6334
    let url = if qdrant_available() {
        "http://localhost:6335"
    } else {
        "http://localhost:6334"
    };
    let runner = HttpVectorRunner::new(
        url,
        &format!("bench_http_vector_{}", size),
        VECTOR_DIM,
    )?;
    runner.create_collection()?;
    runner.load_data(size, SEED)?;
    Ok(runner)
}

fn bench_upsert(c: &mut Criterion) {
    let has_qdrant = qdrant_available();

    let mut group = c.benchmark_group("vector_upsert");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for size in [1000, 5000].iter() {
        // LatticeDB
        let mut lattice = setup_lattice(*size);
        group.bench_with_input(BenchmarkId::new("LatticeDB", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    lattice
                        .bench_upsert(SEED + 1)
                        .expect("LatticeDB upsert failed"),
                )
            })
        });
        drop(lattice);

        // Qdrant (if available)
        if has_qdrant {
            if let Ok(qdrant) = setup_qdrant(*size) {
                group.bench_with_input(BenchmarkId::new("Qdrant", size), size, |b, _| {
                    b.iter(|| {
                        black_box(qdrant.bench_upsert(SEED + 1).expect("Qdrant upsert failed"))
                    })
                });
            }
        }
    }

    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let has_qdrant = qdrant_available();

    let mut group = c.benchmark_group("vector_search");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for size in [1000, 5000].iter() {
        // LatticeDB
        let lattice = setup_lattice(*size);
        group.bench_with_input(BenchmarkId::new("LatticeDB", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    lattice
                        .bench_search(10, SEED)
                        .expect("LatticeDB search failed"),
                )
            })
        });
        drop(lattice);

        // Qdrant (if available)
        if has_qdrant {
            if let Ok(qdrant) = setup_qdrant(*size) {
                group.bench_with_input(BenchmarkId::new("Qdrant", size), size, |b, _| {
                    b.iter(|| {
                        black_box(qdrant.bench_search(10, SEED).expect("Qdrant search failed"))
                    })
                });
            }
        }
    }

    group.finish();
}

fn bench_retrieve(c: &mut Criterion) {
    let has_qdrant = qdrant_available();

    let mut group = c.benchmark_group("vector_retrieve");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let ids: Vec<u64> = (0..10).collect();

    for size in [1000, 5000].iter() {
        // LatticeDB
        let lattice = setup_lattice(*size);
        group.bench_with_input(BenchmarkId::new("LatticeDB", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    lattice
                        .bench_retrieve(&ids)
                        .expect("LatticeDB retrieve failed"),
                )
            })
        });
        drop(lattice);

        // Qdrant (if available)
        if has_qdrant {
            if let Ok(qdrant) = setup_qdrant(*size) {
                group.bench_with_input(BenchmarkId::new("Qdrant", size), size, |b, _| {
                    b.iter(|| {
                        black_box(qdrant.bench_retrieve(&ids).expect("Qdrant retrieve failed"))
                    })
                });
            }
        }
    }

    group.finish();
}

fn bench_scroll(c: &mut Criterion) {
    let has_qdrant = qdrant_available();

    let mut group = c.benchmark_group("vector_scroll");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for size in [1000, 5000].iter() {
        // LatticeDB
        let lattice = setup_lattice(*size);
        group.bench_with_input(BenchmarkId::new("LatticeDB", size), size, |b, _| {
            b.iter(|| black_box(lattice.bench_scroll(100).expect("LatticeDB scroll failed")))
        });
        drop(lattice);

        // Qdrant (if available)
        if has_qdrant {
            if let Ok(qdrant) = setup_qdrant(*size) {
                group.bench_with_input(BenchmarkId::new("Qdrant", size), size, |b, _| {
                    b.iter(|| black_box(qdrant.bench_scroll(100).expect("Qdrant scroll failed")))
                });
            }
        }
    }

    group.finish();
}

// === HTTP-based benchmarks (fair comparison - all over network) ===

fn bench_http_upsert(c: &mut Criterion) {
    let has_lattice = lattice_http_available();
    let has_qdrant = qdrant_available();

    if !has_lattice && !has_qdrant {
        eprintln!("HTTP benchmarks: No servers available (LatticeDB/Qdrant). Skipping.");
        return;
    }

    let mut group = c.benchmark_group("http_vector_upsert");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for size in [1000, 5000].iter() {
        // LatticeDB HTTP
        if has_lattice {
            if let Ok(runner) = setup_http_lattice(*size) {
                group.bench_with_input(BenchmarkId::new("LatticeDB_HTTP", size), size, |b, _| {
                    b.iter(|| {
                        black_box(runner.bench_upsert(SEED + 1).expect("LatticeDB HTTP upsert failed"))
                    })
                });
            }
        }

        // Qdrant HTTP
        if has_qdrant {
            if let Ok(qdrant) = setup_qdrant(*size) {
                group.bench_with_input(BenchmarkId::new("Qdrant_HTTP", size), size, |b, _| {
                    b.iter(|| black_box(qdrant.bench_upsert(SEED + 1).expect("Qdrant upsert failed")))
                });
            }
        }
    }

    group.finish();
}

fn bench_http_search(c: &mut Criterion) {
    let has_lattice = lattice_http_available();
    let has_qdrant = qdrant_available();

    if !has_lattice && !has_qdrant {
        return;
    }

    let mut group = c.benchmark_group("http_vector_search");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for size in [1000, 5000].iter() {
        // LatticeDB HTTP
        if has_lattice {
            if let Ok(runner) = setup_http_lattice(*size) {
                group.bench_with_input(BenchmarkId::new("LatticeDB_HTTP", size), size, |b, _| {
                    b.iter(|| {
                        black_box(runner.bench_search(10, SEED).expect("LatticeDB HTTP search failed"))
                    })
                });
            }
        }

        // Qdrant HTTP
        if has_qdrant {
            if let Ok(qdrant) = setup_qdrant(*size) {
                group.bench_with_input(BenchmarkId::new("Qdrant_HTTP", size), size, |b, _| {
                    b.iter(|| black_box(qdrant.bench_search(10, SEED).expect("Qdrant search failed")))
                });
            }
        }
    }

    group.finish();
}

fn bench_http_retrieve(c: &mut Criterion) {
    let has_lattice = lattice_http_available();
    let has_qdrant = qdrant_available();

    if !has_lattice && !has_qdrant {
        return;
    }

    let mut group = c.benchmark_group("http_vector_retrieve");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let ids: Vec<u64> = (0..10).collect();

    for size in [1000, 5000].iter() {
        // LatticeDB HTTP
        if has_lattice {
            if let Ok(runner) = setup_http_lattice(*size) {
                group.bench_with_input(BenchmarkId::new("LatticeDB_HTTP", size), size, |b, _| {
                    b.iter(|| {
                        black_box(
                            runner
                                .bench_retrieve(&ids)
                                .expect("LatticeDB HTTP retrieve failed"),
                        )
                    })
                });
            }
        }

        // Qdrant HTTP
        if has_qdrant {
            if let Ok(qdrant) = setup_qdrant(*size) {
                group.bench_with_input(BenchmarkId::new("Qdrant_HTTP", size), size, |b, _| {
                    b.iter(|| black_box(qdrant.bench_retrieve(&ids).expect("Qdrant retrieve failed")))
                });
            }
        }
    }

    group.finish();
}

fn bench_http_scroll(c: &mut Criterion) {
    let has_lattice = lattice_http_available();
    let has_qdrant = qdrant_available();

    if !has_lattice && !has_qdrant {
        return;
    }

    let mut group = c.benchmark_group("http_vector_scroll");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for size in [1000, 5000].iter() {
        // LatticeDB HTTP
        if has_lattice {
            if let Ok(runner) = setup_http_lattice(*size) {
                group.bench_with_input(BenchmarkId::new("LatticeDB_HTTP", size), size, |b, _| {
                    b.iter(|| {
                        black_box(runner.bench_scroll(100).expect("LatticeDB HTTP scroll failed"))
                    })
                });
            }
        }

        // Qdrant HTTP
        if has_qdrant {
            if let Ok(qdrant) = setup_qdrant(*size) {
                group.bench_with_input(BenchmarkId::new("Qdrant_HTTP", size), size, |b, _| {
                    b.iter(|| black_box(qdrant.bench_scroll(100).expect("Qdrant scroll failed")))
                });
            }
        }
    }

    group.finish();
}

// In-memory benchmarks (LatticeDB direct vs Qdrant HTTP - original unfair comparison)
criterion_group!(
    inmemory_benches,
    bench_upsert,
    bench_search,
    bench_retrieve,
    bench_scroll
);

// HTTP benchmarks (fair comparison - both over network)
criterion_group!(
    http_benches,
    bench_http_upsert,
    bench_http_search,
    bench_http_retrieve,
    bench_http_scroll
);

criterion_main!(inmemory_benches, http_benches);
