//! Vector operation benchmarks: LatticeDB vs Qdrant
//!
//! ## Prerequisites
//!
//! For HTTP and ACID benchmarks, start LatticeDB server:
//! ```bash
//! cargo run --release -p lattice-server
//! ```
//!
//! Start Qdrant (optional, for comparison):
//! ```bash
//! docker run -d --name qdrant-bench -p 6333:6333 qdrant/qdrant
//! ```
//!
//! ## Running
//!
//! ```bash
//! # Run all vector benchmarks (in-memory + HTTP + ACID)
//! cargo bench -p lattice-bench -- vector
//!
//! # Run only in-memory benchmarks
//! cargo bench -p lattice-bench -- "vector_(upsert|search|retrieve|scroll)/"
//!
//! # Run only HTTP benchmarks (requires LatticeDB server)
//! cargo bench -p lattice-bench -- "http_vector_"
//!
//! # Run HTTP ACID benchmarks (Ephemeral vs Durable over HTTP)
//! cargo bench -p lattice-bench -- "acid_vector_"
//!
//! # Run in-memory ACID benchmarks (true WAL overhead, no HTTP latency)
//! cargo bench -p lattice-bench -- "inmem_acid_"
//!
//! # Run only in-memory ACID upsert (where durability overhead is most visible)
//! cargo bench -p lattice-bench -- "inmem_acid_upsert"
//! ```
//!
//! ## ACID Benchmarks
//!
//! Two sets of ACID benchmarks are available:
//!
//! - **acid_vector_***: HTTP-based, includes network latency (masks WAL overhead)
//! - **inmem_acid_***: In-memory, shows true WAL overhead without network latency
//!
//! Expected overhead (in-memory):
//! - Ephemeral upsert: ~1-5 µs
//! - ACID upsert: ~10-50 µs (WAL serialization + in-memory "sync")

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lattice_bench::http_vector_runner::{lattice_http_available, HttpVectorRunner};
use lattice_bench::lattice_durable_runner::LatticeDurableRunner;
use lattice_bench::lattice_vector_runner::LatticeVectorRunner;
use lattice_bench::qdrant_runner::{qdrant_available, QdrantRunner};
use lattice_bench::DurabilityMode;
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
    let runner = QdrantRunner::new(url, &format!("bench_vector_{}", size), VECTOR_DIM)?;
    runner.create_collection()?;
    runner.load_data(size, SEED)?;
    Ok(runner)
}

fn setup_http_lattice(size: usize) -> Result<HttpVectorRunner, Box<dyn std::error::Error>> {
    setup_http_lattice_with_durability(size, DurabilityMode::Ephemeral)
}

/// Get the LatticeDB server URL
fn get_lattice_url() -> &'static str {
    // Try port 6335 first (used when Qdrant occupies 6334), then 6334
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()
        .unwrap();

    // Check 6335 first
    if client
        .get("http://localhost:6335/")
        .send()
        .map(|r| r.status().is_success())
        .unwrap_or(false)
    {
        return "http://localhost:6335";
    }

    // Fall back to 6334
    "http://localhost:6334"
}

fn setup_http_lattice_with_durability(
    size: usize,
    durability: DurabilityMode,
) -> Result<HttpVectorRunner, Box<dyn std::error::Error>> {
    let url = get_lattice_url();
    let suffix = match durability {
        DurabilityMode::Ephemeral => "ephemeral",
        DurabilityMode::Durable => "acid",
    };
    let runner = HttpVectorRunner::with_durability(
        url,
        &format!("bench_http_vector_{}_{}", size, suffix),
        VECTOR_DIM,
        durability,
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
        println!("HTTP benchmarks: No servers available (LatticeDB/Qdrant). Skipping.");
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
                        black_box(
                            runner
                                .bench_upsert(SEED + 1)
                                .expect("LatticeDB HTTP upsert failed"),
                        )
                    })
                });
            }
        }

        // Qdrant HTTP
        if has_qdrant {
            if let Ok(qdrant) = setup_qdrant(*size) {
                group.bench_with_input(BenchmarkId::new("Qdrant_HTTP", size), size, |b, _| {
                    b.iter(|| {
                        black_box(qdrant.bench_upsert(SEED + 1).expect("Qdrant upsert failed"))
                    })
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
                        black_box(
                            runner
                                .bench_search(10, SEED)
                                .expect("LatticeDB HTTP search failed"),
                        )
                    })
                });
            }
        }

        // Qdrant HTTP
        if has_qdrant {
            if let Ok(qdrant) = setup_qdrant(*size) {
                group.bench_with_input(BenchmarkId::new("Qdrant_HTTP", size), size, |b, _| {
                    b.iter(|| {
                        black_box(qdrant.bench_search(10, SEED).expect("Qdrant search failed"))
                    })
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
                    b.iter(|| {
                        black_box(qdrant.bench_retrieve(&ids).expect("Qdrant retrieve failed"))
                    })
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
                        black_box(
                            runner
                                .bench_scroll(100)
                                .expect("LatticeDB HTTP scroll failed"),
                        )
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

// === ACID vs Ephemeral benchmarks (LatticeDB durability comparison) ===

fn bench_acid_upsert(c: &mut Criterion) {
    if !lattice_http_available() {
        println!("ACID benchmarks: LatticeDB server not available. Skipping.");
        return;
    }

    let mut group = c.benchmark_group("acid_vector_upsert");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for size in [1000, 5000].iter() {
        // Ephemeral mode (baseline)
        if let Ok(runner) = setup_http_lattice_with_durability(*size, DurabilityMode::Ephemeral) {
            group.bench_with_input(
                BenchmarkId::new("LatticeDB_Ephemeral", size),
                size,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            runner
                                .bench_upsert(SEED + 1)
                                .expect("LatticeDB Ephemeral upsert failed"),
                        )
                    })
                },
            );
        }

        // ACID mode (durable)
        if let Ok(runner) = setup_http_lattice_with_durability(*size, DurabilityMode::Durable) {
            group.bench_with_input(BenchmarkId::new("LatticeDB_ACID", size), size, |b, _| {
                b.iter(|| {
                    black_box(
                        runner
                            .bench_upsert(SEED + 1)
                            .expect("LatticeDB ACID upsert failed"),
                    )
                })
            });
        }
    }

    group.finish();
}

fn bench_acid_search(c: &mut Criterion) {
    if !lattice_http_available() {
        return;
    }

    let mut group = c.benchmark_group("acid_vector_search");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for size in [1000, 5000].iter() {
        // Ephemeral mode
        if let Ok(runner) = setup_http_lattice_with_durability(*size, DurabilityMode::Ephemeral) {
            group.bench_with_input(
                BenchmarkId::new("LatticeDB_Ephemeral", size),
                size,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            runner
                                .bench_search(10, SEED)
                                .expect("LatticeDB Ephemeral search failed"),
                        )
                    })
                },
            );
        }

        // ACID mode
        if let Ok(runner) = setup_http_lattice_with_durability(*size, DurabilityMode::Durable) {
            group.bench_with_input(BenchmarkId::new("LatticeDB_ACID", size), size, |b, _| {
                b.iter(|| {
                    black_box(
                        runner
                            .bench_search(10, SEED)
                            .expect("LatticeDB ACID search failed"),
                    )
                })
            });
        }
    }

    group.finish();
}

fn bench_acid_retrieve(c: &mut Criterion) {
    if !lattice_http_available() {
        return;
    }

    let mut group = c.benchmark_group("acid_vector_retrieve");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let ids: Vec<u64> = (0..10).collect();

    for size in [1000, 5000].iter() {
        // Ephemeral mode
        if let Ok(runner) = setup_http_lattice_with_durability(*size, DurabilityMode::Ephemeral) {
            group.bench_with_input(
                BenchmarkId::new("LatticeDB_Ephemeral", size),
                size,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            runner
                                .bench_retrieve(&ids)
                                .expect("LatticeDB Ephemeral retrieve failed"),
                        )
                    })
                },
            );
        }

        // ACID mode
        if let Ok(runner) = setup_http_lattice_with_durability(*size, DurabilityMode::Durable) {
            group.bench_with_input(BenchmarkId::new("LatticeDB_ACID", size), size, |b, _| {
                b.iter(|| {
                    black_box(
                        runner
                            .bench_retrieve(&ids)
                            .expect("LatticeDB ACID retrieve failed"),
                    )
                })
            });
        }
    }

    group.finish();
}

fn bench_acid_scroll(c: &mut Criterion) {
    if !lattice_http_available() {
        return;
    }

    let mut group = c.benchmark_group("acid_vector_scroll");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for size in [1000, 5000].iter() {
        // Ephemeral mode
        if let Ok(runner) = setup_http_lattice_with_durability(*size, DurabilityMode::Ephemeral) {
            group.bench_with_input(
                BenchmarkId::new("LatticeDB_Ephemeral", size),
                size,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            runner
                                .bench_scroll(100)
                                .expect("LatticeDB Ephemeral scroll failed"),
                        )
                    })
                },
            );
        }

        // ACID mode
        if let Ok(runner) = setup_http_lattice_with_durability(*size, DurabilityMode::Durable) {
            group.bench_with_input(BenchmarkId::new("LatticeDB_ACID", size), size, |b, _| {
                b.iter(|| {
                    black_box(
                        runner
                            .bench_scroll(100)
                            .expect("LatticeDB ACID scroll failed"),
                    )
                })
            });
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

// ACID benchmarks (durability comparison - Ephemeral vs ACID over HTTP)
criterion_group!(
    acid_benches,
    bench_acid_upsert,
    bench_acid_search,
    bench_acid_retrieve,
    bench_acid_scroll
);

// === In-memory ACID benchmarks (true WAL overhead, no HTTP latency) ===

fn setup_inmem_ephemeral(size: usize) -> LatticeVectorRunner {
    let mut runner =
        LatticeVectorRunner::new(&format!("bench_inmem_ephemeral_{}", size), VECTOR_DIM).unwrap();
    runner.load_data(size, SEED).unwrap();
    runner
}

async fn setup_inmem_durable_async(size: usize) -> LatticeDurableRunner {
    let mut runner = LatticeDurableRunner::new(&format!("bench_inmem_acid_{}", size), VECTOR_DIM)
        .await
        .unwrap();
    runner.load_data(size, SEED).await.unwrap();
    runner
}

fn bench_inmem_acid_upsert_1000(c: &mut Criterion) {
    use std::mem::ManuallyDrop;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("inmem_acid_upsert");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let size = 1000;

    // Ephemeral mode - use ManuallyDrop to avoid blocking on async indexer shutdown
    // The async indexer has millions of pending items after benchmark iterations
    let mut ephemeral = ManuallyDrop::new(setup_inmem_ephemeral(size));
    group.bench_function(BenchmarkId::new("Ephemeral", size), |b| {
        b.iter(|| {
            black_box(
                ephemeral
                    .bench_upsert(SEED + 1)
                    .expect("Ephemeral upsert failed"),
            )
        })
    });

    // ACID mode
    let mut durable = rt.block_on(setup_inmem_durable_async(size));
    group.bench_function(BenchmarkId::new("ACID", size), |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(
                    durable
                        .bench_upsert(SEED + 1)
                        .await
                        .expect("ACID upsert failed"),
                )
            })
        })
    });

    group.finish();
    // Note: ephemeral is ManuallyDrop, so it won't block on async indexer shutdown
}

fn bench_inmem_acid_upsert_5000(c: &mut Criterion) {
    use std::mem::ManuallyDrop;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("inmem_acid_upsert");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let size = 5000;

    // Ephemeral mode - use ManuallyDrop to avoid blocking on async indexer shutdown
    let mut ephemeral = ManuallyDrop::new(setup_inmem_ephemeral(size));
    group.bench_function(BenchmarkId::new("Ephemeral", size), |b| {
        b.iter(|| {
            black_box(
                ephemeral
                    .bench_upsert(SEED + 1)
                    .expect("Ephemeral upsert failed"),
            )
        })
    });

    // ACID mode
    let mut durable = rt.block_on(setup_inmem_durable_async(size));
    group.bench_function(BenchmarkId::new("ACID", size), |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(
                    durable
                        .bench_upsert(SEED + 1)
                        .await
                        .expect("ACID upsert failed"),
                )
            })
        })
    });

    group.finish();
}

fn bench_inmem_acid_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("inmem_acid_search");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for size in [1000, 5000].iter() {
        // Ephemeral mode
        let ephemeral = setup_inmem_ephemeral(*size);
        group.bench_with_input(
            BenchmarkId::new("Ephemeral", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        ephemeral
                            .bench_search(10, SEED)
                            .expect("Ephemeral search failed"),
                    )
                })
            },
        );
        drop(ephemeral);

        // ACID mode (reads are sync, no async needed)
        let durable = rt.block_on(setup_inmem_durable_async(*size));
        group.bench_with_input(BenchmarkId::new("ACID", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    durable
                        .bench_search(10, SEED)
                        .expect("ACID search failed"),
                )
            })
        });
    }

    group.finish();
}

fn bench_inmem_acid_retrieve(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("inmem_acid_retrieve");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let ids: Vec<u64> = (0..10).collect();

    for size in [1000, 5000].iter() {
        // Ephemeral mode
        let ephemeral = setup_inmem_ephemeral(*size);
        group.bench_with_input(
            BenchmarkId::new("Ephemeral", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        ephemeral
                            .bench_retrieve(&ids)
                            .expect("Ephemeral retrieve failed"),
                    )
                })
            },
        );
        drop(ephemeral);

        // ACID mode (reads are sync)
        let durable = rt.block_on(setup_inmem_durable_async(*size));
        group.bench_with_input(BenchmarkId::new("ACID", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    durable
                        .bench_retrieve(&ids)
                        .expect("ACID retrieve failed"),
                )
            })
        });
    }

    group.finish();
}

fn bench_inmem_acid_scroll(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("inmem_acid_scroll");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for size in [1000, 5000].iter() {
        // Ephemeral mode
        let ephemeral = setup_inmem_ephemeral(*size);
        group.bench_with_input(
            BenchmarkId::new("Ephemeral", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(ephemeral.bench_scroll(100).expect("Ephemeral scroll failed"))
                })
            },
        );
        drop(ephemeral);

        // ACID mode (reads are sync)
        let durable = rt.block_on(setup_inmem_durable_async(*size));
        group.bench_with_input(BenchmarkId::new("ACID", size), size, |b, _| {
            b.iter(|| {
                black_box(durable.bench_scroll(100).expect("ACID scroll failed"))
            })
        });
    }

    group.finish();
}

// In-memory ACID benchmarks (true WAL overhead without HTTP)
criterion_group!(
    inmem_acid_benches,
    bench_inmem_acid_upsert_1000,
    bench_inmem_acid_upsert_5000,
    bench_inmem_acid_search,
    bench_inmem_acid_retrieve,
    bench_inmem_acid_scroll
);

criterion_main!(inmemory_benches, http_benches, acid_benches, inmem_acid_benches);
