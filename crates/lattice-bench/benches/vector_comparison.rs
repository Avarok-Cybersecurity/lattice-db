//! Vector operation benchmarks: LatticeDB vs Qdrant
//!
//! ## Prerequisites
//!
//! Start Qdrant:
//! ```bash
//! docker run -d --name qdrant-bench -p 6333:6333 qdrant/qdrant
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo bench -p lattice-bench -- vector
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
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
    let runner = QdrantRunner::new(
        "http://localhost:6333",
        &format!("bench_vector_{}", size),
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

criterion_group!(
    benches,
    bench_upsert,
    bench_search,
    bench_retrieve,
    bench_scroll
);
criterion_main!(benches);
