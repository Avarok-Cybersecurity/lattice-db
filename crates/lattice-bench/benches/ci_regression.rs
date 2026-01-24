//! CI Performance Regression Benchmarks
//!
//! Self-contained benchmarks for CI performance regression testing.
//! No external services required (Neo4j, Qdrant, etc.).
//!
//! ## Running
//!
//! ```bash
//! # Run all CI benchmarks
//! cargo bench -p lattice-bench -- ci_
//!
//! # Compare against baseline
//! cargo bench -p lattice-bench -- ci_ --save-baseline main
//! cargo bench -p lattice-bench -- ci_ --baseline main
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lattice_bench::lattice_vector_runner::LatticeVectorRunner;
use lattice_core::cypher::parser::CypherParser;
use lattice_core::cypher::{CypherHandler, DefaultCypherHandler};
use lattice_core::engine::collection::CollectionEngine;
use lattice_core::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
use std::collections::HashMap;
use std::time::Duration;

const VECTOR_DIM: usize = 128;
const SEED: u64 = 42;

// =============================================================================
// Vector Operation Benchmarks
// =============================================================================

fn bench_ci_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("ci_vector_search");
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));

    for size in [100, 1000] {
        let mut runner =
            LatticeVectorRunner::new(&format!("ci_bench_{}", size), VECTOR_DIM).unwrap();
        runner.load_data(size, SEED).unwrap();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("lattice", size), &size, |b, _| {
            b.iter(|| {
                // bench_search(k, seed) returns Duration
                black_box(runner.bench_search(10, SEED).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_ci_vector_upsert(c: &mut Criterion) {
    let mut group = c.benchmark_group("ci_vector_upsert");
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));

    group.throughput(Throughput::Elements(1));

    group.bench_function("lattice_single", |b| {
        let mut runner = LatticeVectorRunner::new("ci_upsert_bench", VECTOR_DIM).unwrap();
        b.iter(|| {
            black_box(runner.bench_upsert(SEED).unwrap());
        });
    });

    group.finish();
}

// =============================================================================
// Cypher Parsing Benchmarks
// =============================================================================

fn bench_ci_cypher_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("ci_cypher_parse");
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));

    let parser = CypherParser::new();

    let queries = [
        ("simple_match", "MATCH (n) RETURN n"),
        ("label_match", "MATCH (n:Person) RETURN n.name"),
        (
            "relationship",
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b",
        ),
        (
            "where_clause",
            "MATCH (n:Person) WHERE n.age > 25 RETURN n.name",
        ),
        (
            "projection_order",
            "MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age DESC LIMIT 10",
        ),
    ];

    for (name, query) in queries {
        group.bench_with_input(BenchmarkId::new("parse", name), query, |b, q| {
            b.iter(|| {
                black_box(parser.parse(q).unwrap());
            });
        });
    }

    group.finish();
}

// =============================================================================
// Cypher Execution Benchmarks
// =============================================================================

fn bench_ci_cypher_execute(c: &mut Criterion) {
    let mut group = c.benchmark_group("ci_cypher_execute");
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));

    // Create collection with test data using proper constructor
    let config = CollectionConfig::new(
        "ci_cypher_bench",
        VectorConfig::new(VECTOR_DIM, Distance::Cosine),
        HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 100,
            ef_construction: 200,
        },
    );
    let mut engine = CollectionEngine::new(config).unwrap();

    let handler = DefaultCypherHandler::new();

    // Create test nodes using Cypher (the proper way)
    for i in 0..100 {
        handler
            .query(
                &format!(
                    "CREATE (n:Person {{name: \"Person{}\", age: {}}})",
                    i,
                    20 + (i % 50)
                ),
                &mut engine,
                HashMap::new(),
            )
            .unwrap();
    }

    // Create relationships using Cypher
    // Note: Creating relationships requires existing nodes with IDs
    // For now, focus on node operations which are the main benchmark

    let queries = [
        ("all_nodes", "MATCH (n) RETURN n LIMIT 100"),
        ("label_scan", "MATCH (n:Person) RETURN n LIMIT 100"),
        ("with_filter", "MATCH (n:Person) WHERE n.age > 40 RETURN n"),
        ("projection", "MATCH (n:Person) RETURN n.name, n.age LIMIT 50"),
        (
            "distinct",
            "MATCH (n:Person) RETURN DISTINCT n.age ORDER BY n.age",
        ),
    ];

    for (name, query) in queries {
        group.bench_with_input(BenchmarkId::new("execute", name), query, |b, q| {
            b.iter(|| {
                // Use CypherHandler trait's query() method
                black_box(handler.query(q, &mut engine, HashMap::new()).unwrap());
            });
        });
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    ci_benches,
    bench_ci_vector_search,
    bench_ci_vector_upsert,
    bench_ci_cypher_parse,
    bench_ci_cypher_execute,
);

criterion_main!(ci_benches);
