# Testing

This chapter covers testing guidelines, patterns, and best practices for contributing to LatticeDB.

## Test Organization

### Test Location

| Test Type | Location | Command |
|-----------|----------|---------|
| Unit tests | `src/*.rs` (inline `#[cfg(test)]`) | `cargo test` |
| Integration tests | `tests/*.rs` | `cargo test --test <name>` |
| WASM tests | `src/*.rs` with `wasm_bindgen_test` | `wasm-pack test` |
| Benchmarks | `benches/*.rs` | `cargo bench` |

### Module Structure

```rust
// src/hnsw.rs

pub struct HnswIndex { ... }

impl HnswIndex {
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<SearchResult> {
        // Implementation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_returns_k_results() {
        // Test implementation
    }
}
```

## Writing Unit Tests

### Basic Test Pattern

```rust
#[test]
fn test_function_name_describes_behavior() {
    // Arrange
    let index = HnswIndex::new(test_config(), Distance::Cosine);
    let point = Point::new_vector(1, vec![0.1, 0.2, 0.3]);

    // Act
    index.insert(&point);
    let results = index.search(&[0.1, 0.2, 0.3], 10, 100);

    // Assert
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 1);
}
```

### Test Naming Convention

```rust
// Good: Describes behavior
#[test]
fn test_search_returns_empty_for_empty_index() { ... }

#[test]
fn test_insert_updates_existing_point_with_same_id() { ... }

#[test]
fn test_delete_returns_false_for_nonexistent_point() { ... }

// Bad: Vague names
#[test]
fn test_search() { ... }

#[test]
fn test_insert() { ... }
```

### Test Helpers

```rust
// Common test configuration
fn test_config() -> HnswConfig {
    HnswConfig {
        m: 16,
        m0: 32,
        ml: HnswConfig::recommended_ml(16),
        ef: 100,
        ef_construction: 200,
    }
}

// Random vector generation (deterministic)
fn random_vector(dim: usize, seed: u64) -> Vector {
    let mut rng = seed;
    (0..dim)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng as f64 / u64::MAX as f64) as f32
        })
        .collect()
}

// Approximate equality for floats
fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() < epsilon
}
```

## WASM Tests

### Configuration

```rust
// At the top of the test module
#[cfg(all(target_arch = "wasm32", test))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
```

### Test Attribute

```rust
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

#[test]
fn test_works_on_both_native_and_wasm() {
    // This test runs on both platforms
}
```

### Async WASM Tests

```rust
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
async fn test_async_storage_operation() {
    let storage = OpfsStorage::new().await.unwrap();
    storage.write_page(0, b"hello").await.unwrap();
    let page = storage.read_page(0).await.unwrap();
    assert_eq!(page, b"hello");
}
```

### Running WASM Tests

```bash
# Chrome (headless)
wasm-pack test --headless --chrome crates/lattice-core

# Firefox
wasm-pack test --headless --firefox crates/lattice-core

# With output
wasm-pack test --headless --chrome crates/lattice-core -- --nocapture
```

## Testing Async Code

### Basic Async Test

```rust
#[tokio::test]
async fn test_async_operation() {
    let storage = MemStorage::new();
    storage.write_page(0, b"test").await.unwrap();
    let data = storage.read_page(0).await.unwrap();
    assert_eq!(data, b"test");
}
```

### Testing with Timeouts

```rust
#[tokio::test]
async fn test_operation_completes_quickly() {
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        expensive_operation()
    ).await;

    assert!(result.is_ok(), "Operation timed out");
}
```

## Property-Based Testing

### Using proptest

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_quantize_dequantize_preserves_order(
        a in prop::collection::vec(-1.0f32..1.0, 1..128),
        b in prop::collection::vec(-1.0f32..1.0, 1..128),
    ) {
        // Ensure same length
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let qa = QuantizedVector::quantize(a);
        let qb = QuantizedVector::quantize(b);

        let original_dist = cosine_distance(a, b);
        let quantized_dist = qa.cosine_distance_asymmetric(b);

        // Quantization should preserve relative ordering
        // (within some error margin)
        prop_assert!((original_dist - quantized_dist).abs() < 0.2);
    }
}
```

## Integration Tests

### Test File Structure

```rust
// tests/integration_test.rs
use lattice_core::*;
use lattice_storage::MemStorage;

#[tokio::test]
async fn test_full_workflow() {
    // Create collection
    let config = CollectionConfig::new(
        "test_collection",
        VectorConfig::new(128, Distance::Cosine),
        HnswConfig::default(),
    );
    let storage = MemStorage::new();
    let mut engine = CollectionEngine::new(config, storage).unwrap();

    // Insert points
    for i in 0..100 {
        let point = Point::new_vector(i, random_vector(128, i));
        engine.upsert(point).unwrap();
    }

    // Search
    let query = random_vector(128, 999);
    let results = engine.search(&SearchQuery::new(query).with_limit(10)).unwrap();

    assert_eq!(results.len(), 10);
}
```

## Benchmarks

### Criterion Benchmark

```rust
// benches/search_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_search(c: &mut Criterion) {
    // Setup
    let index = create_test_index(10000);
    let query = random_vector(128, 0);

    c.bench_function("search_10k", |b| {
        b.iter(|| {
            black_box(index.search(&query, 10, 100))
        })
    });
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
```

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench -p lattice-bench

# Run specific benchmark
cargo bench -p lattice-bench -- search

# Compare to baseline
cargo bench -p lattice-bench -- --baseline main
```

## Testing Best Practices

### 1. Test Edge Cases

```rust
#[test]
fn test_search_empty_index() {
    let index = HnswIndex::new(config, Distance::Cosine);
    let results = index.search(&[0.1, 0.2], 10, 100);
    assert!(results.is_empty());
}

#[test]
fn test_search_k_greater_than_index_size() {
    let mut index = HnswIndex::new(config, Distance::Cosine);
    index.insert(&Point::new_vector(1, vec![0.1, 0.2]));

    let results = index.search(&[0.1, 0.2], 100, 100);
    assert_eq!(results.len(), 1);  // Returns available, not k
}

#[test]
fn test_quantize_zero_vector() {
    let quantized = QuantizedVector::quantize(&[0.0, 0.0, 0.0]);
    assert!(!quantized.is_empty());
}
```

### 2. Test Error Conditions

```rust
#[test]
fn test_delete_nonexistent_returns_false() {
    let mut index = HnswIndex::new(config, Distance::Cosine);
    assert!(!index.delete(999));
}

#[test]
fn test_storage_error_on_missing_page() {
    let storage = MemStorage::new();
    let result = storage.read_page(999).await;
    assert!(matches!(result, Err(StorageError::PageNotFound { .. })));
}
```

### 3. Use Descriptive Assertions

```rust
// Good: Clear failure message
assert_eq!(
    results.len(),
    10,
    "Expected 10 results but got {}, query: {:?}",
    results.len(),
    query
);

// Bad: Unhelpful failure
assert!(results.len() == 10);
```

### 4. Isolate Tests

```rust
// Good: Each test is independent
#[test]
fn test_insert_single() {
    let mut index = HnswIndex::new(config, Distance::Cosine);
    index.insert(&point);
    assert_eq!(index.len(), 1);
}

// Bad: Tests depend on shared state
static mut SHARED_INDEX: Option<HnswIndex> = None;

#[test]
fn test_1_insert() {
    unsafe { SHARED_INDEX = Some(HnswIndex::new(...)); }
}

#[test]
fn test_2_search() {
    // Fails if test_1 didn't run first!
}
```

### 5. Test Recall Statistically

```rust
#[test]
fn test_recall_at_least_90_percent() {
    let index = create_index_with_1000_points();
    let distance = DistanceCalculator::new(Distance::Cosine);

    let mut total_recall = 0.0;
    let num_queries = 100;

    for q in 0..num_queries {
        let query = random_vector(128, 10000 + q);

        // Ground truth via brute force
        let gt = brute_force_search(&index, &query, 10);

        // HNSW search
        let results = index.search(&query, 10, 100);

        // Calculate recall
        let hits = gt.iter()
            .filter(|&id| results.iter().any(|r| r.id == *id))
            .count();
        total_recall += hits as f64 / 10.0;
    }

    let avg_recall = total_recall / num_queries as f64;
    assert!(
        avg_recall >= 0.9,
        "Average recall {:.3} below 90% threshold",
        avg_recall
    );
}
```

## Coverage

### Generate Coverage Report

```bash
# Install grcov
cargo install grcov

# Run tests with coverage
CARGO_INCREMENTAL=0 RUSTFLAGS='-Cinstrument-coverage' \
    LLVM_PROFILE_FILE='cargo-test-%p-%m.profraw' \
    cargo test --workspace

# Generate HTML report
grcov . --binary-path ./target/debug/deps/ -s . -t html --branch --ignore-not-existing -o ./target/coverage/

# Open report
open target/coverage/index.html
```

## Next Steps

- [Development Setup](./setup.md) - Environment configuration
- [Architecture](../architecture/overview.md) - Understanding the codebase
