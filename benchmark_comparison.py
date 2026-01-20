#!/usr/bin/env python3
"""Benchmark LatticeDB vs Qdrant performance."""

import time
import random
import statistics
from dataclasses import dataclass
from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configuration
LATTICE_HOST = "localhost"
LATTICE_PORT = 6333
QDRANT_HOST = "localhost"
QDRANT_PORT = 6334

COLLECTION_NAME = "benchmark_collection"
VECTOR_DIM = 128
NUM_POINTS = 10000
NUM_QUERIES = 1000
TOP_K = 10
BATCH_SIZE = 100


@dataclass
class BenchmarkResult:
    name: str
    operation: str
    total_time_ms: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_ops: float


def generate_random_vector(dim: int) -> List[float]:
    return [random.random() for _ in range(dim)]


def generate_points(num_points: int, dim: int) -> List[PointStruct]:
    return [
        PointStruct(
            id=i,
            vector=generate_random_vector(dim),
            payload={"category": f"cat_{i % 10}", "value": i}
        )
        for i in range(num_points)
    ]


def benchmark_create_collection(client: QdrantClient, name: str) -> float:
    """Benchmark collection creation."""
    # Clean up if exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    start = time.perf_counter()
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    end = time.perf_counter()
    return (end - start) * 1000


def benchmark_upsert(client: QdrantClient, points: List[PointStruct], name: str) -> BenchmarkResult:
    """Benchmark point upsert in batches."""
    latencies = []
    total_start = time.perf_counter()

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        start = time.perf_counter()
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    total_end = time.perf_counter()
    total_time = (total_end - total_start) * 1000

    return BenchmarkResult(
        name=name,
        operation=f"upsert ({len(points)} points, batch={BATCH_SIZE})",
        total_time_ms=total_time,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)],
        p99_latency_ms=sorted(latencies)[int(len(latencies) * 0.99)],
        throughput_ops=len(points) / (total_time / 1000)
    )


def benchmark_search(client: QdrantClient, num_queries: int, name: str) -> BenchmarkResult:
    """Benchmark search queries."""
    latencies = []
    total_start = time.perf_counter()

    for _ in range(num_queries):
        query_vector = generate_random_vector(VECTOR_DIM)
        start = time.perf_counter()
        client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K,
            with_payload=True,
        )
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    total_end = time.perf_counter()
    total_time = (total_end - total_start) * 1000

    return BenchmarkResult(
        name=name,
        operation=f"search (top-{TOP_K}, {num_queries} queries)",
        total_time_ms=total_time,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)],
        p99_latency_ms=sorted(latencies)[int(len(latencies) * 0.99)],
        throughput_ops=num_queries / (total_time / 1000)
    )


def benchmark_scroll(client: QdrantClient, name: str) -> BenchmarkResult:
    """Benchmark scrolling through all points."""
    latencies = []
    total_points = 0
    offset = None

    total_start = time.perf_counter()

    while True:
        start = time.perf_counter()
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        total_points += len(points)

        if next_offset is None or len(points) == 0:
            break
        offset = next_offset

    total_end = time.perf_counter()
    total_time = (total_end - total_start) * 1000

    return BenchmarkResult(
        name=name,
        operation=f"scroll ({total_points} points)",
        total_time_ms=total_time,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else max(latencies),
        p99_latency_ms=sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) >= 100 else max(latencies),
        throughput_ops=total_points / (total_time / 1000)
    )


def benchmark_retrieve(client: QdrantClient, num_retrieves: int, name: str) -> BenchmarkResult:
    """Benchmark point retrieval by ID."""
    latencies = []
    total_start = time.perf_counter()

    for _ in range(num_retrieves):
        ids = random.sample(range(NUM_POINTS), min(10, NUM_POINTS))
        start = time.perf_counter()
        client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    total_end = time.perf_counter()
    total_time = (total_end - total_start) * 1000

    return BenchmarkResult(
        name=name,
        operation=f"retrieve ({num_retrieves} batches of 10)",
        total_time_ms=total_time,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)],
        p99_latency_ms=sorted(latencies)[int(len(latencies) * 0.99)],
        throughput_ops=num_retrieves / (total_time / 1000)
    )


def print_result(result: BenchmarkResult):
    print(f"  {result.operation}:")
    print(f"    Total time:  {result.total_time_ms:>10.2f} ms")
    print(f"    Avg latency: {result.avg_latency_ms:>10.2f} ms")
    print(f"    P50 latency: {result.p50_latency_ms:>10.2f} ms")
    print(f"    P95 latency: {result.p95_latency_ms:>10.2f} ms")
    print(f"    P99 latency: {result.p99_latency_ms:>10.2f} ms")
    print(f"    Throughput:  {result.throughput_ops:>10.2f} ops/sec")


def print_comparison(lattice_results: List[BenchmarkResult], qdrant_results: List[BenchmarkResult]):
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Operation':<40} {'LatticeDB':<15} {'Qdrant':<15} {'Ratio':<10}")
    print("-" * 80)

    for l, q in zip(lattice_results, qdrant_results):
        ratio = l.avg_latency_ms / q.avg_latency_ms if q.avg_latency_ms > 0 else float('inf')
        winner = "LatticeDB" if ratio < 1 else "Qdrant"
        ratio_str = f"{ratio:.2f}x" if ratio >= 1 else f"{1/ratio:.2f}x faster"
        print(f"{l.operation:<40} {l.avg_latency_ms:>10.2f} ms   {q.avg_latency_ms:>10.2f} ms   {ratio_str}")

    print("-" * 80)

    # Throughput comparison
    print(f"\n{'Operation':<40} {'LatticeDB':<15} {'Qdrant':<15} {'Ratio':<10}")
    print("-" * 80)
    for l, q in zip(lattice_results, qdrant_results):
        ratio = l.throughput_ops / q.throughput_ops if q.throughput_ops > 0 else float('inf')
        ratio_str = f"{ratio:.2f}x" if ratio >= 1 else f"{1/ratio:.2f}x slower"
        print(f"{l.operation:<40} {l.throughput_ops:>10.0f}/s   {q.throughput_ops:>10.0f}/s   {ratio_str}")


def run_benchmark(client: QdrantClient, name: str, points: List[PointStruct]) -> List[BenchmarkResult]:
    results = []

    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {name}")
    print(f"{'=' * 60}")

    # Create collection
    create_time = benchmark_create_collection(client, name)
    print(f"  Collection created in {create_time:.2f} ms")

    # Upsert
    result = benchmark_upsert(client, points, name)
    results.append(result)
    print_result(result)

    # Search
    result = benchmark_search(client, NUM_QUERIES, name)
    results.append(result)
    print_result(result)

    # Retrieve
    result = benchmark_retrieve(client, 500, name)
    results.append(result)
    print_result(result)

    # Scroll
    result = benchmark_scroll(client, name)
    results.append(result)
    print_result(result)

    # Cleanup
    client.delete_collection(COLLECTION_NAME)

    return results


def main():
    random.seed(42)

    print("=" * 80)
    print("LatticeDB vs Qdrant Performance Benchmark")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Vector dimensions: {VECTOR_DIM}")
    print(f"  Number of points:  {NUM_POINTS}")
    print(f"  Number of queries: {NUM_QUERIES}")
    print(f"  Top-K:             {TOP_K}")
    print(f"  Batch size:        {BATCH_SIZE}")
    print(f"  LatticeDB:         {LATTICE_HOST}:{LATTICE_PORT}")
    print(f"  Qdrant:            {QDRANT_HOST}:{QDRANT_PORT}")

    # Generate test data
    print("\nGenerating test data...")
    points = generate_points(NUM_POINTS, VECTOR_DIM)
    print(f"  Generated {len(points)} points")

    # Connect to both
    print("\nConnecting to databases...")
    lattice_client = QdrantClient(host=LATTICE_HOST, port=LATTICE_PORT, check_compatibility=False)
    print(f"  Connected to LatticeDB at {LATTICE_HOST}:{LATTICE_PORT}")

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"  Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")

    # Run benchmarks
    lattice_results = run_benchmark(lattice_client, "LatticeDB (SIMD)", points)
    qdrant_results = run_benchmark(qdrant_client, "Qdrant", points)

    # Print comparison
    print_comparison(lattice_results, qdrant_results)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
