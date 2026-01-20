#!/usr/bin/env python3
"""
Simple benchmark for LatticeDB performance testing.
Tests upsert, search, retrieve, and scroll operations.
"""

import json
import random
import time
import urllib.request

BASE_URL = "http://localhost:6333"
COLLECTION = "benchmark_test"
DIM = 128
NUM_POINTS = 5000
SEARCH_QUERIES = 100
BATCH_SIZE = 100


def make_request(method, endpoint, data=None):
    """Make HTTP request to LatticeDB."""
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}

    if data:
        data = json.dumps(data).encode('utf-8')

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode('utf-8')}


def random_vector(dim):
    """Generate a random vector."""
    vec = [random.gauss(0, 1) for _ in range(dim)]
    # Normalize
    norm = sum(x*x for x in vec) ** 0.5
    return [x / norm for x in vec]


def setup():
    """Delete collection if exists and create fresh."""
    # Delete if exists
    make_request("DELETE", f"/collections/{COLLECTION}")

    # Create collection
    result = make_request("PUT", f"/collections/{COLLECTION}", {
        "vectors": {"size": DIM, "distance": "Cosine"},
        "hnsw": {"m": 16, "ef": 100, "ef_construction": 200}
    })

    if "error" in result:
        print(f"Failed to create collection: {result}")
        return False
    return True


def benchmark_upsert():
    """Benchmark upsert operation."""
    points = [
        {"id": i, "vector": random_vector(DIM)}
        for i in range(NUM_POINTS)
    ]

    # Batch upserts
    total_time = 0
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]

        start = time.perf_counter()
        result = make_request("PUT", f"/collections/{COLLECTION}/points", {
            "points": batch
        })
        elapsed = time.perf_counter() - start
        total_time += elapsed

        if "error" in result:
            print(f"Upsert failed: {result}")
            return None

    return total_time * 1000  # Convert to ms


def benchmark_search():
    """Benchmark search operation."""
    total_time = 0

    for _ in range(SEARCH_QUERIES):
        query_vec = random_vector(DIM)

        start = time.perf_counter()
        result = make_request("POST", f"/collections/{COLLECTION}/points/search", {
            "vector": query_vec,
            "limit": 10
        })
        elapsed = time.perf_counter() - start
        total_time += elapsed

        if "error" in result:
            print(f"Search failed: {result}")
            return None

    return (total_time / SEARCH_QUERIES) * 1000  # Average in ms


def benchmark_retrieve():
    """Benchmark point retrieval."""
    # Random IDs to retrieve
    ids = random.sample(range(NUM_POINTS), min(100, NUM_POINTS))

    total_time = 0
    iterations = 100

    for _ in range(iterations):
        batch_ids = random.sample(ids, min(10, len(ids)))

        start = time.perf_counter()
        result = make_request("POST", f"/collections/{COLLECTION}/points", {
            "ids": batch_ids,
            "with_vector": True
        })
        elapsed = time.perf_counter() - start
        total_time += elapsed

        if "error" in result:
            print(f"Retrieve failed: {result}")
            return None

    return (total_time / iterations) * 1000  # Average in ms


def benchmark_scroll():
    """Benchmark scroll/pagination."""
    total_time = 0
    iterations = 100

    for _ in range(iterations):
        start = time.perf_counter()
        result = make_request("POST", f"/collections/{COLLECTION}/points/scroll", {
            "limit": 10,
            "with_vector": True
        })
        elapsed = time.perf_counter() - start
        total_time += elapsed

        if "error" in result:
            print(f"Scroll failed: {result}")
            return None

    return (total_time / iterations) * 1000  # Average in ms


def cleanup():
    """Delete test collection."""
    make_request("DELETE", f"/collections/{COLLECTION}")


def main():
    print("=" * 60)
    print("LatticeDB Benchmark")
    print("=" * 60)
    print(f"Points: {NUM_POINTS}, Dimension: {DIM}, Batch Size: {BATCH_SIZE}")
    print()

    # Setup
    print("Setting up collection...")
    if not setup():
        print("Setup failed!")
        return

    # Run benchmarks
    print("\nRunning benchmarks...")
    print("-" * 40)

    # Upsert
    upsert_time = benchmark_upsert()
    if upsert_time:
        print(f"Upsert {NUM_POINTS} points: {upsert_time:.2f}ms total")
        print(f"  Per-point: {upsert_time/NUM_POINTS:.3f}ms")

    # Wait a moment for async indexing to complete
    print("\nWaiting for index to stabilize...")
    time.sleep(2)

    # Search
    search_time = benchmark_search()
    if search_time:
        print(f"\nSearch (avg of {SEARCH_QUERIES}): {search_time:.2f}ms")

    # Retrieve
    retrieve_time = benchmark_retrieve()
    if retrieve_time:
        print(f"Retrieve (avg): {retrieve_time:.2f}ms")

    # Scroll
    scroll_time = benchmark_scroll()
    if scroll_time:
        print(f"Scroll (avg): {scroll_time:.2f}ms")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Operation':<20} {'Time (ms)':<15}")
    print("-" * 35)
    if upsert_time:
        print(f"{'Upsert (total)':<20} {upsert_time:.2f}")
    if search_time:
        print(f"{'Search (avg)':<20} {search_time:.2f}")
    if retrieve_time:
        print(f"{'Retrieve (avg)':<20} {retrieve_time:.2f}")
    if scroll_time:
        print(f"{'Scroll (avg)':<20} {scroll_time:.2f}")

    # Cleanup
    cleanup()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
