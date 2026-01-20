#!/usr/bin/env python3
"""Benchmark batch search vs sequential search"""
import json
import random
import time
import urllib.request

BASE_URL = "http://localhost:6333"
COLLECTION = "batch_test"
DIM = 128
NUM_POINTS = 5000
NUM_QUERIES = 100
BATCH_SIZES = [10, 50, 100]

def make_request(method, endpoint, data=None):
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
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = sum(x*x for x in vec) ** 0.5
    return [x / norm for x in vec]

def setup():
    make_request("DELETE", f"/collections/{COLLECTION}")
    result = make_request("PUT", f"/collections/{COLLECTION}", {
        "vectors": {"size": DIM, "distance": "Cosine"}
    })
    if "error" in result:
        return False

    # Insert points in batches
    points = [{"id": i, "vector": random_vector(DIM)} for i in range(NUM_POINTS)]
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        make_request("PUT", f"/collections/{COLLECTION}/points", {"points": batch})

    time.sleep(2)  # Wait for indexing
    return True

def benchmark_sequential_search(queries):
    """Sequential individual searches"""
    total_time = 0
    for query_vec in queries:
        start = time.perf_counter()
        result = make_request("POST", f"/collections/{COLLECTION}/points/search", {
            "vector": query_vec, "limit": 10
        })
        elapsed = time.perf_counter() - start
        total_time += elapsed
        if "error" in result:
            return None
    return total_time * 1000

def benchmark_batch_search(queries, batch_size):
    """Batch search"""
    total_time = 0
    num_batches = (len(queries) + batch_size - 1) // batch_size

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        searches = [{"vector": q, "limit": 10, "with_payload": False} for q in batch]

        start = time.perf_counter()
        result = make_request("POST", f"/collections/{COLLECTION}/points/search/batch", {
            "searches": searches
        })
        elapsed = time.perf_counter() - start
        total_time += elapsed
        if "error" in result:
            print(f"Batch search error: {result}")
            return None

    return total_time * 1000

def main():
    print("=" * 60)
    print("Batch Search Benchmark")
    print("=" * 60)
    print(f"Points: {NUM_POINTS}, Dimension: {DIM}, Queries: {NUM_QUERIES}")
    print()

    if not setup():
        print("Setup failed!")
        return

    # Generate query vectors
    queries = [random_vector(DIM) for _ in range(NUM_QUERIES)]

    print("Running benchmarks...")
    print("-" * 40)

    # Sequential search
    seq_time = benchmark_sequential_search(queries)
    if seq_time:
        print(f"\nSequential ({NUM_QUERIES} individual searches): {seq_time:.2f}ms total")
        print(f"  Average per query: {seq_time/NUM_QUERIES:.2f}ms")

    # Batch search at different batch sizes
    for batch_size in BATCH_SIZES:
        batch_time = benchmark_batch_search(queries, batch_size)
        if batch_time:
            speedup = seq_time / batch_time if seq_time and batch_time else 0
            print(f"\nBatch (size={batch_size}): {batch_time:.2f}ms total")
            print(f"  Average per query: {batch_time/NUM_QUERIES:.2f}ms")
            print(f"  Speedup: {speedup:.2f}x")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 55)
    if seq_time:
        print(f"{'Sequential':<30} {seq_time:.2f}")
    for batch_size in BATCH_SIZES:
        batch_time = benchmark_batch_search(queries, batch_size)
        if batch_time:
            speedup = seq_time / batch_time if seq_time else 0
            print(f"{'Batch (size=' + str(batch_size) + ')':<30} {batch_time:.2f}{' ':>10}{speedup:.2f}x")

    make_request("DELETE", f"/collections/{COLLECTION}")
    print("\nBenchmark complete.")

if __name__ == "__main__":
    main()
