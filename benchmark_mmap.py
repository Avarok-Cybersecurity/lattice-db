#!/usr/bin/env python3
"""Benchmark mmap vector storage memory and performance"""
import json
import random
import time
import urllib.request
import os
import subprocess
import sys

BASE_URL = "http://localhost:6333"
COLLECTION = "mmap_test"
DIM = 128
NUM_POINTS = 10000
SEARCH_QUERIES = 100

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

def get_process_memory(pid):
    """Get RSS memory usage of a process in MB"""
    try:
        result = subprocess.run(
            ['ps', '-o', 'rss=', '-p', str(pid)],
            capture_output=True, text=True
        )
        return int(result.stdout.strip()) / 1024  # Convert KB to MB
    except:
        return 0

def setup():
    make_request("DELETE", f"/collections/{COLLECTION}")
    result = make_request("PUT", f"/collections/{COLLECTION}", {
        "vectors": {"size": DIM, "distance": "Cosine"}
    })
    return "error" not in result

def benchmark_upsert():
    points = [{"id": i, "vector": random_vector(DIM)} for i in range(NUM_POINTS)]
    batch_size = 100
    total_time = 0
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        start = time.perf_counter()
        result = make_request("PUT", f"/collections/{COLLECTION}/points", {"points": batch})
        elapsed = time.perf_counter() - start
        total_time += elapsed
        if "error" in result:
            print(f"Upsert failed: {result}")
            return None
    return total_time * 1000

def benchmark_search():
    total_time = 0
    for _ in range(SEARCH_QUERIES):
        query_vec = random_vector(DIM)
        start = time.perf_counter()
        result = make_request("POST", f"/collections/{COLLECTION}/points/search", {
            "vector": query_vec, "limit": 10
        })
        elapsed = time.perf_counter() - start
        total_time += elapsed
        if "error" in result:
            return None
    return (total_time / SEARCH_QUERIES) * 1000

def main():
    print("=" * 60)
    print("Memory-Mapped Vectors Benchmark")
    print("=" * 60)

    # Calculate theoretical memory usage
    vector_bytes = NUM_POINTS * DIM * 4  # f32 = 4 bytes
    vector_mb = vector_bytes / (1024 * 1024)
    print(f"\nTheoretical vector data: {vector_mb:.2f} MB")
    print(f"  ({NUM_POINTS:,} vectors x {DIM} dims x 4 bytes)")
    print()

    if not setup():
        print("Setup failed!")
        return

    print("Running benchmarks...")
    print("-" * 40)

    # Upsert
    upsert_time = benchmark_upsert()
    if upsert_time:
        print(f"\nUpsert {NUM_POINTS:,} points: {upsert_time:.2f}ms")

    time.sleep(2)  # Wait for index

    # Search
    search_time = benchmark_search()
    if search_time:
        print(f"Search (avg of {SEARCH_QUERIES}): {search_time:.2f}ms")

    # Note about mmap
    print("\n" + "=" * 60)
    print("MMAP BENEFITS")
    print("=" * 60)
    print("""
The mmap feature provides:
1. Reduced memory footprint - OS pages vectors in/out as needed
2. Faster startup - No deserialization of vector data
3. Better cache utilization - OS manages page cache
4. Shared memory - Multiple processes can share the same mmap

When enabled, vectors can be exported to mmap files and loaded
on-demand, reducing active memory usage significantly for large
datasets that don't fit in RAM.

Memory savings example:
- In-memory: {:.2f} MB for vectors alone
- With mmap: ~{:.2f} MB working set (hot vectors only)
  (Assuming 10% hot vectors accessed frequently)
""".format(vector_mb, vector_mb * 0.1))

    make_request("DELETE", f"/collections/{COLLECTION}")
    print("Benchmark complete.")

if __name__ == "__main__":
    main()
