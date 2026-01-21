# LatticeDB Optimization Log

## Tracking Table

| # | Optimization | Upsert | Search | Retrieve | Scroll | All Win? | Status |
|---|--------------|--------|--------|----------|--------|----------|--------|
| 9 | +Direct iteration in insert path | **1.88x** | **1.72x** | **1.99x** | **1.82x** | **YES** | Current |
| 8 | +Pre-allocate neighbor Vec capacity | **1.55x** | **1.82x** | **2.07x** | **2.03x** | **YES** | - |
| 7 | +FxHashSet in async_indexer | **1.68x** | **1.77x** | **1.83x** | **1.77x** | **YES** | - |
| 6 | +Pre-allocate Vec in calc_distances_batch | **2.03x** | **1.94x** | **1.87x** | **1.99x** | **YES** | - |
| 5 | +FxHashMap (faster integer hashing) | **1.58x** | **1.75x** | **2.09x** | **2.18x** | **YES** | - |
| 4 | +sort_unstable_by | **1.63x** | **1.31x** | **2.47x** | **3.27x** | **YES** | - |
| 3 | +#[inline] on hot path functions | **1.86x** | **1.42x** | **2.20x** | **3.00x** | **YES** | - |
| 2 | +Slice entry_points (no Vec alloc) | **1.56x** | **1.37x** | **2.23x** | **2.42x** | **YES** | - |
| 1 | SmallVec for unvisited neighbors | **1.58x** | **1.45x** | **2.77x** | **2.48x** | **YES** | - |
| 0 | **Baseline** | **1.53x** | **1.47x** | **1.80x** | **2.93x** | **YES** | Baseline |

## Optimization Attempts

### #0 Baseline (Current)
- NEON 2x unrolling
- Candidate Copy trait
- Thread-local scratch space
- 4s indexing delay

---

## Completed This Session

1. ✅ SmallVec for unvisited neighbors - avoids heap allocation for small neighbor lists
2. ✅ Slice entry_points - avoids Vec allocation for single entry point
3. ✅ #[inline] on hot path functions - calc_distance, search_layer_single, search_layer_single_with_shortcut
4. ✅ sort_unstable_by - faster than stable sort when order of equal elements doesn't matter
5. ✅ FxHashMap/FxHashSet - faster hashing for integer keys (u64 PointId)
6. ✅ Pre-allocate Vec in calc_distances_batch - avoids reallocation during collection
7. ✅ FxHashSet in async_indexer - consistent hashing (high variance, minimal impact)
8. ✅ Pre-allocate neighbor Vec capacity in HnswNode::new - reduces reallocations during insert
9. ❌ Inline neighbor processing (reverted) - caused upsert regression
10. ✅ Direct iteration in insert path - avoids intermediate Vec allocation when selecting neighbors

## Next Candidates

1. Software prefetching in HNSW graph traversal
2. Reduce BinaryHeap operations in search_layer
3. SIMD vectorized distance comparison (compare multiple distances at once)
