//! Zero-cost parallel operations abstraction
//!
//! This module provides parallel operations that automatically select the best
//! implementation based on the target platform and data size:
//!
//! - **Native (N >= threshold)**: Uses rayon parallel iterators for `Send` types
//! - **Native (N < threshold)**: Sequential (parallel overhead not worth it)
//! - **WASM**: Always sequential (no threading support)
//!
//! ## Send vs Non-Send Types
//!
//! For types that implement `Send` (primitives, `Arc`, etc.), use the regular
//! functions which will parallelize on native when beneficial.
//!
//! For types that don't implement `Send` (containing `Rc`, etc.), use the `_seq`
//! suffixed functions which are always sequential but avoid the `Send` bound.
//!
//! All functions are zero-cost abstractions with `#[inline]` to enable
//! compiler optimizations.

/// Threshold for switching to parallel operations on native targets.
/// Below this size, sequential operations are typically faster due to
/// parallel overhead (thread spawning, synchronization).
pub const DEFAULT_PARALLEL_THRESHOLD: usize = 500;

/// Upper threshold for parallel sort with expensive comparison functions.
/// At 30K-80K rows with CypherValue enum comparisons, thread sync overhead
/// and cache line bouncing exceed the parallelization benefit.
/// Above this threshold, the workload is large enough that parallel wins again.
pub const EXPENSIVE_CMP_PARALLEL_UPPER: usize = 80_000;

// =============================================================================
// Native implementations (with rayon)
// =============================================================================

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::DEFAULT_PARALLEL_THRESHOLD;
    use rayon::prelude::*;
    use std::cmp::Ordering;

    /// Sort a slice using a comparison function.
    /// Uses parallel sort for large slices.
    #[inline]
    pub fn sort_by<T, F>(slice: &mut [T], compare: F)
    where
        T: Send,
        F: Fn(&T, &T) -> Ordering + Sync,
    {
        if slice.len() >= DEFAULT_PARALLEL_THRESHOLD {
            slice.par_sort_by(compare);
        } else {
            slice.sort_by(compare);
        }
    }

    /// Sort a slice using a comparison function (unstable sort).
    /// Uses parallel sort for large slices. Unstable sort is faster
    /// but doesn't preserve order of equal elements.
    #[inline]
    pub fn sort_unstable_by<T, F>(slice: &mut [T], compare: F)
    where
        T: Send,
        F: Fn(&T, &T) -> Ordering + Sync,
    {
        if slice.len() >= DEFAULT_PARALLEL_THRESHOLD {
            slice.par_sort_unstable_by(compare);
        } else {
            slice.sort_unstable_by(compare);
        }
    }

    /// Sort with expensive comparison function (e.g., CypherValue enum matching).
    /// Uses sequential sort in the 30K-80K range where thread overhead exceeds benefit.
    /// Based on benchmark analysis: LatticeDB 2.3x slower than Neo4j at 50K due to
    /// parallel overhead with expensive comparisons.
    #[inline]
    pub fn sort_by_expensive_cmp<T, F>(slice: &mut [T], compare: F)
    where
        T: Send,
        F: Fn(&T, &T) -> Ordering + Sync,
    {
        use super::EXPENSIVE_CMP_PARALLEL_UPPER;
        let len = slice.len();

        if len < DEFAULT_PARALLEL_THRESHOLD {
            // Too small for parallel
            slice.sort_by(compare);
        } else if len >= EXPENSIVE_CMP_PARALLEL_UPPER {
            // Large enough that parallel wins despite comparison cost
            slice.par_sort_by(compare);
        } else {
            // Awkward zone (500-80K): sequential is faster with expensive comparisons
            slice.sort_by(compare);
        }
    }

    /// Map a slice to a new vector using a transformation function.
    /// Uses parallel iteration for large slices.
    #[inline]
    pub fn map_collect<T, U, F>(slice: &[T], f: F) -> Vec<U>
    where
        T: Sync,
        U: Send,
        F: Fn(&T) -> U + Sync + Send,
    {
        if slice.len() >= DEFAULT_PARALLEL_THRESHOLD {
            slice.par_iter().map(f).collect()
        } else {
            slice.iter().map(f).collect()
        }
    }

    /// Map a vector by value to a new vector using a transformation function.
    /// Uses parallel iteration for large vectors.
    #[inline]
    pub fn map_into_collect<T, U, F>(vec: Vec<T>, f: F) -> Vec<U>
    where
        T: Send,
        U: Send,
        F: Fn(T) -> U + Sync + Send,
    {
        if vec.len() >= DEFAULT_PARALLEL_THRESHOLD {
            vec.into_par_iter().map(f).collect()
        } else {
            vec.into_iter().map(f).collect()
        }
    }

    /// Filter and map a slice to a new vector.
    /// Uses parallel iteration for large slices.
    #[inline]
    pub fn filter_map_collect<T, U, F>(slice: &[T], f: F) -> Vec<U>
    where
        T: Sync,
        U: Send,
        F: Fn(&T) -> Option<U> + Sync + Send,
    {
        if slice.len() >= DEFAULT_PARALLEL_THRESHOLD {
            slice.par_iter().filter_map(f).collect()
        } else {
            slice.iter().filter_map(f).collect()
        }
    }

    /// Filter a slice to a new vector of cloned elements.
    /// Uses parallel iteration for large slices.
    #[inline]
    pub fn filter_collect<T, F>(slice: &[T], f: F) -> Vec<T>
    where
        T: Clone + Send + Sync,
        F: Fn(&T) -> bool + Sync + Send,
    {
        if slice.len() >= DEFAULT_PARALLEL_THRESHOLD {
            slice.par_iter().filter(|x| f(x)).cloned().collect()
        } else {
            slice.iter().filter(|x| f(x)).cloned().collect()
        }
    }

    /// Map a slice with indices to a new vector using a transformation function.
    /// Uses parallel iteration for large slices.
    /// This is useful for Schwartzian transform patterns where index is needed.
    #[inline]
    pub fn enumerate_map_collect<T, U, F>(slice: &[T], f: F) -> Vec<U>
    where
        T: Sync,
        U: Send,
        F: Fn(usize, &T) -> U + Sync + Send,
    {
        if slice.len() >= DEFAULT_PARALLEL_THRESHOLD {
            slice.par_iter().enumerate().map(|(i, x)| f(i, x)).collect()
        } else {
            slice.iter().enumerate().map(|(i, x)| f(i, x)).collect()
        }
    }

    /// Partial sort: efficiently find and sort the top-k elements.
    /// Uses select_nth_unstable for O(n) partitioning, then sorts only top-k.
    /// For ORDER BY ... LIMIT k queries, this is O(n + k log k) vs O(n log n).
    #[inline]
    pub fn partial_sort_by<T, F>(slice: &mut [T], k: usize, compare: F)
    where
        T: Send,
        F: Fn(&T, &T) -> Ordering + Sync,
    {
        if k == 0 || slice.is_empty() {
            return;
        }
        let k = k.min(slice.len());

        if k >= slice.len() / 2 {
            // Full sort is more efficient when k is large relative to n
            sort_by(slice, compare);
            return;
        }

        // Partition: elements [0..k] will contain the k smallest (unsorted)
        // This is O(n) average case
        slice.select_nth_unstable_by(k - 1, &compare);

        // Sort only the top-k elements: O(k log k)
        sort_by(&mut slice[..k], compare);
    }

    /// Filter a vector by consuming it and returning matching elements.
    /// Uses parallel iteration for large vectors.
    /// More efficient than filter_collect when you own the data (no cloning).
    #[inline]
    pub fn filter_into_vec<T, F>(vec: Vec<T>, f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(&T) -> bool + Sync + Send,
    {
        if vec.len() >= DEFAULT_PARALLEL_THRESHOLD {
            vec.into_par_iter().filter(|x| f(x)).collect()
        } else {
            vec.into_iter().filter(|x| f(&x)).collect()
        }
    }

    // =========================================================================
    // Sequential versions for non-Send types (no parallelization)
    // =========================================================================

    /// Sort a slice using a comparison function (always sequential).
    /// Use this for types that don't implement `Send` (e.g., containing `Rc`).
    #[inline]
    pub fn sort_by_seq<T, F>(slice: &mut [T], compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        slice.sort_by(compare);
    }

    /// Sort a slice using a comparison function (unstable, always sequential).
    /// Use this for types that don't implement `Send` (e.g., containing `Rc`).
    #[inline]
    pub fn sort_unstable_by_seq<T, F>(slice: &mut [T], compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        slice.sort_unstable_by(compare);
    }
}

// =============================================================================
// WASM implementations (sequential only)
// =============================================================================

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::cmp::Ordering;

    /// Sort a slice using a comparison function (sequential).
    #[inline]
    pub fn sort_by<T, F>(slice: &mut [T], mut compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        slice.sort_by(|a, b| compare(a, b));
    }

    /// Sort a slice using a comparison function (unstable, sequential).
    #[inline]
    pub fn sort_unstable_by<T, F>(slice: &mut [T], mut compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        slice.sort_unstable_by(|a, b| compare(a, b));
    }

    /// Sort with expensive comparison function (sequential on WASM).
    #[inline]
    pub fn sort_by_expensive_cmp<T, F>(slice: &mut [T], compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        slice.sort_by(compare);
    }

    /// Map a slice to a new vector (sequential).
    #[inline]
    pub fn map_collect<T, U, F>(slice: &[T], f: F) -> Vec<U>
    where
        F: Fn(&T) -> U,
    {
        slice.iter().map(f).collect()
    }

    /// Map a vector by value to a new vector (sequential).
    #[inline]
    pub fn map_into_collect<T, U, F>(vec: Vec<T>, f: F) -> Vec<U>
    where
        F: Fn(T) -> U,
    {
        vec.into_iter().map(f).collect()
    }

    /// Filter and map a slice to a new vector (sequential).
    #[inline]
    pub fn filter_map_collect<T, U, F>(slice: &[T], f: F) -> Vec<U>
    where
        F: Fn(&T) -> Option<U>,
    {
        slice.iter().filter_map(f).collect()
    }

    /// Filter a slice to a new vector of cloned elements (sequential).
    #[inline]
    pub fn filter_collect<T, F>(slice: &[T], f: F) -> Vec<T>
    where
        T: Clone,
        F: Fn(&T) -> bool,
    {
        slice.iter().filter(|x| f(x)).cloned().collect()
    }

    /// Map a slice with indices to a new vector (sequential).
    /// This is useful for Schwartzian transform patterns where index is needed.
    #[inline]
    pub fn enumerate_map_collect<T, U, F>(slice: &[T], f: F) -> Vec<U>
    where
        F: Fn(usize, &T) -> U,
    {
        slice.iter().enumerate().map(|(i, x)| f(i, x)).collect()
    }

    /// Partial sort: efficiently find and sort the top-k elements (sequential).
    /// Uses select_nth_unstable for O(n) partitioning, then sorts only top-k.
    #[inline]
    pub fn partial_sort_by<T, F>(slice: &mut [T], k: usize, mut compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        if k == 0 || slice.is_empty() {
            return;
        }
        let k = k.min(slice.len());

        if k >= slice.len() / 2 {
            slice.sort_by(|a, b| compare(a, b));
            return;
        }

        // Partition: O(n) average
        slice.select_nth_unstable_by(k - 1, &mut compare);

        // Sort only top-k: O(k log k)
        slice[..k].sort_by(compare);
    }

    /// Filter a vector by consuming it and returning matching elements (sequential).
    #[inline]
    pub fn filter_into_vec<T, F>(vec: Vec<T>, f: F) -> Vec<T>
    where
        F: Fn(&T) -> bool,
    {
        vec.into_iter().filter(|x| f(x)).collect()
    }

    // =========================================================================
    // Sequential versions (same as regular on WASM, for API compatibility)
    // =========================================================================

    /// Sort a slice using a comparison function (sequential).
    #[inline]
    pub fn sort_by_seq<T, F>(slice: &mut [T], compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        slice.sort_by(compare);
    }

    /// Sort a slice using a comparison function (unstable, sequential).
    #[inline]
    pub fn sort_unstable_by_seq<T, F>(slice: &mut [T], compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        slice.sort_unstable_by(compare);
    }
}

// =============================================================================
// Public re-exports
// =============================================================================

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

#[cfg(target_arch = "wasm32")]
pub use wasm::*;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_by_small() {
        let mut data = vec![3, 1, 4, 1, 5, 9, 2, 6];
        sort_by(&mut data, |a, b| a.cmp(b));
        assert_eq!(data, vec![1, 1, 2, 3, 4, 5, 6, 9]);
    }

    #[test]
    fn test_sort_by_large() {
        let mut data: Vec<i32> = (0..1000).rev().collect();
        sort_by(&mut data, |a, b| a.cmp(b));
        let expected: Vec<i32> = (0..1000).collect();
        assert_eq!(data, expected);
    }

    #[test]
    fn test_sort_unstable_by() {
        let mut data = vec![3, 1, 4, 1, 5, 9, 2, 6];
        sort_unstable_by(&mut data, |a, b| a.cmp(b));
        assert_eq!(data, vec![1, 1, 2, 3, 4, 5, 6, 9]);
    }

    #[test]
    fn test_map_collect() {
        let data = vec![1, 2, 3, 4, 5];
        let result = map_collect(&data, |x| x * 2);
        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_map_into_collect() {
        let data = vec![1, 2, 3, 4, 5];
        let result = map_into_collect(data, |x| x * 2);
        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_filter_map_collect() {
        let data = vec![1, 2, 3, 4, 5];
        let result = filter_map_collect(&data, |x| if *x % 2 == 0 { Some(x * 2) } else { None });
        assert_eq!(result, vec![4, 8]);
    }

    #[test]
    fn test_filter_collect() {
        let data = vec![1, 2, 3, 4, 5];
        let result = filter_collect(&data, |x| *x % 2 == 0);
        assert_eq!(result, vec![2, 4]);
    }

    #[test]
    fn test_partial_sort_by_small_k() {
        let mut data = vec![9, 3, 7, 1, 5, 8, 2, 6, 4, 0];
        partial_sort_by(&mut data, 3, |a, b| a.cmp(b));
        // Top 3 smallest should be sorted at the beginning
        assert_eq!(&data[..3], &[0, 1, 2]);
    }

    #[test]
    fn test_partial_sort_by_large_k() {
        let mut data = vec![9, 3, 7, 1, 5, 8, 2, 6, 4, 0];
        partial_sort_by(&mut data, 8, |a, b| a.cmp(b));
        // When k >= len/2, does full sort
        assert_eq!(data, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_partial_sort_by_descending() {
        let mut data = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
        partial_sort_by(&mut data, 3, |a, b| b.cmp(a)); // Descending
        // Top 3 largest should be sorted at the beginning
        assert_eq!(&data[..3], &[9, 6, 5]);
    }
}
