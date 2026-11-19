//! Exact (brute-force) nearest-neighbour search.
//!
//! Used by collections configured with `index: flat`: every candidate is
//! scored with the same [`DistanceCalculator`] the HNSW index uses, so scores
//! are directly comparable, and the top `k` are kept in a bounded max-heap.

use crate::index::distance::DistanceCalculator;
use crate::types::point::PointId;
use crate::types::query::SearchResult;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Heap entry ordered by distance, then id (deterministic ties).
struct Scored {
    distance: f32,
    id: PointId,
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Scored {}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.id.cmp(&other.id))
    }
}

/// The `k` candidates closest to `query`, closest first.
pub fn exact_search<'a, I>(
    calc: &DistanceCalculator,
    query: &[f32],
    candidates: I,
    k: usize,
) -> Vec<SearchResult>
where
    I: IntoIterator<Item = (PointId, &'a [f32])>,
{
    if k == 0 {
        return Vec::new();
    }
    let mut heap: BinaryHeap<Scored> = BinaryHeap::with_capacity(k + 1);
    for (id, vector) in candidates {
        let distance = calc.calculate(query, vector);
        if distance.is_nan() {
            continue;
        }
        if heap.len() < k {
            heap.push(Scored { distance, id });
        } else if let Some(worst) = heap.peek() {
            if (Scored { distance, id }) < *worst {
                heap.pop();
                heap.push(Scored { distance, id });
            }
        }
    }
    heap.into_sorted_vec()
        .into_iter()
        .map(|s| SearchResult::new(s.id, s.distance))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::collection::Distance;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn returns_the_true_top_k_in_order() {
        let calc = DistanceCalculator::new(Distance::Euclid);
        let vectors: Vec<(PointId, Vec<f32>)> =
            (0..100u64).map(|i| (i, vec![i as f32, 0.0])).collect();
        let results = exact_search(
            &calc,
            &[41.2, 0.0],
            vectors.iter().map(|(id, v)| (*id, v.as_slice())),
            3,
        );
        let ids: Vec<PointId> = results.iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![41, 42, 40]);
        assert!(results.windows(2).all(|w| w[0].score <= w[1].score));
    }

    #[test]
    fn ties_break_by_id_and_k_is_respected() {
        let calc = DistanceCalculator::new(Distance::Dot);
        let same = [1.0f32, 0.0];
        let results = exact_search(
            &calc,
            &[1.0, 0.0],
            (0..10u64).rev().map(|i| (i, &same[..])),
            4,
        );
        assert_eq!(
            results.iter().map(|r| r.id).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert!(exact_search(&calc, &[1.0, 0.0], [(1u64, &same[..])], 0).is_empty());
    }
}
