//! Reciprocal Rank Fusion.
//!
//! `score(d) = Σ_lists 1 / (k + rank(d))`, with `rank` starting at 1.
//! Only ranks matter, so lists whose scores live on incomparable scales (BM25
//! vs. cosine distance) fuse without any normalisation.

use crate::types::point::PointId;
use rustc_hash::{FxHashMap, FxHashSet};

/// RRF constant `k = 60`, from Cormack, Clarke & Büttcher, "Reciprocal Rank
/// Fusion outperforms Condorcet and Individual Rank Learning Methods",
/// SIGIR 2009. (Qdrant's own default is `k = 2`; pass it explicitly for
/// identical scores.)
pub const RRF_K: u32 = 60;

/// Fuse best-first rankings. A point repeated within one list counts once, at
/// its best rank. Output is ordered by fused score descending, then id
/// ascending, and truncated to `limit`.
pub fn reciprocal_rank_fusion(
    rankings: &[Vec<PointId>],
    k: u32,
    limit: usize,
) -> Vec<(PointId, f32)> {
    let mut scores: FxHashMap<PointId, f64> = FxHashMap::default();
    for ranking in rankings {
        let mut seen: FxHashSet<PointId> = FxHashSet::default();
        let mut rank = 0u64;
        for &id in ranking {
            if !seen.insert(id) {
                continue;
            }
            rank += 1;
            *scores.entry(id).or_insert(0.0) += 1.0 / (f64::from(k) + rank as f64);
        }
    }
    let mut fused: Vec<(PointId, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    fused.truncate(limit);
    fused.into_iter().map(|(id, s)| (id, s as f32)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn hand_computed_scores() {
        // A: rank 1 in list 1, rank 3 in list 2 -> 1/61 + 1/63
        // B: rank 2 in list 1                   -> 1/62
        // C: rank 1 in list 2                   -> 1/61
        // D: rank 2 in list 2                   -> 1/62
        let fused = reciprocal_rank_fusion(&[vec![10, 20], vec![30, 40, 10]], RRF_K, 10);
        let ids: Vec<PointId> = fused.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![10, 30, 20, 40]);
        assert!((fused[0].1 - (1.0 / 61.0 + 1.0 / 63.0) as f32).abs() < 1e-7);
        assert!((fused[1].1 - (1.0f64 / 61.0) as f32).abs() < 1e-7);
        assert_eq!(fused[2].1, fused[3].1, "equal scores tie-break by id");
    }

    #[test]
    fn agreement_beats_a_single_first_place() {
        // 2 is second in both lists; 1 and 3 are each first once.
        // 2/62 > 1/61 + 0, so consensus wins.
        let fused = reciprocal_rank_fusion(&[vec![1, 2], vec![3, 2]], RRF_K, 3);
        assert_eq!(fused[0].0, 2);
    }

    #[test]
    fn duplicates_within_a_list_count_once_and_limit_applies() {
        let fused = reciprocal_rank_fusion(&[vec![5, 5, 6]], RRF_K, 1);
        assert_eq!(fused, vec![(5, (1.0f64 / 61.0) as f32)]);
        assert!(reciprocal_rank_fusion(&[], RRF_K, 10).is_empty());
        assert!(reciprocal_rank_fusion(&[vec![1]], RRF_K, 0).is_empty());
    }

    #[test]
    fn k_changes_the_balance() {
        // 1 is first in one list; 2 is fourth in two lists.
        // k=1:  1/2 = 0.5   vs 2/5  = 0.4     -> 1 wins
        // k=60: 1/61 ≈ .016 vs 2/64 ≈ .031    -> 2 wins
        let lists = [vec![1, 7, 8, 2], vec![9, 10, 11, 2]];
        assert_eq!(reciprocal_rank_fusion(&lists, 1, 1)[0].0, 1);
        assert_eq!(reciprocal_rank_fusion(&lists, RRF_K, 1)[0].0, 2);
    }
}
