//! Qdrant-style sparse vectors: parallel `indices` / `values` arrays.
//!
//! The lexical index turns a text query into a [`SparseVector`] over its own
//! vocabulary (term id -> query term weight) and scores documents against it.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A sparse vector, normalised so that `indices` are strictly increasing.
///
/// JSON shape matches Qdrant: `{"indices": [3, 17], "values": [1.0, 0.5]}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RawSparseVector")]
pub struct SparseVector {
    indices: Vec<u32>,
    values: Vec<f32>,
}

/// Unvalidated wire form; deserialization goes through [`SparseVector::new`].
#[derive(Deserialize)]
struct RawSparseVector {
    indices: Vec<u32>,
    values: Vec<f32>,
}

impl TryFrom<RawSparseVector> for SparseVector {
    type Error = SparseVectorError;

    fn try_from(raw: RawSparseVector) -> Result<Self, Self::Error> {
        Self::new(raw.indices, raw.values)
    }
}

/// Why a sparse vector was rejected.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SparseVectorError {
    #[error("sparse vector has {indices} indices but {values} values")]
    LengthMismatch { indices: usize, values: usize },
    #[error("sparse vector index {index} appears more than once")]
    DuplicateIndex { index: u32 },
    #[error("sparse vector value at index {index} is not finite")]
    NonFiniteValue { index: u32 },
}

impl SparseVector {
    /// Build from parallel arrays in any order; rejects duplicates and NaN/inf.
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Result<Self, SparseVectorError> {
        if indices.len() != values.len() {
            return Err(SparseVectorError::LengthMismatch {
                indices: indices.len(),
                values: values.len(),
            });
        }
        let mut pairs: Vec<(u32, f32)> = indices.into_iter().zip(values).collect();
        pairs.sort_by_key(|&(i, _)| i);
        for window in pairs.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(SparseVectorError::DuplicateIndex { index: window[0].0 });
            }
        }
        if let Some(&(index, _)) = pairs.iter().find(|(_, v)| !v.is_finite()) {
            return Err(SparseVectorError::NonFiniteValue { index });
        }
        let (indices, values) = pairs.into_iter().unzip();
        Ok(Self { indices, values })
    }

    /// Build from `(index, count)` pairs whose indices are already unique
    /// (map keys) and whose values are finite counts — the invariants
    /// [`SparseVector::new`] would otherwise check.
    pub(crate) fn from_unique_counts(mut pairs: Vec<(u32, f32)>) -> Self {
        pairs.sort_by_key(|&(i, _)| i);
        let (indices, values) = pairs.into_iter().unzip();
        Self { indices, values }
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// `(index, value)` pairs in increasing index order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        self.indices
            .iter()
            .copied()
            .zip(self.values.iter().copied())
    }

    /// Dot product by merging the two sorted index lists.
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let (mut i, mut j, mut sum) = (0, 0, 0.0f32);
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    sum += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn sorts_pairs_by_index() {
        let v = SparseVector::new(vec![9, 2, 5], vec![0.9, 0.2, 0.5]).unwrap();
        assert_eq!(v.indices(), &[2, 5, 9]);
        assert_eq!(v.values(), &[0.2, 0.5, 0.9]);
    }

    #[test]
    fn rejects_malformed_input() {
        assert_eq!(
            SparseVector::new(vec![1, 2], vec![1.0]),
            Err(SparseVectorError::LengthMismatch {
                indices: 2,
                values: 1
            })
        );
        assert_eq!(
            SparseVector::new(vec![4, 4], vec![1.0, 2.0]),
            Err(SparseVectorError::DuplicateIndex { index: 4 })
        );
        assert_eq!(
            SparseVector::new(vec![7], vec![f32::NAN]),
            Err(SparseVectorError::NonFiniteValue { index: 7 })
        );
    }

    #[test]
    fn dot_product_over_shared_indices_only() {
        let a = SparseVector::new(vec![1, 3, 5], vec![1.0, 2.0, 3.0]).unwrap();
        let b = SparseVector::new(vec![3, 4, 5], vec![10.0, 100.0, 0.5]).unwrap();
        assert_eq!(a.dot(&b), 2.0 * 10.0 + 3.0 * 0.5);
        assert_eq!(a.dot(&SparseVector::new(vec![], vec![]).unwrap()), 0.0);
    }

    #[test]
    fn json_shape_matches_qdrant() {
        let v = SparseVector::new(vec![3, 1], vec![0.5, 1.0]).unwrap();
        assert_eq!(
            serde_json::to_string(&v).unwrap(),
            r#"{"indices":[1,3],"values":[1.0,0.5]}"#
        );
        let parsed: SparseVector =
            serde_json::from_str(r#"{"indices":[3,1],"values":[0.5,1.0]}"#).unwrap();
        assert_eq!(parsed, v);
        assert!(
            serde_json::from_str::<SparseVector>(r#"{"indices":[1,1],"values":[1,2]}"#).is_err()
        );
    }
}
