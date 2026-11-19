//! How a collection indexes and stores its dense vectors.
//!
//! Both settings live on [`super::collection::VectorConfig`]. Configurations
//! written before they existed (snapshots, WAL checkpoints, REST clients that
//! only send `size`/`distance`) mean exactly `hnsw` + `f32`, which is what
//! every collection was before; that is the only reason the enums have a
//! `Default`.

use half::f16;
use serde::{Deserialize, Serialize};

/// Dense vector search structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VectorIndexKind {
    /// Approximate nearest neighbours over an HNSW graph.
    #[default]
    Hnsw,
    /// Exact scan of every vector. No graph: nothing to build on insert or
    /// load, and recall is 100%. The right choice for small corpora (a few
    /// thousand vectors), where a scan costs about as much as a graph walk.
    Flat,
}

impl VectorIndexKind {
    /// The name used in JSON configuration.
    pub fn as_str(self) -> &'static str {
        match self {
            VectorIndexKind::Hnsw => "hnsw",
            VectorIndexKind::Flat => "flat",
        }
    }
}

/// Precision vectors are kept at.
///
/// Vectors are rounded to this precision when they are written, so search
/// results before and after an export/import round trip are identical and the
/// snapshot stores the compact encoding losslessly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VectorStorage {
    /// 4 bytes per dimension, no rounding.
    #[default]
    F32,
    /// IEEE 754 half precision: 2 bytes per dimension.
    F16,
    /// Symmetric scalar quantization: 1 byte per dimension plus a 1-byte
    /// power-of-two scale per vector.
    Int8,
}

/// Largest int8 code magnitude (symmetric range `-127..=127`).
pub const INT8_MAX_CODE: f32 = 127.0;

impl VectorStorage {
    /// The name used in JSON configuration.
    pub fn as_str(self) -> &'static str {
        match self {
            VectorStorage::F32 => "f32",
            VectorStorage::F16 => "f16",
            VectorStorage::Int8 => "int8",
        }
    }

    /// Round `vector` in place to this storage precision.
    pub fn round_in_place(self, vector: &mut [f32]) {
        match self {
            VectorStorage::F32 => {}
            VectorStorage::F16 => {
                for x in vector.iter_mut() {
                    *x = f16::from_f32(*x).to_f32();
                }
            }
            VectorStorage::Int8 => {
                let exponent = int8_exponent(vector);
                let scale = exp2(exponent);
                for x in vector.iter_mut() {
                    *x = (*x / scale).round().clamp(-INT8_MAX_CODE, INT8_MAX_CODE) * scale;
                }
            }
        }
    }

    /// Encoded bytes per vector of `dim` dimensions (snapshot size planning).
    pub fn bytes_per_vector(self, dim: usize) -> usize {
        match self {
            VectorStorage::F32 => dim * 4,
            VectorStorage::F16 => dim * 2,
            VectorStorage::Int8 => dim + 1,
        }
    }
}

/// The power-of-two scale exponent for an int8 vector: the smallest `e` with
/// `max|x| / 2^e <= 127`.
///
/// A power-of-two scale makes `code * 2^e` exact in f32, and re-quantizing an
/// already-rounded vector yields the same exponent and codes, so rounding at
/// write time and encoding at export time agree bit for bit.
pub fn int8_exponent(vector: &[f32]) -> i8 {
    let max = vector.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if max == 0.0 || !max.is_finite() {
        return 0;
    }
    let exponent = (max / INT8_MAX_CODE).log2().ceil();
    // f32 spans 2^-149..2^128; clamp keeps the exponent in i8 and the scale finite.
    let mut e = exponent.clamp(-126.0, 120.0) as i8;
    // log2/ceil can land one below the bound for values just over a power of two.
    while max / exp2(e) > INT8_MAX_CODE && e < 120 {
        e += 1;
    }
    e
}

/// `2^e` for the int8 scale.
pub fn exp2(e: i8) -> f32 {
    2.0f32.powi(i32::from(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn sample(seed: u64, dim: usize) -> Vec<f32> {
        let mut s = seed;
        (0..dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn rounding_is_idempotent_for_every_storage() {
        for storage in [VectorStorage::F32, VectorStorage::F16, VectorStorage::Int8] {
            for seed in 0..200 {
                let mut v: Vec<f32> = sample(seed, 64)
                    .iter()
                    .map(|x| x * (seed as f32 + 0.3))
                    .collect();
                storage.round_in_place(&mut v);
                let once = v.clone();
                storage.round_in_place(&mut v);
                assert_eq!(v, once, "{storage:?} seed {seed}");
            }
        }
    }

    #[test]
    fn int8_error_is_bounded_by_half_a_step() {
        let original = sample(7, 256);
        let mut v = original.clone();
        VectorStorage::Int8.round_in_place(&mut v);
        let step = exp2(int8_exponent(&original));
        for (a, b) in original.iter().zip(&v) {
            assert!((a - b).abs() <= step / 2.0 + f32::EPSILON, "{a} vs {b}");
        }
    }

    #[test]
    fn zero_and_tiny_vectors_survive() {
        let mut zero = vec![0.0; 8];
        VectorStorage::Int8.round_in_place(&mut zero);
        assert_eq!(zero, vec![0.0; 8]);
        let mut tiny = vec![1e-30, -1e-30];
        VectorStorage::Int8.round_in_place(&mut tiny);
        assert!(tiny.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn names_match_serde() {
        for kind in [VectorIndexKind::Hnsw, VectorIndexKind::Flat] {
            assert_eq!(
                serde_json::to_string(&kind).unwrap(),
                format!("\"{}\"", kind.as_str())
            );
        }
        for storage in [VectorStorage::F32, VectorStorage::F16, VectorStorage::Int8] {
            assert_eq!(
                serde_json::to_string(&storage).unwrap(),
                format!("\"{}\"", storage.as_str())
            );
        }
    }

    #[test]
    fn serde_names_are_snake_case() {
        assert_eq!(
            serde_json::to_string(&VectorIndexKind::Flat).unwrap(),
            r#""flat""#
        );
        assert_eq!(
            serde_json::to_string(&VectorStorage::Int8).unwrap(),
            r#""int8""#
        );
        assert!(serde_json::from_str::<VectorStorage>(r#""f64""#).is_err());
    }
}
