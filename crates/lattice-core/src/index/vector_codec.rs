//! Compact, contiguous encoding of a collection's vectors for snapshots.
//!
//! Vectors are already rounded to the collection's [`VectorStorage`] when
//! they are written (see [`VectorStorage::round_in_place`]), so encoding here
//! is lossless: `decode(encode(v)) == v` bit for bit.

use crate::error::{LatticeError, LatticeResult};
use crate::types::vector_index::{exp2, int8_exponent, VectorStorage};
use half::f16;
use rkyv::{Archive, Deserialize, Serialize};

/// Every vector of a snapshot, concatenated in point order.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EncodedVectors {
    /// The collection has no dense vectors.
    Absent,
    F32(Vec<f32>),
    /// IEEE 754 half-precision bit patterns.
    F16(Vec<u16>),
    /// Codes plus one power-of-two scale exponent per vector.
    Int8 {
        codes: Vec<i8>,
        exponents: Vec<i8>,
    },
}

impl EncodedVectors {
    /// Encode `vectors` (each of length `dim`) at `storage` precision; `None`
    /// storage means the collection has no dense vectors.
    pub fn encode<'a, I>(storage: Option<VectorStorage>, dim: usize, vectors: I) -> Self
    where
        I: ExactSizeIterator<Item = &'a [f32]>,
    {
        let n = vectors.len();
        match storage {
            None => EncodedVectors::Absent,
            Some(VectorStorage::F32) => {
                let mut out = Vec::with_capacity(n * dim);
                vectors.for_each(|v| out.extend_from_slice(v));
                EncodedVectors::F32(out)
            }
            Some(VectorStorage::F16) => {
                let mut out = Vec::with_capacity(n * dim);
                vectors.for_each(|v| out.extend(v.iter().map(|x| f16::from_f32(*x).to_bits())));
                EncodedVectors::F16(out)
            }
            Some(VectorStorage::Int8) => {
                let mut codes = Vec::with_capacity(n * dim);
                let mut exponents = Vec::with_capacity(n);
                for v in vectors {
                    let e = int8_exponent(v);
                    let scale = exp2(e);
                    codes.extend(
                        v.iter()
                            .map(|x| (x / scale).round().clamp(-127.0, 127.0) as i8),
                    );
                    exponents.push(e);
                }
                EncodedVectors::Int8 { codes, exponents }
            }
        }
    }

    /// The storage these vectors were encoded at (`None` = no dense vectors).
    pub fn storage(&self) -> Option<VectorStorage> {
        match self {
            EncodedVectors::Absent => None,
            EncodedVectors::F32(_) => Some(VectorStorage::F32),
            EncodedVectors::F16(_) => Some(VectorStorage::F16),
            EncodedVectors::Int8 { .. } => Some(VectorStorage::Int8),
        }
    }

    /// Decode `count` vectors of `dim` dimensions, failing on any length
    /// mismatch (the input is untrusted).
    pub fn decode(&self, dim: usize, count: usize) -> LatticeResult<Vec<Vec<f32>>> {
        let expect = |actual: usize, expected: usize, what: &str| {
            if actual == expected {
                Ok(())
            } else {
                Err(corrupt(format!(
                    "{what} has {actual} values, expected {expected}"
                )))
            }
        };
        let total = dim
            .checked_mul(count)
            .ok_or_else(|| corrupt("vector section size overflows".to_string()))?;
        match self {
            EncodedVectors::Absent => Ok(vec![Vec::new(); count]),
            EncodedVectors::F32(data) => {
                expect(data.len(), total, "f32 vector section")?;
                Ok(chunks(data, dim, count, |x| *x))
            }
            EncodedVectors::F16(data) => {
                expect(data.len(), total, "f16 vector section")?;
                Ok(chunks(data, dim, count, |x| f16_bits_to_f32(*x)))
            }
            EncodedVectors::Int8 { codes, exponents } => {
                expect(codes.len(), total, "int8 vector section")?;
                expect(exponents.len(), count, "int8 scale section")?;
                Ok((0..count)
                    .map(|i| {
                        let scale = exp2(exponents[i]);
                        codes[i * dim..(i + 1) * dim]
                            .iter()
                            .map(|c| f32::from(*c) * scale)
                            .collect()
                    })
                    .collect())
            }
        }
    }
}

/// Branch-light f16 -> f32 (exact for every input, including subnormals,
/// infinities and NaN). `half`'s portable path is several times slower on
/// wasm32, where it has no hardware conversion, and this runs once per
/// stored value on every snapshot load.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let magnitude = u32::from(bits & 0x7fff);
    if magnitude >= 0x7c00 {
        // Infinity or NaN: all-ones exponent, payload carried over.
        return f32::from_bits(sign | 0x7f80_0000 | ((magnitude & 0x03ff) << 13));
    }
    // Place exponent and mantissa in f32 position, then rebias by 2^112
    // (127 - 15); f32 subnormal arithmetic makes f16 subnormals exact too.
    let rebias = f32::from_bits(0x7780_0000);
    f32::from_bits((f32::from_bits(magnitude << 13) * rebias).to_bits() | sign)
}

fn chunks<T>(data: &[T], dim: usize, count: usize, f: impl Fn(&T) -> f32) -> Vec<Vec<f32>> {
    if dim == 0 {
        return vec![Vec::new(); count];
    }
    data.chunks_exact(dim)
        .map(|c| c.iter().map(&f).collect())
        .collect()
}

fn corrupt(message: String) -> LatticeError {
    LatticeError::Serialization {
        message: format!("Invalid snapshot: {message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn rounded(storage: VectorStorage, seed: u32) -> Vec<f32> {
        let mut v: Vec<f32> = (0..16)
            .map(|i| ((seed * 31 + i * 7) % 97) as f32 / 13.0 - 3.5)
            .collect();
        storage.round_in_place(&mut v);
        v
    }

    #[test]
    fn round_trip_is_lossless_for_rounded_vectors() {
        for storage in [VectorStorage::F32, VectorStorage::F16, VectorStorage::Int8] {
            let vectors: Vec<Vec<f32>> = (0..20).map(|s| rounded(storage, s)).collect();
            let encoded =
                EncodedVectors::encode(Some(storage), 16, vectors.iter().map(|v| v.as_slice()));
            assert_eq!(encoded.storage(), Some(storage));
            assert_eq!(encoded.decode(16, 20).unwrap(), vectors, "{storage:?}");
        }
    }

    #[test]
    fn compact_encodings_shrink() {
        let vectors: Vec<Vec<f32>> = (0..4).map(|s| rounded(VectorStorage::Int8, s)).collect();
        let it = || vectors.iter().map(|v| v.as_slice());
        match EncodedVectors::encode(Some(VectorStorage::Int8), 16, it()) {
            EncodedVectors::Int8 { codes, exponents } => {
                assert_eq!((codes.len(), exponents.len()), (64, 4));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn truncated_sections_are_rejected() {
        let short = EncodedVectors::F16(vec![0; 15]);
        assert!(short.decode(16, 1).is_err());
        let scales = EncodedVectors::Int8 {
            codes: vec![0; 32],
            exponents: vec![0],
        };
        assert!(scales.decode(16, 2).is_err());
        assert!(EncodedVectors::F32(vec![]).decode(usize::MAX, 2).is_err());
    }

    #[test]
    fn fast_f16_conversion_matches_half_for_every_bit_pattern() {
        for bits in 0..=u16::MAX {
            let reference = f16::from_bits(bits).to_f32();
            let fast = f16_bits_to_f32(bits);
            if reference.is_nan() {
                assert!(fast.is_nan(), "{bits:#06x}");
            } else {
                assert_eq!(fast.to_bits(), reference.to_bits(), "{bits:#06x}");
            }
        }
    }

    #[test]
    fn absent_vectors_decode_to_empty() {
        let none = EncodedVectors::encode(None, 0, std::iter::empty::<&[f32]>());
        assert_eq!(none.decode(0, 3).unwrap(), vec![Vec::<f32>::new(); 3]);
    }
}
