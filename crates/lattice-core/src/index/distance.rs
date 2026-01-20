//! Distance calculation functions for vector similarity
//!
//! All distance functions return a value where lower = more similar,
//! making them suitable for min-heap priority queues.
//!
//! # SIMD Optimization
//!
//! When the `simd` feature is enabled and on x86_64, this module uses
//! AVX2 intrinsics for 4-8x speedup on large vectors.

use crate::types::collection::Distance;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::sync::OnceLock;

/// Cached SIMD support detection (avoids per-call overhead from is_x86_feature_detected!)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
static SIMD_AVX2_FMA_SUPPORT: OnceLock<bool> = OnceLock::new();

/// Check if AVX2 and FMA are supported (cached after first call)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn has_avx2_fma() -> bool {
    *SIMD_AVX2_FMA_SUPPORT.get_or_init(|| {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    })
}

/// Distance calculator for vectors
///
/// Provides efficient distance calculations for different metrics.
pub struct DistanceCalculator {
    metric: Distance,
}

impl DistanceCalculator {
    /// Create a new distance calculator
    pub fn new(metric: Distance) -> Self {
        Self { metric }
    }

    /// Calculate distance between two vectors
    ///
    /// Returns a value where lower = more similar.
    /// Panics if vectors have different dimensions.
    #[inline]
    pub fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vector dimension mismatch");

        match self.metric {
            Distance::Cosine => cosine_distance(a, b),
            Distance::Euclid => euclidean_distance(a, b),
            Distance::Dot => dot_distance(a, b),
        }
    }

    /// Get the metric type
    pub fn metric(&self) -> Distance {
        self.metric
    }
}

// ============================================================================
// Scalar implementations (fallback / small vectors)
// ============================================================================

/// Scalar cosine distance implementation
#[inline]
fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    // Unroll by 4 for better auto-vectorization
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        let (a0, a1, a2, a3) = (a[base], a[base + 1], a[base + 2], a[base + 3]);
        let (b0, b1, b2, b3) = (b[base], b[base + 1], b[base + 2], b[base + 3]);

        dot += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        norm_a += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
        norm_b += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
    }

    // Handle remainder
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        dot += a[idx] * b[idx];
        norm_a += a[idx] * a[idx];
        norm_b += b[idx] * b[idx];
    }

    let norm_product = (norm_a * norm_b).sqrt();
    if norm_product == 0.0 {
        return 1.0;
    }

    1.0 - (dot / norm_product)
}

/// Scalar euclidean distance implementation
#[inline]
fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    // Unroll by 4 for better auto-vectorization
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];

        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    // Handle remainder
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        let diff = a[idx] - b[idx];
        sum += diff * diff;
    }

    sum.sqrt()
}

/// Scalar dot product distance implementation
#[inline]
fn dot_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;

    // Unroll by 4 for better auto-vectorization
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        dot += a[base] * b[base]
            + a[base + 1] * b[base + 1]
            + a[base + 2] * b[base + 2]
            + a[base + 3] * b[base + 3];
    }

    // Handle remainder
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        dot += a[idx] * b[idx];
    }

    -dot
}

// ============================================================================
// SIMD implementations (x86_64 with AVX2)
// ============================================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod simd_x86 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// AVX2 cosine distance (processes 8 floats at a time)
    ///
    /// # Safety
    /// Requires AVX2 support
    #[target_feature(enable = "avx2")]
    pub unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let chunks = len / 8;

        let mut dot_sum = _mm256_setzero_ps();
        let mut norm_a_sum = _mm256_setzero_ps();
        let mut norm_b_sum = _mm256_setzero_ps();

        for i in 0..chunks {
            let base = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(base));
            let vb = _mm256_loadu_ps(b.as_ptr().add(base));

            dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
            norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
            norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
        }

        // Horizontal sum
        let dot = hsum_avx(dot_sum);
        let norm_a = hsum_avx(norm_a_sum);
        let norm_b = hsum_avx(norm_b_sum);

        // Handle remainder with scalar
        let remainder_start = chunks * 8;
        let (mut dot_r, mut norm_a_r, mut norm_b_r) = (0.0f32, 0.0f32, 0.0f32);
        for i in remainder_start..len {
            dot_r += a[i] * b[i];
            norm_a_r += a[i] * a[i];
            norm_b_r += b[i] * b[i];
        }

        let total_dot = dot + dot_r;
        let total_norm_a = norm_a + norm_a_r;
        let total_norm_b = norm_b + norm_b_r;

        let norm_product = (total_norm_a * total_norm_b).sqrt();
        if norm_product == 0.0 {
            return 1.0;
        }

        1.0 - (total_dot / norm_product)
    }

    /// AVX2 euclidean distance (processes 8 floats at a time)
    ///
    /// # Safety
    /// Requires AVX2 support
    #[target_feature(enable = "avx2")]
    pub unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let chunks = len / 8;

        let mut sum = _mm256_setzero_ps();

        for i in 0..chunks {
            let base = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(base));
            let vb = _mm256_loadu_ps(b.as_ptr().add(base));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let mut result = hsum_avx(sum);

        // Handle remainder with scalar
        for i in (chunks * 8)..len {
            let diff = a[i] - b[i];
            result += diff * diff;
        }

        result.sqrt()
    }

    /// AVX2 dot product distance (processes 8 floats at a time)
    ///
    /// # Safety
    /// Requires AVX2 support
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let chunks = len / 8;

        let mut sum = _mm256_setzero_ps();

        for i in 0..chunks {
            let base = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(base));
            let vb = _mm256_loadu_ps(b.as_ptr().add(base));
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        let mut result = hsum_avx(sum);

        // Handle remainder with scalar
        for i in (chunks * 8)..len {
            result += a[i] * b[i];
        }

        -result
    }

    /// Horizontal sum of __m256 (8 floats -> 1 float)
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_avx(v: __m256) -> f32 {
        // [a, b, c, d, e, f, g, h] -> [a+e, b+f, c+g, d+h, ...]
        let vlow = _mm256_castps256_ps128(v);
        let vhigh = _mm256_extractf128_ps(v, 1);
        let sum128 = _mm_add_ps(vlow, vhigh);

        // [a+e, b+f, c+g, d+h] -> [a+e+c+g, b+f+d+h, ...]
        let hi64 = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, hi64);

        // [a+e+c+g, b+f+d+h, ...] -> [a+e+c+g+b+f+d+h, ...]
        let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
        let sum32 = _mm_add_ss(sum64, hi32);

        _mm_cvtss_f32(sum32)
    }
}

// ============================================================================
// SIMD implementations (aarch64 with NEON)
// ============================================================================

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod simd_neon {
    use std::arch::aarch64::*;

    /// NEON cosine distance (processes 4 floats at a time)
    ///
    /// # Safety
    /// Requires aarch64 NEON support (always available on aarch64)
    #[inline]
    pub unsafe fn cosine_distance_neon(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let chunks = len / 4;

        let mut dot_sum = vdupq_n_f32(0.0);
        let mut norm_a_sum = vdupq_n_f32(0.0);
        let mut norm_b_sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let base = i * 4;
            let va = vld1q_f32(a.as_ptr().add(base));
            let vb = vld1q_f32(b.as_ptr().add(base));

            dot_sum = vfmaq_f32(dot_sum, va, vb);
            norm_a_sum = vfmaq_f32(norm_a_sum, va, va);
            norm_b_sum = vfmaq_f32(norm_b_sum, vb, vb);
        }

        // Horizontal sum
        let dot = vaddvq_f32(dot_sum);
        let norm_a = vaddvq_f32(norm_a_sum);
        let norm_b = vaddvq_f32(norm_b_sum);

        // Handle remainder with scalar
        let remainder_start = chunks * 4;
        let (mut dot_r, mut norm_a_r, mut norm_b_r) = (0.0f32, 0.0f32, 0.0f32);
        for i in remainder_start..len {
            dot_r += a[i] * b[i];
            norm_a_r += a[i] * a[i];
            norm_b_r += b[i] * b[i];
        }

        let total_dot = dot + dot_r;
        let total_norm_a = norm_a + norm_a_r;
        let total_norm_b = norm_b + norm_b_r;

        let norm_product = (total_norm_a * total_norm_b).sqrt();
        if norm_product == 0.0 {
            return 1.0;
        }

        1.0 - (total_dot / norm_product)
    }

    /// NEON euclidean distance (processes 4 floats at a time)
    #[inline]
    pub unsafe fn euclidean_distance_neon(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let chunks = len / 4;

        let mut sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let base = i * 4;
            let va = vld1q_f32(a.as_ptr().add(base));
            let vb = vld1q_f32(b.as_ptr().add(base));
            let diff = vsubq_f32(va, vb);
            sum = vfmaq_f32(sum, diff, diff);
        }

        let mut result = vaddvq_f32(sum);

        // Handle remainder with scalar
        for i in (chunks * 4)..len {
            let diff = a[i] - b[i];
            result += diff * diff;
        }

        result.sqrt()
    }

    /// NEON dot product distance (processes 4 floats at a time)
    #[inline]
    pub unsafe fn dot_distance_neon(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let chunks = len / 4;

        let mut sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let base = i * 4;
            let va = vld1q_f32(a.as_ptr().add(base));
            let vb = vld1q_f32(b.as_ptr().add(base));
            sum = vfmaq_f32(sum, va, vb);
        }

        let mut result = vaddvq_f32(sum);

        // Handle remainder with scalar
        for i in (chunks * 4)..len {
            result += a[i] * b[i];
        }

        -result
    }
}

// ============================================================================
// Dispatch functions (select best implementation at runtime)
// ============================================================================

/// Cosine distance: 1 - cosine_similarity
///
/// Range: [0, 2] where 0 = identical, 1 = orthogonal, 2 = opposite
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        // Use AVX2 for large vectors if available (cached detection)
        if a.len() >= 32 && has_avx2_fma() {
            return unsafe { simd_x86::cosine_distance_avx2(a, b) };
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        // Use NEON for vectors >= 16 elements (NEON always available on aarch64)
        if a.len() >= 16 {
            return unsafe { simd_neon::cosine_distance_neon(a, b) };
        }
    }

    cosine_distance_scalar(a, b)
}

/// Euclidean distance (L2)
///
/// Range: [0, ∞) where 0 = identical
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        // Use AVX2 for large vectors if available (cached detection)
        if a.len() >= 32 && has_avx2_fma() {
            return unsafe { simd_x86::euclidean_distance_avx2(a, b) };
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        // Use NEON for vectors >= 16 elements (NEON always available on aarch64)
        if a.len() >= 16 {
            return unsafe { simd_neon::euclidean_distance_neon(a, b) };
        }
    }

    euclidean_distance_scalar(a, b)
}

/// Dot product distance: -dot_product
///
/// Negated so lower = more similar (higher dot product = more similar)
/// Range: (-∞, ∞)
#[inline]
pub fn dot_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        // Use AVX2 for large vectors if available (cached detection)
        if a.len() >= 32 && has_avx2_fma() {
            return unsafe { simd_x86::dot_distance_avx2(a, b) };
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        // Use NEON for vectors >= 16 elements (NEON always available on aarch64)
        if a.len() >= 16 {
            return unsafe { simd_neon::dot_distance_neon(a, b) };
        }
    }

    dot_distance_scalar(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_cosine_identical_vectors() {
        let calc = DistanceCalculator::new(Distance::Cosine);
        let v = vec![1.0, 2.0, 3.0];
        let dist = calc.calculate(&v, &v);
        assert!(approx_eq(dist, 0.0), "Expected 0, got {}", dist);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let calc = DistanceCalculator::new(Distance::Cosine);
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dist = calc.calculate(&a, &b);
        assert!(approx_eq(dist, 1.0), "Expected 1, got {}", dist);
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let calc = DistanceCalculator::new(Distance::Cosine);
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let dist = calc.calculate(&a, &b);
        assert!(approx_eq(dist, 2.0), "Expected 2, got {}", dist);
    }

    #[test]
    fn test_cosine_normalized_vectors() {
        let calc = DistanceCalculator::new(Distance::Cosine);
        // 45 degree angle
        let a = vec![1.0, 0.0];
        let sqrt2_inv = 1.0 / 2.0_f32.sqrt();
        let b = vec![sqrt2_inv, sqrt2_inv];
        let dist = calc.calculate(&a, &b);
        // cos(45°) ≈ 0.707, distance = 1 - 0.707 ≈ 0.293
        assert!(
            (dist - 0.293).abs() < 0.01,
            "Expected ~0.293, got {}",
            dist
        );
    }

    #[test]
    fn test_cosine_zero_vectors() {
        let calc = DistanceCalculator::new(Distance::Cosine);
        let a = vec![0.0, 0.0];
        let b = vec![0.0, 0.0];
        let dist = calc.calculate(&a, &b);
        assert!(approx_eq(dist, 1.0), "Zero vectors should be orthogonal");
    }

    #[test]
    fn test_euclid_same_point() {
        let calc = DistanceCalculator::new(Distance::Euclid);
        let v = vec![1.0, 2.0, 3.0];
        let dist = calc.calculate(&v, &v);
        assert!(approx_eq(dist, 0.0), "Expected 0, got {}", dist);
    }

    #[test]
    fn test_euclid_unit_distance() {
        let calc = DistanceCalculator::new(Distance::Euclid);
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        let dist = calc.calculate(&a, &b);
        assert!(approx_eq(dist, 1.0), "Expected 1, got {}", dist);
    }

    #[test]
    fn test_euclid_pythagorean() {
        let calc = DistanceCalculator::new(Distance::Euclid);
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = calc.calculate(&a, &b);
        assert!(approx_eq(dist, 5.0), "Expected 5 (3-4-5 triangle), got {}", dist);
    }

    #[test]
    fn test_dot_product_correctness() {
        let calc = DistanceCalculator::new(Distance::Dot);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // dot = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        // distance = -32 (negated)
        let dist = calc.calculate(&a, &b);
        assert!(approx_eq(dist, -32.0), "Expected -32, got {}", dist);
    }

    #[test]
    fn test_dot_identical_unit_vectors() {
        let calc = DistanceCalculator::new(Distance::Dot);
        let v = vec![1.0, 0.0, 0.0];
        // dot(v, v) = 1, distance = -1
        let dist = calc.calculate(&v, &v);
        assert!(approx_eq(dist, -1.0), "Expected -1, got {}", dist);
    }

    #[test]
    fn test_dot_orthogonal_vectors() {
        let calc = DistanceCalculator::new(Distance::Dot);
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        // dot = 0, distance = 0
        let dist = calc.calculate(&a, &b);
        assert!(approx_eq(dist, 0.0), "Expected 0, got {}", dist);
    }

    #[test]
    fn test_higher_dimensional_vectors() {
        let calc = DistanceCalculator::new(Distance::Euclid);
        let dim = 128;
        let a: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i + 1) as f32).collect();
        // Each dimension differs by 1, so distance = sqrt(128)
        let dist = calc.calculate(&a, &b);
        let expected = (dim as f32).sqrt();
        assert!(
            (dist - expected).abs() < 0.01,
            "Expected {}, got {}",
            expected,
            dist
        );
    }

    #[test]
    fn test_distance_ordering() {
        // Verify that lower distance = more similar for all metrics
        let calc_cos = DistanceCalculator::new(Distance::Cosine);
        let calc_euc = DistanceCalculator::new(Distance::Euclid);
        let calc_dot = DistanceCalculator::new(Distance::Dot);

        let query = vec![1.0, 0.0];
        let similar = vec![0.9, 0.1]; // Close to query
        let dissimilar = vec![0.0, 1.0]; // Far from query

        // Similar should have lower distance than dissimilar
        assert!(calc_cos.calculate(&query, &similar) < calc_cos.calculate(&query, &dissimilar));
        assert!(calc_euc.calculate(&query, &similar) < calc_euc.calculate(&query, &dissimilar));
        assert!(calc_dot.calculate(&query, &similar) < calc_dot.calculate(&query, &dissimilar));
    }

    // SIMD-specific tests (only run when SIMD is enabled and available)
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    mod simd_tests {
        use super::*;

        #[test]
        fn test_simd_cosine_large_vector() {
            let calc = DistanceCalculator::new(Distance::Cosine);
            let dim = 256;
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) / 100.0).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i + 1) as f32) / 100.0).collect();

            // Should use SIMD path on x86_64 with AVX2
            let dist = calc.calculate(&a, &b);

            // Verify result is reasonable (identical vectors would be 0)
            assert!(dist >= 0.0 && dist < 1.0, "Cosine distance out of range: {}", dist);
        }

        #[test]
        fn test_simd_euclidean_large_vector() {
            let calc = DistanceCalculator::new(Distance::Euclid);
            let dim = 256;
            let a: Vec<f32> = vec![0.0; dim];
            let b: Vec<f32> = vec![1.0; dim];

            let dist = calc.calculate(&a, &b);
            let expected = (dim as f32).sqrt();

            assert!(
                (dist - expected).abs() < 0.01,
                "Expected {}, got {}",
                expected,
                dist
            );
        }

        #[test]
        fn test_simd_dot_large_vector() {
            let calc = DistanceCalculator::new(Distance::Dot);
            let dim = 256;
            let a: Vec<f32> = vec![1.0; dim];
            let b: Vec<f32> = vec![2.0; dim];

            let dist = calc.calculate(&a, &b);
            let expected = -(dim as f32 * 2.0);

            assert!(
                (dist - expected).abs() < 0.01,
                "Expected {}, got {}",
                expected,
                dist
            );
        }

        #[test]
        fn test_simd_vs_scalar_consistency() {
            // Verify SIMD and scalar produce the same results
            let dim = 128;
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i * 2) as f32) * 0.1).collect();

            // Force scalar
            let scalar_cos = cosine_distance_scalar(&a, &b);
            let scalar_euc = euclidean_distance_scalar(&a, &b);
            let scalar_dot = dot_distance_scalar(&a, &b);

            // Use dispatch (may use SIMD)
            let dispatch_cos = cosine_distance(&a, &b);
            let dispatch_euc = euclidean_distance(&a, &b);
            let dispatch_dot = dot_distance(&a, &b);

            assert!(
                (scalar_cos - dispatch_cos).abs() < 1e-4,
                "Cosine mismatch: scalar={}, dispatch={}",
                scalar_cos,
                dispatch_cos
            );
            assert!(
                (scalar_euc - dispatch_euc).abs() < 1e-4,
                "Euclidean mismatch: scalar={}, dispatch={}",
                scalar_euc,
                dispatch_euc
            );
            assert!(
                (scalar_dot - dispatch_dot).abs() < 1e-4,
                "Dot mismatch: scalar={}, dispatch={}",
                scalar_dot,
                dispatch_dot
            );
        }
    }

    // NEON SIMD tests for aarch64 (Apple Silicon, etc.)
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    mod neon_tests {
        use super::*;

        #[test]
        fn test_neon_cosine_large_vector() {
            let calc = DistanceCalculator::new(Distance::Cosine);
            let dim = 256;
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) / 100.0).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i + 1) as f32) / 100.0).collect();

            // Should use NEON path on aarch64
            let dist = calc.calculate(&a, &b);

            // Verify result is reasonable (identical vectors would be 0)
            assert!(dist >= 0.0 && dist < 1.0, "Cosine distance out of range: {}", dist);
        }

        #[test]
        fn test_neon_euclidean_large_vector() {
            let calc = DistanceCalculator::new(Distance::Euclid);
            let dim = 256;
            let a: Vec<f32> = vec![0.0; dim];
            let b: Vec<f32> = vec![1.0; dim];

            let dist = calc.calculate(&a, &b);
            let expected = (dim as f32).sqrt();

            assert!(
                (dist - expected).abs() < 0.01,
                "Expected {}, got {}",
                expected,
                dist
            );
        }

        #[test]
        fn test_neon_dot_large_vector() {
            let calc = DistanceCalculator::new(Distance::Dot);
            let dim = 256;
            let a: Vec<f32> = vec![1.0; dim];
            let b: Vec<f32> = vec![2.0; dim];

            let dist = calc.calculate(&a, &b);
            let expected = -(dim as f32 * 2.0);

            assert!(
                (dist - expected).abs() < 0.01,
                "Expected {}, got {}",
                expected,
                dist
            );
        }

        #[test]
        fn test_neon_vs_scalar_consistency() {
            // Verify NEON and scalar produce the same results
            let dim = 128;
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i * 2) as f32) * 0.1).collect();

            // Force scalar
            let scalar_cos = cosine_distance_scalar(&a, &b);
            let scalar_euc = euclidean_distance_scalar(&a, &b);
            let scalar_dot = dot_distance_scalar(&a, &b);

            // Use dispatch (uses NEON on aarch64)
            let dispatch_cos = cosine_distance(&a, &b);
            let dispatch_euc = euclidean_distance(&a, &b);
            let dispatch_dot = dot_distance(&a, &b);

            assert!(
                (scalar_cos - dispatch_cos).abs() < 1e-4,
                "Cosine mismatch: scalar={}, dispatch={}",
                scalar_cos,
                dispatch_cos
            );
            assert!(
                (scalar_euc - dispatch_euc).abs() < 1e-4,
                "Euclidean mismatch: scalar={}, dispatch={}",
                scalar_euc,
                dispatch_euc
            );
            assert!(
                (scalar_dot - dispatch_dot).abs() < 1e-4,
                "Dot mismatch: scalar={}, dispatch={}",
                scalar_dot,
                dispatch_dot
            );
        }
    }
}
