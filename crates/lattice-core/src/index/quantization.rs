//! Scalar quantization for memory-efficient vector storage
//!
//! Int8 scalar quantization reduces memory usage by 4x (f32 -> i8) while
//! maintaining good recall for vector similarity search.
//!
//! # Quantization Formula
//!
//! ```text
//! quantized[i] = round((original[i] - offset) / scale)
//! dequantized[i] = quantized[i] * scale + offset
//! ```
//!
//! where:
//! - offset = min(original)
//! - scale = (max(original) - min(original)) / 255

/// Scalar quantized vector (int8)
///
/// Stores vectors as i8 values with scale and offset for dequantization.
/// Memory usage: N bytes + 8 bytes overhead (vs 4N bytes for f32).
#[derive(Debug, Clone)]
pub struct QuantizedVector {
    /// Quantized data (int8)
    data: Vec<i8>,
    /// Scale factor for dequantization
    scale: f32,
    /// Offset for dequantization
    offset: f32,
}

impl QuantizedVector {
    /// Quantize a f32 vector to int8
    ///
    /// Uses min-max scaling to map values to [-128, 127].
    pub fn quantize(vector: &[f32]) -> Self {
        if vector.is_empty() {
            return Self {
                data: Vec::new(),
                scale: 1.0,
                offset: 0.0,
            };
        }

        // Find min and max
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in vector {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        // Handle constant vectors
        if (max_val - min_val).abs() < f32::EPSILON {
            // For constant vectors, store mid-range quantized value
            // dequantize: (0 + 128) * scale + offset = 128 * 0 + min_val = min_val
            return Self {
                data: vec![0i8; vector.len()],
                scale: 0.0, // Zero scale means constant
                offset: min_val,
            };
        }

        // Compute scale and offset for mapping [min, max] -> [0, 255]
        let scale = (max_val - min_val) / 255.0;
        let offset = min_val;

        // Quantize: map to [0, 255] then shift to [-128, 127]
        let data: Vec<i8> = vector
            .iter()
            .map(|&v| {
                let normalized = (v - offset) / scale;
                let clamped = normalized.clamp(0.0, 255.0);
                (clamped.round() - 128.0) as i8
            })
            .collect();

        Self {
            data,
            scale,
            offset,
        }
    }

    /// Dequantize back to f32 vector
    ///
    /// Note: This is lossy - the original precision is not recovered.
    pub fn dequantize(&self) -> Vec<f32> {
        // Handle constant vectors (scale = 0)
        if self.scale == 0.0 {
            return vec![self.offset; self.data.len()];
        }

        self.data
            .iter()
            .map(|&q| ((q as f32 + 128.0) * self.scale) + self.offset)
            .collect()
    }

    /// Get the quantized data
    pub fn data(&self) -> &[i8] {
        &self.data
    }

    /// Get the scale factor
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get the offset
    pub fn offset(&self) -> f32 {
        self.offset
    }

    /// Get the number of dimensions
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Dequantize a single value inline
    #[inline]
    fn dequantize_value(&self, q: i8) -> f32 {
        if self.scale == 0.0 {
            self.offset
        } else {
            ((q as f32 + 128.0) * self.scale) + self.offset
        }
    }

    /// Asymmetric dot product distance
    ///
    /// Computes distance between a quantized vector (this) and a f32 query.
    /// This is the most common operation in search: quantized database vectors
    /// vs full-precision query vector.
    ///
    /// Returns negated dot product (lower = more similar).
    #[inline]
    pub fn dot_distance_asymmetric(&self, query: &[f32]) -> f32 {
        debug_assert_eq!(self.data.len(), query.len(), "Dimension mismatch");

        let mut dot = 0.0f32;

        // Unroll by 4 for better performance
        let chunks = self.data.len() / 4;
        let remainder = self.data.len() % 4;

        for i in 0..chunks {
            let base = i * 4;
            // Dequantize inline
            let d0 = self.dequantize_value(self.data[base]);
            let d1 = self.dequantize_value(self.data[base + 1]);
            let d2 = self.dequantize_value(self.data[base + 2]);
            let d3 = self.dequantize_value(self.data[base + 3]);

            dot += d0 * query[base]
                + d1 * query[base + 1]
                + d2 * query[base + 2]
                + d3 * query[base + 3];
        }

        for i in 0..remainder {
            let idx = chunks * 4 + i;
            let dequantized = self.dequantize_value(self.data[idx]);
            dot += dequantized * query[idx];
        }

        -dot
    }

    /// Asymmetric euclidean distance
    ///
    /// Computes L2 distance between a quantized vector and a f32 query.
    #[inline]
    pub fn euclidean_distance_asymmetric(&self, query: &[f32]) -> f32 {
        debug_assert_eq!(self.data.len(), query.len(), "Dimension mismatch");

        let mut sum = 0.0f32;

        let chunks = self.data.len() / 4;
        let remainder = self.data.len() % 4;

        for i in 0..chunks {
            let base = i * 4;
            let d0 = self.dequantize_value(self.data[base]) - query[base];
            let d1 = self.dequantize_value(self.data[base + 1]) - query[base + 1];
            let d2 = self.dequantize_value(self.data[base + 2]) - query[base + 2];
            let d3 = self.dequantize_value(self.data[base + 3]) - query[base + 3];

            sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }

        for i in 0..remainder {
            let idx = chunks * 4 + i;
            let diff = self.dequantize_value(self.data[idx]) - query[idx];
            sum += diff * diff;
        }

        sum.sqrt()
    }

    /// Asymmetric cosine distance
    ///
    /// Computes cosine distance between a quantized vector and a f32 query.
    #[inline]
    pub fn cosine_distance_asymmetric(&self, query: &[f32]) -> f32 {
        debug_assert_eq!(self.data.len(), query.len(), "Dimension mismatch");

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for (i, &q) in self.data.iter().enumerate() {
            let dequantized = self.dequantize_value(q);
            dot += dequantized * query[i];
            norm_a += dequantized * dequantized;
            norm_b += query[i] * query[i];
        }

        let norm_product = (norm_a * norm_b).sqrt();
        if norm_product == 0.0 {
            return 1.0;
        }

        1.0 - (dot / norm_product)
    }

    /// Memory size in bytes
    pub fn memory_size(&self) -> usize {
        self.data.len() + 8 // data + scale + offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    const EPSILON: f32 = 0.1; // Quantization has some error

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let original = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let quantized = QuantizedVector::quantize(&original);
        let recovered = quantized.dequantize();

        // Check approximate equality
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!(
                approx_eq(*o, *r, EPSILON),
                "Mismatch: original={}, recovered={}",
                o,
                r
            );
        }
    }

    #[test]
    fn test_quantize_preserves_dimension() {
        let original = vec![1.0; 128];
        let quantized = QuantizedVector::quantize(&original);

        assert_eq!(quantized.len(), 128);
    }

    #[test]
    fn test_quantize_memory_reduction() {
        let dim = 128;
        let original_size = dim * 4; // f32 = 4 bytes
        let quantized = QuantizedVector::quantize(&vec![0.5; dim]);

        // Should be ~4x smaller (+ 8 bytes overhead)
        assert!(
            quantized.memory_size() < original_size / 2,
            "Memory size {} should be less than {}",
            quantized.memory_size(),
            original_size / 2
        );
    }

    #[test]
    fn test_quantize_empty_vector() {
        let quantized = QuantizedVector::quantize(&[]);
        assert!(quantized.is_empty());
        assert_eq!(quantized.dequantize().len(), 0);
    }

    #[test]
    fn test_quantize_constant_vector() {
        let original = vec![0.5; 100];
        let quantized = QuantizedVector::quantize(&original);
        let recovered = quantized.dequantize();

        for r in recovered {
            assert!(approx_eq(r, 0.5, EPSILON), "Recovered value {} != 0.5", r);
        }
    }

    #[test]
    fn test_dot_distance_asymmetric() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let quantized = QuantizedVector::quantize(&a);
        let query = vec![1.0, 1.0, 1.0, 1.0];

        // Dot product = 1*1 + 2*1 + 3*1 + 4*1 = 10
        // Distance = -10
        let distance = quantized.dot_distance_asymmetric(&query);

        // Allow some error due to quantization
        assert!(
            approx_eq(distance, -10.0, 1.0),
            "Distance {} should be close to -10",
            distance
        );
    }

    #[test]
    fn test_euclidean_distance_asymmetric() {
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let quantized = QuantizedVector::quantize(&a);
        let query = vec![3.0, 4.0, 0.0, 0.0];

        // Distance should be 5 (3-4-5 triangle)
        let distance = quantized.euclidean_distance_asymmetric(&query);

        assert!(
            approx_eq(distance, 5.0, 0.5),
            "Distance {} should be close to 5",
            distance
        );
    }

    #[test]
    fn test_cosine_distance_asymmetric() {
        let a = vec![1.0, 0.0];
        let quantized = QuantizedVector::quantize(&a);
        let query = vec![0.0, 1.0];

        // Orthogonal vectors should have cosine distance of 1
        let distance = quantized.cosine_distance_asymmetric(&query);

        assert!(
            approx_eq(distance, 1.0, EPSILON),
            "Distance {} should be close to 1",
            distance
        );
    }

    #[test]
    fn test_quantized_search_ordering() {
        // Verify that quantization preserves relative ordering for search
        let db_vectors = [
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 0.9, 0.1, 0.0],
            vec![0.0, 0.0, 0.9, 0.1],
        ];
        let query = vec![1.0, 0.0, 0.0, 0.0];

        // First vector should be closest to query
        let quantized: Vec<_> = db_vectors
            .iter()
            .map(|v| QuantizedVector::quantize(v))
            .collect();

        let distances: Vec<f32> = quantized
            .iter()
            .map(|q| q.dot_distance_asymmetric(&query))
            .collect();

        assert!(
            distances[0] < distances[1] && distances[0] < distances[2],
            "First vector should be closest: {:?}",
            distances
        );
    }

    #[test]
    fn test_higher_dimensional_quantization() {
        let dim = 256;
        let original: Vec<f32> = (0..dim).map(|i| (i as f32) / 100.0).collect();
        let quantized = QuantizedVector::quantize(&original);
        let recovered = quantized.dequantize();

        // Check that values are approximately preserved
        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!(
                approx_eq(*o, *r, EPSILON),
                "Mismatch at value: original={}, recovered={}",
                o,
                r
            );
        }
    }
}
