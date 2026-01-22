# SIMD Optimization

LatticeDB uses **Single Instruction, Multiple Data (SIMD)** instructions to accelerate distance calculations. This chapter explains the SIMD implementation and how to maximize performance.

## Supported Platforms

| Platform | Instruction Set | Vectors per Cycle | Feature |
|----------|----------------|-------------------|---------|
| x86_64 | AVX2 + FMA | 8 × f32 | `simd` |
| aarch64 | NEON | 4-16 × f32 (unrolled) | `simd` |
| WASM | Scalar (auto-vectorized) | 4 × f32 | - |

## Enabling SIMD

SIMD is enabled by default via the `simd` feature flag:

```toml
# Cargo.toml
[dependencies]
lattice-core = { version = "0.1", features = ["simd"] }  # default
```

To disable SIMD (for debugging or compatibility):

```toml
[dependencies]
lattice-core = { version = "0.1", default-features = false }
```

## Runtime Detection

LatticeDB detects SIMD support at runtime and falls back to scalar code if unavailable:

```rust
// x86_64: Check for AVX2 + FMA
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
static SIMD_SUPPORT: OnceLock<bool> = OnceLock::new();

fn has_avx2_fma() -> bool {
    *SIMD_SUPPORT.get_or_init(|| {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    })
}
```

The check is cached after the first call, so there's no per-operation overhead.

## x86_64 Implementation (AVX2)

### Cosine Distance

Processes 8 floats per iteration:

```rust
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

        // Fused multiply-add: dot_sum += va * vb
        dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
        norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
        norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
    }

    // Horizontal sum and scalar remainder handling
    let dot = hsum_avx(dot_sum) + scalar_remainder(...);
    let norm_a = hsum_avx(norm_a_sum) + scalar_remainder(...);
    let norm_b = hsum_avx(norm_b_sum) + scalar_remainder(...);

    1.0 - (dot / (norm_a * norm_b).sqrt())
}
```

### Euclidean Distance

```rust
#[target_feature(enable = "avx2")]
pub unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);  // sum += diff²
    }

    hsum_avx(sum).sqrt()
}
```

### Horizontal Sum

Reduces 8 floats to 1:

```rust
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx(v: __m256) -> f32 {
    // [a,b,c,d,e,f,g,h] -> [a+e,b+f,c+g,d+h]
    let vlow = _mm256_castps256_ps128(v);
    let vhigh = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(vlow, vhigh);

    // [a+e,b+f,c+g,d+h] -> [a+e+c+g,b+f+d+h]
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);

    // Final reduction
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    _mm_cvtss_f32(_mm_add_ss(sum64, hi32))
}
```

## aarch64 Implementation (NEON)

### 4x Unrolling

Apple Silicon (M1/M2) can sustain 4 FMA operations per cycle, so we unroll 4x:

```rust
pub unsafe fn cosine_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    let chunks16 = a.len() / 16;

    // 4 accumulators for pipeline utilization
    let mut dot0 = vdupq_n_f32(0.0);
    let mut dot1 = vdupq_n_f32(0.0);
    let mut dot2 = vdupq_n_f32(0.0);
    let mut dot3 = vdupq_n_f32(0.0);

    for i in 0..chunks16 {
        let base = i * 16;

        // Load 16 floats (4 NEON registers each)
        let va0 = vld1q_f32(a.as_ptr().add(base));
        let va1 = vld1q_f32(a.as_ptr().add(base + 4));
        let va2 = vld1q_f32(a.as_ptr().add(base + 8));
        let va3 = vld1q_f32(a.as_ptr().add(base + 12));

        let vb0 = vld1q_f32(b.as_ptr().add(base));
        let vb1 = vld1q_f32(b.as_ptr().add(base + 4));
        let vb2 = vld1q_f32(b.as_ptr().add(base + 8));
        let vb3 = vld1q_f32(b.as_ptr().add(base + 12));

        // 4 independent FMAs per cycle
        dot0 = vfmaq_f32(dot0, va0, vb0);
        dot1 = vfmaq_f32(dot1, va1, vb1);
        dot2 = vfmaq_f32(dot2, va2, vb2);
        dot3 = vfmaq_f32(dot3, va3, vb3);
    }

    // Combine accumulators
    let sum = vaddq_f32(vaddq_f32(dot0, dot1), vaddq_f32(dot2, dot3));
    vaddvq_f32(sum)  // NEON horizontal sum
}
```

### Performance Impact

| Platform | Scalar | SIMD | Speedup |
|----------|--------|------|---------|
| x86_64 (AVX2) | 120 ns | 25 ns | 4.8x |
| M1 (NEON, 1x) | 90 ns | 22 ns | 4.1x |
| M1 (NEON, 4x) | 90 ns | 14 ns | 6.4x |

## Scalar Fallback

For small vectors or unsupported platforms:

```rust
fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    // 4x unroll for better compiler auto-vectorization
    let chunks = a.len() / 4;

    for i in 0..chunks {
        let base = i * 4;
        let (a0, a1, a2, a3) = (a[base], a[base+1], a[base+2], a[base+3]);
        let (b0, b1, b2, b3) = (b[base], b[base+1], b[base+2], b[base+3]);

        dot += a0*b0 + a1*b1 + a2*b2 + a3*b3;
        norm_a += a0*a0 + a1*a1 + a2*a2 + a3*a3;
        norm_b += b0*b0 + b1*b1 + b2*b2 + b3*b3;
    }
    // ... handle remainder
}
```

## WASM Considerations

### No SIMD.js by Default

WebAssembly SIMD is available but requires explicit opt-in:

```toml
# .cargo/config.toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+simd128"]
```

LatticeDB uses scalar code for WASM to ensure broad browser compatibility.

### Browser SIMD Support

| Browser | WASM SIMD | Status |
|---------|-----------|--------|
| Chrome 91+ | ✅ | Stable |
| Firefox 89+ | ✅ | Stable |
| Safari 16.4+ | ✅ | Stable |
| Edge 91+ | ✅ | Stable |

## Dispatch Logic

The distance functions automatically select the best implementation:

```rust
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    // x86_64: Use AVX2 for vectors >= 16 elements
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if a.len() >= 16 && has_avx2_fma() {
            return unsafe { simd_x86::cosine_distance_avx2(a, b) };
        }
    }

    // aarch64: Use NEON for vectors >= 8 elements
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if a.len() >= 8 {
            return unsafe { simd_neon::cosine_distance_neon(a, b) };
        }
    }

    // Fallback
    cosine_distance_scalar(a, b)
}
```

## Benchmarking SIMD

### Quick Benchmark

```bash
cargo run -p lattice-bench --release --example quick_vector_bench
```

### Full Criterion Benchmark

```bash
cargo bench -p lattice-bench --bench vector_ops
```

### Compare Scalar vs SIMD

```rust
// Force scalar for comparison
let scalar_time = {
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = cosine_distance_scalar(&a, &b);
    }
    start.elapsed()
};

// Dispatch (uses SIMD if available)
let simd_time = {
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = cosine_distance(&a, &b);
    }
    start.elapsed()
};

println!("Speedup: {:.2}x", scalar_time.as_nanos() / simd_time.as_nanos());
```

## Tips for Maximum Performance

### 1. Use Aligned Vectors

Aligned loads are faster than unaligned:

```rust
// Allocate aligned memory (not directly supported in stable Rust)
// LatticeDB uses unaligned loads for flexibility
```

### 2. Prefer Power-of-2 Dimensions

```rust
// Good: Multiple of 8 (AVX2) or 16 (NEON 4x)
let dim = 128;  // 16 chunks of 8

// Less efficient: Remainder handling needed
let dim = 100;  // 12 chunks of 8 + 4 remainder
```

### 3. Batch Operations

Amortize function call overhead:

```rust
// Good: Batch multiple distances
let distances = index.calc_distances_batch(&query, &neighbor_ids);

// Less efficient: Individual calls
for id in neighbor_ids {
    let dist = calc_distance(&query, &vectors[id]);
}
```

### 4. Keep Vectors Hot

Access vectors sequentially to keep them in cache:

```rust
// Good: Sequential access
for id in 0..n {
    process(vectors.get_by_idx(id));
}

// Less efficient: Random access
for id in random_ids {
    process(vectors.get(id));  // Cache misses
}
```

## Next Steps

- [Performance Tuning](../performance/tuning.md) - Overall optimization strategies
- [Benchmarks](../performance/benchmarks.md) - Detailed performance numbers
