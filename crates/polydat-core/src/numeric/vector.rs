// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The f32 vector bodies: the scalar references, the SIMD dispatch,
//! and the element-wise and reducing operations the `vec_*` nodes and
//! their native lowerings share, so both produce the same bytes.

// ── Scalar reference implementations ──────────────────────────
// Used directly on non-jit builds and as the equivalence oracle
// in tests.

/// The scalar dot product: the reference the SIMD kernel is checked against.
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The scalar squared L2 distance: the reference the SIMD kernel is checked against.
pub fn l2sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Panics with both lengths named when two operands differ in length.
pub fn check_lens(name: &str, a: usize, b: usize) {
    if a != b {
        panic!("{name}: operand lengths differ ({a} vs {b})");
    }
}

// ── Kernel dispatch ────────────────────────────────────────────
//
// One body per operation, shared by the node (which allocates its
// result) and the native helper (which writes into the step's own
// scratch entry, compiled_handles.md §6), so both produce the same
// bytes: the same SIMD kernel or the same scalar loop, in the same
// order.

/// The dot product, through the SIMD kernel where the host has one.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "jit")]
    if let Some(k) = crate::compile::jit::simd::kernels() {
        // SAFETY: both slices live for the call; len is the
        // (equal) element count.
        return unsafe { (k.dot_f32)(a.as_ptr(), b.as_ptr(), a.len() as u64) };
    }
    dot_scalar(a, b)
}

/// The squared L2 distance, through the SIMD kernel where the host has one.
pub fn l2sq_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "jit")]
    if let Some(k) = crate::compile::jit::simd::kernels() {
        return unsafe { (k.l2sq_f32)(a.as_ptr(), b.as_ptr(), a.len() as u64) };
    }
    l2sq_scalar(a, b)
}

/// `a + b` element-wise into `out`, which is cleared first.
pub fn add_f32_into(a: &[f32], b: &[f32], out: &mut Vec<f32>) {
    out.clear();
    out.resize(a.len(), 0.0);
    #[cfg(feature = "jit")]
    if let Some(k) = crate::compile::jit::simd::kernels() {
        // SAFETY: out holds a.len() elements.
        unsafe { (k.add_f32)(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len() as u64) };
        return;
    }
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// `a * k` element-wise into `out`, which is cleared first.
pub fn scale_f32_into(a: &[f32], k_val: f32, out: &mut Vec<f32>) {
    out.clear();
    out.resize(a.len(), 0.0);
    #[cfg(feature = "jit")]
    if let Some(k) = crate::compile::jit::simd::kernels() {
        unsafe { (k.scale_f32)(a.as_ptr(), k_val, out.as_mut_ptr(), a.len() as u64) };
        return;
    }
    for i in 0..a.len() {
        out[i] = a[i] * k_val;
    }
}

/// `a + b` element-wise, as a new vector.
pub fn add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    add_f32_into(a, b, &mut out);
    out
}

/// `a * k` element-wise, as a new vector.
pub fn scale_f32(a: &[f32], k_val: f32) -> Vec<f32> {
    let mut out = Vec::new();
    scale_f32_into(a, k_val, &mut out);
    out
}

/// `a` scaled to unit L2 magnitude into `out`; `a` itself when its
/// magnitude is zero.
pub fn norm_f32_into(a: &[f32], out: &mut Vec<f32>) {
    let mag = (dot_f32(a, a) as f64).sqrt();
    if mag == 0.0 {
        out.clear();
        out.extend_from_slice(a);
    } else {
        scale_f32_into(a, (1.0 / mag) as f32, out);
    }
}

/// The cosine similarity of `vec_cosine`.
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let dot = dot_f32(a, b) as f64;
    let na = (dot_f32(a, a) as f64).sqrt();
    let nb = (dot_f32(b, b) as f64).sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// The estimate of `lid_mle`.
pub fn lid_mle_of(distances: &[f32], k: f64) -> f64 {
    let k = (k as usize).min(distances.len());
    if k < 2 {
        return 0.0;
    }
    let r_k = distances[k - 1] as f64;
    if r_k <= 0.0 {
        return 0.0;
    }
    let ln_rk = r_k.ln();
    let mut logsum = 0.0_f64;
    let mut terms = 0u32;
    for &r_j in &distances[..k - 1] {
        let r_j = r_j as f64;
        if r_j > 0.0 {
            logsum += ln_rk - r_j.ln(); // ln(r_k / r_j) ≥ 0 since r_j ≤ r_k
            terms += 1;
        }
    }
    if terms == 0 || logsum <= 0.0 {
        0.0
    } else {
        terms as f64 / logsum
    }
}

/// The vector of `hash_vec` into `out`, which is cleared first.
pub fn hash_vec_into(seed: u64, dim: u64, out: &mut Vec<f32>) {
    let dim = dim as usize;
    out.clear();
    out.reserve(dim);
    for i in 0..dim {
        let h = crate::numeric::hash::splitmix64_u64(
            seed.wrapping_add((i as u64).wrapping_mul(0x9e3779b97f4a7c15)),
        );
        out.push((h as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32);
    }
}

/// The vector of `xxhash3_vec` into `out`, which is cleared first.
pub fn xxhash3_vec_into(seed: u64, dim: u64, out: &mut Vec<f32>) {
    let dim = dim as usize;
    out.clear();
    out.reserve(dim);
    for i in 0..dim {
        let mut key = [0u8; 16];
        key[..8].copy_from_slice(&seed.to_le_bytes());
        key[8..].copy_from_slice(&(i as u64).to_le_bytes());
        let h = xxhash_rust::xxh3::xxh3_64(&key);
        out.push((h as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32);
    }
}
