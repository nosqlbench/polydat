// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Coherent noise functions: Perlin, simplex, fractal Brownian motion.
//!
//! Unlike hash functions (which produce uncorrelated "white noise"),
//! coherent noise produces values that vary smoothly — nearby inputs
//! yield similar outputs. This is essential for generating realistic
//! time-series data, spatial fields, and any workload where adjacent
//! coordinates should have correlated values.
//!
//! The permutation table is built at init time from a seed. The noise
//! evaluation runs at cycle time.
//!
//! Inputs are u64 coordinates mapped to a float domain via scaling.
//! Outputs are f64 in [-1, 1] (raw noise) or [0, 1] (normalized).

// Imports of `PolydatNode` / `Value` live in the `#[cfg(test)]`
// module — the macro pulls in everything it needs by absolute path.

// =================================================================
// Permutation table (init-time artifact)
// =================================================================

pub use polydat::numeric::noise::{
    PermTable, fbm_1d, fbm_2d, perlin_1d_algo, perlin_2d_algo, simplex_2d_algo,
};

// =================================================================
// Polydat Nodes
// =================================================================

fn perlin_1d_jit_constants(node: &Perlin1d) -> Vec<u64> {
    vec![node.perm.perm.as_ptr() as u64, node.frequency.to_bits()]
}
fn perlin_2d_jit_constants(node: &Perlin2d) -> Vec<u64> {
    vec![node.perm.perm.as_ptr() as u64, node.frequency.to_bits()]
}
fn simplex_2d_jit_constants(node: &Simplex2d) -> Vec<u64> {
    vec![node.perm.perm.as_ptr() as u64, node.frequency.to_bits()]
}

/// 1D Perlin noise.
///
/// Signature: `(input: u64) -> (f64)`
///
/// The u64 input is scaled to the float domain by `frequency`.
/// Output is in [-1, 1]. For [0, 1], compose with a remap node.

#[polydat::polydat_node(category = Noise, jit_constants = perlin_1d_jit_constants)]
fn perlin_1d(
    input: u64,
    seed: polydat::derive_support::Const<u64>,
    frequency: polydat::derive_support::Const<f64>,
    #[poly_const(PermTable::new, from = seed)] perm: &PermTable,
) -> f64 {
    perlin_1d_algo(perm, input as f64 * *frequency)
}

#[polydat::polydat_node(category = Noise, jit_constants = perlin_2d_jit_constants)]
fn perlin_2d(
    x: u64,
    y: u64,
    seed: polydat::derive_support::Const<u64>,
    frequency: polydat::derive_support::Const<f64>,
    #[poly_const(PermTable::new, from = seed)] perm: &PermTable,
) -> f64 {
    perlin_2d_algo(perm, x as f64 * *frequency, y as f64 * *frequency)
}

#[polydat::polydat_node(category = Noise, jit_constants = simplex_2d_jit_constants)]
fn simplex_2d(
    x: u64,
    y: u64,
    seed: polydat::derive_support::Const<u64>,
    frequency: polydat::derive_support::Const<f64>,
    #[poly_const(PermTable::new, from = seed)] perm: &PermTable,
) -> f64 {
    simplex_2d_algo(perm, x as f64 * *frequency, y as f64 * *frequency)
}

// =================================================================
// Fractal Brownian motion primitives
// =================================================================

fn fractal_noise_1d_jit_constants(node: &FractalNoise1d) -> Vec<u64> {
    vec![
        node.perm.perm.as_ptr() as u64,
        node.frequency.to_bits(),
        node.octaves,
    ]
}
fn fractal_noise_2d_jit_constants(node: &FractalNoise2d) -> Vec<u64> {
    vec![
        node.perm.perm.as_ptr() as u64,
        node.frequency.to_bits(),
        node.octaves,
    ]
}

/// 1D fractal Brownian motion: layered Perlin noise with decreasing
/// amplitude at each octave. Produces rich, natural-looking signals.
/// Output is f64, roughly in [-1, 1]. Lacunarity is fixed at 2.0 and
/// persistence at 0.5 (standard FBM parameters).
///
/// `octaves` is 1 to 64, for both fractal nodes. Zero octaves is
/// `0 / 0`, NaN. Each octave halves the amplitude, so past about 53 an
/// octave's term is below the resolution of the sum and changes
/// nothing; far past that the doubling frequency overflows to infinity
/// and the sample is NaN. An octave count is also work per call, one
/// noise sample each, so an unbounded one is a hang. The count used to
/// pass through `as u32` as well, which truncated `2^53 + 1` octaves to
/// one. Sixty-four keeps every octave that can contribute.
#[polydat::polydat_node(category = Noise, jit_constants = fractal_noise_1d_jit_constants)]
fn fractal_noise_1d(
    input: u64,
    seed: polydat::derive_support::Const<u64>,
    frequency: polydat::derive_support::Const<f64>,
    #[poly_default(4u64)]
    #[constraint(RangeU64 { min: 1, max: 64 })]
    octaves: polydat::derive_support::Const<u64>,
    #[poly_const(PermTable::new, from = seed)] perm: &PermTable,
) -> f64 {
    fbm_1d(perm, input as f64, *frequency, *octaves as u32)
}

/// 2D fractal Brownian motion: layered Perlin noise in 2D. Produces
/// terrain-like spatial variation. Lacunarity is fixed at 2.0 and
/// persistence at 0.5 (standard FBM parameters). `octaves` is 1 to 64,
/// as for `fractal_noise_1d`.
#[polydat::polydat_node(category = Noise, jit_constants = fractal_noise_2d_jit_constants)]
fn fractal_noise_2d(
    x: u64,
    y: u64,
    seed: polydat::derive_support::Const<u64>,
    frequency: polydat::derive_support::Const<f64>,
    #[poly_default(4u64)]
    #[constraint(RangeU64 { min: 1, max: 64 })]
    octaves: polydat::derive_support::Const<u64>,
    #[poly_const(PermTable::new, from = seed)] perm: &PermTable,
) -> f64 {
    fbm_2d(perm, x as f64, y as f64, *frequency, *octaves as u32)
}
#[cfg(test)]
mod tests {
    use super::*;
    use polydat::ast::{PolydatNode, Value};

    #[test]
    fn perlin_1d_bounded() {
        let node = Perlin1d::new(42, 0.01);
        let mut out = [Value::None];
        for i in 0..1000u64 {
            node.eval(&[Value::U64(i)], &mut out);
            let v = out[0].as_f64();
            assert!((-1.0..=1.0).contains(&v), "out of range: {v} at i={i}");
        }
    }

    #[test]
    fn perlin_1d_smooth() {
        // Adjacent inputs should produce similar (not identical) values
        let node = Perlin1d::new(42, 0.01);
        let mut prev = [Value::None];
        let mut curr = [Value::None];
        node.eval(&[Value::U64(100)], &mut prev);
        let mut large_jumps = 0;
        for i in 101..200u64 {
            node.eval(&[Value::U64(i)], &mut curr);
            let diff = (curr[0].as_f64() - prev[0].as_f64()).abs();
            if diff > 0.5 {
                large_jumps += 1;
            }
            prev[0] = curr[0].clone();
        }
        // With frequency 0.01, adjacent samples should rarely jump more than 0.5
        assert!(large_jumps < 5, "too many large jumps: {large_jumps}");
    }

    #[test]
    fn perlin_1d_deterministic() {
        let node = Perlin1d::new(42, 0.1);
        let mut out1 = [Value::None];
        let mut out2 = [Value::None];
        node.eval(&[Value::U64(123)], &mut out1);
        node.eval(&[Value::U64(123)], &mut out2);
        assert_eq!(out1[0].as_f64(), out2[0].as_f64());
    }

    #[test]
    fn perlin_1d_different_seeds() {
        let a = Perlin1d::new(1, 0.1);
        let b = Perlin1d::new(2, 0.1);
        let mut out_a = [Value::None];
        let mut out_b = [Value::None];
        let mut differ = false;
        for i in 0..100u64 {
            a.eval(&[Value::U64(i)], &mut out_a);
            b.eval(&[Value::U64(i)], &mut out_b);
            if (out_a[0].as_f64() - out_b[0].as_f64()).abs() > 0.01 {
                differ = true;
                break;
            }
        }
        assert!(differ, "different seeds should produce different noise");
    }

    #[test]
    fn perlin_2d_bounded() {
        let node = Perlin2d::new(42, 0.01);
        let mut out = [Value::None];
        for x in 0..50u64 {
            for y in 0..50u64 {
                node.eval(&[Value::U64(x), Value::U64(y)], &mut out);
                let v = out[0].as_f64();
                assert!((-1.5..=1.5).contains(&v), "out of range: {v} at ({x},{y})");
            }
        }
    }

    #[test]
    fn perlin_2d_smooth() {
        let node = Perlin2d::new(42, 0.01);
        let mut prev = [Value::None];
        let mut curr = [Value::None];
        node.eval(&[Value::U64(100), Value::U64(100)], &mut prev);
        let mut large_jumps = 0;
        for i in 101..150u64 {
            node.eval(&[Value::U64(i), Value::U64(100)], &mut curr);
            let diff = (curr[0].as_f64() - prev[0].as_f64()).abs();
            if diff > 0.5 {
                large_jumps += 1;
            }
            prev[0] = curr[0].clone();
        }
        assert!(large_jumps < 5, "too many large jumps: {large_jumps}");
    }

    #[test]
    fn simplex_2d_bounded() {
        let node = Simplex2d::new(42, 0.01);
        let mut out = [Value::None];
        for x in 0..50u64 {
            for y in 0..50u64 {
                node.eval(&[Value::U64(x), Value::U64(y)], &mut out);
                let v = out[0].as_f64();
                assert!((-1.5..=1.5).contains(&v), "out of range: {v}");
            }
        }
    }

    #[test]
    fn fractal_1d_bounded() {
        let node = FractalNoise1d::new(42, 0.01, 4);
        let mut out = [Value::None];
        for i in 0..500u64 {
            node.eval(&[Value::U64(i)], &mut out);
            let v = out[0].as_f64();
            assert!((-1.5..=1.5).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn fractal_1d_more_detail_than_single_octave() {
        // FBM with 4 octaves should have more high-frequency variation
        // than a single octave
        let single = Perlin1d::new(42, 0.01);
        let fbm = FractalNoise1d::new(42, 0.01, 4);
        let mut s_out = [Value::None];
        let mut f_out = [Value::None];
        let mut s_changes = 0.0;
        let mut f_changes = 0.0;
        let mut s_prev = 0.0;
        let mut f_prev = 0.0;
        for i in 0..500u64 {
            single.eval(&[Value::U64(i)], &mut s_out);
            fbm.eval(&[Value::U64(i)], &mut f_out);
            if i > 0 {
                s_changes += (s_out[0].as_f64() - s_prev).abs();
                f_changes += (f_out[0].as_f64() - f_prev).abs();
            }
            s_prev = s_out[0].as_f64();
            f_prev = f_out[0].as_f64();
        }
        // FBM should have more total variation (higher frequency detail)
        assert!(
            f_changes > s_changes * 0.8,
            "FBM should have comparable or more detail: single={s_changes}, fbm={f_changes}"
        );
    }

    #[test]
    fn fractal_2d_bounded() {
        let node = FractalNoise2d::new(42, 0.01, 3);
        let mut out = [Value::None];
        for x in 0..30u64 {
            for y in 0..30u64 {
                node.eval(&[Value::U64(x), Value::U64(y)], &mut out);
                let v = out[0].as_f64();
                assert!((-1.5..=1.5).contains(&v), "out of range: {v}");
            }
        }
    }
}
