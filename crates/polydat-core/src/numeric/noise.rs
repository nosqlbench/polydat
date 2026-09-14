// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Coherent noise: the permutation table and the Perlin, simplex, and
//! fractal Brownian motion algorithms, the bodies of the noise nodes
//! and of their native lowerings.

/// A permutation table for noise functions. Built from a seed at init
/// time, immutable thereafter. The table is doubled (512 entries) to
/// avoid modular indexing.
pub struct PermTable {
    /// The doubled table: 256 entries repeated, so an index needs no modulus.
    pub perm: [u8; 512],
}

impl PermTable {
    /// The table a seed determines: a Fisher-Yates shuffle driven by xxh3.
    pub fn new(seed: u64) -> Self {
        use xxhash_rust::xxh3::xxh3_64;
        let mut p: Vec<u8> = (0..=255).collect();
        // Fisher-Yates shuffle seeded by hash chain
        let mut s = seed;
        for i in (1..256).rev() {
            s = xxh3_64(&s.to_le_bytes());
            let j = (s as usize) % (i + 1);
            p.swap(i, j);
        }
        let mut perm = [0u8; 512];
        for i in 0..512 {
            perm[i] = p[i & 255];
        }
        Self { perm }
    }

    #[inline]
    /// The entry at `i` modulo 256.
    pub fn hash(&self, i: i32) -> u8 {
        self.perm[(i & 255) as usize]
    }
}

// =================================================================
// Perlin noise primitives
// =================================================================

#[inline]
fn fade(t: f64) -> f64 {
    // 6t^5 - 15t^4 + 10t^3 (improved Perlin smoothstep)
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(t: f64, a: f64, b: f64) -> f64 {
    a + t * (b - a)
}

#[inline]
fn grad1d(hash: u8, x: f64) -> f64 {
    if hash & 1 == 0 { x } else { -x }
}

#[inline]
fn grad2d(hash: u8, x: f64, y: f64) -> f64 {
    match hash & 3 {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        _ => -x - y,
    }
}

/// Evaluate 1D Perlin noise at a given point.
pub fn perlin_1d_algo(perm: &PermTable, x: f64) -> f64 {
    let xi = x.floor() as i32;
    let xf = x - x.floor();
    let u = fade(xf);

    let a = perm.hash(xi);
    let b = perm.hash(xi.wrapping_add(1));

    lerp(u, grad1d(a, xf), grad1d(b, xf - 1.0))
}

/// Evaluate 2D Perlin noise at a given point.
pub fn perlin_2d_algo(perm: &PermTable, x: f64, y: f64) -> f64 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - x.floor();
    let yf = y - y.floor();

    let u = fade(xf);
    let v = fade(yf);

    let aa = perm.hash((perm.hash(xi) as i32).wrapping_add(yi));
    let ab = perm.hash((perm.hash(xi) as i32).wrapping_add(yi).wrapping_add(1));
    let ba = perm.hash((perm.hash(xi.wrapping_add(1)) as i32).wrapping_add(yi));
    let bb = perm.hash(
        (perm.hash(xi.wrapping_add(1)) as i32)
            .wrapping_add(yi)
            .wrapping_add(1),
    );

    lerp(
        v,
        lerp(u, grad2d(aa, xf, yf), grad2d(ba, xf - 1.0, yf)),
        lerp(u, grad2d(ab, xf, yf - 1.0), grad2d(bb, xf - 1.0, yf - 1.0)),
    )
}

// =================================================================
// Simplex noise 2D
// =================================================================

const F2: f64 = 0.3660254037844386; // (sqrt(3) - 1) / 2
const G2: f64 = 0.21132486540518713; // (3 - sqrt(3)) / 6

/// Evaluate 2D simplex noise at a given point.
pub fn simplex_2d_algo(perm: &PermTable, x: f64, y: f64) -> f64 {
    let s = (x + y) * F2;
    let i = (x + s).floor() as i32;
    let j = (y + s).floor() as i32;

    let t = (i + j) as f64 * G2;
    let x0 = x - (i as f64 - t);
    let y0 = y - (j as f64 - t);

    let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };

    let x1 = x0 - i1 as f64 + G2;
    let y1 = y0 - j1 as f64 + G2;
    let x2 = x0 - 1.0 + 2.0 * G2;
    let y2 = y0 - 1.0 + 2.0 * G2;

    let gi0 = perm.hash(i.wrapping_add(perm.hash(j) as i32));
    let gi1 = perm.hash(
        i.wrapping_add(i1)
            .wrapping_add(perm.hash(j.wrapping_add(j1)) as i32),
    );
    let gi2 = perm.hash(
        i.wrapping_add(1)
            .wrapping_add(perm.hash(j.wrapping_add(1)) as i32),
    );

    let mut n0 = 0.0;
    let t0 = 0.5 - x0 * x0 - y0 * y0;
    if t0 > 0.0 {
        let t0 = t0 * t0;
        n0 = t0 * t0 * grad2d(gi0, x0, y0);
    }

    let mut n1 = 0.0;
    let t1 = 0.5 - x1 * x1 - y1 * y1;
    if t1 > 0.0 {
        let t1 = t1 * t1;
        n1 = t1 * t1 * grad2d(gi1, x1, y1);
    }

    let mut n2 = 0.0;
    let t2 = 0.5 - x2 * x2 - y2 * y2;
    if t2 > 0.0 {
        let t2 = t2 * t2;
        n2 = t2 * t2 * grad2d(gi2, x2, y2);
    }

    // Scale to [-1, 1]
    70.0 * (n0 + n1 + n2)
}

impl crate::derive_support::PolydatSetup for PermTable {}

/// FBM lacunarity (frequency multiplier per octave). Standard value.
const FBM_LACUNARITY: f64 = 2.0;
/// FBM persistence (amplitude multiplier per octave). Standard value.
const FBM_PERSISTENCE: f64 = 0.5;

/// 1D fractal Brownian motion: `octaves` layers of Perlin noise, each
/// at twice the frequency and half the amplitude of the last.
pub fn fbm_1d(perm: &PermTable, base_x: f64, frequency: f64, octaves: u32) -> f64 {
    let mut total = 0.0;
    let mut freq = frequency;
    let mut amp = 1.0;
    let mut max_amp = 0.0;

    for _ in 0..octaves {
        total += perlin_1d_algo(perm, base_x * freq) * amp;
        max_amp += amp;
        freq *= FBM_LACUNARITY;
        amp *= FBM_PERSISTENCE;
    }

    // Normalize to [-1, 1]
    total / max_amp
}

/// 2D fractal Brownian motion over Perlin noise, as [`fbm_1d`].
pub fn fbm_2d(perm: &PermTable, base_x: f64, base_y: f64, frequency: f64, octaves: u32) -> f64 {
    let mut total = 0.0;
    let mut freq = frequency;
    let mut amp = 1.0;
    let mut max_amp = 0.0;

    for _ in 0..octaves {
        total += perlin_2d_algo(perm, base_x * freq, base_y * freq) * amp;
        max_amp += amp;
        freq *= FBM_LACUNARITY;
        amp *= FBM_PERSISTENCE;
    }

    total / max_amp
}
