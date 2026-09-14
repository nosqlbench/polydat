// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The round-number families: the powers, Fibonacci steps, and
//! nearest-pick the `floor_*`, `ceiling_*`, and `closest_*` nodes and
//! their native lowerings compute with.

// ---------------------------------------------------------------------------
// Private helpers (not nodes — plain module fns callable from node bodies).
// ---------------------------------------------------------------------------

/// True only for a strictly-positive, finite `x`. All family selectors gate
/// on this and return `0.0` otherwise, so `x <= 0`, `NaN`, and `±inf` all
/// fold to the zero magnitude without special-casing each node.
#[inline]
pub fn positive_finite(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

/// Largest power of ten `10^n <= x`. Assumes `positive_finite(x)`.
///
/// The exponent is taken from `log10(x).floor()` but reconstructed with
/// `powi` (exact for `|result| < 2^53`) and corrected by at most one step,
/// so a one-ulp `log10` error at an exact power-of-ten boundary can't leak.
///
/// `pub` so non-node callers can reuse the exact `floor_base10` formula
/// without re-deriving it (DRY). `floor_base10` the node is just this
/// function behind the `positive_finite` guard, so a caller that already
/// guarantees a strictly-positive, finite `x` (e.g. the CQL batch-stride
/// planner, which floors `budget / row_size`) can call this directly.
#[inline]
pub fn floor_pow10(x: f64) -> f64 {
    let mut e = x.log10().floor() as i32;
    let mut p = 10f64.powi(e);
    if p > x {
        e -= 1;
        p = 10f64.powi(e);
    } else if p * 10.0 <= x {
        e += 1;
        p = 10f64.powi(e);
    }
    p
}

/// Largest power of two `2^n <= x`. Assumes `positive_finite(x)`.
///
/// Same `powi` + one-step-correction scheme as [`floor_pow10`].
#[inline]
pub fn floor_pow2(x: f64) -> f64 {
    let mut e = x.log2().floor() as i32;
    let mut p = 2f64.powi(e);
    if p > x {
        e -= 1;
        p = 2f64.powi(e);
    } else if p * 2.0 <= x {
        e += 1;
        p = 2f64.powi(e);
    }
    p
}

/// Pick whichever of `lo` / `hi` is nearer to `x` by absolute distance.
/// Ties resolve to `lo` (the floor), per the `closest_*` contract.
#[inline]
pub fn pick_closest(x: f64, lo: f64, hi: f64) -> f64 {
    if (hi - x).abs() < (x - lo).abs() {
        hi
    } else {
        lo
    }
}

/// Largest Fibonacci number (`1, 2, 3, 5, 8, …`) that is `<= x`, or `0.0`
/// when `x < 1` (nothing in the sequence is that small). Assumes finite `x`.
pub fn floor_fibonacci_val(x: f64) -> f64 {
    if x < 1.0 {
        return 0.0;
    }
    let (mut a, mut b) = (1.0f64, 2.0f64);
    while b <= x {
        let next = a + b;
        a = b;
        b = next;
        if !b.is_finite() {
            return a;
        }
    }
    a
}

/// Smallest Fibonacci number (`1, 2, 3, 5, 8, …`) that is `>= x`; `1.0` for
/// `x <= 1`. Assumes `positive_finite(x)`.
pub fn ceiling_fibonacci_val(x: f64) -> f64 {
    let (mut a, mut b) = (1.0f64, 2.0f64);
    if x <= a {
        return a;
    }
    loop {
        if b >= x {
            return b;
        }
        let next = a + b;
        a = b;
        b = next;
        if !b.is_finite() {
            return b;
        }
    }
}
