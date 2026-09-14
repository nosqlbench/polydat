// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Round-number selector nodes.
//!
//! General-purpose "snap this magnitude to a nice round number" math on
//! f64 values. Four scale families — powers of ten (`base10`), multiples
//! of the base-ten magnitude (`decade`), Fibonacci numbers (`fibonacci`),
//! and powers of two (`binomial`) — each in `floor_` / `ceiling_` /
//! `closest_` variants, plus a general arbitrary-`interval` rounder.
//!
//! These are pure numeric utilities (no adapter, no prefix), used to pick
//! human-friendly axis ticks, bucket boundaries, and magnitude labels.
//!
//! Edge handling: every family selector returns `0.0` for `x <= 0` and for
//! non-finite `x` — no panics, no NaN/inf leaking. Fractional `x` in `(0,1)`
//! yields fractional powers for `base10`/`binomial` (e.g. `floor_base10(0.5)
//! = 0.1`), which is correct and preserved. The interval rounders return
//! `x` unchanged when `interval` is non-positive or non-finite (identity —
//! never divide by zero).

pub use polydat::numeric::round_numbers::{
    ceiling_fibonacci_val, floor_fibonacci_val, floor_pow2, floor_pow10, pick_closest,
    positive_finite,
};

// ---------------------------------------------------------------------------
// base10 — powers of ten (10^n).
// ---------------------------------------------------------------------------

/// Largest power of ten `<= x`: `10^floor(log10(x))`. `x <= 0` → `0.0`.
#[polydat::polydat_node(category = Math)]
pub(crate) fn floor_base10(x: f64) -> f64 {
    if !positive_finite(x) {
        return 0.0;
    }
    floor_pow10(x)
}

/// Smallest power of ten `>= x`: `10^ceil(log10(x))`. `x <= 0` → `0.0`.
#[polydat::polydat_node(category = Math)]
pub(crate) fn ceiling_base10(x: f64) -> f64 {
    if !positive_finite(x) {
        return 0.0;
    }
    let lo = floor_pow10(x);
    if lo == x { lo } else { lo * 10.0 }
}

/// Power of ten nearest to `x` by absolute distance (ties → floor).
/// `x <= 0` → `0.0`.
#[polydat::polydat_node(category = Math)]
pub(crate) fn closest_base10(x: f64) -> f64 {
    if !positive_finite(x) {
        return 0.0;
    }
    let lo = floor_pow10(x);
    let hi = if lo == x { lo } else { lo * 10.0 };
    pick_closest(x, lo, hi)
}

// ---------------------------------------------------------------------------
// decade — multiples of the base-ten magnitude `base = 10^floor(log10(x))`.
// ---------------------------------------------------------------------------

/// Round `x` down to a multiple of its base-ten magnitude:
/// `floor(x/base)*base`. `x <= 0` → `0.0`.
#[polydat::polydat_node(category = Math)]
pub(crate) fn floor_decade(x: f64) -> f64 {
    if !positive_finite(x) {
        return 0.0;
    }
    let base = floor_pow10(x);
    (x / base).floor() * base
}

/// Round `x` up to a multiple of its base-ten magnitude:
/// `ceil(x/base)*base`. `x <= 0` → `0.0`.
#[polydat::polydat_node(category = Math)]
pub(crate) fn ceiling_decade(x: f64) -> f64 {
    if !positive_finite(x) {
        return 0.0;
    }
    let base = floor_pow10(x);
    (x / base).ceil() * base
}

/// Round `x` to the nearest multiple of its base-ten magnitude:
/// `round(x/base)*base`. `x <= 0` → `0.0`.
#[polydat::polydat_node(category = Math)]
pub(crate) fn closest_decade(x: f64) -> f64 {
    if !positive_finite(x) {
        return 0.0;
    }
    let base = floor_pow10(x);
    (x / base).round() * base
}

// ---------------------------------------------------------------------------
// fibonacci — Fibonacci numbers 1, 2, 3, 5, 8, 13, … (start 1, 2).
// ---------------------------------------------------------------------------

/// Largest Fibonacci number `<= x`. `x < 1` (incl. `x <= 0`) → `0.0`.
#[polydat::polydat_node(category = Math)]
pub(crate) fn floor_fibonacci(x: f64) -> f64 {
    if !positive_finite(x) {
        return 0.0;
    }
    floor_fibonacci_val(x)
}

/// Smallest Fibonacci number `>= x`. `x <= 0` → `0.0`.
#[polydat::polydat_node(category = Math)]
pub(crate) fn ceiling_fibonacci(x: f64) -> f64 {
    if !positive_finite(x) {
        return 0.0;
    }
    ceiling_fibonacci_val(x)
}

/// Fibonacci number nearest to `x` by absolute distance (ties → floor).
/// `x <= 0` → `0.0`.
#[polydat::polydat_node(category = Math)]
pub(crate) fn closest_fibonacci(x: f64) -> f64 {
    if !positive_finite(x) {
        return 0.0;
    }
    pick_closest(x, floor_fibonacci_val(x), ceiling_fibonacci_val(x))
}

// ---------------------------------------------------------------------------
// binomial — powers of two (2^n).
// ---------------------------------------------------------------------------

/// Largest power of two `<= x`: `2^floor(log2(x))`. `x <= 0` → `0.0`.
#[polydat::polydat_node(category = Math)]
pub(crate) fn floor_binomial(x: f64) -> f64 {
    if !positive_finite(x) {
        return 0.0;
    }
    floor_pow2(x)
}

/// Smallest power of two `>= x`: `2^ceil(log2(x))`. `x <= 0` → `0.0`.
#[polydat::polydat_node(category = Math)]
pub(crate) fn ceiling_binomial(x: f64) -> f64 {
    if !positive_finite(x) {
        return 0.0;
    }
    let lo = floor_pow2(x);
    if lo == x { lo } else { lo * 2.0 }
}

/// Power of two nearest to `x` by absolute distance (ties → floor).
/// `x <= 0` → `0.0`.
#[polydat::polydat_node(category = Math)]
pub(crate) fn closest_binomial(x: f64) -> f64 {
    if !positive_finite(x) {
        return 0.0;
    }
    let lo = floor_pow2(x);
    let hi = if lo == x { lo } else { lo * 2.0 };
    pick_closest(x, lo, hi)
}

// ---------------------------------------------------------------------------
// General arbitrary-interval rounders.
// ---------------------------------------------------------------------------

/// Round `x` down to a multiple of `interval`: `floor(x/interval)*interval`.
/// `interval <= 0` or non-finite → returns `x` unchanged (identity).
#[polydat::polydat_node(category = Math)]
pub(crate) fn round_floor(x: f64, interval: f64) -> f64 {
    if !(interval.is_finite() && interval > 0.0) {
        return x;
    }
    (x / interval).floor() * interval
}

/// Round `x` up to a multiple of `interval`: `ceil(x/interval)*interval`.
/// `interval <= 0` or non-finite → returns `x` unchanged (identity).
#[polydat::polydat_node(category = Math)]
pub(crate) fn round_ceiling(x: f64, interval: f64) -> f64 {
    if !(interval.is_finite() && interval > 0.0) {
        return x;
    }
    (x / interval).ceil() * interval
}

/// Round `x` to the nearest multiple of `interval`: `round(x/interval)*interval`.
/// `interval <= 0` or non-finite → returns `x` unchanged (identity).
#[polydat::polydat_node(category = Math)]
pub(crate) fn round_nearest(x: f64, interval: f64) -> f64 {
    if !(interval.is_finite() && interval > 0.0) {
        return x;
    }
    (x / interval).round() * interval
}

#[cfg(test)]
mod tests {
    use super::*;
    use polydat::ast::{PolydatNode, Value};

    fn run1(node: &dyn PolydatNode, x: f64) -> f64 {
        let mut out = [Value::None];
        node.eval(&[Value::F64(x)], &mut out);
        out[0].as_f64()
    }

    fn run2(node: &dyn PolydatNode, x: f64, interval: f64) -> f64 {
        let mut out = [Value::None];
        node.eval(&[Value::F64(x), Value::F64(interval)], &mut out);
        out[0].as_f64()
    }

    // ── base10 ────────────────────────────────────────────
    #[test]
    fn base10_vectors() {
        assert_eq!(run1(&FloorBase10::new(), 1732.234), 1000.0);
        assert_eq!(run1(&CeilingBase10::new(), 1732.0), 10000.0);
        assert_eq!(run1(&ClosestBase10::new(), 1732.0), 1000.0);
        assert_eq!(run1(&ClosestBase10::new(), 6000.0), 10000.0);
    }

    #[test]
    fn base10_exact_power_is_stable() {
        // On an exact power of ten, floor == ceiling == x.
        assert_eq!(run1(&FloorBase10::new(), 1000.0), 1000.0);
        assert_eq!(run1(&CeilingBase10::new(), 1000.0), 1000.0);
    }

    #[test]
    fn base10_fractional_below_one() {
        // (0,1): powers of ten stay fractional — keep it.
        assert!((run1(&FloorBase10::new(), 0.5) - 0.1).abs() < 1e-12);
    }

    // ── decade ────────────────────────────────────────────
    #[test]
    fn decade_vectors() {
        assert_eq!(run1(&FloorDecade::new(), 2734.0), 2000.0);
        assert_eq!(run1(&CeilingDecade::new(), 2734.0), 3000.0);
        assert_eq!(run1(&ClosestDecade::new(), 2734.0), 3000.0);
        assert_eq!(run1(&FloorDecade::new(), 1732.0), 1000.0);
        assert_eq!(run1(&ClosestDecade::new(), 1732.0), 2000.0);
        assert_eq!(run1(&CeilingDecade::new(), 1732.0), 2000.0);
    }

    // ── fibonacci ─────────────────────────────────────────
    #[test]
    fn fibonacci_vectors() {
        assert_eq!(run1(&FloorFibonacci::new(), 1732.0), 1597.0);
        assert_eq!(run1(&CeilingFibonacci::new(), 1732.0), 2584.0);
        assert_eq!(run1(&ClosestFibonacci::new(), 1732.0), 1597.0);
    }

    #[test]
    fn fibonacci_floor_below_one_is_zero() {
        assert_eq!(run1(&FloorFibonacci::new(), 0.5), 0.0);
    }

    #[test]
    fn fibonacci_exact_member_is_stable() {
        assert_eq!(run1(&FloorFibonacci::new(), 1597.0), 1597.0);
        assert_eq!(run1(&CeilingFibonacci::new(), 1597.0), 1597.0);
    }

    // ── binomial ──────────────────────────────────────────
    #[test]
    fn binomial_vectors() {
        assert_eq!(run1(&FloorBinomial::new(), 1732.0), 1024.0);
        assert_eq!(run1(&CeilingBinomial::new(), 1732.0), 2048.0);
        assert_eq!(run1(&ClosestBinomial::new(), 1732.0), 2048.0);
    }

    #[test]
    fn binomial_exact_power_is_stable() {
        assert_eq!(run1(&FloorBinomial::new(), 1024.0), 1024.0);
        assert_eq!(run1(&CeilingBinomial::new(), 1024.0), 1024.0);
    }

    // ── non-positive / non-finite edges ───────────────────
    #[test]
    fn non_positive_inputs_are_zero() {
        assert_eq!(run1(&FloorBase10::new(), 0.0), 0.0);
        assert_eq!(run1(&FloorBase10::new(), -5.0), 0.0);
        assert_eq!(run1(&CeilingBinomial::new(), -1.0), 0.0);
        assert_eq!(run1(&ClosestFibonacci::new(), 0.0), 0.0);
        assert_eq!(run1(&ClosestDecade::new(), -1000.0), 0.0);
        assert_eq!(run1(&FloorBase10::new(), f64::INFINITY), 0.0);
        assert_eq!(run1(&FloorBinomial::new(), f64::NAN), 0.0);
    }

    // ── general interval rounders ─────────────────────────
    #[test]
    fn round_interval_vectors() {
        assert_eq!(run2(&RoundFloor::new(), 1732.0, 500.0), 1500.0);
        assert_eq!(run2(&RoundCeiling::new(), 1732.0, 500.0), 2000.0);
        // (1732/500).round() = 3, so nearest multiple of 500 is 1500.
        assert_eq!(run2(&RoundNearest::new(), 1732.0, 500.0), 1500.0);
        assert_eq!(run2(&RoundNearest::new(), 1700.0, 500.0), 1500.0);
        // interval <= 0 → identity.
        assert_eq!(run2(&RoundFloor::new(), 1732.0, 0.0), 1732.0);
    }

    #[test]
    fn round_interval_identity_on_bad_interval() {
        assert_eq!(run2(&RoundNearest::new(), 1732.0, -5.0), 1732.0);
        assert_eq!(run2(&RoundCeiling::new(), 1732.0, f64::INFINITY), 1732.0);
        assert_eq!(run2(&RoundFloor::new(), 1732.0, f64::NAN), 1732.0);
    }

    // ── registry discovery ────────────────────────────────
    #[test]
    fn all_fifteen_registered_under_math() {
        for name in [
            "floor_base10",
            "ceiling_base10",
            "closest_base10",
            "floor_decade",
            "ceiling_decade",
            "closest_decade",
            "floor_fibonacci",
            "ceiling_fibonacci",
            "closest_fibonacci",
            "floor_binomial",
            "ceiling_binomial",
            "closest_binomial",
            "round_floor",
            "round_ceiling",
            "round_nearest",
        ] {
            let sig = polydat::dsl::registry::lookup(name)
                .unwrap_or_else(|| panic!("node '{name}' not registered"));
            assert_eq!(
                sig.category,
                polydat::dsl::registry::FuncCategory::Math,
                "node '{name}' registered under wrong category",
            );
        }
    }
}
