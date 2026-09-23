// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Inverse CDF distribution builders.
//!
//! Each `dist_*` helper builds a [`LutF64`] containing the precomputed
//! inverse CDF for a specific distribution. The DSL surface
//! (`#[polydat_node]` form below) wraps each helper in a node that
//! caches the LUT at construction via `#[poly_const]` and samples on
//! every cycle.
//!
//! This module also provides the [`UnitInterval`] and [`ClampF64`]
//! conversion nodes that are typically composed with LUT sampling in
//! a DAG.
//!
//! # Supported Distributions
//!
//! **Continuous:**
//! Normal, Exponential, Uniform, Pareto, LogNormal, Weibull, Cauchy,
//! Laplace, Beta, Gamma
//!
//! **Discrete:**
//! Zipf, Poisson, Binomial, Geometric

use crate::sampling::lut::LutF64;
use polydat::numeric::special::{inv_regularized_beta, inv_regularized_gamma_p, probit};

/// Default interpolation table resolution.
pub const DEFAULT_RESOLUTION: usize = 1000;

// =================================================================
// Utility: UnitInterval node
// =================================================================

/// Normalize a u64 to a uniform f64 in [0.0, 1.0).
///
/// Signature: `unit_interval(input: u64) -> (f64)`
///
/// The bridge between the integer hash domain and the continuous
/// probability domain. Place after `hash` and before any node that
/// expects a [0,1) input: distribution LUT samplers, `lerp`, or
/// `inv_lerp`. The mapping is `input as f64 / u64::MAX as f64`, so
/// 0 maps to 0.0 and u64::MAX maps to ~1.0.
///
/// JIT level: P3 (`JitOp::UnitInterval`, inline u64→f64 convert and
/// multiply).
#[polydat::polydat_node(category = Conversions)]
fn unit_interval(input: u64) -> f64 {
    input as f64 / u64::MAX as f64
}

// =================================================================
// Utility: ClampF64 node
// =================================================================

/// Clamp an f64 value to [min, max].
///
/// Signature: `clamp_f64(input: f64, min: f64, max: f64) -> (f64)`
///
/// Hard-limits an f64 to the given bounds. Use after distributions
/// with unbounded tails (normal, Cauchy) to enforce domain
/// constraints: `clamp_f64(normal(72.0, 5.0), 0.0, 100.0)` prevents
/// negative scores. Also useful for guarding against non-finite LUT
/// edge values before downstream arithmetic.
///
/// JIT level: P3 (auto-emitted `compiled_u64` + `jit_constants`
/// via the `#[polydat_node]` macro — the two `Const<f64>` fields
/// are folded into the JIT constant pool as bit-encoded f64s).
#[polydat::polydat_node(category = Conversions)]
fn clamp_f64(
    input: f64,
    #[poly_default(f64::MIN)] min: Const<f64>,
    #[poly_default(f64::MAX)] max: Const<f64>,
) -> f64 {
    input.clamp(*min, *max)
}

// =================================================================
// Continuous distribution LUT builders (free fns reused by nodes)
// =================================================================

/// Normal distribution: N(mean, stddev).
pub fn dist_normal_lut(mean: f64, stddev: f64, resolution: usize) -> LutF64 {
    LutF64::from_fn(|p| mean + stddev * probit(p), resolution)
}

/// Exponential distribution: Exp(rate). Support: [0, +∞).
pub fn dist_exponential_lut(rate: f64, resolution: usize) -> LutF64 {
    LutF64::from_fn(|p| -(1.0 - p).ln() / rate, resolution)
}

/// Uniform continuous distribution: U(min, max).
pub fn dist_uniform_lut(min: f64, max: f64, resolution: usize) -> LutF64 {
    LutF64::from_fn(|p| min + p * (max - min), resolution)
}

/// Pareto distribution: Pareto(scale, shape). Support: [scale, +∞).
pub fn dist_pareto_lut(scale: f64, shape: f64, resolution: usize) -> LutF64 {
    LutF64::from_fn(|p| scale / (1.0 - p).powf(1.0 / shape), resolution)
}

/// Log-normal distribution: LogN(mean, stddev). Support: (0, +∞).
pub fn dist_lognormal_lut(mean: f64, stddev: f64, resolution: usize) -> LutF64 {
    LutF64::from_fn(|p| (mean + stddev * probit(p)).exp(), resolution)
}

/// Weibull distribution: Weibull(shape, scale). Support: [0, +∞).
pub fn dist_weibull_lut(shape: f64, scale: f64, resolution: usize) -> LutF64 {
    LutF64::from_fn(|p| scale * (-(1.0 - p).ln()).powf(1.0 / shape), resolution)
}

/// Cauchy distribution: Cauchy(location, scale). Support: (-∞, +∞).
pub fn dist_cauchy_lut(location: f64, scale: f64, resolution: usize) -> LutF64 {
    LutF64::from_fn(
        |p| location + scale * (std::f64::consts::PI * (p - 0.5)).tan(),
        resolution,
    )
}

/// Laplace distribution: Laplace(location, scale). Support: (-∞, +∞).
pub fn dist_laplace_lut(location: f64, scale: f64, resolution: usize) -> LutF64 {
    LutF64::from_fn(
        |p| {
            if p <= 0.5 {
                location + scale * (2.0 * p).ln()
            } else {
                location - scale * (2.0 * (1.0 - p)).ln()
            }
        },
        resolution,
    )
}

/// Beta distribution: Beta(alpha, beta). Support: [0, 1].
pub fn dist_beta_lut(alpha: f64, beta: f64, resolution: usize) -> LutF64 {
    LutF64::from_fn(|p| inv_regularized_beta(p, alpha, beta), resolution)
}

/// Gamma distribution: Gamma(shape, scale). Support: (0, +∞).
pub fn dist_gamma_lut(shape: f64, scale: f64, resolution: usize) -> LutF64 {
    LutF64::from_fn(|p| scale * inv_regularized_gamma_p(p, shape), resolution)
}

// =================================================================
// Discrete distribution LUT builders
// =================================================================

/// Zipf distribution: Zipf(n, exponent). Support: [1, n].
///
/// The LUT maps [0, 1] → float, which is then truncated to an integer
/// by the caller. The CDF is computed from the PMF:
///   P(k) = (1/k^s) / H(n,s)  where H(n,s) = sum_{i=1}^{n} 1/i^s
pub fn dist_zipf_lut(n: u64, exponent: f64, resolution: usize) -> LutF64 {
    // Reserve first: an `n` no machine could hold must fail before the
    // harmonic sum below walks `1..=n`, or it is a hang rather than an
    // error. `saturating_add` keeps the `+ 1` from wrapping to a small
    // size at the top of the range (`buffer_for`).
    let mut cdf = polydat::derive_support::buffer_for(n.saturating_add(1), "dist_zipf");
    // Precompute CDF
    let harmonic: f64 = (1..=n).map(|k| 1.0 / (k as f64).powf(exponent)).sum();
    cdf.push(0.0);
    let mut cumulative = 0.0;
    for k in 1..=n {
        cumulative += (1.0 / (k as f64).powf(exponent)) / harmonic;
        cdf.push(cumulative);
    }

    // Inverse CDF by binary search
    LutF64::from_fn(
        |p| {
            let p = p.clamp(0.0, 1.0);
            match cdf.binary_search_by(|v| v.partial_cmp(&p).unwrap()) {
                Ok(idx) => idx as f64,
                Err(idx) => (idx as f64).max(1.0).min(n as f64),
            }
        },
        resolution,
    )
}

/// Poisson distribution: Poisson(lambda). Support: [0, +∞).
///
/// Precompute CDF up to a reasonable upper bound, then invert.
pub fn dist_poisson_lut(lambda: f64, resolution: usize) -> LutF64 {
    // `as u64` saturates an enormous or infinite λ to `u64::MAX`, which
    // `buffer_for` then refuses; `upper + 2` in `usize` would have
    // wrapped to a tiny buffer ahead of a loop over the huge `upper`.
    let upper = (lambda + 6.0 * lambda.sqrt() + 10.0).ceil() as u64;

    // Precompute CDF via PMF: P(k) = e^(-λ) * λ^k / k!
    let mut cdf = polydat::derive_support::buffer_for(upper.saturating_add(2), "dist_poisson");
    cdf.push(0.0);
    let mut cumulative = 0.0;
    let mut pmf = (-lambda).exp(); // P(0)
    for k in 0..=upper {
        cumulative += pmf;
        cdf.push(cumulative.min(1.0));
        pmf *= lambda / (k + 1) as f64;
    }

    LutF64::from_fn(
        |p| {
            let p = p.clamp(0.0, 1.0);
            match cdf.binary_search_by(|v| v.partial_cmp(&p).unwrap()) {
                Ok(idx) => idx.saturating_sub(1) as f64,
                Err(idx) => idx.saturating_sub(1) as f64,
            }
        },
        resolution,
    )
}

/// Binomial distribution: Binomial(trials, p). Support: [0, trials].
pub fn dist_binomial_lut(trials: u64, prob: f64, resolution: usize) -> LutF64 {
    // Precompute CDF via PMF
    let mut cdf = polydat::derive_support::buffer_for(trials.saturating_add(2), "dist_binomial");
    let n = trials as usize; // fits: `buffer_for` refused anything that does not
    cdf.push(0.0);
    let mut cumulative = 0.0;
    // P(0) = (1-p)^n. `powi` takes an `i32`, and `n as i32` wraps
    // negative past 2^31 trials, which turned the probability of zero
    // successes into its reciprocal. `powi` is kept wherever it was
    // correct so no existing output moves by a bit; past `i32::MAX` the
    // exponent goes through `powf`.
    let mut pmf = match i32::try_from(n) {
        Ok(e) => (1.0 - prob).powi(e),
        Err(_) => (1.0 - prob).powf(n as f64),
    };
    for k in 0..=n {
        cumulative += pmf;
        cdf.push(cumulative.min(1.0));
        if k < n {
            pmf *= prob / (1.0 - prob) * ((n - k) as f64) / ((k + 1) as f64);
        }
    }

    LutF64::from_fn(
        |p| {
            let p = p.clamp(0.0, 1.0);
            match cdf.binary_search_by(|v| v.partial_cmp(&p).unwrap()) {
                Ok(idx) => idx.saturating_sub(1) as f64,
                Err(idx) => idx.saturating_sub(1) as f64,
            }
        },
        resolution,
    )
}

/// Geometric distribution: Geometric(p). Support: [1, +∞).
///
/// P(X=k) = (1-p)^(k-1) * p, inverse CDF: ceil(ln(1-u) / ln(1-p)).
pub fn dist_geometric_lut(prob: f64, resolution: usize) -> LutF64 {
    let ln_q = (1.0 - prob).ln();
    LutF64::from_fn(
        |p| {
            if p <= 0.0 {
                return 1.0;
            }
            if p >= 1.0 {
                return f64::INFINITY;
            }
            ((1.0 - p).ln() / ln_q).ceil().max(1.0)
        },
        resolution,
    )
}

// =================================================================
// Empirical distribution builders (Rust helpers; the DSL surface for
// `dist_empirical` lives in `crate::sampling::lut`).
// =================================================================

/// Build a LUT from raw data points (continuous empirical distribution).
///
/// The data points are sorted and used directly as the inverse CDF.
/// Linear interpolation between observed values.
pub fn dist_empirical_lut(data: &[f64], resolution: usize) -> LutF64 {
    assert!(!data.is_empty(), "data must not be empty");
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    LutF64::from_fn(
        |p| {
            let pos = p * (sorted.len() - 1) as f64;
            let idx = pos as usize;
            let idx = idx.min(sorted.len() - 2);
            let frac = pos - idx as f64;
            sorted[idx] * (1.0 - frac) + sorted[idx + 1] * frac
        },
        resolution,
    )
}

/// Build a LUT from weighted value-frequency pairs.
///
/// Each (value, weight) pair contributes proportionally to the CDF.
pub fn dist_empirical_weighted_lut(values: &[f64], weights: &[f64], resolution: usize) -> LutF64 {
    assert_eq!(values.len(), weights.len());
    assert!(!values.is_empty());

    // Sort by value, accumulate CDF
    let mut pairs: Vec<(f64, f64)> = values
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let total: f64 = pairs.iter().map(|(_, w)| w).sum();
    let mut cdf_points: Vec<(f64, f64)> = Vec::new(); // (cumulative_prob, value)
    let mut cumulative = 0.0;
    for (val, weight) in &pairs {
        cumulative += weight / total;
        cdf_points.push((cumulative, *val));
    }

    // Inverse CDF by binary search
    LutF64::from_fn(
        |p| match cdf_points.binary_search_by(|&(cp, _)| cp.partial_cmp(&p).unwrap()) {
            Ok(idx) => cdf_points[idx].1,
            Err(idx) => {
                if idx >= cdf_points.len() {
                    cdf_points.last().unwrap().1
                } else {
                    cdf_points[idx].1
                }
            }
        },
        resolution,
    )
}

// =================================================================
// DSL nodes: each dist_* / icd_* function caches a LUT via
// `#[poly_const]` and samples on every cycle. The `PolydatSetup`
// impl for `LutF64` lives in the `lut` module alongside the type.
// =================================================================

fn build_normal_lut(mean: f64, stddev: f64) -> LutF64 {
    dist_normal_lut(mean, stddev, DEFAULT_RESOLUTION)
}

/// Sample from a normal distribution `N(mean, stddev)`.
///
/// Signature: `dist_normal(input: f64, mean: f64, stddev: f64) -> f64`
fn dist_normal_jit_constants(node: &DistNormal) -> Vec<u64> {
    vec![node.lut.as_ptr() as u64, node.lut.len() as u64]
}

/// Sample from a standard normal distribution `N(mean, stddev)`.
///
/// `input` is a uniform value in `[0, 1)` (typically from
/// `unit_interval(hash(cycle))`). The LUT is precomputed at
/// construction; the per-cycle cost is one LUT sample.
#[polydat::polydat_node(category = Distributions, jit_constants = dist_normal_jit_constants)]
fn dist_normal(
    input: f64,
    mean: Const<f64>,
    stddev: Const<f64>,
    #[poly_const(build_normal_lut, from = (mean, stddev))] lut: &LutF64,
) -> f64 {
    let _ = mean;
    let _ = stddev;
    lut.sample(input)
}

fn icd_normal_jit_constants(node: &IcdNormal) -> Vec<u64> {
    vec![node.lut.as_ptr() as u64, node.lut.len() as u64]
}

/// Alias of `dist_normal`. Preserved because workload examples
/// and the host's distribution binding both surface
/// `icd_normal` as the public DSL name.
#[polydat::polydat_node(category = Distributions, jit_constants = icd_normal_jit_constants)]
fn icd_normal(
    input: f64,
    mean: Const<f64>,
    stddev: Const<f64>,
    #[poly_const(build_normal_lut, from = (mean, stddev))] lut: &LutF64,
) -> f64 {
    let _ = mean;
    let _ = stddev;
    lut.sample(input)
}

fn build_exponential_lut(rate: f64) -> LutF64 {
    dist_exponential_lut(rate, DEFAULT_RESOLUTION)
}

fn dist_exponential_jit_constants(node: &DistExponential) -> Vec<u64> {
    vec![node.lut.as_ptr() as u64, node.lut.len() as u64]
}

/// Sample from an exponential distribution `Exp(rate)`.
#[polydat::polydat_node(category = Distributions, jit_constants = dist_exponential_jit_constants)]
fn dist_exponential(
    input: f64,
    rate: Const<f64>,
    #[poly_const(build_exponential_lut, from = rate)] lut: &LutF64,
) -> f64 {
    let _ = rate;
    lut.sample(input)
}

fn icd_exponential_jit_constants(node: &IcdExponential) -> Vec<u64> {
    vec![node.lut.as_ptr() as u64, node.lut.len() as u64]
}

/// Alias of `dist_exponential`.
#[polydat::polydat_node(category = Distributions, jit_constants = icd_exponential_jit_constants)]
fn icd_exponential(
    input: f64,
    rate: Const<f64>,
    #[poly_const(build_exponential_lut, from = rate)] lut: &LutF64,
) -> f64 {
    let _ = rate;
    lut.sample(input)
}

fn build_uniform_lut(min: f64, max: f64) -> LutF64 {
    dist_uniform_lut(min, max, DEFAULT_RESOLUTION)
}

fn dist_uniform_jit_constants(node: &DistUniform) -> Vec<u64> {
    vec![node.lut.as_ptr() as u64, node.lut.len() as u64]
}

/// Sample from a continuous uniform distribution `U(min, max)`.
#[polydat::polydat_node(category = Distributions, jit_constants = dist_uniform_jit_constants)]
fn dist_uniform(
    input: f64,
    min: Const<f64>,
    max: Const<f64>,
    #[poly_const(build_uniform_lut, from = (min, max))] lut: &LutF64,
) -> f64 {
    let _ = min;
    let _ = max;
    lut.sample(input)
}

fn build_pareto_lut(scale: f64, shape: f64) -> LutF64 {
    dist_pareto_lut(scale, shape, DEFAULT_RESOLUTION)
}

fn dist_pareto_jit_constants(node: &DistPareto) -> Vec<u64> {
    vec![node.lut.as_ptr() as u64, node.lut.len() as u64]
}

/// Sample from a Pareto distribution `Pareto(scale, shape)`.
#[polydat::polydat_node(category = Distributions, jit_constants = dist_pareto_jit_constants)]
fn dist_pareto(
    input: f64,
    scale: Const<f64>,
    shape: Const<f64>,
    #[poly_const(build_pareto_lut, from = (scale, shape))] lut: &LutF64,
) -> f64 {
    let _ = scale;
    let _ = shape;
    lut.sample(input)
}

fn build_zipf_lut(n: u64, exponent: f64) -> LutF64 {
    dist_zipf_lut(n, exponent, DEFAULT_RESOLUTION)
}

fn dist_zipf_jit_constants(node: &DistZipf) -> Vec<u64> {
    vec![node.lut.as_ptr() as u64, node.lut.len() as u64]
}

/// Sample from a Zipf distribution `Zipf(n, exponent)`.
#[polydat::polydat_node(category = Distributions, jit_constants = dist_zipf_jit_constants)]
fn dist_zipf(
    input: f64,
    n: Const<u64>,
    exponent: Const<f64>,
    #[poly_const(build_zipf_lut, from = (n, exponent))] lut: &LutF64,
) -> f64 {
    let _ = n;
    let _ = exponent;
    lut.sample(input)
}

// ---------------------------------------------------------------------------
// Inventory stub: histribution and dist_empirical register themselves
// via their own modules; `LutSample` (lut.rs) is a programmatic node
// (not DSL-registered). This module no longer hand-rolls a
// signatures() vec — every DSL node here self-registers via
// `#[polydat_node]`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use polydat::ast::{PolydatNode, Value};

    #[test]
    fn unit_interval_range() {
        let node = UnitInterval::new();
        let mut out = [Value::None];
        node.eval(&[Value::U64(0)], &mut out);
        assert_eq!(out[0].as_f64(), 0.0);
        node.eval(&[Value::U64(u64::MAX)], &mut out);
        assert!((0.999..=1.0).contains(&out[0].as_f64()));
    }

    #[test]
    fn normal_symmetry() {
        let lut = dist_normal_lut(0.0, 1.0, 1000);
        assert!(lut.sample(0.5).abs() < 0.01);
        assert!((lut.sample(0.25) + lut.sample(0.75)).abs() < 0.01);
    }

    #[test]
    fn normal_mean_stddev() {
        let lut = dist_normal_lut(100.0, 10.0, 1000);
        assert!((lut.sample(0.5) - 100.0).abs() < 0.5);
    }

    #[test]
    fn exponential_median() {
        let lut = dist_exponential_lut(1.0, 1000);
        assert!((lut.sample(0.5) - 0.693).abs() < 0.01);
    }

    #[test]
    fn exponential_positive() {
        let lut = dist_exponential_lut(1.0, 1000);
        for i in 1..1000 {
            assert!(lut.sample(i as f64 / 1000.0) >= 0.0);
        }
    }

    #[test]
    fn uniform_linear() {
        let lut = dist_uniform_lut(10.0, 20.0, 1000);
        assert!((lut.sample(0.0) - 10.0).abs() < 0.1);
        assert!((lut.sample(0.5) - 15.0).abs() < 0.1);
        assert!((lut.sample(0.999) - 20.0).abs() < 0.1);
    }

    #[test]
    fn pareto_heavy_tail() {
        let lut = dist_pareto_lut(1.0, 1.0, 1000);
        assert!((lut.sample(0.5) - 2.0).abs() < 0.1);
        assert!(lut.sample(0.99) > 50.0);
    }

    #[test]
    fn cauchy_symmetric() {
        let lut = dist_cauchy_lut(0.0, 1.0, 1000);
        assert!(lut.sample(0.5).abs() < 0.1);
        assert!((lut.sample(0.25) + lut.sample(0.75)).abs() < 0.1);
    }

    #[test]
    fn laplace_symmetric() {
        let lut = dist_laplace_lut(5.0, 2.0, 1000);
        assert!((lut.sample(0.5) - 5.0).abs() < 0.1);
    }

    #[test]
    fn beta_bounded_01() {
        let lut = dist_beta_lut(2.0, 5.0, 1000);
        for i in 0..=1000 {
            let v = lut.sample(i as f64 / 1000.0);
            assert!((0.0..=1.0).contains(&v), "beta out of [0,1]: {v}");
        }
    }

    #[test]
    fn beta_symmetric_at_half() {
        // Beta(2, 2) is symmetric around 0.5
        let lut = dist_beta_lut(2.0, 2.0, 1000);
        assert!(
            (lut.sample(0.5) - 0.5).abs() < 0.1,
            "beta(2,2) median={}, expected ~0.5",
            lut.sample(0.5)
        );
    }

    #[test]
    fn gamma_positive() {
        let lut = dist_gamma_lut(2.0, 1.0, 1000);
        for i in 1..1000 {
            assert!(lut.sample(i as f64 / 1000.0) > 0.0);
        }
    }

    #[test]
    fn gamma_mean() {
        // Gamma(shape=3, scale=2) has mean = shape * scale = 6
        let lut = dist_gamma_lut(3.0, 2.0, 1000);
        assert!((lut.sample(0.5) - 5.0).abs() < 1.5); // median ≈ mean for shape>1
    }

    #[test]
    fn weibull_positive() {
        let lut = dist_weibull_lut(2.0, 1.0, 1000);
        for i in 1..1000 {
            assert!(lut.sample(i as f64 / 1000.0) >= 0.0);
        }
    }

    #[test]
    fn zipf_range() {
        let lut = dist_zipf_lut(100, 1.0, 1000);
        for i in 1..1000 {
            let v = lut.sample(i as f64 / 1000.0);
            assert!((1.0..=100.0).contains(&v), "zipf out of [1,100]: {v}");
        }
    }

    #[test]
    fn zipf_skewed() {
        // Low ranks should be much more common
        let lut = dist_zipf_lut(100, 1.0, 1000);
        let low_quantile = lut.sample(0.5);
        assert!(
            low_quantile < 20.0,
            "median of Zipf(100,1) should be low, got {low_quantile}"
        );
    }

    #[test]
    fn poisson_mean() {
        // Poisson(5): mean and median ≈ 5
        let lut = dist_poisson_lut(5.0, 1000);
        let median = lut.sample(0.5);
        assert!(
            (median - 5.0).abs() < 1.0,
            "poisson median={median}, expected ~5"
        );
    }

    #[test]
    fn poisson_nonnegative() {
        let lut = dist_poisson_lut(3.0, 1000);
        for i in 0..=1000 {
            assert!(lut.sample(i as f64 / 1000.0) >= 0.0);
        }
    }

    #[test]
    fn binomial_range() {
        let lut = dist_binomial_lut(20, 0.5, 1000);
        for i in 0..=1000 {
            let v = lut.sample(i as f64 / 1000.0);
            assert!((0.0..=20.0).contains(&v), "binomial out of [0,20]: {v}");
        }
    }

    #[test]
    fn binomial_mean() {
        // Binomial(20, 0.5): mean = 10
        let lut = dist_binomial_lut(20, 0.5, 1000);
        let median = lut.sample(0.5);
        assert!(
            (median - 10.0).abs() < 1.5,
            "binomial median={median}, expected ~10"
        );
    }

    #[test]
    fn geometric_starts_at_one() {
        let lut = dist_geometric_lut(0.5, 1000);
        assert!(lut.sample(0.001) >= 1.0);
    }

    #[test]
    fn geometric_mean() {
        // Geometric(0.5): mean = 1/p = 2
        let lut = dist_geometric_lut(0.5, 1000);
        let median = lut.sample(0.5);
        assert!(
            (median - 1.0).abs() < 1.0,
            "geometric median={median}, expected ~1-2"
        );
    }

    #[test]
    fn dist_normal_node_eval() {
        let node = DistNormal::new(0.0, 1.0);
        let mut out = [Value::None];
        node.eval(&[Value::F64(0.5)], &mut out);
        assert!(out[0].as_f64().abs() < 0.01);
    }

    #[test]
    fn full_pipeline_hash_normalize_sample() {
        use xxhash_rust::xxh3::xxh3_64;

        let lut = dist_normal_lut(72.0, 5.0, 1000);
        let mut values = Vec::new();
        for i in 0..10_000u64 {
            let hashed = xxh3_64(&i.to_le_bytes());
            let u = hashed as f64 / u64::MAX as f64;
            values.push(lut.sample(u));
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let stddev = variance.sqrt();
        assert!((mean - 72.0).abs() < 0.5, "mean={mean}");
        assert!((stddev - 5.0).abs() < 0.5, "stddev={stddev}");
    }
}
