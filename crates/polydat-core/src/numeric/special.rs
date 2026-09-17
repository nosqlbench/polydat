// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Special functions behind the inverse-CDF samplers: the standard
//! normal quantile and CDF, `ln Γ`, and the regularized incomplete
//! beta and gamma functions with their inverses. The node library's
//! distribution tables (`polydat-nodes`, `sampling::icd`) and the
//! comprehension runtime's named-measure sampling
//! (comprehension_forms.md §10.2 R2) compute from the same bodies,
//! so a `normal` node and a `normal` comprehension source agree bit
//! for bit on the same quantile.

/// Rational approximation of the standard normal quantile function.
/// Abramowitz & Stegun 26.2.23: the absolute error is below
/// 4.5e-4 everywhere in (0, 1). `0` and `1` map to the infinities.
pub fn probit(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 { -result } else { result }
}

/// The standard normal CDF `Φ(x)`, through the regularized lower
/// incomplete gamma function: `Φ(x) = ½(1 + sgn(x)·P(½, x²/2))`.
pub fn normal_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::NEG_INFINITY {
        return 0.0;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    let p = regularized_gamma_p(0.5, x * x / 2.0);
    if x < 0.0 {
        0.5 * (1.0 - p)
    } else {
        0.5 * (1.0 + p)
    }
}

/// Lanczos approximation of ln(Gamma(x)) for x > 0.
pub fn ln_gamma(x: f64) -> f64 {
    let g = 7.0;
    let c = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut sum = c[0];
    for (i, &coeff) in c[1..].iter().enumerate() {
        sum += coeff / (x + i as f64 + 1.0);
    }

    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (t.ln() * (x + 0.5)) - t + sum.ln()
}

/// Regularized incomplete beta function I_x(a, b) via series expansion.
pub fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry relation for better convergence when x > 0.5
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_beta(1.0 - x, b, a);
    }

    let ln_prefix = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let prefix = ln_prefix.exp();

    // Series expansion: I_x(a,b) = (x^a * (1-x)^b) / (a * B(a,b)) * sum
    let mut sum = 0.0;
    let mut term = 1.0;
    for n in 0..300 {
        sum += term;
        term *= x * (a + b + n as f64) / (a + 1.0 + n as f64);
        if term.abs() < 1e-15 * sum.abs() {
            break;
        }
    }

    (prefix * sum / a).clamp(0.0, 1.0)
}

/// Inverse regularized beta via bisection.
pub fn inv_regularized_beta(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }

    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if regularized_beta(mid, a, b) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Regularized lower incomplete gamma function P(a, x) via series.
pub fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x > a + 50.0 {
        return 1.0;
    } // far in the tail

    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..300 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < 1e-14 * sum.abs() {
            break;
        }
    }
    (a * x.ln() - x - ln_gamma(a)).exp() * sum
}

/// Inverse regularized gamma P via bisection.
pub fn inv_regularized_gamma_p(p: f64, a: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Bracket: upper bound heuristic
    let mut hi = a.max(1.0);
    while regularized_gamma_p(a, hi) < p {
        hi *= 2.0;
    }
    let mut lo = 0.0_f64;

    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if regularized_gamma_p(a, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probit_is_odd_about_one_half_and_hits_the_infinities() {
        assert!(probit(0.5).abs() < 1e-3);
        assert!((probit(0.25) + probit(0.75)).abs() < 1e-3);
        assert!((probit(0.975) - 1.96).abs() < 1e-3);
        assert_eq!(probit(0.0), f64::NEG_INFINITY);
        assert_eq!(probit(1.0), f64::INFINITY);
    }

    #[test]
    fn normal_cdf_matches_the_table() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-9);
        assert!((normal_cdf(1.0) - 0.841_344_746).abs() < 1e-6);
        assert!((normal_cdf(-1.96) - 0.024_997_895).abs() < 1e-6);
        assert_eq!(normal_cdf(f64::NEG_INFINITY), 0.0);
        assert_eq!(normal_cdf(f64::INFINITY), 1.0);
    }

    #[test]
    fn normal_cdf_and_probit_are_inverses_to_the_probit_error() {
        for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
            let back = normal_cdf(probit(p));
            assert!((back - p).abs() < 5e-4, "p={p} back={back}");
        }
    }

    #[test]
    fn ln_gamma_matches_factorials() {
        for n in 1..10u32 {
            let fact: f64 = (1..n).map(f64::from).product();
            assert!((ln_gamma(f64::from(n)) - fact.ln()).abs() < 1e-9, "n={n}");
        }
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-9);
    }

    #[test]
    fn regularized_beta_and_its_inverse_round_trip() {
        assert!((regularized_beta(0.5, 2.0, 2.0) - 0.5).abs() < 1e-9);
        for p in [0.05, 0.25, 0.5, 0.75, 0.95] {
            let x = inv_regularized_beta(p, 2.0, 5.0);
            assert!((regularized_beta(x, 2.0, 5.0) - p).abs() < 1e-9, "p={p}");
        }
    }

    #[test]
    fn regularized_gamma_and_its_inverse_round_trip() {
        // P(1, x) = 1 - e^-x.
        assert!((regularized_gamma_p(1.0, 1.0) - (1.0 - (-1.0f64).exp())).abs() < 1e-9);
        for p in [0.05, 0.25, 0.5, 0.75, 0.95] {
            let x = inv_regularized_gamma_p(p, 2.5);
            assert!((regularized_gamma_p(2.5, x) - p).abs() < 1e-9, "p={p}");
        }
    }
}
