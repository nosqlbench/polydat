// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A continuous axis's measure as the sampler maps onto it
//! (comprehension_forms.md §10.2 R2): a sampling strategy draws a
//! point in `[0, 1)` per continuous axis, and the axis's measure
//! carries it onto the axis's interval, affinely for `Uniform` and
//! by the inverse CDF for a named measure. A named measure on an
//! interval narrower than its support is the measure restricted to
//! that interval: the unit point is placed between the CDF values
//! of the two ends and then inverted, so the draws keep the
//! measure's shape inside the interval.

use crate::iteration::comprehension::cardinality::{Interval, MeasureName, ProductMeasure};
use crate::numeric::special::{
    inv_regularized_beta, inv_regularized_gamma_p, normal_cdf, probit, regularized_beta,
    regularized_gamma_p,
};

/// The measure of one continuous axis, with its parameters resolved
/// (see [`MeasureName::parameter_names`]).
#[derive(Debug, Clone, PartialEq)]
pub enum AxisMeasure {
    /// Lebesgue measure scaled to the interval.
    Uniform,
    /// A named distribution with its parameters.
    Named {
        /// The distribution.
        name: MeasureName,
        /// Its parameters, in the order of [`MeasureName::parameter_names`].
        params: Vec<f64>,
    },
}

impl AxisMeasure {
    /// The measure of the `axis`-th continuous axis of `measure`: a
    /// `Product` is indexed, `Uniform` and `Named` apply to every
    /// axis. Named measures here carry no parameters, so they take
    /// the standard ones; a `Source::Distribution` supplies its
    /// own through [`AxisMeasure::named`].
    pub fn from_product(measure: &ProductMeasure, axis: usize) -> Result<Self, String> {
        match measure {
            ProductMeasure::Uniform => Ok(AxisMeasure::Uniform),
            ProductMeasure::Named(name) => Self::named(*name, &[]),
            ProductMeasure::Product(children) => match children.get(axis) {
                Some(child) => Self::from_product(child, 0),
                None => Err(format!(
                    "product measure has {} axes; axis {axis} requested",
                    children.len()
                )),
            },
        }
    }

    /// A named measure with `params` resolved against the measure's
    /// parameter table.
    pub fn named(name: MeasureName, params: &[f64]) -> Result<Self, String> {
        Ok(AxisMeasure::Named {
            name,
            params: name.resolve_params(params)?,
        })
    }

    /// The CDF of a named measure at `x`; `None` for `Uniform`,
    /// whose CDF depends on the interval.
    pub fn cdf(&self, x: f64) -> Option<f64> {
        let AxisMeasure::Named { name, params } = self else {
            return None;
        };
        if x.is_nan() {
            return Some(f64::NAN);
        }
        Some(match name {
            MeasureName::Normal => normal_cdf((x - params[0]) / params[1]),
            MeasureName::Exponential => {
                if x <= 0.0 {
                    0.0
                } else {
                    1.0 - (-params[0] * x).exp()
                }
            }
            MeasureName::Pareto => {
                let (scale, shape) = (params[0], params[1]);
                if x <= scale {
                    0.0
                } else {
                    1.0 - (scale / x).powf(shape)
                }
            }
            MeasureName::Beta => regularized_beta(x, params[0], params[1]),
            MeasureName::LogNormal => {
                if x <= 0.0 {
                    0.0
                } else {
                    normal_cdf((x.ln() - params[0]) / params[1])
                }
            }
            MeasureName::Gamma => {
                if x <= 0.0 {
                    0.0
                } else {
                    regularized_gamma_p(params[0], x / params[1])
                }
            }
            MeasureName::Uniform01 => x.clamp(0.0, 1.0),
        })
    }

    /// The quantile of a named measure at `p ∈ [0, 1]`; `None` for
    /// `Uniform`.
    pub fn quantile(&self, p: f64) -> Option<f64> {
        let AxisMeasure::Named { name, params } = self else {
            return None;
        };
        Some(match name {
            MeasureName::Normal => params[0] + params[1] * probit(p),
            MeasureName::Exponential => -(1.0 - p).ln() / params[0],
            MeasureName::Pareto => params[0] / (1.0 - p).powf(1.0 / params[1]),
            MeasureName::Beta => inv_regularized_beta(p, params[0], params[1]),
            MeasureName::LogNormal => (params[0] + params[1] * probit(p)).exp(),
            MeasureName::Gamma => params[1] * inv_regularized_gamma_p(p, params[0]),
            MeasureName::Uniform01 => p.clamp(0.0, 1.0),
        })
    }

    /// Carry a unit point `u ∈ [0, 1)` onto `interval` under this
    /// measure. `Uniform` is affine; a named measure inverts its CDF
    /// between the CDF values of the interval's ends. The result
    /// lies in the interval, and an open end is never returned: a
    /// point that lands on it moves to the next float inside.
    pub fn map_unit(&self, u: f64, interval: &Interval) -> f64 {
        let x = match self.cdf(interval.lo) {
            None => interval.lo + u * (interval.hi - interval.lo),
            Some(f_lo) => {
                let f_hi = self.cdf(interval.hi).unwrap_or(1.0);
                let p = f_lo + u * (f_hi - f_lo);
                let q = self.quantile(p).unwrap_or(interval.lo);
                if q.is_nan() { interval.lo } else { q }
            }
        };
        inside(x.clamp(interval.lo, interval.hi), interval)
    }

    /// The point at an end of `interval` under this measure: the end
    /// itself, or the next float inside when the end is open.
    pub fn endpoint(&self, interval: &Interval, upper: bool) -> f64 {
        let x = if upper { interval.hi } else { interval.lo };
        inside(x, interval)
    }
}

/// `x` moved off an open end of `interval` onto the next float
/// inside; an infinite end has no next float and stays.
fn inside(x: f64, interval: &Interval) -> f64 {
    if interval.lo_open && x == interval.lo && x.is_finite() {
        x.next_up().min(interval.hi)
    } else if interval.hi_open && x == interval.hi && x.is_finite() {
        x.next_down().max(interval.lo)
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(name: MeasureName, params: &[f64]) -> AxisMeasure {
        AxisMeasure::named(name, params).unwrap()
    }

    #[test]
    fn uniform_is_affine() {
        let iv = Interval::closed(2.0, 4.0);
        assert_eq!(AxisMeasure::Uniform.map_unit(0.0, &iv), 2.0);
        assert_eq!(AxisMeasure::Uniform.map_unit(0.5, &iv), 3.0);
        assert_eq!(AxisMeasure::Uniform.map_unit(0.25, &iv), 2.5);
    }

    #[test]
    fn every_named_measure_inverts_its_own_cdf() {
        let all = [
            named(MeasureName::Normal, &[10.0, 2.0]),
            named(MeasureName::Exponential, &[0.5]),
            named(MeasureName::Pareto, &[2.0, 3.0]),
            named(MeasureName::Beta, &[2.0, 5.0]),
            named(MeasureName::LogNormal, &[0.0, 0.5]),
            named(MeasureName::Gamma, &[2.5, 2.0]),
            named(MeasureName::Uniform01, &[]),
        ];
        for m in &all {
            for p in [0.05, 0.3, 0.5, 0.8, 0.95] {
                let x = m.quantile(p).unwrap();
                let back = m.cdf(x).unwrap();
                // The probit is the least exact piece (4.5e-4).
                assert!((back - p).abs() < 1e-3, "{m:?} p={p} x={x} back={back}");
            }
        }
    }

    #[test]
    fn a_named_measure_on_its_full_support_is_its_quantile() {
        let m = named(MeasureName::Normal, &[0.0, 1.0]);
        let iv = Interval::open(f64::NEG_INFINITY, f64::INFINITY);
        assert_eq!(m.map_unit(0.5, &iv), m.quantile(0.5).unwrap());
        assert_eq!(m.map_unit(0.975, &iv), m.quantile(0.975).unwrap());
    }

    #[test]
    fn a_named_measure_on_a_narrower_interval_is_restricted_to_it() {
        // Exponential(1) on [0, 1]: u maps to F⁻¹(u · F(1)).
        let m = named(MeasureName::Exponential, &[1.0]);
        let iv = Interval::closed(0.0, 1.0);
        let f1 = 1.0 - (-1.0f64).exp();
        for u in [0.0, 0.25, 0.5, 0.9] {
            let expected = -(1.0 - u * f1).ln();
            let got = m.map_unit(u, &iv);
            assert!((got - expected).abs() < 1e-12, "u={u} got={got}");
            assert!((0.0..=1.0).contains(&got));
        }
    }

    #[test]
    fn open_ends_are_never_returned() {
        let iv = Interval::open(0.0, 1.0);
        let u = AxisMeasure::Uniform;
        assert!(u.map_unit(0.0, &iv) > 0.0);
        assert!(u.endpoint(&iv, false) > 0.0);
        assert!(u.endpoint(&iv, true) < 1.0);
        assert_eq!(u.endpoint(&Interval::closed(0.0, 1.0), true), 1.0);
    }

    #[test]
    fn a_product_measure_is_indexed_per_axis() {
        let pm = ProductMeasure::Product(vec![
            ProductMeasure::Uniform,
            ProductMeasure::Named(MeasureName::Exponential),
        ]);
        assert_eq!(
            AxisMeasure::from_product(&pm, 0).unwrap(),
            AxisMeasure::Uniform
        );
        assert_eq!(
            AxisMeasure::from_product(&pm, 1).unwrap(),
            named(MeasureName::Exponential, &[1.0])
        );
        assert!(AxisMeasure::from_product(&pm, 2).is_err());
    }

    #[test]
    fn parameters_are_checked_against_the_table() {
        assert!(AxisMeasure::named(MeasureName::Normal, &[1.0]).is_err());
        assert!(AxisMeasure::named(MeasureName::Normal, &[1.0, f64::NAN]).is_err());
        assert_eq!(
            AxisMeasure::named(MeasureName::Gamma, &[]).unwrap(),
            named(MeasureName::Gamma, &[1.0, 1.0])
        );
    }
}
