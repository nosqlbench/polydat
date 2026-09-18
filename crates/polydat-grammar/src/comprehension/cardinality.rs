// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Cardinality classes — spec §6.1.
//!
//! Six classes describe every comprehension's dispense count.
//! Three are discrete (`Bounded`, `BoundedAtMost`, `Unbounded`);
//! two are continuous-domain (`Continuous`, `ContinuousAtMost`);
//! one is hybrid (`Hybrid`). The class propagates through every
//! constructor per spec §6.1's table.

use serde::{Deserialize, Serialize};

/// Cardinality of a comprehension's dispense stream.
///
/// Six variants per spec §6.1:
///
/// - **Discrete classes** enumerate distinct tuples; the count
///   may be known exactly (`Bounded`), bounded above
///   (`BoundedAtMost`), or unknown (`Unbounded`).
/// - **Continuous classes** describe a measure-theoretic value
///   space; they cannot enumerate and must be sampled via an
///   enclosing `order(_, strategy, Some(n))` per V8.
/// - **Hybrid** is a cartesian whose children mix discrete and
///   continuous axes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CardinalityClass {
    /// Discrete, exactly `n` tuples.
    Bounded(u64),

    /// Discrete, between 0 and `n` tuples (post-filter).
    BoundedAtMost(u64),

    /// Discrete, no known upper bound (generator, live stream).
    Unbounded,

    /// Continuous source — bounded or unbounded real intervals
    /// with an integrable product measure. Sampled rather than
    /// enumerated; V8 requires an enclosing
    /// `order(_, strategy, Some(n))` before reaching a
    /// `PolyStreamer`.
    Continuous {
        /// The interval of each axis.
        intervals: Vec<Interval>,
        /// The measure sampled.
        measure: ProductMeasure,
    },

    /// Filtered continuous source. Measure reduced by the
    /// predicate; still requires sampling.
    ContinuousAtMost {
        /// The interval of each axis.
        intervals: Vec<Interval>,
        /// The measure before the predicate reduces it.
        measure_at_most: ProductMeasure,
    },

    /// Mixed discrete × continuous cartesian. The discrete part
    /// is enumerable; the continuous part needs sampling. V8
    /// applies to the continuous component.
    Hybrid(Hybrid),
}

/// Mixed discrete × continuous cartesian shape.
///
/// Each `discrete_axes` entry is the axis size in tuples; each
/// `continuous_axes` entry is the interval the axis spans.
/// `measure` covers the continuous part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hybrid {
    /// Per-axis cardinality for the discrete axes, in
    /// declaration order.
    pub discrete_axes: Vec<u64>,
    /// Per-axis intervals for the continuous axes, in
    /// declaration order.
    pub continuous_axes: Vec<Interval>,
    /// Product measure over the continuous axes.
    pub measure: ProductMeasure,
}

/// Real interval `[lo, hi]` (or open variants) for continuous
/// sources. Unbounded sides use `f64::NEG_INFINITY` /
/// `f64::INFINITY`; the V8 integrability check determines
/// whether such intervals are valid given the measure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    /// The lower end.
    pub lo: f64,
    /// The upper end.
    pub hi: f64,
    /// Whether the lower end is excluded.
    pub lo_open: bool,
    /// Whether the upper end is excluded.
    pub hi_open: bool,
}

impl Interval {
    /// Closed-closed interval `[lo, hi]`.
    pub fn closed(lo: f64, hi: f64) -> Self {
        Self {
            lo,
            hi,
            lo_open: false,
            hi_open: false,
        }
    }

    /// Half-open `[lo, hi)`.
    pub fn half_open(lo: f64, hi: f64) -> Self {
        Self {
            lo,
            hi,
            lo_open: false,
            hi_open: true,
        }
    }

    /// Open interval `(lo, hi)`.
    pub fn open(lo: f64, hi: f64) -> Self {
        Self {
            lo,
            hi,
            lo_open: true,
            hi_open: true,
        }
    }

    /// `true` if the interval has finite Lebesgue measure
    /// (both endpoints finite). Used by V8's integrability
    /// check together with the measure variant.
    pub fn is_bounded(&self) -> bool {
        self.lo.is_finite() && self.hi.is_finite()
    }
}

/// Product measure over one or more continuous axes.
///
/// `Uniform` is the Lebesgue measure scaled by interval width
/// (requires bounded intervals; V8 rejects unbounded + Uniform).
/// `Named(D)` is a probability distribution with proper density
/// over its declared support — Normal, Exponential, Pareto,
/// Beta, etc. `Product(_)` carries a per-axis product of measures
/// for K-D continuous cartesians.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProductMeasure {
    /// Lebesgue measure scaled to the interval; needs bounded intervals.
    Uniform,
    /// A named probability distribution over its support.
    Named(MeasureName),
    /// One measure per axis.
    Product(Vec<ProductMeasure>),
}

impl ProductMeasure {
    /// `true` if this measure has finite total mass given the
    /// supplied intervals. Used by V8.
    ///
    /// - `Uniform` is integrable iff every interval is bounded.
    /// - `Named(D)` is integrable per its distribution: proper
    ///   probability distributions are always integrable
    ///   (they have unit total mass by definition).
    /// - `Product(children)` is integrable iff every child is.
    pub fn is_integrable(&self, intervals: &[Interval]) -> bool {
        match self {
            ProductMeasure::Uniform => intervals.iter().all(Interval::is_bounded),
            ProductMeasure::Named(name) => name.is_proper_probability_measure(),
            ProductMeasure::Product(children) => {
                if children.len() != intervals.len() {
                    return false;
                }
                children
                    .iter()
                    .zip(intervals.iter())
                    .all(|(m, i)| m.is_integrable(std::slice::from_ref(i)))
            }
        }
    }
}

/// Named continuous distribution. Closed enum per spec
/// §10.7.5's "User-defined extensions" non-goal — new
/// distributions land as coordinated additions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeasureName {
    /// The normal distribution.
    Normal,
    /// The exponential distribution.
    Exponential,
    /// The Pareto distribution.
    Pareto,
    /// The beta distribution.
    Beta,
    /// The log-normal distribution.
    LogNormal,
    /// The gamma distribution.
    Gamma,
    /// The uniform distribution on `[0, 1]`.
    Uniform01,
}

impl MeasureName {
    /// All currently-named distributions are proper probability
    /// measures (unit total mass). V8 accepts them on any
    /// interval that matches the distribution's support.
    pub fn is_proper_probability_measure(self) -> bool {
        true
    }

    /// The distribution's parameters, in the order a
    /// `Source::Distribution`'s `params` lists them:
    ///
    /// | Measure | Parameters | Support |
    /// |---|---|---|
    /// | `Normal` | mean, stddev | (-∞, ∞) |
    /// | `Exponential` | rate | [0, ∞) |
    /// | `Pareto` | scale, shape | [scale, ∞) |
    /// | `Beta` | alpha, beta | [0, 1] |
    /// | `LogNormal` | mean, stddev (of the log) | (0, ∞) |
    /// | `Gamma` | shape, scale | [0, ∞) |
    /// | `Uniform01` | none | [0, 1] |
    pub fn parameter_names(self) -> &'static [&'static str] {
        match self {
            MeasureName::Normal | MeasureName::LogNormal => &["mean", "stddev"],
            MeasureName::Exponential => &["rate"],
            MeasureName::Pareto => &["scale", "shape"],
            MeasureName::Beta => &["alpha", "beta"],
            MeasureName::Gamma => &["shape", "scale"],
            MeasureName::Uniform01 => &[],
        }
    }

    /// The standard parameters, used when a source names the
    /// measure without parameters: `Normal(0, 1)`, `Exponential(1)`,
    /// `Pareto(1, 1)`, `Beta(1, 1)`, `LogNormal(0, 1)`, `Gamma(1, 1)`.
    pub fn default_params(self) -> &'static [f64] {
        match self {
            MeasureName::Normal | MeasureName::LogNormal => &[0.0, 1.0],
            MeasureName::Exponential => &[1.0],
            MeasureName::Pareto | MeasureName::Beta | MeasureName::Gamma => &[1.0, 1.0],
            MeasureName::Uniform01 => &[],
        }
    }

    /// The parameters as the sampler takes them: `params` when it
    /// has the measure's count, the standard parameters when it is
    /// empty, and an error naming the expected parameters otherwise.
    pub fn resolve_params(self, params: &[f64]) -> Result<Vec<f64>, String> {
        let names = self.parameter_names();
        if params.is_empty() {
            return Ok(self.default_params().to_vec());
        }
        if params.len() != names.len() {
            return Err(format!(
                "{self:?} takes {} parameter(s) ({}); found {}",
                names.len(),
                names.join(", "),
                params.len()
            ));
        }
        if let Some(bad) = params.iter().find(|p| !p.is_finite()) {
            return Err(format!("{self:?}: parameter {bad} is not finite"));
        }
        Ok(params.to_vec())
    }

    /// The measure's name in comprehension source text, the spelling
    /// a clause writes: `normal(0, 1)`, `log_normal(0, 0.5)`.
    pub fn text(self) -> &'static str {
        match self {
            MeasureName::Normal => "normal",
            MeasureName::Exponential => "exponential",
            MeasureName::Pareto => "pareto",
            MeasureName::Beta => "beta",
            MeasureName::LogNormal => "log_normal",
            MeasureName::Gamma => "gamma",
            MeasureName::Uniform01 => "uniform01",
        }
    }

    /// The measure a source-text name denotes, or `None` when the
    /// name is not one of the closed set (§10.7.5): the caller reads
    /// the text as something else, a generator call for one.
    pub fn from_text(name: &str) -> Option<Self> {
        [
            MeasureName::Normal,
            MeasureName::Exponential,
            MeasureName::Pareto,
            MeasureName::Beta,
            MeasureName::LogNormal,
            MeasureName::Gamma,
            MeasureName::Uniform01,
        ]
        .into_iter()
        .find(|m| m.text() == name)
    }

    /// The measure's own support under `params`, the interval a
    /// source that names no narrower one draws from. A Pareto's
    /// support starts at its scale; every other measure's is fixed.
    pub fn support(self, params: &[f64]) -> Interval {
        let p = self
            .resolve_params(params)
            .unwrap_or_else(|_| self.default_params().to_vec());
        let inf = f64::INFINITY;
        match self {
            MeasureName::Normal => Interval::open(-inf, inf),
            MeasureName::Exponential | MeasureName::Gamma => Interval {
                lo: 0.0,
                hi: inf,
                lo_open: false,
                hi_open: true,
            },
            MeasureName::Pareto => Interval {
                lo: p[0],
                hi: inf,
                lo_open: false,
                hi_open: true,
            },
            MeasureName::LogNormal => Interval {
                lo: 0.0,
                hi: inf,
                lo_open: true,
                hi_open: true,
            },
            MeasureName::Beta | MeasureName::Uniform01 => Interval::closed(0.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_interval_is_bounded() {
        assert!(Interval::closed(0.0, 1.0).is_bounded());
        assert!(Interval::open(-1.0, 1.0).is_bounded());
    }

    #[test]
    fn unbounded_interval_is_not_bounded() {
        let i = Interval {
            lo: 0.0,
            hi: f64::INFINITY,
            lo_open: false,
            hi_open: true,
        };
        assert!(!i.is_bounded());
    }

    #[test]
    fn uniform_integrable_on_bounded_interval() {
        let m = ProductMeasure::Uniform;
        assert!(m.is_integrable(&[Interval::closed(0.0, 1.0)]));
    }

    #[test]
    fn uniform_not_integrable_on_unbounded_interval() {
        let m = ProductMeasure::Uniform;
        let unbounded = Interval {
            lo: 0.0,
            hi: f64::INFINITY,
            lo_open: false,
            hi_open: true,
        };
        assert!(!m.is_integrable(&[unbounded]));
    }

    #[test]
    fn named_measure_always_integrable() {
        let m = ProductMeasure::Named(MeasureName::Normal);
        let unbounded = Interval {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
            lo_open: true,
            hi_open: true,
        };
        assert!(m.is_integrable(&[unbounded]));
    }

    #[test]
    fn product_measure_requires_matching_arity() {
        let m = ProductMeasure::Product(vec![ProductMeasure::Uniform, ProductMeasure::Uniform]);
        assert!(m.is_integrable(&[Interval::closed(0.0, 1.0), Interval::closed(0.0, 1.0)]));
        assert!(!m.is_integrable(&[Interval::closed(0.0, 1.0)]));
    }

    #[test]
    fn cardinality_class_round_trip_serde() {
        let c = CardinalityClass::Continuous {
            intervals: vec![Interval::closed(0.0, 1.0)],
            measure: ProductMeasure::Uniform,
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: CardinalityClass = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}
