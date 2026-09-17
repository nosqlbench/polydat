// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Sampling over continuous and hybrid spaces (comprehension_forms.md
//! §3.6, §6.3, §10.2 R2): extrema corners of a continuous box, one
//! strategy over a discrete × continuous cartesian, filters over the
//! draws, and named measures by inverse CDF.

use std::collections::HashMap;

use polydat::iteration::comprehension::ast::Comprehension;
use polydat::iteration::comprehension::cardinality::{Interval, MeasureName, ProductMeasure};
use polydat::iteration::comprehension::measure::AxisMeasure;
use polydat::iteration::comprehension::runtime::evaluate_for_iteration;
use polydat::iteration::comprehension::source::Source;
use polydat::iteration::comprehension::strategy::StrategyName;
use polydat::iteration::comprehension::validate::{Mode, ValidationError, validate};

/// Collect the `(px, py)` pairs a traversal body computes, as floats.
fn float_pairs(src: &str) -> Vec<(f64, f64)> {
    let mut k = polydat::dsl::compile_polydat(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    k.set_inputs(&[0]);
    let mut stream = k.traverse(0).unwrap();
    let mut seen = Vec::new();
    while let Some(mut a) = stream.advance().unwrap() {
        let c = a.cycle(0);
        seen.push((c.pull("px").as_f64(), c.pull("py").as_f64()));
    }
    seen
}

fn body(clauses: &str, px: &str, py: &str) -> String {
    format!(
        "input cycle: u64\nfor {clauses} {{\n    px := f64_add({px}, 0.0)\n    py := f64_add({py}, 0.0)\n}}\n"
    )
}

/// Extrema over a continuous box (§3.6, §10.2 R2): `extrema/1` is the
/// box's corners, each axis at an interval end, lex within the stratum.
/// The text's `a..b` is half-open like the integer range, so the upper
/// corner is the last float inside.
#[test]
fn extrema_over_a_continuous_box_is_its_corners() {
    let seen = float_pairs(&body(
        "x in 2.0..4.0, y in 0.0..1.0 order extrema/1",
        "x",
        "y",
    ));
    let (top_x, top_y) = (4.0f64.next_down(), 1.0f64.next_down());
    assert_eq!(
        seen,
        vec![(2.0, 0.0), (2.0, top_y), (top_x, 0.0), (top_x, top_y)]
    );
}

/// A hybrid discrete × continuous cartesian samples through one
/// strategy (§10.2 R2, §11.11): the discrete axis is indexed by
/// position, the continuous axis is carried onto its interval, and the
/// coordinates keep clause order.
#[test]
fn a_hybrid_cartesian_samples_both_kinds_of_axis() {
    let seen = float_pairs(&body(
        "k in 1,2,4,8, theta in 0.0..1.0 order lhs/8",
        "k",
        "theta",
    ));
    assert_eq!(seen.len(), 8);
    assert!(
        seen.iter().all(|(k, _)| [1.0, 2.0, 4.0, 8.0].contains(k)),
        "{seen:?}"
    );
    assert!(seen.iter().all(|(_, t)| (0.0..1.0).contains(t)), "{seen:?}");
    // Latin hypercube: eight draws, one in each eighth of the interval.
    let mut bins: Vec<usize> = seen.iter().map(|(_, t)| (t * 8.0) as usize).collect();
    bins.sort_unstable();
    assert_eq!(bins, (0..8).collect::<Vec<_>>(), "{seen:?}");

    // Extrema over a hybrid: the interval's ends and the discrete
    // axis's ends, four corners, in clause order.
    let seen = float_pairs(&body(
        "theta in 0.0..1.0, k in 1,2,4,8 order extrema/1",
        "theta",
        "k",
    ));
    let top = 1.0f64.next_down();
    assert_eq!(seen, vec![(0.0, 1.0), (0.0, 8.0), (top, 1.0), (top, 8.0)]);

    // Halton, Sobol, and shuffle sample a hybrid too, inside the interval.
    for strategy in ["halton/6", "sobol/6", "shuffle/6"] {
        let seen = float_pairs(&body(
            &format!("k in 1,2,4,8, theta in 0.0..1.0 order {strategy}"),
            "k",
            "theta",
        ));
        assert_eq!(seen.len(), 6, "{strategy}");
        assert!(
            seen.iter()
                .all(|(k, t)| [1.0, 2.0, 4.0, 8.0].contains(k) && (0.0..1.0).contains(t)),
            "{strategy}: {seen:?}"
        );
    }
}

/// Shuffle and Lhs over a continuous interval stay inside it, and Lhs
/// stratifies it (§3.6: "n PRNG draws from the measure"; the classical
/// Latin hypercube over a real box).
#[test]
fn shuffle_and_lhs_draw_inside_a_continuous_interval() {
    let seen = float_pairs(&body(
        "x in 2.0..4.0, y in 10.0..20.0 order shuffle/16",
        "x",
        "y",
    ));
    assert_eq!(seen.len(), 16);
    assert!(
        seen.iter()
            .all(|(x, y)| (2.0..4.0).contains(x) && (10.0..20.0).contains(y)),
        "{seen:?}"
    );
    let seen = float_pairs(&body(
        "x in 2.0..4.0, y in 10.0..20.0 order lhs/5",
        "x",
        "y",
    ));
    let mut xbins: Vec<usize> = seen
        .iter()
        .map(|(x, _)| ((x - 2.0) / 2.0 * 5.0) as usize)
        .collect();
    let mut ybins: Vec<usize> = seen
        .iter()
        .map(|(_, y)| ((y - 10.0) / 10.0 * 5.0) as usize)
        .collect();
    xbins.sort_unstable();
    ybins.sort_unstable();
    assert_eq!(xbins, vec![0, 1, 2, 3, 4], "{seen:?}");
    assert_eq!(ybins, vec![0, 1, 2, 3, 4], "{seen:?}");
}

/// A filter between the order and a continuous clause applies to the
/// drawn points, and a sequence strategy keeps drawing until the count
/// is met (§6.3: a filter shrinks the realized subset).
#[test]
fn a_filter_over_a_continuous_source_applies_to_the_draws() {
    let seen = float_pairs(&body(
        "x in 0.0..1.0, y in 0.0..1.0 where {x} > 0.5 order halton/4",
        "x",
        "y",
    ));
    assert_eq!(seen.len(), 4, "{seen:?}");
    assert!(seen.iter().all(|(x, _)| *x > 0.5), "{seen:?}");
    // The kept points are the first passing points of the sequence:
    // the prefix of an unfiltered draw, filtered.
    let all = float_pairs(&body(
        "x in 0.0..1.0, y in 0.0..1.0 order halton/16",
        "x",
        "y",
    ));
    let expected: Vec<(f64, f64)> = all.into_iter().filter(|(x, _)| *x > 0.5).take(4).collect();
    assert_eq!(seen, expected);
}

// ---- Named measures: the AST form, no text spelling ----

fn scope() -> polydat::kernel::PolydatKernel {
    polydat::dsl::compile_polydat("input cycle: u64\n").unwrap()
}

fn xs(ast: &Comprehension) -> Vec<f64> {
    let tuples = evaluate_for_iteration(ast, &scope(), &HashMap::new(), |_| Ok(())).unwrap();
    tuples.iter().map(|t| t[0].1.as_f64()).collect()
}

fn full_line() -> Interval {
    Interval::open(f64::NEG_INFINITY, f64::INFINITY)
}

/// A `Source::Distribution` samples by the inverse CDF of its measure
/// (§10.2 R2: "inverse-CDF for named measures").
#[test]
fn a_distribution_source_samples_by_its_inverse_cdf() {
    let normal = Comprehension::clause(
        "x",
        Source::Distribution {
            distribution: MeasureName::Normal,
            support: full_line(),
            params: vec![10.0, 2.0],
        },
    );
    let ast = Comprehension::order(normal, StrategyName::Halton, Some(256));
    validate(&ast, Mode::Strict).unwrap();
    let xs = xs(&ast);
    assert_eq!(xs.len(), 256);
    let mean = xs.iter().sum::<f64>() / 256.0;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 256.0;
    assert!((mean - 10.0).abs() < 0.2, "mean {mean}");
    assert!((var.sqrt() - 2.0).abs() < 0.2, "stddev {}", var.sqrt());
    // Each point is the quantile of its draw: the first base-2 Halton
    // point is 1/2, so the first sample is the median.
    let m = AxisMeasure::named(MeasureName::Normal, &[10.0, 2.0]).unwrap();
    assert_eq!(xs[0], m.quantile(0.5).unwrap());
}

#[test]
fn every_named_measure_samples_inside_its_support() {
    let cases = [
        (MeasureName::Exponential, vec![2.0], 0.0, f64::INFINITY),
        (MeasureName::Pareto, vec![3.0, 2.0], 3.0, f64::INFINITY),
        (MeasureName::Beta, vec![2.0, 5.0], 0.0, 1.0),
        (MeasureName::LogNormal, vec![0.0, 0.5], 0.0, f64::INFINITY),
        (MeasureName::Gamma, vec![2.0, 1.5], 0.0, f64::INFINITY),
        (MeasureName::Uniform01, vec![], 0.0, 1.0),
    ];
    for (name, params, lo, hi) in cases {
        let ast = Comprehension::order(
            Comprehension::clause(
                "x",
                Source::Distribution {
                    distribution: name,
                    support: Interval::closed(lo, hi),
                    params,
                },
            ),
            StrategyName::Sobol,
            Some(32),
        );
        let xs = xs(&ast);
        assert_eq!(xs.len(), 32, "{name:?}");
        assert!(
            xs.iter().all(|x| (lo..=hi).contains(x) && x.is_finite()),
            "{name:?}: {xs:?}"
        );
    }
}

/// A named measure on an interval narrower than its support is the
/// measure restricted to the interval (§5 V8's last row).
#[test]
fn a_named_measure_on_a_narrower_interval_is_restricted_to_it() {
    // Exponential(1) on [0, 1]: no draw leaves the interval, and the
    // draws keep the measure's shape (more mass near 0).
    let ast = Comprehension::order(
        Comprehension::clause(
            "x",
            Source::ContinuousInterval {
                interval: Interval::closed(0.0, 1.0),
                measure: ProductMeasure::Named(MeasureName::Exponential),
            },
        ),
        StrategyName::Halton,
        Some(64),
    );
    let xs = xs(&ast);
    assert_eq!(xs.len(), 64);
    assert!(xs.iter().all(|x| (0.0..=1.0).contains(x)), "{xs:?}");
    let low = xs.iter().filter(|x| **x < 0.5).count();
    // F(0.5) / F(1) of Exponential(1) is 0.62: about 40 of 64 sit below
    // the midpoint, against 32 for a uniform draw.
    assert!(
        (36..=44).contains(&low),
        "{low} of 64 below the midpoint: {xs:?}"
    );
    let m = AxisMeasure::named(MeasureName::Exponential, &[1.0]).unwrap();
    assert_eq!(xs[0], m.map_unit(0.5, &Interval::closed(0.0, 1.0)));
}

/// The parameters of a distribution source are checked in the compile
/// (V8, the declared measure).
#[test]
fn distribution_parameters_are_checked_at_compile() {
    let ast = Comprehension::order(
        Comprehension::clause(
            "x",
            Source::Distribution {
                distribution: MeasureName::Normal,
                support: full_line(),
                params: vec![1.0],
            },
        ),
        StrategyName::Halton,
        Some(4),
    );
    let err = validate(&ast, Mode::Permissive).unwrap_err();
    assert!(
        matches!(err, ValidationError::V8ContinuousRequirement { ref reason } if reason.contains("mean, stddev")),
        "{err}"
    );
}
