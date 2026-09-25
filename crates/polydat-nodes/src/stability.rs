// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Signal-settling / steady-state detection over a window of samples.
//!
//! An objective computed from a live metric is volatile: each read
//! follows the current metric window, and after a phase completes that
//! window is empty. A host that needs the level the phase actually
//! produced, and a signal for when that level has settled, keeps the
//! recent samples itself and asks `is_stable` about them.
//!
//! The node is a pure function of the window it is given. It keeps no
//! history of its own, so both outputs are memoized like any other
//! pure step, agree on all four engines and in every fiber, and read
//! the same whichever of the two a host pulls first (runtime_model.md
//! §9.1).

/// The settled level and the steady-state signal of a window of
/// samples.
///
/// Signature: `is_stable(samples: vec_f64, margin: f64, min_samples:
/// u64) -> (stable_value: f64, stable: u64)`
///
/// - `stable_value` is the median of `samples`: a central estimate that
///   one outlier, such as a trailing empty-window sample, barely moves.
///   An empty window reports `0.0`.
/// - `stable` is `1` when the window holds at least `min_samples`
///   samples and their standard deviation is within
///   `margin · max(|median|, 1)`, a relative band with an absolute
///   floor so a level near zero still settles; otherwise `0`.
///
/// The host owns the window: it appends each new sample and drops the
/// oldest, and writes the window as an input before it reads. The
/// cost is linear in the window's length and is paid only when the
/// window changes, so it scales with how often the host updates the
/// window, not with how often samples arrive. A host whose samples
/// arrive much faster than it needs to decide should aggregate them
/// into the window rather than pass every one.
#[polydat::polydat_node(category = Math, output_names(stable_value, stable))]
fn is_stable(
    samples: &[f64],
    #[poly_default(0.05f64)] margin: polydat::derive_support::Const<f64>,
    #[poly_default(8u64)] min_samples: polydat::derive_support::Const<u64>,
) -> (f64, u64) {
    let n = samples.len();
    if n == 0 {
        return (0.0, 0);
    }
    let median = median_of(samples);
    if n < (*min_samples) as usize {
        return (median, 0);
    }
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = samples
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let threshold = (*margin) * median.abs().max(1.0);
    let stable = if var.sqrt() <= threshold { 1 } else { 0 };
    (median, stable)
}

/// The median by selection rather than a full sort: linear in the
/// window's length. The window is copied, since the input is shared.
fn median_of(samples: &[f64]) -> f64 {
    let mut s = samples.to_vec();
    let n = s.len();
    let mid = n / 2;
    let (lower, upper, _) = s.select_nth_unstable_by(mid, f64::total_cmp);
    let upper = *upper;
    if n % 2 == 1 {
        upper
    } else {
        let below = lower.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (below + upper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polydat::ast::{PolydatNode, SliceArc, Value};

    fn window(v: &[f64]) -> Value {
        Value::VecF64(SliceArc::from_vec(v.to_vec()))
    }

    fn eval(node: &IsStable, samples: &[f64]) -> (f64, u64) {
        let mut out = [Value::None, Value::None];
        node.eval(&[window(samples)], &mut out);
        (out[0].as_f64(), out[1].as_u64())
    }

    /// A window that holds a ramp is not steady; once the host's window
    /// holds only the steady tail, the signal settles at its level.
    #[test]
    fn settles_on_a_steady_window_and_reports_the_level() {
        let node = IsStable::new(0.05, 4);
        let (_, stable) = eval(&node, &[0.0, 1.0, 3.0, 4.5, 5.0, 5.0, 5.0, 5.0]);
        assert_eq!(stable, 0, "a window with the ramp in it is not steady");
        let (level, stable) = eval(&node, &[5.0; 8]);
        assert_eq!(stable, 1, "a steady window is stable");
        assert!(
            (level - 5.0).abs() < 1e-9,
            "the level is the steady value, got {level}"
        );
    }

    /// One trailing outlier moves the median barely and breaks the strict
    /// steady-state test.
    #[test]
    fn a_trailing_outlier_does_not_move_the_level() {
        let node = IsStable::new(0.05, 4);
        let (level, stable) = eval(&node, &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0]);
        assert!(
            (level - 5.0).abs() < 1e-9,
            "the median absorbs one outlier, got {level}"
        );
        assert_eq!(stable, 0, "one outlier breaks strict steady state");
    }

    /// Below `min_samples` the signal is never stable, however clean.
    #[test]
    fn reports_unstable_until_min_samples() {
        let node = IsStable::new(0.05, 4);
        assert_eq!(eval(&node, &[5.0]).1, 0);
        assert_eq!(eval(&node, &[5.0; 3]).1, 0);
        assert_eq!(eval(&node, &[5.0; 4]).1, 1);
    }

    /// A level near zero still settles, thanks to the absolute floor.
    #[test]
    fn a_near_zero_level_still_settles() {
        let node = IsStable::new(0.05, 4);
        let (level, stable) = eval(&node, &[0.0; 8]);
        assert_eq!(stable, 1);
        assert!(level.abs() < 1e-9);
    }

    /// An empty window reports no level and no stability, and an even
    /// window's median is the mean of its two middle values.
    #[test]
    fn empty_and_even_windows() {
        let node = IsStable::new(0.05, 1);
        assert_eq!(eval(&node, &[]), (0.0, 0));
        assert!((eval(&node, &[1.0, 4.0, 2.0, 3.0]).0 - 2.5).abs() < 1e-12);
    }

    /// Both outputs come from the same window whichever is pulled first,
    /// and pulling them again returns the same values: the node keeps no
    /// history.
    #[test]
    fn both_outputs_read_one_window_in_any_order() {
        let mut k = polydat::dsl::compile::compile_polydat_kernel(
            "extern samples: vec_f64\n(stable_value, stable) := is_stable(samples, 0.05, 4)",
        )
        .expect("compile");
        k.set_input("samples", window(&[5.0; 8]))
            .expect("write the window");
        assert_eq!(k.pull("stable").as_u64(), 1);
        assert!((k.pull("stable_value").as_f64() - 5.0).abs() < 1e-9);
        assert_eq!(k.pull("stable").as_u64(), 1);
    }
}
