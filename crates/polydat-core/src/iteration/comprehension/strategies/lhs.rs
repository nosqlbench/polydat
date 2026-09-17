// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `Lhs` (Latin Hypercube Sampling) strategy — spec §3.6.
//!
//! Per-axis stratified permutation. For K-D + n samples:
//!
//! 1. Stratify each axis's `[0, size)` into n equal bins.
//! 2. For each axis, generate a random permutation of `0..n`.
//! 3. Sample i = zip per-axis `permutations[i]`.
//!
//! The result is n tuples that cover each axis's bins
//! uniformly (Latin square property in K-D). Native to
//! continuous K-D boxes — the stratification is the
//! mathematical definition. Over discrete inputs, the
//! stratified positions are floored to integer indices.
//!
//! 1-axis Lhs is degenerate (equivalent to Shuffle); spec
//! §5.8 emits a warning when this composition is detected
//! (handled in `validate.rs`).
//!
//! ## References
//!
//! - M. D. McKay, R. J. Beckman, W. J. Conover, "A Comparison of
//!   Three Methods for Selecting Values of Input Variables in the
//!   Analysis of Output from a Computer Code," *Technometrics* 21(2)
//!   (1979), 239–245.
//!   doi:[10.2307/1268522](https://doi.org/10.2307/1268522). The
//!   original Latin Hypercube design.
//! - The defining property: with `N` samples each axis is stratified
//!   into `N` equal bins and **each bin is hit exactly once** — i.e.
//!   the per-axis stratum assignment is a permutation of `0..N`. This
//!   marginal-stratification guarantee is verified in
//!   `tests::lhs_latin_property_each_axis_is_a_permutation`.

use super::{
    EvaluatedInput, MultiIndex, Strategy, Tuple, index_fn_dim, index_fn_size,
    index_fn_supports_lookup, multi_index_to_flat, prng::Prng,
};
use crate::iteration::comprehension::metadata::IndexFn;
use crate::iteration::comprehension::strategy::StrategyName;

/// Latin hypercube samples.
pub struct Lhs;

/// Seed base when none is authored; the input length is added per
/// call. Per-streamer
/// seeding is not implemented.
const SEED: u64 = 0x1A50_4577_3EED_BEEF;

impl Strategy for Lhs {
    fn name(&self) -> StrategyName {
        StrategyName::Lhs
    }

    fn accepts_input(&self, idx: Option<&IndexFn>) -> bool {
        idx.is_some()
    }

    fn has_closed_form_for(&self, _idx: &IndexFn) -> bool {
        true
    }

    fn apply(&self, input: &EvaluatedInput, truncation: Option<u64>) -> Vec<Tuple> {
        self.apply_seeded(input, truncation, None)
    }

    fn apply_seeded(
        &self,
        input: &EvaluatedInput,
        truncation: Option<u64>,
        seed: Option<u64>,
    ) -> Vec<Tuple> {
        if index_fn_supports_lookup(&input.index_fn) {
            let mis = lhs_multi_indices(&input.index_fn, truncation, seed);
            mis.into_iter()
                .filter_map(|mi| multi_index_to_flat(&input.index_fn, &mi))
                .filter_map(|flat| input.tuples.get(flat).cloned())
                .collect()
        } else {
            naive_lhs_over_tuples(&input.tuples, truncation, seed)
        }
    }
}

fn naive_lhs_over_tuples(
    input: &[Tuple],
    truncation: Option<u64>,
    seed: Option<u64>,
) -> Vec<Tuple> {
    let total = input.len() as u64;
    if total == 0 {
        return Vec::new();
    }
    let n = match truncation {
        Some(t) => t.min(total),
        None => total,
    };
    let mut rng = Prng::new(seed.unwrap_or(SEED).wrapping_add(total));
    let mut indices: Vec<u64> = (0..total).collect();
    rng.shuffle(&mut indices);
    indices
        .into_iter()
        .take(n as usize)
        .map(|i| input[i as usize].clone())
        .collect()
}

/// The multi-indices of a Latin hypercube over `idx`, `truncation`
/// of them, from the authored `seed` or the default.
pub(crate) fn lhs_multi_indices(
    idx: &IndexFn,
    truncation: Option<u64>,
    seed: Option<u64>,
) -> Vec<MultiIndex> {
    let dim = index_fn_dim(idx);
    if dim == 0 {
        return Vec::new();
    }
    let total = index_fn_size(idx);
    let n = match (truncation, total) {
        (Some(t), 0) => t,
        (Some(t), tot) => t.min(tot),
        (None, 0) => return Vec::new(),
        (None, tot) => tot,
    };
    if n == 0 {
        return Vec::new();
    }

    let axis_sizes = axis_sizes_for(idx, dim);

    let mut rng = Prng::new(seed.unwrap_or(SEED).wrapping_add(n));
    let mut per_axis_perms: Vec<Vec<u64>> = Vec::with_capacity(dim);
    for _ in 0..dim {
        let mut perm: Vec<u64> = (0..n).collect();
        rng.shuffle(&mut perm);
        per_axis_perms.push(perm);
    }

    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let mut mi: MultiIndex = Vec::with_capacity(dim);
        for axis in 0..dim {
            let stratum = per_axis_perms[axis][i as usize];
            let size = axis_sizes[axis];
            mi.push(if size == u64::MAX {
                // A continuous axis: one draw inside the stratum's bin
                // of the unit interval, as a 53-bit fraction.
                let jitter = (rng.next_u64() >> 11) as f64 / UNIT_SCALE;
                ((stratum as f64 + jitter) / n as f64 * UNIT_SCALE) as u64
            } else if size >= n {
                (stratum * size) / n
            } else {
                stratum % size
            });
        }
        out.push(mi);
    }
    out
}

/// The scale of a continuous code: a point of `[0, 1)` as a 53-bit
/// fraction, the encoding the runtime's sampler reads.
const UNIT_SCALE: f64 = (1u64 << 53) as f64;

fn axis_sizes_for(idx: &IndexFn, dim: usize) -> Vec<u64> {
    match idx {
        IndexFn::Lattice { axis_sizes } | IndexFn::Modular { axis_sizes } => axis_sizes.clone(),
        IndexFn::Lockstep { length } => vec![*length],
        IndexFn::Concatenation { segment_sizes } => vec![segment_sizes.iter().sum()],
        IndexFn::Continuous { .. } => vec![u64::MAX; dim],
        IndexFn::Hybrid {
            discrete_axes,
            continuous_axes,
            ..
        } => {
            let mut s = discrete_axes.clone();
            s.extend(continuous_axes.iter().map(|_| u64::MAX));
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lhs_per_axis_stratification_2d_discrete() {
        let idx = IndexFn::Lattice {
            axis_sizes: vec![10, 10],
        };
        let out = lhs_multi_indices(&idx, Some(5), None);
        assert_eq!(out.len(), 5);

        let axis_0_values: std::collections::HashSet<u64> = out.iter().map(|mi| mi[0]).collect();
        let axis_1_values: std::collections::HashSet<u64> = out.iter().map(|mi| mi[1]).collect();
        assert_eq!(axis_0_values.len(), 5);
        assert_eq!(axis_1_values.len(), 5);
    }

    #[test]
    fn lhs_continuous_each_stratum_used() {
        use crate::iteration::comprehension::cardinality::{Interval, ProductMeasure};
        let idx = IndexFn::Continuous {
            intervals: vec![Interval::closed(0.0, 1.0), Interval::closed(0.0, 1.0)],
            measure: ProductMeasure::Uniform,
        };
        let out = lhs_multi_indices(&idx, Some(10), None);
        assert_eq!(out.len(), 10);
        let axis_0: std::collections::HashSet<u64> = out.iter().map(|mi| mi[0]).collect();
        let axis_1: std::collections::HashSet<u64> = out.iter().map(|mi| mi[1]).collect();
        assert_eq!(axis_0.len(), 10);
        assert_eq!(axis_1.len(), 10);
    }

    #[test]
    fn lhs_latin_property_each_axis_is_a_permutation() {
        // McKay/Beckman/Conover 1979: the defining marginal property
        // — for N samples over a continuous box, each axis's N strata
        // are exactly {0,1,…,N-1} (each bin used once). On a
        // continuous box each code is a 53-bit fraction drawn inside
        // its stratum's bin, so the bins the codes fall in must be
        // the full 0..N range.

        use crate::iteration::comprehension::cardinality::{Interval, ProductMeasure};
        let n = 16u64;
        let idx = IndexFn::Continuous {
            intervals: vec![
                Interval::closed(0.0, 1.0),
                Interval::closed(0.0, 1.0),
                Interval::closed(0.0, 1.0),
            ],
            measure: ProductMeasure::Uniform,
        };
        let out = lhs_multi_indices(&idx, Some(n), None);
        assert_eq!(out.len(), n as usize);
        let expected: std::collections::BTreeSet<u64> = (0..n).collect();
        for axis in 0..3 {
            let got: std::collections::BTreeSet<u64> =
                out.iter().map(|mi| mi[axis] * n / (1u64 << 53)).collect();
            assert_eq!(
                got, expected,
                "axis {axis}: LHS strata must be a permutation of 0..{n}"
            );
        }
    }

    #[test]
    fn deterministic() {
        let idx = IndexFn::Lattice {
            axis_sizes: vec![20, 20],
        };
        let a = lhs_multi_indices(&idx, Some(10), None);
        let b = lhs_multi_indices(&idx, Some(10), None);
        assert_eq!(a, b);
    }

    /// A continuous axis's codes are 53-bit fractions, one in each of
    /// the n equal bins of the unit interval (the Latin hypercube over
    /// a real box, spec §10.2 R2).
    #[test]
    fn continuous_axis_codes_are_stratified_unit_fractions() {
        use crate::iteration::comprehension::cardinality::{Interval, ProductMeasure};
        let idx = IndexFn::Continuous {
            intervals: vec![Interval::closed(0.0, 1.0), Interval::closed(0.0, 1.0)],
            measure: ProductMeasure::Uniform,
        };
        let out = lhs_multi_indices(&idx, Some(8), None);
        assert_eq!(out.len(), 8);
        for axis in 0..2 {
            let mut bins: Vec<u64> = out
                .iter()
                .map(|mi| {
                    assert!(mi[axis] < (1u64 << 53));
                    mi[axis] * 8 / (1u64 << 53)
                })
                .collect();
            bins.sort_unstable();
            assert_eq!(bins, (0..8).collect::<Vec<_>>(), "axis {axis}: {out:?}");
        }
        // A hybrid: the discrete axis is a position, the continuous one a code.
        let idx = IndexFn::Hybrid {
            discrete_axes: vec![4],
            continuous_axes: vec![Interval::closed(0.0, 1.0)],
            measure: ProductMeasure::Uniform,
        };
        let out = lhs_multi_indices(&idx, Some(8), None);
        assert!(
            out.iter().all(|mi| mi[0] < 4 && mi[1] < (1u64 << 53)),
            "{out:?}"
        );
    }

    /// An authored seed selects a different permutation from the
    /// default and the same one on every call (comprehension_forms.md
    /// §3.6: state from the authored seed and structural identity).
    #[test]
    fn an_authored_seed_is_deterministic_and_distinct() {
        let idx = IndexFn::Lattice {
            axis_sizes: vec![6, 6],
        };
        let default = lhs_multi_indices(&idx, Some(12), None);
        let seeded = lhs_multi_indices(&idx, Some(12), Some(42));
        let again = lhs_multi_indices(&idx, Some(12), Some(42));
        assert_eq!(seeded, again);
        assert_ne!(seeded, default);
        assert_ne!(seeded, lhs_multi_indices(&idx, Some(12), Some(43)));
    }
}
