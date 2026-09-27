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
    MultiIndex, Selection, Strategy, capped, index_fn_dim, index_fn_size, index_fn_supports_lookup,
    prng::Prng,
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

    /// The hypercube stratifies each axis of the index space.
    fn selects_from_shape(&self) -> bool {
        true
    }

    fn accepts_input(&self, idx: Option<&IndexFn>) -> bool {
        idx.is_some()
    }

    fn has_closed_form_for(&self, _idx: &IndexFn) -> bool {
        true
    }

    fn select(
        &self,
        index_fn: &IndexFn,
        cardinality: u64,
        truncation: Option<u64>,
        seed: Option<u64>,
    ) -> Selection {
        if index_fn_supports_lookup(index_fn) {
            let mis = lhs_multi_indices(index_fn, truncation, seed);
            Selection::from_multi_indices(index_fn, mis, cardinality)
        } else {
            Selection::Positions(naive_lhs_positions(cardinality, truncation, seed))
        }
    }
}

/// A seeded permutation of `0..total` positions, as one axis, cut to
/// the truncation.
fn naive_lhs_positions(total: u64, truncation: Option<u64>, seed: Option<u64>) -> Vec<u64> {
    if total == 0 {
        return Vec::new();
    }
    let n = capped(truncation, total);
    let mut rng = Prng::new(seed.unwrap_or(SEED).wrapping_add(total));
    let mut indices: Vec<u64> = (0..total).collect();
    rng.shuffle(&mut indices);
    indices.truncate(n as usize);
    indices
}

/// The multi-indices of a Latin hypercube over `idx`, `truncation`
/// of them, from the authored `seed` or the default.
pub(crate) fn lhs_multi_indices(
    idx: &IndexFn,
    truncation: Option<u64>,
    seed: Option<u64>,
) -> Vec<MultiIndex> {
    try_lhs_multi_indices(idx, truncation, seed).unwrap_or_else(|e| panic!("{e}"))
}

/// [`lhs_multi_indices`], refusing a draw count that cannot be held:
/// over a continuous space the count comes from the spec text, and the
/// strata take one permutation of that length per axis besides the
/// draws themselves.
pub(crate) fn try_lhs_multi_indices(
    idx: &IndexFn,
    truncation: Option<u64>,
    seed: Option<u64>,
) -> Result<Vec<MultiIndex>, String> {
    let dim = index_fn_dim(idx);
    if dim == 0 {
        return Ok(Vec::new());
    }
    let total = index_fn_size(idx);
    // A continuous space has no tuple count, so the truncation is the
    // number of draws; an empty discrete input has nothing to draw.
    let n = match (truncation, total) {
        (Some(t), 0) if idx.has_continuous_axis() => t,
        (_, 0) => return Ok(Vec::new()),
        (Some(t), tot) => t.min(tot),
        (None, tot) => tot,
    };
    if n == 0 {
        return Ok(Vec::new());
    }

    let axis_sizes = axis_sizes_for(idx, dim);

    let mut rng = Prng::new(seed.unwrap_or(SEED).wrapping_add(n));
    let mut per_axis_perms: Vec<Vec<u64>> = Vec::with_capacity(dim);
    for _ in 0..dim {
        let mut perm: Vec<u64> = crate::derive_support::try_buffer_for(n, "order lhs")?;
        perm.extend(0..n);
        rng.shuffle(&mut perm);
        per_axis_perms.push(perm);
    }

    let mut out = crate::derive_support::try_buffer_for(n, "order lhs")?;
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
    Ok(out)
}

/// The scale of a continuous code: a point of `[0, 1)` as a 53-bit
/// fraction, the encoding the runtime's sampler reads.
const UNIT_SCALE: f64 = (1u64 << 53) as f64;

fn axis_sizes_for(idx: &IndexFn, dim: usize) -> Vec<u64> {
    match idx {
        IndexFn::Lattice { axis_sizes } => axis_sizes.clone(),
        IndexFn::Modular { .. } => vec![index_fn_size(idx)],
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
