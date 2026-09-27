// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `Shuffle` strategy — spec §3.6.
//!
//! Random permutation. PRNG seed captured at materialization
//! per spec §3.6 — same comprehension instance produces the
//! same shuffle on every dispense pass. Spec §9.5.2's
//! independence contract has different `CoordinateStream`
//! instances against the same comprehension get independent
//! shuffles via a per-streamer seed.
//!
//! The seed here is a module constant plus the input length;
//! per-streamer seeding is not implemented.
//!
//! ## References
//!
//! - R. A. Fisher & F. Yates, *Statistical Tables for Biological,
//!   Agricultural and Medical Research*, 3rd ed. (1948), the original
//!   shuffle. The in-place O(n) form (used via [`super::prng::Prng::shuffle`])
//!   is R. Durstenfeld, "Algorithm 235: Random permutation,"
//!   *Comm. ACM* 7(7) (1964), 420.
//!   doi:[10.1145/364520.364540](https://doi.org/10.1145/364520.364540);
//!   see also Knuth, *TAOCP* Vol. 2 §3.4.2 (Algorithm P). Correctness
//!   = the output is a *permutation* of the input (each element
//!   exactly once), verified in `tests::apply_preserves_elements`.
//!
//! Accepts any non-`None` `IndexFn` including continuous: over a
//! continuous or hybrid space it draws `n` codes, one 53-bit unit
//! fraction per continuous axis, and the runtime's sampler carries
//! each onto its axis's measure (spec §3.6, §10.2 R2).

use super::{
    MultiIndex, Selection, Strategy, index_fn_dim, index_fn_size, index_fn_supports_lookup,
    prng::Prng,
};
use crate::iteration::comprehension::metadata::IndexFn;
use crate::iteration::comprehension::strategy::StrategyName;

/// A seeded permutation.
pub struct Shuffle;

/// Seed base when none is authored; the input length is added per
/// call. Per-streamer
/// seeding is not implemented.
const DEFAULT_SEED: u64 = 0xD1CE_5EED_C0FF_EE42;

impl Strategy for Shuffle {
    fn name(&self) -> StrategyName {
        StrategyName::Shuffle
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
            let mis = shuffle_multi_indices(index_fn, truncation, seed);
            Selection::from_multi_indices(index_fn, mis, cardinality)
        } else {
            Selection::Positions(naive_shuffle_positions(cardinality, truncation, seed))
        }
    }
}

/// A seeded permutation of `0..total` positions, as one axis, cut to
/// the truncation.
fn naive_shuffle_positions(total: u64, truncation: Option<u64>, seed: Option<u64>) -> Vec<u64> {
    let mut rng = Prng::new(seed.unwrap_or(DEFAULT_SEED).wrapping_add(total));
    let mut positions: Vec<u64> = (0..total).collect();
    rng.shuffle(&mut positions);
    if let Some(n) = truncation {
        positions.truncate(usize::try_from(n).unwrap_or(usize::MAX));
    }
    positions
}

/// The multi-indices of a shuffle over `idx`, `truncation` of them,
/// from the authored `seed` or the default.
pub(crate) fn shuffle_multi_indices(
    idx: &IndexFn,
    truncation: Option<u64>,
    seed: Option<u64>,
) -> Vec<MultiIndex> {
    try_shuffle_multi_indices(idx, truncation, seed).unwrap_or_else(|e| panic!("{e}"))
}

/// [`shuffle_multi_indices`], refusing a draw count that cannot be
/// held. Over a continuous space the count is the order's own, from
/// the spec text; over a discrete one it is bounded by the tuples the
/// input already holds.
pub(crate) fn try_shuffle_multi_indices(
    idx: &IndexFn,
    truncation: Option<u64>,
    seed: Option<u64>,
) -> Result<Vec<MultiIndex>, String> {
    let total = index_fn_size(idx);
    let continuous = matches!(idx, IndexFn::Continuous { .. } | IndexFn::Hybrid { .. });
    // A continuous space has no tuple count: the truncation is the
    // number of draws.
    let n = match (truncation, continuous) {
        (Some(t), true) => t,
        (Some(t), false) => t.min(total),
        (None, true) => return Ok(Vec::new()),
        (None, false) => total,
    };
    if n == 0 {
        return Ok(Vec::new());
    }

    let dim = index_fn_dim(idx);
    let axis_sizes = axis_sizes_for(idx);
    // The seed follows the draw count over a continuous space, which
    // has no tuple count of its own.
    let base = seed.unwrap_or(DEFAULT_SEED);
    let mut rng = Prng::new(base.wrapping_add(if continuous { n } else { total }));

    Ok(match idx {
        IndexFn::Continuous { intervals, .. } => {
            let _ = intervals;
            let mut out = crate::derive_support::try_buffer_for(n, "order shuffle")?;
            out.extend((0..n).map(|_| (0..dim).map(|_| rng.next_u64() >> 11).collect()));
            out
        }
        IndexFn::Hybrid {
            discrete_axes,
            continuous_axes,
            ..
        } => {
            let _ = continuous_axes;
            let mut out = crate::derive_support::try_buffer_for(n, "order shuffle")?;
            out.extend((0..n).map(|_| {
                let mut mi = Vec::with_capacity(dim);
                for size in discrete_axes {
                    mi.push(rng.next_bounded(*size));
                }
                for _ in 0..continuous_axes.len() {
                    mi.push(rng.next_u64() >> 11);
                }
                mi
            }));
            out
        }
        _ => {
            if n == total {
                let mut indices: Vec<u64> = (0..total).collect();
                rng.shuffle(&mut indices);
                indices
                    .into_iter()
                    .map(|i| linear_to_multi(i, &axis_sizes))
                    .collect()
            } else {
                // A partial Fisher–Yates over the pool `0..total`, held
                // sparsely: only the slots a draw has displaced are
                // stored, so `n` draws hold at most `n` entries whatever
                // the input's size, and draw what the dense pool draws.
                let mut displaced: std::collections::HashMap<u64, u64> =
                    std::collections::HashMap::new();
                let mut out = Vec::with_capacity(n as usize);
                for i in 0..n {
                    let last = total - i - 1;
                    let j = rng.next_bounded(total - i);
                    let pick = displaced.get(&j).copied().unwrap_or(j);
                    out.push(linear_to_multi(pick, &axis_sizes));
                    let tail = displaced.remove(&last).unwrap_or(last);
                    if j != last {
                        displaced.insert(j, tail);
                    }
                }
                out
            }
        }
    })
}

fn axis_sizes_for(idx: &IndexFn) -> Vec<u64> {
    match idx {
        IndexFn::Lattice { axis_sizes } => axis_sizes.clone(),
        IndexFn::Modular { .. } => vec![index_fn_size(idx)],
        IndexFn::Lockstep { length } => vec![*length],
        IndexFn::Concatenation { segment_sizes } => vec![segment_sizes.iter().sum()],
        IndexFn::Continuous { .. } | IndexFn::Hybrid { .. } => Vec::new(),
    }
}

fn linear_to_multi(mut linear: u64, axis_sizes: &[u64]) -> MultiIndex {
    let mut out = vec![0u64; axis_sizes.len()];
    for i in (0..axis_sizes.len()).rev() {
        out[i] = linear % axis_sizes[i];
        linear /= axis_sizes[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iteration::comprehension::strategies::{EvaluatedInput, Tuple, TupleValue};

    /// The sparse partial shuffle draws what a dense pool of every
    /// position draws, step for step.
    #[test]
    fn partial_shuffle_matches_the_dense_pool() {
        for (total, n, seed) in [(10u64, 3u64, 1u64), (97, 40, 7), (1000, 999, 3), (5, 1, 9)] {
            let idx = IndexFn::Lattice {
                axis_sizes: vec![total],
            };
            let sparse: Vec<u64> = shuffle_multi_indices(&idx, Some(n), Some(seed))
                .into_iter()
                .map(|mi| mi[0])
                .collect();
            let mut rng = Prng::new(seed.wrapping_add(total));
            let mut pool: Vec<u64> = (0..total).collect();
            let mut dense = Vec::new();
            for i in 0..n {
                let j = rng.next_bounded(total - i) as usize;
                dense.push(pool[j]);
                let last = pool.len() - 1;
                pool.swap(j, last);
                pool.pop();
            }
            assert_eq!(sparse, dense, "total {total}, n {n}, seed {seed}");
        }
    }

    fn tup(k: i64) -> Tuple {
        Tuple::new().with("k", TupleValue::I64(k))
    }

    fn input_with(tuples: Vec<Tuple>) -> EvaluatedInput {
        let n = tuples.len() as u64;
        EvaluatedInput {
            tuples,
            cardinality: n,
            index_fn: IndexFn::Lattice {
                axis_sizes: vec![n],
            },
        }
    }

    #[test]
    fn apply_preserves_elements() {
        let inp = input_with(vec![tup(1), tup(2), tup(3), tup(4), tup(5)]);
        let mut out = Shuffle.apply(&inp, None);
        let mut sorted_in = inp.tuples.clone();
        out.sort_by_key(|t| match t.bindings[0].1 {
            TupleValue::I64(v) => v,
            _ => panic!(),
        });
        sorted_in.sort_by_key(|t| match t.bindings[0].1 {
            TupleValue::I64(v) => v,
            _ => panic!(),
        });
        assert_eq!(out, sorted_in);
    }

    #[test]
    fn apply_deterministic() {
        let inp = input_with(vec![tup(1), tup(2), tup(3), tup(4), tup(5)]);
        let a = Shuffle.apply(&inp, None);
        let b = Shuffle.apply(&inp, None);
        assert_eq!(a, b);
    }

    #[test]
    fn shuffle_multi_indices_produces_unique_discrete() {
        let idx = IndexFn::Lattice {
            axis_sizes: vec![3, 4],
        };
        let out = shuffle_multi_indices(&idx, Some(10), None);
        assert_eq!(out.len(), 10);
        let mut seen = std::collections::HashSet::new();
        for mi in &out {
            assert!(seen.insert(mi.clone()), "duplicate: {mi:?}");
        }
        for mi in &out {
            assert!(mi[0] < 3);
            assert!(mi[1] < 4);
        }
    }

    #[test]
    fn shuffle_multi_indices_full_lattice() {
        let idx = IndexFn::Lattice {
            axis_sizes: vec![2, 2],
        };
        let out = shuffle_multi_indices(&idx, None, None);
        assert_eq!(out.len(), 4);
        let mut sorted = out.clone();
        sorted.sort();
        assert_eq!(sorted, vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);
    }

    #[test]
    fn linear_to_multi_round_trip() {
        let sizes = vec![3u64, 4, 5];
        for linear in 0..60u64 {
            let mi = linear_to_multi(linear, &sizes);
            let mut back = 0u64;
            for (s, m) in sizes.iter().zip(mi.iter()) {
                back = back * s + m;
            }
            assert_eq!(back, linear);
        }
    }

    #[test]
    fn accepts_any_non_none() {
        assert!(Shuffle.accepts_input(Some(&IndexFn::Lattice {
            axis_sizes: vec![3]
        })));
        assert!(!Shuffle.accepts_input(None));
    }

    /// Over a continuous or hybrid space the truncation is the number
    /// of draws (spec §3.6: "n PRNG draws from the measure"), and a
    /// continuous code is a 53-bit fraction of the unit interval.
    #[test]
    fn continuous_draws_are_counted_by_the_truncation() {
        use crate::iteration::comprehension::cardinality::{Interval, ProductMeasure};
        let idx = IndexFn::Continuous {
            intervals: vec![Interval::closed(2.0, 4.0)],
            measure: ProductMeasure::Uniform,
        };
        let out = shuffle_multi_indices(&idx, Some(16), None);
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|mi| mi[0] < (1u64 << 53)), "{out:?}");
        assert!(shuffle_multi_indices(&idx, None, None).is_empty());
        let idx = IndexFn::Hybrid {
            discrete_axes: vec![3],
            continuous_axes: vec![Interval::closed(0.0, 1.0)],
            measure: ProductMeasure::Uniform,
        };
        let out = shuffle_multi_indices(&idx, Some(5), None);
        assert_eq!(out.len(), 5);
        assert!(
            out.iter().all(|mi| mi[0] < 3 && mi[1] < (1u64 << 53)),
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
        let default = shuffle_multi_indices(&idx, Some(12), None);
        let seeded = shuffle_multi_indices(&idx, Some(12), Some(42));
        let again = shuffle_multi_indices(&idx, Some(12), Some(42));
        assert_eq!(seeded, again);
        assert_ne!(seeded, default);
        assert_ne!(seeded, shuffle_multi_indices(&idx, Some(12), Some(43)));
    }
}
