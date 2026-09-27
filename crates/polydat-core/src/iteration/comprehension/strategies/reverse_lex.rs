// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `ReverseLex` strategy — spec §3.6.
//!
//! Reverses the input's Lex enumeration order. Index-sampling
//! family: accepts any non-`None` IndexFn (per spec §3.6's
//! per-strategy table). Continuous input is rejected (no
//! canonical reverse over a measure).
//!
//! ## References
//!
//! - The reverse of the lexicographic / mixed-radix order of
//!   [`super::lex`] (Knuth, *TAOCP* Vol. 4A §7.2.1.1) — equivalently
//!   colex from the high end. The exact reversal is cross-checked in
//!   `tests::reverse_lex_multi_indices_2d`.

use super::{MultiIndex, Selection, Strategy, capped, index_fn_size, lex::lex_multi_indices};
use crate::iteration::comprehension::metadata::IndexFn;
use crate::iteration::comprehension::strategy::StrategyName;

/// Lexicographic order with the leftmost index varying fastest.
pub struct ReverseLex;

impl Strategy for ReverseLex {
    fn name(&self) -> StrategyName {
        StrategyName::ReverseLex
    }

    /// The reversal is of the sequence as it arrives.
    fn selects_from_shape(&self) -> bool {
        false
    }

    fn accepts_input(&self, idx: Option<&IndexFn>) -> bool {
        match idx {
            None => false,
            Some(i) => !i.has_continuous_axis(),
        }
    }

    fn has_closed_form_for(&self, idx: &IndexFn) -> bool {
        !idx.has_continuous_axis()
    }

    fn select(
        &self,
        _index_fn: &IndexFn,
        cardinality: u64,
        truncation: Option<u64>,
        _seed: Option<u64>,
    ) -> Selection {
        // The input's natural enumeration is its Lex order, so the
        // reverse is its positions from the last down, cut to the
        // truncation.
        Selection::Reverse {
            total: cardinality,
            len: capped(truncation, cardinality),
        }
    }
}

/// Reverse-Lex multi-indices over `idx`. Public to crate for
/// tests + future R2 push-down consumers.
#[allow(dead_code)] // exposed for tests and future R2 consumers
pub(crate) fn reverse_lex_multi_indices(idx: &IndexFn, truncation: Option<u64>) -> Vec<MultiIndex> {
    let total = index_fn_size(idx);
    let n = match truncation {
        Some(t) => t.min(total),
        None => total,
    };
    let lex = lex_multi_indices(idx, None);
    let mut rev: Vec<MultiIndex> = lex.into_iter().rev().collect();
    rev.truncate(n as usize);
    rev
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iteration::comprehension::strategies::{EvaluatedInput, Tuple, TupleValue};

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
    fn apply_reverses() {
        let inp = input_with(vec![tup(1), tup(2), tup(3)]);
        let out = ReverseLex.apply(&inp, None);
        assert_eq!(out, vec![tup(3), tup(2), tup(1)]);
    }

    #[test]
    fn apply_truncates_after_reverse() {
        let inp = input_with(vec![tup(1), tup(2), tup(3), tup(4)]);
        let out = ReverseLex.apply(&inp, Some(2));
        assert_eq!(out, vec![tup(4), tup(3)]);
    }

    #[test]
    fn reverse_lex_multi_indices_2d() {
        let idx = IndexFn::Lattice {
            axis_sizes: vec![2, 2],
        };
        let out = reverse_lex_multi_indices(&idx, None);
        // Lex of 2x2: (0,0), (0,1), (1,0), (1,1). Reverse:
        assert_eq!(out, vec![vec![1, 1], vec![1, 0], vec![0, 1], vec![0, 0]]);
    }

    #[test]
    fn accepts_discrete_only() {
        use crate::iteration::comprehension::cardinality::{Interval, ProductMeasure};
        assert!(ReverseLex.accepts_input(Some(&IndexFn::Lattice {
            axis_sizes: vec![3]
        })));
        assert!(!ReverseLex.accepts_input(Some(&IndexFn::Continuous {
            intervals: vec![Interval::closed(0.0, 1.0)],
            measure: ProductMeasure::Uniform,
        })));
        assert!(!ReverseLex.accepts_input(None));
    }
}
