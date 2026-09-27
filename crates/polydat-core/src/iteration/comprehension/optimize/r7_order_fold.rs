// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! R7 — order chain folding (comprehension_forms.md §7.4 O1).
//!
//! `order(order(c, s1, None), s2, t) → order(c, s2, t)` when `s2`
//! selects from its input's shape.
//!
//! A strategy that selects from the shape (`halton`, `sobol`, `lhs`,
//! `extrema`, `shells`, `diagonal`, `antidiagonal`) places each tuple
//! by its position in the index space beneath the inner order, which
//! only permutes those tuples, so the inner order has no effect and
//! the outer strategy applies its own truncation. A strategy that
//! selects from the sequence (`lex`, `reverse_lex`, `shuffle`) chooses
//! other tuples after a permutation: `lex/2` after `shuffle` is two
//! shuffled tuples, not the first two. The rule leaves such a chain
//! as written, and both orders run.
//!
//! An inner truncation keeps the rule dormant too:
//! `order(order(c, s1, Some(n)), s2, t)` picks n tuples in `s1` and
//! then orders those (O2).
//!
//! Guard:
//! - Outer is `Order` whose strategy selects from the shape.
//! - Child is `Order` with `truncation: None`.

use crate::iteration::comprehension::ast::Comprehension;
use crate::iteration::comprehension::strategies::for_name;

/// Drop an untruncated order under an order whose strategy selects
/// from the shape; `None` otherwise.
pub fn apply(ast: &Comprehension) -> Option<Comprehension> {
    let Comprehension::Order {
        child: outer_child,
        strategy: outer_strat,
        truncation: outer_trunc,
        seed: outer_seed,
    } = ast
    else {
        return None;
    };
    if !for_name(*outer_strat).selects_from_shape() {
        return None;
    }
    let Comprehension::Order {
        child: inner_child,
        truncation: None,
        ..
    } = outer_child.as_ref()
    else {
        return None;
    };
    Some(Comprehension::Order {
        child: inner_child.clone(),
        strategy: *outer_strat,
        truncation: *outer_trunc,
        seed: *outer_seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iteration::comprehension::source::{LiteralValue, Source};
    use crate::iteration::comprehension::strategy::StrategyName;

    fn clause(name: &str, vs: &[i64]) -> Comprehension {
        Comprehension::clause(
            name,
            Source::Literal {
                values: vs.iter().map(|n| LiteralValue::Int(*n)).collect(),
            },
        )
    }

    #[test]
    fn r7_folds_two_orders_when_inner_untruncated() {
        let inner = clause("k", &[1, 2, 3]);
        let o1 = Comprehension::order(inner.clone(), StrategyName::Shuffle, None);
        let o2 = Comprehension::order(o1, StrategyName::Halton, Some(2));
        let result = apply(&o2).unwrap();
        match result {
            Comprehension::Order {
                child,
                strategy: StrategyName::Halton,
                truncation: Some(2),
                ..
            } => {
                assert_eq!(&*child, &inner);
            }
            other => panic!("expected Order(Halton, Some(2)) → clause, got {other:?}"),
        }
    }

    #[test]
    fn r7_does_not_fire_when_inner_truncated() {
        let inner = clause("k", &[1, 2, 3]);
        let o1 = Comprehension::order(inner, StrategyName::Shuffle, Some(2));
        let o2 = Comprehension::order(o1, StrategyName::Halton, Some(1));
        // O2 — meaningful two-stage composition.
        assert_eq!(apply(&o2), None);
    }

    #[test]
    fn r7_does_not_fire_for_single_order() {
        let ast = Comprehension::order(clause("k", &[1]), StrategyName::Lex, None);
        assert_eq!(apply(&ast), None);
    }

    /// The fold keeps the outer order's seed: it is the outer order
    /// that survives.
    #[test]
    fn fold_keeps_the_outer_seed() {
        let inner = clause("k", &[1, 2, 3]);
        let o1 = Comprehension::order_seeded(inner.clone(), StrategyName::Shuffle, None, Some(1));
        let o2 = Comprehension::order_seeded(o1, StrategyName::Lhs, Some(2), Some(42));
        let result = apply(&o2).unwrap();
        assert_eq!(
            result,
            Comprehension::order_seeded(inner, StrategyName::Lhs, Some(2), Some(42))
        );
    }

    /// An outer strategy that selects from the sequence keeps the inner
    /// order: `lex/2` after `shuffle` is two shuffled tuples.
    #[test]
    fn a_sequence_strategy_keeps_the_inner_order() {
        for outer in [
            StrategyName::Lex,
            StrategyName::ReverseLex,
            StrategyName::Shuffle,
        ] {
            let o1 = Comprehension::order(clause("k", &[1, 2, 3]), StrategyName::Shuffle, None);
            let o2 = Comprehension::order(o1, outer, Some(2));
            assert_eq!(apply(&o2), None, "{outer:?}");
        }
    }
}
