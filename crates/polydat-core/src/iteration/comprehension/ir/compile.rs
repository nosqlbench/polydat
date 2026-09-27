// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! AST → IR compiler — comprehension_forms.md §9.1.
//!
//! Bottom-up tree walk: each AST node emits its children's IR
//! sequences in left-to-right order, then its own operator(s).
//! `cartesian` / `zip` / `union` use N-arity opcodes; `filter`
//! and `order(Lex, _)` use unary wrappers. The terminal
//! `Dispense` is appended at the end.
//!
//! `order(Lex, _)` compiles to `Op::OrderStreaming` (R1). A
//! non-`Lex` order compiles to one `Op::OrderMaterialize` holding
//! its input, which the interpreter evaluates as a traversal does:
//! R2 over an index-addressable input, the survivors' ranking over a
//! filter, sampling over a continuous space. No IR is emitted for
//! that input.

use crate::iteration::comprehension::ast::Comprehension;
use crate::iteration::comprehension::metadata::cycle_operands;
use crate::iteration::comprehension::strategy::{StrategyName, ZipMode};

use super::op::{Op, OrderStreamingKind};
use super::program::Program;

/// Compile an optimized AST to a `Program`. The result is
/// ready for execution by [`super::interpreter::interpret`].
///
/// Per comprehension_forms.md §9.4 and §10.6 the input AST has
/// already been validated as written (§5) and then optimized (§10).
/// Compiling an unoptimized AST is well-defined but may produce
/// catastrophic working sets (§10's motivating example).
pub fn compile(ast: &Comprehension) -> Program {
    let mut ops = Vec::new();
    emit(ast, &mut ops);
    ops.push(Op::Dispense);
    Program::new(ops)
}

/// Recursive emit walker.
fn emit(ast: &Comprehension, ops: &mut Vec<Op>) {
    match ast {
        Comprehension::Clause { name, source } => {
            ops.push(Op::PushClause {
                name: name.clone(),
                source: source.clone(),
            });
        }
        Comprehension::Cartesian { children } => {
            for child in children {
                emit(child, ops);
            }
            ops.push(Op::Cartesian { n: children.len() });
        }
        Comprehension::Zip { children, mode } => {
            for child in children {
                emit(child, ops);
            }
            let operands = match mode {
                ZipMode::Cycle => cycle_operands(
                    &children
                        .iter()
                        .map(Comprehension::metadata)
                        .collect::<Vec<_>>(),
                ),
                ZipMode::Strict | ZipMode::Truncate => Vec::new(),
            };
            ops.push(Op::Zip {
                n: children.len(),
                mode: *mode,
                operands,
            });
        }
        Comprehension::Union { children } => {
            for child in children {
                emit(child, ops);
            }
            ops.push(Op::Union { n: children.len() });
        }
        Comprehension::Filter { child, predicate } => {
            emit(child, ops);
            ops.push(Op::Filter {
                predicate: predicate.clone(),
            });
        }
        Comprehension::Order {
            child,
            strategy,
            truncation,
            seed,
        } => {
            if matches!(strategy, StrategyName::Lex) {
                emit(child, ops);
                ops.push(Op::OrderStreaming {
                    kind: OrderStreamingKind::Lex,
                    truncation: *truncation,
                });
            } else {
                ops.push(Op::OrderMaterialize {
                    strategy: *strategy,
                    truncation: *truncation,
                    seed: *seed,
                    input_index_fn: child.metadata().index_addressable,
                    input: child.clone(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iteration::comprehension::source::{LiteralValue, Source};

    fn clause(name: &str, vs: &[i64]) -> Comprehension {
        Comprehension::clause(
            name,
            Source::Literal {
                values: vs.iter().map(|n| LiteralValue::Int(*n)).collect(),
            },
        )
    }

    #[test]
    fn compile_single_clause_emits_3_opcodes() {
        let ast = clause("k", &[1, 2, 3]);
        let prog = compile(&ast);
        assert_eq!(prog.len(), 2);
        // [PUSH_CLAUSE, DISPENSE]
        assert!(matches!(prog.ops()[0], Op::PushClause { .. }));
        assert!(matches!(prog.ops()[1], Op::Dispense));
    }

    #[test]
    fn compile_cartesian_emits_children_then_combinator() {
        let ast = Comprehension::cartesian(vec![clause("a", &[1, 2]), clause("b", &[10, 20])]);
        let prog = compile(&ast);
        // [PUSH a, PUSH b, CARTESIAN(2), DISPENSE]
        assert_eq!(prog.len(), 4);
        match &prog.ops()[2] {
            Op::Cartesian { n: 2 } => {}
            other => panic!("expected Cartesian(2), got {other:?}"),
        }
    }

    #[test]
    fn compile_order_lex_emits_streaming() {
        let inner = clause("k", &[1, 2, 3]);
        let ast = Comprehension::order(inner, StrategyName::Lex, Some(2));
        let prog = compile(&ast);
        // [PUSH, ORDER_STREAMING(Lex, Some(2)), DISPENSE]
        assert!(matches!(
            prog.ops()[1],
            Op::OrderStreaming {
                kind: OrderStreamingKind::Lex,
                truncation: Some(2)
            }
        ));
    }

    #[test]
    fn compile_order_halton_emits_materialize_indexed() {
        // halton over a cartesian (index-addressable) → R2 fires.
        let cart = Comprehension::cartesian(vec![clause("a", &[1, 2, 3]), clause("b", &[10, 20])]);
        let ast = Comprehension::order(cart, StrategyName::Halton, Some(3));
        let prog = compile(&ast);
        let last_non_dispense = &prog.ops()[prog.len() - 2];
        match last_non_dispense {
            Op::OrderMaterialize {
                strategy: StrategyName::Halton,
                truncation: Some(3),
                input_index_fn: Some(_),
                ..
            } => {}
            other => panic!("expected OrderMaterialize indexed, got {other:?}"),
        }
    }

    #[test]
    fn compile_order_halton_over_filter_is_naive() {
        // Filter destroys index addressability → R2 does NOT
        // fire → the op carries no index function.
        let cart = Comprehension::cartesian(vec![clause("a", &[1, 2, 3]), clause("b", &[10, 20])]);
        let filtered = Comprehension::filter(cart, "{a} > 0");
        let ast = Comprehension::order(filtered, StrategyName::Halton, Some(2));
        let prog = compile(&ast);
        let order_op = prog
            .ops()
            .iter()
            .find(|op| matches!(op, Op::OrderMaterialize { .. }))
            .unwrap();
        assert!(matches!(
            order_op,
            Op::OrderMaterialize {
                input_index_fn: None,
                ..
            }
        ));
    }

    #[test]
    fn compile_zip_emits_zip_with_mode() {
        let ast = Comprehension::zip(
            vec![clause("x", &[1, 2, 3]), clause("y", &[10, 20, 30])],
            ZipMode::Strict,
        );
        let prog = compile(&ast);
        assert!(matches!(
            prog.ops()[2],
            Op::Zip {
                n: 2,
                mode: ZipMode::Strict,
                ..
            }
        ));
    }

    /// A cycle zip carries its operands' plan: addressable operands
    /// are indexed, and of the rest the largest streams.
    #[test]
    fn compile_zip_cycle_carries_the_operand_plan() {
        use crate::iteration::comprehension::metadata::CycleOperand;
        let ast = Comprehension::zip(
            vec![
                clause("x", &[1, 2, 3]),
                Comprehension::filter(clause("y", &[1, 2]), "{y} > 0"),
                Comprehension::filter(clause("z", &[1, 2, 3, 4]), "{z} > 0"),
            ],
            ZipMode::Cycle,
        );
        let prog = compile(&ast);
        let plan = prog.ops().iter().find_map(|op| match op {
            Op::Zip { operands, .. } => Some(operands.clone()),
            _ => None,
        });
        assert_eq!(
            plan,
            Some(vec![
                CycleOperand::Indexed,
                CycleOperand::Buffered { bound: Some(2) },
                CycleOperand::Streamed,
            ])
        );
    }

    #[test]
    fn compile_terminates_with_dispense() {
        let ast = clause("k", &[1]);
        let prog = compile(&ast);
        assert!(matches!(prog.ops().last(), Some(Op::Dispense)));
    }

    /// An order's authored seed compiles into its materialize op.
    #[test]
    fn compile_carries_the_authored_seed() {
        let ast = Comprehension::order_seeded(
            clause("k", &[1, 2, 3, 4]),
            StrategyName::Shuffle,
            Some(2),
            Some(42),
        );
        let prog = compile(&ast);
        assert!(
            prog.ops().iter().any(|op| matches!(
                op,
                Op::OrderMaterialize {
                    strategy: StrategyName::Shuffle,
                    truncation: Some(2),
                    seed: Some(42),
                    ..
                }
            )),
            "{:?}",
            prog.ops()
        );
    }
}
