// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Operator IR — comprehension_forms.md §9.1.
//!
//! Every well-formed comprehension AST compiles to a finite
//! sequence of these 8 opcodes. Every operator is a stream
//! transducer; operands flow as tuple streams via
//! `advance() -> Option<Tuple>`, never as materialized
//! `Vec<Tuple>`. The two materialization barriers — non-Lex
//! `ORDER_MATERIALIZE` and `ZIP(Cycle)`'s buffering of operands
//! that are not index-addressable — are called out explicitly.

use serde::{Deserialize, Serialize};

use crate::iteration::comprehension::ast::Comprehension;
use crate::iteration::comprehension::metadata::{CycleOperand, cycle_plan_is_empty};
use crate::iteration::comprehension::source::Source;
use crate::iteration::comprehension::strategy::{StrategyName, ZipMode};

/// The 8-opcode IR set (comprehension_forms.md §9.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    /// Push a single-name tuple stream produced by `source`.
    /// Streaming; O(1) per pull above the source's own state.
    PushClause {
        /// The name the stream binds.
        name: String,
        /// Where its values come from.
        source: Source,
    },

    /// Replace the top-N stream operands with one stream that
    /// enumerates their cross product in Lex order. Streaming.
    Cartesian {
        /// Operands combined.
        n: usize,
    },

    /// Replace the top-N stream operands with their lockstep
    /// diagonal. Streaming under Strict/Truncate; under `Cycle`
    /// each operand is held as `operands` plans
    /// (comprehension_forms.md §3.3, §6.2):
    /// indexed, streamed, or buffered.
    Zip {
        /// Operands combined.
        n: usize,
        /// The length policy.
        mode: ZipMode,
        /// Under `Cycle`, how each operand is held, from the
        /// operands' metadata ([`crate::iteration::comprehension::metadata::cycle_operands`]).
        /// Empty under Strict and Truncate, and for a program built
        /// without one, whose operands the interpreter plans from the
        /// streams themselves.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        operands: Vec<CycleOperand>,
    },

    /// Replace the top-N stream operands with a stream that
    /// concatenates them in operand order. Streaming.
    Union {
        /// Operands concatenated.
        n: usize,
    },

    /// Wrap the top operand with a per-tuple predicate check.
    /// Streaming.
    Filter {
        /// The predicate, a boolean expression over the tuple.
        predicate: String,
    },

    /// Wrap the top operand with a counter / pass-through.
    /// Used for `order(Lex, _)` per comprehension_forms.md §10.2 R1; this is
    /// the "streaming" order opcode.
    OrderStreaming {
        /// The streaming order's kind.
        kind: OrderStreamingKind,
        /// The output cap, if any.
        truncation: Option<u64>,
    },

    /// MATERIALIZATION BARRIER. Push a stream of the tuples a
    /// non-`Lex` order selects from `input`. Truncation is the output
    /// cap.
    ///
    /// The order is evaluated as a traversal evaluates it
    /// ([`crate::iteration::comprehension::runtime::evaluate_indexed`]),
    /// on the first pull, in the empty scope: the strategy selects
    /// positions from the input's evaluated shape and length, and
    /// the stream computes each selected tuple as it emits it. Per
    /// comprehension_forms.md §10.2 R2, over an index-addressable
    /// input the working set is the selection, O(output); over a
    /// filter it is the filter's input and the positions of the
    /// survivors the strategy keeps (§5 V5); over any other input it
    /// is the input's tuples. An order over a continuous axis samples
    /// the input's space and holds the samples. No IR is emitted for
    /// `input`: the op pops nothing and pushes one stream.
    ///
    /// `input_index_fn` is the input's addressing scheme as the
    /// metadata propagator claims it at compile time (§10.7.6),
    /// which the bounds checker reads; `None` when it claims none.
    OrderMaterialize {
        /// The strategy applied.
        strategy: StrategyName,
        /// The output cap, if any.
        truncation: Option<u64>,
        /// The authored seed a seeded strategy (`Shuffle`, `Lhs`)
        /// derives its state from; its fixed default when `None`.
        seed: Option<u64>,
        /// The input's IndexFn from its metadata at compile time.
        input_index_fn: Option<crate::iteration::comprehension::metadata::IndexFn>,
        /// The comprehension ordered.
        input: Box<Comprehension>,
    },

    /// Bind the top stream as the comprehension's result.
    /// Must be the last opcode in a well-formed Program.
    Dispense,
}

/// Variant marker for [`Op::OrderStreaming`]. Lex is the one
/// streaming strategy (comprehension_forms.md §6.2's footprint
/// table); the kind is an enum so the opcode set stays fixed
/// whatever strategies stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderStreamingKind {
    /// Lexicographic order: the natural enumeration, counted.
    Lex,
}

impl Op {
    /// Arity for stack-effect computation: how many stream
    /// operands this opcode pops, and how many it pushes.
    /// Always pushes 1 for stream-producing ops; `Dispense`
    /// pushes 0 (it consumes the final stream).
    pub fn stack_effect(&self) -> (usize, usize) {
        match self {
            Op::PushClause { .. } => (0, 1),
            Op::Cartesian { n } => (*n, 1),
            Op::Zip { n, .. } => (*n, 1),
            Op::Union { n } => (*n, 1),
            Op::Filter { .. } => (1, 1),
            Op::OrderStreaming { .. } => (1, 1),
            Op::OrderMaterialize { .. } => (0, 1),
            Op::Dispense => (1, 0),
        }
    }

    /// `true` if this opcode is a materialization barrier per
    /// comprehension_forms.md §6.2 and §6.3. Used by the bounds checker. A `Cycle` zip
    /// is one when it buffers an operand, or when it carries no plan
    /// and may have to; one with an operand known empty holds nothing.
    pub fn is_barrier(&self) -> bool {
        match self {
            Op::OrderMaterialize { .. } => true,
            Op::Zip {
                mode: ZipMode::Cycle,
                operands,
                ..
            } => {
                operands.is_empty()
                    || (!cycle_plan_is_empty(operands)
                        && operands
                            .iter()
                            .any(|o| matches!(o, CycleOperand::Buffered { .. })))
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_effect_basics() {
        assert_eq!(
            Op::PushClause {
                name: "k".into(),
                source: Source::Literal { values: vec![] },
            }
            .stack_effect(),
            (0, 1)
        );
        assert_eq!(Op::Cartesian { n: 3 }.stack_effect(), (3, 1));
        assert_eq!(Op::Dispense.stack_effect(), (1, 0));
    }

    fn input() -> Box<Comprehension> {
        Box::new(Comprehension::clause(
            "k",
            Source::IntRange {
                lo: 0,
                hi: 50,
                step: 1,
            },
        ))
    }

    #[test]
    fn barrier_classification() {
        assert!(
            Op::OrderMaterialize {
                strategy: StrategyName::Halton,
                truncation: Some(10),
                seed: None,
                input_index_fn: None,
                input: input(),
            }
            .is_barrier()
        );
        assert!(
            Op::Zip {
                n: 2,
                mode: ZipMode::Cycle,
                operands: vec![
                    CycleOperand::Streamed,
                    CycleOperand::Buffered { bound: Some(3) }
                ],
            }
            .is_barrier()
        );
        assert!(
            !Op::Zip {
                n: 2,
                mode: ZipMode::Cycle,
                operands: vec![CycleOperand::Indexed, CycleOperand::Streamed],
            }
            .is_barrier(),
            "indexing and streaming buffer nothing"
        );
        assert!(
            !Op::Zip {
                n: 2,
                mode: ZipMode::Strict,
                operands: Vec::new(),
            }
            .is_barrier()
        );
        assert!(
            !Op::OrderStreaming {
                kind: OrderStreamingKind::Lex,
                truncation: None,
            }
            .is_barrier()
        );
    }

    #[test]
    fn serde_round_trip() {
        let op = Op::OrderMaterialize {
            strategy: StrategyName::Halton,
            truncation: Some(50),
            seed: None,
            input_index_fn: Some(
                crate::iteration::comprehension::metadata::IndexFn::Lattice {
                    axis_sizes: vec![10, 5],
                },
            ),
            input: input(),
        };
        let json = serde_json::to_string(&op).unwrap();
        let back: Op = serde_json::from_str(&json).unwrap();
        assert_eq!(op, back);
    }
}
