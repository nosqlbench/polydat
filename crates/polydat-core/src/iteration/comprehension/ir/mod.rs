// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! IR + compiler + interpreter — comprehension_forms.md §9.1, §9.2,
//! §9.3.
//!
//! The IR is a finite linear sequence of opcodes (the 8-op
//! set in [`Op`]) that compiles from an optimized AST and
//! executes on a stack-machine interpreter ([`interpreter`]). Any
//! other execution model, such as a stream-fusion compiler, must
//! produce the same dispense sequences (§9.2).
//!
//! ## Module layout
//!
//! - [`op`] — the 8-opcode enum + supporting parameter types.
//! - [`program`] — `#[non_exhaustive] Program` wrapper:
//!   immutable, accessible by value (§9.1).
//! - [`compile`](fn@compile) — bottom-up AST → IR walker.
//! - [`interpreter`] — stack-machine interpreter; produces a
//!   tuple stream that pulls lazily.
//! - [`bounds`] — §9.3 closed-form peak-memory checker.

pub mod bounds;
pub mod compile;
pub mod interpreter;
pub mod op;
pub mod program;

pub use bounds::{Bound, ResourceBound, check_bounds};
pub use compile::compile;
pub use interpreter::{TupleStream, interpret};
pub use op::{Op, OrderStreamingKind};
pub use program::Program;
