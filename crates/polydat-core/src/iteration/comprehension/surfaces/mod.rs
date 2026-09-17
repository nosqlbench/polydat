// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Consumption surfaces — spec §9.5.
//!
//! Three independent first-class consumption surfaces over a
//! shared compiled IR:
//!
//! - [`CoordinateStream`] (first-order) — dispenses coordinate
//!   tuples (`Vec<(String, TupleValue)>`).
//! - [`ScopedKernelStream<K>`] (second-order) — dispenses
//!   scoped kernel instances; functor over the first-order
//!   via `K`'s `KernelScope` impl.
//! - [`scope_once`](fn@scope_once) (one-shot) — non-streamed; takes a single
//!   coord tuple and produces a single scoped kernel instance.
//!
//! All three surfaces share the underlying `Program` via
//! `Arc<Program>` but maintain independent dispense state per
//! spec §9.5.2's independence contract:
//!
//! > Each call to `coordinate_stream` or
//! > `scoped_kernel_stream` returns a fresh streamer with its
//! > own dispense cursor. The streamers share the underlying
//! > compiled IR but allocate their own per-streamer state.
//!
//! The entry point is [`CompiledComprehension`], obtained via
//! [`compile`]`(&ast)` or `CompiledComprehension::from_ast`.

use super::ast::Comprehension;

pub mod compiled;
pub mod coord_stream;
pub mod instance;
pub mod polydat_kernel;
pub mod scope_once;
pub mod scoped_stream;

pub use compiled::CompiledComprehension;
pub use coord_stream::CoordinateStream;
pub use instance::{KernelScope, ScopedKernelInstance};
pub use polydat_kernel::{
    PolydatKernelScope, polydat_value_to_tuple_value, tuple_value_to_polydat_value,
};
pub use scope_once::scope_once;
pub use scoped_stream::ScopedKernelStream;

/// Compile an AST into a [`CompiledComprehension`] ready to
/// dispense: the §10 optimizer runs first, then the AST → IR pass.
/// Equivalent to `CompiledComprehension::from_ast(ast)`.
pub fn compile(ast: &Comprehension) -> CompiledComprehension {
    CompiledComprehension::from_ast(ast)
}
