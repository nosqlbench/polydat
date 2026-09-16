// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! # polydat-core
//!
//! The Polydat runtime: the value model, the graph compiler, the
//! execution engines, the kernels, the comprehension runtime, the node
//! macro's support surface, the nodes the compiler synthesizes itself,
//! and the numeric bodies the native lowerings share with the node
//! library.
//!
//! A program declares typed inputs and a graph of named functions; the
//! compiler produces a kernel whose named outputs are pulled on demand.
//! The same inputs always yield the same outputs, on any thread, any
//! host, and any engine, with no state carried between evaluations.
//!
//! Most programs depend on the `polydat` facade, which re-exports this
//! crate together with the node library (`polydat-nodes`) and the
//! language (`polydat-grammar`) at the paths they always had. Depend on
//! `polydat-core` directly to assemble your own node set without the
//! standard library linked, or to build a tool that needs the compiler
//! and engines alone.
//!
//! ## Quick start
//!
//! The runtime compiles any program whose functions are linked. The
//! standard functions such as `hash` live in `polydat-nodes` and
//! register at link time, so a program that calls them needs that
//! crate linked as well:
//!
//! ```rust,ignore
//! use polydat_core::dsl::compile_polydat_with;
//! use polydat_core::{Engine, Provenance};
//!
//! let mut kernel = compile_polydat_with(
//!     r#"
//!         input cycle: u64
//!         id := mod(hash(cycle), 1000)
//!     "#,
//!     Engine::Closures(Provenance::PushPull),
//! )?;
//!
//! kernel.set_inputs(&[7]);
//! assert!(kernel.pull("id").as_u64() < 1000);
//! ```
//!
//! For programmatic construction, [`compile::assembly::PolydatAssembler`]
//! wires boxed nodes by name and compiles the result the same way.
//!
//! ## Engines
//!
//! One program compiles to any of three engines and gives the same
//! values on each; the host names one with [`Engine`], and
//! [`Engine::default`] is the fastest the build has.
//!
//! - [`Engine::Interpreter`]: boxed nodes over typed value buffers,
//!   with as much of the graph fused into native cones as its
//!   [`JitMode`] allows.
//! - [`Engine::Closures`]: one generated closure per node over a flat
//!   slot buffer.
//! - [`Engine::Native`]: Cranelift machine code where a node has a
//!   lowering and the node's closure elsewhere. Needs the `jit`
//!   feature.
//!
//! [`Provenance`] chooses how much re-evaluation a changed input
//! triggers; it is an optimization and never changes a result. Every
//! engine accepts every program the interpreter accepts and drives it
//! through the one [`Kernel`] trait.
//!
//! ## Program and state
//!
//! ```text
//! inputs (u64 tuple, cursors, externs)
//!     │
//!     ▼
//! ┌──────────────────────────────────┐
//! │ KernelProgram   immutable, Arc   │  shared by every thread
//! │  nodes · wiring · outputs · consts│
//! └───────────────┬──────────────────┘
//!                 │ create_kernel()
//!                 ▼
//! ┌──────────────────────────────────┐
//! │ Kernel          one per thread   │  no locks, no shared writes
//! │  slot buffers · provenance masks │
//! └───────────────┬──────────────────┘
//!                 ▼
//!          pull("id") → Value
//! ```
//!
//! A [`KernelProgram`] is the compiled, immutable half, shared by
//! reference; a [`Kernel`] is one thread's private state over it.
//! Outputs are owned by their provenance: a value stands until an
//! input that reaches it is written.
//!
//! ## Cargo features
//!
//! - **`jit`** (default): the native engine, on Cranelift.
//! - **`vectordata`**: vector-dataset access nodes for ML/AI-oriented
//!   workloads.
//!
//! ## Modules
//!
//! - [`ast`]: the value model and node contract: [`ast::Value`],
//!   the [`ast::PolydatNode`] trait, [`ast::Port`].
//! - [`dsl`]: compiling Polydat source:
//!   [`dsl::compile_polydat_with`] for a chosen engine,
//!   [`dsl::compile_polydat_kernel`] for the default, and
//!   [`dsl::compile_polydat`] for the interpreter kernel; the node
//!   registry, factories, and compile events.
//! - [`compile`]: graph construction and the engines:
//!   [`compile::assembly`] (the assembler and adapter insertion),
//!   [`compile::fusion`], [`compile::closures`], [`compile::hybrid`]
//!   (the native engine's kernel), `compile::jit` (Cranelift lowering,
//!   feature-gated), [`compile::select`] (engine and provenance
//!   selection).
//! - [`kernel`]: the runtime: the [`Kernel`] and [`KernelProgram`]
//!   traits, the interpreter's [`kernel::PolydatProgram`] and
//!   [`kernel::PolydatState`], shared cells, scopes, subcontexts,
//!   traversal activation.
//! - [`iteration`]: comprehensions, cursors, partitions, and the
//!   coordinate algebra.
//! - [`library`]: the nodes the compiler keeps: adapters
//!   ([`library::polyfill`]), assertions, constants, identity,
//!   formatting, tile rendering ([`library::tile_render`]), and the
//!   library-internal support ([`library::support`]). The node library
//!   proper is `polydat-nodes`.
//! - [`numeric`]: the numeric bodies shared by the node library and the
//!   native lowerings.
//! - [`tile`]: Polytile at the host boundary.
//! - [`binder`], [`derive_support`], [`resource`], [`audit`]: the typed
//!   binding contracts, the `#[polydat_node]` macro's support surface,
//!   the host resource bridge, and the log sink.
//! - [`viz`]: AST and graph visualization, re-exported from the grammar.
//!
//! The narrative documentation lives in the repository under
//! `crates/polydat/docs/`, organized by the
//! [documentation index](https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/README.md);
//! the [runtime model](https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/design/runtime_model.md),
//! the [graph compiler](https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/design/graph_compiler.md),
//! and the [engines](https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/design/engines.md)
//! design documents are the ones to read first.

// Unit tests use round-number float literals (`3.14`, `1.57`,
// `2.71`, …) as arbitrary fixture data. clippy's `approx_constant`
// is a deny-by-default correctness lint that reads those as
// fat-fingered `std::f*::consts::*` — true for production code,
// noise for test data. Scope the allowance to `cfg(test)` so the
// lint still guards real code.
#![cfg_attr(test, allow(clippy::approx_constant))]
#![warn(missing_docs)]

// SRD-80 PR B.3 — let the `#[polydat_node]` macro's emitted
// `polydat::...` paths resolve when the macro is invoked from
// INSIDE the polydat crate itself (library nodes migrating to
// the macro form). External callers don't need this — they
// reference `polydat` via the regular crate-name lookup.
extern crate self as polydat;

pub mod ast;
pub mod binder;
pub mod compile;
pub mod dsl;
pub mod iteration;
pub mod kernel;
pub mod library;
pub mod numeric;
pub use polydat_grammar::viz;

/// Polytile at the host boundary (SRD 114 §5.6): build a tile from
/// template text, from structural JSON text, or from a parsed JSON
/// value, then compile it with a program via
/// [`tile::compile_polydat_with_tiles`].
pub mod tile {
    pub use crate::dsl::ast::{TileBodyKind, TileDef, TileOptions, TilePiece};
    pub use crate::dsl::compile::{compile_polydat_kernel_with_tiles, compile_polydat_with_tiles};
    pub use crate::dsl::lexer::Span;
    pub use crate::dsl::tile::{parse_template, render_template};
    pub use crate::dsl::tile_structural::{
        ENCODINGS, template_text_from_value, tile_from_json_text, tile_from_json_value,
        tile_from_text,
    };
}

// SRD-104 — dependency-inverted resource-accessor bridge. A
// type-erased trait + process-global install point by which a
// kernel node reaches a live, host-owned resource by fingerprint,
// without polydat depending on the host runtime.
pub mod resource;

// SRD-80 — proc-macro trait surface. The `polydat-derive`
// crate emits paths like `polydat::derive_support::FromValue` /
// `IntoValue` that resolve here.
pub mod derive_support;

// SRD-80 PR B.5 — `Const<T>` wrapper re-exported at crate root
// for ergonomic use in `#[polydat_node]` function signatures.
pub use derive_support::Const;

/// How much of the interpreter's graph is fused into native cones:
/// what `Engine::Interpreter` carries.
pub use compile::cone::JitMode;
/// The engine a host chooses and the one error of every constructor
/// that takes it (docs/design/engine_parity.md, step 4).
pub use compile::select::{Engine, EnginePlan, KernelError, Provenance};
/// One kernel API for every engine.
pub use kernel::{Kernel, KernelProgram};

// SRD-82 §"Panic reporting: one full render" — host runtimes with
// their own panic reporting declare it so the eval-panic hook
// prints a short notice instead of the full diagnostic.
pub use kernel::set_panic_reporting_downstream;

// SRD-80 — re-export the `#[polydat_node]` attribute so
// library callers can write `#[polydat::polydat_node]` without
// a separate `use polydat_derive::polydat_node;` line.
pub use polydat_derive::polydat_node;

// SRD-80 — re-export `inventory` so the macro's emitted
// `::polydat::inventory::submit!` path resolves at every call
// site without users having to add `inventory` to their own
// dependencies.
pub use inventory;

/// Re-exported for `#[polydat_node]`-generated Phase-2 buffer
/// casts on `half::f16`-typed wires (the generated code spells
/// `polydat::half::f16`, which `extern crate self as polydat`
/// resolves inside this crate too).
pub use half;

/// SRD-104 — the resource-accessor bridge at the crate root so the
/// host installs via `polydat::RESOURCE_ACCESSOR` and nodes resolve
/// via `polydat::resource_lookup`, without reaching a deep module
/// path (D6).
pub use resource::{RESOURCE_ACCESSOR, ResourceAccessor, resource_lookup};

/// Host-log sink bridge — the sanctioned public path for installing
/// a leveled log sink into the kernel (`set_log_fn`) and for emitting
/// through it (`warn` / `info` / …). The activity runner installs its
/// `observer::log` here so polydat's cycle-time data-source audit lines
/// land in `session.log`. This is the one public entry point for the
/// audit channel; the implementation lives under `library::support`,
/// which is library-internal and must not be reached directly.
pub use library::support::audit;
