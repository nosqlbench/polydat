// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Polydat — a variates construction engine: one function graph,
//! compiled from Polydat source or built programmatically, run on the
//! interpreter, as closures, or as native code, with the same values
//! on every engine.
//!
//! This crate is the facade over three crates and re-exports each at
//! the paths it always had: [`polydat_grammar`] (the language),
//! [`polydat_core`] (the runtime), and [`polydat_nodes`] (the node
//! library, under [`library`] beside the nodes the core keeps). A
//! program that compiles against `polydat` sees one crate.
//!
//! # Quick start
//!
//! ## From DSL source
//!
//! The simplest way to build a kernel is from Polydat DSL source, on the
//! default engine (native code with the `jit` feature, closures without):
//!
//! ```rust
//! use polydat::dsl::compile_polydat_kernel;
//!
//! let mut kernel = compile_polydat_kernel(r#"
//!     input cycle: u64
//!     hashed := hash(cycle)
//!     user_id := mod(hashed, 1000000)
//! "#).unwrap();
//!
//! kernel.set_inputs(&[42]);
//! let user_id = kernel.pull("user_id").as_u64();
//! assert!(user_id < 1_000_000);
//! ```
//!
//! ## From the assembler API
//!
//! For programmatic construction:
//!
//! ```rust
//! use polydat::compile::assembly::{PolydatAssembler, WireRef};
//! use polydat::library::hash::Hash;
//! use polydat::library::arithmetic::Mod;
//!
//! let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
//! asm.add_node("hashed", Box::new(Hash::new()), vec![WireRef::input("cycle")]);
//! asm.add_node("user_id", Box::new(Mod::new(1_000_000)), vec![WireRef::node("hashed")]);
//! asm.add_output("user_id", WireRef::node("user_id"));
//!
//! let mut kernel = asm.compile().unwrap();
//! kernel.set_inputs(&[42]);
//! assert!(kernel.pull("user_id").as_u64() < 1_000_000);
//! ```
//!
//! (The crate runs no doctests; both examples are
//! `tests/rustdoc_examples.rs`.)
//!
//! # Documentation
//!
//! The rustdoc covers the API. The narrative documentation lives in the
//! repository under `crates/polydat/docs/`, organized by the
//! [documentation index](https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/README.md).

#![warn(missing_docs)]

pub use polydat_core::{
    Const, Engine, EnginePlan, JitMode, Kernel, KernelError, KernelProgram, Provenance,
    RESOURCE_ACCESSOR, ResourceAccessor, SlotKernel, ast, audit, binder, compile, derive_support,
    dsl, half, inventory, iteration, kernel, numeric, polydat_node, resource, resource_lookup,
    set_panic_reporting_downstream, tile, viz,
};

/// The node library: the nodes the compiler keeps
/// (`polydat_core::library`) and the node library (`polydat_nodes`),
/// side by side at the paths they always had.
pub mod library {
    pub use polydat_core::library::*;
    pub use polydat_nodes::*;
}
