// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Parent-gated Polydat sub-context construction.
//!
//! This module is the typed entry point for constructing a Polydat
//! child kernel as a function of a parent kernel. It implements
//! the protocol from
//! `crates/polydat/docs/design/subcontext_construction.md`:
//!
//! 1. Parent yields a builder via [`ScopeKernel::subcontext_builder`].
//! 2. Builder accumulates module matter (imports, exports, body
//!    fragments, pull consumers) via [`SubcontextBuilder`].
//! 3. `finalize` closes the builder, validates imports against the
//!    parent's exports, compiles the body into a [`ScopeModule`] with
//!    a typed [`ScopeContract`].
//! 4. Parent spawns the child via [`ScopeKernel::spawn`] — the single
//!    chokepoint where every cross-binding is resolved.
//!
//! ## Cross-binding rules
//!
//! The comments in this module name the cross-binding rules by
//! number. They are enforced at [`SubcontextBuilder::finalize`] and
//! [`ScopeKernel::spawn`]:
//!
//! * Rule 1 — import resolution at finalize
//!   (subcontext_construction.md §2.2, SC4).
//! * Rule 2 — export collision with a `const` parent output surfaces
//!   as [`ContractViolation::FinalShadow`]; collision with a `shared`
//!   cell visible at the parent rewrites the body's `X := <expr>` into
//!   `extern X: <type>` + `__write_X := <expr>` and records a
//!   `WriteThroughBinding` on the artifact (§3.1). Per-cycle eval
//!   calls [`ScopeKernel::commit_write_throughs`] to fan values
//!   through the parent's `SharedCell` (§5).
//! * Rule 3 — the child is initialized after binding, so its consts
//!   see the bound inputs (§4, §9).
//! * Rule 4 — coordinate routing handled by `materialize_wiring_from_outer`'s
//!   IterationExtern input-kind.
//! * Rule 5 — closure-binding economy: magic externs are injected only
//!   when referenced (§3.2), and unused imports surface as finalize
//!   diagnostics rather than errors (§2.2).
//!
//! ## Cross-crate boundary
//!
//! [`PullConsumer`] is defined as a trait so that `nbrs-runtime`'s
//! `ScopeFixture` can implement it without `polydat-core`
//! depending on `nbrs-runtime` (the crate dependency runs the
//! other way). The trait carries a minimal shape — the names a
//! consumer wants to pull at cycle time — sufficient for the
//! activity-side `ScopeFixture::register_consumer` adapter to
//! absorb. Sealing the host's pull plan happens on the activity
//! side; the artifact only carries the requested names
//! (subcontext_construction.md §2.4).
//!
//! ## Walled-off invariant
//!
//! The low-level cross-binding primitives are sealed
//! (subcontext_construction.md §8, SC1): `PolydatKernel::from_program` is
//! `pub(crate)` and `materialize_wiring_from_outer` is private to the kernel.
//! External consumers must go through the typed surface:
//! [`SubcontextBuilder`] / [`ScopeKernel::spawn`] or [`PolydatMatter::build_under`]
//! for child construction, and `Construction::root` / `KernelProgram::create_kernel`
//! for parentless re-instancing of a compiled program.
//!
//! The compile-fail cases under the facade crate's `crates/polydat/tests/ui/seal/` guard the seal: one
//! calls `materialize_wiring_from_outer` from outside the crate, one
//! calls `PolydatKernel::from_program`, and neither may compile. If
//! either starts compiling, the seal is broken.

mod builder;
mod error;
mod kernel;
mod module;
mod name;
mod pull;
mod spec;

#[cfg(test)]
mod tests;

pub use builder::{CompileOptions, ParentView, SubcontextBuilder};
pub use error::{ContractViolation, SourceContext};
pub(crate) use kernel::PolydatMatterInner;
pub use kernel::{Child, PolydatMatter, PolydatMatterBuilder, RootMarker};
pub use kernel::{ScopeKernel, SharedCellInScope, wrap_root_kernel};
pub use module::{BodyFragment, ScopeContract, ScopeModule};
pub use name::ChildName;
pub use pull::{NamedPullConsumer, PullConsumer, RegisteredPullConsumer};
pub use spec::{ExportClassification, ExportSpec, ImportClassification, ImportSpec};
