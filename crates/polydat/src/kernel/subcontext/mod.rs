// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD-67 — parent-gated Polydat sub-context construction.
//!
//! This module is the typed entry point for constructing a Polydat
//! child kernel as a function of a parent kernel. It implements
//! the protocol from
//! [subcontext_construction.md](../../../docs/design/subcontext_construction.md):
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
//! ## Phase scope
//!
//! SRD-67's five phases have all landed: the typed surface
//! (Phase 1), the Rule 2 write-through rewrite in
//! [`SubcontextBuilder::finalize`] (Phase 2), the synthesiser
//! migrations (Phase 3), the `pub(crate)` seal on the legacy
//! construction primitives (Phase 4), and
//! [`SubcontextBuilder::add_result_bindings`] (Phase 5). The
//! design document is the implemented specification; "Phase N"
//! in this module's comments names the push that landed a piece,
//! not pending work.
//!
//! Cross-binding rules from SRD-67 §"Cross-binding rules" are
//! enforced at [`SubcontextBuilder::finalize`] and
//! [`ScopeKernel::spawn`]:
//!
//! * Rule 1 — import resolution at finalize.
//! * Rule 2 — export collision with `final` parent surfaces as
//!   [`ContractViolation::FinalShadow`]; collision with `shared`
//!   parent rewrites the body's `X := <expr>` into
//!   `extern X: <type>` + `__write_X := <expr>` and records a
//!   `WriteThroughBinding` on the artifact. Per-cycle eval
//!   calls [`ScopeKernel::commit_write_throughs`] to fan values
//!   through the parent's `SharedCell`.
//! * Rule 4 — coordinate routing handled by `materialize_wiring_from_outer`'s
//!   IterationExtern input-kind.
//! * Rule 5 — closure-binding economy: unused imports surface as
//!   finalize diagnostics rather than errors.
//!
//! ## Cross-crate boundary
//!
//! [`PullConsumer`] is defined as a trait so that `nbrs-runtime`'s
//! `ScopeFixture` can implement it without `polydat`
//! depending on `nbrs-runtime` (the crate dependency runs the
//! other way). Phase 1 ships a minimal trait shape — the names a
//! consumer wants to pull at cycle time — sufficient for the
//! activity-side `ScopeFixture::register_consumer` adapter to
//! absorb. The eventual seal of the SRD-32 `PullPlan` happens on
//! the activity side; the artifact only carries the requested
//! names.
//!
//! ## Walled-off invariant (SRD-67 Phase 4)
//!
//! Per SRD-67 §"Walled-off invariant", the legacy
//! cross-binding primitives `PolydatKernel::materialize_wiring_from_outer` and
//! `PolydatKernel::from_program` are `pub(crate)` after Phase 4.
//! External consumers must go through the typed surface:
//! [`SubcontextBuilder`] / [`ScopeKernel::spawn`] for child
//! construction, `instance_program` for parentless re-instancing
//! of a compiled program, `chain_kernel_under_parent` for top-
//! level kernel chaining.
//!
//! The following compile-fail doctests guard the seal — if any
//! of them starts compiling, the seal is broken and a Phase 4
//! invariant has regressed.
//!
//! `materialize_wiring_from_outer` is not public:
//!
//! ```compile_fail
//! use polydat::dsl::compile::compile_polydat;
//! let mut inner = compile_polydat("input cycle: u64\n").unwrap();
//! let outer = compile_polydat("input cycle: u64\n").unwrap();
//! inner.materialize_wiring_from_outer(&outer); // pub(crate) — must not compile
//! ```
//!
//! `PolydatKernel::from_program` is not public:
//!
//! ```compile_fail
//! use polydat::dsl::compile::compile_polydat;
//! use polydat::kernel::PolydatKernel;
//! let kernel = compile_polydat("input cycle: u64\n").unwrap();
//! let _ = PolydatKernel::from_program(kernel.program().clone()); // pub(crate)
//! ```

mod builder;
mod error;
mod kernel;
mod module;
mod name;
mod pull;
mod spec;

#[cfg(test)]
mod tests;

pub use builder::{CompileOptions, SubcontextBuilder};
pub use error::{ContractViolation, SourceContext};
pub(crate) use kernel::PolydatMatterInner;
pub use kernel::{Child, PolydatMatter, PolydatMatterBuilder, RootMarker};
pub use kernel::{ScopeKernel, SharedCellInScope};
pub use module::{BodyFragment, ScopeContract, ScopeModule};
pub use name::ChildName;
pub use pull::{NamedPullConsumer, PullConsumer, RegisteredPullConsumer};
pub use spec::{ExportClassification, ExportSpec, ImportClassification, ImportSpec};
