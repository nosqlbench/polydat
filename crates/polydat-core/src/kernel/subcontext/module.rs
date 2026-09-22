// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! [`ScopeModule<M>`] — closed, immutable module-matter artifact.
//!
//! Per SRD-67 §"Step 3 — Artifact is a closed value": the
//! artifact carries everything the parent needs to spawn — type
//! contracts, the compiled program, registered pull consumers —
//! but holds no live reference to the parent. The artifact can
//! be moved, hashed, debug-printed, and (per Phase 4) cached for
//! reuse.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use crate::dsl::ast::Statement;
use crate::kernel::PolydatProgram;

use super::error::SourceContext;
use super::pull::RegisteredPullConsumer;
use super::spec::{ExportSpec, ImportSpec};

/// Body fragment — what the builder accepts via
/// [`super::SubcontextBuilder::body`].
///
/// Per SRD-67 §"Decision 4". `PolydatSource` is for user-facing
/// `bindings:` / `result:` content (parsed at finalize);
/// `Statements` is for synthesisers that already produce GK
/// programmatically.
#[derive(Debug, Clone)]
pub enum BodyFragment {
    /// User-facing Polydat source. Parsed via the existing
    /// `lexer + parser` pipeline at finalize.
    PolydatSource(String),
    /// Pre-parsed statements — submitted directly without
    /// round-tripping through Polydat source strings. Reuses
    /// [`Statement`] from the existing AST, so synthesisers
    /// don't carry a parallel enum.
    Statements(Vec<Statement>),
}

/// Typed handle bundle (per SRD-13e §1.2).
///
/// The bundle is minimal by design: a handle for each declared
/// import / export, identified by name. Slot resolution happens
/// against the compiled program (`find_input` /
/// `output_map_lookup`) rather than through cached indices here.
///
/// `M` is the module-identity phantom — [`super::Child<P>`] for
/// modules built under parent `P`. Handles issued by one module
/// can't be applied to a sibling at the type level.
pub struct ScopeContract<M> {
    imports: Vec<ImportHandle<M>>,
    exports: Vec<ExportHandle<M>>,
    _module: PhantomData<fn() -> M>,
}

impl<M> ScopeContract<M> {
    pub(crate) fn from_specs(imports: &[ImportSpec], exports: &[ExportSpec]) -> Self {
        Self {
            imports: imports
                .iter()
                .map(|s| ImportHandle {
                    name: s.name.clone(),
                    _module: PhantomData,
                })
                .collect(),
            exports: exports
                .iter()
                .map(|s| ExportHandle {
                    name: s.name.clone(),
                    _module: PhantomData,
                })
                .collect(),
            _module: PhantomData,
        }
    }

    /// The imports the module declares, in declaration order.
    pub fn imports(&self) -> &[ImportHandle<M>] {
        &self.imports
    }

    /// The exports the module declares, in declaration order.
    pub fn exports(&self) -> &[ExportHandle<M>] {
        &self.exports
    }
}

impl<M> std::fmt::Debug for ScopeContract<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopeContract")
            .field("imports", &self.imports)
            .field("exports", &self.exports)
            .finish()
    }
}

/// A typed handle to a named import slot. Brand `M` ties it to
/// the module that issued it.
pub struct ImportHandle<M> {
    name: String,
    _module: PhantomData<fn() -> M>,
}

impl<M> ImportHandle<M> {
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<M> std::fmt::Debug for ImportHandle<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ImportHandle").field(&self.name).finish()
    }
}

/// A typed handle to a named export slot. Brand `M` ties it to
/// the module that issued it.
pub struct ExportHandle<M> {
    name: String,
    _module: PhantomData<fn() -> M>,
}

impl<M> ExportHandle<M> {
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<M> std::fmt::Debug for ExportHandle<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ExportHandle").field(&self.name).finish()
    }
}

/// A Rule 2 write-through binding produced by the builder when a
/// child's `X := <expr>` collides with a parent `shared X`
/// export. The child's compiled program produces a synthetic
/// output named [`Self::source_output`] (typically
/// `__write_<X>`); spawn wires the parent's `SharedCell` into
/// the child's `X` input slot via `materialize_wiring_from_outer`. After
/// per-cycle eval, [`super::ScopeKernel::commit_write_throughs`]
/// pulls the synthetic output and stores its value through the
/// child's input slot, which propagates to the cell.
#[derive(Debug, Clone)]
pub struct WriteThroughBinding {
    /// The name as declared on the parent (and as the child sees
    /// it via `extern`). The parent's shared cell is keyed on
    /// this name.
    pub export_name: String,
    /// The synthetic output the rewrite emits in the child
    /// program — its `pull` produces the value to write through.
    pub source_output: String,
}

/// Closed, immutable module-matter artifact.
///
/// Produced by [`super::SubcontextBuilder::finalize`]; consumed
/// by [`super::ScopeKernel::spawn`]. The artifact carries no
/// live reference to its parent — it can be stored, hashed,
/// inspected, or moved freely.
pub struct ScopeModule<M> {
    pub(crate) imports: Vec<ImportSpec>,
    pub(crate) exports: Vec<ExportSpec>,
    pub(crate) program: Arc<PolydatProgram>,
    /// The body after the Rule 2 rewrite, and the settings it was
    /// compiled under: what [`ScopeModule::program_on`] needs to build
    /// the same body for another engine.
    pub(crate) statements: Vec<Statement>,
    pub(crate) options: crate::dsl::compile::CompileOptions,
    /// This module's program per engine, built on the first ask for
    /// that engine and shared by every instance after it — the same
    /// carrier a `for` body and a tile's projection body use
    /// (`dsl::traversal::BodySource`).
    ///
    /// The interpreter's is the one `finalize` already built, so it is
    /// in the table from the start. A module instantiated a thousand
    /// times compiles once per engine and allocates a state per
    /// instance.
    pub(crate) programs: Mutex<HashMap<crate::Engine, Arc<dyn crate::kernel::KernelProgram>>>,
    pub(crate) contract: ScopeContract<M>,
    pub(crate) context: SourceContext,
    pub(crate) consumers: Vec<RegisteredPullConsumer>,
    /// Rule 2 write-through bindings — child exports rewritten
    /// at finalize to feed parent `shared` cells. Empty for the
    /// vast majority of modules; populated only when the parent
    /// has a `shared` export with a name the child redefines.
    pub(crate) write_throughs: Vec<WriteThroughBinding>,
    /// Diagnostics emitted during finalize (warnings, etc.). Free-
    /// form strings; downstream tooling can surface them.
    pub(crate) diagnostics: Vec<String>,
    pub(crate) _module: PhantomData<fn() -> M>,
}

impl<M> ScopeModule<M> {
    /// This module's program on `engine`, compiled on the first ask
    /// for that engine and shared by every instance after it.
    ///
    /// This is the kernel image. A module instantiated many times —
    /// once per coordinate, per fiber, per scenario visit — compiles
    /// once per engine here and pays only a state per instance, which
    /// is what [`Self::instantiate_under`] does with it.
    pub fn program_on(
        &self,
        engine: crate::Engine,
    ) -> Result<Arc<dyn crate::kernel::KernelProgram>, crate::KernelError> {
        let mut programs = self
            .programs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(program) = programs.get(&engine) {
            return Ok(program.clone());
        }
        let kernel = crate::dsl::compile::compile_ast_with_engine(
            &crate::dsl::ast::PolydatFile {
                statements: self.statements.clone(),
            },
            "",
            &self.options,
            None,
            engine,
        )?;
        let program = kernel.into_program();
        programs.insert(engine, program.clone());
        Ok(program)
    }

    /// One instance of this module under `parent`: a kernel on
    /// `engine` over the program [`Self::program_on`] holds, with the
    /// parent's cells attached, its values copied in, and
    /// `iter_bindings` written before either.
    ///
    /// The image is shared; the instance is a state. Calling this a
    /// second time with different coordinates or different input
    /// values compiles nothing.
    ///
    /// `engine` is the caller's to name, and the parent's own is the
    /// answer a caller usually wants: a child belongs to the kernel it
    /// was bound under, the way a `for` body belongs to the kernel
    /// that opened it.
    pub fn instantiate_under(
        &self,
        parent: &dyn crate::kernel::Kernel,
        engine: crate::Engine,
        iter_bindings: &[(String, crate::ast::Value)],
    ) -> Result<Box<dyn crate::kernel::Kernel>, crate::KernelError> {
        let program = self.program_on(engine)?;
        let mut child = program.create_kernel();
        for (var, value) in iter_bindings {
            // Before the wiring, so the child's own scope-coordinate
            // snapshot sees them.
            let _ = child.set_input(var, value.clone());
        }
        crate::kernel::PolydatKernel::wire_child_under(child.as_mut(), parent);
        Ok(child)
    }

    /// The imports the module declares.
    pub fn imports(&self) -> &[ImportSpec] {
        &self.imports
    }

    /// The exports the module declares.
    pub fn exports(&self) -> &[ExportSpec] {
        &self.exports
    }

    /// The body's compiled program.
    pub fn program(&self) -> &Arc<PolydatProgram> {
        &self.program
    }

    /// The contract the module was built against.
    pub fn contract(&self) -> &ScopeContract<M> {
        &self.contract
    }

    /// Where the module comes from.
    pub fn context(&self) -> &SourceContext {
        &self.context
    }

    /// The pull consumers registered on the module's outputs.
    pub fn consumers(&self) -> &[RegisteredPullConsumer] {
        &self.consumers
    }

    /// Diagnostics the build recorded.
    pub fn diagnostics(&self) -> &[String] {
        &self.diagnostics
    }

    /// Rule 2 write-through bindings produced by the builder for
    /// child exports that collide with a parent `shared` export.
    pub fn write_throughs(&self) -> &[WriteThroughBinding] {
        &self.write_throughs
    }
}

impl<M> std::fmt::Debug for ScopeModule<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopeModule")
            .field("imports", &self.imports)
            .field("exports", &self.exports)
            .field("context", &self.context)
            .field("consumer_count", &self.consumers.len())
            .field("write_throughs", &self.write_throughs)
            .field("diagnostics", &self.diagnostics)
            .finish()
    }
}
