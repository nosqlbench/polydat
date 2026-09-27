// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! [`ScopeKernel<M>`] — typed wrapper around a kernel of any engine.
//!
//! `ScopeKernel<M>` is the typed surface (subcontext_construction.md
//! §1); the interpreter's `PolydatKernel` stays public, and its
//! construction primitives are sealed (`from_program` is crate-private,
//! `materialize_wiring_from_outer` private), so a child is built only
//! through the typed surface or the binder. A child is built on its
//! parent's engine.
//!
//! The kernel exposes:
//! - [`Self::subcontext_builder`] — yields a typed
//!   [`super::SubcontextBuilder`] over this kernel's
//!   [`super::ParentView`]. The single public entry point for child
//!   construction on the typed path.
//! - [`Self::spawn`] — the single chokepoint where every
//!   cross-binding is resolved; takes a closed
//!   [`super::ScopeModule`] artifact, applies the
//!   cross-binding rules (subcontext_construction.md §4), returns a typed child kernel and
//!   records the spawn under `name` in this kernel's registry.
//! - [`Self::release_child`] — drop a registry entry to allow
//!   re-spawn under the same name (for per-iteration
//!   re-traversal).

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use crate::ast::{PortType, Value};
use crate::kernel::{Kernel, PolydatKernel, SharedCell};

use super::builder::{ParentView, SubcontextBuilder};
use super::error::{ContractViolation, SourceContext};
use super::module::{ScopeModule, WriteThroughBinding};
use super::name::ChildName;
use super::pull::RegisteredPullConsumer;

/// Phantom-marker brand for the workload-root scope kernel —
/// the top of any spawn type chain. Tests / examples that need
/// a "starting" identity use this.
#[derive(Debug)]
pub struct RootMarker;

/// Phantom-marker brand for "child of `P`". `spawn` returns
/// `ScopeKernel<Child<P>>`, distinct at the type level from a
/// sibling's `Child<P>` *value* but type-compatible at the
/// module-identity level (subcontext_construction.md §1).
#[derive(Debug)]
pub struct Child<P>(PhantomData<fn() -> P>);

/// Internal record of a spawned child — used for the
/// duplicate-spawn diagnostic.
#[derive(Debug)]
struct ChildEntry {
    site: SourceContext,
}

/// Typed wrapper around a kernel of any engine.
///
/// Construction via this type goes through the subcontext protocol
/// (`subcontext_builder` → `finalize` → `spawn`), which builds each
/// child on its parent's engine; a root is wrapped with
/// [`wrap_root_kernel`].
pub struct ScopeKernel<M> {
    name: ChildName,
    inner: Arc<Mutex<Box<dyn Kernel>>>,
    site: SourceContext,
    children: Mutex<HashMap<ChildName, ChildEntry>>,
    consumers: Mutex<Vec<RegisteredPullConsumer>>,
    /// Rule 2 write-through bindings. Per-cycle eval of this
    /// kernel must call [`Self::commit_write_throughs`] after
    /// producing values to fan them through the parent's
    /// `SharedCell`s.
    write_throughs: Vec<WriteThroughBinding>,
    _module: PhantomData<fn() -> M>,
}

impl<M> std::fmt::Debug for ScopeKernel<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopeKernel")
            .field("name", &self.name)
            .field("site", &self.site)
            .finish()
    }
}

/// One shared cell visible at a parent scope, reified for
/// transitive cross-binding. Returned by
/// [`ScopeKernel::shared_cells_in_scope`].
///
/// Carries the name a child must use to bind to the cell,
/// the port type (so Rule 2 / `extern` synthesis at finalize
/// can declare a typed input slot), and the cell handle (so
/// spawn can attach it to the child's matching input).
///
/// "In scope" semantics: a cell visible at the parent is one
/// the parent itself can read or write at this scope —
/// covering both:
///
/// 1. Cells the parent declared via its own program
///    (`shared X := <init>` produces a cell-bound input slot).
/// 2. Cells inherited from the parent's own ancestors
///    (attached during the parent's spawn). Without this
///    case, a `shared` cell at the workload root would not
///    propagate to grand-children whose immediate parent's
///    body never references the name.
///
/// Both cases are answered by walking the parent's input
/// slots and reading `PolydatState::shared_cell` for each (see
/// `PolydatKernel::shared_cells_in_scope`).
#[derive(Clone)]
pub struct SharedCellInScope {
    /// The binding's name.
    pub name: String,
    /// The cell's declared type.
    pub port_type: PortType,
    /// The cell.
    pub cell: SharedCell,
}

impl std::fmt::Debug for SharedCellInScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedCellInScope")
            .field("name", &self.name)
            .field("port_type", &self.port_type)
            .finish()
    }
}

impl<M> ScopeKernel<M> {
    /// Enumerate every shared cell visible at this scope: the
    /// kernel's [`Kernel::cells_in_scope`], its own and those it
    /// carries for its descendants.
    pub fn shared_cells_in_scope(&self) -> Vec<SharedCellInScope> {
        let inner = self.lock_inner();
        inner
            .cells_in_scope()
            .into_iter()
            .map(|e| SharedCellInScope {
                name: e.name,
                port_type: e.port_type,
                cell: e.cell,
            })
            .collect()
    }

    /// Internal constructor — only the spawn path and
    /// [`wrap_root_kernel`] produce a `ScopeKernel` directly. Public
    /// callers go through the protocol.
    pub(crate) fn new_internal(
        name: ChildName,
        kernel: Box<dyn Kernel>,
        site: SourceContext,
        consumers: Vec<RegisteredPullConsumer>,
    ) -> Self {
        Self::new_with_write_throughs(name, kernel, site, consumers, Vec::new())
    }

    pub(crate) fn new_with_write_throughs(
        name: ChildName,
        kernel: Box<dyn Kernel>,
        site: SourceContext,
        consumers: Vec<RegisteredPullConsumer>,
        write_throughs: Vec<WriteThroughBinding>,
    ) -> Self {
        Self {
            name,
            inner: Arc::new(Mutex::new(kernel)),
            site,
            children: Mutex::new(HashMap::new()),
            consumers: Mutex::new(consumers),
            write_throughs,
            _module: PhantomData,
        }
    }

    /// The structured name this kernel was spawned under (for
    /// child kernels) or its self-label (for root kernels).
    pub fn name(&self) -> &ChildName {
        &self.name
    }

    /// Diagnostic site for this kernel's construction.
    pub fn site(&self) -> &SourceContext {
        &self.site
    }

    /// Borrow the underlying kernel. The lock is released when the
    /// returned guard is dropped. Exposed for callers that drive the
    /// kernel directly.
    pub fn lock_inner(&self) -> std::sync::MutexGuard<'_, Box<dyn Kernel>> {
        self.inner
            .lock()
            .expect("ScopeKernel inner kernel poisoned")
    }

    /// The pull consumers registered with this kernel. Used by
    /// the activity-side fixture adapter at seal time.
    pub fn consumers(&self) -> Vec<RegisteredPullConsumer> {
        self.consumers
            .lock()
            .expect("ScopeKernel consumers poisoned")
            .clone()
    }

    /// Whether `name` is recorded in this kernel's named-child
    /// registry. Diagnostic; the subcontext tests assert against this.
    pub fn has_child(&self, name: &ChildName) -> bool {
        self.children
            .lock()
            .expect("ScopeKernel children registry poisoned")
            .contains_key(name)
    }

    /// Drop the named child from this kernel's registry. The
    /// child kernel itself is unaffected — only the registry
    /// entry. After release, the same name may be spawned again
    /// (typical for comprehension scopes that re-traverse per
    /// iteration). See subcontext_construction.md §4.1.
    pub fn release_child(&self, name: &ChildName) {
        self.children
            .lock()
            .expect("ScopeKernel children registry poisoned")
            .remove(name);
    }

    /// Begin construction of a child sub-context
    /// (subcontext_construction.md §1): the builder takes the
    /// parent's [`ParentView`] — its names, modifiers, ledger, and
    /// in-scope cells, which is everything finalize reads of a
    /// parent — accumulates
    /// module matter, and produces a closed [`ScopeModule`] artifact
    /// at finalize. It holds no reference to the parent kernel, so a
    /// caller that only has a kernel needs no `ScopeKernel` to stand up
    /// a child.
    pub fn subcontext_builder(&self) -> SubcontextBuilder<M> {
        SubcontextBuilder::new(ParentView::of_kernel(self.lock_inner().as_ref()))
    }

    /// Spawn a child kernel from a closed [`ScopeModule`]
    /// artifact (subcontext_construction.md §4): this is the single
    /// chokepoint where every cross-binding is resolved.
    ///
    /// The artifact arrives with Rule 1 (name closure) and Rule 2
    /// (the shared write-through rewrite) already applied by
    /// [`SubcontextBuilder::finalize`]. Spawn instantiates the
    /// closed program on this parent's engine
    /// ([`ScopeModule::instantiate_under`]), whose binder does the
    /// live binding: attaches every parent-visible `SharedCell` to
    /// a matching child slot and forwards the rest as transit
    /// (Rule 2's cell attach, SC8), value-copies or cell-attaches
    /// parent outputs into child externs (Rules 4 and 5),
    /// initializes the child so its consts see post-bind inputs
    /// (Rule 3), and freezes the scope coordinates. A refused copy or
    /// a failing const is [`ContractViolation::Bind`]. Per-cycle
    /// publication to the cells is [`Self::commit_write_throughs`].
    pub fn spawn(
        self: &Arc<Self>,
        name: ChildName,
        artifact: ScopeModule<Child<M>>,
    ) -> Result<ScopeKernel<Child<M>>, ContractViolation> {
        // ----- Named-child registry guard
        // (subcontext_construction.md §4.1) -----
        {
            let mut children = self
                .children
                .lock()
                .expect("ScopeKernel children registry poisoned");
            if let Some(prior) = children.get(&name) {
                return Err(ContractViolation::DuplicateChild {
                    name: name.clone(),
                    prior_site: Box::new(prior.site.clone()),
                    this_site: artifact.context.clone(),
                });
            }
            children.insert(
                name.clone(),
                ChildEntry {
                    site: artifact.context.clone(),
                },
            );
        }

        // ----- Cross-binding resolution -----
        // Single chokepoint: `materialize_wiring_from_outer` walks every
        // cell visible at the parent (own slots + transit
        // cells inherited from ancestors), attaches each to
        // any matching child slot, and forwards the rest as
        // transit on the child kernel. This is the transitive
        // cascade — an ancestral `shared X` cell remains
        // visible to deep descendants regardless of how many
        // intermediate scopes' bodies skip the name.
        //
        // Honours Rule 1 (import resolution validated at
        // finalize), Rule 2 (write-through rewrite produces
        // the matching child input slot finalize-side),
        // Rule 4 (coordinate routing via IterationExtern
        // input-kind), Rule 5 (closure-binding economy —
        // unreferenced names skip cell attachment but still
        // ride the transit channel for grand-children).
        let built = {
            let parent_inner = self.lock_inner();
            artifact.instantiate_under(parent_inner.as_ref(), parent_inner.engine(), &[])
        };
        let child_kernel = match built {
            Ok(kernel) => kernel,
            Err(e) => {
                // A child that was never built leaves its name free.
                self.release_child(&name);
                return Err(e.into());
            }
        };

        let child_site = artifact.context.clone();
        let child_consumers = artifact.consumers.clone();
        let child_write_throughs = artifact.write_throughs.clone();

        Ok(ScopeKernel::new_with_write_throughs(
            name,
            child_kernel,
            child_site,
            child_consumers,
            child_write_throughs,
        ))
    }

    /// The Rule 2 write-through bindings carried by this kernel.
    /// Empty for the vast majority of kernels; populated only
    /// when the artifact's `finalize` rewrote a child export to
    /// a parent `shared` cell write.
    pub fn write_throughs(&self) -> &[WriteThroughBinding] {
        &self.write_throughs
    }

    /// Per-cycle Rule 2 commit: pulls every write-through's
    /// synthetic source output (`__write_<X>`) and stores its
    /// value through the corresponding child input slot for
    /// `<X>`. Because `materialize_wiring_from_outer` attached the parent's
    /// `SharedCell` to that slot, the write propagates to the
    /// cell, where it becomes visible to the parent and to any
    /// sibling that shares the same cell.
    ///
    /// No-op for kernels with no write-throughs.
    ///
    /// TYPE-STABLE (scope_model.md §6.1): each pending
    /// value passes the boundary of [`Kernel::commit_write_throughs`]:
    /// matching types pass, catalog adapters heal (widening), and an
    /// unhealable mismatch is an `Err` at the write site.
    pub fn commit_write_throughs(&self) -> Result<(), String> {
        if self.write_throughs.is_empty() {
            return Ok(());
        }
        self.lock_inner().commit_write_throughs()
    }
}

/// Construct a workload-root [`ScopeKernel<RootMarker>`] from a
/// compiled kernel of any engine: the door into the typed scope path,
/// where a host spawns children it keeps and releases by name rather
/// than dropping a kernel on the floor. Every child spawned under it
/// runs on its engine.
pub fn wrap_root_kernel(
    kernel: Box<dyn Kernel>,
    label: impl Into<String>,
) -> Arc<ScopeKernel<RootMarker>> {
    let label = label.into();
    let name = ChildName::from_segments([label.clone()]);
    let site = SourceContext::new(label);
    Arc::new(ScopeKernel::new_internal(name, kernel, site, Vec::new()))
}

/// Typed Polydat matter accepted by both kernel-construction
/// paths — root and subscope. Opaque externally: the only way
/// to obtain a `PolydatMatter` value is via [`PolydatMatter::builder`].
///
/// Internally carries one of three input forms — fresh source,
/// pre-parsed statements (the "module parser" output), or a
/// pre-compiled program. The builder validates that exactly
/// one form is provided.
pub struct PolydatMatter<'a> {
    pub(crate) inner: PolydatMatterInner<'a>,
}

pub(crate) enum PolydatMatterInner<'a> {
    Source(SourceMatter),
    Statements(StatementsMatter),
    Program(ProgramMatter<'a>),
}

pub(crate) struct SourceMatter {
    pub(crate) label: String,
    pub(crate) body: String,
    pub(crate) result_bindings: Option<String>,
    pub(crate) inherited_outputs: Vec<String>,
    pub(crate) options: super::builder::CompileOptions,
}

pub(crate) struct StatementsMatter {
    pub(crate) label: String,
    pub(crate) statements: Vec<crate::dsl::ast::Statement>,
    pub(crate) result_bindings: Option<String>,
    pub(crate) inherited_outputs: Vec<String>,
    pub(crate) options: super::builder::CompileOptions,
}

pub(crate) struct ProgramMatter<'a> {
    pub(crate) program: Arc<dyn crate::kernel::KernelProgram>,
    pub(crate) iter_bindings: &'a [(String, Value)],
}

impl<'a> PolydatMatter<'a> {
    /// Begin building Polydat matter. The builder is the only
    /// constructor of `PolydatMatter`; the variants and their
    /// fields are not exposed.
    #[inline]
    pub fn builder() -> PolydatMatterBuilder<'a> {
        PolydatMatterBuilder::new()
    }
}

/// Builder for [`PolydatMatter`]. Configure exactly one input form
/// (source, pre-parsed statements, or program), plus optional
/// metadata, then call [`Self::build`].
#[derive(Default)]
pub struct PolydatMatterBuilder<'a> {
    label: Option<String>,
    body: Option<String>,
    statements: Option<Vec<crate::dsl::ast::Statement>>,
    program: Option<Arc<dyn crate::kernel::KernelProgram>>,
    iter_bindings: &'a [(String, Value)],
    result_bindings: Option<String>,
    inherited_outputs: Vec<String>,
    options: super::builder::CompileOptions,
}

impl<'a> PolydatMatterBuilder<'a> {
    fn new() -> Self {
        Self::default()
    }

    /// Diagnostic label for this matter. Surfaces in compile
    /// errors and as the child's `SourceContext`.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Provide Polydat source as a string. Mutually exclusive with
    /// [`Self::statements`] and [`Self::program`].
    pub fn source(mut self, body: impl Into<String>) -> Self {
        self.body = Some(body.into());
        self
    }

    /// Provide Polydat source as pre-parsed AST statements. Mutually
    /// exclusive with [`Self::source`] and [`Self::program`].
    /// Use when the caller has already run the module parser
    /// (e.g. when synthesising scope source from a structured
    /// model and wanting to skip a string round-trip).
    pub fn statements(mut self, stmts: Vec<crate::dsl::ast::Statement>) -> Self {
        self.statements = Some(stmts);
        self
    }

    /// Provide a pre-compiled program on any engine
    /// ([`crate::Kernel::into_program`], or an interpreter
    /// `Arc<PolydatProgram>`). Mutually exclusive with
    /// [`Self::source`] and [`Self::statements`]. Used for per-
    /// fiber state forks, comprehension iteration, and other
    /// call sites that hold a compiled program directly.
    pub fn program(mut self, program: Arc<dyn crate::kernel::KernelProgram>) -> Self {
        self.program = Some(program);
        self
    }

    /// Iter-var bindings applied before the parent binds the
    /// child. Only meaningful for the program form.
    pub fn iter_bindings(mut self, bindings: &'a [(String, Value)]) -> Self {
        self.iter_bindings = bindings;
        self
    }

    /// Result-binding source (subcontext_construction.md §3.2). Folded through
    /// [`super::SubcontextBuilder::add_result_bindings`] at
    /// finalize. Only meaningful for source / statements forms.
    pub fn result_bindings(mut self, src: impl Into<String>) -> Self {
        self.result_bindings = Some(src.into());
        self
    }

    /// Names to pass through `mark_inherited_outputs` so the
    /// scope tree can distinguish own exports from cascade-
    /// inherited names. Source / statements forms only.
    pub fn inherited_outputs(mut self, names: Vec<String>) -> Self {
        self.inherited_outputs = names;
        self
    }

    /// Compile-time knobs (lib paths, strict mode, required
    /// outputs, cursor limit). Source / statements forms only.
    pub fn options(mut self, options: super::builder::CompileOptions) -> Self {
        self.options = options;
        self
    }

    /// Validate and produce typed matter. Errors when zero or
    /// more than one input form is configured.
    pub fn build(self) -> Result<PolydatMatter<'a>, String> {
        let forms = [
            self.body.is_some(),
            self.statements.is_some(),
            self.program.is_some(),
        ];
        let count = forms.iter().filter(|x| **x).count();
        if count == 0 {
            return Err(
                "PolydatMatter::builder: no input form set (use .source / .statements / .program)"
                    .into(),
            );
        }
        if count > 1 {
            return Err(
                "PolydatMatter::builder: multiple input forms set; choose exactly one".into(),
            );
        }
        let label = self.label.unwrap_or_else(|| "(matter)".to_string());
        let inner = if let Some(body) = self.body {
            PolydatMatterInner::Source(SourceMatter {
                label,
                body,
                result_bindings: self.result_bindings,
                inherited_outputs: self.inherited_outputs,
                options: self.options,
            })
        } else if let Some(stmts) = self.statements {
            PolydatMatterInner::Statements(StatementsMatter {
                label,
                statements: stmts,
                result_bindings: self.result_bindings,
                inherited_outputs: self.inherited_outputs,
                options: self.options,
            })
        } else {
            // program
            PolydatMatterInner::Program(ProgramMatter {
                program: self.program.expect("program form set per count above"),
                iter_bindings: self.iter_bindings,
            })
        };
        Ok(PolydatMatter { inner })
    }
}

impl PolydatMatter<'_> {
    /// THE subscope-construction path. Per the kernel-construction
    /// invariant, this is how a parent kernel of any engine produces a
    /// child. `compile_polydat` produces root kernels; everything else
    /// is a subscope and routes here.
    ///
    /// A child from source or statements is compiled once and bound on
    /// `parent`'s engine through the binder every engine shares
    /// ([`super::ScopeModule::instantiate_under`]). A child from a
    /// compiled program is bound on that program's engine
    /// ([`crate::kernel::bind_under`]). Cell propagation,
    /// scope-coordinate plumbing, and Rule 2 write-throughs flow from
    /// `parent` into the returned child.
    pub fn build_under(self, parent: &dyn Kernel) -> Result<Box<dyn Kernel>, ContractViolation> {
        use super::module::BodyFragment;
        let (label, strict, inherited, options, body, result_bindings) = match self.inner {
            PolydatMatterInner::Program(p) => {
                return Ok(crate::kernel::bind_under(
                    parent,
                    p.program,
                    p.iter_bindings,
                )?);
            }
            PolydatMatterInner::Source(s) => (
                s.label,
                s.options.strict,
                s.inherited_outputs,
                s.options,
                BodyFragment::PolydatSource(s.body),
                s.result_bindings,
            ),
            PolydatMatterInner::Statements(s) => (
                s.label,
                s.options.strict,
                s.inherited_outputs,
                s.options,
                BodyFragment::Statements(s.statements),
                s.result_bindings,
            ),
        };
        let mut builder: SubcontextBuilder<RootMarker> =
            SubcontextBuilder::new(ParentView::of_kernel(parent));
        builder
            .context(SourceContext::new(label.clone()))
            .mark_inherited_outputs(inherited)
            .with_compile_options(options)
            .body(body);
        if let Some(src) = result_bindings {
            builder.add_result_bindings(&src)?;
        }
        let module = builder.finalize()?;
        let mut child = module.instantiate_under(parent, parent.engine(), &[])?;
        enforce_l2f_strict(child.as_mut(), strict, &label)?;
        Ok(child)
    }
}

impl PolydatKernel {
    /// [`PolydatMatter::build_under`] this kernel: the child runs on
    /// the interpreter, this kernel's engine.
    pub fn build_subscope(
        &self,
        matter: PolydatMatter<'_>,
    ) -> Result<Box<dyn Kernel>, ContractViolation> {
        matter.build_under(self)
    }
}

/// L2.f strict-mode hardening — when strict is on, escalate a
/// const's silent fall-through to a hard error
/// (`ContractViolation::StrictNonePropagation`,
/// subcontext_construction.md §7). Per
/// composition_substrate.md L2.f's strict-mode hardening
/// clause: an intermediate-layer `const X := <expr>` whose
/// RHS evaluates to `Value::None` at initialization normally
/// falls through to the outer scope's `X` via the
/// conditional-shadow semantics (none_semantics.md). Strict
/// mode rejects this silent fall-through, forcing the author
/// to either ensure the const yields a defined value or
/// remove the binding and declare `extern X` explicitly if
/// fall-through to outer was intended.
///
/// A const's own value is its expression's: `__init_<name>` for a
/// const captured at initialization, the output itself for one
/// folded at build.
fn enforce_l2f_strict(
    child: &mut dyn Kernel,
    strict: bool,
    label: &str,
) -> Result<(), ContractViolation> {
    if !strict {
        return Ok(());
    }
    let consts: Vec<String> = child
        .output_names()
        .into_iter()
        .filter(|n| child.output_modifier(n).is_const() && !n.starts_with("__init_"))
        .collect();
    let bindings: Vec<String> = consts
        .into_iter()
        .filter(|name| {
            let own = child
                .const_inits()
                .iter()
                .find(|c| &c.name == name && !c.register)
                .map_or_else(|| name.clone(), |c| c.source.clone());
            matches!(child.pull(&own), Value::None)
        })
        .collect();
    if bindings.is_empty() {
        return Ok(());
    }
    Err(ContractViolation::StrictNonePropagation {
        bindings,
        site: SourceContext::new(label.to_string()),
    })
}

// Per the kernel-construction invariant, two paths exist:
//
//   1. Root kernel built from source via `compile_polydat` (and family).
//   2. Subscope kernel bound under a parent kernel of any engine via
//      `PolydatMatter::build_under`, `ScopeKernel::spawn`,
//      `ScopeModule::instantiate_under`, or `bind_under`, each
//      parent-supervised through the one binder.
