// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The kernel API: one surface for every engine.
//!
//! A host holds a kernel as `Box<dyn Kernel>` whatever engine built
//! it, and drives it through [`Kernel`]: the coordinate and extern
//! writes that invalidate their dependents, `pull` for one output and `eval` for
//! every one, the names and types of its inputs and outputs, the
//! traversals its program declares, its cells, and `into_program`, the
//! program shared across threads that [`KernelProgram::create_kernel`]
//! makes a kernel of per thread. Every engine that accepts a program
//! computes what the interpreter computes, and every call means the
//! same thing on every engine: the one write rule at `set_input`, one
//! `invalidate_all`, outputs in declaration order, and a created kernel
//! starting from the program's defaults.
//!
//! [`PolydatKernel`] is the interpreter's kernel as a concrete type:
//! a program (immutable, shared through `Arc<PolydatProgram>`) and
//! one evaluation state. It implements [`Kernel`] and keeps three
//! traits of its own, for the interpreter alone:
//!
//! - [`Dataflow`], the healing write: `set_wire` runs the boundary
//!   adapter catalog before a typed rejection, where `Kernel::set_input`
//!   refuses a value of another type outright.
//! - [`Metadata`], structural queries the program answers directly.
//! - [`Construction`], the subcontext protocol: a root from source
//!   matter, a subscope built against this kernel with new matter.
//!
//! The construction-time hooks the compile path and the program
//! sharing use (attaching traversals, resolving cursor extents,
//! nesting) live on a sealed supertrait a host neither sees nor
//! implements.
//!
//! [`PolydatKernel`]: super::PolydatKernel

use crate::ast::{PortType, Value};
use crate::kernel::{SharedCell, SharedCellEntry};

/// Error returned by [`Dataflow::set_wire_idx`] /
/// [`Dataflow::set_wire`] when the typed-write contract at the
/// composition-substrate boundary cannot be satisfied.
///
/// Per composition_substrate.md axiom S4, "T1 + T2 ensure
/// writes are type-checked at the boundary" — the typed-write
/// API rejects writes whose Value variant doesn't match the
/// declared slot port type, after first attempting auto-adapter
/// healing. This error names the rejection reason.
#[derive(Debug, Clone, PartialEq)]
pub enum WriteError {
    /// The wire key did not resolve to a known input slot.
    /// Carries the name that was looked up; for indexed writes
    /// the index is reported instead.
    UnknownWire {
        /// The name or index looked up.
        key: String,
        /// The kernel's input slots, as
        /// [`Kernel::input_names`] reports them and in the same order,
        /// coordinates included. Empty only where the writer does not
        /// have the list.
        ///
        /// Coordinates are in it although writing one by name is
        /// [`Self::CoordinateSlot`] rather than a success: a caller who
        /// mistyped a coordinate meant a name this kernel has, and is
        /// not helped by a list that leaves it out. The exact match is
        /// what the other variant is for. Every engine answers alike,
        /// and a host can check the list against `input_names`.
        known: Vec<String>,
    },

    /// The value's port type did not match the slot's declared
    /// port type and no auto-adapter exists to heal the
    /// mismatch. Both expected and provided port types are
    /// reported for diagnostic clarity.
    TypeMismatch {
        /// The slot written.
        slot: String,
        /// Its declared type.
        expected: PortType,
        /// The value's type.
        got: PortType,
    },

    /// The slot is a coordinate, which advances through
    /// `set_inputs` rather than being written by name or index.
    /// Writing one here would put the coordinate prefix out of
    /// step with the values a pull is about to read.
    CoordinateSlot {
        /// The coordinate named.
        slot: String,
    },

    /// The slot holds a `const` binding's value, which only
    /// [`Kernel::init`] writes. A const is fixed for the life of the
    /// kernel; to change it, write the inputs it reads and initialize.
    ConstSlot {
        /// The const's slot.
        slot: String,
    },
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteError::UnknownWire { key, known } => {
                write!(f, "unknown wire '{key}': no input slot by this name")?;
                if !known.is_empty() {
                    write!(f, "; this kernel's are {known:?}")?;
                }
                Ok(())
            }
            WriteError::CoordinateSlot { slot } => {
                write!(
                    f,
                    "'{slot}' is a coordinate: advance it with set_inputs, not by name"
                )
            }
            WriteError::ConstSlot { slot } => {
                write!(
                    f,
                    "'{slot}' holds a const, which only initialization writes: write the \
                     inputs it reads and call init()"
                )
            }
            WriteError::TypeMismatch {
                slot,
                expected,
                got,
            } => {
                write!(
                    f,
                    "type mismatch writing to slot '{slot}': expected {expected:?}, got {got:?} (no auto-adapter available)"
                )?;
                // Vec → scalar is intentionally excluded from the
                // polyfill matrix (type_system.md §3) — there is
                // no single natural collection-to-scalar
                // convention. Point the author at the explicit
                // helpers rather than leaving them to guess.
                if matches!(got, PortType::VecF32 | PortType::VecI32)
                    && !matches!(
                        expected,
                        PortType::VecF32
                            | PortType::VecI32
                            | PortType::Str
                            | PortType::Bytes
                            | PortType::Json
                    )
                {
                    write!(
                        f,
                        " — collection → scalar requires an explicit \
                         reduction node in the program (the library \
                         provides none; `vec_dot` and `vec_norm` are \
                         the vector reductions that exist)"
                    )?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for WriteError {}

/// A wire reference — either a pre-resolved index (fast path)
/// or a name (resolved against the context's input map).
///
/// Lets `set_wire` / `get_wire` accept either form so callers
/// can hold an index when they have one and a name when they
/// don't, without needing two distinct method names.
///
/// Sealed: only the in-crate impls (`usize`, `&str`, `String`)
/// are valid wire keys. External implementors are not
/// permitted because the resolution semantics are tied to the
/// context's input layout.
pub trait WireKey: sealed::Sealed {
    /// Resolve to a wire index in `metadata`. Returns `None`
    /// when the key doesn't match a wire on this context.
    fn resolve<M: Metadata + ?Sized>(self, metadata: &M) -> Option<usize>;

    /// Diagnostic rendering of this key — used by
    /// [`Dataflow::set_wire`] when constructing
    /// [`WriteError::UnknownWire`] so the error names what the
    /// caller passed.
    fn describe(&self) -> String;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for usize {}
    impl Sealed for &str {}
    impl Sealed for String {}
    impl Sealed for &String {}
}

impl WireKey for usize {
    #[inline]
    fn resolve<M: Metadata + ?Sized>(self, _: &M) -> Option<usize> {
        Some(self)
    }
    #[inline]
    fn describe(&self) -> String {
        format!("wire[{self}]")
    }
}

impl WireKey for &str {
    #[inline]
    fn resolve<M: Metadata + ?Sized>(self, metadata: &M) -> Option<usize> {
        metadata.find_input(self)
    }
    #[inline]
    fn describe(&self) -> String {
        (*self).to_string()
    }
}

impl WireKey for String {
    #[inline]
    fn resolve<M: Metadata + ?Sized>(self, metadata: &M) -> Option<usize> {
        metadata.find_input(&self)
    }
    #[inline]
    fn describe(&self) -> String {
        self.clone()
    }
}

impl WireKey for &String {
    #[inline]
    fn resolve<M: Metadata + ?Sized>(self, metadata: &M) -> Option<usize> {
        metadata.find_input(self)
    }
    #[inline]
    fn describe(&self) -> String {
        (*self).clone()
    }
}

/// Read-only metadata about the interpreter's kernel: structural
/// shape, types, names, scope layering. Everything that's a property
/// of the compiled program (or fiber-state instance) but isn't itself
/// a runtime value. Interpreter-only: [`Kernel`] carries the names and
/// types every engine reports.
pub trait Metadata {
    /// Resolve an input name to its wire index, if present.
    fn find_input(&self, name: &str) -> Option<usize>;

    /// All declared input wire names, in declaration order.
    fn input_names(&self) -> Vec<String>;

    /// All declared output wire names, in declaration order.
    fn output_names(&self) -> Vec<String>;

    /// Number of coordinate inputs (the leading prefix of the
    /// input slot vector — written via the cycle dispatcher).
    fn coord_count(&self) -> usize;

    /// Declared port type of an input wire, if known.
    fn input_port_type(&self, name: &str) -> Option<PortType>;

    /// Declared port type of an input wire by index. The
    /// indexed counterpart of [`input_port_type`](Self::input_port_type) — used by
    /// the typed-write fast path so [`Dataflow::set_wire_idx`]
    /// can look up the slot's expected type without first
    /// reverse-resolving an index to a name.
    fn input_port_type_by_idx(&self, idx: usize) -> Option<PortType>;

    /// Declared port type of an output wire, if present.
    /// Symmetric counterpart to [`input_port_type`](Self::input_port_type). Used by
    /// the binder verification path
    /// (`crate::binder::verify_against_kernel`) to look up wire
    /// types for type-checking adapter binding shapes.
    fn output_port_type(&self, name: &str) -> Option<PortType>;
}

/// The interpreter kernel's healing write and raw read: write inputs,
/// read wires.
///
/// Four core methods. The indexed pair is the fast path; the named
/// pair resolves against the context's metadata then delegates to the
/// indexed pair. A write runs the boundary adapter catalog before a
/// typed rejection, where [`Kernel::set_input`] refuses a value of
/// another type outright. Interpreter-only.
pub trait Dataflow: Metadata {
    /// Write a value to wire `idx` with typed enforcement.
    ///
    /// Per composition_substrate.md axiom S4, the typed-write
    /// boundary enforces T1 + T2: the value's port type must
    /// match the slot's declared port type, with auto-adapter
    /// healing where the implementation supports it. Mismatches
    /// the boundary cannot heal return [`WriteError::TypeMismatch`].
    /// An out-of-range index returns
    /// [`WriteError::UnknownWire`].
    #[deprecated(
        since = "0.5.0",
        note = "a write is never converted (input_variance.md): use `Kernel::set_input_at`, \
                converting first with `polydat::convert::to_port`, or open the input with \
                `CompileOptions::input_variance`"
    )]
    fn set_wire_idx(&mut self, idx: usize, value: Value) -> Result<(), WriteError>;

    /// Read the current value of wire `idx`. Out-of-range
    /// behaviour returns the slot's default `Value::None` (the
    /// read path is non-fallible; type information is structural
    /// and reads cannot fail typewise).
    fn get_wire_idx(&self, idx: usize) -> Value;

    /// Write a value to a wire identified by `key` (index or
    /// name). Returns `Ok(())` on success, `Err(WriteError)` on
    /// failure (unknown wire or type mismatch the boundary
    /// cannot heal).
    #[deprecated(
        since = "0.5.0",
        note = "a write is never converted (input_variance.md): use `Kernel::set_input`, \
                converting first with `polydat::convert::to_port`, or open the input with \
                `CompileOptions::input_variance`"
    )]
    #[inline]
    #[allow(deprecated)]
    fn set_wire<W: WireKey>(&mut self, key: W, value: Value) -> Result<(), WriteError> {
        // Capture a string form of the key for diagnostic
        // reporting before resolution consumes it. The
        // WireKey::describe method provides this; the default
        // impl renders index keys as "wire[N]" and name keys
        // as the name itself.
        let key_desc = key.describe();
        match key.resolve(self) {
            Some(idx) => self.set_wire_idx(idx, value),
            None => Err(WriteError::UnknownWire {
                key: key_desc,
                known: Vec::new(),
            }),
        }
    }

    /// Read the current value of a wire identified by `key`
    /// (index or name). Returns `None` when the wire is not
    /// found.
    #[inline]
    fn get_wire<W: WireKey>(&self, key: W) -> Option<Value> {
        key.resolve(self).map(|idx| self.get_wire_idx(idx))
    }
}

/// Construction interface — the two sanctioned construction
/// paths. Per the kernel-construction invariant:
///
/// 1. **Root** — built from Polydat matter, no parent.
/// 2. **Subscope** — built from Polydat matter against an existing
///    context.
///
/// Both paths take the same typed Polydat matter
/// ([`super::subcontext::PolydatMatter`]). The only
/// difference is whether a parent context supervises
/// construction. Nothing else is allowed.
pub trait Construction: Sized {
    /// Construction error type.
    type Error;

    /// Path 1: build a root context from Polydat matter. No parent.
    /// Subscope-only fields on the matter (result-binding
    /// rewrites, inherited-output cascade, finalize-time
    /// contract checks) are not applicable here and are
    /// ignored.
    fn root(matter: super::subcontext::PolydatMatter<'_>) -> Result<Self, Self::Error>;

    /// Path 2: build a subscope context against `self` from
    /// Polydat matter. The parent supervises: cell cascade, Rule 2
    /// rewrites, scope-coordinate threading, init-binding
    /// contract checks all flow from `self` into the child.
    fn subscope(&self, matter: super::subcontext::PolydatMatter<'_>) -> Result<Self, Self::Error>;
}

// ── One kernel API for every engine (engines.md §3.5) ──────

/// A kernel on any engine: the interpreter, the closure tier, the
/// hybrid kernel, or pure native code. Every engine accepts every
/// program the interpreter accepts, or refuses it at construction
/// with a reason, and computes the same values for the same inputs;
/// the choice of engine changes how fast a program runs and nothing
/// else. This trait is the surface a host drives an engine through
/// without knowing which one it has.
///
/// The interpreter kernel and the compiled kernels also keep their
/// inherent methods (raw slot readers, `eval(&[u64])`, `engine_counts`)
/// as engine-specific extras; where a name is shared, the inherent
/// method is the one a call on the concrete type reaches, and the
/// trait's is reached through `dyn Kernel` or `Kernel::pull(&mut k, …)`.
pub trait Kernel: Send + Sync + internals::KernelInternals {
    /// The engine this kernel runs on.
    fn engine(&self) -> crate::compile::select::Engine;

    /// Set the coordinate inputs for the next evaluation.
    fn set_inputs(&mut self, coords: &[u64]);

    /// Set an extern by name. One rule on every engine: the value must
    /// satisfy the declared port type (a carrier's bit-stuffed forms
    /// included) or be `None`, which clears the extern; a value of
    /// another type is refused at the write, never healed. A coordinate
    /// is set with [`Self::set_inputs`], not here. An unknown name is
    /// an error naming the known ones.
    fn set_input(&mut self, name: &str, value: Value) -> Result<(), WriteError>;

    /// Narrow a cursor to one partition: its `Ext` slot and its six
    /// scalar projections are set.
    fn set_cursor(
        &mut self,
        name: &str,
        partition: &crate::iteration::cursor_partition::Partition,
    ) -> Result<(), crate::kernel::WriteError>;

    /// Evaluate every output for the inputs set so far.
    fn eval(&mut self);

    /// The named output for the inputs set so far, evaluating what it
    /// needs and no more: the output's cone, on the interpreter, the
    /// closure tier, and the hybrid kernel alike (pure native code,
    /// being one function, evaluates the program). A side channel in
    /// the cone fires when the output is pulled; a failing node fails
    /// when pulled, with the same attributed message on every engine:
    /// the node's name, the outputs it feeds, the program's context,
    /// and its inputs. The value is owned; a handle is never returned to
    /// the host, and a slot that holds `None` reads as `None`.
    fn pull(&mut self, name: &str) -> Value;

    /// Every input by name, the coordinates first.
    fn input_names(&self) -> Vec<String>;

    /// Every named output.
    fn output_names(&self) -> Vec<String>;

    /// The declared port type of a named output.
    fn output_type(&self, name: &str) -> Option<PortType>;

    /// The externs by name and declared type.
    fn externs(&self) -> Vec<(String, PortType)>;

    /// The cursors the program declares, with the partitions the
    /// compiler resolved where it could.
    fn cursor_schemas(&self) -> &[crate::iteration::source::SourceSchema];

    /// What this kernel's engine decided for the program: how much of
    /// it runs as native segments, as closure steps, and on the
    /// interpreter. The one planning detail a kernel exposes.
    fn plan(&self) -> crate::EnginePlan;

    /// The value of a named input as the kernel holds it now, an extern
    /// or a coordinate; `None` for a name that is not an input.
    fn input_value(&self, name: &str) -> Option<Value>;

    /// The index of a named input among [`Self::input_names`], the
    /// coordinates first: what [`Self::set_input_at`] takes.
    fn input_index(&self, name: &str) -> Option<usize> {
        self.input_names().iter().position(|n| n == name)
    }

    /// [`Self::set_input`] by index, for a host that binds the same
    /// inputs every cycle: the name is resolved once, with
    /// [`Self::input_index`], and no lookup runs per write.
    fn set_input_at(&mut self, index: usize, value: Value) -> Result<(), WriteError> {
        let name =
            self.input_names()
                .get(index)
                .cloned()
                .ok_or_else(|| WriteError::UnknownWire {
                    key: format!("wire[{index}]"),
                    known: self.input_names(),
                })?;
        self.set_input(&name, value)
    }

    /// The index of a named output among [`Self::output_names`]: what
    /// [`Self::pull_at`] takes.
    fn output_index(&self, name: &str) -> Option<usize> {
        self.output_names().iter().position(|n| n == name)
    }

    /// [`Self::pull`] by index, for a host that reads the same outputs
    /// every cycle: the name is resolved once, with
    /// [`Self::output_index`], and no lookup runs per pull.
    fn pull_at(&mut self, index: usize) -> Value {
        let name = self
            .output_names()
            .get(index)
            .cloned()
            .unwrap_or_else(|| panic!("no output at index {index}"));
        self.pull(&name)
    }

    /// The `const` bindings this kernel initializes, in the order
    /// [`Self::init`] evaluates them: a const that reads another comes
    /// after it.
    fn const_inits(&self) -> &[crate::kernel::ConstInit];

    /// Write an input as part of initialization. It is
    /// [`Self::set_input_at`] except that a const's slot is accepted,
    /// which is how [`Self::init`] stores each const's value.
    fn init_input_at(&mut self, index: usize, value: Value) -> Result<(), WriteError>;

    /// Initialize the kernel: evaluate every `const` binding once, in
    /// dependency order, and store its value for the rest of the
    /// kernel's life.
    ///
    /// Every way a kernel comes into existence initializes it: a build, a
    /// kernel created from a shared program, a child bound under a
    /// parent (after the parent's values are bound), and a traversal
    /// activation (after its tuple is bound). A host calls it again when
    /// it wants the consts recomputed from the inputs as they are now,
    /// for example after setting externs a const reads. Inputs keep
    /// their values; only the consts change.
    ///
    /// A const whose expression yields `None` takes the value the binder
    /// copied from the enclosing scope, so the outer binding stays
    /// visible. A const whose expression fails makes initialization
    /// fail, naming the const; a slow one makes initialization slow.
    fn init(&mut self) -> Result<(), crate::KernelError> {
        let inits = self.const_inits().to_vec();
        for c in &inits {
            let own =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.pull(&c.source)))
                    .map_err(|payload| crate::KernelError::ConstInit {
                        name: c.name.clone(),
                        reason: crate::kernel::panic_message(&payload),
                    })?;
            let value = match own {
                Value::None => c
                    .fallback
                    .as_deref()
                    .and_then(|f| self.input_value(f))
                    .unwrap_or(Value::None),
                v => v,
            };
            // Pure native code carries no `None`: a const with no value
            // is refused there, as an unset extern is (engines.md §8),
            // rather than read as a zero.
            if let (Value::None, engine @ crate::Engine::PureNative(_)) = (&value, self.engine()) {
                return Err(crate::KernelError::Refused {
                    engine,
                    reason: format!(
                        "the const '{}' has no value, and pure native code cannot carry a \
                         `None`; give it a value or run this program on `native`",
                        c.name
                    ),
                });
            }
            let index = self
                .input_index(&c.slot)
                .expect("a const's slot is an input of its own program");
            self.init_input_at(index, value)
                .map_err(crate::KernelError::Write)?;
            // Bring the const's output up to date now, so a reader of its
            // buffer (a binder's lookup) sees the captured value.
            let _ = self.pull(&c.name);
        }
        Ok(())
    }

    /// The traversals the program declares, in document order.
    fn traversals(&self) -> &[crate::dsl::traversal::Traversal];

    /// Open the traversal at `index` against this kernel's current
    /// values (SRD 113 §3.6): the comprehension's sources see the wires
    /// they reference as this kernel holds them now, and the cascaded
    /// wires are snapshotted into every activation. On every engine
    /// (engine parity, step 8).
    fn traverse(&mut self, index: usize) -> Result<crate::kernel::TraversalStream, String>;

    /// Open every traversal, in document order.
    fn traverse_all(&mut self) -> Result<Vec<crate::kernel::TraversalStream>, String> {
        (0..self.traversals().len())
            .map(|i| self.traverse(i))
            .collect()
    }

    /// Begin the next cycle with nothing current, so every step, a side
    /// channel included, runs again when pulled. The runtime model makes
    /// a cycle whose inputs did not move cost nothing; this is how a
    /// host runs such a cycle anyway, as the `polydat` binary does when
    /// every input is fixed.
    fn invalidate_all(&mut self);

    /// The cells this kernel's `shared` bindings are bound to (scope
    /// model §6): one register per binding, which every kernel holding
    /// the cell reads and writes.
    fn shared_cells(&self) -> Vec<SharedCellEntry>;

    /// The broadcast cell for a *computed* output, created on the first
    /// ask: a descendant that binds its matching input slot to this
    /// cell reads the value each of this kernel's pulls publishes
    /// through it, rather than a copy taken once when the descendant
    /// was built (cross_fiber_invalidation.md §3.1).
    ///
    /// `None` when the name is not an output of this kernel, and on an
    /// engine that has no broadcast cells at all. The interpreter seeds
    /// one per output at construction; the closure tier and the hybrid
    /// make them on demand, so a program with no descendant bound to it
    /// allocates none.
    fn output_cell(&self, _name: &str) -> Option<SharedCell> {
        None
    }

    /// The binding modifier a named output was declared with — `const`,
    /// `shared`, `final`, or none. A binder reads it to decide how a
    /// descendant takes the output: a `const` is effectively fixed for
    /// the scope's life and is value-copied, where a computed output is
    /// bound to its broadcast cell (scope_model.md §4).
    ///
    /// `NONE` for a name this kernel does not declare.
    fn output_modifier(&self, _name: &str) -> crate::dsl::ast::BindingModifier {
        crate::dsl::ast::BindingModifier::NONE
    }

    /// Every cell a descendant of this kernel could bind to: the ones
    /// its own `shared` slots hold, plus the ones it carries forward
    /// for a descendant without holding a slot for them itself. The
    /// second kind is why an ancestral `shared` reaches a grandchild
    /// whose parent's program never names it.
    ///
    /// [`Self::shared_cells`] is the first kind alone.
    fn cells_in_scope(&self) -> Vec<SharedCellEntry> {
        self.shared_cells()
    }

    /// Carry `cells` forward for this kernel's descendants. The binder
    /// writes what the parent had and this kernel holds no slot for.
    fn set_transit_cells(&mut self, _cells: Vec<SharedCellEntry>) {}

    /// This kernel's place in the comprehension nest its scope was
    /// built under, outermost last: a child's path is its own followed
    /// by its parent's. Empty for a root, which is every kernel a host
    /// compiles rather than binds, so the compiled engines answer
    /// empty until one is bound under a parent.
    fn scope_coordinates(&self) -> &[super::ScopeCoord] {
        &[]
    }

    /// Append `outer` to this kernel's own scope-coordinate path, which
    /// the binder does once the child's inputs are in. A no-op on an
    /// engine that keeps no path.
    fn extend_scope_coordinates(&mut self, _outer: &[super::ScopeCoord]) {}

    /// The declared type of a named input slot, coordinates included.
    /// The binder reads it to adapt a value the parent supplies into
    /// the type the child's slot declares.
    fn input_port_type(&self, _name: &str) -> Option<PortType> {
        None
    }

    /// Bind the named input slot to `cell`, whether or not the slot was
    /// built as a `shared` register, and answer whether it was bound.
    ///
    /// This is what a parent does to a child, not what a host does to
    /// two kernels. A child declares its imports `extern`; it is the
    /// parent binding it that decides one of them reads a register
    /// rather than a copied value. [`Self::attach_shared_cell`] is the
    /// host's operation and refuses a slot that is not already a
    /// register on both sides, which is the right answer for joining
    /// two kernels and the wrong one for building a child.
    fn bind_input_cell(&mut self, _name: &str, _cell: SharedCell) -> bool {
        false
    }

    /// Bind the `shared` binding `name` to `cell`, so this kernel and
    /// every other holder of the cell read and write one register:
    /// a write on any of them is what the others read next, and a
    /// dependent output is recomputed. A name that is not a `shared`
    /// binding is an error naming the ones that are.
    fn attach_shared_cell(&mut self, name: &str, cell: SharedCell) -> Result<(), String>;

    /// The program this kernel runs, shareable across threads: each
    /// thread creates its own kernel from it with
    /// [`KernelProgram::create_kernel`].
    ///
    /// **What this kernel was set to does not travel with it.** A
    /// kernel created from the program starts at the program: every
    /// extern at its declared default and every `shared` binding with
    /// a cell of its own, whatever this kernel had been written to
    /// before it became one. That holds on every engine.
    ///
    /// The reason is that an extern is per-kernel state, in the same
    /// family as the coordinates: both are writes into declared slots
    /// of a running kernel, and neither is part of the compiled
    /// program. A host that wants a value fixed *for the program*
    /// fixes it before compiling, with
    /// [`transform::assign_values`](crate::dsl::transform::assign_values)
    /// or an `extern` default in the source; a host that wants every
    /// thread to see one register attaches a cell with
    /// [`Self::attach_shared_cell`].
    fn into_program(self: Box<Self>) -> std::sync::Arc<dyn KernelProgram>;

    /// The compile ledger of the program tree this kernel belongs to:
    /// what compiling it and everything opened from it has built.
    fn ledger(&self) -> &std::sync::Arc<crate::kernel::CompileLedger>;

    // ── The per-cycle scope-tree surface (native_scope_trees.md §3) ──
    //
    // Index arguments are positions in `input_names`, coordinates
    // first, resolved once by the host. None of these allocates or
    // looks a name up.

    /// How many of the inputs are coordinates: they come first in
    /// `input_names`, and `set_inputs` writes them.
    fn coord_count(&self) -> usize;

    /// The value input `index` holds now: a coordinate's pending or
    /// current value, an extern's current value. `None` past the end.
    fn input_value_at(&self, index: usize) -> Option<Value>;

    /// The value input `index` starts with: an extern's declared
    /// default, `U64(0)` for a coordinate. `None` past the end.
    fn input_default_at(&self, index: usize) -> Option<Value>;

    /// Whether input `index` is bound to a shared cell, so that its
    /// value is the cell's and a reset leaves it alone.
    fn input_is_cell_bound(&self, index: usize) -> bool;

    /// Every input that is not a coordinate and not bound to a cell
    /// back at its default, and whatever depends on a changed one not
    /// current. What a host does at a boundary where values written for
    /// the last stretch must not leak into the next.
    fn reset_inputs(&mut self);

    /// A new kernel over the same program with this kernel's state: its
    /// inputs, its current outputs, and its cells, which stay shared
    /// (a cell is the scope's register, not a value it holds), transit
    /// cells included. Callable concurrently on a kernel shared across
    /// threads (native_scope_trees.md §4).
    fn fork(&self) -> Box<dyn Kernel>;

    /// Pull every output a descendant bound to by cell, so the
    /// descendant reads the current value. Nothing happens on a kernel
    /// nothing is bound under. A failing output is left for the pull
    /// that needs it to report.
    fn publish_broadcasts(&mut self);

    /// Commit the Rule 2 write-throughs: pull each synthetic
    /// `__write_<name>` output and write it through the cell of the
    /// shared binding it exports to. No-op for a kernel without them.
    fn commit_write_throughs(&mut self) -> Result<(), String>;

    /// How input `name`'s type was established: written by the author,
    /// or inferred by the compiler and so open to
    /// `CompileOptions::input_variance` (input_variance.md §3). A host
    /// that compiles many scopes at `Info` reports each open input once
    /// from here rather than from every compile's log.
    fn input_type_origin(&self, name: &str) -> Option<crate::kernel::TypeOrigin>;

    /// The identity of this kernel's program: equal for kernels created
    /// from one program and for forks, different for any two programs,
    /// the same program compiled twice included. What a host seals a
    /// plan of pre-resolved indices against.
    fn program_id(&self) -> ProgramId;
}

/// The identity of a compiled program; see [`Kernel::program_id`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProgramId(pub(crate) usize);

/// The construction-time hooks of a kernel, sealed: the compile path
/// and the program sharing call them once, before a kernel is shared,
/// and a host neither sees nor implements them.
pub(crate) mod internals {
    use crate::ast::{PortType, Value};

    pub trait KernelInternals {
        /// Attach the traversals the program declares and the producer
        /// bindings they may traverse.
        fn set_traversals(
            &mut self,
            traversals: Vec<crate::dsl::traversal::Traversal>,
            producers: Vec<crate::dsl::traversal::Producer>,
        );

        /// The value at a buffer slot decoded as `ty`, for the compile
        /// log's record of the constants folded at build; `None` on the
        /// interpreter, whose program logs its own fold.
        fn slot_value(&self, _slot: usize, _ty: PortType) -> Value {
            Value::None
        }

        /// The value the build folded for output `name`, if it folded
        /// one: what the compile path reads to resolve a cursor extent
        /// computed from constants, on every engine.
        fn folded_value(&self, name: &str) -> Option<Value>;

        /// Record the extent of cursor `index` once the compile path
        /// has resolved it from the folded constants.
        fn set_cursor_extent(&mut self, index: usize, extent: u64);

        /// Start over from the program: every input at its declared
        /// default, every `shared` binding with a cell of its own,
        /// nothing current. What a kernel created from a shared program
        /// starts with; the interpreter's is built that way and needs
        /// nothing.
        fn reset_to_program(&mut self) {}

        /// Record the Rule 2 write-throughs this kernel commits, as
        /// `(export_name, source_output)` pairs: what a scope module
        /// hands the kernels it instantiates.
        fn set_write_throughs(&mut self, pairs: Vec<(String, String)>);
    }
}

/// A program on some engine, shared across threads through an `Arc`;
/// every kernel created from it computes the same values and owns its
/// own inputs, buffers, and outputs.
pub trait KernelProgram: Send + Sync {
    /// The engine the program was built for.
    fn engine(&self) -> crate::compile::select::Engine;

    /// A kernel of this program for the calling thread, initialized. It
    /// starts from the program on every engine: every input at its
    /// declared default, whatever the kernel that became the program had
    /// been set to; every `shared` binding with a cell of its own; and
    /// every `const` evaluated from those inputs ([`Kernel::init`]).
    ///
    /// # Panics
    ///
    /// When a const's expression fails. A const is evaluated when a
    /// kernel is initialized, so a const that fails makes creation fail;
    /// the build of the program evaluated the same consts from the same
    /// defaults, so this happens only for a const that reads something
    /// outside the program, such as a clock or the environment.
    fn create_kernel(self: std::sync::Arc<Self>) -> Box<dyn Kernel> {
        let mut kernel = self.create_uninitialized();
        if let Err(e) = kernel.init() {
            panic!("{e}");
        }
        kernel
    }

    /// A kernel of this program whose consts are not yet evaluated: what
    /// a binder creates, writes the enclosing scope's values into, and
    /// then initializes, so the consts are evaluated once, from the
    /// bound values.
    fn create_uninitialized(self: std::sync::Arc<Self>) -> Box<dyn Kernel>;

    /// The interpreter's program, when this is one: the graph a
    /// diagnostic describes node by node. `None` for a compiled
    /// engine's program.
    fn as_interpreter(
        self: std::sync::Arc<Self>,
    ) -> Option<std::sync::Arc<crate::kernel::PolydatProgram>> {
        None
    }

    /// The compile ledger of the program tree this program belongs to.
    fn ledger(&self) -> &std::sync::Arc<crate::kernel::CompileLedger>;

    /// This program's identity, equal to [`Kernel::program_id`] of every
    /// kernel created from it.
    fn program_id(&self) -> ProgramId;
}

/// A compiled kernel as a shared program: its steps are shared, and a
/// created kernel is a clone that owns its own buffer, table, scratch,
/// and externs.
pub(crate) struct SharedKernel<K>(pub(crate) K);

impl<K: Kernel + Clone + Send + Sync + 'static> KernelProgram for SharedKernel<K> {
    fn engine(&self) -> crate::compile::select::Engine {
        self.0.engine()
    }
    fn create_uninitialized(self: std::sync::Arc<Self>) -> Box<dyn Kernel> {
        let mut kernel = self.0.clone();
        // A created kernel starts from the program, as an interpreter
        // state created from one does: inputs at their defaults, cells
        // of its own; a host sets what it wants and attaches what it
        // shares.
        kernel.reset_to_program();
        Box::new(kernel)
    }
    fn ledger(&self) -> &std::sync::Arc<crate::kernel::CompileLedger> {
        self.0.ledger()
    }
    fn program_id(&self) -> ProgramId {
        self.0.program_id()
    }
}
