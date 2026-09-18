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
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteError::UnknownWire { key } => {
                write!(f, "unknown wire '{key}': no input slot by this name")
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
    #[inline]
    fn set_wire<W: WireKey>(&mut self, key: W, value: Value) -> Result<(), WriteError> {
        // Capture a string form of the key for diagnostic
        // reporting before resolution consumes it. The
        // WireKey::describe method provides this; the default
        // impl renders index keys as "wire[N]" and name keys
        // as the name itself.
        let key_desc = key.describe();
        match key.resolve(self) {
            Some(idx) => self.set_wire_idx(idx, value),
            None => Err(WriteError::UnknownWire { key: key_desc }),
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
pub trait Kernel: Send + internals::KernelInternals {
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
    fn set_input(&mut self, name: &str, value: Value) -> Result<(), String>;

    /// Narrow a cursor to one partition: its `Ext` slot and its six
    /// scalar projections are set.
    fn set_cursor(
        &mut self,
        name: &str,
        partition: &crate::iteration::cursor_partition::Partition,
    ) -> Result<(), String>;

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
    fn set_input_at(&mut self, index: usize, value: Value) -> Result<(), String> {
        let name = self
            .input_names()
            .get(index)
            .cloned()
            .ok_or_else(|| format!("no input at index {index}"))?;
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

    /// Bind the `shared` binding `name` to `cell`, so this kernel and
    /// every other holder of the cell read and write one register:
    /// a write on any of them is what the others read next, and a
    /// dependent output is recomputed. A name that is not a `shared`
    /// binding is an error naming the ones that are.
    fn attach_shared_cell(&mut self, name: &str, cell: SharedCell) -> Result<(), String>;

    /// The program this kernel runs, shareable across threads: each
    /// thread creates its own kernel from it with
    /// [`KernelProgram::create_kernel`].
    fn into_program(self: Box<Self>) -> std::sync::Arc<dyn KernelProgram>;

    /// The compile ledger of the program tree this kernel belongs to:
    /// what compiling it and everything opened from it has built.
    fn ledger(&self) -> &std::sync::Arc<crate::kernel::CompileLedger>;
}

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
    }
}

/// A program on some engine, shared across threads through an `Arc`;
/// every kernel created from it computes the same values and owns its
/// own inputs, buffers, and outputs.
pub trait KernelProgram: Send + Sync {
    /// The engine the program was built for.
    fn engine(&self) -> crate::compile::select::Engine;

    /// A kernel of this program for the calling thread. It starts from
    /// the program on every engine: every input at its declared
    /// default, whatever the kernel that became the program had been
    /// set to; every `shared` binding with a cell of its own.
    fn create_kernel(self: std::sync::Arc<Self>) -> Box<dyn Kernel>;

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
}

/// A compiled kernel as a shared program: its steps are shared, and a
/// created kernel is a clone that owns its own buffer, table, scratch,
/// and externs.
pub(crate) struct SharedKernel<K>(pub(crate) K);

impl<K: Kernel + Clone + Send + Sync + 'static> KernelProgram for SharedKernel<K> {
    fn engine(&self) -> crate::compile::select::Engine {
        self.0.engine()
    }
    fn create_kernel(self: std::sync::Arc<Self>) -> Box<dyn Kernel> {
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
}
