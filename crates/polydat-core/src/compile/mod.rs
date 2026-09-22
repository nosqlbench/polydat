// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Kernel compilation: assembled DAG → fast executable kernel.
//!
//! Everything in this module is on the path between
//! [`assembly::PolydatAssembler`] and an executable kernel. The
//! pipeline:
//!
//! ```text
//! PolydatAssembler ──resolve──▶ ResolvedDag ──compile_with(Engine)──┐
//!   (fusion, adapters,                                              │
//!    round-trip lint,          ┌────────────────────────────────────┤
//!    topo sort)                ▼                  ▼                 ▼
//!                  Interpreter(JitMode)  Closures(Provenance)  Native(Provenance)
//!                  cone::extract_jit_cones  closures::           hybrid::
//!                  → PolydatKernel          CompiledKernel*      HybridKernel*
//! ```
//!
//! The host names the engine ([`select::Engine`]); under
//! `Provenance::Auto` the selector picks the provenance mode from the
//! graph's shape. Pure native code (`jit::JitKernel*`) is the
//! differential tier behind the hybrid kernel.
//!
//! - [`assembly`]: the public construction surface
//!   ([`assembly::PolydatAssembler`] + [`assembly::WireRef`]) and
//!   the per-engine compile paths.
//! - [`fusion`]: graph-level subgraph fusion pass; runs during
//!   assembly after wiring resolution.
//! - [`roundtrip_lint`]: the structural type-round-trip lint run at
//!   resolution.
//! - [`cone`]: cone-level JIT inside the interpreter kernel
//!   (SRD-105), under a [`cone::JitMode`].
//! - [`lattice`]: the engine-mix report of a compiled program.
//! - [`select`]: the engine and provenance enums, and the heuristic
//!   that picks a provenance mode under `Provenance::Auto`.
//! - [`closures`]: the closure tier, one generated op per node over
//!   a flat u64 slot buffer, by-reference outputs as `Ref2` pairs.
//! - [`hybrid`]: the native engine (native segments + closure steps
//!   sharing a flat u64 buffer).
//! - [`marshal`]: the boundary marshalling between slots and `Value`s.
//! - `externs`: extern inputs and `shared` cells on the compiled
//!   engines.
//! - [`simd_plan`], `simd_tier1`: scalar-flow SIMD promotion.
//! - `jit`: Cranelift lowering and the pure native kernels
//!   (feature-gated on `jit`).

pub mod assembly;
pub mod closures;
pub mod cone;
#[cfg(all(test, feature = "jit"))]
mod cone_tests;
pub(crate) mod externs;
pub mod fusion;
pub mod hybrid;
#[cfg(feature = "jit")]
pub mod jit;
pub mod lattice;
/// The boundary marshalling a compiled node kit reads its arguments
/// and writes its outputs through (compiled_handles.md §4): a node
/// crate's kits use it as the core's own do.
pub mod marshal;
pub mod roundtrip_lint;
pub mod select;
pub mod simd_plan;
#[cfg(feature = "jit")]
pub mod simd_tier1;

/// Axiom S2 typed accessors, shared by the P2 and hybrid kernel
/// types (both expose `self.core.ref_entry(slot)`). Each returns
/// a borrow whose lifetime ties to `&self`, so the borrow checker
/// statically prevents holding a slice across the next
/// `eval(&mut self)` — stale Ref reads are compile errors.
macro_rules! ref_readers {
    () => {
        /// Borrow a `vec_f32` output's current contents.
        pub fn read_vec_f32(&self, slot: usize) -> &[f32] {
            match self.core.ref_entry(slot) {
                crate::ast::ScratchBuf::F32(v) => v,
                other => panic!("slot {slot} is not f32-lane scratch: {other:?}"),
            }
        }
        /// Borrow a `vec_f64` output's current contents.
        pub fn read_vec_f64(&self, slot: usize) -> &[f64] {
            match self.core.ref_entry(slot) {
                crate::ast::ScratchBuf::F64(v) => v,
                other => panic!("slot {slot} is not f64-lane scratch: {other:?}"),
            }
        }
        /// Borrow a `vec_f16` output's current contents.
        pub fn read_vec_f16(&self, slot: usize) -> &[half::f16] {
            match self.core.ref_entry(slot) {
                crate::ast::ScratchBuf::F16(v) => v,
                other => panic!("slot {slot} is not f16-lane scratch: {other:?}"),
            }
        }
        /// Borrow a `vec_i8` output's current contents.
        pub fn read_vec_i8(&self, slot: usize) -> &[i8] {
            match self.core.ref_entry(slot) {
                crate::ast::ScratchBuf::I8(v) => v,
                other => panic!("slot {slot} is not i8-lane scratch: {other:?}"),
            }
        }
        /// Borrow a `vec_i16` output's current contents.
        pub fn read_vec_i16(&self, slot: usize) -> &[i16] {
            match self.core.ref_entry(slot) {
                crate::ast::ScratchBuf::I16(v) => v,
                other => panic!("slot {slot} is not i16-lane scratch: {other:?}"),
            }
        }
        /// Borrow a `vec_i32` output's current contents.
        pub fn read_vec_i32(&self, slot: usize) -> &[i32] {
            match self.core.ref_entry(slot) {
                crate::ast::ScratchBuf::I32(v) => v,
                other => panic!("slot {slot} is not i32-lane scratch: {other:?}"),
            }
        }
        /// Borrow a `vec_i64` output's current contents.
        pub fn read_vec_i64(&self, slot: usize) -> &[i64] {
            match self.core.ref_entry(slot) {
                crate::ast::ScratchBuf::I64(v) => v,
                other => panic!("slot {slot} is not i64-lane scratch: {other:?}"),
            }
        }
    };
}
pub(crate) use ref_readers;

/// The accessors every compiled kernel type carries, whichever tier it
/// belongs to: reading an output by name or by slot, the externs and
/// the cursors it declares, and applying the coordinates a `Kernel`
/// caller left pending before a pull or an eval. Every one of them
/// delegates to `self.core`, so none decides anything about evaluation
/// — which is why seven types can share one copy.
///
/// `$set_coords` is the coordinate writer the type uses: the provenance
/// modes that track a changed-input mask apply coordinates through
/// `set_inputs`, the rest through `set_coords`. It is the only thing
/// that varies, and the closure tier and the hybrid had a macro each to
/// vary it.
macro_rules! kernel_accessors {
    ($set_coords:ident) => {
        /// The coordinate inputs.
        pub fn coord_count(&self) -> usize {
            self.core.coord_count
        }

        /// The slot of a named output.
        pub fn resolve_output(&self, name: &str) -> Option<usize> {
            self.core.output_map.get(name).copied()
        }

        /// Read an output by pre-resolved slot index. Panics on
        /// Ref2-colored slots (axiom S2) — use `read_vec_*`.
        #[inline]
        pub fn get_slot(&self, slot: usize) -> u64 {
            self.core.guard_ref_slot(slot);
            self.core.buffer[slot]
        }

        /// Read a named output variate after `eval()`. Panics on
        /// Ref2-colored outputs (axiom S2) — use `read_vec_*`.
        #[inline]
        pub fn get(&self, name: &str) -> u64 {
            let slot = self.core.output_map[name];
            self.core.guard_ref_slot(slot);
            self.core.buffer[slot]
        }

        /// The named output as a typed `Value`, decoded by its port
        /// type: a `Ref2` output is copied out through its pair
        /// (compiled_handles.md §4), so the caller never holds a
        /// pointer; a slot that holds `None` reads as `None`.
        pub fn get_value(&self, name: &str) -> crate::ast::Value {
            self.core.value_of(name)
        }

        /// A named output, its cone run if a write is pending.
        pub fn pull_output(&mut self, name: &str) -> crate::ast::Value {
            self.core.pull_named(name)
        }

        /// The named output through the `Kernel` trait: the pending
        /// coordinates are applied, a round begins if a write is pending, and
        /// only the output's cone runs.
        fn pull_value(&mut self, name: &str) -> crate::ast::Value {
            let coords = std::mem::take(&mut self.core.drive.coords);
            self.$set_coords(&coords);
            self.core.drive.coords = coords;
            self.pull_output(name)
        }

        /// [`Self::pull_value`] by output index.
        fn pull_value_at(&mut self, index: usize) -> crate::ast::Value {
            let coords = std::mem::take(&mut self.core.drive.coords);
            self.$set_coords(&coords);
            self.core.drive.coords = coords;
            self.core.pull_at(index)
        }

        /// `eval` through the `Kernel` trait: the pending coordinates,
        /// then every step.
        fn eval_pending(&mut self) {
            let coords = std::mem::take(&mut self.core.drive.coords);
            self.eval(&coords);
            self.core.drive.coords = coords;
        }

        /// The kernel's externs by name and declared type.
        pub fn externs(&self) -> Vec<(&str, crate::ast::PortType)> {
            self.core.externs.names()
        }

        /// The cursors the program declares, with the partitions the
        /// compiler resolved where its `over` clause and extent were
        /// constant, as `PolydatProgram::cursor_schemas` reports them.
        pub fn cursor_schemas(&self) -> &[crate::iteration::source::SourceSchema] {
            self.core.externs.cursor_schemas()
        }

        /// Narrow a cursor to one partition, as `narrow_cursor` does on
        /// the interpreter: its `Ext` slot and six scalar projections
        /// are set as externs.
        pub fn set_cursor(
            &mut self,
            name: &str,
            partition: &crate::iteration::cursor_partition::Partition,
        ) -> Result<(), crate::kernel::WriteError> {
            for (slot, value) in self.core.externs.cursor_writes(name, partition)? {
                self.set_input(&slot, value)?;
            }
            Ok(())
        }

        crate::compile::ref_readers!();
    };
}
pub(crate) use kernel_accessors;

/// SRD-74's fusion rule: whether a node may join the run of fused
/// code being formed, as far as `None` is concerned. The one predicate
/// both fusers apply — the interpreter's cone planner and the hybrid's
/// segment batcher — so that what one admits the other admits.
///
/// Fused code answers a `None` on a boundary input by making every one
/// of its outputs `None`, because native code cannot carry one. That is
/// SRD-74 Rule 1 and it is the right answer for a node that propagates
/// a `None`. It is the wrong answer for a node that *consumes* one and
/// keeps going — `to_json` writes `null`, a `printf` with an `Option`
/// argument writes its own text — so such a node may join only when
/// every one of its inputs comes from inside, where a `None` cannot
/// arrive: either no boundary input is `None` and the run proceeds
/// normally, or one is and the whole run answers `None` without any
/// member running at all.
///
/// `eligible` is indexed by node and filled in topological order, so a
/// node's producers are decided before it is.
#[cfg(feature = "jit")]
pub(crate) fn none_rule_admits(
    accepts_none: bool,
    wiring: &[crate::kernel::WireSource],
    eligible: &[bool],
) -> bool {
    !accepts_none
        || wiring
            .iter()
            .all(|src| matches!(src, crate::kernel::WireSource::NodeOutput(j, _) if eligible[*j]))
}

/// The highest tier a node can reach, given the types of the wires
/// feeding it: the one answer to a question four places were asking
/// separately.
///
/// The order is the one every builder walks. Native first, since a node
/// with a lowering joins a segment; then the compiled forms, in the
/// order `assembly::node_step_op` tries them — a copy step, the scalar
/// `compiled_u64`, the node's own slot kit; then the interpreter.
///
/// The wire types are not optional. `compiled_slot` is offered per call
/// site with the types the kernel fixed, so a caller without them can
/// only ask the first two questions, and the three callers that
/// reported a tier rather than selecting one did exactly that — they
/// asked `compiled_u64().is_some()` and called a node with a slot kit
/// `Phase1`, which is what the binary printed for `printf`.
pub fn node_tier(
    node: &dyn crate::ast::PolydatNode,
    wire_types: &[crate::ast::PortType],
) -> crate::ast::CompileLevel {
    #[cfg(feature = "jit")]
    if !matches!(
        crate::compile::jit::classify_node_typed(node, wire_types),
        crate::compile::jit::JitOp::Fallback
    ) {
        return crate::ast::CompileLevel::Phase3;
    }
    let meta = node.meta();
    let is_copy =
        (meta.name == "identity" || meta.name.starts_with("__port_")) && meta.outs.len() == 1;
    let has_kit = node
        .compiled_slot(
            wire_types,
            crate::compile::select::Engine::Closures(crate::compile::select::Provenance::Auto),
        )
        .is_some();
    if is_copy || node.compiled_u64().is_some() || has_kit {
        crate::ast::CompileLevel::Phase2
    } else {
        crate::ast::CompileLevel::Phase1
    }
}

/// The provenance of every buffer slot: which input slots reach it,
/// as an exact multi-word mask, for the pull-side cone guard of every
/// compiled kernel. `input_dependents` is indexed by input slot (a
/// multi-slot input repeats its list per slot) and lists the steps
/// downstream of that slot; `step_output_slots` gives each step's
/// output slots, which all take the step's mask. A coordinate slot's
/// provenance is itself.
pub(crate) fn slot_provenance(
    coord_count: usize,
    total_slots: usize,
    step_output_slots: &[&[usize]],
    input_dependents: &[Vec<usize>],
) -> Vec<crate::kernel::ProvMask> {
    use crate::kernel::ProvMask;
    let step_count = step_output_slots.len();
    let mut step_prov: Vec<ProvMask> = (0..step_count).map(|_| ProvMask::empty()).collect();
    for (input_slot, deps) in input_dependents.iter().enumerate() {
        for &step in deps {
            if step < step_count {
                step_prov[step].set(input_slot);
            }
        }
    }
    let mut slots: Vec<ProvMask> = (0..total_slots).map(|_| ProvMask::empty()).collect();
    for (i, slot) in slots.iter_mut().enumerate().take(coord_count) {
        slot.set(i);
    }
    for (step, outs) in step_output_slots.iter().enumerate() {
        for &slot in outs.iter() {
            if slot < slots.len() {
                slots[slot] = step_prov[step].clone();
            }
        }
    }
    slots
}

/// The coordinates a host set last on a compiled kernel and whether
/// they have been evaluated: what the [`Kernel`](crate::kernel::Kernel)
/// trait's `set_inputs` and `pull` keep between calls.
#[derive(Clone, Default)]
pub(crate) struct Drive {
    pub(crate) coords: Vec<u64>,
    pub(crate) stale: bool,
}

/// The slot surface of a compiled kernel: the extended API, over and
/// above the [`Kernel`](crate::kernel::Kernel) trait every engine
/// answers.
///
/// Every compiled engine lays its program out over one flat `u64` slot
/// buffer (engines.md §6). That layout is an implementation detail, and
/// this trait is where it is admitted: a slot index instead of an
/// output name, a raw `u64` instead of a `Value`, a borrow into the
/// scratch a by-reference output writes. The interpreter does not
/// implement it and cannot — its buffers are typed `Value`s and it has
/// no slot to name — which is the point: the shape of this trait *is*
/// the thing the compiled tiers share and the interpreter does not.
///
/// **This is not the surface for running a program.** Driving a kernel
/// is `Kernel`, on every engine, and a host that never names an engine
/// never sees this trait. Reach for it when the implementation detail
/// is the subject: a differential test asserting on what was laid out,
/// a benchmark measuring a tier without the `Value` construction and
/// the name lookup a `pull` pays, a diagnostic reporting on a slot.
///
/// It is a subtrait rather than a wider `Kernel`, so it is opt-in at
/// the import: a caller who does not write `use SlotKernel` does not
/// have these methods on their kernel at all. And it is reachable
/// without naming a kernel type, through
/// [`PolydatAssembler::compile_slots`](crate::compile::assembly::PolydatAssembler::compile_slots),
/// which hands back a `Box<dyn SlotKernel>` that upcasts to
/// `Box<dyn Kernel>` wherever the ordinary surface will do.
pub trait SlotKernel: crate::kernel::Kernel {
    /// The buffer slot a named output writes, resolved once so a
    /// caller reading the same output every cycle pays no lookup.
    fn resolve_output(&self, name: &str) -> Option<usize>;

    /// The raw `u64` in `slot`, as it stands: no evaluation, no
    /// decoding. Panics on a `Ref2` slot (axiom S2), which has no
    /// scalar to read — use the `read_vec_*` borrows.
    fn get_slot(&self, slot: usize) -> u64;

    /// [`Self::get_slot`] by output name.
    fn get(&self, name: &str) -> u64;

    /// A named output decoded by its port type, a `Ref2` output copied
    /// out through its pair so the caller never holds a pointer. Reads
    /// what is there; [`Kernel::pull`](crate::kernel::Kernel::pull)
    /// evaluates first.
    fn get_value(&self, name: &str) -> crate::ast::Value;

    /// Set the coordinates, evaluate what `slot` needs, and return its
    /// raw `u64`. The whole read in one call and one `u64`, which is
    /// what a tier benchmark wants: `pull_at` gives the same value
    /// through a `Value` it has to construct.
    fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64;

    /// Set the coordinates and run every step, the whole program in
    /// one call. [`Kernel::eval`](crate::kernel::Kernel::eval) is the
    /// same evaluation over coordinates already written with
    /// `set_inputs`; this is the form that takes them, which is what a
    /// loop over a coordinate range wants.
    fn eval_at(&mut self, coords: &[u64]);

    /// Borrow a `vec_f32` output's current contents. The borrow ties to
    /// `&self`, so holding one across the next evaluation is a compile
    /// error rather than a stale read (axiom S2).
    fn read_vec_f32(&self, slot: usize) -> &[f32];
    /// Borrow a `vec_f64` output's current contents.
    fn read_vec_f64(&self, slot: usize) -> &[f64];
    /// Borrow a `vec_f16` output's current contents.
    fn read_vec_f16(&self, slot: usize) -> &[half::f16];
    /// Borrow a `vec_i8` output's current contents.
    fn read_vec_i8(&self, slot: usize) -> &[i8];
    /// Borrow a `vec_i16` output's current contents.
    fn read_vec_i16(&self, slot: usize) -> &[i16];
    /// Borrow a `vec_i32` output's current contents.
    fn read_vec_i32(&self, slot: usize) -> &[i32];
    /// Borrow a `vec_i64` output's current contents.
    fn read_vec_i64(&self, slot: usize) -> &[i64];
}

/// [`SlotKernel`] for a compiled kernel, forwarding to the inherent
/// methods the type already has. The trait is the surface; the
/// inherent copies are what it forwards to and what this crate calls.
macro_rules! impl_slot_kernel {
    ($ty:ident) => {
        impl crate::compile::SlotKernel for $ty {
            fn resolve_output(&self, name: &str) -> Option<usize> {
                $ty::resolve_output(self, name)
            }
            fn get_slot(&self, slot: usize) -> u64 {
                $ty::get_slot(self, slot)
            }
            fn get(&self, name: &str) -> u64 {
                $ty::get(self, name)
            }
            fn get_value(&self, name: &str) -> crate::ast::Value {
                $ty::get_value(self, name)
            }
            fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
                $ty::eval_for_slot(self, coords, slot)
            }
            fn eval_at(&mut self, coords: &[u64]) {
                $ty::eval(self, coords)
            }
            fn read_vec_f32(&self, slot: usize) -> &[f32] {
                $ty::read_vec_f32(self, slot)
            }
            fn read_vec_f64(&self, slot: usize) -> &[f64] {
                $ty::read_vec_f64(self, slot)
            }
            fn read_vec_f16(&self, slot: usize) -> &[half::f16] {
                $ty::read_vec_f16(self, slot)
            }
            fn read_vec_i8(&self, slot: usize) -> &[i8] {
                $ty::read_vec_i8(self, slot)
            }
            fn read_vec_i16(&self, slot: usize) -> &[i16] {
                $ty::read_vec_i16(self, slot)
            }
            fn read_vec_i32(&self, slot: usize) -> &[i32] {
                $ty::read_vec_i32(self, slot)
            }
            fn read_vec_i64(&self, slot: usize) -> &[i64] {
                $ty::read_vec_i64(self, slot)
            }
        }
    };
}
pub(crate) use impl_slot_kernel;

/// The [`Kernel`](crate::kernel::Kernel) impl every compiled kernel
/// shares: the type's inherent `eval_pending`, `pull_value`,
/// `pull_value_at`, `set_input`, `set_input_at`, `set_cursor`,
/// `mark_all_dirty`, and a `core` with `drive`, `externs`,
/// `coord_count`, `output_types`, `output_map`, `buffer`,
/// `traversals`, and `plan`/`invalidate_all`/`attach_cell`/
/// `slot_value`.
macro_rules! impl_kernel_trait {
    ($ty:ident) => {
        impl crate::kernel::Kernel for $ty {
            fn engine(&self) -> crate::compile::select::Engine {
                self.core.engine
            }
            fn set_inputs(&mut self, coords: &[u64]) {
                self.core.drive.coords.clear();
                self.core.drive.coords.extend_from_slice(coords);
                self.core.drive.stale = true;
            }
            fn set_input(
                &mut self,
                name: &str,
                value: crate::ast::Value,
            ) -> Result<(), crate::kernel::WriteError> {
                self.core.drive.stale = true;
                $ty::set_input(self, name, value)
            }
            fn set_cursor(
                &mut self,
                name: &str,
                partition: &crate::iteration::cursor_partition::Partition,
            ) -> Result<(), crate::kernel::WriteError> {
                self.core.drive.stale = true;
                $ty::set_cursor(self, name, partition)
            }
            fn eval(&mut self) {
                self.eval_pending();
                self.core.drive.stale = false;
            }
            fn pull(&mut self, name: &str) -> crate::ast::Value {
                self.pull_value(name)
            }
            fn input_names(&self) -> Vec<String> {
                self.core.externs.input_names().to_vec()
            }
            /// In declaration order, as the interpreter lists them: the
            /// assembler sets them on every compiled kernel.
            fn output_names(&self) -> Vec<String> {
                self.core.externs.output_names().to_vec()
            }
            fn output_type(&self, name: &str) -> Option<crate::ast::PortType> {
                self.core.output_types.get(name).copied()
            }
            fn externs(&self) -> Vec<(String, crate::ast::PortType)> {
                self.core
                    .externs
                    .names()
                    .into_iter()
                    .map(|(n, t)| (n.to_string(), t))
                    .collect()
            }
            fn cursor_schemas(&self) -> &[crate::iteration::source::SourceSchema] {
                self.core.externs.cursor_schemas()
            }
            fn input_value(&self, name: &str) -> Option<crate::ast::Value> {
                self.core.externs.value(name).or_else(|| {
                    let i = self
                        .core
                        .externs
                        .input_names()
                        .iter()
                        .position(|n| n == name)?;
                    if i < self.core.coord_count {
                        let pending = self.core.drive.coords.get(i).copied();
                        Some(crate::ast::Value::U64(
                            pending.unwrap_or(self.core.buffer[i]),
                        ))
                    } else {
                        None
                    }
                })
            }
            fn traversals(&self) -> &[crate::dsl::traversal::Traversal] {
                &self.core.traversals
            }
            fn plan(&self) -> crate::EnginePlan {
                self.core.plan()
            }
            fn input_index(&self, name: &str) -> Option<usize> {
                self.core
                    .externs
                    .input_names()
                    .iter()
                    .position(|n| n == name)
            }
            fn set_input_at(
                &mut self,
                index: usize,
                value: crate::ast::Value,
            ) -> Result<(), crate::kernel::WriteError> {
                self.core.drive.stale = true;
                $ty::set_input_at(self, index, value)
            }
            fn output_index(&self, name: &str) -> Option<usize> {
                self.core
                    .externs
                    .output_names()
                    .iter()
                    .position(|n| n == name)
            }
            fn pull_at(&mut self, index: usize) -> crate::ast::Value {
                self.pull_value_at(index)
            }
            fn traverse(&mut self, index: usize) -> Result<crate::kernel::TraversalStream, String> {
                let traversal = self.core.traversals.get(index).cloned().ok_or_else(|| {
                    format!(
                        "no traversal at index {index}; the program declares {}",
                        self.core.traversals.len()
                    )
                })?;
                crate::kernel::activation::open_traversal(self, traversal)
            }
            fn invalidate_all(&mut self) {
                self.mark_all_dirty();
                self.core.invalidate_all();
            }
            fn shared_cells(&self) -> Vec<crate::kernel::SharedCellEntry> {
                self.core.externs.shared_cells()
            }
            fn output_cell(&mut self, name: &str) -> Option<crate::kernel::SharedCell> {
                self.core.output_cell_for(name)
            }
            fn output_modifier(&self, name: &str) -> crate::dsl::ast::BindingModifier {
                self.core.externs.output_modifier(name)
            }
            fn cells_in_scope(&self) -> Vec<crate::kernel::SharedCellEntry> {
                self.core.externs.cells_in_scope()
            }
            fn set_transit_cells(&mut self, cells: Vec<crate::kernel::SharedCellEntry>) {
                self.core.externs.set_transit_cells(cells);
            }
            fn attach_shared_cell(
                &mut self,
                name: &str,
                cell: crate::kernel::SharedCell,
            ) -> Result<(), String> {
                self.core.attach_cell(name, cell)
            }
            fn into_program(
                mut self: Box<Self>,
            ) -> std::sync::Arc<dyn crate::kernel::KernelProgram> {
                self.mark_all_dirty();
                self.core.drive.stale = true;
                std::sync::Arc::new(crate::kernel::SharedKernel(*self))
            }
            fn ledger(&self) -> &std::sync::Arc<crate::kernel::CompileLedger> {
                self.core.externs.ledger()
            }
        }

        impl crate::kernel::KernelInternals for $ty {
            /// A compiled kernel keeps the traversals; each carries the
            /// comprehension its producer resolved to at compile time.
            fn set_traversals(
                &mut self,
                traversals: Vec<crate::dsl::traversal::Traversal>,
                _producers: Vec<crate::dsl::traversal::Producer>,
            ) {
                self.core.traversals = traversals.into();
            }
            fn slot_value(&self, slot: usize, ty: crate::ast::PortType) -> crate::ast::Value {
                self.core.slot_value(slot, ty)
            }
            fn folded_value(&self, name: &str) -> Option<crate::ast::Value> {
                let slot = *self.core.output_map.get(name)?;
                let ty = *self.core.output_types.get(name)?;
                Some(self.core.slot_value(slot, ty))
            }
            fn set_cursor_extent(&mut self, index: usize, extent: u64) {
                self.core.externs.set_cursor_extent(index, extent);
            }
            fn reset_to_program(&mut self) {
                self.core.externs.reset_to_program(&mut self.core.buffer);
                self.mark_all_dirty();
            }
        }
    };
}
pub(crate) use impl_kernel_trait;

/// The bookkeeping every compiled engine keeps, whatever its steps
/// are: the evaluation round and what ran in it, the clean flags and
/// what a write dirties, the extern writes and the cell refresh, the
/// reference pairs a step publishes, and reading an output back.
///
/// Both compiled cores carry the same fields for these and, until this
/// macro, the same eighteen method bodies byte for byte. None of them
/// touches the step list, which is the one thing the two tiers
/// genuinely differ about: a step on the closure tier is always a
/// closure, and on the native tier it is a closure or a run of native
/// code. That difference lives in the run loops, which stay per tier.
macro_rules! shared_core_methods {
    () => {
        /// Axiom S9: every reference pair in the buffer names the
        /// scratch entry that owns it. A slot is skipped when nothing
        /// has been published into it — its step has not run, or it
        /// carries `None`.
        ///
        /// The two tiers wrote this assertion separately and their
        /// skip predicates had drifted apart: one skipped a `None`
        /// slot only when a step owned it, the other whenever the
        /// slot was `None`. Nothing is published either way, so the
        /// looser test is the right one and is now the only one.
        ///
        /// Axiom S2 typed accessor core: resolve a Ref pair's first
        /// slot to its kernel-owned scratch entry. The returned
        /// borrow ties to `&self`, so holding it across the next
        /// `eval(&mut self)` is a compile error — stale reads are
        /// statically impossible.
        fn ref_entry(&self, slot: usize) -> &crate::ast::ScratchBuf {
            match self.ref_scratch.iter().find(|(s, _)| *s == slot) {
                Some(&(_, idx)) => &self.scratch[idx],
                None if self.ref_slots.get(slot).copied().unwrap_or(false) => panic!(
                    "slot {slot} is a Ref pair owned by the CALLER (a kernel \
                     input) — read it on the caller side"
                ),
                None => panic!("slot {slot} is not a Ref2-colored slot"),
            }
        }

        /// Run `body` with the failure path armed: a panic inside a
        /// step is recorded quietly and re-raised enriched, as the
        /// interpreter re-raises a node's (A7), and every reference
        /// pair is checked afterwards in a debug build.
        ///
        /// `#[inline]` is load-bearing: this wraps every `eval`, and
        /// without it the native rung of the ladder pays a call and
        /// about seven percent.
        #[inline]
        fn run_guarded(&mut self, body: impl FnOnce(&mut Self)) {
            let capture = crate::kernel::engines::EvalPanicCaptureGuard::arm();
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(self)));
            drop(capture);
            if let Err(payload) = outcome {
                let sites = std::sync::Arc::clone(&self.sites);
                let node = self.failing_node();
                sites.reraise(payload, node, &self.buffer, Some(&self.none));
            }
            #[cfg(debug_assertions)]
            self.validate_refs();
        }

        /// Gated to `debug_assertions` to match its call sites, which
        /// compile out in release.
        #[cfg(debug_assertions)]
        fn validate_refs(&self) {
            for &(slot, idx) in &self.ref_scratch {
                let unpublished = self.none[slot]
                    || matches!(self.slot_step.get(slot), Some(Some(step)) if self.ran[*step] == 0);
                if unpublished {
                    continue;
                }
                let (p, l) = self.scratch[idx].ptr_len();
                assert!(
                    self.buffer[slot] == p && self.buffer[slot + 1] == l,
                    "S9 ref-validator: slot pair ({slot}, {}) = ({:#x}, {}) \
                     does not match scratch[{idx}] = ({p:#x}, {l}) — a slot \
                     op failed to republish or wrote the wrong slots",
                    slot + 1,
                    self.buffer[slot],
                    self.buffer[slot + 1],
                );
            }
        }

        fn attach_cell(
            &mut self,
            name: &str,
            cell: crate::kernel::SharedCell,
        ) -> Result<(), String> {
            let slot = self.externs.attach_cell(name, cell)?;
            self.dirty_input(slot);
            self.drive.stale = true;
            Ok(())
        }

        #[inline]
        fn begin_epoch(&mut self) {
            if self.externs.cells_dirty() {
                self.externs.refresh_cells(&mut self.buffer);
            }
            self.dirty_refreshed();
            self.epoch += 1;
            self.all_ran = false;
            for &i in self.volatile_steps.iter() {
                self.clean[i] = false;
            }
            self.drive.stale = false;
        }

        #[inline]
        fn dirty_input(&mut self, slot: usize) {
            if let Some(deps) = self.dirty.get(slot) {
                for &i in deps {
                    self.clean[i] = false;
                }
            }
        }

        #[inline]
        fn dirty_refreshed(&mut self) {
            if !self.externs.has_changed() {
                return;
            }
            let changed = self.externs.take_changed();
            for &slot in &changed {
                if let Some(deps) = self.plan.input_dependents.get(slot) {
                    for &i in deps {
                        self.ran[i] = 0;
                        self.clean[i] = false;
                    }
                    self.all_ran = false;
                }
            }
            self.externs.return_changed(changed);
        }

        #[inline]
        fn eval_all(&mut self) {
            let fresh = self.drive.stale;
            if fresh {
                self.begin_epoch();
            } else {
                self.refresh_cells();
            }
            if fresh && !self.use_clean && !self.any_none {
                self.run_guarded(|core| core.run_fresh());
            } else {
                let all = std::sync::Arc::clone(&self.all);
                self.run_steps(&all);
            }
        }

        fn extern_written(&mut self, slot: usize, unset: bool) {
            self.none[slot] = unset;
            let was = self.any_none;
            self.any_none = self.externs.any_unset();
            if was && !self.any_none {
                self.none.fill(false);
            }
            self.dirty_input(slot);
            self.drive.stale = true;
        }

        #[inline]
        fn guard_ref_slot(&self, slot: usize) {
            if self.ref_slots.get(slot).copied().unwrap_or(false) {
                panic!(
                    "S2 pointer containment: slot {slot} is Ref2-colored; raw u64 readers \
                     would leak an interior address. Use the typed borrow-checked accessor \
                     (read_vec_*), the boundary decode, or copy out."
                );
            }
        }

        fn invalidate_all(&mut self) {
            self.clean.fill(false);
            self.all_ran = false;
            self.drive.stale = true;
        }

        fn pull_at(&mut self, index: usize) -> crate::ast::Value {
            if self.resolved_outputs.len() <= index {
                self.resolved_outputs.resize(index + 1, None);
            }
            if self.resolved_outputs[index].is_none() {
                let name = self
                    .externs
                    .output_names()
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| {
                        panic!(
                            "no output at index {index}; this kernel declares {}",
                            self.externs.output_names().len()
                        )
                    });
                let slot = self.output_map[&name];
                let ty = self
                    .output_types
                    .get(&name)
                    .copied()
                    .unwrap_or(crate::ast::PortType::U64);
                let cone = self
                    .plan
                    .cones
                    .get(&name)
                    .map(|c| std::sync::Arc::from(c.as_slice()));
                self.resolved_outputs[index] = Some((slot, ty, cone));
            }
            if self.drive.stale {
                self.begin_epoch();
            } else {
                self.refresh_cells();
            }
            let (slot, ty, cone) = self.resolved_outputs[index]
                .clone()
                .expect("resolved above");
            if let Some(order) = cone {
                self.run_steps(&order);
            }
            self.slot_value(slot, ty)
        }

        fn pull_named(&mut self, name: &str) -> crate::ast::Value {
            if self.drive.stale {
                self.begin_epoch();
            } else {
                self.refresh_cells();
            }
            let plan = std::sync::Arc::clone(&self.plan);
            if let Some(order) = plan.cones.get(name) {
                self.run_steps(order);
            }
            let value = self.value_of(name);
            self.broadcast(name, &value);
            value
        }

        /// Publish a freshly computed output through its broadcast cell,
        /// if a descendant asked for one, so a child that bound its
        /// matching input slot to the same cell reads the new value
        /// (cross_fiber_invalidation.md §3.1, "broadcast outputs").
        ///
        /// A program nobody built a subscope under has no cells at all,
        /// and pays the emptiness check.
        #[inline]
        fn broadcast(&mut self, name: &str, value: &crate::ast::Value) {
            if !self.externs.broadcasts() {
                return;
            }
            if let Some(&slot) = self.output_map.get(name)
                && let Some(cell) = self.externs.published_output(slot)
            {
                cell.publish(value.clone());
            }
        }

        /// The broadcast cell for a named output, created on the first
        /// ask. The interpreter seeds one per output at construction;
        /// a compiled kernel makes them only when a descendant binds to
        /// one, so a program with no subscope under it allocates none.
        ///
        /// Keyed by the output's slot, which is what `output_map`
        /// answers and what the buffer is indexed by, so the vector is
        /// as long as the buffer rather than as long as the output list.
        fn output_cell_for(&mut self, name: &str) -> Option<crate::kernel::SharedCell> {
            let slot = *self.output_map.get(name)?;
            let initial = self.value_of(name);
            Some(self.externs.output_cell(slot, initial))
        }

        #[inline]
        fn refresh_cells(&mut self) {
            if self.externs.cells_dirty() {
                self.externs.refresh_cells(&mut self.buffer);
                self.dirty_refreshed();
            }
        }

        fn republish_refs(&mut self) {
            for &(slot, idx) in &self.ref_scratch {
                let (p, l) = self.scratch[idx].ptr_len();
                self.buffer[slot] = p;
                self.buffer[slot + 1] = l;
            }
            self.externs.seed(&mut self.buffer, None);
        }

        #[inline]
        fn run_steps(&mut self, order: &[usize]) {
            self.run_guarded(|core| core.run_order(order));
        }

        /// `run_steps` for the build-time constant fold: the same steps
        /// under the same guard, but a failure comes back as the
        /// message [`Attribution::reraise`] would have raised. A step
        /// that fails here fails before any kernel exists, so it is an
        /// error the builder returns rather than a panic out of a
        /// constructor.
        fn fold_steps(&mut self, order: &[usize]) -> Result<(), crate::KernelError> {
            let capture = crate::kernel::engines::EvalPanicCaptureGuard::arm();
            let outcome =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.run_order(order)));
            drop(capture);
            if let Err(payload) = outcome {
                let sites = std::sync::Arc::clone(&self.sites);
                let node = self.failing_node();
                return Err(crate::KernelError::ConstantFold {
                    reason: sites.describe(payload, node, &self.buffer, Some(&self.none)),
                });
            }
            Ok(())
        }

        fn set_extern(
            &mut self,
            name: &str,
            value: crate::ast::Value,
        ) -> Result<usize, crate::kernel::WriteError> {
            let (slot, unset) = self.externs.set(name, value, &mut self.buffer)?;
            self.extern_written(slot, unset);
            Ok(slot)
        }

        fn set_extern_at(
            &mut self,
            index: usize,
            value: crate::ast::Value,
        ) -> Result<usize, crate::kernel::WriteError> {
            let (slot, unset) = self.externs.set_at(index, value, &mut self.buffer)?;
            self.extern_written(slot, unset);
            Ok(slot)
        }

        fn slot_value(&self, slot: usize, ty: crate::ast::PortType) -> crate::ast::Value {
            if self.none.get(slot).copied().unwrap_or(false) {
                return crate::ast::Value::None;
            }
            crate::compile::marshal::decode_output(&self.buffer, slot, ty)
        }

        fn value_of(&self, name: &str) -> crate::ast::Value {
            let slot = self.output_map[name];
            let ty = self
                .output_types
                .get(name)
                .copied()
                .unwrap_or(crate::ast::PortType::U64);
            self.slot_value(slot, ty)
        }
    };
}
pub(crate) use shared_core_methods;

/// The dirty-register plan of a compiled kernel: which steps each input
/// slot invalidates when it changes, and which steps each named output
/// needs. The evaluation loops consume only this; provenance derives it
/// today, and a host that knows its write and read patterns may supply
/// a narrower plan later without touching the loops
/// (docs/design/engines.md §3.1).
pub(crate) struct Invalidation {
    /// Per input slot (coordinates and externs alike), the steps that
    /// depend on it, transitively.
    pub(crate) input_dependents: Vec<Vec<usize>>,
    /// Per named output, the steps of its cone in evaluation order.
    pub(crate) cones: std::collections::HashMap<String, Vec<usize>>,
}

impl Invalidation {
    /// The plan provenance gives: every step downstream of an input is
    /// invalidated by it, and every step upstream of an output is in
    /// its cone. `inputs` and `outputs` are each step's slots;
    /// `output_slots` names the outputs.
    pub(crate) fn from_provenance(
        input_dependents: Vec<Vec<usize>>,
        step_inputs: &[&[usize]],
        step_outputs: &[&[usize]],
        output_slots: &std::collections::HashMap<String, usize>,
        total_slots: usize,
    ) -> Self {
        let step_count = step_inputs.len();
        let mut slot_step: Vec<Option<usize>> = vec![None; total_slots];
        for (i, outs) in step_outputs.iter().enumerate() {
            for &s in outs.iter() {
                slot_step[s] = Some(i);
            }
        }
        let cones = output_slots
            .iter()
            .map(|(name, &slot)| {
                let mut wanted = vec![false; step_count];
                let mut stack: Vec<usize> = slot_step[slot].into_iter().collect();
                while let Some(i) = stack.pop() {
                    if wanted[i] {
                        continue;
                    }
                    wanted[i] = true;
                    stack.extend(step_inputs[i].iter().filter_map(|&s| slot_step[s]));
                }
                (
                    name.clone(),
                    (0..step_count).filter(|&i| wanted[i]).collect(),
                )
            })
            .collect();
        Self {
            input_dependents,
            cones,
        }
    }
}

/// Where each compiled step came from, for the failure path only
/// (engines.md §3.4). A step's panic is caught at the step
/// boundary and re-raised enriched exactly as the interpreter enriches
/// a node's: the node's name, the outputs it feeds, the program's
/// diagnostic context, and its input values decoded from the buffer
/// where the slot types allow. `sites` is indexed by program node:
/// the closure and pure-native kernels have one step per node, and
/// the hybrid kernel names the failing member of a segment through
/// its tracker slot.
#[derive(Default)]
pub(crate) struct Attribution {
    pub(crate) sites: Vec<NodeSite>,
    /// The program's diagnostic context (`PolydatProgram::context`).
    pub(crate) context: String,
}

/// One node's identity for the failure path.
pub(crate) struct NodeSite {
    pub(crate) name: String,
    /// The declared outputs the node feeds, sorted.
    pub(crate) outputs: Vec<String>,
    /// `(first slot, port type)` of every input port, in port order.
    pub(crate) inputs: Vec<(usize, crate::ast::PortType)>,
}

impl Attribution {
    /// The inputs of `step` as diagnostic text, from the buffer, each
    /// copied out and printed as the interpreter prints the same value:
    /// `None` where the mask says so, and the port type alone where the
    /// slot cannot be decoded, so the report itself never fails.
    fn inputs_of(&self, step: usize, buffer: &[u64], none: Option<&[bool]>) -> Vec<String> {
        let Some(site) = self.sites.get(step) else {
            return Vec::new();
        };
        let _quiet = crate::kernel::engines::EvalPanicCaptureGuard::arm();
        site.inputs
            .iter()
            .map(|&(slot, ty)| {
                if none.is_some_and(|m| m.get(slot).copied().unwrap_or(false)) {
                    return "None".to_string();
                }
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::kernel::engines::format_value_for_diag(&marshal::decode_output(
                        buffer, slot, ty,
                    ))
                }))
                .unwrap_or_else(|_| format!("{ty:?}"))
            })
            .collect()
    }

    /// Re-raise a step's panic enriched as the interpreter enriches a
    /// node's (`kernel::engines::enrich_panic`). `step` beyond the
    /// sites (native code that failed before naming a step) reports an
    /// unknown node, as the interpreter does for an index it lacks.
    pub(crate) fn reraise(
        &self,
        payload: Box<dyn std::any::Any + Send>,
        step: usize,
        buffer: &[u64],
        none: Option<&[bool]>,
    ) -> ! {
        crate::kernel::engines::reraise_enriched(self.describe(payload, step, buffer, none))
    }

    /// The same message [`Self::reraise`] raises, returned instead. The
    /// build-time constant fold uses it: a step that fails there fails
    /// before any kernel exists, so it is an error the builder returns
    /// and not a panic out of a constructor.
    pub(crate) fn describe(
        &self,
        payload: Box<dyn std::any::Any + Send>,
        step: usize,
        buffer: &[u64],
        none: Option<&[bool]>,
    ) -> String {
        let site = self.sites.get(step);
        let name = site
            .map(|s| s.name.clone())
            .unwrap_or_else(|| format!("<unknown node #{step}>"));
        let outputs: Vec<&str> = site
            .map(|s| s.outputs.iter().map(String::as_str).collect())
            .unwrap_or_default();
        let inputs = self.inputs_of(step, buffer, none);
        crate::kernel::engines::enrich_panic(payload, &name, &outputs, &self.context, &inputs)
    }
}
