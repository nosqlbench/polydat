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
        let site = self.sites.get(step);
        let name = site
            .map(|s| s.name.clone())
            .unwrap_or_else(|| format!("<unknown node #{step}>"));
        let outputs: Vec<&str> = site
            .map(|s| s.outputs.iter().map(String::as_str).collect())
            .unwrap_or_default();
        let inputs = self.inputs_of(step, buffer, none);
        let enriched =
            crate::kernel::engines::enrich_panic(payload, &name, &outputs, &self.context, &inputs);
        crate::kernel::engines::reraise_enriched(enriched)
    }
}
