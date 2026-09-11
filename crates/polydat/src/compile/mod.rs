// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Kernel compilation: assembled DAG → fast executable kernel.
//!
//! Everything in this module is on the path between
//! [`assembly::PolydatAssembler`] and an executable kernel. The
//! pipeline:
//!
//! ```text
//! PolydatAssembler  ──(fusion pass)──▶  fused DAG
//!                                       │
//!                            (select::choose_kernel)
//!                                       │
//!                  ┌────────────────────┼────────────────────┐
//!                  ▼                    ▼                    ▼
//!         closures::Kernel       hybrid::Kernel       jit::Kernel
//!         (Phase 2 u64 closures) (per-node optimal)   (Phase 3 native)
//! ```
//!
//! - [`assembly`]: the public construction surface
//!   ([`assembly::PolydatAssembler`] + [`assembly::WireRef`]).
//! - [`fusion`]: graph-level subgraph fusion pass; runs during
//!   assembly after wiring resolution.
//! - [`select`]: variant-selection heuristic; chooses the
//!   monomorphic kernel type at construction time.
//! - [`closures`]: Phase 2 monomorphic u64-only kernels.
//! - [`hybrid`]: per-node optimal kernel (JIT segments + closure
//!   segments sharing a flat u64 buffer).
//! - `jit`: Phase 3 Cranelift JIT compilation
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
pub(crate) mod marshal;
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

/// The coordinates a host set last on a compiled kernel and whether
/// they have been evaluated: what the [`Kernel`](crate::kernel::Kernel)
/// trait's `set_inputs` and `pull` keep between calls.
#[derive(Clone, Default)]
pub(crate) struct Drive {
    pub(crate) coords: Vec<u64>,
    pub(crate) stale: bool,
}

/// The [`Kernel`](crate::kernel::Kernel) impl every compiled kernel
/// shares: the type's inherent `eval`, `set_input`, `set_cursor`,
/// `get_value`, `output_names`, `mark_all_dirty`, and a `core` with a
/// `drive`, `externs`, `coord_count`, and `output_types`.
macro_rules! impl_kernel_trait {
    ($ty:ident, $engine:expr) => {
        impl crate::kernel::Kernel for $ty {
            fn engine(&self) -> crate::compile::select::Engine {
                $engine
            }
            fn set_inputs(&mut self, coords: &[u64]) {
                self.core.drive.coords.clear();
                self.core.drive.coords.extend_from_slice(coords);
                self.core.drive.stale = true;
            }
            fn set_input(&mut self, name: &str, value: crate::ast::Value) -> Result<(), String> {
                self.core.drive.stale = true;
                $ty::set_input(self, name, value)
            }
            fn set_cursor(
                &mut self,
                name: &str,
                partition: &crate::iteration::cursor_partition::Partition,
            ) -> Result<(), String> {
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
            /// In declaration order, as the interpreter lists them; a kernel
            /// built without a program surface lists its slots.
            fn output_names(&self) -> Vec<String> {
                let declared = self.core.externs.output_names();
                if declared.is_empty() {
                    $ty::output_names(self)
                        .into_iter()
                        .map(String::from)
                        .collect()
                } else {
                    declared.to_vec()
                }
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
            fn traverse(&mut self, index: usize) -> Result<crate::kernel::TraversalStream, String> {
                let traversal = self.core.traversals.get(index).cloned().ok_or_else(|| {
                    format!(
                        "no traversal at index {index}; the program declares {}",
                        self.core.traversals.len()
                    )
                })?;
                crate::kernel::activation::open_traversal(self, traversal)
            }
            fn set_traversals(&mut self, traversals: Vec<crate::dsl::traversal::Traversal>) {
                self.core.traversals = traversals.into();
            }
            fn invalidate_all(&mut self) {
                self.mark_all_dirty();
                self.core.invalidate_all();
            }
            fn slot_value(&self, slot: usize, ty: crate::ast::PortType) -> crate::ast::Value {
                self.core.slot_value(slot, ty)
            }
            fn nest(&mut self) {
                self.set_owns_cycle(false);
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
            fn reseed_shared_cells(&mut self) {
                self.core.externs.reseed_cells();
            }
            fn into_program(
                mut self: Box<Self>,
            ) -> std::sync::Arc<dyn crate::kernel::KernelProgram> {
                self.mark_all_dirty();
                self.core.drive.stale = true;
                std::sync::Arc::new(crate::kernel::SharedKernel(*self))
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
/// (docs/design/engine_parity.md, step 5).
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
/// (engine_parity.md, A7). A step's panic is caught at the step
/// boundary and re-raised enriched exactly as the interpreter enriches
/// a node's: the node's name, the outputs it feeds, the program's
/// diagnostic context, and its input values decoded from the buffer
/// where the slot types allow. Step index is node index on every
/// compiled engine.
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
    /// The inputs of `step` as diagnostic text, from the buffer: `None`
    /// where the mask says so, the port type alone for a vector (as the
    /// interpreter prints one), and the port type again where the slot
    /// cannot be decoded, so the report itself never fails.
    fn inputs_of(
        &self,
        step: usize,
        buffer: &[u64],
        none: Option<&[bool]>,
        table: &crate::kernel::ValueTable,
    ) -> Vec<String> {
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
                if ty.slot_color() == crate::ast::SlotColor::Ref2 {
                    return format!("{ty:?}");
                }
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::kernel::engines::format_value_for_diag(&marshal::decode_output(
                        buffer, slot, ty, table,
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
        table: &crate::kernel::ValueTable,
    ) -> ! {
        let site = self.sites.get(step);
        let name = site
            .map(|s| s.name.clone())
            .unwrap_or_else(|| format!("<unknown node #{step}>"));
        let outputs: Vec<&str> = site
            .map(|s| s.outputs.iter().map(String::as_str).collect())
            .unwrap_or_default();
        let inputs = self.inputs_of(step, buffer, none, table);
        let enriched =
            crate::kernel::engines::enrich_panic(payload, &name, &outputs, &self.context, &inputs);
        crate::kernel::engines::reraise_enriched(enriched)
    }
}
