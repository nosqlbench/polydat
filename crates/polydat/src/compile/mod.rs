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
            fn output_names(&self) -> Vec<String> {
                $ty::output_names(self)
                    .into_iter()
                    .map(String::from)
                    .collect()
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
