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
/// macro, the same seventeen method bodies byte for byte. None of them
/// touches the step list, which is the one thing the two tiers
/// genuinely differ about: a step on the closure tier is always a
/// closure, and on the native tier it is a closure or a run of native
/// code. That difference lives in the run loops, which stay per tier.
macro_rules! shared_core_methods {
    () => {
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

        fn dirty_input(&mut self, slot: usize) {
            if let Some(deps) = self.dirty.get(slot) {
                for &i in deps {
                    self.clean[i] = false;
                }
            }
        }

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
            self.value_of(name)
        }

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
