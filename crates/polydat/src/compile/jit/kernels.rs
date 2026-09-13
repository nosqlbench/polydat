// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! JIT kernel types: structs and impls for all four kernel variants.
//!
//! `JitCore` holds the shared buffer, slot map, and module handle.
//! The four kernel structs (`JitKernelRaw`, `JitKernelPush`,
//! `JitKernelPull`, `JitKernelPushPull`) wrap a `JitCore` and a
//! compiled function pointer, providing `eval` and accessor methods.

use std::collections::HashMap;

use cranelift_jit::JITModule;

use crate::ast::PolydatNode;
use crate::kernel::ProvMask;

/// Finalized native code, shared by every kernel created from one
/// program. The module's memory is never written after finalization,
/// so sharing it across threads is sound; the wrapper exists so a
/// kernel clone is a new state over the same code. The slot kits the
/// code calls by address live beside it, for as long as it does.
#[derive(Clone)]
pub struct JitCode(#[allow(dead_code)] std::sync::Arc<FinalizedModule>);

/// A JIT module after finalization, which nothing writes again, and
/// the kits its code calls.
struct FinalizedModule(
    #[allow(dead_code)] JITModule,
    #[allow(dead_code)] Vec<super::codegen::SlotKitRef>,
);

/// The scratch a native kernel's state owns: one entry per entry the
/// steps' kits declare, and the `(first slot, entry)` pairs of the
/// scratch-backed `Ref2` outputs among them (axiom S9(a)).
#[derive(Clone, Default)]
pub(crate) struct ScratchPlan {
    pub(crate) elems: Vec<crate::ast::ScratchElem>,
    pub(crate) refs: Vec<(usize, usize)>,
}

// SAFETY: the module is finalized before it is wrapped and never
// touched again; only its code runs, from any thread.
unsafe impl Send for FinalizedModule {}
unsafe impl Sync for FinalizedModule {}

impl JitCode {
    pub(crate) fn new(module: JITModule, kits: Vec<super::codegen::SlotKitRef>) -> Self {
        JitCode(std::sync::Arc::new(FinalizedModule(module, kits)))
    }
}

/// A raw native kernel taken apart: its entry point and its code.
pub type JitParts = (super::codegen::NativeFn, JitCode);

/// Shared fields for all JIT kernel variants. A clone is a new state
/// of the same program: the code and the nodes are shared, everything
/// else is the clone's own (engine_parity.md, step 4), and every extern
/// pair in its buffer points into its own storage (axiom S3), never
/// into the state it was cloned from.
pub(super) struct JitCore {
    pub(super) buffer: Vec<u64>,
    pub(super) coord_count: usize,
    pub(super) output_map: HashMap<String, usize>,
    /// Slots the raw readers refuse: `Ref2` pairs (axiom S2). Set by
    /// the assembler once the layout is known; empty means no such
    /// slot.
    pub(super) guard_slots: Vec<bool>,
    /// Port type of each named output, for `get_value`'s decode.
    pub(super) output_types: HashMap<String, crate::ast::PortType>,
    /// The extern inputs, written through at every set.
    pub(super) externs: crate::compile::externs::Externs,
    /// The traversals the program declares (SRD 113), opened through the
    /// `Kernel` trait.
    pub(super) traversals: std::sync::Arc<[crate::dsl::traversal::Traversal]>,
    pub(super) _module: JitCode,
    pub(super) _nodes: std::sync::Arc<Vec<Box<dyn PolydatNode>>>,
    /// The coordinates set through the `Kernel` trait, pending
    /// evaluation.
    pub(super) drive: crate::compile::Drive,
    /// Where each step came from, for the failure path (A7).
    pub(super) sites: std::sync::Arc<crate::compile::Attribution>,
    /// The slot past the layout where native code names the step it
    /// is in before calling a helper; `u64::MAX` before any.
    pub(super) tracker: usize,
    /// The scratch entries the steps' kits write into, owned by this
    /// state (axiom S3); native code receives the base pointer.
    pub(super) scratch: Vec<crate::ast::ScratchBuf>,
    /// Axiom S9(a): (first slot of a Ref pair → scratch index) for
    /// every scratch-backed Ref output.
    pub(super) ref_scratch: Vec<(usize, usize)>,
    /// The steps that are never current (runtime_model.md, R1.v): a
    /// nondeterministic node or one downstream of it. The kernels
    /// with a clean flag per step clear theirs at every write, and
    /// the kernels with a cone guard run whenever one exists and a
    /// write happened, since one native function is the program.
    pub(super) volatile_steps: Vec<usize>,
}

impl Clone for JitCore {
    fn clone(&self) -> Self {
        let mut core = JitCore {
            buffer: self.buffer.clone(),
            coord_count: self.coord_count,
            output_map: self.output_map.clone(),
            guard_slots: self.guard_slots.clone(),
            output_types: self.output_types.clone(),
            externs: self.externs.clone(),
            traversals: self.traversals.clone(),
            _module: self._module.clone(),
            _nodes: self._nodes.clone(),
            drive: self.drive.clone(),
            sites: self.sites.clone(),
            tracker: self.tracker,
            scratch: self.scratch.clone(),
            ref_scratch: self.ref_scratch.clone(),
            volatile_steps: self.volatile_steps.clone(),
        };
        // Every pair points into this state's own storage (axiom S3):
        // a step's scratch entry, the value an extern stores.
        for &(slot, idx) in &core.ref_scratch {
            let (p, l) = core.scratch[idx].ptr_len();
            core.buffer[slot] = p;
            core.buffer[slot + 1] = l;
        }
        core.externs.seed(&mut core.buffer, None);
        core
    }
}

impl JitCore {
    /// The value at `slot` decoded as `ty`, a pair copied out.
    pub(super) fn slot_value(&self, slot: usize, ty: crate::ast::PortType) -> crate::ast::Value {
        crate::compile::marshal::decode_output(&self.buffer, slot, ty)
    }

    /// One native function is the program.
    pub(super) fn plan(&self) -> crate::EnginePlan {
        crate::EnginePlan {
            native_segments: 1,
            ..Default::default()
        }
    }

    /// The next evaluation runs the program: one native function.
    pub(super) fn invalidate_all(&mut self) {
        self.drive.stale = true;
    }

    pub(super) fn new(
        total_slots: usize,
        coord_count: usize,
        output_map: HashMap<String, usize>,
        code: JitCode,
        nodes: Vec<Box<dyn PolydatNode>>,
        scratch: ScratchPlan,
        volatile_steps: Vec<usize>,
    ) -> Self {
        Self {
            buffer: vec![0u64; total_slots + 1],
            coord_count,
            output_map,
            guard_slots: Vec::new(),
            output_types: HashMap::new(),
            externs: crate::compile::externs::Externs::default(),
            traversals: Vec::new().into(),
            _module: code,
            _nodes: std::sync::Arc::new(nodes),
            drive: crate::compile::Drive::default(),
            sites: std::sync::Arc::default(),
            tracker: total_slots,
            scratch: scratch
                .elems
                .iter()
                .map(|e| crate::ast::ScratchBuf::new(*e))
                .collect(),
            ref_scratch: scratch.refs,
            volatile_steps,
        }
    }

    /// Whether a write must run the program regardless of the cone
    /// guard: a never-current step exists (R1.v).
    #[inline]
    fn has_volatile(&self) -> bool {
        !self.volatile_steps.is_empty()
    }

    /// Axiom S9(a): every scratch-backed pair in the buffer names its
    /// own entry, checked after a run in debug builds.
    #[cfg(debug_assertions)]
    fn validate_refs(&self) {
        for &(slot, idx) in &self.ref_scratch {
            let (p, l) = self.scratch[idx].ptr_len();
            assert!(
                self.buffer[slot] == p && self.buffer[slot + 1] == l,
                "S9 ref-validator: slot pair ({slot}, {}) = ({:#x}, {}) does not match \
                 scratch[{idx}] = ({p:#x}, {l})",
                slot + 1,
                self.buffer[slot],
                self.buffer[slot + 1],
            );
        }
    }

    /// Install the extern inputs, written through into the buffer now.
    pub(super) fn set_externs(&mut self, externs: crate::compile::externs::Externs) {
        externs.seed(&mut self.buffer, None);
        self.externs = externs;
    }

    /// Set an extern by name; returns its slot for dirty marking.
    fn set_extern(&mut self, name: &str, value: crate::ast::Value) -> Result<usize, String> {
        Ok(self.externs.set(name, value, &mut self.buffer)?.0)
    }

    /// [`Self::set_extern`] by input index.
    fn set_extern_at(&mut self, index: usize, value: crate::ast::Value) -> Result<usize, String> {
        Ok(self.externs.set_at(index, value, &mut self.buffer)?.0)
    }

    /// Bind a `shared` binding to `cell` (engine parity, step 9). Native
    /// code evaluates the whole program, so the next run reads it.
    fn attach_cell(&mut self, name: &str, cell: crate::kernel::SharedCell) -> Result<(), String> {
        self.externs.attach_cell(name, cell)?;
        self.drive.stale = true;
        Ok(())
    }

    /// Run one native evaluation: take what cells other holders
    /// published, refuse an unset extern (native code cannot carry a
    /// `None`; engine_parity.md, A12), run inside the longjmp catch.
    #[inline]
    pub(super) fn run(&mut self, native: impl FnOnce()) {
        if self.externs.cells_dirty() {
            self.externs.refresh_cells(&mut self.buffer);
        }
        if let Some((name, ty)) = self.externs.first_unset() {
            panic!(
                "extern '{name}' ({ty}) has no value: it has no default, so set it with \
                 set_input before the first run (native code cannot carry `None`; \
                 docs/design/engine_parity.md, A12)"
            );
        }
        // Native code names the step it is in before each helper call;
        // a failure before any names none. The capture guard is armed
        // for the run, so the helper's panic is recorded quietly and
        // re-raised enriched, as the interpreter re-raises a node's (A7).
        self.buffer[self.tracker] = u64::MAX;
        let capture = crate::kernel::engines::EvalPanicCaptureGuard::arm();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            super::codegen::invoke_with_catch(native)
        }));
        drop(capture);
        if let Err(payload) = outcome {
            let step = self.buffer[self.tracker] as usize;
            let sites = std::sync::Arc::clone(&self.sites);
            sites.reraise(payload, step, &self.buffer, None);
        }
        #[cfg(debug_assertions)]
        self.validate_refs();
    }
}

macro_rules! jit_accessors {
    () => {
        /// Returns the number of coordinate inputs this kernel accepts.
        pub fn coord_count(&self) -> usize {
            self.core.coord_count
        }

        /// Returns the buffer slot index for the named output, if present.
        pub fn resolve_output(&self, name: &str) -> Option<usize> {
            self.core.output_map.get(name).copied()
        }

        /// Returns the raw u64 value stored in the named output slot.
        #[inline]
        pub fn get(&self, name: &str) -> u64 {
            self.get_slot(self.core.output_map[name])
        }

        /// Returns the raw u64 value stored at the given buffer slot
        /// index. Refuses a reference or handle slot (axioms S2, H1):
        /// read those through [`Self::get_value`].
        #[inline]
        pub fn get_slot(&self, slot: usize) -> u64 {
            if self.core.guard_slots.get(slot).copied().unwrap_or(false) {
                panic!(
                    "slot {slot} is Ref2-colored; a raw u64 read would leak an interior \
                     address. Use get_value to decode it."
                );
            }
            self.core.buffer[slot]
        }

        /// The named output as a typed `Value`, decoded by its port type:
        /// a reference pair is copied out, so the caller never holds a
        /// reference into the buffer.
        pub fn get_value(&self, name: &str) -> crate::ast::Value {
            let slot = self.core.output_map[name];
            let ty = self
                .core
                .output_types
                .get(name)
                .copied()
                .unwrap_or(crate::ast::PortType::U64);
            crate::compile::marshal::decode_output(&self.core.buffer, slot, ty)
        }

        /// Record the slots raw readers must refuse and each output's
        /// port type. Called by the assembler after construction.
        pub(crate) fn set_slot_info(
            &mut self,
            guard_slots: Vec<bool>,
            output_types: HashMap<String, crate::ast::PortType>,
        ) {
            self.core.guard_slots = guard_slots;
            self.core.output_types = output_types;
        }

        /// Where each step came from, for the failure path (A7).
        pub(crate) fn set_attribution(
            &mut self,
            sites: std::sync::Arc<crate::compile::Attribution>,
        ) {
            self.core.sites = sites;
        }

        /// Set an extern by name, as `PolydatState::set_input` does on
        /// the interpreter. The value must be of the declared port
        /// type. A carrier takes effect at once; a string, JSON, or
        /// extension value is written at the start of the next run, and
        /// every step downstream of the extern reruns.
        pub fn set_input(&mut self, name: &str, value: crate::ast::Value) -> Result<(), String> {
            let slot = self.core.set_extern(name, value)?;
            self.mark_input_changed(slot);
            Ok(())
        }

        /// [`Self::set_input`] by input index.
        pub fn set_input_at(
            &mut self,
            index: usize,
            value: crate::ast::Value,
        ) -> Result<(), String> {
            let slot = self.core.set_extern_at(index, value)?;
            self.mark_input_changed(slot);
            Ok(())
        }

        /// The kernel's externs by name and declared type.
        pub fn externs(&self) -> Vec<(&str, crate::ast::PortType)> {
            self.core.externs.names()
        }

        /// Every step downstream of a coordinate reruns at the next
        /// evaluation: the state a kernel created from a shared program
        /// starts in.
        fn mark_all_dirty(&mut self) {
            for i in 0..self.core.coord_count {
                self.mark_input_changed(i);
            }
        }

        /// The named output through the `Kernel` trait. Native code
        /// evaluates the whole program, so a pull after new inputs is an
        /// evaluation.
        fn pull_value(&mut self, name: &str) -> crate::ast::Value {
            // A cell another holder published to is a changed input.
            if self.core.drive.stale || self.core.externs.cells_dirty() {
                self.eval_pending();
                self.core.drive.stale = false;
            }
            self.get_value(name)
        }

        /// [`Self::pull_value`] by output index: one native function is
        /// the program, so the index names the output and nothing more.
        fn pull_value_at(&mut self, index: usize) -> crate::ast::Value {
            let name = self
                .core
                .externs
                .output_names()
                .get(index)
                .cloned()
                .unwrap_or_else(|| panic!("no output at index {index}"));
            self.pull_value(&name)
        }

        /// `eval` through the `Kernel` trait: the pending coordinates.
        fn eval_pending(&mut self) {
            let coords = std::mem::take(&mut self.core.drive.coords);
            self.eval(&coords);
            self.core.drive.coords = coords;
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
        ) -> Result<(), String> {
            for (slot, value) in self.core.externs.cursor_writes(name, partition)? {
                self.set_input(&slot, value)?;
            }
            Ok(())
        }
    };
}

// ── JitKernelRaw ───────────────────────────────────────────

/// Raw JIT kernel: no provenance, all nodes evaluate unconditionally.
#[derive(Clone)]
#[doc(hidden)]
pub struct JitKernelRaw {
    pub(super) core: JitCore,
    pub(super) code_fn: super::codegen::NativeFn,
}

impl JitKernelRaw {
    /// Evaluate the kernel with the given coordinate values.
    ///
    /// Predicate violations (`is_positive`, `in_range`,
    /// `is_one_of`) from JIT-lowered code surface as normal
    /// Rust panics carrying the violation message. The
    /// longjmp wrapper in `super::codegen::invoke_with_catch`
    /// handles the transition back to Rust land.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.core.buffer[..self.core.coord_count.min(coords.len())]
            .copy_from_slice(&coords[..self.core.coord_count.min(coords.len())]);
        let code_fn = self.code_fn;
        let buf_ptr_const = self.core.buffer.as_ptr();
        let buf_ptr_mut = self.core.buffer.as_mut_ptr();
        let sc = self.core.scratch.as_mut_ptr();
        self.core.run(move || unsafe {
            (code_fn)(buf_ptr_const, buf_ptr_mut, sc);
        });
    }

    /// Evaluate and return the value at the given buffer slot index.
    #[inline]
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        self.eval(coords);
        self.core.buffer[slot]
    }

    /// Decompose into raw parts for hybrid kernel integration: the
    /// entry point and its module.
    pub fn into_parts(self) -> JitParts {
        (self.code_fn, self.core._module)
    }

    /// Every run evaluates everything; a changed input needs no mark.
    fn mark_input_changed(&mut self, _slot: usize) {}

    jit_accessors!();
}

// ── JitKernelPush ──────────────────────────────────────────

/// Push-only JIT kernel: per-node dirty tracking, no cone guard.
#[derive(Clone)]
#[doc(hidden)]
pub struct JitKernelPush {
    pub(super) core: JitCore,
    pub(super) code_fn_prov: super::codegen::NativeProvFn,
    pub(super) node_clean: Vec<u8>,
    pub(super) input_dependents: Vec<Vec<usize>>,
}

impl JitKernelPush {
    #[inline]
    fn set_inputs(&mut self, coords: &[u64]) {
        for (i, &c) in coords.iter().enumerate().take(self.core.coord_count) {
            if self.core.buffer[i] != c {
                self.core.buffer[i] = c;
                self.mark_input_changed(i);
            }
        }
        // A write makes every never-current step run again (R1.v).
        for &step_idx in &self.core.volatile_steps {
            self.node_clean[step_idx] = 0;
        }
    }

    /// Every step downstream of the slot reruns, and every
    /// never-current step with it (R1.v).
    fn mark_input_changed(&mut self, slot: usize) {
        if slot < self.input_dependents.len() {
            for &step_idx in &self.input_dependents[slot] {
                self.node_clean[step_idx] = 0;
            }
        }
        for &step_idx in &self.core.volatile_steps {
            self.node_clean[step_idx] = 0;
        }
    }

    /// Evaluate the kernel with the given coordinate values.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.set_inputs(coords);
        let code_fn = self.code_fn_prov;
        let buf_const = self.core.buffer.as_ptr();
        let buf_mut = self.core.buffer.as_mut_ptr();
        let sc = self.core.scratch.as_mut_ptr();
        let clean_mut = self.node_clean.as_mut_ptr();
        self.core.run(move || unsafe {
            (code_fn)(buf_const, buf_mut, sc, clean_mut);
        });
    }

    /// Evaluate and return the value at the given buffer slot index.
    #[inline]
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        self.eval(coords);
        self.core.buffer[slot]
    }

    jit_accessors!();
}

// ── JitKernelPull ──────────────────────────────────────────

/// Pull-only JIT kernel: cone guard, but all nodes run when cone is dirty.
/// Uses the raw (non-provenance) JIT function — no per-node clean checks.
#[derive(Clone)]
#[doc(hidden)]
pub struct JitKernelPull {
    pub(super) core: JitCore,
    pub(super) code_fn: super::codegen::NativeFn,
    pub(super) slot_provenance: Vec<ProvMask>,
    pub(super) changed_mask: ProvMask,
    /// Set by `set_input`: an extern changed, so the next evaluation
    /// runs whatever the cone guard says.
    pub(super) force_run: bool,
}

impl JitKernelPull {
    #[inline]
    fn set_inputs(&mut self, coords: &[u64]) {
        self.changed_mask.clear();
        for (i, &c) in coords.iter().enumerate().take(self.core.coord_count) {
            if self.core.buffer[i] != c {
                self.core.buffer[i] = c;
                self.changed_mask.set(i);
            }
        }
        // A never-current step runs again after every write (R1.v),
        // and one native function is the program.
        if self.core.has_volatile() {
            self.force_run = true;
        }
    }

    /// The next evaluation runs regardless of the cone guard, since
    /// `set_inputs` rebuilds the changed set from the coordinates alone.
    fn mark_input_changed(&mut self, _slot: usize) {
        self.force_run = true;
    }

    /// Evaluate the kernel with the given coordinate values.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.set_inputs(coords);
        self.force_run = false;
        let code_fn = self.code_fn;
        let buf_const = self.core.buffer.as_ptr();
        let buf_mut = self.core.buffer.as_mut_ptr();
        let sc = self.core.scratch.as_mut_ptr();
        self.core.run(move || unsafe {
            (code_fn)(buf_const, buf_mut, sc);
        });
    }

    /// Evaluate and return the value at the given buffer slot index,
    /// skipping evaluation if the slot's provenance cone is unaffected.
    #[inline]
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        self.set_inputs(coords);
        if !self.force_run
            && slot < self.slot_provenance.len()
            && !self.slot_provenance[slot].intersects(&self.changed_mask)
        {
            return self.core.buffer[slot];
        }
        self.force_run = false;
        let code_fn = self.code_fn;
        let buf_const = self.core.buffer.as_ptr();
        let buf_mut = self.core.buffer.as_mut_ptr();
        let sc = self.core.scratch.as_mut_ptr();
        self.core.run(move || unsafe {
            (code_fn)(buf_const, buf_mut, sc);
        });
        self.core.buffer[slot]
    }

    jit_accessors!();
}

// ── JitKernelPushPull ──────────────────────────────────────

/// Full optimization: push-side dirty tracking + pull-side cone guard.
#[derive(Clone)]
#[doc(hidden)]
pub struct JitKernelPushPull {
    pub(super) core: JitCore,
    pub(super) code_fn_prov: super::codegen::NativeProvFn,
    pub(super) node_clean: Vec<u8>,
    pub(super) input_dependents: Vec<Vec<usize>>,
    pub(super) slot_provenance: Vec<ProvMask>,
    pub(super) changed_mask: ProvMask,
    /// Set by `set_input`: an extern changed, so the next evaluation
    /// runs whatever the cone guard says.
    pub(super) force_run: bool,
}

impl JitKernelPushPull {
    #[inline]
    fn set_inputs(&mut self, coords: &[u64]) {
        self.changed_mask.clear();
        for (i, &c) in coords.iter().enumerate().take(self.core.coord_count) {
            if self.core.buffer[i] != c {
                self.core.buffer[i] = c;
                self.changed_mask.set(i);
                if i < self.input_dependents.len() {
                    for &step_idx in &self.input_dependents[i] {
                        self.node_clean[step_idx] = 0;
                    }
                }
            }
        }
        // A write makes every never-current step run again (R1.v),
        // whatever the cone guard would say of the pulled output.
        if self.core.has_volatile() {
            for &step_idx in &self.core.volatile_steps {
                self.node_clean[step_idx] = 0;
            }
            self.force_run = true;
        }
    }

    /// Every step downstream of the slot reruns, and the next
    /// evaluation runs whatever the cone guard says.
    fn mark_input_changed(&mut self, slot: usize) {
        if slot < self.input_dependents.len() {
            for &step_idx in &self.input_dependents[slot] {
                self.node_clean[step_idx] = 0;
            }
        }
        for &step_idx in &self.core.volatile_steps {
            self.node_clean[step_idx] = 0;
        }
        self.force_run = true;
    }

    /// Evaluate the kernel with the given coordinate values.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.set_inputs(coords);
        self.force_run = false;
        let code_fn = self.code_fn_prov;
        let buf_const = self.core.buffer.as_ptr();
        let buf_mut = self.core.buffer.as_mut_ptr();
        let sc = self.core.scratch.as_mut_ptr();
        let clean_mut = self.node_clean.as_mut_ptr();
        self.core.run(move || unsafe {
            (code_fn)(buf_const, buf_mut, sc, clean_mut);
        });
    }

    /// Evaluate and return the value at the given buffer slot index,
    /// applying both push and pull optimizations.
    #[inline]
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        self.set_inputs(coords);
        if !self.force_run
            && slot < self.slot_provenance.len()
            && !self.slot_provenance[slot].intersects(&self.changed_mask)
        {
            return self.core.buffer[slot];
        }
        self.force_run = false;
        let code_fn = self.code_fn_prov;
        let buf_const = self.core.buffer.as_ptr();
        let buf_mut = self.core.buffer.as_mut_ptr();
        let sc = self.core.scratch.as_mut_ptr();
        let clean_mut = self.node_clean.as_mut_ptr();
        self.core.run(move || unsafe {
            (code_fn)(buf_const, buf_mut, sc, clean_mut);
        });
        self.core.buffer[slot]
    }

    jit_accessors!();
}

// ── The engine-independent surface (engine_parity.md, step 4) ──────

use crate::compile::select::{Engine, Provenance};

crate::compile::impl_kernel_trait!(JitKernelRaw, Engine::Native(Provenance::Raw));
crate::compile::impl_kernel_trait!(JitKernelPush, Engine::Native(Provenance::Push));
crate::compile::impl_kernel_trait!(JitKernelPull, Engine::Native(Provenance::Pull));
crate::compile::impl_kernel_trait!(JitKernelPushPull, Engine::Native(Provenance::PushPull));
