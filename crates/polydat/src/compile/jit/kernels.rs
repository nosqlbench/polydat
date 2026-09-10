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
/// kernel clone is a new state over the same code.
#[derive(Clone)]
pub struct JitCode(#[allow(dead_code)] std::sync::Arc<FinalizedModule>);

/// A JIT module after finalization, which nothing writes again.
struct FinalizedModule(#[allow(dead_code)] JITModule);

// SAFETY: the module is finalized before it is wrapped and never
// touched again; only its code runs, from any thread.
unsafe impl Send for FinalizedModule {}
unsafe impl Sync for FinalizedModule {}

impl JitCode {
    pub(crate) fn new(module: JITModule) -> Self {
        JitCode(std::sync::Arc::new(FinalizedModule(module)))
    }
}

/// A raw native kernel taken apart: its entry point, its code, and the
/// table-kind `(slot, entry)` pairs the code writes.
pub type JitParts = (
    unsafe fn(*const u64, *mut u64),
    JitCode,
    Vec<(usize, usize)>,
);

/// Shared fields for all JIT kernel variants. A clone is a new state
/// of the same program: the code and the nodes are shared, everything
/// else is the clone's own (engine_parity.md, step 4).
#[derive(Clone)]
pub(super) struct JitCore {
    pub(super) buffer: Vec<u64>,
    pub(super) coord_count: usize,
    pub(super) output_map: HashMap<String, usize>,
    /// Slots the raw readers refuse: `Ref2` pairs (axiom S2) and
    /// `Hdl1` handles (SRD 115, axiom H1). Set by the assembler once the
    /// layout is known; empty means no such slot.
    pub(super) guard_slots: Vec<bool>,
    /// Port type of each named output, for `get_value`'s decode.
    pub(super) output_types: HashMap<String, crate::ast::PortType>,
    /// The kernel's value table (SRD 115 §3): one entry per table-kind
    /// slot the native code writes, owned for the kernel's lifetime.
    pub(super) table: crate::kernel::ValueTable,
    /// `(slot, entry)` for every table-kind slot, from codegen; the
    /// validator checks each slot's handle names its own entry.
    pub(super) table_entries: Vec<(usize, usize)>,
    /// True when this kernel is driven directly by a host and so begins
    /// a root cycle at every eval (SRD 115 §4). False when a state that
    /// owns the cycle wraps it.
    pub(super) owns_cycle: bool,
    /// The extern inputs, materialized at the start of every run.
    pub(super) externs: crate::compile::externs::Externs,
    pub(super) _module: JitCode,
    pub(super) _nodes: std::sync::Arc<Vec<Box<dyn PolydatNode>>>,
    /// The coordinates set through the `Kernel` trait, pending
    /// evaluation.
    pub(super) drive: crate::compile::Drive,
}

impl JitCore {
    pub(super) fn new(
        total_slots: usize,
        coord_count: usize,
        output_map: HashMap<String, usize>,
        table_entries: Vec<(usize, usize)>,
        module: JITModule,
        nodes: Vec<Box<dyn PolydatNode>>,
    ) -> Self {
        let table_len = table_entries.iter().map(|&(_, e)| e + 1).max().unwrap_or(0);
        Self {
            buffer: vec![0u64; total_slots],
            coord_count,
            output_map,
            guard_slots: Vec::new(),
            output_types: HashMap::new(),
            table: crate::kernel::ValueTable::new(table_len),
            table_entries,
            owns_cycle: true,
            externs: crate::compile::externs::Externs::default(),
            _module: JitCode::new(module),
            _nodes: std::sync::Arc::new(nodes),
            drive: crate::compile::Drive::default(),
        }
    }

    /// Install the extern inputs: table-kind externs take entries after
    /// the ones the code owns, carriers are seeded now, and every run
    /// materializes the handle kinds.
    pub(super) fn set_externs(&mut self, mut externs: crate::compile::externs::Externs) {
        externs.renumber_entries(
            self.table_entries
                .iter()
                .map(|&(_, e)| e + 1)
                .max()
                .unwrap_or(0),
        );
        self.table_entries.extend(externs.table_entries());
        let table_len = self
            .table_entries
            .iter()
            .map(|&(_, e)| e + 1)
            .max()
            .unwrap_or(0);
        self.table = crate::kernel::ValueTable::new(table_len);
        externs.seed(&mut self.buffer);
        self.externs = externs;
    }

    /// Set an extern by name; returns its slot for dirty marking.
    fn set_extern(&mut self, name: &str, value: crate::ast::Value) -> Result<usize, String> {
        self.externs.set(name, value, &mut self.buffer)
    }

    /// Run one native evaluation: begin the cycle it belongs to, install
    /// the kernel's value table for the helpers, run inside the longjmp
    /// catch, then check the table invariants.
    #[inline]
    pub(super) fn run(&mut self, native: impl FnOnce()) {
        let generation = if self.owns_cycle {
            crate::kernel::begin_root_cycle()
        } else {
            crate::kernel::cycle_generation()
        };
        self.table.set_generation(generation);
        // Extern handles belong to this run (H3, H4).
        self.externs
            .materialize(&mut self.buffer, &mut self.table, None);
        crate::kernel::with_value_table(&mut self.table, || {
            super::codegen::invoke_with_catch(native)
        });
        self.validate_table();
    }

    /// SRD 115 axiom H4 validator (the S9 analogue for handles): after
    /// a native run, every table-kind slot holds a table handle of this
    /// generation naming exactly the entry the layout assigned it, and
    /// that entry was written. Debug builds only.
    #[inline]
    fn validate_table(&self) {
        if cfg!(debug_assertions) {
            for &(slot, entry) in &self.table_entries {
                let handle = self.buffer[slot];
                assert_eq!(
                    handle & crate::kernel::TAG_MASK,
                    crate::kernel::TAG_RES,
                    "H4: slot {slot} should hold a table handle, holds {handle:#x}"
                );
                let (_, generation, named) = crate::kernel::decode_table_handle(handle);
                assert_eq!(
                    named, entry,
                    "H4: slot {slot} names entry {named}; the layout assigned it entry {entry}"
                );
                assert_eq!(
                    generation,
                    self.table.generation() & 0xFF_FFFF,
                    "H3: slot {slot} holds a handle from another cycle generation"
                );
                assert!(
                    self.table.is_written(entry),
                    "H4: entry {entry} was not written by the run that produced slot {slot}"
                );
            }
        }
    }
}

/// Compute slot provenance from input_dependents.
///
/// `input_dependents` is indexed by coordinate SLOT (callers
/// expand per-input lists across multi-slot inputs per §8.4
/// layer 1); `step_output_slots` carries every step's flattened
/// output slot list so multi-slot outputs share their step's
/// provenance word.
pub(super) fn compute_jit_slot_provenance(
    coord_count: usize,
    buffer_len: usize,
    step_output_slots: &[Vec<usize>],
    input_dependents: &[Vec<usize>],
) -> Vec<ProvMask> {
    let step_count = step_output_slots.len();
    let mut step_prov: Vec<ProvMask> = (0..step_count).map(|_| ProvMask::empty()).collect();
    for (input_slot, deps) in input_dependents.iter().enumerate() {
        for &step_idx in deps {
            if step_idx < step_count {
                step_prov[step_idx].set(input_slot);
            }
        }
    }
    let mut slot_prov: Vec<ProvMask> = (0..buffer_len).map(|_| ProvMask::empty()).collect();
    for (i, slot) in slot_prov.iter_mut().enumerate().take(coord_count) {
        slot.set(i);
    }
    for (i, outs) in step_output_slots.iter().enumerate() {
        for &slot in outs {
            if slot < slot_prov.len() {
                slot_prov[slot] = step_prov[i].clone();
            }
        }
    }
    slot_prov
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

        /// The named outputs, as every engine reports them.
        pub fn output_names(&self) -> Vec<&str> {
            self.core.output_map.keys().map(|s| s.as_str()).collect()
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
                    "slot {slot} is Ref2- or Hdl1-colored; a raw u64 read would leak an \
                     interior address or a handle. Use get_value to decode it."
                );
            }
            self.core.buffer[slot]
        }

        /// The named output as a typed `Value`, decoded by its port type:
        /// a handle slot is copied out of the arena or the value table
        /// (SRD 115 §5), so the caller never holds a handle.
        pub fn get_value(&self, name: &str) -> crate::ast::Value {
            let slot = self.core.output_map[name];
            let ty = self
                .core
                .output_types
                .get(name)
                .copied()
                .unwrap_or(crate::ast::PortType::U64);
            crate::compile::marshal::decode_output(&self.core.buffer, slot, ty, &self.core.table)
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

        /// Whether each eval begins a root cycle (SRD 115 §4). A state
        /// that owns the cycle and wraps this kernel sets this false.
        #[allow(dead_code)] // no state wraps the push-only variant
        pub(crate) fn set_owns_cycle(&mut self, owns: bool) {
            self.core.owns_cycle = owns;
        }

        /// Entries in the kernel's value table.
        pub fn table_len(&self) -> usize {
            self.core.table.len()
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
            if self.core.drive.stale {
                self.eval_pending();
                self.core.drive.stale = false;
            }
            self.get_value(name)
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
pub struct JitKernelRaw {
    pub(super) core: JitCore,
    pub(super) code_fn: unsafe fn(*const u64, *mut u64),
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
        self.core.run(move || unsafe {
            (code_fn)(buf_ptr_const, buf_ptr_mut);
        });
    }

    /// Evaluate and return the value at the given buffer slot index.
    #[inline]
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        self.eval(coords);
        self.core.buffer[slot]
    }

    /// Decompose into raw parts for hybrid kernel integration: the
    /// entry point, its module, and the table-kind `(slot, entry)`
    /// pairs the code writes.
    pub fn into_parts(self) -> JitParts {
        (self.code_fn, self.core._module, self.core.table_entries)
    }

    /// Every run evaluates everything; a changed input needs no mark.
    fn mark_input_changed(&mut self, _slot: usize) {}

    jit_accessors!();
}

// ── JitKernelPush ──────────────────────────────────────────

/// Push-only JIT kernel: per-node dirty tracking, no cone guard.
#[derive(Clone)]
pub struct JitKernelPush {
    pub(super) core: JitCore,
    pub(super) code_fn_prov: unsafe fn(*const u64, *mut u64, *mut u8),
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
    }

    /// Every step downstream of the slot reruns.
    fn mark_input_changed(&mut self, slot: usize) {
        if slot < self.input_dependents.len() {
            for &step_idx in &self.input_dependents[slot] {
                self.node_clean[step_idx] = 0;
            }
        }
    }

    /// Evaluate the kernel with the given coordinate values.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.set_inputs(coords);
        let code_fn = self.code_fn_prov;
        let buf_const = self.core.buffer.as_ptr();
        let buf_mut = self.core.buffer.as_mut_ptr();
        let clean_mut = self.node_clean.as_mut_ptr();
        self.core.run(move || unsafe {
            (code_fn)(buf_const, buf_mut, clean_mut);
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
pub struct JitKernelPull {
    pub(super) core: JitCore,
    pub(super) code_fn: unsafe fn(*const u64, *mut u64),
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
        self.core.run(move || unsafe {
            (code_fn)(buf_const, buf_mut);
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
        self.core.run(move || unsafe {
            (code_fn)(buf_const, buf_mut);
        });
        self.core.buffer[slot]
    }

    jit_accessors!();
}

// ── JitKernelPushPull ──────────────────────────────────────

/// Full optimization: push-side dirty tracking + pull-side cone guard.
#[derive(Clone)]
pub struct JitKernelPushPull {
    pub(super) core: JitCore,
    pub(super) code_fn_prov: unsafe fn(*const u64, *mut u64, *mut u8),
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
    }

    /// Every step downstream of the slot reruns, and the next
    /// evaluation runs whatever the cone guard says.
    fn mark_input_changed(&mut self, slot: usize) {
        if slot < self.input_dependents.len() {
            for &step_idx in &self.input_dependents[slot] {
                self.node_clean[step_idx] = 0;
            }
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
        let clean_mut = self.node_clean.as_mut_ptr();
        self.core.run(move || unsafe {
            (code_fn)(buf_const, buf_mut, clean_mut);
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
        let clean_mut = self.node_clean.as_mut_ptr();
        self.core.run(move || unsafe {
            (code_fn)(buf_const, buf_mut, clean_mut);
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
