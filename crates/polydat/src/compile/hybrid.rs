// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Hybrid kernel: per-node optimal compilation level.
//!
//! Splits the DAG into segments based on each node's compilation
//! capability. JIT-able nodes are batched into native code segments.
//! Non-JIT-able nodes run as Phase 2 closures. All segments share
//! the same flat u64 buffer.
//!
//! This is the "best of all worlds" kernel — no node pays more
//! overhead than it needs to.
//!
//! Three monomorphic kernel types, each with no runtime branching:
//!
//! | Type | Push (per-step skip) | Pull (cone guard) |
//! |------|---------------------|-------------------|
//! | `HybridKernelRaw` | — | — |
//! | `HybridKernelPull` | — | yes |
//! | `HybridKernelPushPull` | yes | yes |

use std::collections::HashMap;

use crate::ast::{CompiledU64Op, PolydatNode};
use crate::kernel::WireSource;

#[cfg(feature = "jit")]
use crate::compile::jit::{self, JitOp};

/// A step in the hybrid kernel: either JIT native code or a Phase 2 closure.
enum HybridStep {
    /// A batch of nodes compiled to native code via Cranelift.
    /// The function reads/writes directly to the shared buffer.
    #[cfg(feature = "jit")]
    Jit(JitSegment),
    /// A single node executed via its Phase 2 closure.
    Closure(ClosureStep),
}

#[cfg(feature = "jit")]
struct JitSegment {
    code_fn: unsafe fn(*const u64, *mut u64),
    /// The finalized native code, shared by every kernel created from
    /// one program.
    _module: crate::compile::jit::JitCode,
    /// The slots the segment reads and writes, for cones and for the
    /// `None` check native code cannot make itself.
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
}

impl HybridStep {
    fn input_slots(&self) -> &[usize] {
        match self {
            #[cfg(feature = "jit")]
            HybridStep::Jit(seg) => &seg.input_slots,
            HybridStep::Closure(cs) => &cs.input_slots,
        }
    }
    fn output_slots(&self) -> &[usize] {
        match self {
            #[cfg(feature = "jit")]
            HybridStep::Jit(seg) => &seg.output_slots,
            HybridStep::Closure(cs) => &cs.output_slots,
        }
    }
    /// SRD-74 Rule 2: the step runs on `None` inputs. Native code never
    /// does; a node downstream of an unset extern is a closure.
    fn accepts_none(&self) -> bool {
        match self {
            #[cfg(feature = "jit")]
            HybridStep::Jit(_) => false,
            HybridStep::Closure(cs) => cs.accepts_none,
        }
    }
}

/// A closure step's op: pure-scalar u64 closure, or a slot op
/// with kernel-owned scratch for typed-slice ports (§8.4 L3).
enum ClosureOp {
    U64(CompiledU64Op),
    Slot(crate::ast::CompiledSlotOp),
}

struct ClosureStep {
    op: ClosureOp,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
    /// `[start, end)` into the kernel's scratch arena.
    scratch_range: (usize, usize),
    /// SRD-74 Rule 2: the closure runs on `None` inputs.
    accepts_none: bool,
}

/// Common fields shared by all hybrid kernel variants. A clone is a new
/// state of the same program: the steps and the nodes are shared,
/// everything else is the clone's own (engine_parity.md, step 4).
#[derive(Clone)]
struct HybridCore {
    buffer: Vec<u64>,
    coord_count: usize,
    steps: std::sync::Arc<Vec<HybridStep>>,
    output_map: HashMap<String, usize>,
    gather_buf: Vec<u64>,
    scatter_buf: Vec<u64>,
    /// Kernel-owned vector storage; vector-producing ports'
    /// (ptr, len) slots view entries here (§8.4 layer 3).
    scratch: Vec<crate::ast::ScratchBuf>,
    /// Axiom S2: per-slot Ref2 mask — raw readers panic on these.
    ref_slots: Vec<bool>,
    /// Axiom S9(a): (first slot of a Ref pair → scratch index).
    ref_scratch: Vec<(usize, usize)>,
    /// The kernel's value table (SRD 115 §3), shared by every JIT
    /// segment; entries are numbered across segments at build.
    table: crate::kernel::ValueTable,
    /// `(slot, entry)` for every table-kind slot, from the JIT
    /// segments and the handle closures; the H4 validator checks each
    /// slot's handle names its own entry.
    table_entries: Vec<(usize, usize)>,
    /// Port type of each named output, for `get_value`.
    output_types: HashMap<String, crate::ast::PortType>,
    /// Per step, true when it writes a handle slot: such a step is
    /// never marked clean, because its arena bytes or table entry
    /// belong to the cycle that ran it (SRD 115 §4).
    step_rerun: Vec<bool>,
    /// The extern inputs, materialized at the start of every run.
    externs: crate::compile::externs::Externs,
    /// Keep source nodes alive so JIT-baked pointers remain valid.
    _nodes: std::sync::Arc<Vec<Box<dyn PolydatNode>>>,
    /// The coordinates set through the `Kernel` trait, pending
    /// evaluation; `stale` means the next evaluation begins a cycle.
    drive: crate::compile::Drive,
    /// Per slot: the slot holds `None` (SRD-74 on a compiled kernel).
    none: Vec<bool>,
    /// Per step: ran in the cycle that is open.
    ran: Vec<bool>,
    /// Per step: its outputs are current for the inputs it depends on.
    /// Cleared through the plan when an input changes, whichever call
    /// changed it; never set for a volatile or handle-writing step.
    clean: Vec<bool>,
    /// Whether this kernel's provenance mode skips current steps.
    use_clean: bool,
    /// The dirty-register plan: what each input invalidates, what each
    /// output needs.
    plan: std::sync::Arc<crate::compile::Invalidation>,
    /// Per step: nondeterministic or downstream of one, never current.
    volatile: std::sync::Arc<[bool]>,
    /// Per step: a side channel, skipped when current in every mode.
    side: std::sync::Arc<[bool]>,
    /// Per slot: the step that writes it.
    slot_step: std::sync::Arc<[Option<usize>]>,
    /// Where each step came from, for the failure path (A7).
    sites: std::sync::Arc<crate::compile::Attribution>,
    /// The step running, for the failure path.
    cur_step: usize,
}

/// Per-slot `Hdl1` mask over the nodes' output ports.
fn handle_slot_mask_of(
    nodes: &[Box<dyn PolydatNode>],
    port_offsets: &[Vec<usize>],
    total_slots: usize,
) -> Vec<bool> {
    let mut mask = vec![false; total_slots];
    for (node_idx, node) in nodes.iter().enumerate() {
        for (p, out) in node.meta().outs.iter().enumerate() {
            if out.typ.slot_color() == crate::ast::SlotColor::Hdl1 {
                mask[port_offsets[node_idx][p]] = true;
            }
        }
    }
    mask
}

impl HybridCore {
    /// Axiom S9(a) — deterministic Ref validation (see
    /// `jit_boundary.md` §"Slot-state axioms"). Gated to
    /// `debug_assertions` to match its call sites, which compile
    /// out in release.
    #[cfg(debug_assertions)]
    fn validate_refs(&self) {
        // A step that did not run in this cycle, or that propagated
        // `None`, left its slots as they were.
        let skip = |slot: usize| {
            self.none[slot]
                || matches!(self.slot_step.get(slot), Some(Some(step)) if !self.ran[*step])
        };
        for &(slot, idx) in &self.ref_scratch {
            if skip(slot) {
                continue;
            }
            let (p, l) = self.scratch[idx].ptr_len();
            assert!(
                self.buffer[slot] == p && self.buffer[slot + 1] == l,
                "S9 ref-validator: slot pair ({slot}, {}) = ({:#x}, {}) \
                 does not match scratch[{idx}] = ({p:#x}, {l})",
                slot + 1,
                self.buffer[slot],
                self.buffer[slot + 1],
            );
        }
        // SRD 115 axiom H4, the same check for handles: every
        // table-kind slot names its own entry, in this generation.
        for &(slot, entry) in &self.table_entries {
            if skip(slot) {
                continue;
            }
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

    /// Axiom S2 guard for raw u64 readers, and SRD 115 axiom H1 for
    /// handle slots.
    #[inline]
    fn guard_ref_slot(&self, slot: usize) {
        if self.ref_slots.get(slot).copied().unwrap_or(false) {
            panic!(
                "S2 pointer containment: slot {slot} is Ref2- or Hdl1-colored; raw \
                 u64 readers would leak an interior address or a handle. Use the \
                 typed borrow-checked accessor (read_vec_*), the boundary decode, \
                 or copy out."
            );
        }
    }

    /// Axiom S2 typed accessor core (borrow ties to &self).
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
}

impl HybridCore {
    /// Begin a cycle (SRD 115 §4; engine_parity.md, step 5): a hybrid
    /// kernel is driven by a host, never wrapped by a state, so the
    /// cycle is a root one; the externs are written (an unset one as
    /// `None`), what ran last cycle is forgotten, and the volatile
    /// steps are invalidated, as the interpreter does at every
    /// `set_inputs`.
    fn begin_cycle(&mut self) {
        self.table.set_generation(crate::kernel::begin_root_cycle());
        self.externs
            .materialize(&mut self.buffer, &mut self.table, Some(&mut self.none));
        for r in &mut self.ran {
            *r = false;
        }
        for i in 0..self.steps.len() {
            if self.volatile[i] {
                self.clean[i] = false;
            }
        }
        self.drive.stale = false;
    }

    /// An input slot changed, through whichever call: every step the
    /// plan lists for it is no longer current.
    fn dirty_input(&mut self, slot: usize) {
        if let Some(deps) = self.plan.input_dependents.get(slot) {
            for &i in deps {
                self.clean[i] = false;
            }
        }
    }

    /// Run the steps of `order` that have not run in this cycle and are
    /// not current, as the closure kernels do: one rule for every step,
    /// whatever reaches it. A handle-writing step runs every cycle and
    /// a volatile one is never current.
    fn run_steps(&mut self, order: &[usize]) {
        // The table is installed around every step, closures included,
        // so handle closures write through it as the segments' helpers do.
        let mut table = std::mem::take(&mut self.table);
        // The capture guard is armed for the run, so a step's panic is
        // recorded quietly and re-raised enriched, as the interpreter
        // re-raises a node's (A7).
        let capture = crate::kernel::engines::EvalPanicCaptureGuard::arm();
        let outcome = {
            let _installed = crate::kernel::install_value_table(&mut table);
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.run_order(order)))
        };
        self.table = table;
        drop(capture);
        if let Err(payload) = outcome {
            let sites = std::sync::Arc::clone(&self.sites);
            sites.reraise(
                payload,
                self.cur_step,
                &self.buffer,
                Some(&self.none),
                &self.table,
            );
        }
        #[cfg(debug_assertions)]
        self.validate_refs();
    }

    /// The steps of `order` that have not run in the cycle, in order.
    fn run_order(&mut self, order: &[usize]) {
        let steps = std::sync::Arc::clone(&self.steps);
        for &i in order {
            if self.ran[i] {
                continue;
            }
            let never = self.step_rerun[i] || self.volatile[i];
            if (self.use_clean || self.side[i]) && self.clean[i] && !never {
                self.ran[i] = true;
                continue;
            }
            self.cur_step = i;
            self.run_or_propagate(&steps[i]);
            self.ran[i] = true;
            self.clean[i] = !never;
        }
    }

    /// One step: SRD-74 Rule 1, then the segment or the closure. A step
    /// that does not accept `None` emits `None` on every output when
    /// any input is `None`, without running; native code never accepts
    /// it, and a node downstream of an unset extern is a closure, so a
    /// `None` reaches a segment only when a host cleared an extern
    /// after the build.
    #[inline]
    fn run_or_propagate(&mut self, step: &HybridStep) {
        if step.input_slots().iter().any(|&s| self.none[s]) {
            #[cfg(feature = "jit")]
            if let HybridStep::Jit(_) = step {
                panic!(
                    "a `None` reached native code in a hybrid kernel: an extern was cleared \
                     after the build (docs/design/engine_parity.md, A12)"
                );
            }
            if !step.accepts_none() {
                for &s in step.output_slots() {
                    self.none[s] = true;
                }
                return;
            }
        }
        match step {
            #[cfg(feature = "jit")]
            HybridStep::Jit(seg) => {
                // Funnel through the setjmp wrapper so JIT predicate
                // violations surface as catchable panics instead of
                // aborting, as every stand-alone JIT kernel does. The
                // kernel's value table is installed for the segment's
                // helpers (SRD 115 §3).
                let code_fn = seg.code_fn;
                let buf_const = self.buffer.as_ptr();
                let buf_mut = self.buffer.as_mut_ptr();
                crate::compile::jit::invoke_with_catch(move || unsafe {
                    (code_fn)(buf_const, buf_mut);
                });
            }
            HybridStep::Closure(cs) => {
                for (i, &slot) in cs.input_slots.iter().enumerate() {
                    self.gather_buf[i] = self.buffer[slot];
                }
                match &cs.op {
                    ClosureOp::U64(op) => op(
                        &self.gather_buf[..cs.input_slots.len()],
                        &mut self.scatter_buf[..cs.output_slots.len()],
                    ),
                    ClosureOp::Slot(op) => op(
                        &self.gather_buf[..cs.input_slots.len()],
                        &mut self.scatter_buf[..cs.output_slots.len()],
                        &mut self.scratch[cs.scratch_range.0..cs.scratch_range.1],
                    ),
                }
                for (i, &slot) in cs.output_slots.iter().enumerate() {
                    self.buffer[slot] = self.scatter_buf[i];
                }
            }
        }
        for &s in step.output_slots() {
            self.none[s] = false;
        }
    }

    /// Evaluate every output: begin the cycle if none is open, then run
    /// every step that has not run.
    fn eval_all(&mut self) {
        if self.drive.stale {
            self.begin_cycle();
        }
        let all: Vec<usize> = (0..self.steps.len()).collect();
        self.run_steps(&all);
    }

    /// The named output for the cycle's inputs, running only its cone.
    fn pull_named(&mut self, name: &str) -> crate::ast::Value {
        if self.drive.stale {
            self.begin_cycle();
        }
        let plan = std::sync::Arc::clone(&self.plan);
        if let Some(order) = plan.cones.get(name) {
            self.run_steps(order);
        }
        self.value_of(name)
    }

    /// The named output as a typed `Value`, `None` where the slot holds
    /// one; a vector from scratch; a handle copied out.
    fn value_of(&self, name: &str) -> crate::ast::Value {
        let slot = self.output_map[name];
        if self.none.get(slot).copied().unwrap_or(false) {
            return crate::ast::Value::None;
        }
        let ty = self
            .output_types
            .get(name)
            .copied()
            .unwrap_or(crate::ast::PortType::U64);
        if let Some(&(_, idx)) = self.ref_scratch.iter().find(|(s, _)| *s == slot) {
            return self.scratch[idx].to_value();
        }
        crate::compile::marshal::decode_output(&self.buffer, slot, ty, &self.table)
    }
}

/// Everything the evaluation loops once did, kept for the raw kernel's
/// `eval`, which evaluates every step in a new cycle.
#[inline]
fn eval_all_hybrid_steps(core: &mut HybridCore) {
    core.drive.stale = true;
    core.eval_all();
}

/// Compute per-slot provenance bitmasks for the hybrid kernel.
///
/// Returns `slot_provenance[slot]` = bitmask of which inputs affect
/// that buffer slot. Used by pull-side cone guard.
fn compute_hybrid_slot_provenance(
    coord_count: usize,
    total_slots: usize,
    step_dependents: &[Vec<usize>],
    steps: &[HybridStep],
) -> Vec<u64> {
    let step_count = steps.len();
    let mut step_prov = vec![0u64; step_count];
    for (input_idx, deps) in step_dependents.iter().enumerate() {
        for &step_idx in deps {
            if step_idx < step_count {
                step_prov[step_idx] |= 1u64 << input_idx;
            }
        }
    }
    let mut slot_provenance = vec![0u64; total_slots];
    for (i, slot) in slot_provenance
        .iter_mut()
        .enumerate()
        .take(coord_count.min(64))
    {
        *slot = 1u64 << i;
    }
    for (step_idx, step) in steps.iter().enumerate() {
        // Only closure steps carry explicit output_slots; JIT steps use the
        // same buffer region but slot assignment is managed by the JIT code.
        // (Under `not(feature = "jit")` Closure is the only variant, making
        // this pattern irrefutable — that's fine, not a bug.)
        #[allow(irrefutable_let_patterns)]
        if let HybridStep::Closure(cs) = step {
            for &slot in &cs.output_slots {
                if slot < slot_provenance.len() {
                    slot_provenance[slot] = step_prov[step_idx];
                }
            }
        }
    }
    slot_provenance
}

// ═══════════════════════════════════════════════════════════════
// Raw: no provenance, no cone guard. Eval runs all steps.
// ═══════════════════════════════════════════════════════════════

impl HybridCore {
    /// How many steps run as native segments and how many as closures:
    /// what the per-node engine choice decided for this graph.
    fn engine_counts(&self) -> (usize, usize) {
        let closures = self
            .steps
            .iter()
            .filter(|s| matches!(s, HybridStep::Closure(_)))
            .count();
        (self.steps.len() - closures, closures)
    }

    /// Set an extern by name; returns its slot for dirty marking.
    /// Set an extern by name; returns its slot. The plan invalidates
    /// what depends on it, as a changed coordinate is invalidated, and
    /// the next evaluation begins a cycle.
    fn set_extern(&mut self, name: &str, value: crate::ast::Value) -> Result<usize, String> {
        let slot = self.externs.set(name, value, &mut self.buffer)?;
        self.dirty_input(slot);
        self.drive.stale = true;
        Ok(slot)
    }
}

/// Hybrid kernel with no provenance tracking.
///
/// Every `eval()` call runs all steps unconditionally. Useful as a
/// baseline and for graphs where inputs change on every cycle.
#[derive(Clone)]
pub struct HybridKernelRaw {
    core: HybridCore,
}

impl HybridKernelRaw {
    /// The coordinates of the next cycle; a changed one invalidates
    /// its dependents through the plan, as in every mode.
    #[inline]
    fn set_coords(&mut self, coords: &[u64]) {
        for (i, &c) in coords.iter().enumerate().take(self.core.coord_count) {
            if self.core.buffer[i] != c {
                self.core.buffer[i] = c;
                self.core.dirty_input(i);
            }
        }
    }

    /// Evaluate all hybrid steps unconditionally: a new cycle.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.set_coords(coords);
        eval_all_hybrid_steps(&mut self.core);
    }

    #[cfg(feature = "jit")]
    fn pull_in_cycle(&mut self, name: &str) -> crate::ast::Value {
        self.core.pull_named(name)
    }

    /// Set an extern by name, as `PolydatState::set_input` does on the
    /// interpreter. Every run evaluates everything, so it takes effect
    /// at the next run.
    pub fn set_input(&mut self, name: &str, value: crate::ast::Value) -> Result<(), String> {
        self.core.set_extern(name, value).map(|_| ())
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
    /// the interpreter: its `Ext` slot and six scalar projections are
    /// set as externs.
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

    /// Eval all steps and return the value at `slot`.
    #[inline]
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        self.core.guard_ref_slot(slot);
        self.eval(coords);
        self.core.buffer[slot]
    }

    /// Read a named output after `eval()`. Panics on Ref2 slots
    /// (axiom S2) — use `read_vec_*`.
    #[inline]
    pub fn get(&self, name: &str) -> u64 {
        let slot = self.core.output_map[name];
        self.core.guard_ref_slot(slot);
        self.core.buffer[slot]
    }

    /// Read by slot index. Panics on Ref2 slots (axiom S2) —
    /// use `read_vec_*`.
    #[inline]
    pub fn get_slot(&self, slot: usize) -> u64 {
        self.core.guard_ref_slot(slot);
        self.core.buffer[slot]
    }

    crate::compile::ref_readers!();

    /// The named output as a typed `Value`, decoded by its port type:
    /// a handle slot is copied out of the arena or the value table
    /// (SRD 115 §5), so the caller never holds a handle.
    pub fn get_value(&self, name: &str) -> crate::ast::Value {
        self.core.value_of(name)
    }

    /// Entries in the kernel's value table.
    pub fn table_len(&self) -> usize {
        self.core.table.len()
    }

    /// Number of coordinate inputs.
    pub fn coord_count(&self) -> usize {
        self.core.coord_count
    }

    /// The number of native segments and of closure steps in this
    /// kernel, in that order: what the per-node engine choice decided.
    pub fn engine_counts(&self) -> (usize, usize) {
        self.core.engine_counts()
    }

    /// Resolve an output name to its buffer slot.
    pub fn resolve_output(&self, name: &str) -> Option<usize> {
        self.core.output_map.get(name).copied()
    }

    /// The named outputs, as every engine reports them.
    pub fn output_names(&self) -> Vec<&str> {
        self.core.output_map.keys().map(|s| s.as_str()).collect()
    }

    /// Store owned nodes to keep JIT-baked pointers valid.
    pub fn retain_nodes(&mut self, nodes: Vec<Box<dyn PolydatNode>>) {
        self.core._nodes = std::sync::Arc::new(nodes);
    }
}

// ═══════════════════════════════════════════════════════════════
// Pull: cone guard only, no per-step skip.
// set_inputs tracks changed_mask. eval_for_slot checks the cone
// then runs ALL steps if dirty.
// ═══════════════════════════════════════════════════════════════

/// Hybrid kernel with pull-side cone guard.
///
/// `eval_for_slot()` checks whether the output's transitive input
/// cone changed before running steps. If nothing in the cone changed,
/// the cached value is returned without re-evaluation.
#[derive(Clone)]
pub struct HybridKernelPull {
    core: HybridCore,
    slot_provenance: Vec<u64>,
    changed_mask: u64,
    /// Set by `set_input`: an extern changed, so the next evaluation
    /// runs whatever the cone guard says.
    force_run: bool,
}

impl HybridKernelPull {
    /// Track which inputs changed (for the cone guard), and invalidate
    /// their dependents through the plan, as in every mode.
    #[inline]
    fn set_inputs(&mut self, coords: &[u64]) {
        self.changed_mask = 0;
        for (i, &c) in coords.iter().enumerate().take(self.core.coord_count) {
            if self.core.buffer[i] != c {
                self.core.buffer[i] = c;
                self.changed_mask |= 1u64 << i;
                self.core.dirty_input(i);
            }
        }
    }

    /// Evaluate all steps (no cone guard): a new cycle.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.set_inputs(coords);
        self.force_run = false;
        eval_all_hybrid_steps(&mut self.core);
    }

    fn pull_in_cycle(&mut self, name: &str) -> crate::ast::Value {
        self.core.pull_named(name)
    }

    /// Cone guard: if the output's cone is clean, skip eval entirely.
    /// Otherwise run ALL steps (no per-step skip).
    #[inline]
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        self.core.guard_ref_slot(slot);
        self.set_inputs(coords);
        if !self.force_run
            && slot < self.slot_provenance.len()
            && self.slot_provenance[slot] & self.changed_mask == 0
        {
            return self.core.buffer[slot];
        }
        self.force_run = false;
        eval_all_hybrid_steps(&mut self.core);
        self.core.buffer[slot]
    }

    /// Set an extern by name, as `PolydatState::set_input` does on the
    /// interpreter. A carrier takes effect at once; a string, JSON, or
    /// extension value is written at the start of the next run, which
    /// runs whatever the cone guard says.
    pub fn set_input(&mut self, name: &str, value: crate::ast::Value) -> Result<(), String> {
        self.core.set_extern(name, value)?;
        self.force_run = true;
        Ok(())
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
    /// the interpreter: its `Ext` slot and six scalar projections are
    /// set as externs.
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

    /// Read a named output after `eval()`. Panics on Ref2 slots
    /// (axiom S2) — use `read_vec_*`.
    #[inline]
    pub fn get(&self, name: &str) -> u64 {
        let slot = self.core.output_map[name];
        self.core.guard_ref_slot(slot);
        self.core.buffer[slot]
    }

    /// Read by slot index. Panics on Ref2 slots (axiom S2) —
    /// use `read_vec_*`.
    #[inline]
    pub fn get_slot(&self, slot: usize) -> u64 {
        self.core.guard_ref_slot(slot);
        self.core.buffer[slot]
    }

    crate::compile::ref_readers!();

    /// The named output as a typed `Value`, decoded by its port type:
    /// a handle slot is copied out of the arena or the value table
    /// (SRD 115 §5), so the caller never holds a handle.
    pub fn get_value(&self, name: &str) -> crate::ast::Value {
        self.core.value_of(name)
    }

    /// Entries in the kernel's value table.
    pub fn table_len(&self) -> usize {
        self.core.table.len()
    }

    /// Number of coordinate inputs.
    pub fn coord_count(&self) -> usize {
        self.core.coord_count
    }

    /// The number of native segments and of closure steps in this
    /// kernel, in that order: what the per-node engine choice decided.
    pub fn engine_counts(&self) -> (usize, usize) {
        self.core.engine_counts()
    }

    /// Resolve an output name to its buffer slot.
    pub fn resolve_output(&self, name: &str) -> Option<usize> {
        self.core.output_map.get(name).copied()
    }

    /// The named outputs, as every engine reports them.
    pub fn output_names(&self) -> Vec<&str> {
        self.core.output_map.keys().map(|s| s.as_str()).collect()
    }

    /// Store owned nodes to keep JIT-baked pointers valid.
    pub fn retain_nodes(&mut self, nodes: Vec<Box<dyn PolydatNode>>) {
        self.core._nodes = std::sync::Arc::new(nodes);
    }
}

// ═══════════════════════════════════════════════════════════════
// PushPull: push-side per-step skip + pull-side cone guard.
// Full optimization — the production default.
// ═══════════════════════════════════════════════════════════════

/// Hybrid kernel with both push-side per-step skip and pull-side cone guard.
///
/// Push side: `set_inputs()` marks only steps that depend on changed inputs
/// as dirty; clean steps are skipped during `eval()`.
///
/// Pull side: `eval_for_slot()` first checks whether the output's cone of
/// influence changed at all. If not, the cached value is returned without
/// entering the eval loop.
#[derive(Clone)]
pub struct HybridKernelPushPull {
    core: HybridCore,
    slot_provenance: Vec<u64>,
    changed_mask: u64,
    /// Set by `set_input`: an extern changed, so the next evaluation
    /// runs whatever the cone guard says.
    force_run: bool,
}

impl HybridKernelPushPull {
    /// Set an extern by name, as `PolydatState::set_input` does on the
    /// interpreter. A carrier takes effect at once; a string, JSON, or
    /// extension value is written at the start of the next run. Every
    /// step downstream of the extern reruns, and the next evaluation
    /// runs whatever the cone guard says.
    pub fn set_input(&mut self, name: &str, value: crate::ast::Value) -> Result<(), String> {
        self.core.set_extern(name, value)?;
        self.force_run = true;
        Ok(())
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
    /// the interpreter: its `Ext` slot and six scalar projections are
    /// set as externs.
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

    /// Track which inputs changed and dirty affected steps.
    #[inline]
    fn set_inputs(&mut self, coords: &[u64]) {
        self.changed_mask = 0;
        for (i, &c) in coords.iter().enumerate().take(self.core.coord_count) {
            if self.core.buffer[i] != c {
                self.core.buffer[i] = c;
                self.changed_mask |= 1u64 << i;
                self.core.dirty_input(i);
            }
        }
    }

    /// Evaluate with push-side step skip (no cone guard): a new cycle.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.set_inputs(coords);
        self.force_run = false;
        self.core.drive.stale = true;
        self.core.eval_all();
    }

    fn pull_in_cycle(&mut self, name: &str) -> crate::ast::Value {
        self.core.pull_named(name)
    }

    /// Cone guard + push-side skip: the full optimization.
    #[inline]
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        self.core.guard_ref_slot(slot);
        self.set_inputs(coords);
        if !self.force_run
            && slot < self.slot_provenance.len()
            && self.slot_provenance[slot] & self.changed_mask == 0
        {
            return self.core.buffer[slot];
        }
        self.force_run = false;
        self.core.drive.stale = true;
        self.core.eval_all();
        self.core.buffer[slot]
    }

    /// Read a named output after `eval()`. Panics on Ref2 slots
    /// (axiom S2) — use `read_vec_*`.
    #[inline]
    pub fn get(&self, name: &str) -> u64 {
        let slot = self.core.output_map[name];
        self.core.guard_ref_slot(slot);
        self.core.buffer[slot]
    }

    /// Read by slot index. Panics on Ref2 slots (axiom S2) —
    /// use `read_vec_*`.
    #[inline]
    pub fn get_slot(&self, slot: usize) -> u64 {
        self.core.guard_ref_slot(slot);
        self.core.buffer[slot]
    }

    crate::compile::ref_readers!();

    /// The named output as a typed `Value`, decoded by its port type:
    /// a handle slot is copied out of the arena or the value table
    /// (SRD 115 §5), so the caller never holds a handle.
    pub fn get_value(&self, name: &str) -> crate::ast::Value {
        self.core.value_of(name)
    }

    /// Entries in the kernel's value table.
    pub fn table_len(&self) -> usize {
        self.core.table.len()
    }

    /// Number of coordinate inputs.
    pub fn coord_count(&self) -> usize {
        self.core.coord_count
    }

    /// The number of native segments and of closure steps in this
    /// kernel, in that order: what the per-node engine choice decided.
    pub fn engine_counts(&self) -> (usize, usize) {
        self.core.engine_counts()
    }

    /// Resolve an output name to its buffer slot.
    pub fn resolve_output(&self, name: &str) -> Option<usize> {
        self.core.output_map.get(name).copied()
    }

    /// The named outputs, as every engine reports them.
    pub fn output_names(&self) -> Vec<&str> {
        self.core.output_map.keys().map(|s| s.as_str()).collect()
    }

    /// Store owned nodes to keep JIT-baked pointers valid.
    pub fn retain_nodes(&mut self, nodes: Vec<Box<dyn PolydatNode>>) {
        self.core._nodes = std::sync::Arc::new(nodes);
    }
}

/// Type alias for the default hybrid kernel (PushPull — full optimization).
///
/// Assembler and bench code that references `HybridKernel` uses the full
/// push+pull variant. Rename uses to the concrete type if different
/// optimization trade-offs are needed.
pub type HybridKernel = HybridKernelPushPull;

/// Flattened slot list for one node's wire inputs under per-port
/// widths (type_system_alignment.md §8.4 layer 1): every source
/// contributes `slot_width` consecutive slots.
fn flatten_input_slots(
    wiring: &[Vec<WireSource>],
    nodes: &[Box<dyn PolydatNode>],
    node_idx: usize,
    port_offsets: &[Vec<usize>],
    input_starts: &[usize],
    input_widths: &[usize],
) -> Vec<usize> {
    let mut slots = Vec::new();
    for source in &wiring[node_idx] {
        let (start, w) = match source {
            WireSource::Input(c) => (
                input_starts.get(*c).copied().unwrap_or(*c),
                input_widths.get(*c).copied().unwrap_or(1),
            ),
            WireSource::NodeOutput(u, p) => (
                port_offsets[*u][*p],
                nodes[*u].meta().outs[*p].typ.slot_width(),
            ),
        };
        slots.extend(start..start + w);
    }
    slots
}

/// First slot of each Ref2-colored output port of one node, in
/// port order (axiom S3 pairing with CompiledSlotKit scratch).
fn flatten_ref_output_starts(
    nodes: &[Box<dyn PolydatNode>],
    node_idx: usize,
    port_offsets: &[Vec<usize>],
) -> Vec<usize> {
    nodes[node_idx]
        .meta()
        .outs
        .iter()
        .enumerate()
        .filter(|(_, out)| out.typ.slot_color() == crate::ast::SlotColor::Ref2)
        .map(|(p, _)| port_offsets[node_idx][p])
        .collect()
}

/// Flattened slot list for one node's outputs.
fn flatten_output_slots(
    nodes: &[Box<dyn PolydatNode>],
    node_idx: usize,
    port_offsets: &[Vec<usize>],
) -> Vec<usize> {
    let mut slots = Vec::new();
    for (p, out) in nodes[node_idx].meta().outs.iter().enumerate() {
        let start = port_offsets[node_idx][p];
        slots.extend(start..start + out.typ.slot_width());
    }
    slots
}

/// Build a hybrid kernel from resolved DAG data.
///
/// Each node is classified: if it can be JIT-compiled, it goes into
/// a JIT segment. If not, it becomes a closure step. Adjacent JIT-able
/// nodes are batched into a single JIT segment for efficiency.
///
/// Returns a `HybridKernelPushPull` (the production default).
#[cfg(feature = "jit")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_hybrid(
    nodes: &[Box<dyn PolydatNode>],
    wiring: &[Vec<WireSource>],
    coord_count: usize,
    total_slots: usize,
    port_offsets: &[Vec<usize>],
    input_starts: &[usize],
    input_widths: &[usize],
    output_map: HashMap<String, usize>,
    ref_slots: Vec<bool>,
    input_types: &[crate::ast::PortType],
    externs: crate::compile::externs::Externs,
    constant: Vec<bool>,
    volatile: Vec<bool>,
    attribution: std::sync::Arc<crate::compile::Attribution>,
) -> Result<HybridKernelPushPull, String> {
    let mut steps: Vec<HybridStep> = Vec::new();
    let mut scratch: Vec<crate::ast::ScratchBuf> = Vec::new();
    let mut ref_scratch: Vec<(usize, usize)> = Vec::new();
    let mut max_inputs = 0usize;
    let mut max_outputs = 0usize;
    // `(slot, entry)` of every table-kind slot across all JIT segments
    // (SRD 115 §3); the kernel's value table is sized from it.
    let mut table_entries: Vec<(usize, usize)> = Vec::new();
    let handle_mask = handle_slot_mask_of(nodes, port_offsets, total_slots);
    let mut step_rerun: Vec<bool> = Vec::new();

    // Classify each node
    let classifications: Vec<(JitOp, Vec<usize>, Vec<usize>)> = nodes
        .iter()
        .enumerate()
        .map(|(node_idx, node)| {
            // Classified with the wire types known (SRD 115 §6.1), as
            // cones and pure-P3 layouts are: a variadic node whose
            // wires its helper cannot decode falls back to its closure.
            let wire_types: Vec<crate::ast::PortType> = wiring[node_idx]
                .iter()
                .map(|src| match src {
                    WireSource::Input(c) => input_types
                        .get(*c)
                        .copied()
                        .unwrap_or(crate::ast::PortType::U64),
                    WireSource::NodeOutput(j, p) => nodes[*j].meta().outs[*p].typ,
                })
                .collect();
            let jit_op = jit::classify_node_typed(node.as_ref(), &wire_types);

            let input_slots = flatten_input_slots(
                wiring,
                nodes,
                node_idx,
                port_offsets,
                input_starts,
                input_widths,
            );
            let output_slots = flatten_output_slots(nodes, node_idx, port_offsets);

            max_inputs = max_inputs.max(input_slots.len());
            max_outputs = max_outputs.max(output_slots.len());

            (jit_op, input_slots, output_slots)
        })
        .collect();
    // A node downstream of an extern with no value runs as a closure:
    // a `None` propagates through closures as it does on the
    // interpreter (SRD-74), and native code cannot carry one
    // (engine_parity.md, A12).
    let mut classifications = classifications;
    let unset = externs.unset_slots();
    if !unset.is_empty() {
        let mut tainted = vec![false; nodes.len()];
        for node_idx in 0..nodes.len() {
            tainted[node_idx] = wiring[node_idx].iter().any(|src| match src {
                WireSource::Input(c) => unset.contains(&input_starts[*c]),
                WireSource::NodeOutput(j, _) => tainted[*j],
            });
            if tainted[node_idx] {
                classifications[node_idx].0 = JitOp::Fallback;
            }
        }
    }

    // Batch adjacent JIT-able nodes into segments
    let mut i = 0;
    while i < classifications.len() {
        if matches!(classifications[i].0, JitOp::Fallback) {
            // This node needs a closure — scalar u64 op preferred,
            // slot op for slice-bearing nodes (§8.4 layer 3).
            let node = &nodes[i];
            let (_, ref input_slots, ref output_slots) = classifications[i];
            let scratch_start = scratch.len();
            // A handle closure (SRD 115 §7) owns the next entries of
            // the kernel's one table, numbered with the JIT segments'.
            let entry_base = table_entries.len();
            let wire_types: Vec<crate::ast::PortType> = wiring[i]
                .iter()
                .map(|src| match src {
                    WireSource::Input(c) => input_types
                        .get(*c)
                        .copied()
                        .unwrap_or(crate::ast::PortType::U64),
                    WireSource::NodeOutput(j, p) => nodes[*j].meta().outs[*p].typ,
                })
                .collect();
            // A copy of a table handle re-enters the value (axiom H4),
            // so it owns an entry like a handle closure does.
            let table_op = crate::compile::assembly::table_copy_op(node.as_ref(), entry_base)
                .or_else(|| {
                    if node.compiled_u64().is_some() {
                        None
                    } else {
                        node.compiled_handle(entry_base, &wire_types)
                    }
                });
            let op = if let Some(op) = table_op {
                let mut slot = output_slots.iter().copied();
                for port in &node.meta().outs {
                    let first = slot.next();
                    for _ in 1..port.typ.slot_width() {
                        slot.next();
                    }
                    if port.typ.handle_kind() == Some(crate::ast::HandleKind::Table)
                        && let Some(first) = first
                    {
                        table_entries.push((first, table_entries.len()));
                    }
                }
                ClosureOp::U64(op)
            } else if let Some(op) = node.compiled_u64() {
                ClosureOp::U64(op)
            } else if let Some(op) = crate::compile::assembly::identity_op(node.as_ref()) {
                ClosureOp::U64(op)
            } else if let Some(kit) = node.compiled_slot() {
                scratch.extend(kit.scratch.iter().map(|e| crate::ast::ScratchBuf::new(*e)));
                // Axiom S9(a): map this step's Ref output pairs to
                // its scratch entries (port order — S3 contract).
                let starts = flatten_ref_output_starts(nodes, i, port_offsets);
                assert_eq!(
                    starts.len(),
                    kit.scratch.len(),
                    "slot-op scratch/Ref-output mismatch on '{}'",
                    node.meta().name
                );
                for (k, &slot) in starts.iter().enumerate() {
                    ref_scratch.push((slot, scratch_start + k));
                }
                ClosureOp::Slot(kit.op)
            } else {
                return Err(format!(
                    "node '{}' has no compiled form and can't be JIT-compiled",
                    node.meta().name
                ));
            };
            step_rerun.push(output_slots.iter().any(|&s| handle_mask[s]));
            steps.push(HybridStep::Closure(ClosureStep {
                op,
                input_slots: input_slots.clone(),
                output_slots: output_slots.clone(),
                scratch_range: (scratch_start, scratch.len()),
                accepts_none: node.accepts_none_inputs(),
            }));
            i += 1;
        } else {
            // Batch consecutive JIT-able nodes
            let batch_start = i;
            while i < classifications.len() && !matches!(classifications[i].0, JitOp::Fallback) {
                i += 1;
            }
            let batch: Vec<(JitOp, Vec<usize>, Vec<usize>)> =
                classifications[batch_start..i].to_vec();

            // Compile the batch to native code
            let empty_map = HashMap::new();
            let _jit_kernel =
                jit::compile_jit_raw(coord_count, total_slots, batch, empty_map, Vec::new())?;

            // For now, compile each JIT-able node as its own JIT segment.
            // Batching multiple nodes into one segment is a future optimization.
            for (jit_op, input_slots, output_slots) in &classifications[batch_start..i] {
                let single_batch =
                    vec![(jit_op.clone(), input_slots.clone(), output_slots.clone())];
                // Table-kind slots are numbered across every segment
                // so the kernel's one value table serves them all
                // (SRD 115 §3).
                // Slots closures fill with handles or Ref pairs are
                // handle slots to the H1 verifier: a segment may only
                // load, store, and pass them.
                let guarded_slots: Vec<usize> = ref_slots
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| **r)
                    .map(|(s, _)| s)
                    .collect();
                let (code_fn, module, entries) = jit::compile_jit_entry(
                    &single_batch,
                    table_entries.len(),
                    &guarded_slots,
                    None,
                )?;
                table_entries.extend(entries);
                step_rerun.push(output_slots.iter().any(|&s| handle_mask[s]));
                steps.push(HybridStep::Jit(JitSegment {
                    code_fn,
                    _module: crate::compile::jit::JitCode::new(module),
                    input_slots: input_slots.clone(),
                    output_slots: output_slots.clone(),
                }));
            }
        }
    }

    let output_types = output_types_of(nodes, port_offsets, input_starts, input_types, &output_map);
    build_pushpull_from_steps(
        steps,
        scratch,
        ref_scratch,
        ref_slots,
        wiring,
        nodes,
        coord_count,
        total_slots,
        output_map,
        max_inputs,
        max_outputs,
        input_starts,
        input_widths,
        table_entries,
        output_types,
        step_rerun,
        externs,
        constant,
        volatile,
        attribution,
    )
}

/// The port type of each named output, by the slot it names: a node
/// output port's type, or a coordinate input's declared type.
fn output_types_of(
    nodes: &[Box<dyn PolydatNode>],
    port_offsets: &[Vec<usize>],
    input_starts: &[usize],
    input_types: &[crate::ast::PortType],
    output_map: &HashMap<String, usize>,
) -> HashMap<String, crate::ast::PortType> {
    let mut slot_types: HashMap<usize, crate::ast::PortType> = HashMap::new();
    for (start, ty) in input_starts.iter().zip(input_types) {
        slot_types.insert(*start, *ty);
    }
    for (node_idx, node) in nodes.iter().enumerate() {
        for (p, out) in node.meta().outs.iter().enumerate() {
            slot_types.insert(port_offsets[node_idx][p], out.typ);
        }
    }
    output_map
        .iter()
        .map(|(name, slot)| {
            (
                name.clone(),
                slot_types
                    .get(slot)
                    .copied()
                    .unwrap_or(crate::ast::PortType::U64),
            )
        })
        .collect()
}

/// Build a hybrid kernel without JIT (all closures).
#[cfg(not(feature = "jit"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_hybrid(
    nodes: &[Box<dyn PolydatNode>],
    wiring: &[Vec<WireSource>],
    coord_count: usize,
    total_slots: usize,
    port_offsets: &[Vec<usize>],
    input_starts: &[usize],
    input_widths: &[usize],
    output_map: HashMap<String, usize>,
    ref_slots: Vec<bool>,
    input_types: &[crate::ast::PortType],
    externs: crate::compile::externs::Externs,
    constant: Vec<bool>,
    volatile: Vec<bool>,
    attribution: std::sync::Arc<crate::compile::Attribution>,
) -> Result<HybridKernelPushPull, String> {
    let mut steps: Vec<HybridStep> = Vec::new();
    let mut scratch: Vec<crate::ast::ScratchBuf> = Vec::new();
    let mut ref_scratch: Vec<(usize, usize)> = Vec::new();
    let mut max_inputs = 0usize;
    let mut max_outputs = 0usize;
    let handle_mask = handle_slot_mask_of(nodes, port_offsets, total_slots);
    let mut step_rerun: Vec<bool> = Vec::new();
    let mut table_entries: Vec<(usize, usize)> = Vec::new();

    for (node_idx, node) in nodes.iter().enumerate() {
        let input_slots = flatten_input_slots(
            wiring,
            nodes,
            node_idx,
            port_offsets,
            input_starts,
            input_widths,
        );
        let output_slots = flatten_output_slots(nodes, node_idx, port_offsets);

        max_inputs = max_inputs.max(input_slots.len());
        max_outputs = max_outputs.max(output_slots.len());

        let scratch_start = scratch.len();
        // A handle closure (SRD 115 §7) owns the next entries of the
        // kernel's one table, exactly as in the JIT-enabled builder.
        let entry_base = table_entries.len();
        let wire_types: Vec<crate::ast::PortType> = wiring[node_idx]
            .iter()
            .map(|src| match src {
                WireSource::Input(c) => input_types
                    .get(*c)
                    .copied()
                    .unwrap_or(crate::ast::PortType::U64),
                WireSource::NodeOutput(j, p) => nodes[*j].meta().outs[*p].typ,
            })
            .collect();
        // A copy of a table handle re-enters the value (axiom H4), so
        // it owns an entry like a handle closure does.
        let table_op =
            crate::compile::assembly::table_copy_op(node.as_ref(), entry_base).or_else(|| {
                if node.compiled_u64().is_some() {
                    None
                } else {
                    node.compiled_handle(entry_base, &wire_types)
                }
            });
        let op = if let Some(op) = table_op {
            let mut slot = output_slots.iter().copied();
            for port in &node.meta().outs {
                let first = slot.next();
                for _ in 1..port.typ.slot_width() {
                    slot.next();
                }
                if port.typ.handle_kind() == Some(crate::ast::HandleKind::Table)
                    && let Some(first) = first
                {
                    table_entries.push((first, table_entries.len()));
                }
            }
            ClosureOp::U64(op)
        } else if let Some(op) = node.compiled_u64() {
            ClosureOp::U64(op)
        } else if let Some(op) = crate::compile::assembly::identity_op(node.as_ref()) {
            ClosureOp::U64(op)
        } else if let Some(kit) = node.compiled_slot() {
            scratch.extend(kit.scratch.iter().map(|e| crate::ast::ScratchBuf::new(*e)));
            // Axiom S9(a): map this step's Ref output pairs to
            // its scratch entries (port order — S3 contract).
            let starts = flatten_ref_output_starts(nodes, node_idx, port_offsets);
            assert_eq!(
                starts.len(),
                kit.scratch.len(),
                "slot-op scratch/Ref-output mismatch on '{}'",
                node.meta().name
            );
            for (k, &slot) in starts.iter().enumerate() {
                ref_scratch.push((slot, scratch_start + k));
            }
            ClosureOp::Slot(kit.op)
        } else {
            return Err(format!("node '{}' has no compiled form", node.meta().name));
        };
        step_rerun.push(output_slots.iter().any(|&s| handle_mask[s]));
        steps.push(HybridStep::Closure(ClosureStep {
            op,
            input_slots,
            output_slots,
            scratch_range: (scratch_start, scratch.len()),
            accepts_none: node.accepts_none_inputs(),
        }));
    }

    let output_types = output_types_of(nodes, port_offsets, input_starts, input_types, &output_map);
    build_pushpull_from_steps(
        steps,
        scratch,
        ref_scratch,
        ref_slots,
        wiring,
        nodes,
        coord_count,
        total_slots,
        output_map,
        max_inputs,
        max_outputs,
        input_starts,
        input_widths,
        table_entries,
        output_types,
        step_rerun,
        externs,
        constant,
        volatile,
        attribution,
    )
}

/// Shared construction of `HybridKernelPushPull` from assembled steps.
///
/// Computes provenance bitmasks from the DAG wiring and builds the
/// step_dependents list for push-side invalidation and the slot_provenance
/// table for pull-side cone guard.
#[allow(clippy::too_many_arguments)]
fn build_pushpull_from_steps(
    steps: Vec<HybridStep>,
    scratch: Vec<crate::ast::ScratchBuf>,
    ref_scratch: Vec<(usize, usize)>,
    ref_slots: Vec<bool>,
    wiring: &[Vec<WireSource>],
    nodes: &[Box<dyn PolydatNode>],
    coord_count: usize,
    total_slots: usize,
    output_map: HashMap<String, usize>,
    max_inputs: usize,
    max_outputs: usize,
    _input_starts: &[usize],
    input_widths: &[usize],
    mut table_entries: Vec<(usize, usize)>,
    output_types: HashMap<String, crate::ast::PortType>,
    step_rerun: Vec<bool>,
    mut externs: crate::compile::externs::Externs,
    constant: Vec<bool>,
    volatile: Vec<bool>,
    attribution: std::sync::Arc<crate::compile::Attribution>,
) -> Result<HybridKernelPushPull, String> {
    let step_count = steps.len();
    debug_assert_eq!(step_rerun.len(), step_count);
    // Table-kind externs own entries after every segment's and closure's.
    externs.renumber_entries(table_entries.iter().map(|&(_, e)| e + 1).max().unwrap_or(0));
    table_entries.extend(externs.table_entries());
    let table_len = table_entries.iter().map(|&(_, e)| e + 1).max().unwrap_or(0);
    let mut buffer = vec![0u64; total_slots];
    externs.seed(&mut buffer);

    // Compute per-node provenance and invert into per-input step dependents.
    // Since each step currently maps to one node, step index == node index.
    // Dependents come back per-INPUT; expand to per-SLOT so the kernels'
    // slot-indexed dirty tracking / changed-mask bits stay coherent under
    // multi-slot inputs (§8.4 layer 1). Identity for all-scalar inputs.
    let node_provenance = crate::kernel::PolydatProgram::compute_provenance(nodes, wiring);
    let input_dependents =
        crate::kernel::PolydatProgram::compute_dependents(&node_provenance, input_widths.len());
    let step_dependents: Vec<Vec<usize>> = input_widths
        .iter()
        .enumerate()
        .flat_map(|(i, w)| {
            std::iter::repeat_n(input_dependents.get(i).cloned().unwrap_or_default(), *w)
        })
        .collect();

    let slot_provenance =
        compute_hybrid_slot_provenance(coord_count, total_slots, &step_dependents, &steps);

    // The runtime model's lifecycle classification, passed in from the
    // one rule the interpreter's fold applies (step index is node index
    // here).
    debug_assert_eq!(constant.len(), step_count);
    debug_assert_eq!(volatile.len(), step_count);
    let side: Vec<bool> = nodes
        .iter()
        .map(|n| matches!(n.purity(), crate::ast::Purity::SideChannel { .. }))
        .collect();
    let constants: Vec<usize> = (0..step_count).filter(|&i| constant[i]).collect();
    let step_inputs: Vec<&[usize]> = steps.iter().map(|s| s.input_slots()).collect();
    let step_outputs: Vec<&[usize]> = steps.iter().map(|s| s.output_slots()).collect();
    let plan = crate::compile::Invalidation::from_provenance(
        step_dependents.clone(),
        &step_inputs,
        &step_outputs,
        &output_map,
        total_slots,
    );
    let mut slot_step: Vec<Option<usize>> = vec![None; total_slots];
    for (i, outs) in step_outputs.iter().enumerate() {
        for &s in outs.iter() {
            slot_step[s] = Some(i);
        }
    }
    drop(step_inputs);
    drop(step_outputs);

    let mut kernel = HybridKernelPushPull {
        core: HybridCore {
            buffer,
            coord_count,
            steps: std::sync::Arc::new(steps),
            output_map,
            gather_buf: vec![0u64; max_inputs.max(1)],
            scatter_buf: vec![0u64; max_outputs.max(1)],
            scratch,
            ref_slots,
            ref_scratch,
            table: crate::kernel::ValueTable::new(table_len),
            table_entries,
            output_types,
            step_rerun,
            externs,
            _nodes: std::sync::Arc::new(Vec::new()),
            drive: crate::compile::Drive {
                coords: Vec::new(),
                stale: true,
            },
            none: vec![false; total_slots],
            ran: vec![false; step_count],
            clean: vec![false; step_count],
            use_clean: true,
            plan: std::sync::Arc::new(plan),
            volatile: volatile.into(),
            side: side.into(),
            slot_step: slot_step.into(),
            sites: attribution,
            cur_step: 0,
        },
        slot_provenance,
        changed_mask: u64::MAX, // all dirty on first eval
        force_run: false,
    };
    // The compile-constant fold of the runtime model, on this engine: a
    // step no input reaches runs at build, once, and is current from
    // then on, so what is knowable at build is known at build and fails
    // at build.
    kernel.core.begin_cycle();
    kernel.core.run_steps(&constants);
    kernel.core.drive.stale = true;
    Ok(kernel)
}

// ── The engine-independent surface (engine_parity.md, step 4) ──────

#[cfg(feature = "jit")]
impl HybridKernelRaw {
    /// Nothing to mark: every run evaluates everything.
    fn mark_all_dirty(&mut self) {}
}

impl HybridKernelPull {
    /// The next evaluation runs whatever the cone guard says.
    fn mark_all_dirty(&mut self) {
        self.changed_mask = u64::MAX;
        self.force_run = true;
    }
}

impl HybridKernelPushPull {
    /// Every step reruns at the next evaluation.
    fn mark_all_dirty(&mut self) {
        for c in &mut self.core.clean {
            *c = false;
        }
        self.changed_mask = u64::MAX;
        self.force_run = true;
    }

    /// The same program with no provenance: every run evaluates
    /// everything.
    #[cfg(feature = "jit")]
    pub(crate) fn into_raw(self) -> HybridKernelRaw {
        let mut core = self.core;
        core.use_clean = false;
        HybridKernelRaw { core }
    }

    /// The same program with the cone guard only.
    pub(crate) fn into_pull(self) -> HybridKernelPull {
        let mut core = self.core;
        core.use_clean = false;
        HybridKernelPull {
            core,
            slot_provenance: self.slot_provenance,
            changed_mask: u64::MAX,
            force_run: false,
        }
    }
}

use crate::compile::select::{Engine, Provenance};

#[cfg(feature = "jit")]
crate::compile::impl_kernel_trait!(HybridKernelRaw, Engine::Hybrid(Provenance::Raw));
crate::compile::impl_kernel_trait!(HybridKernelPull, Engine::Hybrid(Provenance::Pull));
crate::compile::impl_kernel_trait!(HybridKernelPushPull, Engine::Hybrid(Provenance::PushPull));

/// The pending coordinates through the `Kernel` trait, for the hybrid
/// kernels: every evaluation runs the whole program until the hybrid
/// kernel keeps the closure tier's cone and `None` bookkeeping.
macro_rules! hybrid_drive {
    ($ty:ident, $set_coords:ident) => {
        impl $ty {
            /// The named output through the `Kernel` trait: the pending
            /// coordinates are applied, a cycle begins if none is open,
            /// and only the output's cone runs.
            fn pull_value(&mut self, name: &str) -> crate::ast::Value {
                let coords = std::mem::take(&mut self.core.drive.coords);
                self.$set_coords(&coords);
                self.core.drive.coords = coords;
                self.pull_in_cycle(name)
            }
            fn eval_pending(&mut self) {
                let coords = std::mem::take(&mut self.core.drive.coords);
                self.eval(&coords);
                self.core.drive.coords = coords;
            }
        }
    };
}
#[cfg(feature = "jit")]
hybrid_drive!(HybridKernelRaw, set_coords);
hybrid_drive!(HybridKernelPull, set_inputs);
hybrid_drive!(HybridKernelPushPull, set_inputs);
