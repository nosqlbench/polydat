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
    /// The program nodes in the segment, in step order; the tracker
    /// slot names the one a failure belongs to.
    nodes: Vec<usize>,
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

    /// The program node a failure in this step belongs to: a closure's
    /// own, or the member native code named in the tracker slot.
    #[cfg_attr(not(feature = "jit"), allow(unused_variables))]
    fn failing_node(&self, buffer: &[u64], tracker: usize) -> usize {
        match self {
            #[cfg(feature = "jit")]
            HybridStep::Jit(seg) => seg
                .nodes
                .get(buffer[tracker] as usize)
                .copied()
                .unwrap_or(usize::MAX),
            HybridStep::Closure(cs) => cs.node,
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
    /// The program node, for the failure path.
    node: usize,
}

/// An output resolved for the index-keyed pull: its slot, its type,
/// and the steps of its cone.
type ResolvedOutput = (usize, crate::ast::PortType, Option<std::sync::Arc<[usize]>>);

/// Common fields shared by all hybrid kernel variants. A clone is a new
/// state of the same program: the steps and the nodes are shared,
/// everything else is the clone's own (engine_parity.md, step 4), and
/// every pair in its buffer points into its own storage (axiom S3),
/// never into the state it was cloned from.
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
    /// Port type of each named output, for `get_value`.
    output_types: HashMap<String, crate::ast::PortType>,
    /// The extern inputs, written through at every set.
    externs: crate::compile::externs::Externs,
    /// The traversals the program declares (SRD 113), opened through the
    /// `Kernel` trait.
    traversals: std::sync::Arc<[crate::dsl::traversal::Traversal]>,
    /// Per declared output, its slot, type, and cone, resolved on the
    /// first index-keyed pull (SRD 117 step 3).
    resolved_outputs: Vec<Option<ResolvedOutput>>,
    /// Keep source nodes alive so JIT-baked pointers remain valid.
    _nodes: std::sync::Arc<Vec<Box<dyn PolydatNode>>>,
    /// The coordinates set through the `Kernel` trait, pending
    /// evaluation; `stale` means a write happened since the last
    /// evaluation round.
    drive: crate::compile::Drive,
    /// Per slot: the slot holds `None` (SRD-74 on a compiled kernel).
    none: Vec<bool>,
    /// Per step: the evaluation round it last ran in, so a new round
    /// forgets every run without a scan.
    ran: Vec<u64>,
    /// The evaluation round: advanced by the first evaluation after a
    /// write, so a mode without per-step currency runs a step once per
    /// round rather than once per reader. Bookkeeping only: it wipes
    /// nothing, and every output stands until an input in its
    /// provenance is written. 0 is never a round.
    epoch: u64,
    /// Every step ran in the round: a full evaluation happened.
    all_ran: bool,
    /// Per step: its outputs are current for the inputs it depends on.
    /// Cleared through the plan when an input changes, whichever call
    /// changed it; never set for a volatile step.
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
    /// The slot past the layout where a segment names the member it is
    /// in before calling a helper.
    tracker: usize,
    /// Every step, in order: what `eval` runs.
    all: std::sync::Arc<[usize]>,
    /// Per input slot, the steps an input change marks not current:
    /// the plan's dependents in a push mode; in a raw or pull-only
    /// mode, which never consult a pure step's currency, only the side
    /// channels among them (an optimization over the plan, not a change
    /// to it).
    dirty: std::sync::Arc<[Vec<usize>]>,
    /// Some slot holds `None` in this cycle: an unset extern, which is
    /// the only way one enters (SRD-74). When none does, the steps run
    /// without the mask.
    any_none: bool,
    /// The steps that are never current, invalidated at every round.
    volatile_steps: std::sync::Arc<[usize]>,
}

impl Clone for HybridCore {
    fn clone(&self) -> Self {
        let mut core = HybridCore {
            buffer: self.buffer.clone(),
            coord_count: self.coord_count,
            steps: self.steps.clone(),
            output_map: self.output_map.clone(),
            gather_buf: self.gather_buf.clone(),
            scatter_buf: self.scatter_buf.clone(),
            scratch: self.scratch.clone(),
            ref_slots: self.ref_slots.clone(),
            ref_scratch: self.ref_scratch.clone(),
            output_types: self.output_types.clone(),
            externs: self.externs.clone(),
            traversals: self.traversals.clone(),
            resolved_outputs: self.resolved_outputs.clone(),
            _nodes: self._nodes.clone(),
            drive: self.drive.clone(),
            none: self.none.clone(),
            ran: self.ran.clone(),
            epoch: self.epoch,
            all_ran: self.all_ran,
            clean: self.clean.clone(),
            use_clean: self.use_clean,
            plan: self.plan.clone(),
            volatile: self.volatile.clone(),
            side: self.side.clone(),
            slot_step: self.slot_step.clone(),
            sites: self.sites.clone(),
            cur_step: self.cur_step,
            tracker: self.tracker,
            all: self.all.clone(),
            dirty: self.dirty.clone(),
            any_none: self.any_none,
            volatile_steps: self.volatile_steps.clone(),
        };
        core.republish_refs();
        core
    }
}

impl HybridCore {
    /// Point every pair in the buffer into this state's own storage: a
    /// step's scratch entry for its `Ref2` outputs, the stored value
    /// for an extern's (axiom S3). What a clone needs, whose buffer
    /// was copied from a state whose storage it does not share.
    fn republish_refs(&mut self) {
        for &(slot, idx) in &self.ref_scratch {
            let (p, l) = self.scratch[idx].ptr_len();
            self.buffer[slot] = p;
            self.buffer[slot + 1] = l;
        }
        self.externs.seed(&mut self.buffer, None);
    }
}

/// Whether a node has a `Ref2`-colored port on either side: such a
/// node runs as a closure step until the native lowering carries
/// reference pairs.
#[cfg(feature = "jit")]
fn has_ref_port(node: &dyn PolydatNode, wire_types: &[crate::ast::PortType]) -> bool {
    let is_ref = |t: &crate::ast::PortType| t.slot_color() == crate::ast::SlotColor::Ref2;
    wire_types.iter().any(is_ref) || node.meta().outs.iter().any(|o| is_ref(&o.typ))
}

impl HybridCore {
    /// Axiom S9(a) — deterministic Ref validation (see
    /// `jit_boundary.md` §"Slot-state axioms"). Gated to
    /// `debug_assertions` to match its call sites, which compile
    /// out in release.
    #[cfg(debug_assertions)]
    fn validate_refs(&self) {
        // A step that has never run, or that propagated `None`, left
        // its slots as they were.
        let skip = |slot: usize| {
            self.none[slot]
                || matches!(self.slot_step.get(slot), Some(Some(step)) if self.ran[*step] == 0)
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
    }

    /// Axiom S2 guard for raw u64 readers.
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
    /// Begin an evaluation round after a write: take what cells other
    /// holders published, forget what ran in the last round, and
    /// invalidate the volatile steps, as the interpreter does at every
    /// write. Nothing else changes: every output stands until an input
    /// in its provenance is written (runtime_model.md, R1).
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

    /// Whether this kernel skips current steps, and with it which steps
    /// an input change marks: every dependent, or only the side
    /// channels when a pure step's currency is never consulted.
    #[cfg(feature = "jit")]
    fn set_use_clean(&mut self, on: bool) {
        self.use_clean = on;
        let side = std::sync::Arc::clone(&self.side);
        self.dirty = self
            .plan
            .input_dependents
            .iter()
            .map(|deps| {
                if on {
                    deps.clone()
                } else {
                    deps.iter().copied().filter(|&i| side[i]).collect()
                }
            })
            .collect::<Vec<_>>()
            .into();
    }

    /// Every dependent of a slot a cell refresh changed runs again,
    /// inside the open cycle too, as the interpreter re-evaluates a node
    /// whose cell moved on its next read: the plan's dependents, whatever
    /// the mode, are neither run nor current.
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

    /// Take the current value of every cell another holder published
    /// to, and mark its dependents, so a pull inside a cycle sees the
    /// register as the interpreter's revision check does.
    #[inline]
    fn refresh_cells(&mut self) {
        if self.externs.cells_dirty() {
            self.externs.refresh_cells(&mut self.buffer);
            self.dirty_refreshed();
        }
    }

    /// Bind a `shared` binding to `cell` (engine parity, step 9): this
    /// kernel reads and writes that register from now on.
    fn attach_cell(&mut self, name: &str, cell: crate::kernel::SharedCell) -> Result<(), String> {
        let slot = self.externs.attach_cell(name, cell)?;
        self.dirty_input(slot);
        self.drive.stale = true;
        Ok(())
    }

    /// An input slot changed, through whichever call: every step the
    /// plan lists for it is no longer current.
    #[inline]
    fn dirty_input(&mut self, slot: usize) {
        if let Some(deps) = self.dirty.get(slot) {
            for &i in deps {
                self.clean[i] = false;
            }
        }
    }

    /// Run the steps of `order` that have not run in this round and are
    /// not current, as the closure kernels do: one rule for every step,
    /// whatever reaches it; a volatile step is never current.
    #[inline]
    fn run_steps(&mut self, order: &[usize]) {
        self.run_guarded(|core| core.run_order(order));
    }

    /// Run `body` with the capture guard armed, so a step's panic is
    /// recorded quietly and re-raised enriched, as the interpreter
    /// re-raises a node's (A7).
    #[inline]
    fn run_guarded(&mut self, body: impl FnOnce(&mut Self)) {
        let capture = crate::kernel::engines::EvalPanicCaptureGuard::arm();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(self)));
        drop(capture);
        if let Err(payload) = outcome {
            let sites = std::sync::Arc::clone(&self.sites);
            let node = self.steps[self.cur_step].failing_node(&self.buffer, self.tracker);
            sites.reraise(payload, node, &self.buffer, Some(&self.none));
        }
        #[cfg(debug_assertions)]
        self.validate_refs();
    }

    /// The steps of `order` that have not run in the round, in order.
    #[inline]
    fn run_order(&mut self, order: &[usize]) {
        let steps = &self.steps;
        let none_free = !self.any_none;
        for &i in order {
            if self.all_ran || self.ran[i] == self.epoch {
                continue;
            }
            let never = self.volatile[i];
            if (self.use_clean || self.side[i]) && self.clean[i] && !never {
                self.ran[i] = self.epoch;
                continue;
            }
            self.cur_step = i;
            run_hybrid_step(
                &steps[i],
                none_free,
                &mut self.buffer,
                &mut self.none,
                &mut self.gather_buf,
                &mut self.scatter_buf,
                &mut self.scratch,
            );
            self.ran[i] = self.epoch;
            self.clean[i] = !never;
        }
    }

    /// Every step, in order, in a round just begun, in a mode without
    /// per-step skipping and with no `None` in play: the same steps the
    /// general loop would run, without the bookkeeping a partial round
    /// needs. A current side channel is still skipped, since its run is
    /// observed.
    #[inline]
    fn run_fresh(&mut self) {
        let steps = &self.steps;
        for (i, step) in steps.iter().enumerate() {
            if self.side[i] {
                let never = self.volatile[i];
                if self.clean[i] && !never {
                    continue;
                }
                self.clean[i] = !never;
            }
            self.cur_step = i;
            run_hybrid_step(
                step,
                true,
                &mut self.buffer,
                &mut self.none,
                &mut self.gather_buf,
                &mut self.scatter_buf,
                &mut self.scratch,
            );
        }
        self.all_ran = true;
    }

    /// Evaluate every output: begin the cycle if none is open, then run
    /// every step that has not run.
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

    /// The named output for the cycle's inputs, running only its cone.
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

    /// [`Self::pull_named`] by output index: the name is resolved to
    /// its slot, type, and cone once, so a pull costs no string lookup
    /// (SRD 117 step 3).
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

    /// The named output as a typed `Value`, `None` where the slot holds
    /// one; a vector from scratch; a handle copied out.
    fn value_of(&self, name: &str) -> crate::ast::Value {
        let slot = self.output_map[name];
        let ty = self
            .output_types
            .get(name)
            .copied()
            .unwrap_or(crate::ast::PortType::U64);
        self.slot_value(slot, ty)
    }

    /// The value at `slot` decoded as `ty`: `None` where the mask says
    /// so, a Ref pair copied out through the pair.
    fn slot_value(&self, slot: usize, ty: crate::ast::PortType) -> crate::ast::Value {
        if self.none.get(slot).copied().unwrap_or(false) {
            return crate::ast::Value::None;
        }
        crate::compile::marshal::decode_output(&self.buffer, slot, ty)
    }

    /// The native segments and the closure steps.
    fn plan(&self) -> crate::EnginePlan {
        let (native_segments, closure_steps) = self.engine_counts();
        crate::EnginePlan {
            native_segments,
            closure_steps,
            interpreted_nodes: 0,
        }
    }

    /// Nothing is current: every step runs at the next evaluation.
    fn invalidate_all(&mut self) {
        self.clean.fill(false);
        self.all_ran = false;
        self.drive.stale = true;
    }
}

/// Everything the evaluation loops once did, kept for the raw kernel's
/// `eval`, which evaluates every step in a new cycle.
#[inline]
fn eval_all_hybrid_steps(core: &mut HybridCore) {
    core.drive.stale = true;
    core.eval_all();
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
        let (slot, unset) = self.externs.set(name, value, &mut self.buffer)?;
        self.extern_written(slot, unset);
        Ok(slot)
    }

    /// [`Self::set_extern`] by input index.
    fn set_extern_at(&mut self, index: usize, value: crate::ast::Value) -> Result<usize, String> {
        let (slot, unset) = self.externs.set_at(index, value, &mut self.buffer)?;
        self.extern_written(slot, unset);
        Ok(slot)
    }

    /// An extern was written: its dependents are no longer current, the
    /// `None` mask records whether it is unset (SRD-74 on a compiled
    /// kernel), and the next evaluation begins a round. When the last
    /// unset extern is set, no slot can hold a `None` any more, so the
    /// mask is cleared and the steps run without it.
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

    /// [`Self::set_input`] by input index.
    pub fn set_input_at(&mut self, index: usize, value: crate::ast::Value) -> Result<(), String> {
        self.core.set_extern_at(index, value).map(|_| ())
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
    slot_provenance: Vec<crate::kernel::ProvMask>,
    changed_mask: crate::kernel::ProvMask,
    /// Set by `set_input`: an extern changed, so the next evaluation
    /// runs whatever the cone guard says.
    force_run: bool,
}

impl HybridKernelPull {
    /// Track which inputs changed (for the cone guard), and invalidate
    /// their dependents through the plan, as in every mode.
    #[inline]
    fn set_inputs(&mut self, coords: &[u64]) {
        self.changed_mask.clear();
        for (i, &c) in coords.iter().enumerate().take(self.core.coord_count) {
            if self.core.buffer[i] != c {
                self.core.buffer[i] = c;
                self.changed_mask.set(i);
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
            && !self.slot_provenance[slot].intersects(&self.changed_mask)
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

    /// [`Self::set_input`] by input index.
    pub fn set_input_at(&mut self, index: usize, value: crate::ast::Value) -> Result<(), String> {
        self.core.set_extern_at(index, value)?;
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
    slot_provenance: Vec<crate::kernel::ProvMask>,
    changed_mask: crate::kernel::ProvMask,
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

    /// [`Self::set_input`] by input index.
    pub fn set_input_at(&mut self, index: usize, value: crate::ast::Value) -> Result<(), String> {
        self.core.set_extern_at(index, value)?;
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
        self.changed_mask.clear();
        for (i, &c) in coords.iter().enumerate().take(self.core.coord_count) {
            if self.core.buffer[i] != c {
                self.core.buffer[i] = c;
                self.changed_mask.set(i);
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
            && !self.slot_provenance[slot].intersects(&self.changed_mask)
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
            // A node with a `Ref2` port runs as a closure until the
            // native lowering carries reference pairs.
            let jit_op = if has_ref_port(node.as_ref(), &wire_types) {
                JitOp::Fallback
            } else {
                jit::classify_node_typed(node.as_ref(), &wire_types)
            };

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

    // Per node, the step it runs in: its own closure step or its segment.
    let mut node_step = vec![usize::MAX; nodes.len()];
    // Batch adjacent JIT-able nodes into segments
    let mut i = 0;
    while i < classifications.len() {
        if matches!(classifications[i].0, JitOp::Fallback) {
            // This node needs a closure — scalar u64 op preferred,
            // slot op for slice-bearing nodes (§8.4 layer 3).
            let node = &nodes[i];
            let (_, ref input_slots, ref output_slots) = classifications[i];
            let scratch_start = scratch.len();
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
            let op = if let Some(op) = node.compiled_u64() {
                ClosureOp::U64(op)
            } else if let Some(op) = crate::compile::assembly::identity_op(node.as_ref()) {
                ClosureOp::U64(op)
            } else if let Some(kit) = ref_copy_or_slot(node.as_ref(), &wire_types) {
                scratch.extend(kit.scratch.iter().map(|e| crate::ast::ScratchBuf::new(*e)));
                let starts = flatten_ref_output_starts(nodes, i, port_offsets);
                ref_scratch.extend(crate::compile::assembly::scratch_pairs(
                    &node.meta().name,
                    &starts,
                    &kit.scratch,
                    scratch_start,
                ));
                ClosureOp::Slot(kit.op)
            } else {
                return Err(format!(
                    "node '{}' has no compiled form and can't be JIT-compiled",
                    node.meta().name
                ));
            };
            node_step[i] = steps.len();
            steps.push(HybridStep::Closure(ClosureStep {
                op,
                input_slots: input_slots.clone(),
                output_slots: output_slots.clone(),
                scratch_range: (scratch_start, scratch.len()),
                accepts_none: node.accepts_none_inputs(),
                node: i,
            }));
            i += 1;
        } else {
            // Batch consecutive JIT-able nodes of one lifecycle: a segment is
            // folded at build only if every member is compile-constant, so
            // a constant node never joins a segment that is not, or the
            // constant steps after it would run before their producer.
            let batch_start = i;
            while i < classifications.len()
                && !matches!(classifications[i].0, JitOp::Fallback)
                && constant[i] == constant[batch_start]
            {
                i += 1;
            }
            // One native segment for the batch: its boundary inputs are
            // the slots the batch reads and does not write, its outputs
            // every slot it writes. Table-kind slots are numbered across
            // every segment so the kernel's one value table serves them
            // all (SRD 115 §3); slots closures fill with handles or Ref
            // pairs are handle slots to the H1 verifier, which a segment
            // may only load, store, and pass. Native code names the
            // member it is in through the tracker slot, for the failure
            // path (A7).
            let batch: Vec<(JitOp, Vec<usize>, Vec<usize>)> =
                classifications[batch_start..i].to_vec();
            let written: std::collections::HashSet<usize> = batch
                .iter()
                .flat_map(|(_, _, o)| o.iter().copied())
                .collect();
            let mut input_slots: Vec<usize> = Vec::new();
            for (_, ins, _) in &batch {
                for &s in ins {
                    if !written.contains(&s) && !input_slots.contains(&s) {
                        input_slots.push(s);
                    }
                }
            }
            let output_slots: Vec<usize> = batch
                .iter()
                .flat_map(|(_, _, o)| o.iter().copied())
                .collect();
            let (code_fn, module) = jit::compile_jit_entry(&batch, Some(total_slots))?;
            let segment = steps.len();
            for s in &mut node_step[batch_start..i] {
                *s = segment;
            }
            steps.push(HybridStep::Jit(JitSegment {
                code_fn,
                _module: crate::compile::jit::JitCode::new(module),
                input_slots,
                output_slots,
                nodes: (batch_start..i).collect(),
            }));
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
        output_types,
        externs,
        constant,
        volatile,
        attribution,
        node_step,
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
        let op = if let Some(op) = node.compiled_u64() {
            ClosureOp::U64(op)
        } else if let Some(op) = crate::compile::assembly::identity_op(node.as_ref()) {
            ClosureOp::U64(op)
        } else if let Some(kit) = ref_copy_or_slot(node.as_ref(), &wire_types) {
            scratch.extend(kit.scratch.iter().map(|e| crate::ast::ScratchBuf::new(*e)));
            let starts = flatten_ref_output_starts(nodes, node_idx, port_offsets);
            ref_scratch.extend(crate::compile::assembly::scratch_pairs(
                &node.meta().name,
                &starts,
                &kit.scratch,
                scratch_start,
            ));
            ClosureOp::Slot(kit.op)
        } else {
            return Err(format!("node '{}' has no compiled form", node.meta().name));
        };
        steps.push(HybridStep::Closure(ClosureStep {
            op,
            input_slots,
            output_slots,
            scratch_range: (scratch_start, scratch.len()),
            accepts_none: node.accepts_none_inputs(),
            node: node_idx,
        }));
    }
    let node_step: Vec<usize> = (0..nodes.len()).collect();

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
        output_types,
        externs,
        constant,
        volatile,
        attribution,
        node_step,
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
    output_types: HashMap<String, crate::ast::PortType>,
    externs: crate::compile::externs::Externs,
    constant: Vec<bool>,
    volatile: Vec<bool>,
    attribution: std::sync::Arc<crate::compile::Attribution>,
    node_step: Vec<usize>,
) -> Result<HybridKernelPushPull, String> {
    let step_count = steps.len();
    debug_assert_eq!(node_step.len(), nodes.len());
    debug_assert!(node_step.iter().all(|&s| s < step_count));
    // Node lists from the runtime model become step lists: a segment
    // depends on what any member depends on.
    let to_steps = |list: &[usize]| -> Vec<usize> {
        let mut v: Vec<usize> = list.iter().map(|&n| node_step[n]).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    // One slot past the layout is the tracker (A7).
    let mut buffer = vec![0u64; total_slots + 1];
    let mut none = vec![false; total_slots];
    let any_none = externs.seed(&mut buffer, Some(&mut none));

    // Compute per-node provenance and invert into per-input step dependents.
    // Since each step currently maps to one node, step index == node index.
    // Dependents come back per-INPUT; expand to per-SLOT so the kernels'
    // slot-indexed dirty tracking / changed-mask bits stay coherent under
    // multi-slot inputs (§8.4 layer 1). Identity for all-scalar inputs.
    let node_provenance = crate::kernel::PolydatProgram::compute_provenance(nodes, wiring);
    let input_dependents: Vec<Vec<usize>> =
        crate::kernel::PolydatProgram::compute_dependents(&node_provenance, input_widths.len())
            .iter()
            .map(|d| to_steps(d))
            .collect();
    let step_dependents: Vec<Vec<usize>> = input_widths
        .iter()
        .enumerate()
        .flat_map(|(i, w)| {
            std::iter::repeat_n(input_dependents.get(i).cloned().unwrap_or_default(), *w)
        })
        .collect();

    let step_outs: Vec<&[usize]> = steps.iter().map(|s| s.output_slots()).collect();
    let slot_provenance =
        crate::compile::slot_provenance(coord_count, total_slots, &step_outs, &step_dependents);

    // The runtime model's lifecycle classification, passed in per node
    // from the one rule the interpreter's fold applies, folded onto the
    // steps: a segment is constant only if every member is, volatile
    // or a side channel if any member is.
    debug_assert_eq!(constant.len(), nodes.len());
    debug_assert_eq!(volatile.len(), nodes.len());
    let mut step_constant = vec![true; step_count];
    let mut step_volatile = vec![false; step_count];
    let mut side = vec![false; step_count];
    for (n, node) in nodes.iter().enumerate() {
        let s = node_step[n];
        step_constant[s] &= constant[n];
        step_volatile[s] |= volatile[n];
        side[s] |= matches!(node.purity(), crate::ast::Purity::SideChannel { .. });
    }
    let volatile = step_volatile;
    let constants: Vec<usize> = (0..step_count).filter(|&i| step_constant[i]).collect();
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

    let dirty: Vec<Vec<usize>> = plan.input_dependents.clone();
    let volatile_steps: Vec<usize> = (0..step_count).filter(|&i| volatile[i]).collect();
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
            output_types,
            externs,
            traversals: Vec::new().into(),
            resolved_outputs: Vec::new(),
            _nodes: std::sync::Arc::new(Vec::new()),
            drive: crate::compile::Drive {
                coords: Vec::new(),
                stale: true,
            },
            none,
            ran: vec![0; step_count],
            epoch: 0,
            all_ran: false,
            clean: vec![false; step_count],
            use_clean: true,
            plan: std::sync::Arc::new(plan),
            volatile: volatile.into(),
            side: side.into(),
            slot_step: slot_step.into(),
            sites: attribution,
            cur_step: 0,
            tracker: total_slots,
            all: (0..step_count).collect::<Vec<usize>>().into(),
            dirty: dirty.into(),
            any_none,
            volatile_steps: volatile_steps.into(),
        },
        slot_provenance,
        changed_mask: crate::kernel::ProvMask::all_below(coord_count), // all dirty on first eval
        force_run: false,
    };
    // The compile-constant fold of the runtime model, on this engine: a
    // step no input reaches runs at build, once, and is current from
    // then on, so what is knowable at build is known at build and fails
    // at build.
    kernel.core.begin_epoch();
    kernel.core.run_steps(&constants);
    kernel.core.drive.stale = true;
    Ok(kernel)
}

/// The slot kit for a closure step: a copy of a `Ref2` value into the
/// step's own scratch (`identity`, a `__port_` passthrough; axiom S3),
/// else the node's own kit.
fn ref_copy_or_slot(
    node: &dyn PolydatNode,
    wire_types: &[crate::ast::PortType],
) -> Option<crate::ast::CompiledSlotKit> {
    let meta = node.meta();
    if (meta.name == "identity" || meta.name.starts_with("__port_"))
        && meta.outs.len() == 1
        && meta.outs[0].typ.slot_color() == crate::ast::SlotColor::Ref2
    {
        return crate::compile::assembly::ref_copy_kit(meta.outs[0].typ);
    }
    node.compiled_slot(wire_types)
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
        self.changed_mask = crate::kernel::ProvMask::all_below(self.core.coord_count);
        self.force_run = true;
    }
}

impl HybridKernelPushPull {
    /// Every step reruns at the next evaluation.
    fn mark_all_dirty(&mut self) {
        for c in &mut self.core.clean {
            *c = false;
        }
        self.changed_mask = crate::kernel::ProvMask::all_below(self.core.coord_count);
        self.force_run = true;
    }

    /// The same program with no provenance: every run evaluates
    /// everything.
    #[cfg(feature = "jit")]
    pub(crate) fn into_raw(self) -> HybridKernelRaw {
        let mut core = self.core;
        core.set_use_clean(false);
        HybridKernelRaw { core }
    }

    /// The same program with the cone guard only.
    #[cfg(feature = "jit")]
    pub(crate) fn into_pull(self) -> HybridKernelPull {
        let mut core = self.core;
        core.set_use_clean(false);
        let changed_mask = crate::kernel::ProvMask::all_below(core.coord_count);
        HybridKernelPull {
            core,
            slot_provenance: self.slot_provenance,
            changed_mask,
            force_run: false,
        }
    }
}

use crate::compile::select::{Engine, Provenance};

#[cfg(feature = "jit")]
crate::compile::impl_kernel_trait!(HybridKernelRaw, Engine::Native(Provenance::Raw));
crate::compile::impl_kernel_trait!(HybridKernelPull, Engine::Native(Provenance::Pull));
crate::compile::impl_kernel_trait!(HybridKernelPushPull, Engine::Native(Provenance::PushPull));

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
            /// [`Self::pull_value`] by output index.
            fn pull_value_at(&mut self, index: usize) -> crate::ast::Value {
                let coords = std::mem::take(&mut self.core.drive.coords);
                self.$set_coords(&coords);
                self.core.drive.coords = coords;
                self.core.pull_at(index)
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

/// One step: SRD-74 Rule 1, then the segment or the closure. A step
/// that does not accept `None` emits `None` on every output when any
/// input is `None`, without running; native code never accepts it, and
/// a node downstream of an unset extern is a closure, so a `None`
/// reaches a segment only when a host cleared an extern after the
/// build. With `none_free` the mask is known clear and is not read.
#[inline(always)]
fn run_hybrid_step(
    step: &HybridStep,
    none_free: bool,
    buffer: &mut [u64],
    none: &mut [bool],
    gather: &mut [u64],
    scatter: &mut [u64],
    scratch: &mut [crate::ast::ScratchBuf],
) {
    if !none_free && step.input_slots().iter().any(|&s| none[s]) {
        #[cfg(feature = "jit")]
        if let HybridStep::Jit(_) = step {
            panic!(
                "a `None` reached native code in a hybrid kernel: an extern was cleared \
                 after the build (docs/design/engine_parity.md, A12)"
            );
        }
        if !step.accepts_none() {
            for &s in step.output_slots() {
                none[s] = true;
            }
            return;
        }
    }
    match step {
        #[cfg(feature = "jit")]
        HybridStep::Jit(seg) => {
            // Through the setjmp wrapper, so a helper's failure is the
            // longjmp the kernel catches rather than an abort.
            let code_fn = seg.code_fn;
            let buf_const = buffer.as_ptr();
            let buf_mut = buffer.as_mut_ptr();
            crate::compile::jit::invoke_with_catch(move || unsafe {
                (code_fn)(buf_const, buf_mut);
            });
        }
        HybridStep::Closure(cs) => {
            for (i, &slot) in cs.input_slots.iter().enumerate() {
                gather[i] = buffer[slot];
            }
            match &cs.op {
                ClosureOp::U64(op) => op(
                    &gather[..cs.input_slots.len()],
                    &mut scatter[..cs.output_slots.len()],
                ),
                ClosureOp::Slot(op) => op(
                    &gather[..cs.input_slots.len()],
                    &mut scatter[..cs.output_slots.len()],
                    &mut scratch[cs.scratch_range.0..cs.scratch_range.1],
                ),
            }
            for (i, &slot) in cs.output_slots.iter().enumerate() {
                buffer[slot] = scatter[i];
            }
        }
    }
    if !none_free {
        for &s in step.output_slots() {
            none[s] = false;
        }
    }
}
