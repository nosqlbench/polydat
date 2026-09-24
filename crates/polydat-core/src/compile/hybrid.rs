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
//! Three kernel types. They differ in what `set_inputs` marks and
//! whether `eval_for_slot` consults the cone guard; the shared step
//! loop reads the mode's `use_clean` flag per step:
//!
//! | Type | Push (per-step skip) | Pull (cone guard) |
//! |------|---------------------|-------------------|
//! | `HybridKernelRaw` | — | — |
//! | `HybridKernelPull` | — | yes |
//! | `HybridKernelPushPull` | yes | yes |

use std::collections::HashMap;

use crate::ast::SlotShape;
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
    code_fn: crate::compile::jit::NativeFn,
    /// The finalized native code, shared by every kernel created from
    /// one program.
    _module: crate::compile::jit::JitCode,
    /// Whether the code calls a helper, and so runs under the longjmp
    /// catch; code with no call runs bare.
    fallible: bool,
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
/// with kernel-owned scratch for typed-slice ports
/// (type_system_alignment.md §4, compiled_handles.md §3).
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
/// everything else is the clone's own (engines.md §3.5), and
/// every pair in its buffer points into its own storage (axiom S3),
/// never into the state it was cloned from.
struct HybridCore {
    /// The engine this kernel runs, as it reports it: the tier and
    /// the provenance mode it was built with. State rather than a
    /// property of the type, so one kernel type can serve a tier
    /// that runs native code and one that runs none.
    engine: crate::compile::select::Engine,
    buffer: Vec<u64>,
    coord_count: usize,
    steps: std::sync::Arc<Vec<HybridStep>>,
    output_map: HashMap<String, usize>,
    gather_buf: Vec<u64>,
    scatter_buf: Vec<u64>,
    /// Kernel-owned vector storage; vector-producing ports'
    /// (ptr, len) slots view entries here (type_system_alignment.md
    /// §4, compiled_handles.md §3).
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
    /// Some slot holds `None`: an unset extern, which is
    /// the only way one enters (SRD-74). When none does, the steps run
    /// without the mask.
    any_none: bool,
    /// The steps that are never current, invalidated at every round.
    volatile_steps: std::sync::Arc<[usize]>,
}

impl Clone for HybridCore {
    fn clone(&self) -> Self {
        let mut core = HybridCore {
            engine: self.engine,
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
    crate::compile::shared_core_methods!();
}

impl HybridCore {
    /// Whether this kernel skips current steps, and with it which steps
    /// an input change marks: every dependent, or only the side
    /// channels when a pure step's currency is never consulted.
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

    /// The program node the step now running belongs to, for the
    /// failure path (A7). A step here can be a run of native code over
    /// several nodes, so the tracker slot names the member. Read by
    /// `run_guarded` and by the build-time `fold_steps`, so the two
    /// always name the same node.
    #[inline]
    fn failing_node(&self) -> usize {
        self.steps[self.cur_step].failing_node(&self.buffer, self.tracker)
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

    /// The native segments and the closure steps.
    fn plan(&self) -> crate::EnginePlan {
        let (native_segments, closure_steps) = self.engine_counts();
        crate::EnginePlan {
            native_segments,
            closure_steps,
            interpreted_nodes: 0,
        }
    }
}

/// Everything the evaluation loops once did, kept for the raw kernel's
/// `eval`, which evaluates every step in a new round.
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
}

/// Hybrid kernel with no provenance tracking.
///
/// Every `eval()` call runs all steps unconditionally. Useful as a
/// baseline and for graphs where inputs change on every evaluation.
#[derive(Clone)]
pub struct HybridKernelRaw {
    core: HybridCore,
}

impl HybridKernelRaw {
    crate::compile::kernel_accessors!(set_coords);
    /// The coordinates, written; a changed one invalidates
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

    /// Evaluate all hybrid steps unconditionally: a new round.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.set_coords(coords);
        eval_all_hybrid_steps(&mut self.core);
    }

    /// Set an extern by name, as `PolydatState::set_input` does on the
    /// interpreter. Every run evaluates everything, so it takes effect
    /// at the next run.
    pub fn set_input(
        &mut self,
        name: &str,
        value: crate::ast::Value,
    ) -> Result<(), crate::kernel::WriteError> {
        self.core.set_extern(name, value).map(|_| ())
    }

    /// [`Self::set_input`] by input index.
    pub fn set_input_at(
        &mut self,
        index: usize,
        value: crate::ast::Value,
    ) -> Result<(), crate::kernel::WriteError> {
        self.core.set_extern_at(index, value).map(|_| ())
    }

    /// Eval all steps and return the value at `slot`.
    #[inline]
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        self.core.guard_ref_slot(slot);
        self.eval(coords);
        self.core.buffer[slot]
    }

    /// The number of native segments and of closure steps in this
    /// kernel, in that order: what the per-node engine choice decided.
    pub fn engine_counts(&self) -> (usize, usize) {
        self.core.engine_counts()
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
    crate::compile::kernel_accessors!(set_inputs);
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

    /// Evaluate all steps (no cone guard): a new round.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.set_inputs(coords);
        self.force_run = false;
        eval_all_hybrid_steps(&mut self.core);
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
    /// interpreter. Every kind is written through at once, and the
    /// next run runs whatever the cone guard says.
    pub fn set_input(
        &mut self,
        name: &str,
        value: crate::ast::Value,
    ) -> Result<(), crate::kernel::WriteError> {
        self.core.set_extern(name, value)?;
        self.force_run = true;
        Ok(())
    }

    /// [`Self::set_input`] by input index.
    pub fn set_input_at(
        &mut self,
        index: usize,
        value: crate::ast::Value,
    ) -> Result<(), crate::kernel::WriteError> {
        self.core.set_extern_at(index, value)?;
        self.force_run = true;
        Ok(())
    }

    /// The number of native segments and of closure steps in this
    /// kernel, in that order: what the per-node engine choice decided.
    pub fn engine_counts(&self) -> (usize, usize) {
        self.core.engine_counts()
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
    crate::compile::kernel_accessors!(set_inputs);
    /// Set an extern by name, as `PolydatState::set_input` does on the
    /// interpreter. Every kind is written through at once. Every step
    /// downstream of the extern reruns, and the next evaluation runs
    /// whatever the cone guard says.
    pub fn set_input(
        &mut self,
        name: &str,
        value: crate::ast::Value,
    ) -> Result<(), crate::kernel::WriteError> {
        self.core.set_extern(name, value)?;
        self.force_run = true;
        Ok(())
    }

    /// [`Self::set_input`] by input index.
    pub fn set_input_at(
        &mut self,
        index: usize,
        value: crate::ast::Value,
    ) -> Result<(), crate::kernel::WriteError> {
        self.core.set_extern_at(index, value)?;
        self.force_run = true;
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

    /// Evaluate with push-side step skip (no cone guard): a new round.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.set_inputs(coords);
        self.force_run = false;
        self.core.drive.stale = true;
        self.core.eval_all();
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

    /// The number of native segments and of closure steps in this
    /// kernel, in that order: what the per-node engine choice decided.
    pub fn engine_counts(&self) -> (usize, usize) {
        self.core.engine_counts()
    }

    /// Store owned nodes to keep JIT-baked pointers valid.
    pub fn retain_nodes(&mut self, nodes: Vec<Box<dyn PolydatNode>>) {
        self.core._nodes = std::sync::Arc::new(nodes);
    }
}

/// Type alias for the default hybrid kernel (PushPull — full optimization).
///
/// The assembler's `compile_hybrid` returns this alias. Rename uses to
/// the concrete type if different optimization trade-offs are needed.
pub type HybridKernel = HybridKernelPushPull;

/// Flattened slot list for one node's wire inputs under per-port
/// widths (type_system_alignment.md §6): every source
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
/// A node this engine cannot lay out, as a refusal naming the engine.
/// Distinct from the fold failure a built kernel can still report,
/// which is the program's and not this engine's
/// ([`KernelError::ConstantFold`](crate::KernelError::ConstantFold)).
fn refused(reason: String) -> crate::KernelError {
    crate::KernelError::Refused {
        engine: crate::compile::select::Engine::Native(crate::compile::select::Provenance::Auto),
        reason,
    }
}

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
) -> Result<HybridKernelPushPull, crate::KernelError> {
    let mut steps: Vec<HybridStep> = Vec::new();
    let mut scratch: Vec<crate::ast::ScratchBuf> = Vec::new();
    let mut ref_scratch: Vec<(usize, usize)> = Vec::new();
    let mut max_inputs = 0usize;
    let mut max_outputs = 0usize;
    let graph = GraphView {
        nodes,
        wiring,
        port_offsets,
        input_types,
    };

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
    // A node that would have produced a value from a `None` runs as a
    // closure, always. A segment's answer to a `None` on one of its
    // boundary inputs is `None` on all of its outputs — SRD-74 Rule 1,
    // the only answer native code can give, since it cannot carry one.
    // That answer is right for every node that propagates a `None` and
    // wrong for a node that consumes one (`to_json` keeps going, a
    // `printf` with an `Option` arg writes its own text), so such a
    // node must not be inside a segment for the rule to hold. The cone
    // planner makes the same exclusion for the same reason.
    let mut classifications = classifications;
    let mut eligible = vec![false; nodes.len()];
    for (node_idx, node) in nodes.iter().enumerate() {
        if matches!(classifications[node_idx].0, JitOp::Fallback) {
            continue;
        }
        if !crate::compile::none_rule_admits(
            node.accepts_none_inputs(),
            &wiring[node_idx],
            &eligible,
        ) {
            classifications[node_idx].0 = JitOp::Fallback;
            continue;
        }
        eligible[node_idx] = true;
    }
    // A node downstream of an extern with no value runs as a closure
    // too. This is the narrower case — the extern is already unset at
    // build — and it stays because it also keeps the `None` out of
    // segments downstream, where the boundary guard would otherwise be
    // the only thing catching it.
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
    // The step order: every compile-constant node first, then the rest
    // in the graph's order. A constant depends on constants alone, so
    // hoisting them keeps every dependency ahead of its consumer, and
    // it keeps the cycle-time nodes contiguous: a literal between two
    // cycle-time statements no longer cuts a segment in two (the tile
    // ladder's twenty-hole case ran as dozens of segments that way,
    // each paying the segment's catch and step bookkeeping).
    let order: Vec<usize> = (0..nodes.len())
        .filter(|&k| constant[k])
        .chain((0..nodes.len()).filter(|&k| !constant[k]))
        .collect();
    let mut rank = vec![0usize; nodes.len()];
    for (pos, &k) in order.iter().enumerate() {
        rank[k] = pos;
    }
    // Segments are the fusion units (SRD-105, compile::fusion_units):
    // connected, convex groups of native nodes, so two chains that share
    // nothing are two segments and a pull runs only its own. Nodes fuse
    // within one lifecycle: a segment is folded at build only if every
    // member is compile-constant, and a volatile node never joins pure
    // ones (the segment would be never current and rerun them at every
    // round). A side channel never joins any other node (it would fire
    // whenever the segment ran, rather than when its own inputs changed).
    let is_side = |k: usize| matches!(nodes[k].purity(), crate::ast::Purity::SideChannel { .. });
    let preds: Vec<Vec<usize>> = wiring
        .iter()
        .map(|w| {
            w.iter()
                .filter_map(|src| match src {
                    WireSource::NodeOutput(j, _) => Some(*j),
                    WireSource::Input(_) => None,
                })
                .collect()
        })
        .collect();
    let fusible: Vec<bool> = (0..nodes.len())
        .map(|k| !matches!(classifications[k].0, JitOp::Fallback) && !is_side(k))
        .collect();
    let class: Vec<u64> = (0..nodes.len())
        .map(|k| constant[k] as u64 | (volatile[k] as u64) << 1)
        .collect();
    let plan =
        crate::compile::fusion_units::plan_units(&preds, &fusible, &class, &rank, &|c| c & 1 == 1);
    for members in plan.units {
        let i = members[0];
        if matches!(classifications[i].0, JitOp::Fallback) {
            // This node needs a closure — scalar u64 op preferred,
            // slot op for slice-bearing nodes (type_system_alignment.md
            // §4, compiled_handles.md §3).
            let (_, ref input_slots, ref output_slots) = classifications[i];
            let step = closure_step_for(
                &graph,
                i,
                input_slots.clone(),
                output_slots.clone(),
                &mut scratch,
                &mut ref_scratch,
            )?;
            node_step[i] = steps.len();
            steps.push(HybridStep::Closure(step));
        } else {
            // Each step's scratch entries are placed in the kernel's
            // scratch (axiom S3), and its reference outputs recorded
            // for the validator (S9(a)).
            for &k in &members {
                let base = scratch.len();
                classifications[k].0.place_scratch(base);
                let elems = classifications[k].0.scratch_elems().to_vec();
                ref_scratch.extend(crate::compile::assembly::scratch_pairs(
                    &nodes[k].meta().name,
                    &flatten_ref_output_starts(nodes, k, port_offsets),
                    &elems,
                    base,
                ));
                scratch.extend(elems.iter().map(|e| crate::ast::ScratchBuf::new(*e)));
            }
            // One native segment for the batch: its boundary inputs are
            // the slots the batch reads and does not write, its outputs
            // every slot it writes. Slots closures fill with Ref pairs
            // are Ref2 slots to the S2/S9 validator, which a segment
            // may only load, store, and pass. Native code names the
            // member it is in through the tracker slot, for the failure
            // path (A7).
            let batch: Vec<(JitOp, Vec<usize>, Vec<usize>)> = members
                .iter()
                .map(|&k| classifications[k].clone())
                .collect();
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
            let (code_fn, code) =
                jit::compile_jit_entry(&batch, Some(total_slots)).map_err(refused)?;
            let segment = steps.len();
            for &k in &members {
                node_step[k] = segment;
            }
            steps.push(HybridStep::Jit(JitSegment {
                code_fn,
                fallible: code.fallible(),
                _module: code,
                input_slots,
                output_slots,
                nodes: members,
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
) -> Result<HybridKernelPushPull, crate::KernelError> {
    let mut steps: Vec<HybridStep> = Vec::new();
    let mut scratch: Vec<crate::ast::ScratchBuf> = Vec::new();
    let mut ref_scratch: Vec<(usize, usize)> = Vec::new();
    let mut max_inputs = 0usize;
    let mut max_outputs = 0usize;
    let graph = GraphView {
        nodes,
        wiring,
        port_offsets,
        input_types,
    };

    for node_idx in 0..nodes.len() {
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

        let step = closure_step_for(
            &graph,
            node_idx,
            input_slots,
            output_slots,
            &mut scratch,
            &mut ref_scratch,
        )?;
        steps.push(HybridStep::Closure(step));
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

/// The graph a builder reads a node's shape out of: the nodes, how
/// they are wired, where each port's slots begin, and the coordinate
/// and extern types a wire from an input takes. The four always travel
/// together and neither builder modifies them.
#[derive(Clone, Copy)]
struct GraphView<'a> {
    nodes: &'a [Box<dyn PolydatNode>],
    wiring: &'a [Vec<WireSource>],
    port_offsets: &'a [Vec<usize>],
    input_types: &'a [crate::ast::PortType],
}

/// The closure step for one node: its op, its scratch placed in the
/// kernel's arena, and its reference outputs recorded for the S9(a)
/// validator. A scalar op first, then the compiler's slot copy, then
/// the node's own kit — the same ladder `assembly::node_step_op` walks
/// for the closure tier.
///
/// Both builders reach here for a node that runs as a closure: the one
/// with the JIT for a node it classified `Fallback`, the one without
/// for every node, since without the feature there is nothing else a
/// node can be. They had the block twice, differing in where the slots
/// came from, which is why it is a parameter.
fn closure_step_for(
    graph: &GraphView<'_>,
    node_idx: usize,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
    scratch: &mut Vec<crate::ast::ScratchBuf>,
    ref_scratch: &mut Vec<(usize, usize)>,
) -> Result<ClosureStep, crate::KernelError> {
    let GraphView {
        nodes,
        wiring,
        port_offsets,
        input_types,
    } = *graph;
    let node = &nodes[node_idx];
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
        return Err(refused(format!(
            "node '{}' has no compiled form (docs/design/engines.md §8)",
            node.meta().name
        )));
    };
    Ok(ClosureStep {
        op,
        input_slots,
        output_slots,
        scratch_range: (scratch_start, scratch.len()),
        accepts_none: node.accepts_none_inputs(),
        node: node_idx,
    })
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
) -> Result<HybridKernelPushPull, crate::KernelError> {
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
    // Dependents come back per node; `to_steps` folds them onto steps (a
    // segment depends on what any member depends on). They also come back
    // per-INPUT; expand to per-SLOT so the kernels' slot-indexed dirty
    // tracking / changed-mask bits stay coherent under multi-slot inputs
    // (type_system_alignment.md §6). Identity for all-scalar inputs.
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
            engine: Engine::Native(Provenance::PushPull),
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
    kernel.core.fold_steps(&constants)?;
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
    node.compiled_slot(
        wire_types,
        crate::compile::select::Engine::Native(crate::compile::select::Provenance::Auto),
    )
}

// ── The engine-independent surface (engines.md §3.5) ──────

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
        self.core.clean.fill(false);
        self.changed_mask = crate::kernel::ProvMask::all_below(self.core.coord_count);
        self.force_run = true;
    }

    /// The same program with no provenance: every run evaluates
    /// everything.
    pub(crate) fn into_raw(self) -> HybridKernelRaw {
        let mut core = self.core;
        core.set_use_clean(false);
        core.engine = Engine::Native(Provenance::Raw);
        HybridKernelRaw { core }
    }

    pub(crate) fn into_pull(self) -> HybridKernelPull {
        let mut core = self.core;
        core.set_use_clean(false);
        core.engine = Engine::Native(Provenance::Pull);
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

crate::compile::impl_kernel_trait!(HybridKernelRaw);
crate::compile::impl_kernel_trait!(HybridKernelPull);
crate::compile::impl_kernel_trait!(HybridKernelPushPull);
crate::compile::impl_slot_kernel!(HybridKernelRaw);
crate::compile::impl_slot_kernel!(HybridKernelPull);
crate::compile::impl_slot_kernel!(HybridKernelPushPull);

/// One step: SRD-74 Rule 1, then the segment or the closure. A step
/// that does not accept `None` emits `None` on every output when any
/// input is `None`, without running.
///
/// A segment is such a step and always was — native code cannot carry
/// a `None` — so a `None` on one of its boundary inputs makes all of
/// its outputs `None`, which is the same answer the closure tier and
/// the interpreter give. It reaches a segment only when a host cleared
/// an extern after the build; an extern unset at build already keeps
/// the nodes downstream of it out of segments, and a node that would
/// have *consumed* the `None` rather than propagated it is kept out
/// unconditionally, so this answer is never the wrong one.
///
/// With `none_free` the mask is known clear and is not read.
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
    if !none_free && !step.accepts_none() && step.input_slots().iter().any(|&s| none[s]) {
        for &s in step.output_slots() {
            none[s] = true;
        }
        return;
    }
    match step {
        #[cfg(feature = "jit")]
        HybridStep::Jit(seg) => {
            // Through the setjmp wrapper when the code calls a helper,
            // so its failure is the longjmp the kernel catches rather
            // than an abort; bare when it calls nothing.
            let code_fn = seg.code_fn;
            let buf_const = buffer.as_ptr();
            let buf_mut = buffer.as_mut_ptr();
            let sc = scratch.as_mut_ptr();
            if seg.fallible {
                crate::compile::jit::invoke_with_catch(move || unsafe {
                    (code_fn)(buf_const, buf_mut, sc);
                });
            } else {
                unsafe { (code_fn)(buf_const, buf_mut, sc) };
            }
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
