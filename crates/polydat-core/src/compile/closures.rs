// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The closure tier: every node runs its generated op over a flat u64
//! slot buffer, by-reference outputs as `Ref2` pairs into step-owned
//! scratch (compiled_handles.md).
//!
//! Four kernel types, each produced by a distinct compiler path. They
//! differ in what `set_inputs` marks and whether `eval_for_slot`
//! consults the cone guard; the shared step loop reads the mode's
//! `use_clean` flag per step.
//!
//! | Type | Push (per-node skip) | Pull (cone guard) |
//! |------|---------------------|-------------------|
//! | `CompiledKernelRaw` | — | — |
//! | `CompiledKernelPush` | yes | — |
//! | `CompiledKernelPull` | — | yes |
//! | `CompiledKernelPushPull` | yes | yes |

use std::collections::HashMap;

use crate::ast::{CompiledSlotOp, CompiledU64Op, PortType, ScratchBuf, ScratchElem};

/// What a P2 kernel needs beyond its steps: each named output's port
/// type for the typed reader, the externs, the provenance, and the
/// attribution for the failure path.
#[derive(Default)]
pub(crate) struct P2Extras {
    pub(crate) output_types: HashMap<String, PortType>,
    /// The kernel's extern inputs: seeded at build, host-settable,
    /// written through at every set (`compile::externs`).
    pub(crate) externs: crate::compile::externs::Externs,
    /// Per input slot, coordinates and externs alike, the steps that
    /// depend on it: the provenance the plan is derived from.
    pub(crate) input_dependents: Vec<Vec<usize>>,
    /// Where each step came from, for the failure path (A7).
    pub(crate) attribution: std::sync::Arc<crate::compile::Attribution>,
}

/// A single evaluation step in the compiled kernel.
/// A compiled step's op: pure-scalar u64 closure, or a slot op
/// with kernel-owned scratch for typed-slice ports
/// (type_system_alignment.md §4, compiled_handles.md §3).
pub(crate) enum StepOp {
    U64(CompiledU64Op),
    Slot(CompiledSlotOp),
    /// A slot copy (`identity`, the compiler's `__port_` passthrough),
    /// run inline: no closure call, no gather.
    Copy,
}
/// One compiled step plus its slice of the scratch arena.
pub(crate) struct P2Step {
    /// The node's name, for construction-time diagnostics.
    pub(crate) name: String,
    pub(crate) op: StepOp,
    pub(crate) input_slots: Vec<usize>,
    pub(crate) output_slots: Vec<usize>,
    /// Scratch element declarations (consumed by build_core).
    pub(crate) scratch: Vec<ScratchElem>,
    /// First slot of each Ref2-colored output port, in port
    /// order — zipped with the scratch entries to build the
    /// slot→arena map for axiom S9(a)'s validator and the S2
    /// accessors.
    pub(crate) ref_output_starts: Vec<usize>,
    /// The node handles `None` inputs itself (SRD-74 Rule 2); every
    /// other node emits `None` when any input is `None` (Rule 1).
    pub(crate) accepts_none: bool,
    /// The node is nondeterministic or downstream of one (the runtime
    /// model's per-cycle invalidation set): never current.
    pub(crate) volatile: bool,
    /// No input reaches the node and it is not volatile: the runtime
    /// model's compile-constant lifecycle, folded at build.
    pub(crate) constant: bool,
    /// The node is a side channel: it runs exactly when the interpreter
    /// would run it, in every provenance mode, because its run is
    /// observable.
    pub(crate) side: bool,
}

struct CompiledStep {
    op: StepOp,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
    scratch_range: (usize, usize),
    /// SRD-74 Rule 2: the closure runs on `None` inputs.
    accepts_none: bool,
    /// Never current: nondeterministic or downstream of one.
    volatile: bool,
    /// Compile-constant: folded at build, current from then on.
    constant: bool,
    /// A side channel: skipped when current in every mode, since a
    /// redundant run would be observed.
    side: bool,
}

/// An output resolved for the index-keyed pull: its slot, its type,
/// and the steps of its cone.
type ResolvedOutput = (usize, crate::ast::PortType, Option<std::sync::Arc<[usize]>>);

/// Common fields shared by all kernel variants. A clone is a new state
/// of the same program: the steps are shared, everything else is the
/// clone's own (engines.md §3.5), and every pair in its buffer
/// points into its own storage (axiom S3), never into the state it was
/// cloned from.
struct KernelCore {
    buffer: Vec<u64>,
    coord_count: usize,
    steps: std::sync::Arc<[CompiledStep]>,
    output_map: HashMap<String, usize>,
    gather_buf: Vec<u64>,
    scatter_buf: Vec<u64>,
    /// Kernel-owned vector storage; vector-producing ports'
    /// (ptr, len) slots view entries here (type_system_alignment.md
    /// §4, compiled_handles.md §3).
    scratch: Vec<ScratchBuf>,
    /// Axiom S2: per-slot Ref2 mask — the raw readers panic on
    /// these instead of leaking addresses.
    ref_slots: Vec<bool>,
    /// Axiom S9(a): (first slot of a Ref pair → scratch arena
    /// index) for every scratch-backed Ref output.
    ref_scratch: Vec<(usize, usize)>,
    /// Port type of each named output, for `get_value`.
    output_types: HashMap<String, PortType>,
    /// The extern inputs, written through at every set.
    externs: crate::compile::externs::Externs,
    /// The traversals the program declares (SRD 113), opened through the
    /// `Kernel` trait.
    traversals: std::sync::Arc<[crate::dsl::traversal::Traversal]>,
    /// Per declared output, its slot, type, and cone, resolved on the
    /// first index-keyed pull (SRD 117 step 3).
    resolved_outputs: Vec<Option<ResolvedOutput>>,
    /// The coordinates set through the `Kernel` trait, pending
    /// evaluation; `stale` means a write happened since the last
    /// evaluation round.
    drive: crate::compile::Drive,
    /// Per slot: the slot holds `None` (SRD-74 on a compiled kernel):
    /// an unset extern, or an output of a step that propagated one.
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
    /// Whether this kernel's provenance mode skips current steps
    /// (push-side); a mode without per-step skipping runs every step
    /// in the cone once per round.
    use_clean: bool,
    /// The dirty-register plan: what each input invalidates, what each
    /// output needs.
    plan: std::sync::Arc<crate::compile::Invalidation>,
    /// Per slot: the step that writes it, for the validator.
    slot_step: std::sync::Arc<[Option<usize>]>,
    /// Where each step came from, for the failure path (A7).
    sites: std::sync::Arc<crate::compile::Attribution>,
    /// The step running, for the failure path.
    cur_step: usize,
    /// Every step, in order: what `eval` runs.
    all: std::sync::Arc<[usize]>,
    /// Per input slot, the steps an input change marks not current:
    /// the plan's dependents in a push mode; in a raw or pull-only
    /// mode, which never consult a pure step's currency, only the side
    /// channels among them (an optimization over the plan, not a change
    /// to it).
    dirty: std::sync::Arc<[Vec<usize>]>,
    /// The steps that are never current.
    volatile_steps: std::sync::Arc<[usize]>,
    /// Some slot holds `None`: an unset extern, which is
    /// the only way one enters (SRD-74). When none does, the steps run
    /// without the mask.
    any_none: bool,
}

impl Clone for KernelCore {
    fn clone(&self) -> Self {
        let mut core = KernelCore {
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
            drive: self.drive.clone(),
            none: self.none.clone(),
            ran: self.ran.clone(),
            epoch: self.epoch,
            all_ran: self.all_ran,
            clean: self.clean.clone(),
            use_clean: self.use_clean,
            plan: self.plan.clone(),
            slot_step: self.slot_step.clone(),
            sites: self.sites.clone(),
            cur_step: self.cur_step,
            all: self.all.clone(),
            dirty: self.dirty.clone(),
            volatile_steps: self.volatile_steps.clone(),
            any_none: self.any_none,
        };
        core.republish_refs();
        core
    }
}

impl KernelCore {
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

impl KernelCore {
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

    /// Every dependent of a slot a cell refresh changed runs again,
    /// between writes too, as the interpreter re-evaluates a node
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
    /// to, and mark its dependents, so a pull between writes sees the
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
    /// not current: one rule for every step, whatever reaches it; a
    /// volatile step is never current.
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
            sites.reraise(payload, self.cur_step, &self.buffer, Some(&self.none));
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
            let step = &steps[i];
            // A pure step may be recomputed redundantly in a mode
            // without per-step skipping; a side channel may not.
            if (self.use_clean || step.side) && self.clean[i] && !step.volatile {
                self.ran[i] = self.epoch;
                continue;
            }
            self.cur_step = i;
            if none_free {
                run_step_fast(
                    step,
                    &mut self.buffer,
                    &mut self.gather_buf,
                    &mut self.scatter_buf,
                    &mut self.scratch,
                );
            } else {
                run_step(
                    step,
                    &mut self.buffer,
                    &mut self.none,
                    &mut self.gather_buf,
                    &mut self.scatter_buf,
                    &mut self.scratch,
                );
            }
            self.ran[i] = self.epoch;
            self.clean[i] = !step.volatile;
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
            if step.side {
                if self.clean[i] && !step.volatile {
                    continue;
                }
                self.clean[i] = !step.volatile;
            }
            self.cur_step = i;
            run_step_fast(
                step,
                &mut self.buffer,
                &mut self.gather_buf,
                &mut self.scatter_buf,
                &mut self.scratch,
            );
        }
        self.all_ran = true;
    }

    /// Evaluate every output: begin a round if a write happened, then
    /// run every step that has not run.
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

    /// The named output for the current inputs, running only its cone
    /// (A6): the interpreter's `pull`, on a compiled kernel.
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

    /// Every step is a closure.
    fn plan(&self) -> crate::EnginePlan {
        crate::EnginePlan {
            closure_steps: self.steps.len(),
            ..Default::default()
        }
    }

    /// Nothing is current: every step runs at the next evaluation.
    fn invalidate_all(&mut self) {
        self.clean.fill(false);
        self.all_ran = false;
        self.drive.stale = true;
    }

    /// Set an extern by name; returns its slot. The plan invalidates
    /// what depends on it, as a changed coordinate is invalidated, and
    /// the next evaluation begins a round.
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

    /// Axiom S9(a) — deterministic Ref validation: every
    /// scratch-backed Ref pair in the buffer must equal its
    /// owning entry's current `(as_ptr(), len())`. Run after
    /// every eval in debug/test builds; a violation names the
    /// slot instead of dangling. Gated to `debug_assertions` to
    /// match its call sites, which compile out in release.
    #[cfg(debug_assertions)]
    fn validate_refs(&self) {
        for &(slot, idx) in &self.ref_scratch {
            // A step that has never run has not published; one that
            // propagated `None` left its slots as they were.
            if let Some(Some(step)) = self.slot_step.get(slot)
                && (self.ran[*step] == 0 || self.none[slot])
            {
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

    /// Axiom S2 guard for the raw u64 readers.
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

    /// Axiom S2 typed accessor core: resolve a Ref pair's first
    /// slot to its kernel-owned scratch entry. The returned
    /// borrow ties to `&self`, so holding it across the next
    /// `eval(&mut self)` is a compile error — stale reads are
    /// statically impossible.
    fn ref_entry(&self, slot: usize) -> &ScratchBuf {
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

/// Build kernel core from raw step data. `use_clean` is whether the
/// kernel's provenance mode skips current steps.
fn build_core(
    coord_count: usize,
    total_slots: usize,
    steps: Vec<P2Step>,
    output_map: HashMap<String, usize>,
    ref_slots: Vec<bool>,
    extras: P2Extras,
    use_clean: bool,
) -> KernelCore {
    let P2Extras {
        output_types,
        externs,
        input_dependents,
        attribution,
    } = extras;
    let max_inputs = steps.iter().map(|s| s.input_slots.len()).max().unwrap_or(0);
    let max_outputs = steps
        .iter()
        .map(|s| s.output_slots.len())
        .max()
        .unwrap_or(0);
    let mut scratch: Vec<ScratchBuf> = Vec::new();
    let mut ref_scratch: Vec<(usize, usize)> = Vec::new();
    let compiled_steps: Vec<CompiledStep> = steps
        .into_iter()
        .map(|step| {
            let start = scratch.len();
            scratch.extend(step.scratch.iter().map(|e| ScratchBuf::new(*e)));
            ref_scratch.extend(crate::compile::assembly::scratch_pairs(
                &step.name,
                &step.ref_output_starts,
                &step.scratch,
                start,
            ));
            CompiledStep {
                op: step.op,
                input_slots: step.input_slots,
                output_slots: step.output_slots,
                scratch_range: (start, scratch.len()),
                accepts_none: step.accepts_none,
                volatile: step.volatile,
                constant: step.constant,
                side: step.side,
            }
        })
        .collect();
    let mut slot_step: Vec<Option<usize>> = vec![None; total_slots];
    for (i, step) in compiled_steps.iter().enumerate() {
        for &s in &step.output_slots {
            slot_step[s] = Some(i);
        }
    }
    let step_inputs: Vec<&[usize]> = compiled_steps
        .iter()
        .map(|s| s.input_slots.as_slice())
        .collect();
    let step_outputs: Vec<&[usize]> = compiled_steps
        .iter()
        .map(|s| s.output_slots.as_slice())
        .collect();
    let plan = crate::compile::Invalidation::from_provenance(
        input_dependents,
        &step_inputs,
        &step_outputs,
        &output_map,
        total_slots,
    );
    let dirty: Vec<Vec<usize>> = plan
        .input_dependents
        .iter()
        .map(|deps| {
            if use_clean {
                deps.clone()
            } else {
                deps.iter()
                    .copied()
                    .filter(|&i| compiled_steps[i].side)
                    .collect()
            }
        })
        .collect();
    let volatile_steps: Vec<usize> = (0..compiled_steps.len())
        .filter(|&i| compiled_steps[i].volatile)
        .collect();
    let mut buffer = vec![0u64; total_slots];
    let mut none = vec![false; total_slots];
    let any_none = externs.seed(&mut buffer, Some(&mut none));
    let step_count = compiled_steps.len();
    let constants: Vec<usize> = compiled_steps
        .iter()
        .enumerate()
        .filter(|(_, s)| s.constant)
        .map(|(i, _)| i)
        .collect();
    let mut core = KernelCore {
        buffer,
        coord_count,
        steps: compiled_steps.into(),
        output_map,
        gather_buf: vec![0u64; max_inputs],
        scatter_buf: vec![0u64; max_outputs],
        scratch,
        ref_slots,
        ref_scratch,
        output_types,
        externs,
        traversals: Vec::new().into(),
        resolved_outputs: Vec::new(),
        drive: crate::compile::Drive {
            coords: Vec::new(),
            stale: true,
        },
        none,
        ran: vec![0; step_count],
        epoch: 0,
        all_ran: false,
        clean: vec![false; step_count],
        use_clean,
        plan: std::sync::Arc::new(plan),
        slot_step: slot_step.into(),
        sites: attribution,
        cur_step: 0,
        all: (0..step_count).collect::<Vec<usize>>().into(),
        dirty: dirty.into(),
        volatile_steps: volatile_steps.into(),
        any_none,
    };
    // The compile-constant fold of the runtime model, on this engine: a
    // step no input reaches runs at build, once, and is current from
    // then on, so what is knowable at build is known at build and fails
    // at build.
    core.begin_epoch();
    core.run_steps(&constants);
    core.drive.stale = true;
    core
}

/// The provenance of every slot, from the steps' output slots
/// ([`crate::compile::slot_provenance`]).
fn compute_slot_provenance(
    coord_count: usize,
    total_slots: usize,
    input_dependents: &[Vec<usize>],
    steps: &[CompiledStep],
) -> Vec<crate::kernel::ProvMask> {
    let outs: Vec<&[usize]> = steps.iter().map(|s| s.output_slots.as_slice()).collect();
    crate::compile::slot_provenance(coord_count, total_slots, &outs, input_dependents)
}

// ── Shared accessor methods ────────────────────────────────────

macro_rules! kernel_accessors {
    () => {
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

        /// The named output through the `Kernel` trait: the pending
        /// coordinates are applied, a round begins if a write is pending, and
        /// only the output's cone runs.
        fn pull_value(&mut self, name: &str) -> crate::ast::Value {
            let coords = std::mem::take(&mut self.core.drive.coords);
            self.set_coords(&coords);
            self.core.drive.coords = coords;
            self.pull_output(name)
        }

        /// [`Self::pull_value`] by output index.
        fn pull_value_at(&mut self, index: usize) -> crate::ast::Value {
            let coords = std::mem::take(&mut self.core.drive.coords);
            self.set_coords(&coords);
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

        /// Set an extern by name, as `PolydatState::set_input` does on
        /// the interpreter. The value must be of the declared port
        /// type. Every kind is written through at once, and every step
        /// downstream of the extern reruns.
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

        crate::compile::ref_readers!();
    };
}

// ═══════════════════════════════════════════════════════════════
// Raw: no provenance, no cone guard. Eval runs all steps.
// ═══════════════════════════════════════════════════════════════

#[derive(Clone)]
/// The closure tier with no provenance: every evaluation runs every step.
pub struct CompiledKernelRaw {
    core: KernelCore,
}

impl CompiledKernelRaw {
    pub(crate) fn new(
        coord_count: usize,
        total_slots: usize,
        steps: Vec<P2Step>,
        output_map: HashMap<String, usize>,
        ref_slots: Vec<bool>,
        extras: P2Extras,
    ) -> Self {
        Self {
            core: build_core(
                coord_count,
                total_slots,
                steps,
                output_map,
                ref_slots,
                extras,
                false,
            ),
        }
    }

    /// The plan invalidates what depends on the input; this mode runs
    /// every step of a cone once per round regardless.
    fn mark_input_changed(&mut self, slot: usize) {
        self.core.dirty_input(slot);
    }

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

    /// Evaluate every step for `coords`: a new round.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.set_coords(coords);
        self.core.drive.stale = true;
        self.core.eval_all();
    }

    fn pull_output(&mut self, name: &str) -> crate::ast::Value {
        self.core.pull_named(name)
    }

    /// Eval + return a specific slot. No cone guard — always evaluates.
    #[inline]
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        self.core.guard_ref_slot(slot);
        self.eval(coords);
        self.core.buffer[slot]
    }

    kernel_accessors!();
}

// ═══════════════════════════════════════════════════════════════
// Push: per-step skip, no cone guard. A changed input invalidates
// its dependents through the plan; a current step is skipped.
// ═══════════════════════════════════════════════════════════════

#[derive(Clone)]
/// The closure tier with per-step skipping: a changed input invalidates
/// its dependents through the plan, and a current step is skipped.
pub struct CompiledKernelPush {
    core: KernelCore,
}

impl CompiledKernelPush {
    pub(crate) fn new(
        coord_count: usize,
        total_slots: usize,
        steps: Vec<P2Step>,
        output_map: HashMap<String, usize>,
        input_dependents: Vec<Vec<usize>>,
        ref_slots: Vec<bool>,
        extras: P2Extras,
    ) -> Self {
        // The plan in `extras` carries the dependents.
        let _ = input_dependents;
        Self {
            core: build_core(
                coord_count,
                total_slots,
                steps,
                output_map,
                ref_slots,
                extras,
                true,
            ),
        }
    }

    #[inline]
    fn set_coords(&mut self, coords: &[u64]) {
        for (i, &c) in coords.iter().enumerate().take(self.core.coord_count) {
            if self.core.buffer[i] != c {
                self.core.buffer[i] = c;
                self.core.dirty_input(i);
            }
        }
    }

    /// Every step downstream of the slot reruns.
    fn mark_input_changed(&mut self, slot: usize) {
        self.core.dirty_input(slot);
    }

    /// Evaluate every step that is not current for `coords`: a new round.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.set_coords(coords);
        self.core.drive.stale = true;
        self.core.eval_all();
    }

    fn pull_output(&mut self, name: &str) -> crate::ast::Value {
        self.core.pull_named(name)
    }

    /// Eval + return a specific slot. No cone guard — always enters eval loop.
    #[inline]
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        self.core.guard_ref_slot(slot);
        self.eval(coords);
        self.core.buffer[slot]
    }

    kernel_accessors!();
}

// ═══════════════════════════════════════════════════════════════
// Pull: cone guard only, no per-step skip.
// set_inputs tracks changed_mask. eval_for_slot checks cone
// then runs ALL steps if dirty.
// ═══════════════════════════════════════════════════════════════

#[derive(Clone)]
/// The closure tier with the cone guard: an output whose cone no changed
/// input reaches is not recomputed.
pub struct CompiledKernelPull {
    core: KernelCore,
    slot_provenance: Vec<crate::kernel::ProvMask>,
    changed_mask: crate::kernel::ProvMask,
    /// Set by `set_input`: an extern changed, so the next evaluation
    /// runs whatever the cone guard says.
    force_run: bool,
}

impl CompiledKernelPull {
    pub(crate) fn new(
        coord_count: usize,
        total_slots: usize,
        steps: Vec<P2Step>,
        output_map: HashMap<String, usize>,
        input_dependents: &[Vec<usize>],
        ref_slots: Vec<bool>,
        extras: P2Extras,
    ) -> Self {
        let core = build_core(
            coord_count,
            total_slots,
            steps,
            output_map,
            ref_slots,
            extras,
            false,
        );
        let slot_provenance =
            compute_slot_provenance(coord_count, total_slots, input_dependents, &core.steps);
        Self {
            core,
            slot_provenance,
            changed_mask: crate::kernel::ProvMask::all_below(coord_count), // all dirty initially
            force_run: false,
        }
    }

    /// Track which inputs changed (for the cone guard), and invalidate
    /// their dependents through the plan, as in every mode.
    #[inline]
    fn set_coords(&mut self, coords: &[u64]) {
        self.changed_mask.clear();
        for (i, &c) in coords.iter().enumerate().take(self.core.coord_count) {
            if self.core.buffer[i] != c {
                self.core.buffer[i] = c;
                self.changed_mask.set(i);
                self.core.dirty_input(i);
            }
        }
    }

    /// The next evaluation runs regardless of the cone guard, since
    /// `set_inputs` rebuilds the changed set from the coordinates alone.
    fn mark_input_changed(&mut self, slot: usize) {
        self.core.dirty_input(slot);
        self.force_run = true;
    }

    /// Evaluate eagerly (no cone guard). Runs all steps: a new round.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.set_coords(coords);
        self.force_run = false;
        self.core.drive.stale = true;
        self.core.eval_all();
    }

    fn pull_output(&mut self, name: &str) -> crate::ast::Value {
        self.core.pull_named(name)
    }

    /// Cone guard: if the output's cone is clean, skip eval entirely.
    /// Otherwise run ALL steps (no per-node skip).
    #[inline]
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        self.core.guard_ref_slot(slot);
        self.set_coords(coords);
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

    kernel_accessors!();
}

// ═══════════════════════════════════════════════════════════════
// PushPull: push-side per-step skip + pull-side cone guard.
// Full optimization.
// ═══════════════════════════════════════════════════════════════

#[derive(Clone)]
/// The closure tier with per-step skipping and the cone guard.
pub struct CompiledKernelPushPull {
    core: KernelCore,
    slot_provenance: Vec<crate::kernel::ProvMask>,
    changed_mask: crate::kernel::ProvMask,
    /// Set by `set_input`: an extern changed, so the next evaluation
    /// runs whatever the cone guard says.
    force_run: bool,
}

impl CompiledKernelPushPull {
    pub(crate) fn new(
        coord_count: usize,
        total_slots: usize,
        steps: Vec<P2Step>,
        output_map: HashMap<String, usize>,
        input_dependents: Vec<Vec<usize>>,
        ref_slots: Vec<bool>,
        extras: P2Extras,
    ) -> Self {
        let core = build_core(
            coord_count,
            total_slots,
            steps,
            output_map,
            ref_slots,
            extras,
            true,
        );
        let slot_provenance =
            compute_slot_provenance(coord_count, total_slots, &input_dependents, &core.steps);
        Self {
            core,
            slot_provenance,
            changed_mask: crate::kernel::ProvMask::all_below(coord_count),
            force_run: false,
        }
    }

    #[inline]
    fn set_coords(&mut self, coords: &[u64]) {
        self.changed_mask.clear();
        for (i, &c) in coords.iter().enumerate().take(self.core.coord_count) {
            if self.core.buffer[i] != c {
                self.core.buffer[i] = c;
                self.changed_mask.set(i);
                self.core.dirty_input(i);
            }
        }
    }

    /// Every step downstream of the slot reruns, and the next
    /// evaluation runs whatever the cone guard says.
    fn mark_input_changed(&mut self, slot: usize) {
        self.core.dirty_input(slot);
        self.force_run = true;
    }

    /// Eval with push-side skip (no cone guard): a new round.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.set_coords(coords);
        self.force_run = false;
        self.core.drive.stale = true;
        self.core.eval_all();
    }

    fn pull_output(&mut self, name: &str) -> crate::ast::Value {
        self.core.pull_named(name)
    }

    /// Cone guard + push-side skip: the full optimization.
    #[inline]
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        self.core.guard_ref_slot(slot);
        self.set_coords(coords);
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

    kernel_accessors!();
}

// ── The engine-independent surface (engines.md §3.5) ──────

use crate::compile::select::{Engine, Provenance};

crate::compile::impl_kernel_trait!(CompiledKernelRaw, Engine::Closures(Provenance::Raw));
crate::compile::impl_kernel_trait!(CompiledKernelPush, Engine::Closures(Provenance::Push));
crate::compile::impl_kernel_trait!(CompiledKernelPull, Engine::Closures(Provenance::Pull));
crate::compile::impl_kernel_trait!(
    CompiledKernelPushPull,
    Engine::Closures(Provenance::PushPull)
);

/// One step: SRD-74 Rule 1, then gather, run the closure, scatter. A
/// node that does not accept `None` emits `None` on every output when
/// any input is `None`, without running.
#[inline(always)]
fn run_step(
    step: &CompiledStep,
    buffer: &mut [u64],
    none: &mut [bool],
    gather: &mut [u64],
    scatter: &mut [u64],
    scratch: &mut [ScratchBuf],
) {
    let mut any_none = false;
    for (i, &s) in step.input_slots.iter().enumerate() {
        gather[i] = buffer[s];
        any_none |= none[s];
    }
    if any_none && !step.accepts_none {
        for &s in &step.output_slots {
            none[s] = true;
        }
        return;
    }
    if matches!(step.op, StepOp::Copy) {
        for (&i, &o) in step.input_slots.iter().zip(&step.output_slots) {
            buffer[o] = buffer[i];
            none[o] = false;
        }
        return;
    }
    let (n_in, n_out) = (step.input_slots.len(), step.output_slots.len());
    match &step.op {
        StepOp::Copy => unreachable!(),
        StepOp::U64(op) => op(&gather[..n_in], &mut scatter[..n_out]),
        StepOp::Slot(op) => op(
            &gather[..n_in],
            &mut scatter[..n_out],
            &mut scratch[step.scratch_range.0..step.scratch_range.1],
        ),
    }
    for (i, &s) in step.output_slots.iter().enumerate() {
        buffer[s] = scatter[i];
        none[s] = false;
    }
}

/// [`run_step`] when no slot holds `None`: gather, run, scatter.
#[inline(always)]
fn run_step_fast(
    step: &CompiledStep,
    buffer: &mut [u64],
    gather: &mut [u64],
    scatter: &mut [u64],
    scratch: &mut [ScratchBuf],
) {
    if matches!(step.op, StepOp::Copy) {
        for (&i, &o) in step.input_slots.iter().zip(&step.output_slots) {
            buffer[o] = buffer[i];
        }
        return;
    }
    for (i, &s) in step.input_slots.iter().enumerate() {
        gather[i] = buffer[s];
    }
    let (n_in, n_out) = (step.input_slots.len(), step.output_slots.len());
    match &step.op {
        StepOp::Copy => unreachable!(),
        StepOp::U64(op) => op(&gather[..n_in], &mut scatter[..n_out]),
        StepOp::Slot(op) => op(
            &gather[..n_in],
            &mut scatter[..n_out],
            &mut scratch[step.scratch_range.0..step.scratch_range.1],
        ),
    }
    for (i, &s) in step.output_slots.iter().enumerate() {
        buffer[s] = scatter[i];
    }
}
