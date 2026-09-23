// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! JIT kernel types: structs and impls for all four kernel variants.
//!
//! `JitCore` holds the shared buffer, slot map, and module handle.
//! The two kernel structs (`JitKernelRaw` and `JitKernelPushPull`)
//! wrap a `JitCore` and a
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
pub struct JitCode(std::sync::Arc<FinalizedModule>);

/// A JIT module after finalization, which nothing writes again, the
/// kits its code calls, and whether the code calls anything at all.
struct FinalizedModule {
    #[allow(dead_code)]
    module: JITModule,
    #[allow(dead_code)]
    kits: Vec<super::codegen::SlotKitRef>,
    fallible: bool,
}

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
    pub(crate) fn new(
        module: JITModule,
        kits: Vec<super::codegen::SlotKitRef>,
        fallible: bool,
    ) -> Self {
        JitCode(std::sync::Arc::new(FinalizedModule {
            module,
            kits,
            fallible,
        }))
    }

    /// Whether the code can fail: it calls a helper, and a helper can
    /// raise a node's failure through the longjmp catch. Code with no
    /// call is arithmetic over the buffer, which cannot fail, so the
    /// site that runs it needs no catch around it (the jump buffer, the
    /// panic capture, and the unwind guard are the fixed cost of an
    /// evaluation on the pure tier).
    pub(crate) fn fallible(&self) -> bool {
        self.0.fallible
    }
}

/// Shared fields for all JIT kernel variants. A clone is a new state
/// of the same program: the code and the nodes are shared, everything
/// else is the clone's own (engines.md §3.5), and every extern
/// pair in its buffer points into its own storage (axiom S3), never
/// into the state it was cloned from.
pub(super) struct JitCore {
    /// The engine this kernel runs, as it reports it: the tier and
    /// the provenance mode it was built with. State rather than a
    /// property of the type, so one kernel type can serve a tier
    /// that runs native code and one that runs none.
    pub(super) engine: crate::compile::select::Engine,
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
    /// Whether the code calls a helper, and so runs under the catch.
    pub(super) fallible: bool,
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
    /// nondeterministic node or one downstream of it. Every write
    /// clears their clean flags, so the next pull whose cone holds one
    /// runs it again.
    pub(super) volatile_steps: Vec<usize>,
    /// The program's steps as slots, for the cone a pull runs.
    pub(super) cones: ConePlan,
    /// Each output by index, as the host names it: its slot and type,
    /// filled on the first pull by index so a pull does not look an
    /// output up by name.
    pub(super) outputs_at: Vec<(usize, crate::ast::PortType)>,
}

/// What a pull on this tier runs: the steps its output depends on,
/// found from the steps' slots, and nothing else (engines.md §3.1). The
/// native code takes a want flag per step and skips every step the
/// caller did not ask for; a full evaluation asks for all of them.
#[derive(Clone, Default)]
pub(super) struct ConePlan {
    /// Each step's input slots.
    inputs: std::sync::Arc<[Box<[usize]>]>,
    /// The step that writes each slot, or `usize::MAX` for a slot no
    /// step writes (an input or an extern).
    producer: std::sync::Arc<[usize]>,
    /// A want flag for every step, for a full evaluation.
    all: std::sync::Arc<[u8]>,
    /// The cones found so far, by the output slot they end in.
    found: HashMap<usize, Box<[u8]>>,
}

impl ConePlan {
    pub(super) fn new(
        steps: &[(super::codegen::JitOp, Vec<usize>, Vec<usize>)],
        slots: usize,
    ) -> Self {
        let mut producer = vec![usize::MAX; slots + 1];
        for (i, (_, _, outs)) in steps.iter().enumerate() {
            for &s in outs {
                if s < producer.len() {
                    producer[s] = i;
                }
            }
        }
        ConePlan {
            inputs: steps
                .iter()
                .map(|(_, ins, _)| ins.clone().into_boxed_slice())
                .collect(),
            producer: producer.into(),
            all: vec![1u8; steps.len()].into(),
            found: HashMap::new(),
        }
    }

    /// The want flags of every step.
    pub(super) fn all(&self) -> *const u8 {
        self.all.as_ptr()
    }

    /// The want flags of the steps `slot` depends on, found once. The
    /// pointer is to a boxed slice this plan keeps, so it stays valid
    /// while other cones are added.
    pub(super) fn of(&mut self, slot: usize) -> *const u8 {
        if !self.found.contains_key(&slot) {
            let mut want = vec![0u8; self.inputs.len()];
            let producer_of = |s: usize| self.producer.get(s).copied().filter(|&p| p != usize::MAX);
            let mut stack: Vec<usize> = producer_of(slot).into_iter().collect();
            while let Some(step) = stack.pop() {
                if want[step] != 0 {
                    continue;
                }
                want[step] = 1;
                stack.extend(self.inputs[step].iter().filter_map(|&s| producer_of(s)));
            }
            self.found.insert(slot, want.into_boxed_slice());
        }
        self.found[&slot].as_ptr()
    }
}

impl Clone for JitCore {
    fn clone(&self) -> Self {
        let mut core = JitCore {
            engine: self.engine,
            buffer: self.buffer.clone(),
            coord_count: self.coord_count,
            output_map: self.output_map.clone(),
            guard_slots: self.guard_slots.clone(),
            output_types: self.output_types.clone(),
            externs: self.externs.clone(),
            traversals: self.traversals.clone(),
            _module: self._module.clone(),
            fallible: self.fallible,
            _nodes: self._nodes.clone(),
            drive: self.drive.clone(),
            sites: self.sites.clone(),
            tracker: self.tracker,
            scratch: self.scratch.clone(),
            ref_scratch: self.ref_scratch.clone(),
            volatile_steps: self.volatile_steps.clone(),
            cones: self.cones.clone(),
            outputs_at: self.outputs_at.clone(),
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
    /// Nothing to mark here: a cell another holder published to is
    /// handled where the pending write is applied, where the kernels
    /// mark every step dirty, since the dependents lists do not name a
    /// cell's readers.
    fn dirty_input(&mut self, _slot: usize) {}

    /// The pure tier broadcasts nothing. It is the differential oracle
    /// behind the hybrid and Tier-1's carrier, not a surface a host
    /// composes under (engines.md §1, §8), so no descendant binds to
    /// one of its outputs and it makes no cell to bind to.
    fn output_cell_for(&self, _name: &str) -> Option<crate::kernel::SharedCell> {
        None
    }

    /// Axiom S2 typed accessor core (borrow ties to `&self`), as the
    /// closure tier and the hybrid have it. The pure tier owns the
    /// same scratch and the same `(slot → entry)` map, so the typed
    /// borrows read the same way here; `guard_slots` is this core's
    /// name for the per-slot `Ref2` mask.
    fn ref_entry(&self, slot: usize) -> &crate::ast::ScratchBuf {
        match self.ref_scratch.iter().find(|(s, _)| *s == slot) {
            Some(&(_, idx)) => &self.scratch[idx],
            None if self.guard_slots.get(slot).copied().unwrap_or(false) => panic!(
                "slot {slot} is a Ref pair owned by the CALLER (a kernel \
                 input) — read it on the caller side"
            ),
            None => panic!("slot {slot} is not a Ref2-colored slot"),
        }
    }

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

    /// The next pull or evaluation applies the pending write first.
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
        let mut core = Self {
            // The pure tier, not `Native`: this core belongs to a
            // kernel that refused every node without a native lowering
            // rather than running its closure, and `engine()` reports
            // what ran. The raw builder overwrites the mode.
            engine: crate::compile::select::Engine::PureNative(
                crate::compile::select::Provenance::PushPull,
            ),
            buffer: vec![0u64; total_slots + 1],
            coord_count,
            output_map,
            guard_slots: Vec::new(),
            output_types: HashMap::new(),
            externs: crate::compile::externs::Externs::default(),
            traversals: Vec::new().into(),
            fallible: code.fallible(),
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
            cones: ConePlan::default(),
            outputs_at: Vec::new(),
        };
        // Every pair names its own entry from the start (axiom S3), as
        // a clone's do. A pull runs only its cone, so a step outside
        // every cone pulled so far has not run, and its pair must still
        // name its entry, empty until the step writes it (S9(a)).
        for &(slot, idx) in &core.ref_scratch {
            let (p, l) = core.scratch[idx].ptr_len();
            core.buffer[slot] = p;
            core.buffer[slot + 1] = l;
        }
        core
    }

    /// The output at `index` in the host's order: its slot and type.
    fn output_at(&mut self, index: usize) -> (usize, crate::ast::PortType) {
        if self.outputs_at.is_empty() {
            self.outputs_at = self
                .externs
                .output_names()
                .iter()
                .map(|n| {
                    let slot = self.output_map[n];
                    let ty = self
                        .output_types
                        .get(n)
                        .copied()
                        .unwrap_or(crate::ast::PortType::U64);
                    (slot, ty)
                })
                .collect();
        }
        *self
            .outputs_at
            .get(index)
            .unwrap_or_else(|| panic!("no output at index {index}"))
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
    fn set_extern(
        &mut self,
        name: &str,
        value: crate::ast::Value,
    ) -> Result<usize, crate::kernel::WriteError> {
        Ok(self.externs.set(name, value, &mut self.buffer)?.0)
    }

    /// [`Self::set_extern`] by input index.
    fn set_extern_at(
        &mut self,
        index: usize,
        value: crate::ast::Value,
    ) -> Result<usize, crate::kernel::WriteError> {
        Ok(self.externs.set_at(index, value, &mut self.buffer)?.0)
    }

    /// Bind a `shared` binding to `cell` (engine parity, step 9). The
    /// next pull or evaluation reads it.
    fn attach_cell(&mut self, name: &str, cell: crate::kernel::SharedCell) -> Result<(), String> {
        self.externs.attach_cell(name, cell)?;
        self.drive.stale = true;
        Ok(())
    }

    /// Run one native evaluation: take what cells other holders
    /// published, refuse an unset extern (native code cannot carry a
    /// `None`; engines.md §3.3), run inside the longjmp catch.
    #[inline]
    pub(super) fn run(&mut self, native: impl FnOnce()) {
        if self.externs.cells_dirty() {
            self.externs.refresh_cells(&mut self.buffer);
        }
        if let Some((name, ty)) = self.externs.first_unset() {
            panic!(
                "extern '{name}' ({ty}) has no value on the pure native tier, which \
                 cannot carry a `None`: every step is native code and there is no \
                 closure to propagate one through. Either it was declared without a \
                 default and never set, or a host cleared it after the build. Set it \
                 with set_input before pulling, or run this program on `native`, which \
                 answers a cleared extern with `None` as the interpreter does \
                 (docs/design/engines.md §3.3)"
            );
        }
        // Code that calls no helper cannot fail: it runs bare. Otherwise
        // native code names the step it is in before each helper call;
        // a failure before any names none. The capture guard is armed
        // for the run, so the helper's panic is recorded quietly and
        // re-raised enriched, as the interpreter re-raises a node's (A7).
        if !self.fallible {
            native();
        } else {
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
        }
        #[cfg(debug_assertions)]
        self.validate_refs();
    }

    /// The compile-constant fold of the runtime model on this tier: a
    /// step no input reaches runs at build, once, and is current from
    /// then on, so what is knowable at build is known at build and
    /// fails at build.
    ///
    /// The other two compiled tiers keep a step list and run the
    /// constant steps out of it. This tier has one compiled function
    /// and no list, so the constant steps are compiled a second time
    /// into an entry of their own, run once over this core's buffer and
    /// scratch, and dropped with the code that held them. The steps
    /// carry absolute slot indices, so the entry writes the same slots
    /// the whole-program function would have.
    ///
    /// Externs are not consulted: a compile-constant step is one no
    /// input reaches, extern inputs included, so a program whose
    /// externs are still unset folds its constants anyway. That is the
    /// difference from [`Self::run`], which refuses an unset extern
    /// because a real evaluation reads them.
    pub(super) fn fold_constants(
        &mut self,
        folded: &[(super::codegen::JitOp, Vec<usize>, Vec<usize>)],
        origin: &[usize],
        total_slots: usize,
    ) -> Result<(), crate::KernelError> {
        if folded.is_empty() {
            return Ok(());
        }
        // Graph order is topological and a constant depends on
        // constants alone, so the filtered order is a valid order.
        let (code_fn, code) = super::codegen::compile_jit_entry(folded, Some(total_slots))
            .map_err(|reason| crate::KernelError::ConstantFold { reason })?;
        let buf_ptr_const = self.buffer.as_ptr();
        let buf_ptr_mut = self.buffer.as_mut_ptr();
        let sc = self.scratch.as_mut_ptr();
        let native = move || unsafe {
            (code_fn)(buf_ptr_const, buf_ptr_mut, sc);
        };
        if !code.fallible() {
            native();
        } else {
            self.buffer[self.tracker] = u64::MAX;
            let capture = crate::kernel::engines::EvalPanicCaptureGuard::arm();
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                super::codegen::invoke_with_catch(native)
            }));
            drop(capture);
            if let Err(payload) = outcome {
                // The entry counts its own steps, so the tracker holds
                // an index into `folded`; the attribution is keyed by
                // the program's step, which `origin` gives back.
                let step = self.buffer[self.tracker] as usize;
                let step = origin.get(step).copied().unwrap_or(step);
                let sites = std::sync::Arc::clone(&self.sites);
                return Err(crate::KernelError::ConstantFold {
                    reason: sites.describe(payload, step, &self.buffer, None),
                });
            }
        }
        // `code` owns the executable memory the call ran in, so it is
        // kept alive to here and dropped after, not before.
        drop(code);
        Ok(())
    }
}

macro_rules! jit_accessors {
    () => {
        crate::compile::ref_readers!();

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
        /// index. Refuses a `Ref2` slot (axiom S2): read those
        /// through [`Self::get_value`].
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

        /// Run this program's compile-constant steps once, at build; see
        /// [`JitCore::fold_constants`]. Called by the assembler after
        /// the attribution is in place, so a constant that fails names
        /// its node.
        pub(crate) fn fold_constants(
            &mut self,
            folded: &[(super::codegen::JitOp, Vec<usize>, Vec<usize>)],
            origin: &[usize],
            total_slots: usize,
        ) -> Result<(), crate::KernelError> {
            self.core.fold_constants(folded, origin, total_slots)
        }

        /// Set an extern by name, as `PolydatState::set_input` does on
        /// the interpreter. The value must be of the declared port
        /// type. The value is written through into the buffer at once,
        /// whatever its color, and every step downstream of the extern
        /// reruns at the next evaluation.
        pub fn set_input(
            &mut self,
            name: &str,
            value: crate::ast::Value,
        ) -> Result<(), crate::kernel::WriteError> {
            let slot = self.core.set_extern(name, value)?;
            self.mark_input_changed(slot);
            Ok(())
        }

        /// [`Self::set_input`] by input index.
        pub fn set_input_at(
            &mut self,
            index: usize,
            value: crate::ast::Value,
        ) -> Result<(), crate::kernel::WriteError> {
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

        /// The named output through the `Kernel` trait: the pending
        /// writes are applied and the output's cone runs, and nothing
        /// else (engines.md §3.1).
        fn pull_value(&mut self, name: &str) -> crate::ast::Value {
            let slot = self.core.output_map[name];
            let ty = self
                .core
                .output_types
                .get(name)
                .copied()
                .unwrap_or(crate::ast::PortType::U64);
            self.pull_slot(slot, ty)
        }

        /// [`Self::pull_value`] by output index, through the index's
        /// slot and type rather than its name.
        fn pull_value_at(&mut self, index: usize) -> crate::ast::Value {
            let (slot, ty) = self.core.output_at(index);
            self.pull_slot(slot, ty)
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
        ) -> Result<(), crate::kernel::WriteError> {
            for (slot, value) in self.core.externs.cursor_writes(name, partition)? {
                self.set_input(&slot, value)?;
            }
            Ok(())
        }
    };
}

// ── JitKernelRaw ───────────────────────────────────────────

/// Raw JIT kernel: no provenance. Every write begins a round in which
/// every step is dirty; `eval` runs them all, and a pull runs its
/// output's cone, each step at most once in the round.
#[derive(Clone)]
#[doc(hidden)]
pub struct JitKernelRaw {
    pub(super) core: JitCore,
    pub(super) code_fn: super::codegen::NativeProvFn,
    /// Which steps have run in the round.
    pub(super) node_clean: Vec<u8>,
}

impl JitKernelRaw {
    /// Run the steps `want` names that have not run in the round.
    #[inline]
    fn run(&mut self, want: *const u8) {
        let code_fn = self.code_fn;
        let buf_const = self.core.buffer.as_ptr();
        let buf_mut = self.core.buffer.as_mut_ptr();
        let sc = self.core.scratch.as_mut_ptr();
        let clean = self.node_clean.as_mut_ptr();
        self.core.run(move || unsafe {
            (code_fn)(buf_const, buf_mut, sc, clean, want);
        });
    }

    /// A pull through the `Kernel` trait: a pending write begins a
    /// round, then the output's cone runs.
    fn pull_slot(&mut self, slot: usize, ty: crate::ast::PortType) -> crate::ast::Value {
        if self.core.drive.stale || self.core.externs.cells_dirty() {
            let coords = std::mem::take(&mut self.core.drive.coords);
            self.write_coords(&coords);
            self.core.drive.coords = coords;
            self.core.drive.stale = false;
            self.node_clean.fill(0);
        }
        let want = self.core.cones.of(slot);
        self.run(want);
        self.core.slot_value(slot, ty)
    }

    /// The coordinates, written into the buffer.
    #[inline]
    fn write_coords(&mut self, coords: &[u64]) {
        // Written one by one, as the other kernels write them: a slice
        // copy of a runtime length is a call to memcpy, which costs
        // more than the three stores it replaces.
        for (i, &c) in coords.iter().enumerate().take(self.core.coord_count) {
            if self.core.buffer[i] != c {
                self.core.buffer[i] = c;
            }
        }
    }
    /// Evaluate the kernel with the given coordinate values.
    ///
    /// Predicate violations (`is_positive`, `in_range`,
    /// `is_one_of`) from JIT-lowered code surface as normal
    /// Rust panics carrying the violation message. The
    /// longjmp wrapper in `super::codegen::invoke_with_catch`
    /// handles the transition back to Rust land when the code
    /// calls a helper; code that calls none cannot fail and
    /// runs bare.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.write_coords(coords);
        self.node_clean.fill(0);
        let all = self.core.cones.all();
        self.run(all);
    }

    /// Evaluate and return the value at the given buffer slot index.
    #[inline]
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        self.eval(coords);
        self.core.buffer[slot]
    }

    /// A write begins a round: every step is dirty again.
    fn mark_input_changed(&mut self, _slot: usize) {
        self.node_clean.fill(0);
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

    /// Run the steps `want` names that are dirty.
    #[inline]
    fn run(&mut self, want: *const u8) {
        let code_fn = self.code_fn_prov;
        let buf_const = self.core.buffer.as_ptr();
        let buf_mut = self.core.buffer.as_mut_ptr();
        let sc = self.core.scratch.as_mut_ptr();
        let clean_mut = self.node_clean.as_mut_ptr();
        self.core.run(move || unsafe {
            (code_fn)(buf_const, buf_mut, sc, clean_mut, want);
        });
    }

    /// A pull through the `Kernel` trait: the pending coordinates dirty
    /// their dependents, then the output's cone runs its dirty steps.
    fn pull_slot(&mut self, slot: usize, ty: crate::ast::PortType) -> crate::ast::Value {
        if self.core.drive.stale {
            let coords = std::mem::take(&mut self.core.drive.coords);
            self.set_inputs(&coords);
            self.core.drive.coords = coords;
            self.core.drive.stale = false;
        }
        // A cell another holder published to is a changed input whose
        // readers the dependents lists do not name: every step reruns.
        if self.core.externs.cells_dirty() {
            self.node_clean.fill(0);
        }
        let want = self.core.cones.of(slot);
        self.run(want);
        self.core.slot_value(slot, ty)
    }

    /// Evaluate the kernel with the given coordinate values.
    #[inline]
    pub fn eval(&mut self, coords: &[u64]) {
        self.set_inputs(coords);
        self.force_run = false;
        let all = self.core.cones.all();
        self.run(all);
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
        let want = self.core.cones.of(slot);
        self.run(want);
        self.core.buffer[slot]
    }

    jit_accessors!();
}

// ── The engine-independent surface (engines.md §3.5) ──────

crate::compile::impl_kernel_trait!(JitKernelRaw);
crate::compile::impl_kernel_trait!(JitKernelPushPull);
crate::compile::impl_slot_kernel!(JitKernelRaw);
crate::compile::impl_slot_kernel!(JitKernelPushPull);
