// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Cone-level JIT inside the interpreter kernel (engines.md §2, §8).
//!
//! At assembly time, maximal cones of JIT-eligible nodes with
//! scalar boundaries collapse into one synthetic `JitConeNode`
//! each, compiled to native code by the P3 codegen. The
//! cone node is an ordinary `PolydatNode`: the walker, scope
//! chains, shared cells, None propagation, node_clean caching, and
//! the enrich-and-re-raise panic contract all see a plain node.
//!
//! Boundary marshalling covers every one-slot immediate and every
//! `Ref2` kind, borrowed into its pair for the call and copied out
//! after it; interior fusion follows whatever the P3 classifier
//! accepts. Extraction is recoverable:
//! member nodes move into the cone only after codegen succeeds, so
//! a cone whose code generation fails leaves its members exactly as
//! the interpreter would have compiled them. Under `JitMode::Auto` the
//! failure is recorded on the tree's `CompileLedger` and the compile
//! goes on; under `JitMode::Force` it fails the compile (engines.md
//! §2.1).
//!
//! A component that is not convex is first split into the convex pieces
//! native code and pure native code form from it
//! (`fusion_units::convex_pieces`, engines.md §8). A cone reads at most
//! [`MAX_CONE_INPUTS`] distinct boundary inputs, and a convex piece that
//! reads more is cut into pieces within the bound, each compiled on its
//! own (engines.md §2.2).

/// The most distinct boundary inputs one cone piece reads: an
/// implementation bound on each piece, not on the component it is cut
/// from. A component over it is cut into pieces within it, and a single
/// node over it stays on the interpreter and is recorded on the ledger
/// (engines.md §2.2).
pub const MAX_CONE_INPUTS: usize = 64;

/// How much of the interpreter's graph is fused into native cones: the
/// interpreter engine's one knob, carried by
/// [`Engine::Interpreter`](crate::Engine::Interpreter) and settable per
/// assembler with `set_jit_mode`. It is a property of the kernel being
/// built, never of the process: two hosts in one process compiling
/// under different modes get the kernels they each asked for.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum JitMode {
    /// Pure interpreter, no native code: the differential baseline.
    Off,
    /// Cone extraction with the cost model (fused cones of >= 2 nodes):
    /// what a host gets when it names none.
    #[default]
    Auto,
    /// Every eligible node joins a cone (threshold 1). Used by the
    /// differential battery and for isolating marshalling regressions.
    Force,
}

#[cfg(not(feature = "jit"))]
pub(crate) fn extract_jit_cones(
    _dag: &mut super::assembly::ResolvedDag,
    _mode: JitMode,
) -> Result<(), super::assembly::AssemblyError> {
    Ok(())
}

#[cfg(feature = "jit")]
pub(crate) use jit_impl::extract_jit_cones;

#[cfg(feature = "jit")]
mod jit_impl {
    use super::{JitMode, MAX_CONE_INPUTS};
    use crate::ast::{NodeMeta, PolydatNode, Port, PortType, Purity, Slot, SlotShape, Value};
    use crate::compile::assembly::{AssemblyError, PolydatAssembler, ResolvedDag};
    use crate::compile::jit::{JitOp, classify_node_typed};
    use crate::kernel::{ConeFallback, ConeFallbackKind};
    use crate::kernel::{InputDef, InputKind, WireSource};
    use std::collections::HashMap;

    /// A fused subgraph compiled to native code, standing in the
    /// program as one ordinary node (engines.md §2). The node is shared by
    /// every state of the program; the slot buffer its native code
    /// runs over, and the scratch entries its members' kits write
    /// into, belong to the state that evaluates it, which hands them
    /// in through [`PolydatNode::eval_in`] (axiom S3).
    pub(crate) struct JitConeNode {
        meta: NodeMeta,
        code_fn: crate::compile::jit::NativeFn,
        total_slots: usize,
        /// The members' scratch entries, after the slot buffer in the
        /// cone's scratch layout, with the validator's pairs.
        scratch: crate::compile::jit::ScratchPlan,
        /// Where each member lives, for the failure path (engines.md §3.4): the
        /// member that failed is named as the program names it, with
        /// its outputs under the program's names; the cone is no frame.
        attribution: std::sync::Arc<crate::compile::Attribution>,
        /// First buffer slot per boundary input, in port order.
        in_slots: Vec<usize>,
        /// Buffer slot per output port, in `meta.outs` order.
        out_slots: Vec<usize>,
        in_types: Vec<PortType>,
        out_types: Vec<PortType>,
        /// The original member nodes — kept alive for the LUT /
        /// constant memory the native code references, and walked
        /// by identity hashing (`fusion_subgraph`).
        members: Vec<Box<dyn PolydatNode>>,
        /// Local member wiring (`Input(i)` = this node's i-th
        /// outer input; `NodeOutput(j, p)` = member j) — the
        /// stored subgraph identity hashing recurses through.
        sub_wiring: Vec<Vec<WireSource>>,
        /// Per output port: (local member index, member port).
        out_ports: Vec<(usize, usize)>,
        /// The finalized code and the kits it calls, kept alive for
        /// the life of the program.
        _module: crate::compile::jit::JitCode,
        /// Whether the code calls a helper, and so runs under the
        /// catch; code with no call runs bare.
        fallible: bool,
    }

    impl PolydatNode for JitConeNode {
        fn meta(&self) -> &NodeMeta {
            &self.meta
        }

        fn fusion_subgraph(&self) -> Option<crate::ast::FusionSubgraph<'_>> {
            Some(crate::ast::FusionSubgraph {
                members: &self.members,
                wiring: &self.sub_wiring,
                out_ports: &self.out_ports,
            })
        }

        /// The state owns the cone's slot buffer and its members'
        /// scratch entries (axiom S3): one `Slots` entry, then the
        /// entries the members' kits declared, handed in at every
        /// evaluation.
        fn scratch_layout(&self) -> Vec<crate::ast::ScratchElem> {
            let mut layout = vec![crate::ast::ScratchElem::Slots];
            layout.extend(self.scratch.elems.iter().copied());
            layout
        }

        fn eval_in(
            &self,
            scratch: &mut [crate::ast::ScratchBuf],
            inputs: &[Value],
            outputs: &mut [Value],
        ) {
            let (slots, members) = scratch.split_at_mut(1);
            let crate::ast::ScratchBuf::Slots(buf) = &mut slots[0] else {
                unreachable!("a cone's scratch is its slot buffer");
            };
            self.eval_with(buf, members, inputs, outputs)
        }

        /// An evaluation without a state's scratch (a node evaluated
        /// on its own): a buffer and entries of the call's own.
        fn eval(&self, inputs: &[Value], outputs: &mut [Value]) {
            let mut buf = Vec::new();
            let mut members: Vec<crate::ast::ScratchBuf> = self
                .scratch
                .elems
                .iter()
                .map(|e| crate::ast::ScratchBuf::new(*e))
                .collect();
            self.eval_with(&mut buf, &mut members, inputs, outputs)
        }
    }

    impl JitConeNode {
        /// Evaluate over `buf` and the members' scratch: the boundary
        /// inputs are borrowed into their slots for the duration of the
        /// call, the native code runs, and every output is copied out
        /// as an owned `Value` (the interpreter never holds a reference
        /// into a buffer).
        fn eval_with(
            &self,
            buf: &mut Vec<u64>,
            members: &mut [crate::ast::ScratchBuf],
            inputs: &[Value],
            outputs: &mut [Value],
        ) {
            buf.clear();
            buf.resize(self.total_slots + 1, 0);
            for (i, v) in inputs.iter().enumerate() {
                let start = self.in_slots[i];
                if crate::compile::marshal::encode_slots(v, self.in_types[i], &mut buf[start..])
                    .is_none()
                {
                    panic!(
                        "cone `{}` boundary input [{i}] expected {:?}, got {:?}",
                        self.meta.name,
                        self.in_types[i],
                        v.port_type()
                    );
                }
            }
            // Native code names the member it is in before each helper
            // call (the slot past the layout); a failure is re-raised
            // attributed to that member with the program's context and
            // output names, and the interpreter re-raises it as is
            // (engines.md §3.4).
            let code_fn = self.code_fn;
            let cp = buf.as_ptr();
            let mp = buf.as_mut_ptr();
            let sc = members.as_mut_ptr();
            if !self.fallible {
                // Code that calls no helper cannot fail: it runs bare.
                unsafe { (code_fn)(cp, mp, sc) };
            } else {
                buf[self.total_slots] = u64::MAX;
                let capture = crate::kernel::engines::EvalPanicCaptureGuard::arm();
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::compile::jit::invoke_with_catch(move || unsafe {
                        (code_fn)(cp, mp, sc);
                    })
                }));
                drop(capture);
                if let Err(payload) = outcome {
                    let step = buf[self.total_slots] as usize;
                    self.attribution.reraise(payload, step, buf, None);
                }
            }
            #[cfg(debug_assertions)]
            for &(slot, idx) in &self.scratch.refs {
                let (p, l) = members[idx].ptr_len();
                assert!(
                    buf[slot] == p && buf[slot + 1] == l,
                    "S9 ref-validator: cone `{}` slot pair ({slot}, {}) does not name \
                     scratch[{idx}]",
                    self.meta.name,
                    slot + 1
                );
            }
            for (k, slot) in self.out_slots.iter().enumerate() {
                outputs[k] = crate::compile::marshal::decode_output(buf, *slot, self.out_types[k]);
            }
        }
    }

    /// A planned-but-rejected cone is diagnosable state, never
    /// silent (audit channel, Debug level — rejections are normal
    /// cost-model outcomes, not user-facing failures).
    fn audit_skip(member_count: usize, reason: &str) {
        crate::library::support::audit::debug(&format!(
            "jit cone: leaving a {member_count}-member component on the interpreter: {reason}"
        ));
    }

    /// Marshalable boundary types: every one-slot immediate, encoded
    /// as the bits its `Wire` impl injects (a signed narrow carrier
    /// sign-extended, an unsigned or float one as its bits;
    /// type_system_alignment.md §2), and every `Ref2` kind, borrowed
    /// into its pair for the call and copied out after it
    /// (compiled_handles.md §4). The 128-bit immediates stay out until
    /// they have a boundary encoding of their own.
    fn scalar_ok(ty: PortType) -> bool {
        use crate::ast::SlotColor;
        match ty.slot_color() {
            SlotColor::Imm1 | SlotColor::Ref2 => true,
            SlotColor::Imm2 => false,
        }
    }

    /// A node may join a cone iff the P3 classifier can lower it with
    /// its wire types known, it is pure, and every wire port is a
    /// single-slot value this push can marshal. The None rule
    /// (engines.md §3.3)
    /// is applied by the caller, which knows where each input comes
    /// from.
    fn node_eligible(node: &dyn PolydatNode, wire_types: &[PortType]) -> bool {
        matches!(node.purity(), Purity::Pure)
            && !matches!(classify_node_typed(node, wire_types), JitOp::Fallback)
            && node.meta().outs.iter().all(|p| scalar_ok(p.typ))
            && wire_types.iter().all(|t| scalar_ok(*t))
            && node.meta().wire_inputs().iter().all(|p| scalar_ok(p.typ))
    }

    /// The hoisting classes (compile-constant, scope-init, dynamic;
    /// graph_compiler.md §3), read from the one
    /// classifier the program carries, so that extraction can
    /// restrict fusion to per-cycle work. Const and scope-init
    /// subgraphs belong to the fold passes (which evaluate them
    /// once); fusing them would demote them to per-pull native
    /// evaluation and — for multi-output cones — block
    /// `fold_init_constants`' single-output replacement, breaking
    /// `get_constant` consumers like `eval_const_expr`.
    ///
    /// Reading the program's classifier, rather than walking the graph
    /// again, keeps the `volatile` output modifier and volatility's
    /// downstream propagation in the answer, so a node a program
    /// declares volatile never reads here as const and is never fused
    /// into a cone the fold evaluates once.
    ///
    /// Returns each node's lifecycle and whether it is volatile.
    fn classify_lifecycles(dag: &ResolvedDag) -> (Vec<crate::kernel::EvalLifecycle>, Vec<bool>) {
        let classes = crate::kernel::PolydatProgram::classify_lifecycle(
            &dag.nodes,
            &dag.wiring,
            &dag.input_defs,
            &dag.output_map,
            &dag.output_modifiers,
        );
        (classes.lifecycle, classes.nondeterministic)
    }

    /// Dedup/lookup key for a boundary wire source.
    fn src_key(src: &WireSource) -> (u8, usize, usize) {
        match src {
            WireSource::Input(i) => (0, *i, 0),
            WireSource::NodeOutput(j, p) => (1, *j, *p),
        }
    }

    struct ConePlan {
        /// Member node indices, ascending (inherits topo order).
        members: Vec<usize>,
        /// Boundary input sources, deduped, in first-use order.
        boundary_in: Vec<WireSource>,
        in_types: Vec<PortType>,
        /// Boundary output ports `(member_idx, port)`, first-use order.
        boundary_out: Vec<(usize, usize)>,
        out_types: Vec<PortType>,
    }

    /// Replace eligible cones in `dag` with compiled cone nodes.
    ///
    /// A cone whose code generation fails keeps its members as
    /// interpreter nodes. Under `Auto` the failure is recorded on the
    /// tree's ledger and the DAG is left valid and topologically sorted;
    /// under `Force` it is returned as [`AssemblyError::NativeCone`]
    /// (engines.md §2.1). A node that alone reads more than
    /// [`MAX_CONE_INPUTS`] boundary inputs stays on the interpreter under
    /// either mode and is recorded (engines.md §2.2).
    pub(crate) fn extract_jit_cones(
        dag: &mut ResolvedDag,
        mode: JitMode,
    ) -> Result<(), AssemblyError> {
        let min_members = match mode {
            JitMode::Off => return Ok(()),
            JitMode::Auto => 2,
            JitMode::Force => 1,
        };
        let n = dag.nodes.len();
        if n == 0 {
            return Ok(());
        }

        let (lifecycles, volatile) = classify_lifecycles(dag);
        // Eligibility in topological order, because the None rule
        // (engines.md §3.3) for a None-tolerant node depends on its sources: the
        // kernel guard makes a fused cone None whenever a boundary
        // input is None, so a node that would have seen the None and
        // produced a value (`tile_encode` writes `null`, `to_json`
        // keeps going) may join only when every input is an intra-cone
        // wire from an eligible node, where no None can arrive. Every
        // other node is guarded the same way fused or not.
        let mut eligible: Vec<bool> = vec![false; n];
        for i in 0..n {
            if lifecycles[i] != crate::kernel::EvalLifecycle::Dynamic {
                continue;
            }
            let nd = dag.nodes[i].as_ref();
            if !node_eligible(nd, &crate::compile::assembly::wire_types_of(dag, i)) {
                continue;
            }
            if !crate::compile::none_rule_admits(
                nd.accepts_none_inputs(),
                &dag.wiring[i],
                &eligible,
            ) {
                continue;
            }
            eligible[i] = true;
        }

        // Connected components over eligible-to-eligible wires, by the
        // rule every fusing engine shares (compile::fusion_units). A
        // volatile node never shares a cone with a node that is not: a
        // cone runs whole, so every read that re-evaluates the volatile
        // node would re-run the cached work upstream of it too.
        let preds: Vec<Vec<usize>> = dag
            .wiring
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
        // Nor does a node that reads an extern a host can clear share
        // one with a node that does not depend on it: a `None` on a
        // cone's boundary makes every output of the cone `None`, and it
        // must reach only the outputs that depend on the extern.
        let class: Vec<u64> = volatile.iter().map(|&v| v as u64).collect();
        let unset_read = crate::compile::externs::unset_read_inputs(
            &dag.input_defs,
            dag.coord_count,
            &dag.const_inits,
        );
        let reads: Vec<Vec<usize>> = dag
            .wiring
            .iter()
            .map(|w| {
                w.iter()
                    .filter_map(|src| match src {
                        WireSource::Input(c) if unset_read.get(*c) == Some(&true) => Some(*c),
                        _ => None,
                    })
                    .collect()
            })
            .collect();
        let class = crate::compile::fusion_units::refine_by_externs(&preds, &reads, &class);
        let components = crate::compile::fusion_units::components(&preds, &eligible, &class);

        // Consumer adjacency over the ORIGINAL node graph — the
        // convexity walk below routes through it.
        let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, ps) in preds.iter().enumerate() {
            for &j in ps {
                consumers[j].push(i);
            }
        }

        // Every node's output types, read before the members leave the
        // graph: a cone's boundary input may be an output of a node an
        // earlier cone took.
        let out_types: Vec<Vec<PortType>> = dag
            .nodes
            .iter()
            .map(|nd| nd.meta().outs.iter().map(|p| p.typ).collect())
            .collect();
        let mut nodes_opt: Vec<Option<Box<dyn PolydatNode>>> = std::mem::take(&mut dag.nodes)
            .into_iter()
            .map(Some)
            .collect();
        let mut cones: Vec<(ConePlan, JitConeNode)> = Vec::new();

        // A connected component is not necessarily convex: an
        // eligible→ineligible→eligible sandwich whose ends connect
        // through another eligible path lands both ends in one component
        // while the middle stays out, and fusing it would make the
        // middle both a consumer and a producer of the cone, a cycle in
        // the spliced graph. Such a component is split into the convex
        // pieces every fusing engine forms from it (engines.md §8), and
        // each piece is then cut to the input bound. The nodes are in
        // topological order, the order `convex_pieces` walks.
        let topo: Vec<usize> = (0..n).collect();
        let convex: Vec<Vec<usize>> = components
            .into_iter()
            .filter(|members| members.len() >= min_members)
            .flat_map(|members| {
                if crate::compile::fusion_units::is_convex(&members, &consumers) {
                    vec![members]
                } else {
                    crate::compile::fusion_units::convex_pieces(&members, &preds, &topo, false)
                }
            })
            .collect();

        for members in &convex {
            if members.len() < min_members {
                continue;
            }
            let split = split_by_inputs(dag, members, &preds);
            for (node, inputs) in split.over_bound {
                record_fallback(
                    dag,
                    &[node],
                    &nodes_opt,
                    inputs,
                    ConeFallbackKind::InputBound,
                    format!(
                        "{inputs} distinct boundary inputs exceed the {MAX_CONE_INPUTS}-input \
                         piece bound"
                    ),
                );
            }
            for piece in &split.pieces {
                if piece.len() < min_members {
                    continue;
                }
                let Some(plan) = plan_cone(dag, piece, &nodes_opt, &out_types) else {
                    // plan_cone audit-logs its own rejection reason;
                    // the piece stays on the interpreter.
                    continue;
                };
                match build_cone(dag, &plan, &mut nodes_opt) {
                    Ok(cone) => {
                        // Formation is diagnosable state too: cone-aware
                        // bench reporting keys on this line to verify
                        // extraction actually ran.
                        crate::library::support::audit::debug(&format!(
                            "jit cone: fused {} members ({} boundary in, {} out): {}",
                            plan.members.len(),
                            plan.boundary_in.len(),
                            plan.boundary_out.len(),
                            cone.meta().name,
                        ));
                        cones.push((plan, cone));
                    }
                    // build_cone restored the members. Force builds
                    // native code or fails; Auto keeps the members on
                    // the interpreter and records the fallback.
                    Err(e) => {
                        if mode == JitMode::Force {
                            return Err(AssemblyError::NativeCone {
                                cone: label_of(plan.members.iter().map(|&m| {
                                    nodes_opt[m]
                                        .as_ref()
                                        .map_or("", |nd| nd.meta().name.as_str())
                                })),
                                reason: e,
                            });
                        }
                        record_fallback(
                            dag,
                            &plan.members,
                            &nodes_opt,
                            plan.boundary_in.len(),
                            ConeFallbackKind::Codegen,
                            e,
                        );
                    }
                }
            }
        }

        if cones.is_empty() {
            dag.nodes = nodes_opt.into_iter().map(Option::unwrap).collect();
            return Ok(());
        }
        rebuild(dag, nodes_opt, cones);
        Ok(())
    }

    /// Record on the tree's ledger that the cone of `members` stays on
    /// the interpreter, and write the same to the audit channel.
    fn record_fallback(
        dag: &ResolvedDag,
        members: &[usize],
        nodes: &[Option<Box<dyn PolydatNode>>],
        boundary_inputs: usize,
        kind: ConeFallbackKind,
        reason: String,
    ) {
        let names: Vec<String> = members
            .iter()
            .map(|&m| {
                nodes[m]
                    .as_ref()
                    .map_or_else(String::new, |nd| nd.meta().name.clone())
            })
            .collect();
        let mut outputs: Vec<String> = dag
            .output_map
            .iter()
            .filter(|(_, (j, _))| members.contains(j))
            .map(|(name, _)| name.clone())
            .collect();
        outputs.sort_unstable();
        crate::library::support::audit::warn(&format!(
            "jit cone: {} stays on the interpreter ({kind:?}): {reason}",
            label_of(names.iter().map(String::as_str)),
        ));
        dag.ledger.record_cone_fallback(ConeFallback {
            context: dag.context.clone(),
            members: names,
            outputs,
            boundary_inputs,
            kind,
            reason,
        });
    }

    /// A component cut into pieces within [`MAX_CONE_INPUTS`], and the
    /// nodes no piece can hold, each with the boundary inputs it reads.
    struct InputSplit {
        pieces: Vec<Vec<usize>>,
        over_bound: Vec<(usize, usize)>,
    }

    /// Cut the convex component `members` (ascending, so topological)
    /// into pieces of at most [`MAX_CONE_INPUTS`] distinct boundary
    /// inputs each. A component within the bound is one piece.
    ///
    /// Members join the open piece in topological order until the next
    /// one would take its boundary over the bound; that member opens the
    /// next piece. Each piece is a run of the component's topological
    /// order, and a run of a convex component is convex: a path between
    /// two of its members passes only through nodes between them in that
    /// order, and a member between them is in the run, while a path
    /// through a node outside the component would leave the component
    /// and come back. Every wire between pieces runs forward, so the
    /// pieces form no cycle. Each run is then cut into its connected
    /// parts, which stay convex and never read more than the run did, so
    /// a pull runs only the part its output needs. Every piece keeps the
    /// component's class, so lifecycle, volatility, extern set, and
    /// purity hold as they held for the component. A member that alone
    /// reads more than the bound is in no piece.
    fn split_by_inputs(dag: &ResolvedDag, members: &[usize], preds: &[Vec<usize>]) -> InputSplit {
        let mut runs: Vec<Vec<usize>> = Vec::new();
        let mut over_bound = Vec::new();
        let mut run: Vec<usize> = Vec::new();
        let mut read: std::collections::HashSet<(u8, usize, usize)> = Default::default();
        // The distinct sources `m` reads from outside `run`, not yet in
        // `read`.
        let fresh = |m: usize, run: &[usize], read: &std::collections::HashSet<_>| {
            let mut keys: Vec<(u8, usize, usize)> = dag.wiring[m]
                .iter()
                .filter(|src| {
                    !matches!(src, WireSource::NodeOutput(j, _) if run.binary_search(j).is_ok())
                })
                .map(src_key)
                .filter(|k| !read.contains(k))
                .collect();
            keys.sort_unstable();
            keys.dedup();
            keys
        };
        for &m in members {
            let keys = fresh(m, &run, &read);
            if read.len() + keys.len() <= MAX_CONE_INPUTS {
                read.extend(keys);
                run.push(m);
                continue;
            }
            if !run.is_empty() {
                runs.push(std::mem::take(&mut run));
                read.clear();
            }
            let keys = fresh(m, &run, &read);
            if keys.len() > MAX_CONE_INPUTS {
                over_bound.push((m, keys.len()));
                continue;
            }
            read.extend(keys);
            run.push(m);
        }
        if !run.is_empty() {
            runs.push(run);
        }
        if runs.len() == 1 && over_bound.is_empty() {
            return InputSplit {
                pieces: runs,
                over_bound,
            };
        }
        let mut in_run = vec![false; preds.len()];
        let classes = vec![0u64; preds.len()];
        let mut pieces = Vec::new();
        for run in runs {
            for &m in &run {
                in_run[m] = true;
            }
            pieces.extend(crate::compile::fusion_units::components(
                preds, &in_run, &classes,
            ));
            for &m in &run {
                in_run[m] = false;
            }
        }
        InputSplit { pieces, over_bound }
    }

    /// Compute the cone's boundaries; `None` rejects the piece (dead
    /// outputs, a mistyped or None-tolerant boundary, an unmarshalable
    /// edge type).
    /// `out_types` are every node's output types, the graph's before any
    /// cone took its members.
    fn plan_cone(
        dag: &ResolvedDag,
        members: &[usize],
        nodes: &[Option<Box<dyn PolydatNode>>],
        out_types: &[Vec<PortType>],
    ) -> Option<ConePlan> {
        let is_member = |j: usize| members.binary_search(&j).is_ok();

        let mut boundary_in: Vec<WireSource> = Vec::new();
        let mut in_types: Vec<PortType> = Vec::new();
        let mut seen_in: HashMap<(u8, usize, usize), usize> = HashMap::new();
        for &m in members {
            let member = nodes[m].as_ref()?;
            let member_ports: Vec<PortType> =
                member.meta().wire_inputs().iter().map(|p| p.typ).collect();
            let wire_types: Vec<PortType> = dag.wiring[m]
                .iter()
                .map(|src| match src {
                    WireSource::Input(i) => dag.input_defs[*i].port_type,
                    WireSource::NodeOutput(j, p) => out_types[*j][*p],
                })
                .collect();
            // A node that lowers as a slot call runs the kit built for
            // its wire types (compiled_handles.md §6), so its advertised
            // port types do not bind its wires: a variadic that inspects
            // `Value`s at P1 reads each wire as the wire is. A named
            // native lowering takes its ports as declared.
            let typed_by_wires = matches!(
                classify_node_typed(member.as_ref(), &wire_types),
                JitOp::SlotCall { .. }
            );
            for (k, src) in dag.wiring[m].iter().enumerate() {
                let ty = wire_types[k];
                // Inside a cone every wire is exactly its port's type.
                if !typed_by_wires
                    && let Some(expected) = member_ports.get(k)
                    && *expected != ty
                {
                    audit_skip(
                        members.len(),
                        &format!(
                            "input [{k}] of `{}` is a {ty:?} wire on a {expected:?} port",
                            member.meta().name
                        ),
                    );
                    return None;
                }
                let intra = matches!(src, WireSource::NodeOutput(j, _) if is_member(*j));
                // engines.md §3.3: a None-tolerant member must not sit on the
                // boundary, where a None could reach it (see the
                // eligibility pass); a component split can put it there.
                if !intra && member.accepts_none_inputs() {
                    audit_skip(
                        members.len(),
                        &format!(
                            "`{}` tolerates None inputs and input [{k}] is a boundary wire",
                            member.meta().name
                        ),
                    );
                    return None;
                }
                if intra {
                    continue;
                }
                let key = src_key(src);
                if seen_in.contains_key(&key) {
                    continue;
                }
                if !scalar_ok(ty) {
                    audit_skip(
                        members.len(),
                        &format!("boundary input of type {ty:?} is not marshalable"),
                    );
                    return None;
                }
                seen_in.insert(key, boundary_in.len());
                boundary_in.push(src.clone());
                in_types.push(ty);
            }
        }
        // `split_by_inputs` cut the component into pieces within the
        // bound.
        debug_assert!(boundary_in.len() <= MAX_CONE_INPUTS);
        // A cone with no boundary inputs is a compile-time
        // constant: it would evaluate exactly once (node_clean)
        // and belongs to const folding, not per-cycle fusion.
        // It also breaks lifecycle analysis (a no-input node
        // claiming per-cycle outputs). Leave it interpreted.
        if boundary_in.is_empty() {
            // Normal outcome for const subgraphs — the fold passes
            // own them; not worth an audit line.
            return None;
        }

        let mut boundary_out: Vec<(usize, usize)> = Vec::new();
        let mut seen_out: HashMap<(usize, usize), usize> = HashMap::new();
        let mut note_out = |j: usize, p: usize| {
            if let std::collections::hash_map::Entry::Vacant(e) = seen_out.entry((j, p)) {
                e.insert(boundary_out.len());
                boundary_out.push((j, p));
            }
        };
        for (i, wiring) in dag.wiring.iter().enumerate() {
            if is_member(i) {
                continue;
            }
            for src in wiring {
                if let WireSource::NodeOutput(j, p) = src
                    && is_member(*j)
                {
                    note_out(*j, *p);
                }
            }
        }
        for (j, p) in dag.output_map.values() {
            if is_member(*j) {
                note_out(*j, *p);
            }
        }
        if boundary_out.is_empty() {
            // Dead subgraph (no observable outputs) — DCE
            // territory, not worth an audit line.
            return None;
        }
        let out_types: Vec<PortType> = boundary_out
            .iter()
            .map(|(j, p)| out_types[*j][*p])
            .collect();
        if out_types.iter().any(|t| !scalar_ok(*t)) {
            audit_skip(members.len(), "a boundary output type is not marshalable");
            return None;
        }

        Some(ConePlan {
            members: members.to_vec(),
            boundary_in,
            in_types,
            boundary_out,
            out_types,
        })
    }

    /// A boundary input's declared default, of its own type; the cone
    /// is always evaluated with its inputs bound, so the default is
    /// never read, but the definition is typed like any input's.
    fn default_for(ty: PortType) -> Value {
        match ty {
            PortType::F64 => Value::F64(0.0),
            PortType::Bool => Value::Bool(false),
            PortType::Str => Value::Str("".into()),
            PortType::Bytes => Value::Bytes(Vec::new().into()),
            PortType::Json => Value::Json(std::sync::Arc::new(serde_json::Value::Null)),
            PortType::U64 => Value::U64(0),
            _ => Value::None,
        }
    }

    /// Attempt native compilation of the planned cone. Codegen runs
    /// before the members leave the graph permanently: on any error
    /// they are restored and the caller keeps the interpreter form.
    fn build_cone(
        dag: &ResolvedDag,
        plan: &ConePlan,
        nodes: &mut [Option<Box<dyn PolydatNode>>],
    ) -> Result<JitConeNode, String> {
        let local: HashMap<usize, usize> = plan
            .members
            .iter()
            .enumerate()
            .map(|(l, &g)| (g, l))
            .collect();
        let in_pos: HashMap<(u8, usize, usize), usize> = plan
            .boundary_in
            .iter()
            .enumerate()
            .map(|(i, s)| (src_key(s), i))
            .collect();

        let sub_wiring: Vec<Vec<WireSource>> = plan
            .members
            .iter()
            .map(|&m| {
                dag.wiring[m]
                    .iter()
                    .map(|src| match src {
                        WireSource::NodeOutput(j, p) if local.contains_key(j) => {
                            WireSource::NodeOutput(local[j], *p)
                        }
                        other => WireSource::Input(in_pos[&src_key(other)]),
                    })
                    .collect()
            })
            .collect();
        let sub_input_defs: Vec<InputDef> = plan
            .in_types
            .iter()
            .enumerate()
            .map(|(i, ty)| InputDef {
                name: format!("c{i}"),
                default: default_for(*ty),
                port_type: *ty,
                kind: InputKind::Coordinate,
                type_origin: crate::kernel::TypeOrigin::Declared,
                converts_to: None,
            })
            .collect();
        let mut sub_output_map: HashMap<String, (usize, usize)> = HashMap::new();
        let mut sub_output_order: Vec<String> = Vec::new();
        for (k, (j, p)) in plan.boundary_out.iter().enumerate() {
            let name = format!("o{k}");
            sub_output_map.insert(name.clone(), (local[j], *p));
            sub_output_order.push(name);
        }

        let taken: Vec<Box<dyn PolydatNode>> = plan
            .members
            .iter()
            .map(|&m| nodes[m].take().expect("cone member present"))
            .collect();
        let member_label = label_of(taken.iter().map(|n| n.meta().name.as_str()));

        let mut sub = ResolvedDag {
            nodes: taken,
            wiring: sub_wiring,
            input_defs: sub_input_defs,
            coord_count: plan.boundary_in.len(),
            output_map: sub_output_map,
            output_order: sub_output_order,
            cursor_schemas: Vec::new(),
            source: String::new(),
            // A member's failure is reported against the program the
            // cone stands in, as the same node's failure is reported on
            // every other engine (engines.md §3.4); the cone is not a
            // frame of its own.
            context: dag.context.clone(),
            output_modifiers: HashMap::new(),
            const_outputs: std::collections::HashSet::new(),
            const_inits: Vec::new(),
            // A cone is a fragment of the program that stands in the
            // tree's ledger already, not a program of its own: its
            // kernel is recorded nowhere.
            ledger: crate::kernel::CompileLedger::new(),
        };

        let restore = |sub_nodes: Vec<Box<dyn PolydatNode>>,
                       nodes: &mut [Option<Box<dyn PolydatNode>>]| {
            for (&m, nd) in plan.members.iter().zip(sub_nodes) {
                nodes[m] = Some(nd);
            }
        };

        let layout = match PolydatAssembler::build_jit_layout(&sub) {
            Ok(l) => l,
            Err(e) => {
                restore(sub.nodes, nodes);
                return Err(e);
            }
        };
        let (coord_slots, total_slots, jit_steps, jit_outputs, scratch, _volatile) = layout;
        // Boundary inputs occupy the first slots, each as wide as its
        // type.
        let mut in_slots = Vec::with_capacity(plan.in_types.len());
        let mut next = 0usize;
        for ty in &plan.in_types {
            in_slots.push(next);
            next += ty.slot_width();
        }
        debug_assert_eq!(coord_slots, next);
        let compiled = crate::compile::jit::compile_jit_entry(&jit_steps, Some(total_slots));
        let (code_fn, code) = match compiled {
            Ok(parts) => parts,
            Err(e) => {
                restore(sub.nodes, nodes);
                return Err(e);
            }
        };

        let out_slots: Vec<usize> = (0..plan.boundary_out.len())
            .map(|k| jit_outputs[&format!("o{k}")])
            .collect();
        // Port metadata mirrors the fused subgraph rather than
        // being synthesized: outputs clone the member's original
        // port (lifecycle analysis and downstream diagnostics see
        // what the interpreter form would have declared); inputs
        // clone the source port where one exists (graph inputs are
        // per-cycle by definition).
        let meta = NodeMeta {
            name: member_label,
            ins: plan
                .boundary_in
                .iter()
                .zip(&plan.in_types)
                .enumerate()
                .map(|(i, (src, ty))| {
                    // A boundary producer is an ineligible node, still
                    // in the slot vec, or a member of an earlier piece
                    // of the same component, already taken, whose port
                    // is built from the wire's type.
                    let mut port = match src {
                        WireSource::NodeOutput(j, p) => nodes[*j]
                            .as_ref()
                            .map(|nd| nd.meta().outs[*p].clone())
                            .unwrap_or_else(|| Port::new("", *ty)),
                        WireSource::Input(_) => Port::new("", *ty),
                    };
                    port.name = format!("c{i}");
                    port.constraint = None;
                    Slot::Wire(port)
                })
                .collect(),
            outs: plan
                .boundary_out
                .iter()
                .enumerate()
                .map(|(k, (j, p))| {
                    let mut port = sub.nodes[local[j]].meta().outs[*p].clone();
                    port.name = format!("o{k}");
                    port.constraint = None;
                    port
                })
                .collect(),
        };
        let out_ports: Vec<(usize, usize)> = plan
            .boundary_out
            .iter()
            .map(|(j, p)| (local[j], *p))
            .collect();
        // A member's failure names the member's outputs as the program
        // names them (engines.md §3.4), not as the cone numbers them: the boundary
        // outputs take the program's names for the attribution.
        let mut named = sub.output_map.clone();
        for (k, (j, p)) in plan.boundary_out.iter().enumerate() {
            let names: Vec<String> = dag
                .output_map
                .iter()
                .filter(|(_, v)| **v == (*j, *p))
                .map(|(n, _)| n.clone())
                .collect();
            if !names.is_empty()
                && let Some(target) = named.remove(&format!("o{k}"))
            {
                for n in names {
                    named.insert(n, target);
                }
            }
        }
        let numbered = std::mem::replace(&mut sub.output_map, named);
        let attribution = std::sync::Arc::new(PolydatAssembler::attribution_of(&sub));
        sub.output_map = numbered;
        Ok(JitConeNode {
            attribution,
            in_slots,
            meta,
            code_fn,
            total_slots,
            out_slots,
            in_types: plan.in_types.clone(),
            out_types: plan.out_types.clone(),
            members: sub.nodes,
            sub_wiring: sub.wiring,
            out_ports,
            scratch,
            fallible: code.fallible(),
            _module: code,
        })
    }

    /// Diagnostic name carrying the fused members, so an enriched
    /// eval panic attributes the interior functions.
    fn label_of<'a>(members: impl ExactSizeIterator<Item = &'a str>) -> String {
        const SHOWN: usize = 6;
        let count = members.len();
        let names: Vec<&str> = members.take(SHOWN).collect();
        let suffix = if count > SHOWN {
            format!("+{} more", count - SHOWN)
        } else {
            String::new()
        };
        format!("jit_cone[{}{}]", names.join("+"), suffix)
    }

    /// Splice the compiled cones into the DAG and restore
    /// topological order.
    fn rebuild(
        dag: &mut ResolvedDag,
        nodes_opt: Vec<Option<Box<dyn PolydatNode>>>,
        cones: Vec<(ConePlan, JitConeNode)>,
    ) {
        let old_n = nodes_opt.len();
        // (old_idx, port) → (cone_ordinal, cone_out_port)
        let mut cone_port: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        for (ci, (plan, _)) in cones.iter().enumerate() {
            for (k, (j, p)) in plan.boundary_out.iter().enumerate() {
                cone_port.insert((*j, *p), (ci, k));
            }
        }

        let mut kept_map: HashMap<usize, usize> = HashMap::new();
        let mut new_nodes: Vec<Box<dyn PolydatNode>> = Vec::new();
        let mut new_wiring: Vec<Vec<WireSource>> = Vec::new();
        for (old, slot) in nodes_opt.into_iter().enumerate() {
            if let Some(node) = slot {
                kept_map.insert(old, new_nodes.len());
                new_nodes.push(node);
                new_wiring.push(dag.wiring[old].clone());
            }
        }
        let cone_base = new_nodes.len();
        let mut cone_plans: Vec<ConePlan> = Vec::with_capacity(cones.len());
        for (plan, cone) in cones {
            new_nodes.push(Box::new(cone));
            new_wiring.push(plan.boundary_in.clone());
            cone_plans.push(plan);
        }

        let remap = |src: &WireSource| -> WireSource {
            match src {
                WireSource::Input(i) => WireSource::Input(*i),
                WireSource::NodeOutput(j, p) => {
                    if let Some(&nj) = kept_map.get(j) {
                        WireSource::NodeOutput(nj, *p)
                    } else {
                        let (ci, k) = cone_port[&(*j, *p)];
                        WireSource::NodeOutput(cone_base + ci, k)
                    }
                }
            }
        };
        for wiring in new_wiring.iter_mut() {
            for src in wiring.iter_mut() {
                *src = remap(src);
            }
        }
        let mut new_output_map: HashMap<String, (usize, usize)> = HashMap::new();
        for (name, (j, p)) in dag.output_map.iter() {
            let (nj, np) = match remap(&WireSource::NodeOutput(*j, *p)) {
                WireSource::NodeOutput(a, b) => (a, b),
                WireSource::Input(_) => unreachable!("outputs map to nodes"),
            };
            new_output_map.insert(name.clone(), (nj, np));
        }

        // Kahn topo sort — consumers of cone interiors may sit at
        // indices below the spliced cone node.
        let m = new_nodes.len();
        let mut indegree = vec![0usize; m];
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); m];
        for (i, wiring) in new_wiring.iter().enumerate() {
            let mut producers: Vec<usize> = wiring
                .iter()
                .filter_map(|s| match s {
                    WireSource::NodeOutput(j, _) => Some(*j),
                    WireSource::Input(_) => None,
                })
                .collect();
            producers.sort_unstable();
            producers.dedup();
            indegree[i] = producers.len();
            for j in producers {
                dependents[j].push(i);
            }
        }
        let mut order: Vec<usize> = Vec::with_capacity(m);
        let mut ready: std::collections::BinaryHeap<std::cmp::Reverse<usize>> = (0..m)
            .filter(|&i| indegree[i] == 0)
            .map(std::cmp::Reverse)
            .collect();
        while let Some(std::cmp::Reverse(i)) = ready.pop() {
            order.push(i);
            for &d in &dependents[i] {
                indegree[d] -= 1;
                if indegree[d] == 0 {
                    ready.push(std::cmp::Reverse(d));
                }
            }
        }
        assert_eq!(
            order.len(),
            m,
            "cone splice must not introduce a cycle (old_n={old_n})"
        );
        let mut pos = vec![0usize; m];
        for (new_idx, &i) in order.iter().enumerate() {
            pos[i] = new_idx;
        }

        let mut sorted_nodes: Vec<Option<Box<dyn PolydatNode>>> =
            new_nodes.into_iter().map(Some).collect();
        dag.nodes = order
            .iter()
            .map(|&i| sorted_nodes[i].take().expect("each node placed once"))
            .collect();
        dag.wiring = order
            .iter()
            .map(|&i| {
                new_wiring[i]
                    .iter()
                    .map(|s| match s {
                        WireSource::Input(k) => WireSource::Input(*k),
                        WireSource::NodeOutput(j, p) => WireSource::NodeOutput(pos[*j], *p),
                    })
                    .collect()
            })
            .collect();
        dag.output_map = new_output_map
            .into_iter()
            .map(|(name, (j, p))| (name, (pos[j], p)))
            .collect();
        let _ = cone_plans;
    }
}
