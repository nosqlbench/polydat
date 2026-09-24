// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The engine a host chooses, the provenance mode a compiled engine is
//! built with, and the selector that picks a mode from a graph's shape
//! when the host leaves it to `Provenance::Auto`.

use crate::ast::PolydatNode;
use crate::kernel::WireSource;
use std::collections::HashMap;

/// Which provenance optimization the compiler selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvMode {
    /// No provenance — eval runs all nodes unconditionally.
    Raw,
    /// Pull-side cone guard only. `set_inputs` tracks `changed_mask`,
    /// `eval_for_slot` skips eval when the output cone is clean.
    /// Zero overhead on all-dirty graphs.
    Pull,
    /// Push + pull. Per-node dirty tracking in `set_inputs` +
    /// cone guard. Selected whenever the graph has two or more inputs.
    PushPull,
}

/// Graph analysis results used by the engine selection heuristic.
#[derive(Debug, Clone)]
pub struct GraphAnalysis {
    /// Nodes in the graph.
    pub total_nodes: usize,
    /// Inputs.
    pub num_inputs: usize,
    /// Named outputs.
    pub num_outputs: usize,
    /// Per-output cone size (number of nodes in transitive dependency).
    pub output_cone_sizes: Vec<(String, usize)>,
    /// max(cone_size) / total_nodes
    pub max_cone_ratio: f64,
    /// Average cone_size / total_nodes
    pub avg_cone_ratio: f64,
}

/// Analyze a resolved DAG to compute structural metrics.
pub fn analyze_graph(
    nodes: &[Box<dyn PolydatNode>],
    wiring: &[Vec<WireSource>],
    output_map: &HashMap<String, (usize, usize)>,
) -> GraphAnalysis {
    let total_nodes = nodes.len();

    // Compute per-output cone size: count nodes reachable from each output.
    // A node is in the cone if its provenance overlaps with the output's.
    // Actually, we need the transitive upstream set, which is the set of
    // nodes that can reach this output. We compute this by walking backward
    // from the output node.
    let mut output_cone_sizes = Vec::new();
    for (name, &(node_idx, _port)) in output_map {
        let cone_size = compute_cone_size(node_idx, wiring);
        output_cone_sizes.push((name.clone(), cone_size));
    }

    let max_cone = output_cone_sizes.iter().map(|(_, s)| *s).max().unwrap_or(0);
    let avg_cone: f64 = if output_cone_sizes.is_empty() {
        0.0
    } else {
        output_cone_sizes
            .iter()
            .map(|(_, s)| *s as f64)
            .sum::<f64>()
            / output_cone_sizes.len() as f64
    };

    let max_cone_ratio = if total_nodes > 0 {
        max_cone as f64 / total_nodes as f64
    } else {
        1.0
    };
    let avg_cone_ratio = if total_nodes > 0 {
        avg_cone / total_nodes as f64
    } else {
        1.0
    };

    // Count distinct inputs
    let mut max_input = 0usize;
    for sources in wiring {
        for s in sources {
            if let WireSource::Input(idx) = s {
                max_input = max_input.max(*idx + 1);
            }
        }
    }

    GraphAnalysis {
        total_nodes,
        num_inputs: max_input,
        num_outputs: output_map.len(),
        output_cone_sizes,
        max_cone_ratio,
        avg_cone_ratio,
    }
}

/// Count the number of nodes in the transitive upstream cone of a node.
fn compute_cone_size(node_idx: usize, wiring: &[Vec<WireSource>]) -> usize {
    let mut visited = vec![false; wiring.len()];
    let mut stack = vec![node_idx];
    let mut count = 0;
    while let Some(idx) = stack.pop() {
        if idx >= visited.len() || visited[idx] {
            continue;
        }
        visited[idx] = true;
        count += 1;
        for source in &wiring[idx] {
            if let WireSource::NodeOutput(upstream, _) = source
                && !visited[*upstream]
            {
                stack.push(*upstream);
            }
        }
    }
    count
}

/// Select the optimal provenance mode based on graph analysis.
///
/// Heuristic (`crates/polydat/docs/design/engines.md`):
/// - Pull has zero overhead on all-dirty graphs (cone check ~2ns)
/// - Pull is the safe default for selective output access
/// - PushPull when multiple inputs exist (push skip helps within
///   dirty cones when some subgraphs are stable)
/// - Raw only for tiny single-input graphs
pub fn select_prov_mode(analysis: &GraphAnalysis) -> ProvMode {
    // Tiny single-input graphs: skip provenance data entirely.
    // The overhead of tracking changed_mask isn't worth it.
    if analysis.total_nodes < 15 && analysis.num_inputs <= 1 {
        return ProvMode::Raw;
    }

    // Multiple inputs: some may be stable at runtime, enabling both
    // push-side skip (within dirty cones) and pull-side skip (clean cones).
    // PushPull is the right choice because:
    // - If an output's cone is clean: pull guard skips eval entirely
    // - If an output's cone is dirty: push skip avoids stable nodes
    // The push overhead (~10ns in set_inputs for dependent marking)
    // is justified by the potential to skip 30-80% of nodes within
    // dirty cones.
    if analysis.num_inputs >= 2 {
        return ProvMode::PushPull;
    }

    // Single input: every node's cone includes the one input, so the
    // pull guard will always see "dirty" and fall through. Pull still
    // has zero overhead (the AND + branch costs ~2ns), so prefer it
    // over Raw as free insurance for future multi-output scenarios
    // where not all outputs are pulled every cycle.
    ProvMode::Pull
}

// ── The engine a host chooses (engines.md §3.5) ──────────

/// How much of a kernel's work is skipped when inputs repeat: the
/// provenance mode a compiled engine is built with. Every mode computes
/// the same values; the modes differ in what they recompute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provenance {
    /// Every evaluation runs every step.
    Raw,
    /// A changed input reruns only the steps downstream of it.
    Push,
    /// An output whose cone no changed input reaches is not recomputed.
    Pull,
    /// Both: per-step skipping and the cone guard.
    PushPull,
    /// The selector's choice from the graph's shape
    /// ([`select_prov_mode`]).
    Auto,
}

/// The engine a program runs on. Every engine accepts every program the
/// interpreter accepts, or refuses it with a reason
/// ([`KernelError::Refused`]); the choice changes how fast the program
/// runs and nothing else (docs/design/engines.md §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Engine {
    /// The interpreter, with as much of its graph fused into native
    /// cones as the [`JitMode`](crate::JitMode) says: what `compile()`
    /// builds, under the assembler's mode.
    Interpreter(crate::compile::cone::JitMode),
    /// The closure tier: every node runs its generated closure over
    /// one slot buffer.
    Closures(Provenance),
    /// Native code where a node has a lowering, its closure elsewhere:
    /// the P3 tier. Built in every configuration: a build without the
    /// `jit` feature runs the same kernel with every step a closure and
    /// no native segment in it, which `plan()` reports.
    Native(Provenance),
    /// Native code and nothing else: the differential tier behind
    /// [`Engine::Native`] (engines.md §8). It differs from `Native` in
    /// what it does with a node that has no native lowering, which is
    /// to refuse the program rather than run that node's closure. A
    /// host asks for it to be told whether its program is fully native,
    /// which `Native` can never answer because it always succeeds.
    /// Refused by a build without the `jit` feature.
    ///
    /// Only [`Provenance::Raw`] and [`Provenance::PushPull`] have a
    /// pure kernel; the other named modes are refused, and
    /// [`Provenance::Auto`] resolves to one of the two.
    PureNative(Provenance),
}

impl Default for Engine {
    /// The engine a host gets when it names none: the fastest this build
    /// has, P3 with the `jit` feature and the closure tier without, with
    /// the provenance mode left to the selector. Compiled code is the
    /// default; the interpreter is a choice.
    fn default() -> Self {
        #[cfg(feature = "jit")]
        {
            Engine::Native(Provenance::Auto)
        }
        #[cfg(not(feature = "jit"))]
        {
            Engine::Closures(Provenance::Auto)
        }
    }
}

impl std::fmt::Display for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Engine::Interpreter(crate::compile::cone::JitMode::Auto) => write!(f, "interpreter"),
            Engine::Interpreter(crate::compile::cone::JitMode::Off) => {
                write!(f, "interpreter (cones off)")
            }
            Engine::Interpreter(crate::compile::cone::JitMode::Force) => {
                write!(f, "interpreter (cones forced)")
            }
            Engine::Closures(p) => write!(f, "closures ({p:?})"),
            Engine::Native(p) => write!(f, "native ({p:?})"),
            Engine::PureNative(p) => write!(f, "pure native ({p:?})"),
        }
    }
}

/// What a kernel's engine decided for its program: how much of it runs
/// as native segments, as closure steps, and on the interpreter. The one
/// planning detail a kernel exposes, on every engine
/// ([`Kernel::plan`](crate::Kernel::plan)).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EnginePlan {
    /// Runs of nodes compiled to one native function each; on the
    /// interpreter, its native cones.
    pub native_segments: usize,
    /// Nodes that run their generated closure.
    pub closure_steps: usize,
    /// Nodes the interpreter dispatches itself.
    pub interpreted_nodes: usize,
}

impl std::fmt::Display for EnginePlan {
    /// The non-zero counts, native first: `4 native segment(s), 3
    /// closure step(s)`; `nothing` for an empty program.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts = Vec::new();
        if self.native_segments > 0 {
            parts.push(format!("{} native segment(s)", self.native_segments));
        }
        if self.closure_steps > 0 {
            parts.push(format!("{} closure step(s)", self.closure_steps));
        }
        if self.interpreted_nodes > 0 {
            parts.push(format!("{} interpreted node(s)", self.interpreted_nodes));
        }
        if parts.is_empty() {
            write!(f, "nothing")
        } else {
            write!(f, "{}", parts.join(", "))
        }
    }
}

/// Why a kernel was not built: the one error type of every constructor
/// that takes an [`Engine`].
#[derive(Debug)]
pub enum KernelError {
    /// The source did not parse or compile; the message is the DSL
    /// front end's.
    Source(String),
    /// The graph did not assemble.
    Assembly(crate::compile::assembly::AssemblyError),
    /// The engine refuses this graph, which the interpreter accepts;
    /// `reason` names the node or construct.
    Refused {
        /// The engine that refused.
        engine: Engine,
        /// The node or construct it cannot run.
        reason: String,
    },
    /// A compile-constant step could not be computed. A step no input
    /// reaches runs once at build, so what it does there it will do on
    /// every pull; there is nothing a later evaluation could supply
    /// that would make it succeed. Distinct from [`Self::Refused`],
    /// which is one engine declining a program the others accept:
    /// every engine reports this one, and reports it at build.
    ConstantFold {
        /// The failure as the node reported it, enriched with the
        /// node's name, the outputs it feeds, and the program's
        /// diagnostic context, as an evaluation failure would be.
        reason: String,
    },
    /// A value written into a kernel while building it was refused: an
    /// iteration binding or a value a binder copied from a parent that
    /// does not satisfy the child's declared input.
    Write(crate::kernel::WriteError),
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelError::Source(e) => write!(f, "{e}"),
            KernelError::Assembly(e) => write!(f, "{e}"),
            KernelError::Refused { engine, reason } => {
                write!(f, "the {engine} engine refuses this program: {reason}")
            }
            KernelError::ConstantFold { reason } => {
                write!(
                    f,
                    "a value this program computes at build could not be computed: {reason}"
                )
            }
            KernelError::Write(e) => write!(f, "a value written while binding was refused: {e}"),
        }
    }
}

impl std::error::Error for KernelError {}

impl From<crate::compile::assembly::AssemblyError> for KernelError {
    fn from(e: crate::compile::assembly::AssemblyError) -> Self {
        // The interpreter's build path speaks `AssemblyError`, so a
        // fold failure travels as one; it is the same failure the
        // compiled engines report directly, and reads as the same kind
        // here.
        match e {
            crate::compile::assembly::AssemblyError::ConstantFold(reason) => {
                KernelError::ConstantFold { reason }
            }
            other => KernelError::Assembly(other),
        }
    }
}
