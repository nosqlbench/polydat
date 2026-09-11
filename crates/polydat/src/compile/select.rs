// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Engine selection: analyze a compiled DAG and choose the optimal
//! monomorphic kernel variant.
//!
//! The heuristic is applied at construction time by the assembler.
//! The resulting kernel has the selected optimizations baked in with
//! no runtime strategy branching.

use crate::ast::PolydatNode;
use crate::compile::closures::{CompiledKernelPull, CompiledKernelPushPull, CompiledKernelRaw};
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
    /// cone guard. Only selected when output cones are large AND
    /// partially stable.
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
/// Heuristic (from benchmark findings in memo 09):
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

// ── Compiled engine wrappers ───────────────────────────────────

/// Auto-selected P2 compiled kernel. The variant is chosen at
/// construction time based on graph analysis. One outer branch
/// per eval call (perfectly predicted), zero branches inside.
pub enum P2Engine {
    /// Every evaluation runs every step.
    Raw(CompiledKernelRaw),
    /// The cone guard alone.
    Pull(CompiledKernelPull),
    /// Per-step skipping and the cone guard.
    PushPull(CompiledKernelPushPull),
}

impl P2Engine {
    #[inline]
    /// Evaluate for the coordinates and read one slot.
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        match self {
            P2Engine::Raw(k) => k.eval_for_slot(coords, slot),
            P2Engine::Pull(k) => k.eval_for_slot(coords, slot),
            P2Engine::PushPull(k) => k.eval_for_slot(coords, slot),
        }
    }

    #[inline]
    /// Evaluate for the coordinates.
    pub fn eval(&mut self, coords: &[u64]) {
        match self {
            P2Engine::Raw(k) => k.eval(coords),
            P2Engine::Pull(k) => k.eval(coords),
            P2Engine::PushPull(k) => k.eval(coords),
        }
    }

    #[inline]
    /// Read a slot of the last evaluation.
    pub fn get_slot(&self, slot: usize) -> u64 {
        match self {
            P2Engine::Raw(k) => k.get_slot(slot),
            P2Engine::Pull(k) => k.get_slot(slot),
            P2Engine::PushPull(k) => k.get_slot(slot),
        }
    }

    /// The slot of a named output.
    pub fn resolve_output(&self, name: &str) -> Option<usize> {
        match self {
            P2Engine::Raw(k) => k.resolve_output(name),
            P2Engine::Pull(k) => k.resolve_output(name),
            P2Engine::PushPull(k) => k.resolve_output(name),
        }
    }

    /// The coordinate inputs.
    pub fn coord_count(&self) -> usize {
        match self {
            P2Engine::Raw(k) => k.coord_count(),
            P2Engine::Pull(k) => k.coord_count(),
            P2Engine::PushPull(k) => k.coord_count(),
        }
    }

    /// The provenance mode the selector chose.
    pub fn prov_mode(&self) -> ProvMode {
        match self {
            P2Engine::Raw(_) => ProvMode::Raw,
            P2Engine::Pull(_) => ProvMode::Pull,
            P2Engine::PushPull(_) => ProvMode::PushPull,
        }
    }
}

/// Auto-selected P3 kernel: the hybrid kernel in the provenance mode
/// the selector chose (engine_parity.md, step 7).
#[cfg(feature = "jit")]
pub enum P3Engine {
    /// Every evaluation runs every step.
    Raw(crate::compile::hybrid::HybridKernelRaw),
    /// The cone guard alone.
    Pull(crate::compile::hybrid::HybridKernelPull),
    /// Per-step skipping and the cone guard.
    PushPull(crate::compile::hybrid::HybridKernelPushPull),
}

#[cfg(feature = "jit")]
impl P3Engine {
    #[inline]
    /// Evaluate for the coordinates and read one slot.
    pub fn eval_for_slot(&mut self, coords: &[u64], slot: usize) -> u64 {
        match self {
            P3Engine::Raw(k) => k.eval_for_slot(coords, slot),
            P3Engine::Pull(k) => k.eval_for_slot(coords, slot),
            P3Engine::PushPull(k) => k.eval_for_slot(coords, slot),
        }
    }

    #[inline]
    /// Evaluate for the coordinates.
    pub fn eval(&mut self, coords: &[u64]) {
        match self {
            P3Engine::Raw(k) => k.eval(coords),
            P3Engine::Pull(k) => k.eval(coords),
            P3Engine::PushPull(k) => k.eval(coords),
        }
    }

    #[inline]
    /// Read a slot of the last evaluation.
    pub fn get_slot(&self, slot: usize) -> u64 {
        match self {
            P3Engine::Raw(k) => k.get_slot(slot),
            P3Engine::Pull(k) => k.get_slot(slot),
            P3Engine::PushPull(k) => k.get_slot(slot),
        }
    }

    /// The slot of a named output.
    pub fn resolve_output(&self, name: &str) -> Option<usize> {
        match self {
            P3Engine::Raw(k) => k.resolve_output(name),
            P3Engine::Pull(k) => k.resolve_output(name),
            P3Engine::PushPull(k) => k.resolve_output(name),
        }
    }

    /// The coordinate inputs.
    pub fn coord_count(&self) -> usize {
        match self {
            P3Engine::Raw(k) => k.coord_count(),
            P3Engine::Pull(k) => k.coord_count(),
            P3Engine::PushPull(k) => k.coord_count(),
        }
    }

    /// The provenance mode the selector chose.
    pub fn prov_mode(&self) -> ProvMode {
        match self {
            P3Engine::Raw(_) => ProvMode::Raw,
            P3Engine::Pull(_) => ProvMode::Pull,
            P3Engine::PushPull(_) => ProvMode::PushPull,
        }
    }
}

// ── The engine a host chooses (engine_parity.md, step 4) ──────────

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
/// runs and nothing else (docs/design/engine_parity.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Engine {
    /// The interpreter, with native cones per the assembler's
    /// [`JitMode`](crate::JitMode): what `compile()` builds.
    Interpreter,
    /// The closure tier: every node runs its generated closure over
    /// one slot buffer.
    Closures(Provenance),
    /// Native code where a node has a lowering, its closure elsewhere:
    /// the P3 tier. Refused by a build without the `jit` feature.
    Native(Provenance),
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
            Engine::Interpreter => write!(f, "interpreter"),
            Engine::Closures(p) => write!(f, "closures ({p:?})"),
            Engine::Native(p) => write!(f, "native ({p:?})"),
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
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelError::Source(e) => write!(f, "{e}"),
            KernelError::Assembly(e) => write!(f, "{e}"),
            KernelError::Refused { engine, reason } => {
                write!(f, "the {engine} engine refuses this program: {reason}")
            }
        }
    }
}

impl std::error::Error for KernelError {}

impl From<crate::compile::assembly::AssemblyError> for KernelError {
    fn from(e: crate::compile::assembly::AssemblyError) -> Self {
        KernelError::Assembly(e)
    }
}
