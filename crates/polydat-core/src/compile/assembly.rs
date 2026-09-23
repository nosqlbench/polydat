// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Programmatic assembly API for building Polydat Kernels.
//!
//! The assembler validates wiring and types, auto-inserts edge adapters,
//! topologically sorts nodes, and builds a kernel on any engine: a host
//! adds nodes and wires (or takes the assembler the DSL built from
//! source) and calls [`PolydatAssembler::compile_kernel`] for the default
//! engine, [`PolydatAssembler::compile_with`] for a named one, or
//! [`PolydatAssembler::compile`] for the interpreter kernel as a concrete
//! type. The `try_compile*` constructors build one engine's kernel as its
//! concrete type for the differential suites and the ladder.

use std::collections::HashMap;

use crate::ast::SlotShape;
use crate::ast::{PolydatNode, PortType};
use crate::compile::closures::{
    CompiledKernelPull, CompiledKernelPush, CompiledKernelPushPull, CompiledKernelRaw,
};
use crate::compile::select::{self, ProvMode};
use crate::kernel::{PolydatKernel, PolydatProgram, WireSource};
use crate::library::convert::{F64ToString, U64ToF64, U64ToString};
use crate::library::json::JsonToStr;

/// A reference to a value in the assembler: either a coordinate or a
/// node output port.
#[derive(Debug, Clone)]
pub enum WireRef {
    /// A graph input, by name.
    Input(String),
    /// A node output: `(node_name, output_port_index)`.
    Node(String, usize),
}

impl WireRef {
    /// Convenience: reference the first (or only) output of a named node.
    pub fn node(name: impl Into<String>) -> Self {
        WireRef::Node(name.into(), 0)
    }

    /// Reference a specific output port of a named node.
    pub fn node_port(name: impl Into<String>, port: usize) -> Self {
        WireRef::Node(name.into(), port)
    }

    /// Reference a graph input by name.
    pub fn input(name: impl Into<String>) -> Self {
        WireRef::Input(name.into())
    }
}

struct PendingNode {
    name: String,
    node: Box<dyn PolydatNode>,
    inputs: Vec<WireRef>,
}

/// Errors that can occur during assembly.
#[derive(Debug)]
pub enum AssemblyError {
    /// A wire reference names no node output or input.
    UnknownWire(String),
    /// A wire's type does not match the port it feeds and no adapter heals it.
    TypeMismatch {
        /// The producing node.
        from_node: String,
        /// Its output port index.
        from_port: usize,
        /// The output's type.
        from_type: PortType,
        /// The consuming node.
        to_node: String,
        /// Its input port index.
        to_port: usize,
        /// The type the port requires.
        to_type: PortType,
    },
    /// Two nodes were added under one name.
    DuplicateNode(String),
    /// The wiring has a cycle.
    CycleDetected,
    /// A node was wired with the wrong number of inputs.
    ArityMismatch {
        /// The node.
        node_name: String,
        /// Inputs its signature takes.
        expected: usize,
        /// Inputs it was given.
        got: usize,
    },
    /// A compile-constant step could not be computed; see
    /// [`KernelError::ConstantFold`], which this becomes at the kernel
    /// boundary. Carried here so the interpreter's build path, which
    /// speaks `AssemblyError`, reports the same kind as the compiled
    /// engines do rather than folding it into `Other`.
    ConstantFold(String),
    /// Catch-all for errors from downstream phases (e.g., strict mode).
    Other(String),
}

impl std::fmt::Display for AssemblyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssemblyError::UnknownWire(name) => {
                write!(f, "unknown wire: '{name}'\n\n")?;
                writeln!(f, "  No node output or coordinate named '{name}' exists.")?;
                write!(
                    f,
                    "  Check spelling, or add a node that produces this output."
                )
            }
            AssemblyError::TypeMismatch {
                from_node,
                from_port,
                from_type,
                to_node,
                to_port,
                to_type,
            } => {
                writeln!(
                    f,
                    "type mismatch: cannot connect {from_type} output to {to_type} input"
                )?;
                writeln!(f)?;
                writeln!(
                    f,
                    "  {from_node} [{from_port}]  ──({from_type})──▶  {to_node} [{to_port}] expects {to_type}"
                )?;
                writeln!(f)?;
                // Suggest auto-adapters that exist
                let suggestion = match (from_type, to_type) {
                    (PortType::U64, PortType::Str) => {
                        Some("This should auto-convert. If you see this, file a bug.")
                    }
                    (PortType::F64, PortType::Str) => {
                        Some("This should auto-convert. If you see this, file a bug.")
                    }
                    (PortType::U64, PortType::F64) => {
                        Some("This should auto-convert. If you see this, file a bug.")
                    }
                    (PortType::U64, PortType::Bytes) => {
                        Some("Add u64_to_bytes() between them to convert.")
                    }
                    (PortType::Str, PortType::Bytes) => {
                        Some("String cannot be directly used as bytes.")
                    }
                    (PortType::U64, PortType::Json) => {
                        Some("Add to_json() between them to wrap as JSON.")
                    }
                    (PortType::Str, PortType::Json) => {
                        Some("Add str_to_json() to parse the string as JSON.")
                    }
                    (PortType::Bytes, PortType::Str) => {
                        Some("Add to_hex() or to_base64() to convert bytes to string.")
                    }
                    (PortType::Bytes, PortType::U64) => {
                        Some("Bytes cannot be directly converted to u64.")
                    }
                    _ => None,
                };
                if let Some(hint) = suggestion {
                    write!(f, "  Hint: {hint}")?;
                }
                Ok(())
            }
            AssemblyError::DuplicateNode(name) => {
                write!(f, "duplicate node name: '{name}'\n\n")?;
                write!(f, "  Two nodes cannot share the same name.")
            }
            AssemblyError::CycleDetected => {
                write!(f, "cycle detected in DAG\n\n")?;
                writeln!(
                    f,
                    "  The graph contains a loop. Polydat graphs must be acyclic"
                )?;
                write!(f, "  (data flows in one direction only).")
            }
            AssemblyError::ArityMismatch {
                node_name,
                expected,
                got,
            } => {
                write!(f, "wrong number of inputs for '{node_name}'\n\n")?;
                writeln!(f, "  Expected {expected} input(s), but got {got}.")?;
                if *got < *expected {
                    write!(f, "  Connect more wires to this node's input ports.")
                } else {
                    write!(f, "  Disconnect extra wires from this node.")
                }
            }
            AssemblyError::ConstantFold(msg) => write!(
                f,
                "a value this program computes at build could not be computed: {msg}"
            ),
            AssemblyError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for AssemblyError {}

/// Validated, topologically sorted intermediate form.
pub(crate) struct ResolvedDag {
    /// Nodes in topological order.
    pub(crate) nodes: Vec<Box<dyn PolydatNode>>,
    /// Per-node wiring (in topological order).
    pub(crate) wiring: Vec<Vec<WireSource>>,
    /// All input definitions (coordinates + captures).
    pub(crate) input_defs: Vec<crate::kernel::InputDef>,
    /// Number of coordinate inputs.
    pub(crate) coord_count: usize,
    /// Output name → (node_index_in_sorted, output_port_index).
    pub(crate) output_map: HashMap<String, (usize, usize)>,
    /// Output names in declaration order.
    pub(crate) output_order: Vec<String>,
    /// Source text for diagnostics.
    pub(crate) source: String,
    /// Diagnostic context.
    pub(crate) context: String,
    /// Output binding modifiers.
    pub(crate) output_modifiers: HashMap<String, crate::dsl::ast::BindingModifier>,
    /// Names declared with `init` (SRD 11 §"Init Binding Contract").
    pub(crate) const_outputs: std::collections::HashSet<String>,
    /// The cursors the program declares.
    pub(crate) cursor_schemas: Vec<crate::iteration::source::SourceSchema>,
    /// The compile ledger every program built from this graph records in.
    pub(crate) ledger: std::sync::Arc<crate::kernel::CompileLedger>,
}

impl ResolvedDag {
    /// Coordinate input names (for P2/P3 kernels that use positional u64 buffers).
    fn input_names(&self) -> Vec<String> {
        self.input_defs[..self.coord_count]
            .iter()
            .map(|d| d.name.clone())
            .collect()
    }
}

/// Per-port slot layout for compiled kernels
/// (type_system_alignment.md §6). Each port occupies
/// `PortType::slot_width()` consecutive buffer slots: an immediate
/// is one slot; a 128-bit value or a `Ref2` pair is two.
struct SlotLayout {
    /// Per kernel input: first slot index.
    input_starts: Vec<usize>,
    /// Total slots occupied by kernel inputs.
    coord_slots: usize,
    /// Per node, per output port: first slot index.
    port_offsets: Vec<Vec<usize>>,
    /// Total buffer length.
    total_slots: usize,
}

fn slot_layout(resolved: &ResolvedDag) -> SlotLayout {
    let mut input_starts = Vec::with_capacity(resolved.coord_count);
    let mut next = 0usize;
    for d in &resolved.input_defs {
        input_starts.push(next);
        next += d.port_type.slot_width();
    }
    let coord_slots = next;
    let mut port_offsets: Vec<Vec<usize>> = Vec::with_capacity(resolved.nodes.len());
    for node in &resolved.nodes {
        let mut po = Vec::with_capacity(node.meta().outs.len());
        for out in &node.meta().outs {
            po.push(next);
            next += out.typ.slot_width();
        }
        port_offsets.push(po);
    }
    SlotLayout {
        input_starts,
        coord_slots,
        port_offsets,
        total_slots: next,
    }
}

/// Compiled-op selection for one node: a copy step inline, then the
/// pure-scalar `compiled_u64` (cheapest dispatch), then the slot kit
/// for every other shape (type_system_alignment.md §4,
/// compiled_handles.md §3), else `None` → typed-eval
/// fallback. `wire_types` is the type of each wire input.
fn node_step_op(
    node: &dyn crate::ast::PolydatNode,
    wire_types: &[PortType],
) -> Option<(
    crate::compile::closures::StepOp,
    Vec<crate::ast::ScratchElem>,
)> {
    // A plain copy (`identity`, a `__port_` passthrough): an inline
    // slot copy of an immediate; a `Ref2` value is copied into the
    // step's own scratch, since a pair is never forwarded (axiom S3).
    let meta = node.meta();
    if (meta.name == "identity" || meta.name.starts_with("__port_")) && meta.outs.len() == 1 {
        return Some(match meta.outs[0].typ.slot_color() {
            crate::ast::SlotColor::Ref2 => {
                let kit = ref_copy_kit(meta.outs[0].typ)?;
                (crate::compile::closures::StepOp::Slot(kit.op), kit.scratch)
            }
            _ => (crate::compile::closures::StepOp::Copy, Vec::new()),
        });
    }
    if let Some(op) = node.compiled_u64() {
        return Some((crate::compile::closures::StepOp::U64(op), Vec::new()));
    }
    node.compiled_slot(
        wire_types,
        crate::compile::select::Engine::Closures(crate::compile::select::Provenance::Auto),
    )
    .map(|kit| (crate::compile::closures::StepOp::Slot(kit.op), kit.scratch))
}

/// Axiom S9(a): the `(first slot, scratch index)` pairs of a step's
/// scratch-backed `Ref2` outputs. A kit's scratch entries pair with
/// the step's `Ref2` output ports in port order, skipping the entries
/// that publish no pair (a native cone's slot buffer, a render's body
/// kernels, a node's own state); a `Ref2` output beyond the kit's publishing entries is
/// not scratch-backed (a pair into interned bytes) and is validated by
/// nothing. `base` is the index of the kit's first entry in the
/// kernel's scratch. A kit with more publishing entries than the step
/// has `Ref2` outputs is a macro or builder bug, caught at
/// construction (axiom S3).
pub(crate) fn scratch_pairs(
    name: &str,
    ref_starts: &[usize],
    scratch: &[crate::ast::ScratchElem],
    base: usize,
) -> Vec<(usize, usize)> {
    use crate::ast::ScratchElem;
    let publishing: Vec<usize> = scratch
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            !matches!(
                e,
                ScratchElem::Slots | ScratchElem::Kernels | ScratchElem::State
            )
        })
        .map(|(k, _)| base + k)
        .collect();
    assert!(
        publishing.len() <= ref_starts.len(),
        "slot-op step '{name}' declares {} publishing scratch entries for {} Ref output ports",
        publishing.len(),
        ref_starts.len()
    );
    ref_starts.iter().copied().zip(publishing).collect()
}

/// The compiled form of a copy of a `Ref2` value (`identity`, the
/// compiler's `__port_<name>` passthrough, a type assertion): the pair
/// is never forwarded (axiom S3), so the elements are copied into this
/// step's own scratch entry and its pair is published. `None` for an
/// immediate color, which is copied inline.
pub(crate) fn ref_copy_kit(ty: PortType) -> Option<crate::ast::CompiledSlotKit> {
    use crate::ast::ScratchBuf;
    let elem = ty.scratch_elem()?;
    Some(crate::ast::CompiledSlotKit {
        scratch: vec![elem],
        op: Box::new(
            move |inputs: &[u64], outputs: &mut [u64], scratch: &mut [ScratchBuf]| {
                let (p, n) = (inputs[0] as usize, inputs[1] as usize);
                macro_rules! copy_into {
                    ($v:expr, $t:ty) => {{
                        $v.clear();
                        // SAFETY: the pair was published by the producing
                        // step into storage alive until it reruns (axioms
                        // S3, S4), and the layout typed it `$t`.
                        $v.extend_from_slice(unsafe {
                            std::slice::from_raw_parts(p as *const $t, n)
                        });
                    }};
                }
                match &mut scratch[0] {
                    ScratchBuf::Str(v) | ScratchBuf::Bytes(v) => copy_into!(v, u8),
                    ScratchBuf::F32(v) => copy_into!(v, f32),
                    ScratchBuf::F64(v) => copy_into!(v, f64),
                    ScratchBuf::F16(v) => copy_into!(v, half::f16),
                    ScratchBuf::I8(v) => copy_into!(v, i8),
                    ScratchBuf::I16(v) => copy_into!(v, i16),
                    ScratchBuf::I32(v) => copy_into!(v, i32),
                    ScratchBuf::I64(v) => copy_into!(v, i64),
                    ScratchBuf::Value(v) => {
                        v.clear();
                        if n > 0 {
                            // SAFETY: as above; a value pair names one `Value`.
                            v.push(unsafe { (*(p as *const crate::ast::Value)).clone() });
                        }
                    }
                    ScratchBuf::Slots(_) | ScratchBuf::Kernels(_) | ScratchBuf::State(_) => {
                        unreachable!("a copy owns only a value entry")
                    }
                }
                let (ptr, len) = scratch[0].ptr_len();
                outputs[0] = ptr;
                outputs[1] = len;
            },
        ),
    })
}

/// The compiled form of `identity`, synthesized by the builder: a slot
/// copy, for every port color except `Ref2`, which
/// [`ref_copy_kit`] carries. The node itself is polymorphic over
/// `Value` and so has no kit of its own; the builder knows the
/// resolved port type and can supply one.
pub(crate) fn identity_op(node: &dyn crate::ast::PolydatNode) -> Option<crate::ast::CompiledU64Op> {
    let meta = node.meta();
    if meta.name != "identity" || meta.outs.len() != 1 {
        return None;
    }
    if meta.outs[0].typ.slot_color() == crate::ast::SlotColor::Ref2 {
        return None;
    }
    Some(Box::new(|inputs: &[u64], outputs: &mut [u64]| {
        outputs.copy_from_slice(inputs)
    }))
}

impl SlotLayout {
    /// Flattened input slot list for one node: every wire source
    /// contributes its full width, in port order.
    fn input_slots(&self, resolved: &ResolvedDag, node_idx: usize) -> Vec<usize> {
        let mut slots = Vec::new();
        for source in &resolved.wiring[node_idx] {
            let (start, w) = match source {
                WireSource::Input(c) => (
                    self.input_starts.get(*c).copied().unwrap_or(*c),
                    resolved
                        .input_defs
                        .get(*c)
                        .map(|d| d.port_type.slot_width())
                        .unwrap_or(1),
                ),
                WireSource::NodeOutput(u, p) => (
                    self.port_offsets[*u][*p],
                    resolved.nodes[*u].meta().outs[*p].typ.slot_width(),
                ),
            };
            slots.extend(start..start + w);
        }
        slots
    }

    /// Flattened output slot list for one node.
    fn output_slots(&self, resolved: &ResolvedDag, node_idx: usize) -> Vec<usize> {
        let mut slots = Vec::new();
        for (p, out) in resolved.nodes[node_idx].meta().outs.iter().enumerate() {
            let start = self.port_offsets[node_idx][p];
            slots.extend(start..start + out.typ.slot_width());
        }
        slots
    }

    /// Output name → first slot of the named port.
    fn named_outputs(&self, resolved: &ResolvedDag) -> HashMap<String, usize> {
        resolved
            .output_map
            .iter()
            .map(|(name, (n, p))| (name.clone(), self.port_offsets[*n][*p]))
            .collect()
    }

    /// Axiom S2: per-slot mask of the slots raw readers must refuse,
    /// over the whole buffer — kernel inputs and node outputs alike.
    /// Both slots of a Ref pair are masked, since their bits are an
    /// address and a length rather than a value; only a typed accessor
    /// or a boundary decode may read them.
    fn ref_slot_mask(&self, resolved: &ResolvedDag) -> Vec<bool> {
        use crate::ast::SlotColor;
        let mut mask = vec![false; self.total_slots];
        let mut mark = |start: usize, color: SlotColor| match color {
            SlotColor::Ref2 => {
                mask[start] = true;
                mask[start + 1] = true;
            }
            SlotColor::Imm1 | SlotColor::Imm2 => {}
        };
        for (i, d) in resolved.input_defs.iter().enumerate() {
            mark(self.input_starts[i], d.port_type.slot_color());
        }
        for (n, node) in resolved.nodes.iter().enumerate() {
            for (p, out) in node.meta().outs.iter().enumerate() {
                mark(self.port_offsets[n][p], out.typ.slot_color());
            }
        }
        mask
    }

    /// First slot of each Ref2-colored output port of one node,
    /// in port order — pairs with the node's `CompiledSlotKit`
    /// scratch entries (axiom S3).
    fn ref_output_starts(&self, resolved: &ResolvedDag, node_idx: usize) -> Vec<usize> {
        resolved.nodes[node_idx]
            .meta()
            .outs
            .iter()
            .enumerate()
            .filter(|(_, out)| out.typ.slot_color() == crate::ast::SlotColor::Ref2)
            .map(|(p, _)| self.port_offsets[node_idx][p])
            .collect()
    }

    /// Expand per-INPUT dependent-step lists to per-SLOT lists so
    /// the kernels' slot-indexed dirty tracking / changed-mask
    /// bits stay coherent under multi-slot inputs (every slot of
    /// one input shares that input's dependents). Identity for
    /// all-scalar inputs.
    fn expand_dependents(&self, resolved: &ResolvedDag, deps: &[Vec<usize>]) -> Vec<Vec<usize>> {
        let mut out = Vec::with_capacity(self.coord_slots);
        for (i, d) in resolved.input_defs.iter().enumerate() {
            for _ in 0..d.port_type.slot_width() {
                out.push(deps.get(i).cloned().unwrap_or_default());
            }
        }
        out
    }
}

/// Builder for assembling a Polydat Kernel programmatically.
pub struct PolydatAssembler {
    /// All input definitions. Coordinates come first (indices 0..coord_count).
    input_defs: Vec<crate::kernel::InputDef>,
    /// How many of the inputs are coordinates.
    coord_count: usize,
    nodes: Vec<PendingNode>,
    /// Output declarations in insertion order.
    output_order: Vec<String>,
    outputs: HashMap<String, WireRef>,
    /// Original source text for diagnostics. Set by the DSL compiler.
    source: String,
    /// Diagnostic context (e.g., "workload.yaml bindings").
    context: String,
    /// Binding modifiers for named outputs.
    output_modifiers: HashMap<String, crate::dsl::ast::BindingModifier>,
    /// Names declared with the `const` keyword. Subject to the
    /// init-binding contract (SRD 11 §"Init Binding Contract").
    const_outputs: std::collections::HashSet<String>,
    /// SRD 15 §"Strict Wire Mode": when true, the resolver
    /// auto-inserts `AssertValue` nodes in front of every wire
    /// input whose declared `Port.constraint` can't be statically
    /// proven satisfied by the source.
    pub(crate) strict_values: bool,
    /// SRD 15: when true, the resolver auto-inserts `AssertType`
    /// nodes in front of wires where the source's runtime variant
    /// can't be statically proven to match the sink's declared
    /// `PortType`. Today this is mainly latent — the type system
    /// already proves variants match for nearly every wire — so
    /// the flag exists for forward compatibility with dynamic
    /// JSON navigation, `Ext` unwraps, and cross-adapter values.
    pub(crate) strict_types: bool,
    /// Strict mode: an implicit type coercion is refused at wire
    /// resolution, and a config wire fed from a cycle-time source, a
    /// nondeterministic node no `volatile` output acknowledges, and a
    /// binding nothing reads are refused at build, on every engine.
    pub(crate) strict: bool,
    /// How much of the interpreter's graph `compile()` fuses into native
    /// cones; `None` is [`JitMode::Auto`](crate::compile::cone::JitMode).
    /// `compile_with(Engine::Interpreter(mode))` takes its mode from the
    /// engine.
    pub(crate) jit_mode: Option<crate::compile::cone::JitMode>,
    /// The compile ledger every program built from this assembler
    /// records in: a fresh one unless the compiler hands down the
    /// tree's.
    pub(crate) ledger: std::sync::Arc<crate::kernel::CompileLedger>,
    /// The cursors the program declares (engines.md §3.5), set
    /// by the DSL compiler so every kernel built from this assembler
    /// knows them.
    cursor_schemas: Vec<crate::iteration::source::SourceSchema>,
}

/// `(coord_slots, total_slots, steps, named outputs, ref-slot
/// mask)` — the Phase-2 compiled layout shared by the closure
/// kernel builders.
type P2Layout = (
    usize,
    usize,
    Vec<crate::compile::closures::P2Step>,
    HashMap<String, usize>,
    Vec<bool>,
    crate::compile::closures::P2Extras,
);

/// `(coord_slots, total_slots, JIT steps, named outputs, scratch,
/// volatile steps)` — the JIT compiled layout shared by the native
/// kernel builders; the scratch is what a state owns for the steps'
/// kits, with each step's entries placed, and the volatile steps are
/// the never-current ones (runtime_model.md, R1.v).
#[cfg(feature = "jit")]
type JitLayout = (
    usize,
    usize,
    Vec<(crate::compile::jit::JitOp, Vec<usize>, Vec<usize>)>,
    HashMap<String, usize>,
    crate::compile::jit::ScratchPlan,
    Vec<usize>,
);

impl PolydatAssembler {
    /// Create a new assembler with the given coordinate names.
    pub fn new(input_names: Vec<String>) -> Self {
        let coord_count = input_names.len();
        let input_defs: Vec<crate::kernel::InputDef> = input_names
            .into_iter()
            .map(|name| crate::kernel::InputDef {
                name,
                default: crate::ast::Value::U64(0),
                port_type: crate::ast::PortType::U64,
                kind: crate::kernel::InputKind::Coordinate,
            })
            .collect();
        Self {
            input_defs,
            coord_count,
            nodes: Vec::new(),
            output_order: Vec::new(),
            outputs: HashMap::new(),
            source: String::new(),
            context: "(assembler)".into(),
            output_modifiers: HashMap::new(),
            const_outputs: std::collections::HashSet::new(),
            strict_values: false,
            strict_types: false,
            strict: false,
            jit_mode: None,
            cursor_schemas: Vec::new(),
            ledger: crate::kernel::CompileLedger::new(),
        }
    }

    /// Record the cursors the program declares, with the partitions the
    /// compiler resolved for each. Every kernel built from this
    /// assembler reports them through `cursor_schemas` and narrows one
    /// through `set_cursor`.
    pub fn set_cursor_schemas(&mut self, schemas: Vec<crate::iteration::source::SourceSchema>) {
        self.cursor_schemas = schemas;
    }

    /// The cursors the program declares.
    pub fn cursor_schemas(&self) -> &[crate::iteration::source::SourceSchema] {
        &self.cursor_schemas
    }

    /// Enable strict-wire-mode auto-insertion of value/type assertion
    /// nodes (SRD 15 §"Strict Wire Mode"). Off by default — the
    /// caller (compiler / DSL pragma extractor) opts in.
    pub fn set_strict_wires(&mut self, strict_types: bool, strict_values: bool) {
        self.strict_types = strict_types;
        self.strict_values = strict_values;
    }

    /// Strict mode, on every engine this assembler builds for: an
    /// implicit type coercion, a config wire fed from a cycle-time
    /// source, a nondeterministic node no `volatile` output
    /// acknowledges, and a binding nothing reads are errors. Off by
    /// default; the DSL sets it from its `strict` option.
    pub fn set_strict(&mut self, strict: bool) {
        self.strict = strict;
    }

    /// Override the engine-mix mode for this compile (SRD-105).
    /// Unset means `JitMode::Auto`.
    pub fn set_jit_mode(&mut self, mode: crate::compile::cone::JitMode) {
        self.jit_mode = Some(mode);
    }

    /// Set the source text and diagnostic context for this assembler.
    /// Called by the DSL compiler to attach the original Polydat source.
    pub fn set_context(&mut self, source: &str, context: &str) {
        self.source = source.to_string();
        self.context = context.to_string();
    }

    /// Add a node to the assembler with the given name and input wiring.
    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        node: Box<dyn PolydatNode>,
        inputs: Vec<WireRef>,
    ) -> &mut Self {
        self.nodes.push(PendingNode {
            name: name.into(),
            node,
            inputs,
        });
        self
    }

    /// Set the binding modifier for a named output.
    pub fn set_output_modifier(&mut self, name: &str, modifier: crate::dsl::ast::BindingModifier) {
        if modifier != crate::dsl::ast::BindingModifier::NONE {
            self.output_modifiers.insert(name.to_string(), modifier);
        }
    }

    /// Mark an output as declared with the `const` keyword. Compile-
    /// time and scope-activation checks (SRD 11 §"Init Binding
    /// Contract") read this set to enforce const-like-constraint
    /// semantics on the binding.
    pub fn mark_const_output(&mut self, name: &str) {
        self.const_outputs.insert(name.to_string());
    }

    /// How many nodes the graph holds so far.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Designate a wire as a named output variate.
    pub fn add_output(&mut self, name: impl Into<String>, wire: WireRef) -> &mut Self {
        let name = name.into();
        if !self.outputs.contains_key(&name) {
            self.output_order.push(name.clone());
        }
        self.outputs.insert(name, wire);
        self
    }

    /// Declare an additional named input.
    ///
    /// Added after coordinate inputs. Nodes wire to it via
    /// `WireRef::input(name)` — same as coordinate inputs.
    /// `kind` controls the lifecycle classification used by the
    /// init-binding contract (see
    /// `crates/polydat/docs/design/evaluation_model.md`
    /// §"Effectively-Const Nodes"): `IterationExtern` for slots
    /// populated by `materialize_wiring_from_outer`, `ExternalWrite` for slots
    /// written by capture extraction.
    pub fn add_input(
        &mut self,
        name: impl Into<String>,
        default: crate::ast::Value,
        port_type: crate::ast::PortType,
        kind: crate::kernel::InputKind,
    ) -> &mut Self {
        self.input_defs.push(crate::kernel::InputDef {
            name: name.into(),
            default,
            port_type,
            kind,
        });
        self
    }

    /// Override a declared input's port type. `new` seeds every
    /// `input_names` entry with `PortType::U64`; this applies the type
    /// from an `input <name>: <type>` declaration. No-op if the input
    /// isn't present.
    pub fn set_input_type(&mut self, name: &str, port_type: crate::ast::PortType) {
        if let Some(d) = self.input_defs.iter_mut().find(|d| d.name == name) {
            d.port_type = port_type;
        }
    }

    /// Return the names of all inputs (coordinates + captures).
    pub fn input_names(&self) -> Vec<&str> {
        self.input_defs.iter().map(|d| d.name.as_str()).collect()
    }

    /// Query the output port type of a named node (first output).
    /// Returns `None` if the node is not found or has no output
    /// ports; callers surface the absence as a loud diagnostic
    /// rather than silently substituting a default.
    pub fn node_output_type(&self, name: &str) -> Option<crate::ast::PortType> {
        self.nodes
            .iter()
            .find(|n| n.name == name)
            .and_then(|n| n.node.meta().outs.first())
            .map(|p| p.typ)
    }

    /// Return the names of declared outputs.
    pub fn output_names(&self) -> Vec<&str> {
        self.outputs.keys().map(|s| s.as_str()).collect()
    }

    /// The node type a named node has, when the name is a node.
    pub fn node_type_of(&self, name: &str) -> Option<String> {
        self.nodes
            .iter()
            .find(|pn| pn.name == name)
            .map(|pn| pn.node.meta().name.clone())
    }

    /// Look up the output port type of a named node.
    ///
    /// Returns the first output port's `PortType` if the node exists.
    pub fn output_type(&self, name: &str) -> Option<PortType> {
        self.nodes
            .iter()
            .find(|pn| pn.name == name)
            .and_then(|pn| pn.node.meta().outs.first())
            .map(|port| port.typ)
    }

    /// Look up the port type of a graph input by name.
    pub fn input_type(&self, name: &str) -> Option<PortType> {
        self.input_defs
            .iter()
            .find(|d| d.name == name)
            .map(|d| d.port_type)
    }

    /// Look up the produced port type of a `WireRef`. Returns `None`
    /// if the wire's source isn't yet known to the assembler (e.g.
    /// it points to a not-yet-added node — a bug in the binding
    /// compiler if it happens).
    pub fn wire_type(&self, wire: &WireRef) -> Option<PortType> {
        match wire {
            WireRef::Input(name) => self.input_type(name),
            WireRef::Node(name, port_idx) => self
                .nodes
                .iter()
                .find(|pn| &pn.name == name)
                .and_then(|pn| pn.node.meta().outs.get(*port_idx))
                .map(|p| p.typ),
        }
    }

    /// Validate, resolve, and produce a Phase 1 runtime kernel.
    pub fn compile(self) -> Result<PolydatKernel, AssemblyError> {
        self.compile_with_log(None)
    }

    /// Compile with diagnostic event logging.
    pub fn compile_with_log(
        self,
        mut log: Option<&mut crate::dsl::events::CompileEventLog>,
    ) -> Result<PolydatKernel, AssemblyError> {
        let jit_mode = self.jit_mode.unwrap_or_default();
        let strict = self.strict;
        let mut resolved = self.resolve_with_log(log.as_deref_mut())?;
        let (node_total, output_total) = (resolved.nodes.len(), resolved.output_order.len());
        crate::compile::cone::extract_jit_cones(&mut resolved, jit_mode);
        let _coord_names = resolved.input_names();
        let modifiers = resolved.output_modifiers.clone();
        let cursors = std::mem::take(&mut resolved.cursor_schemas);
        let mut kernel = PolydatKernel::new_with_inputs(
            resolved.nodes,
            resolved.wiring,
            resolved.input_defs,
            resolved.coord_count,
            resolved.output_map,
            resolved.output_order,
            resolved.const_outputs,
            modifiers,
            &resolved.source,
            &resolved.context,
            log.as_deref_mut(),
            strict,
            resolved.ledger.clone(),
        )?;
        if !cursors.is_empty() {
            kernel.set_cursor_schemas(cursors);
        }
        kernel.set_cone_mode(jit_mode);
        Self::log_summary(log, node_total, output_total);
        Ok(kernel)
    }

    /// Strict mode's build-time refusals on a resolved graph, the ones
    /// the interpreter's fold makes: what a compiled engine checks
    /// before it builds, so strict means the same thing on every engine.
    fn refuse_strict(resolved: &ResolvedDag) -> Result<(), AssemblyError> {
        let classes = PolydatProgram::classify_lifecycle(
            &resolved.nodes,
            &resolved.wiring,
            &resolved.input_defs,
            &resolved.output_map,
            &resolved.output_modifiers,
        );
        let is_init: Vec<bool> = classes
            .lifecycle
            .iter()
            .map(|lc| *lc == crate::kernel::EvalLifecycle::CompileConst)
            .collect();
        match PolydatProgram::strict_violation(
            &resolved.nodes,
            &resolved.wiring,
            &is_init,
            &resolved.output_map,
            &resolved.output_modifiers,
        ) {
            Some(violation) => Err(AssemblyError::Other(violation)),
            None => Ok(()),
        }
    }

    /// A node with no closure form, as a refusal naming the closure
    /// tier and the reason the layout gave.
    fn refused_by_closures(reason: String) -> KernelError {
        KernelError::Refused {
            engine: Engine::Closures(Provenance::Auto),
            reason,
        }
    }

    /// A node native code cannot run, as a refusal naming the native
    /// engine and the reason the layout gave.
    fn refused_by_native(reason: String) -> KernelError {
        KernelError::Refused {
            engine: Engine::Native(Provenance::Auto),
            reason,
        }
    }

    /// The same, naming the pure tier: on `Native` a node without a
    /// native lowering runs its closure, so only the pure tier turns
    /// that into a refusal, and the error should say which engine
    /// refused.
    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    fn refused_by_pure_native(reason: String) -> KernelError {
        KernelError::Refused {
            engine: Engine::PureNative(Provenance::Auto),
            reason,
        }
    }

    /// Shared: extract P2 compiled steps + slot layout from resolved DAG.
    /// Returns None if any node lacks a compiled form.
    fn build_p2_layout(resolved: &ResolvedDag) -> Result<P2Layout, String> {
        let layout = slot_layout(resolved);

        let mut compiled_ops = Vec::with_capacity(resolved.nodes.len());
        let mut extras = crate::compile::closures::P2Extras::default();
        for (node_idx, node) in resolved.nodes.iter().enumerate() {
            compiled_ops.push(
                node_step_op(node.as_ref(), &wire_types_of(resolved, node_idx)).ok_or_else(
                    || {
                        format!(
                            "node '{}' has no compiled form (docs/design/engines.md §8)",
                            node.meta().name
                        )
                    },
                )?,
            );
        }
        extras.externs = crate::compile::externs::Externs::new(
            &resolved.input_defs,
            resolved.coord_count,
            &layout.input_starts,
            &resolved.cursor_schemas,
            &shared_outputs_of(resolved),
            resolved.ledger.clone(),
        )?;
        extras.externs.set_output_names(&resolved.output_order);
        extras
            .externs
            .set_output_modifiers(&resolved.output_modifiers);
        extras.output_types = resolved
            .output_map
            .iter()
            .map(|(name, (n, p))| (name.clone(), resolved.nodes[*n].meta().outs[*p].typ))
            .collect();

        // The runtime model's lifecycle classification, the one rule the
        // interpreter's fold applies, and the provenance the plan is
        // derived from.
        let classes = PolydatProgram::classify_lifecycle(
            &resolved.nodes,
            &resolved.wiring,
            &resolved.input_defs,
            &resolved.output_map,
            &resolved.output_modifiers,
        );
        let inventory = PolydatProgram::compute_node_inventory(&resolved.nodes, &resolved.wiring);
        let per_input = PolydatProgram::compute_dependents(
            &inventory.input_provenance,
            resolved.input_defs.len(),
        );
        extras.input_dependents = layout.expand_dependents(resolved, &per_input);
        extras.attribution = std::sync::Arc::new(Self::attribution_of(resolved));

        let mut steps = Vec::with_capacity(resolved.nodes.len());
        for (node_idx, (op, scratch)) in compiled_ops.into_iter().enumerate() {
            steps.push(crate::compile::closures::P2Step {
                name: resolved.nodes[node_idx].meta().name.clone(),
                op,
                input_slots: layout.input_slots(resolved, node_idx),
                output_slots: layout.output_slots(resolved, node_idx),
                ref_output_starts: layout.ref_output_starts(resolved, node_idx),
                scratch,
                accepts_none: resolved.nodes[node_idx].accepts_none_inputs(),
                volatile: classes.nondeterministic[node_idx],
                constant: classes.lifecycle[node_idx] == crate::kernel::EvalLifecycle::CompileConst,
                side: matches!(
                    resolved.nodes[node_idx].purity(),
                    crate::ast::Purity::SideChannel { .. }
                ),
            });
        }
        let output_map = layout.named_outputs(resolved);
        let ref_slots = layout.ref_slot_mask(resolved);

        Ok((
            layout.coord_slots,
            layout.total_slots,
            steps,
            output_map,
            ref_slots,
            extras,
        ))
    }

    /// Shared: resolve nodes to JIT steps + slot layout.
    #[cfg(feature = "jit")]
    pub(crate) fn build_jit_layout(resolved: &ResolvedDag) -> Result<JitLayout, String> {
        let layout = slot_layout(resolved);

        // Every step's scratch entries are placed in the state's
        // scratch as the steps are laid out (axiom S3): a reference
        // output's pair names its own entry, wherever the step runs.
        let mut scratch = crate::compile::jit::ScratchPlan::default();
        let mut jit_steps = Vec::new();
        for (node_idx, node) in resolved.nodes.iter().enumerate() {
            let mut jit_op = crate::compile::jit::classify_node_typed(
                node.as_ref(),
                &wire_types_of(resolved, node_idx),
            );
            if matches!(jit_op, crate::compile::jit::JitOp::Fallback) {
                return Err(format!(
                    "node '{}' has no native form and no kit; pure native code cannot run it",
                    node.meta().name
                ));
            }
            let base = scratch.elems.len();
            jit_op.place_scratch(base);
            let elems = jit_op.scratch_elems().to_vec();
            scratch.refs.extend(scratch_pairs(
                &node.meta().name,
                &layout.ref_output_starts(resolved, node_idx),
                &elems,
                base,
            ));
            scratch.elems.extend(elems);
            jit_steps.push((
                jit_op,
                layout.input_slots(resolved, node_idx),
                layout.output_slots(resolved, node_idx),
            ));
        }

        let output_map = layout.named_outputs(resolved);
        // The runtime model's lifecycle classification, the one rule the
        // interpreter's fold applies: a nondeterministic node, or one
        // downstream of it, is never current on any engine.
        let classes = PolydatProgram::classify_lifecycle(
            &resolved.nodes,
            &resolved.wiring,
            &resolved.input_defs,
            &resolved.output_map,
            &resolved.output_modifiers,
        );
        let volatile: Vec<usize> = (0..resolved.nodes.len())
            .filter(|&i| classes.nondeterministic[i])
            .collect();
        Ok((
            layout.coord_slots,
            layout.total_slots,
            jit_steps,
            output_map,
            scratch,
            volatile,
        ))
    }

    /// The slots a pure-P3 kernel's raw readers must refuse and the
    /// port type of each named output, for typed decode (SRD 115 §5).
    #[cfg(feature = "jit")]
    fn jit_slot_info(resolved: &ResolvedDag) -> (Vec<bool>, HashMap<String, PortType>) {
        let layout = slot_layout(resolved);
        let guard = layout.ref_slot_mask(resolved);
        let types = resolved
            .output_map
            .iter()
            .map(|(name, (n, p))| (name.clone(), resolved.nodes[*n].meta().outs[*p].typ))
            .collect();
        (guard, types)
    }

    #[cfg(feature = "jit")]
    fn jit_push_pull_from(
        resolved: ResolvedDag,
    ) -> Result<crate::compile::jit::JitKernelPushPull, KernelError> {
        let _coord_names = resolved.input_names();
        let (coord_count, total_slots, jit_steps, output_map, scratch, volatile) =
            Self::build_jit_layout(&resolved).map_err(Self::refused_by_pure_native)?;
        let (guard, types) = Self::jit_slot_info(&resolved);
        let deps = slot_layout(&resolved).expand_dependents(
            &resolved,
            &PolydatProgram::compute_dependents(
                &PolydatProgram::compute_provenance(&resolved.nodes, &resolved.wiring),
                resolved.input_defs.len(),
            ),
        );
        let externs = Self::externs_of(&resolved).map_err(Self::refused_by_pure_native)?;
        let attribution = std::sync::Arc::new(Self::attribution_of(&resolved));
        let (folded, origin) = Self::constant_steps(&resolved, &jit_steps);
        let mut k = crate::compile::jit::compile_jit_push_pull(
            coord_count,
            total_slots,
            jit_steps,
            output_map,
            resolved.nodes,
            deps,
            externs,
            scratch,
            volatile,
        )
        .map_err(Self::refused_by_pure_native)?;
        k.set_slot_info(guard, types);
        k.set_attribution(attribution);
        // After the attribution, so a constant that fails at build names
        // its node as it would at evaluation.
        k.fold_constants(&folded, &origin, total_slots)?;
        Ok(k)
    }

    /// This graph's compile-constant steps, and the program step each
    /// one came from. The closure tier and the hybrid run their
    /// constant steps out of the step list they keep; the pure tier
    /// compiles one function over every step and keeps no list, so its
    /// constants are compiled a second time into an entry of their own
    /// and run once over the kernel's buffer. Same classification as
    /// the other two engines use, from the runtime model's lifecycle.
    #[cfg(feature = "jit")]
    #[allow(clippy::type_complexity)]
    fn constant_steps(
        resolved: &ResolvedDag,
        jit_steps: &[(crate::compile::jit::JitOp, Vec<usize>, Vec<usize>)],
    ) -> (
        Vec<(crate::compile::jit::JitOp, Vec<usize>, Vec<usize>)>,
        Vec<usize>,
    ) {
        let classes = PolydatProgram::classify_lifecycle(
            &resolved.nodes,
            &resolved.wiring,
            &resolved.input_defs,
            &resolved.output_map,
            &resolved.output_modifiers,
        );
        // One step per node, pushed in node order by `build_jit_layout`,
        // so a step's index is its node's.
        jit_steps
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                classes.lifecycle.get(*i) == Some(&crate::kernel::EvalLifecycle::CompileConst)
            })
            .map(|(i, s)| (s.clone(), i))
            .unzip()
    }

    /// The extern inputs of a resolved graph, at the slots the layout
    /// gives them.
    fn externs_of(resolved: &ResolvedDag) -> Result<crate::compile::externs::Externs, String> {
        let layout = slot_layout(resolved);
        let mut externs = crate::compile::externs::Externs::new(
            &resolved.input_defs,
            resolved.coord_count,
            &layout.input_starts,
            &resolved.cursor_schemas,
            &shared_outputs_of(resolved),
            resolved.ledger.clone(),
        )?;
        externs.set_output_names(&resolved.output_order);
        externs.set_output_modifiers(&resolved.output_modifiers);
        Ok(externs)
    }

    /// Pure native code, raw; see [`Self::try_compile_pure_jit`].
    #[doc(hidden)]
    #[cfg(feature = "jit")]
    pub(crate) fn try_compile_pure_jit_raw(
        self,
    ) -> Result<crate::compile::jit::JitKernelRaw, KernelError> {
        let resolved = self.resolve().map_err(KernelError::Assembly)?;
        Self::jit_raw_from(resolved)
    }

    // ── The typed tier constructors (feature `bench-tiers`) ──────
    //
    // The same kernels [`Self::compile_slots`] builds, returned as
    // their own types instead of `Box<dyn SlotKernel>`.
    //
    // There is one contract — [`Kernel`](crate::kernel::Kernel) and the
    // [`SlotKernel`](crate::compile::SlotKernel) that extends it — and
    // these do not add a second. They change only how a caller *holds*
    // it: a boxed kernel dispatches, a named one monomorphizes, and
    // both are bound by the same trait with the same semantics.
    //
    // The normative path is `compile_slots`, which picks an engine from
    // a runtime value and therefore cannot return a statically known
    // type. Only a caller that knows its tier at compile time can use
    // these, and only one kind of caller does: a benchmark measuring a
    // tier, which otherwise measures the dispatch instead of the
    // kernel. On the engine ladder that difference is about a fifth of
    // the native tier's per-cycle cost, which is large enough to hide
    // the regressions the ladder exists to catch.
    //
    // Off by default, so an ordinary build and anything a consumer
    // links has exactly one door to a kernel.

    /// The closure tier with no provenance, as its own type.
    #[cfg(feature = "bench-tiers")]
    pub fn compile_closures_raw(
        self,
    ) -> Result<crate::compile::closures::CompiledKernelRaw, KernelError> {
        let resolved = self.resolve_with_log(None)?;
        let (coord_count, total_slots, steps, output_map, ref_slots, extras) =
            Self::build_p2_layout(&resolved).map_err(Self::refused_by_closures)?;
        crate::compile::closures::CompiledKernelRaw::new(
            coord_count,
            total_slots,
            steps,
            output_map,
            ref_slots,
            extras,
        )
    }

    /// The native tier with no provenance, as its own type.
    #[cfg(all(feature = "bench-tiers", feature = "jit"))]
    pub fn compile_native_raw(
        self,
    ) -> Result<crate::compile::hybrid::HybridKernelRaw, KernelError> {
        let resolved = self.resolve_with_log(None)?;
        Ok(Self::hybrid_from(resolved)?.into_raw())
    }

    /// Pure native code with no provenance, as its own type.
    #[cfg(all(feature = "bench-tiers", feature = "jit"))]
    pub fn compile_pure_native_raw(self) -> Result<crate::compile::jit::JitKernelRaw, KernelError> {
        self.try_compile_pure_jit_raw()
    }

    /// Where each node lives, for the failure path (A7): its name, the
    /// outputs it feeds, and `(first slot, port type)` per input port,
    /// so a compiled kernel can report a step's failure as the
    /// interpreter reports the node's.
    pub(crate) fn attribution_of(resolved: &ResolvedDag) -> crate::compile::Attribution {
        let layout = slot_layout(resolved);
        let sites = resolved
            .nodes
            .iter()
            .enumerate()
            .map(|(node_idx, node)| {
                let mut outputs: Vec<String> = resolved
                    .output_map
                    .iter()
                    .filter(|(_, (n, _))| *n == node_idx)
                    .map(|(name, _)| name.clone())
                    .collect();
                outputs.sort();
                let inputs = resolved.wiring[node_idx]
                    .iter()
                    .map(|source| match source {
                        WireSource::Input(c) => (
                            layout.input_starts.get(*c).copied().unwrap_or(*c),
                            resolved
                                .input_defs
                                .get(*c)
                                .map(|d| d.port_type)
                                .unwrap_or(PortType::U64),
                        ),
                        WireSource::NodeOutput(u, p) => (
                            layout.port_offsets[*u][*p],
                            resolved.nodes[*u].meta().outs[*p].typ,
                        ),
                    })
                    .collect();
                crate::compile::NodeSite {
                    name: node.meta().name.to_string(),
                    outputs,
                    inputs,
                }
            })
            .collect();
        crate::compile::Attribution {
            sites,
            context: resolved.context.clone(),
        }
    }

    #[cfg(feature = "jit")]
    fn jit_raw_from(
        resolved: ResolvedDag,
    ) -> Result<crate::compile::jit::JitKernelRaw, KernelError> {
        let _coord_names = resolved.input_names();
        let (coord_count, total_slots, jit_steps, output_map, scratch, volatile) =
            Self::build_jit_layout(&resolved).map_err(Self::refused_by_pure_native)?;
        let (guard, types) = Self::jit_slot_info(&resolved);
        let externs = Self::externs_of(&resolved).map_err(Self::refused_by_pure_native)?;
        let attribution = std::sync::Arc::new(Self::attribution_of(&resolved));
        let (folded, origin) = Self::constant_steps(&resolved, &jit_steps);
        let mut k = crate::compile::jit::compile_jit_raw_with(
            coord_count,
            total_slots,
            jit_steps,
            output_map,
            resolved.nodes,
            externs,
            scratch,
            volatile,
        )
        .map_err(Self::refused_by_pure_native)?;
        k.set_slot_info(guard, types);
        k.set_attribution(attribution);
        // After the attribution, so a constant that fails at build names
        // its node as it would at evaluation.
        k.fold_constants(&folded, &origin, total_slots)?;
        Ok(k)
    }

    /// Compile the conservative perfect-ordinal Tier-1 SIMD execution plan.
    ///
    /// Ordinary `compile()` semantics are unchanged. This explicit surface
    /// retains the selected scalar DAG as a fallback and synthesizes a second,
    /// register-typed DAG for one named output and driving cursor input.
    #[cfg(feature = "jit")]
    #[doc(hidden)]
    pub fn try_compile_tier1_simd_ordinal(
        self,
        driving_input: &str,
        output: &str,
    ) -> Result<
        crate::compile::simd_tier1::Tier1SimdExecutor,
        crate::compile::simd_tier1::Tier1SimdError,
    > {
        let resolved = self.resolve().map_err(|error| {
            crate::compile::simd_tier1::Tier1SimdError::VectorGraphBuild(error.to_string())
        })?;
        crate::compile::simd_tier1::compile_tier1_ordinal(resolved, driving_input, output)
    }

    fn hybrid_from(
        resolved: ResolvedDag,
    ) -> Result<crate::compile::hybrid::HybridKernel, KernelError> {
        let _coord_names = resolved.input_names();
        let layout = slot_layout(&resolved);

        let output_map = layout.named_outputs(&resolved);
        let input_widths: Vec<usize> = resolved
            .input_defs
            .iter()
            .map(|d| d.port_type.slot_width())
            .collect();

        let ref_slots = layout.ref_slot_mask(&resolved);
        let input_types: Vec<PortType> = resolved.input_defs.iter().map(|d| d.port_type).collect();
        let externs = Self::externs_of(&resolved).map_err(Self::refused_by_native)?;
        let attribution = std::sync::Arc::new(Self::attribution_of(&resolved));
        // The runtime model's lifecycle classification, the one rule the
        // interpreter's fold applies.
        let classes = PolydatProgram::classify_lifecycle(
            &resolved.nodes,
            &resolved.wiring,
            &resolved.input_defs,
            &resolved.output_map,
            &resolved.output_modifiers,
        );
        let constant: Vec<bool> = classes
            .lifecycle
            .iter()
            .map(|lc| *lc == crate::kernel::EvalLifecycle::CompileConst)
            .collect();
        let mut kernel = crate::compile::hybrid::build_hybrid(
            &resolved.nodes,
            &resolved.wiring,
            layout.coord_slots,
            layout.total_slots,
            &layout.port_offsets,
            &layout.input_starts,
            &input_widths,
            output_map,
            ref_slots,
            &input_types,
            externs,
            constant,
            classes.nondeterministic,
            attribution,
        )?;
        kernel.retain_nodes(resolved.nodes);
        Ok(kernel)
    }

    /// Internal: validate, resolve wiring, insert adapters, topological sort.
    /// Report the compiled form each node has
    /// (`CompileEvent::CompileLevelSelected`), a property of the node
    /// and its wire types, so the log is the same on every engine
    /// (engines.md §7): a native form, a compiled `u64` op, a slot
    /// kit, a slot copy, or interpretation only. A node is named by
    /// the output it produces when it produces one.
    fn log_forms(resolved: &ResolvedDag, log: &mut crate::dsl::events::CompileEventLog) {
        for (node_idx, node) in resolved.nodes.iter().enumerate() {
            let wire_types = wire_types_of(resolved, node_idx);
            // Without the `jit` feature there is no native form to
            // report: every node reaches its closure, its slot copy, or
            // interpretation, which the arms below name.
            #[cfg(feature = "jit")]
            let native = !matches!(
                crate::compile::jit::classify_node_typed(node.as_ref(), &wire_types),
                crate::compile::jit::JitOp::Fallback
            );
            #[cfg(not(feature = "jit"))]
            let native = false;
            let level = if native {
                "native"
            } else {
                match node_step_op(node.as_ref(), &wire_types) {
                    Some((crate::compile::closures::StepOp::Copy, _)) => "slot copy",
                    Some((crate::compile::closures::StepOp::U64(_), _)) => "compiled u64 op",
                    Some((crate::compile::closures::StepOp::Slot(_), _)) => "slot kit",
                    None => "interpreted",
                }
            };
            let name = resolved
                .output_map
                .iter()
                .find(|(_, (ni, _))| *ni == node_idx)
                .map(|(n, _)| n.clone())
                .unwrap_or_else(|| node.meta().name.clone());
            log.push(crate::dsl::events::CompileEvent::CompileLevelSelected {
                node: name,
                level: level.to_string(),
            });
        }
    }

    /// Close the log with the program's shape
    /// (`CompileEvent::Summary`): the resolved node and output counts,
    /// the same on every engine, and the constants the build folded,
    /// counted from the log itself.
    fn log_summary(
        log: Option<&mut crate::dsl::events::CompileEventLog>,
        nodes: usize,
        outputs: usize,
    ) {
        if let Some(log) = log {
            let constants_folded = log
                .events()
                .iter()
                .filter(|e| matches!(e, crate::dsl::events::CompileEvent::ConstantFolded { .. }))
                .count();
            log.push(crate::dsl::events::CompileEvent::Summary {
                nodes,
                outputs,
                constants_folded,
            });
        }
    }

    /// Resolve with no log. Only the pure-native paths take it, and
    /// those need code generation, so it is gated as they are.
    #[cfg(feature = "jit")]
    fn resolve(self) -> Result<ResolvedDag, AssemblyError> {
        self.resolve_with_log(None)
    }

    fn resolve_with_log(
        self,
        mut log: Option<&mut crate::dsl::events::CompileEventLog>,
    ) -> Result<ResolvedDag, AssemblyError> {
        // An extern without a default is `None` until the host sets it,
        // and every consumer reads `None` through it; the log names each
        // one so a host knows what it must set (engines.md §3.3).
        // A cursor's slots are `None` until narrowed by design and are
        // not externs a host sets by value.
        if let Some(log) = log.as_deref_mut() {
            let cursor_slot = |name: &str| {
                self.cursor_schemas
                    .iter()
                    .any(|s| name.starts_with(&format!("{}__cursor", s.name)))
            };
            for def in &self.input_defs {
                if matches!(
                    def.kind,
                    crate::kernel::InputKind::ExternalWrite
                        | crate::kernel::InputKind::IterationExtern
                ) && def.default == crate::ast::Value::None
                    && !cursor_slot(&def.name)
                {
                    log.push(crate::dsl::events::CompileEvent::ExternWithoutDefault {
                        name: def.name.clone(),
                        port_type: def.port_type.to_string(),
                    });
                }
            }
        }
        // Build name → index map for nodes
        let mut name_to_idx: HashMap<String, usize> = HashMap::new();
        for (i, pn) in self.nodes.iter().enumerate() {
            if name_to_idx.contains_key(&pn.name) {
                return Err(AssemblyError::DuplicateNode(pn.name.clone()));
            }
            name_to_idx.insert(pn.name.clone(), i);
        }

        // Build input name → index map (covers both coords and captures)
        let input_to_idx: HashMap<String, usize> = self
            .input_defs
            .iter()
            .enumerate()
            .map(|(i, d)| (d.name.clone(), i))
            .collect();

        // Validate arity
        for pn in &self.nodes {
            let expected = pn.node.meta().wire_inputs().len();
            let got = pn.inputs.len();
            if expected != got {
                return Err(AssemblyError::ArityMismatch {
                    node_name: pn.name.clone(),
                    expected,
                    got,
                });
            }
        }

        let mut all_nodes: Vec<PendingNode> = Vec::new();
        let mut all_name_to_idx: HashMap<String, usize> = HashMap::new();
        let mut adapter_count = 0usize;
        let mut assertion_count = 0usize;
        let strict_values = self.strict_values;
        let strict_types = self.strict_types;
        let strict = self.strict;

        for pn in self.nodes {
            let idx = all_nodes.len();
            all_name_to_idx.insert(pn.name.clone(), idx);
            all_nodes.push(pn);
        }

        let mut resolved_wiring: Vec<Vec<WireSource>> = Vec::new();

        for node_idx in 0..all_nodes.len() {
            let mut node_wiring = Vec::new();

            for (port_idx, wire_ref) in all_nodes[node_idx].inputs.clone().iter().enumerate() {
                let port = all_nodes[node_idx].node.meta().wire_inputs()[port_idx].clone();
                let expected_type = port.typ;

                let (source, source_type) = match wire_ref {
                    WireRef::Input(name) => {
                        let input_idx = input_to_idx
                            .get(name)
                            .ok_or_else(|| AssemblyError::UnknownWire(name.clone()))?;
                        let source_type = self.input_defs[*input_idx].port_type;
                        (WireSource::Input(*input_idx), source_type)
                    }
                    WireRef::Node(name, out_port) => {
                        let src_idx = all_name_to_idx
                            .get(name)
                            .ok_or_else(|| AssemblyError::UnknownWire(name.clone()))?;
                        let src_type = all_nodes[*src_idx].node.meta().outs[*out_port].typ;
                        (WireSource::NodeOutput(*src_idx, *out_port), src_type)
                    }
                };

                // A port that takes the wire as it is gets no
                // adapter and no check: converting the value would
                // change what the node reads. The port says so
                // itself (`Port::accepts_any_type`) — this used to be
                // decided from a list of thirteen node names, which
                // disabled the check on every port of those nodes,
                // `pick`'s `Bool` selectors included.
                if port.accepts_any_type || source_type == expected_type {
                    node_wiring.push(source);
                } else if let Some(adapter) = auto_adapter(source_type, expected_type) {
                    if strict {
                        return Err(AssemblyError::Other(format!(
                            "strict mode: implicit type coercion {source_type} → {expected_type} \
                             into '{}'. Use an explicit conversion function (e.g., to_f64, \
                             to_i64, f64_to_u64).",
                            all_nodes[node_idx].name
                        )));
                    }
                    let adapter_name = format!("__adapt_{adapter_count}");
                    adapter_count += 1;
                    let adapter_idx = all_nodes.len();

                    if let Some(ref mut log) = log {
                        let from_name = match wire_ref {
                            WireRef::Input(n) => n.clone(),
                            WireRef::Node(n, _) => n.clone(),
                        };
                        let to_name = all_nodes[node_idx].name.clone();
                        log.push(if is_lossless_widening(source_type, expected_type) {
                            crate::dsl::events::CompileEvent::TypeWidening {
                                from: source_type.to_keyword(),
                                to: expected_type.to_keyword(),
                                context: format!("{from_name} → {to_name}"),
                            }
                        } else {
                            crate::dsl::events::CompileEvent::TypeAdapterInserted {
                                from_node: from_name,
                                to_node: to_name,
                                adapter: format!("{source_type:?}→{expected_type:?}"),
                            }
                        });
                    }

                    all_name_to_idx.insert(adapter_name.clone(), adapter_idx);

                    let adapter_wiring = vec![source];
                    while resolved_wiring.len() <= adapter_idx {
                        resolved_wiring.push(Vec::new());
                    }
                    resolved_wiring[adapter_idx] = adapter_wiring;

                    all_nodes.push(PendingNode {
                        name: adapter_name,
                        node: adapter,
                        inputs: vec![],
                    });

                    node_wiring.push(WireSource::NodeOutput(adapter_idx, 0));
                } else {
                    let from_name = match wire_ref {
                        WireRef::Input(n) => n.clone(),
                        WireRef::Node(n, _) => n.clone(),
                    };
                    return Err(AssemblyError::TypeMismatch {
                        from_node: from_name,
                        from_port: match wire_ref {
                            WireRef::Input(_) => 0,
                            WireRef::Node(_, p) => *p,
                        },
                        from_type: source_type,
                        to_node: all_nodes[node_idx].name.clone(),
                        to_port: port_idx,
                        to_type: expected_type,
                    });
                }

                // === Strict-wire assertion insertion (SRD 15) ===
                //
                // After a wire is resolved (and any type adapter
                // inserted), look at the sink port's declared
                // `constraint`. If strict_values is on, we either
                // prove the source already satisfies it (skip) or
                // splice an `AssertValue` node in front of the
                // sink. The skip cases mirror the four bullets in
                // SRD 15 §"Strict Wire Mode": static type match is
                // already handled by the adapter pass above; here
                // we cover constant sources and upstream-assertion
                // chains for value constraints.
                let sink_port = &all_nodes[node_idx].node.meta().wire_inputs()[port_idx];
                if let Some(constraint) = sink_port.constraint {
                    let last_source = node_wiring.last().expect("wire just pushed").clone();
                    if strict_values
                        && !value_constraint_proven(&all_nodes, &last_source, &constraint)
                    {
                        let assert_name = format!("__assert_v_{assertion_count}");
                        assertion_count += 1;
                        let assert_idx = all_nodes.len();

                        if let Some(ref mut log) = log {
                            let from_name = match wire_ref {
                                WireRef::Input(n) => n.clone(),
                                WireRef::Node(n, _) => n.clone(),
                            };
                            log.push(crate::dsl::events::CompileEvent::AssertionInserted {
                                from_node: from_name,
                                to_node: all_nodes[node_idx].name.clone(),
                                kind: format!("{:?} value-assert {:?}", expected_type, constraint),
                            });
                        }

                        all_name_to_idx.insert(assert_name.clone(), assert_idx);
                        let assert_wiring = vec![last_source];
                        while resolved_wiring.len() <= assert_idx {
                            resolved_wiring.push(Vec::new());
                        }
                        resolved_wiring[assert_idx] = assert_wiring;

                        all_nodes.push(PendingNode {
                            name: assert_name,
                            node: crate::library::assertions::assert_value_node(
                                expected_type,
                                constraint,
                            ),
                            inputs: vec![],
                        });

                        // Replace the just-pushed source with the
                        // assertion's output.
                        *node_wiring.last_mut().unwrap() = WireSource::NodeOutput(assert_idx, 0);
                    } else if let Some(ref mut log) = log {
                        let from_name = match wire_ref {
                            WireRef::Input(n) => n.clone(),
                            WireRef::Node(n, _) => n.clone(),
                        };
                        log.push(crate::dsl::events::CompileEvent::AssertionSkipped {
                            from_node: from_name,
                            to_node: all_nodes[node_idx].name.clone(),
                            reason: assertion_skip_reason(
                                strict_values,
                                &all_nodes,
                                &last_source,
                                &constraint,
                            ),
                        });
                    }
                } else if strict_types && source_type != expected_type {
                    // Type mismatch was already adapted above; the
                    // post-adapter wire is statically the right
                    // type. No assertion needed. Tracking the skip
                    // here is forward-compatible — once dynamic
                    // type cases (JSON nav, Ext unwraps) appear,
                    // this is where the AssertType insertion would
                    // hook in.
                }
            }

            while resolved_wiring.len() <= node_idx {
                resolved_wiring.push(Vec::new());
            }
            resolved_wiring[node_idx] = node_wiring;
        }

        while resolved_wiring.len() < all_nodes.len() {
            resolved_wiring.push(Vec::new());
        }

        // --- Node fusion optimization ---
        //
        // Recognize fusible subgraph patterns and replace them with
        // semantically equivalent fused nodes. See SRD 36.
        {
            let rules = crate::compile::fusion::default_rules();
            if !rules.is_empty() {
                // Collect node indices that are directly referenced by outputs.
                // These nodes must not be consumed as interior nodes by fusion.
                let mut output_nodes: Vec<usize> = Vec::new();
                for wire_ref in self.outputs.values() {
                    if let WireRef::Node(node_name, _) = wire_ref
                        && let Some(&idx) = all_name_to_idx.get(node_name)
                    {
                        output_nodes.push(idx);
                    }
                }

                // Convert to Option<Box<dyn PolydatNode>> for the fusion pass.
                let mut opt_nodes: Vec<Option<Box<dyn PolydatNode>>> =
                    all_nodes.into_iter().map(|pn| Some(pn.node)).collect();

                let fused_count = crate::compile::fusion::apply_fusions(
                    &mut opt_nodes,
                    &mut resolved_wiring,
                    &mut all_name_to_idx,
                    &rules,
                    &output_nodes,
                );
                if fused_count > 0
                    && let Some(ref mut log) = log
                {
                    log.push(crate::dsl::events::CompileEvent::FusionApplied {
                        pattern: "subgraph".into(),
                        nodes_replaced: fused_count,
                    });
                }

                // Convert back, rebuilding PendingNode wrappers.
                // Fused-away nodes (None) get placeholder names.
                all_nodes = opt_nodes
                    .into_iter()
                    .enumerate()
                    .map(|(i, opt)| PendingNode {
                        name: all_name_to_idx
                            .iter()
                            .find(|&(_, &idx)| idx == i)
                            .map(|(n, _)| n.clone())
                            .unwrap_or_else(|| format!("__removed_{i}")),
                        node: opt.unwrap_or_else(|| {
                            Box::new(crate::library::identity::Identity::new(
                                crate::ast::PortType::U64,
                            ))
                        }),
                        inputs: vec![], // wiring is in resolved_wiring
                    })
                    .collect();
            }
        }

        // --- Dead code elimination ---
        //
        // Trace backward from output nodes to find all reachable nodes.
        // Only reachable nodes participate in the topological sort and
        // end up in the final kernel. This prunes unused binding chains
        // when the caller requests a subset of outputs.
        let node_count = all_nodes.len();
        let mut reachable = vec![false; node_count];
        {
            let mut worklist: Vec<usize> = Vec::new();
            // Seed with output nodes
            for wire_ref in self.outputs.values() {
                if let WireRef::Node(node_name, _) = wire_ref
                    && let Some(&idx) = all_name_to_idx.get(node_name)
                {
                    worklist.push(idx);
                }
            }
            // Side-effecting nodes are pinned alive regardless
            // of reachability from a declared output. `log_info`
            // and friends emit one audit-log line per eval as a
            // deliberate side effect — DCE-pruning them would
            // silently drop diagnostic logging the operator
            // explicitly asked for. The set is closed and
            // matched by node-meta name so the marker survives
            // any wiring shape (passthrough, captured-but-unused,
            // synthesised wrapper, etc.).
            for (idx, pn) in all_nodes.iter().enumerate() {
                if matches!(
                    pn.node.meta().name.as_str(),
                    "log_debug" | "log_info" | "log_warn" | "log_error"
                ) {
                    worklist.push(idx);
                }
            }
            // Walk backward through wiring
            while let Some(idx) = worklist.pop() {
                if reachable[idx] {
                    continue;
                }
                reachable[idx] = true;
                for source in &resolved_wiring[idx] {
                    if let WireSource::NodeOutput(upstream, _) = source
                        && !reachable[*upstream]
                    {
                        worklist.push(*upstream);
                    }
                }
            }
        }
        let live_count = reachable.iter().filter(|&&r| r).count();

        // Topological sort (Kahn's algorithm) over reachable nodes only
        let mut in_degree = vec![0usize; node_count];
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); node_count];

        for (node_idx, wiring) in resolved_wiring.iter().enumerate() {
            if !reachable[node_idx] {
                continue;
            }
            for source in wiring {
                if let WireSource::NodeOutput(upstream, _) = source {
                    in_degree[node_idx] += 1;
                    dependents[*upstream].push(node_idx);
                }
            }
        }

        let mut queue: Vec<usize> = (0..node_count)
            .filter(|i| reachable[*i] && in_degree[*i] == 0)
            .collect();
        let mut sorted_order: Vec<usize> = Vec::with_capacity(live_count);

        while let Some(idx) = queue.pop() {
            sorted_order.push(idx);
            for &dep in &dependents[idx] {
                in_degree[dep] -= 1;
                if in_degree[dep] == 0 {
                    queue.push(dep);
                }
            }
        }

        if sorted_order.len() != live_count {
            return Err(AssemblyError::CycleDetected);
        }

        let mut old_to_new = vec![0usize; node_count];
        for (new_idx, &old_idx) in sorted_order.iter().enumerate() {
            old_to_new[old_idx] = new_idx;
        }

        let mut sorted_nodes: Vec<Option<Box<dyn PolydatNode>>> =
            all_nodes.into_iter().map(|pn| Some(pn.node)).collect();

        let final_nodes: Vec<Box<dyn PolydatNode>> = sorted_order
            .iter()
            .map(|&old_idx| sorted_nodes[old_idx].take().unwrap())
            .collect();

        let final_wiring: Vec<Vec<WireSource>> = sorted_order
            .iter()
            .map(|&old_idx| {
                resolved_wiring[old_idx]
                    .iter()
                    .map(|source| match source {
                        WireSource::Input(c) => WireSource::Input(*c),
                        WireSource::NodeOutput(old_up, port) => {
                            WireSource::NodeOutput(old_to_new[*old_up], *port)
                        }
                    })
                    .collect()
            })
            .collect();

        let mut final_output_map: HashMap<String, (usize, usize)> = HashMap::new();
        for (name, wire_ref) in &self.outputs {
            match wire_ref {
                WireRef::Input(coord_name) => {
                    return Err(AssemblyError::UnknownWire(format!(
                        "output '{name}' references coordinate '{coord_name}' directly; \
                         wire through a node instead"
                    )));
                }
                WireRef::Node(node_name, port) => {
                    let old_idx = all_name_to_idx
                        .get(node_name)
                        .ok_or_else(|| AssemblyError::UnknownWire(node_name.clone()))?;
                    final_output_map.insert(name.clone(), (old_to_new[*old_idx], *port));
                }
            }
        }

        // C6b — structural type-round-trip lint (see
        // `compile::roundtrip_lint`): a value modulated `T → Y → … → T`
        // through pure conversion/formatting machinery violates the
        // native-types-stay-native principle. Warning by default; a
        // hard error under strict-values mode, matching the SRD 15
        // strict-wire constraint discipline.
        for f in crate::compile::roundtrip_lint::lint_type_round_trips(
            &final_nodes,
            &final_wiring,
            &self.input_defs,
        ) {
            if strict_values {
                return Err(AssemblyError::Other(f.message()));
            }
            eprintln!("warning: {}", f.message());
            if let Some(ref mut log) = log {
                log.push(crate::dsl::events::CompileEvent::Warning {
                    message: f.message(),
                });
            }
        }

        if let Some(log) = log {
            let resolved_view = ResolvedDag {
                nodes: final_nodes,
                wiring: final_wiring,
                input_defs: self.input_defs,
                coord_count: self.coord_count,
                output_map: final_output_map,
                output_order: self.output_order,
                source: self.source,
                context: self.context,
                output_modifiers: self.output_modifiers,
                const_outputs: self.const_outputs,
                cursor_schemas: self.cursor_schemas,
                ledger: self.ledger,
            };
            Self::log_forms(&resolved_view, log);
            return Ok(resolved_view);
        }
        Ok(ResolvedDag {
            nodes: final_nodes,
            wiring: final_wiring,
            input_defs: self.input_defs,
            coord_count: self.coord_count,
            output_map: final_output_map,
            output_order: self.output_order,
            source: self.source,
            context: self.context,
            output_modifiers: self.output_modifiers,
            const_outputs: self.const_outputs,
            cursor_schemas: self.cursor_schemas,
            ledger: self.ledger,
        })
    }
}

/// Decide whether the source feeding `wire_source` already
/// guarantees the sink's value `constraint` at compile time.
/// Returns `true` if the assertion can be safely skipped.
///
/// Today we recognise two skip cases (SRD 15 §"Strict Wire Mode"):
///
/// 1. **Constant source.** The source node has no wire inputs and
///    its name matches the convention used by `fixed::ConstU64`
///    et al. Const sources have already been validated against
///    their `ParamSpec.constraint` at the factory layer, so any
///    further runtime check would be redundant.
/// 2. **Upstream assertion.** The source is itself an
///    `AssertValue` node (its name starts with `__assert_v_`),
///    which already enforces the same or stronger contract.
fn value_constraint_proven(
    all_nodes: &[PendingNode],
    src: &WireSource,
    _constraint: &crate::dsl::const_constraints::ConstConstraint,
) -> bool {
    match src {
        WireSource::Input(_) => false,
        WireSource::NodeOutput(idx, _) => {
            let meta = all_nodes[*idx].node.meta();
            // Const-source heuristic: a node with no wire inputs
            // is a constant. Today's `ConstU64` / `ConstF64` /
            // `ConstBool` (in `nodes::fixed`) and the synthesised
            // `ConstNode` from compile-time folding both qualify.
            let no_wire_inputs = meta.wire_inputs().is_empty();
            if no_wire_inputs {
                return true;
            }
            // Upstream assertion: skip stacking the same guard.
            // Conservative — any `__assert_v_*` upstream counts as
            // proof. A fancier analysis would compare constraint
            // shapes; for now, idempotency is good enough.
            if meta.name.starts_with("__assert_v_") || meta.name.starts_with("assert_") {
                return true;
            }
            false
        }
    }
}

/// Format the reason a strict-wire assertion was skipped, for the
/// `AssertionSkipped` advisory event. Mirrors the bullets in SRD 15
/// §"Strict Wire Mode" so the log is grep-able.
fn assertion_skip_reason(
    strict_values: bool,
    all_nodes: &[PendingNode],
    src: &WireSource,
    _constraint: &crate::dsl::const_constraints::ConstConstraint,
) -> String {
    if !strict_values {
        return "strict_values not enabled".into();
    }
    match src {
        WireSource::Input(_) => "raw input wire".into(),
        WireSource::NodeOutput(idx, _) => {
            let meta = all_nodes[*idx].node.meta();
            if meta.wire_inputs().is_empty() {
                "constant source already validated".into()
            } else if meta.name.starts_with("__assert_v_") || meta.name.starts_with("assert_") {
                "upstream assertion".into()
            } else {
                "no skip rule matched".into()
            }
        }
    }
}

/// The `shared` bindings of a resolved graph, by name: each is an
/// extern the compiled kernels bind to a cell (engine parity, step 9).
pub(crate) fn shared_outputs_of(resolved: &ResolvedDag) -> Vec<&str> {
    let mut shared: Vec<&str> = resolved
        .output_modifiers
        .iter()
        .filter(|(_, m)| **m == crate::dsl::ast::BindingModifier::SHARED)
        .map(|(name, _)| name.as_str())
        .collect();
    shared.sort();
    shared
}

/// The port type of each wire input of a node, from its sources: the
/// type a compiled lowering sees (SRD 115 §6).
/// Whether the adapter from `from` to `to` is a lossless numeric
/// widening, the class the adapter table lists first: reported as a
/// `TypeWidening`, where every other adapter is a `TypeAdapterInserted`.
fn is_lossless_widening(from: PortType, to: PortType) -> bool {
    use PortType as P;
    matches!(
        (from, to),
        (P::U64, P::F64)
            | (P::U32, P::U64)
            | (P::U32, P::I64)
            | (P::U32, P::F64)
            | (P::I32, P::I64)
            | (P::I32, P::F64)
            | (P::I64, P::F64)
            | (P::F32, P::F64)
    )
}

pub(crate) fn wire_types_of(resolved: &ResolvedDag, node_idx: usize) -> Vec<PortType> {
    resolved.wiring[node_idx]
        .iter()
        .map(|src| match src {
            crate::kernel::WireSource::Input(i) => resolved.input_defs[*i].port_type,
            crate::kernel::WireSource::NodeOutput(j, p) => resolved.nodes[*j].meta().outs[*p].typ,
        })
        .collect()
}

/// The lossless adapter node from one port type to another, if the
/// catalog has one: what the assembler inserts between a wire and a port
/// of different types.
pub fn auto_adapter(from: PortType, to: PortType) -> Option<Box<dyn PolydatNode>> {
    use crate::library::convert::{
        BoolToStr, BoolToU64, F32ToF64, F32ToString, I32ToF64, I32ToI64, I32ToString, I64ToF64,
        I64ToString, U32ToF64, U32ToI64, U32ToString, U32ToU64,
    };
    use crate::library::polyfill as P;
    use crate::library::polyfill_128 as W;
    use crate::library::polyfill_complete as C;
    use crate::library::polyfill_narrow as N;
    match (from, to) {
        // ── Numeric widening (lossless) ─────────────────────────
        (PortType::U64, PortType::F64) => Some(Box::new(U64ToF64::new())),
        (PortType::U32, PortType::U64) => Some(Box::new(U32ToU64::new())),
        (PortType::U32, PortType::I64) => Some(Box::new(U32ToI64::new())),
        (PortType::U32, PortType::F64) => Some(Box::new(U32ToF64::new())),
        (PortType::I32, PortType::I64) => Some(Box::new(I32ToI64::new())),
        (PortType::I32, PortType::F64) => Some(Box::new(I32ToF64::new())),
        // Rounds past 2^24 and never fails, which is class A —
        // totality, not losslessness. The node existed in `polyfill`
        // and the element-wise `VecI32 -> VecF32` below was already
        // auto-inserted; only this wiring was missing, so the scalar
        // of the same two types fell through to a type mismatch.
        (PortType::I32, PortType::F32) => Some(Box::new(P::I32ToF32::new())),
        (PortType::I64, PortType::F64) => Some(Box::new(I64ToF64::new())),
        (PortType::F32, PortType::F64) => Some(Box::new(F32ToF64::new())),

        // ── X → Str (every type renders as a string) ────────────
        (PortType::U64, PortType::Str) => Some(Box::new(U64ToString::new())),
        (PortType::F64, PortType::Str) => Some(Box::new(F64ToString::new())),
        (PortType::Bool, PortType::Str) => Some(Box::new(BoolToStr::new())),
        (PortType::Json, PortType::Str) => Some(Box::new(JsonToStr::new())),
        (PortType::U32, PortType::Str) => Some(Box::new(U32ToString::new())),
        (PortType::I32, PortType::Str) => Some(Box::new(I32ToString::new())),
        (PortType::I64, PortType::Str) => Some(Box::new(I64ToString::new())),
        (PortType::F32, PortType::Str) => Some(Box::new(F32ToString::new())),

        // ── Bool ↔ numeric (always-defined; 1/0 mapping) ────────
        (PortType::Bool, PortType::U64) => Some(Box::new(BoolToU64::new())),
        (PortType::Bool, PortType::U32) => Some(Box::new(P::BoolToU32::new())),
        (PortType::Bool, PortType::I64) => Some(Box::new(P::BoolToI64::new())),
        (PortType::Bool, PortType::I32) => Some(Box::new(P::BoolToI32::new())),
        (PortType::Bool, PortType::F64) => Some(Box::new(P::BoolToF64::new())),
        (PortType::Bool, PortType::F32) => Some(Box::new(P::BoolToF32::new())),
        (PortType::U64, PortType::Bool) => {
            Some(Box::new(crate::library::convert::U64ToBool::new()))
        }
        (PortType::U32, PortType::Bool) => Some(Box::new(P::U32ToBool::new())),
        (PortType::I64, PortType::Bool) => Some(Box::new(P::I64ToBool::new())),
        (PortType::I32, PortType::Bool) => Some(Box::new(P::I32ToBool::new())),
        (PortType::F64, PortType::Bool) => Some(Box::new(P::F64ToBool::new())),
        (PortType::F32, PortType::Bool) => Some(Box::new(P::F32ToBool::new())),

        // ── X → Bytes (little-endian serialize, always-defined) ─
        (PortType::U64, PortType::Bytes) => Some(Box::new(P::U64ToBytes::new())),
        (PortType::U32, PortType::Bytes) => Some(Box::new(P::U32ToBytes::new())),
        (PortType::I64, PortType::Bytes) => Some(Box::new(P::I64ToBytes::new())),
        (PortType::I32, PortType::Bytes) => Some(Box::new(P::I32ToBytes::new())),
        (PortType::F64, PortType::Bytes) => Some(Box::new(P::F64ToBytes::new())),
        (PortType::F32, PortType::Bytes) => Some(Box::new(P::F32ToBytes::new())),
        (PortType::Bool, PortType::Bytes) => Some(Box::new(P::BoolToBytes::new())),
        (PortType::VecF32, PortType::Bytes) => Some(Box::new(P::VecF32ToBytes::new())),
        (PortType::VecI32, PortType::Bytes) => Some(Box::new(P::VecI32ToBytes::new())),

        // ── X → Json (integer / bool wraps; F* and VecF32 are
        //              boundary-only because non-finite floats
        //              aren't representable in JSON) ────────────
        (PortType::U64, PortType::Json) => Some(Box::new(P::U64ToJson::new())),
        (PortType::U32, PortType::Json) => Some(Box::new(P::U32ToJson::new())),
        (PortType::I64, PortType::Json) => Some(Box::new(P::I64ToJson::new())),
        (PortType::I32, PortType::Json) => Some(Box::new(P::I32ToJson::new())),
        (PortType::Bool, PortType::Json) => Some(Box::new(P::BoolToJson::new())),
        (PortType::VecI32, PortType::Json) => Some(Box::new(P::VecI32ToJson::new())),

        // ── Vec ↔ Vec (VecI32 → VecF32 is lossless) ─────────────
        (PortType::VecI32, PortType::VecF32) => Some(Box::new(P::VecI32ToVecF32::new())),

        // ── Narrow cranelift widths (u8/i8/u16/i16/f16) ─────────
        // Lossless widenings + Display renders + Bool maps + LE
        // byte / JSON wraps, mirroring the u32/i32/f32 rows.
        // (type_system_alignment.md §2)
        (PortType::U8, PortType::U64) => Some(Box::new(N::U8ToU64::new())),
        (PortType::U8, PortType::U32) => Some(Box::new(N::U8ToU32::new())),
        (PortType::U8, PortType::U16) => Some(Box::new(N::U8ToU16::new())),
        (PortType::U8, PortType::F64) => Some(Box::new(N::U8ToF64::new())),
        (PortType::U16, PortType::U64) => Some(Box::new(N::U16ToU64::new())),
        (PortType::U16, PortType::U32) => Some(Box::new(N::U16ToU32::new())),
        (PortType::U16, PortType::F64) => Some(Box::new(N::U16ToF64::new())),
        (PortType::I8, PortType::I64) => Some(Box::new(N::I8ToI64::new())),
        (PortType::I8, PortType::I32) => Some(Box::new(N::I8ToI32::new())),
        (PortType::I8, PortType::I16) => Some(Box::new(N::I8ToI16::new())),
        (PortType::I8, PortType::F64) => Some(Box::new(N::I8ToF64::new())),
        (PortType::I16, PortType::I64) => Some(Box::new(N::I16ToI64::new())),
        (PortType::I16, PortType::I32) => Some(Box::new(N::I16ToI32::new())),
        (PortType::I16, PortType::F64) => Some(Box::new(N::I16ToF64::new())),
        (PortType::F16, PortType::F32) => Some(Box::new(N::F16ToF32::new())),
        (PortType::F16, PortType::F64) => Some(Box::new(N::F16ToF64::new())),
        // Totality fills: unsigned → strictly-larger signed, and
        // narrow int → f32 (exact, magnitude ≤ 2^24). All class A.
        (PortType::U8, PortType::I16) => Some(Box::new(N::U8ToI16::new())),
        (PortType::U8, PortType::I32) => Some(Box::new(N::U8ToI32::new())),
        (PortType::U8, PortType::I64) => Some(Box::new(N::U8ToI64::new())),
        (PortType::U8, PortType::F32) => Some(Box::new(N::U8ToF32::new())),
        (PortType::U16, PortType::I32) => Some(Box::new(N::U16ToI32::new())),
        (PortType::U16, PortType::I64) => Some(Box::new(N::U16ToI64::new())),
        (PortType::U16, PortType::F32) => Some(Box::new(N::U16ToF32::new())),
        (PortType::I8, PortType::F32) => Some(Box::new(N::I8ToF32::new())),
        (PortType::I16, PortType::F32) => Some(Box::new(N::I16ToF32::new())),
        (PortType::U8, PortType::F16) => Some(Box::new(N::U8ToF16::new())),
        (PortType::I8, PortType::F16) => Some(Box::new(N::I8ToF16::new())),
        (PortType::U8, PortType::Str) => Some(Box::new(N::U8ToString::new())),
        (PortType::U16, PortType::Str) => Some(Box::new(N::U16ToString::new())),
        (PortType::I8, PortType::Str) => Some(Box::new(N::I8ToString::new())),
        (PortType::I16, PortType::Str) => Some(Box::new(N::I16ToString::new())),
        (PortType::F16, PortType::Str) => Some(Box::new(N::F16ToString::new())),
        (PortType::Bool, PortType::U8) => Some(Box::new(N::BoolToU8::new())),
        (PortType::Bool, PortType::U16) => Some(Box::new(N::BoolToU16::new())),
        (PortType::Bool, PortType::I8) => Some(Box::new(N::BoolToI8::new())),
        (PortType::Bool, PortType::I16) => Some(Box::new(N::BoolToI16::new())),
        (PortType::Bool, PortType::F16) => Some(Box::new(N::BoolToF16::new())),
        (PortType::U8, PortType::Bool) => Some(Box::new(N::U8ToBool::new())),
        (PortType::U16, PortType::Bool) => Some(Box::new(N::U16ToBool::new())),
        (PortType::I8, PortType::Bool) => Some(Box::new(N::I8ToBool::new())),
        (PortType::I16, PortType::Bool) => Some(Box::new(N::I16ToBool::new())),
        (PortType::F16, PortType::Bool) => Some(Box::new(N::F16ToBool::new())),
        (PortType::U8, PortType::Bytes) => Some(Box::new(N::U8ToBytes::new())),
        (PortType::U16, PortType::Bytes) => Some(Box::new(N::U16ToBytes::new())),
        (PortType::I8, PortType::Bytes) => Some(Box::new(N::I8ToBytes::new())),
        (PortType::I16, PortType::Bytes) => Some(Box::new(N::I16ToBytes::new())),
        (PortType::F16, PortType::Bytes) => Some(Box::new(N::F16ToBytes::new())),
        (PortType::U8, PortType::Json) => Some(Box::new(N::U8ToJson::new())),
        (PortType::U16, PortType::Json) => Some(Box::new(N::U16ToJson::new())),
        (PortType::I8, PortType::Json) => Some(Box::new(N::I8ToJson::new())),
        (PortType::I16, PortType::Json) => Some(Box::new(N::I16ToJson::new())),

        // ── 128-bit integers (cranelift I128) ───────────────────
        // Widenings from the 64-bit carriers, Display renders,
        // LE byte / decimal-string JSON wraps. → f64 mirrors
        // u64→f64's class-A treatment (defined for every input).
        (PortType::U64, PortType::U128) => Some(Box::new(W::U64ToU128::new())),
        (PortType::U64, PortType::I128) => Some(Box::new(W::U64ToI128::new())),
        (PortType::I64, PortType::I128) => Some(Box::new(W::I64ToI128::new())),
        // Totality fills: every ≤64-bit integer widens losslessly
        // into the 128-bit carriers (unsigned → both signednesses,
        // signed → i128), `bool` widens to both, and the nonzero
        // test `128 → bool` is total. All class A.
        (PortType::U8, PortType::U128) => Some(Box::new(W::U8ToU128::new())),
        (PortType::U8, PortType::I128) => Some(Box::new(W::U8ToI128::new())),
        (PortType::U16, PortType::U128) => Some(Box::new(W::U16ToU128::new())),
        (PortType::U16, PortType::I128) => Some(Box::new(W::U16ToI128::new())),
        (PortType::U32, PortType::U128) => Some(Box::new(W::U32ToU128::new())),
        (PortType::U32, PortType::I128) => Some(Box::new(W::U32ToI128::new())),
        (PortType::I8, PortType::I128) => Some(Box::new(W::I8ToI128::new())),
        (PortType::I16, PortType::I128) => Some(Box::new(W::I16ToI128::new())),
        (PortType::I32, PortType::I128) => Some(Box::new(W::I32ToI128::new())),
        (PortType::Bool, PortType::U128) => Some(Box::new(W::BoolToU128::new())),
        (PortType::Bool, PortType::I128) => Some(Box::new(W::BoolToI128::new())),
        (PortType::U128, PortType::Bool) => Some(Box::new(W::U128ToBool::new())),
        (PortType::I128, PortType::Bool) => Some(Box::new(W::I128ToBool::new())),
        (PortType::U128, PortType::F64) => Some(Box::new(W::U128ToF64::new())),
        (PortType::I128, PortType::F64) => Some(Box::new(W::I128ToF64::new())),
        (PortType::U128, PortType::Str) => Some(Box::new(W::U128ToString::new())),
        (PortType::I128, PortType::Str) => Some(Box::new(W::I128ToString::new())),
        (PortType::U128, PortType::Bytes) => Some(Box::new(W::U128ToBytes::new())),
        (PortType::I128, PortType::Bytes) => Some(Box::new(W::I128ToBytes::new())),
        (PortType::U128, PortType::Json) => Some(Box::new(W::U128ToJson::new())),
        (PortType::I128, PortType::Json) => Some(Box::new(W::I128ToJson::new())),

        // ── Register views (free bitcasts) ──────────────────────
        // Any reg→reg pair heals with a zero-cost retag — the
        // materialized "views are free bitcasts" rule
        // (type_system_alignment.md §3).
        (from, to) if crate::library::register_view::is_reg_port(from) => {
            crate::library::register_view::reg_view(to)
        }

        // ── Vector lane completion — class A (total) ────────────
        // Lossless inter-lane widenings, `→ Bytes` serialise, and
        // integer-lane `→ Json`/`→ Str`. See library/polyfill_complete.rs.
        (PortType::VecI8, PortType::VecI16) => Some(Box::new(C::VecI8ToVecI16::new())),
        (PortType::VecI8, PortType::VecI32) => Some(Box::new(C::VecI8ToVecI32::new())),
        (PortType::VecI8, PortType::VecI64) => Some(Box::new(C::VecI8ToVecI64::new())),
        (PortType::VecI8, PortType::VecF16) => Some(Box::new(C::VecI8ToVecF16::new())),
        (PortType::VecI8, PortType::VecF32) => Some(Box::new(C::VecI8ToVecF32::new())),
        (PortType::VecI8, PortType::VecF64) => Some(Box::new(C::VecI8ToVecF64::new())),
        (PortType::VecI16, PortType::VecI32) => Some(Box::new(C::VecI16ToVecI32::new())),
        (PortType::VecI16, PortType::VecI64) => Some(Box::new(C::VecI16ToVecI64::new())),
        (PortType::VecI16, PortType::VecF32) => Some(Box::new(C::VecI16ToVecF32::new())),
        (PortType::VecI16, PortType::VecF64) => Some(Box::new(C::VecI16ToVecF64::new())),
        (PortType::VecI32, PortType::VecI64) => Some(Box::new(C::VecI32ToVecI64::new())),
        (PortType::VecI32, PortType::VecF64) => Some(Box::new(C::VecI32ToVecF64::new())),
        (PortType::VecI64, PortType::VecF64) => Some(Box::new(C::VecI64ToVecF64::new())),
        (PortType::VecF16, PortType::VecF32) => Some(Box::new(C::VecF16ToVecF32::new())),
        (PortType::VecF16, PortType::VecF64) => Some(Box::new(C::VecF16ToVecF64::new())),
        (PortType::VecF32, PortType::VecF64) => Some(Box::new(C::VecF32ToVecF64::new())),
        (PortType::VecF64, PortType::Bytes) => Some(Box::new(C::VecF64ToBytes::new())),
        (PortType::VecI64, PortType::Bytes) => Some(Box::new(C::VecI64ToBytes::new())),
        (PortType::VecF16, PortType::Bytes) => Some(Box::new(C::VecF16ToBytes::new())),
        (PortType::VecI16, PortType::Bytes) => Some(Box::new(C::VecI16ToBytes::new())),
        (PortType::VecI8, PortType::Bytes) => Some(Box::new(C::VecI8ToBytes::new())),
        (PortType::VecI64, PortType::Json) => Some(Box::new(C::VecI64ToJson::new())),
        (PortType::VecI16, PortType::Json) => Some(Box::new(C::VecI16ToJson::new())),
        (PortType::VecI8, PortType::Json) => Some(Box::new(C::VecI8ToJson::new())),
        (PortType::VecI32, PortType::Str) => Some(Box::new(P::VecI32ToStr::new())),
        (PortType::VecI64, PortType::Str) => Some(Box::new(C::VecI64ToStr::new())),
        (PortType::VecI16, PortType::Str) => Some(Box::new(C::VecI16ToStr::new())),
        (PortType::VecI8, PortType::Str) => Some(Box::new(C::VecI8ToStr::new())),

        _ => None,
    }
}

/// Boundary adapter catalog. Consulted by
/// `adapt_boundary_value` when a host-injected scope value
/// crosses into a typed slot. Strictly a superset of
/// [`auto_adapter`]: every intra-graph adapter is also a
/// boundary adapter, plus all the lossy / parseable / shape-
/// checking adapters that can panic on input the assembler
/// can't statically verify.
///
/// Boundary-only adapters fall into four classes:
///
/// - **Numeric narrowings** — `U64→{U32, I64, I32, F32}`,
///   `F64→{U64, U32, I64, I32, F32}`, etc. Range-checked,
///   panic on out-of-range.
/// - **Str → X parsers** — workload-param flow (YAML string
///   interpolations, comma-split iter-values). Panic on
///   unparseable input.
/// - **Bytes → X parsers** — wrong-length panics. Numeric
///   reads expect exactly sizeof(N) bytes; Vec reads expect
///   a multiple of sizeof(element).
/// - **Json → X extractors** — shape mismatch panics
///   (`Json::Array` expected for Vec; `Json::Number` for
///   numerics; etc.).
///
/// Plus a small set of "almost-auto" adapters that the
/// assembler can't promote because they panic on non-finite
/// floats: `F64→Json`, `F32→Json`, `VecF32→Json`,
/// `VecF32→Str`.
///
/// See `polydat/docs/design/type_system.md`.
pub fn boundary_adapter(from: PortType, to: PortType) -> Option<Box<dyn PolydatNode>> {
    if let Some(adapter) = auto_adapter(from, to) {
        return Some(adapter);
    }
    use crate::library::convert::{StrToBool, StrToF64, StrToU64};
    use crate::library::polyfill as P;
    use crate::library::polyfill_128 as W;
    use crate::library::polyfill_complete as C;
    use crate::library::polyfill_narrow as N;
    match (from, to) {
        // ── Numeric narrowings + non-widening casts ─────────────
        (PortType::U64, PortType::U32) => Some(Box::new(P::U64ToU32::new())),
        (PortType::U64, PortType::I64) => Some(Box::new(P::U64ToI64::new())),
        (PortType::U64, PortType::I32) => Some(Box::new(P::U64ToI32::new())),
        (PortType::U64, PortType::F32) => Some(Box::new(P::U64ToF32::new())),
        (PortType::U32, PortType::I32) => Some(Box::new(P::U32ToI32::new())),
        (PortType::U32, PortType::F32) => Some(Box::new(P::U32ToF32::new())),
        (PortType::I64, PortType::U64) => Some(Box::new(P::I64ToU64::new())),
        (PortType::I64, PortType::U32) => Some(Box::new(P::I64ToU32::new())),
        (PortType::I64, PortType::I32) => Some(Box::new(P::I64ToI32::new())),
        (PortType::I64, PortType::F32) => Some(Box::new(P::I64ToF32::new())),
        (PortType::I32, PortType::U64) => Some(Box::new(P::I32ToU64::new())),
        (PortType::I32, PortType::U32) => Some(Box::new(P::I32ToU32::new())),
        (PortType::I32, PortType::F32) => Some(Box::new(P::I32ToF32::new())),
        (PortType::F64, PortType::U64) => Some(Box::new(P::F64ToU64Checked::new())),
        (PortType::F64, PortType::U32) => Some(Box::new(P::F64ToU32::new())),
        (PortType::F64, PortType::I64) => Some(Box::new(P::F64ToI64::new())),
        (PortType::F64, PortType::I32) => Some(Box::new(P::F64ToI32::new())),
        (PortType::F64, PortType::F32) => Some(Box::new(P::F64ToF32::new())),
        (PortType::F32, PortType::U64) => Some(Box::new(P::F32ToU64::new())),
        (PortType::F32, PortType::U32) => Some(Box::new(P::F32ToU32::new())),
        (PortType::F32, PortType::I64) => Some(Box::new(P::F32ToI64::new())),
        (PortType::F32, PortType::I32) => Some(Box::new(P::F32ToI32::new())),

        // ── Str → X parsers (boundary-only: panic on unparseable)
        (PortType::Str, PortType::Bool) => Some(Box::new(StrToBool::new())),
        (PortType::Str, PortType::U64) => Some(Box::new(StrToU64::new())),
        (PortType::Str, PortType::F64) => Some(Box::new(StrToF64::new())),
        (PortType::Str, PortType::U32) => Some(Box::new(P::StrToU32::new())),
        (PortType::Str, PortType::I64) => Some(Box::new(P::StrToI64::new())),
        (PortType::Str, PortType::I32) => Some(Box::new(P::StrToI32::new())),
        (PortType::Str, PortType::F32) => Some(Box::new(P::StrToF32::new())),
        (PortType::Str, PortType::Bytes) => Some(Box::new(P::StrToBytes::new())),
        (PortType::Str, PortType::Json) => Some(Box::new(P::StrToJson::new())),
        (PortType::Str, PortType::VecF32) => Some(Box::new(P::StrToVecF32::new())),
        (PortType::Str, PortType::VecI32) => Some(Box::new(P::StrToVecI32::new())),

        // ── Bytes → X (length-checked, little-endian) ───────────
        (PortType::Bytes, PortType::U64) => Some(Box::new(P::BytesToU64::new())),
        (PortType::Bytes, PortType::U32) => Some(Box::new(P::BytesToU32::new())),
        (PortType::Bytes, PortType::I64) => Some(Box::new(P::BytesToI64::new())),
        (PortType::Bytes, PortType::I32) => Some(Box::new(P::BytesToI32::new())),
        (PortType::Bytes, PortType::F64) => Some(Box::new(P::BytesToF64::new())),
        (PortType::Bytes, PortType::F32) => Some(Box::new(P::BytesToF32::new())),
        (PortType::Bytes, PortType::Bool) => Some(Box::new(P::BytesToBool::new())),
        (PortType::Bytes, PortType::Str) => Some(Box::new(P::BytesToStr::new())),
        (PortType::Bytes, PortType::Json) => Some(Box::new(P::BytesToJson::new())),
        (PortType::Bytes, PortType::VecF32) => Some(Box::new(P::BytesToVecF32::new())),
        (PortType::Bytes, PortType::VecI32) => Some(Box::new(P::BytesToVecI32::new())),

        // ── Json → X (shape-checked) ────────────────────────────
        (PortType::Json, PortType::U64) => Some(Box::new(P::JsonToU64::new())),
        (PortType::Json, PortType::U32) => Some(Box::new(P::JsonToU32::new())),
        (PortType::Json, PortType::I64) => Some(Box::new(P::JsonToI64::new())),
        (PortType::Json, PortType::I32) => Some(Box::new(P::JsonToI32::new())),
        (PortType::Json, PortType::F64) => Some(Box::new(P::JsonToF64::new())),
        (PortType::Json, PortType::F32) => Some(Box::new(P::JsonToF32::new())),
        (PortType::Json, PortType::Bool) => Some(Box::new(P::JsonToBool::new())),
        (PortType::Json, PortType::Bytes) => Some(Box::new(P::JsonToBytes::new())),
        (PortType::Json, PortType::VecF32) => Some(Box::new(P::JsonToVecF32::new())),
        (PortType::Json, PortType::VecI32) => Some(Box::new(P::JsonToVecI32::new())),

        // ── Almost-auto (panic on non-finite floats) ────────────
        (PortType::F64, PortType::Json) => Some(Box::new(P::F64ToJson::new())),
        (PortType::F32, PortType::Json) => Some(Box::new(P::F32ToJson::new())),
        (PortType::VecF32, PortType::Json) => Some(Box::new(P::VecF32ToJson::new())),
        (PortType::VecF32, PortType::Str) => Some(Box::new(P::VecF32ToStr::new())),

        // ── Vec ↔ Vec (lossy round) ─────────────────────────────
        (PortType::VecF32, PortType::VecI32) => Some(Box::new(P::VecF32ToVecI32::new())),

        // ── Narrow cranelift widths (u8/i8/u16/i16/f16) ─────────
        // Range-checked narrowings + parsers + shape-checked
        // extractors, mirroring the u32/i32/f32 rows.
        (PortType::U64, PortType::U8) => Some(Box::new(N::U64ToU8::new())),
        (PortType::U32, PortType::U8) => Some(Box::new(N::U32ToU8::new())),
        (PortType::U16, PortType::U8) => Some(Box::new(N::U16ToU8::new())),
        (PortType::I64, PortType::U8) => Some(Box::new(N::I64ToU8::new())),
        (PortType::F64, PortType::U8) => Some(Box::new(N::F64ToU8::new())),
        (PortType::U64, PortType::U16) => Some(Box::new(N::U64ToU16::new())),
        (PortType::U32, PortType::U16) => Some(Box::new(N::U32ToU16::new())),
        (PortType::I64, PortType::U16) => Some(Box::new(N::I64ToU16::new())),
        (PortType::F64, PortType::U16) => Some(Box::new(N::F64ToU16::new())),
        (PortType::I64, PortType::I8) => Some(Box::new(N::I64ToI8::new())),
        (PortType::I32, PortType::I8) => Some(Box::new(N::I32ToI8::new())),
        (PortType::U64, PortType::I8) => Some(Box::new(N::U64ToI8::new())),
        (PortType::F64, PortType::I8) => Some(Box::new(N::F64ToI8::new())),
        (PortType::I64, PortType::I16) => Some(Box::new(N::I64ToI16::new())),
        (PortType::I32, PortType::I16) => Some(Box::new(N::I32ToI16::new())),
        (PortType::U64, PortType::I16) => Some(Box::new(N::U64ToI16::new())),
        (PortType::F64, PortType::I16) => Some(Box::new(N::F64ToI16::new())),
        (PortType::F64, PortType::F16) => Some(Box::new(N::F64ToF16::new())),
        (PortType::F32, PortType::F16) => Some(Box::new(N::F32ToF16::new())),
        (PortType::U64, PortType::F16) => Some(Box::new(N::U64ToF16::new())),
        (PortType::Str, PortType::U8) => Some(Box::new(N::StrToU8::new())),
        (PortType::Str, PortType::U16) => Some(Box::new(N::StrToU16::new())),
        (PortType::Str, PortType::I8) => Some(Box::new(N::StrToI8::new())),
        (PortType::Str, PortType::I16) => Some(Box::new(N::StrToI16::new())),
        (PortType::Str, PortType::F16) => Some(Box::new(N::StrToF16::new())),
        (PortType::Bytes, PortType::U8) => Some(Box::new(N::BytesToU8::new())),
        (PortType::Bytes, PortType::U16) => Some(Box::new(N::BytesToU16::new())),
        (PortType::Bytes, PortType::I8) => Some(Box::new(N::BytesToI8::new())),
        (PortType::Bytes, PortType::I16) => Some(Box::new(N::BytesToI16::new())),
        (PortType::Bytes, PortType::F16) => Some(Box::new(N::BytesToF16::new())),
        (PortType::Json, PortType::U8) => Some(Box::new(N::JsonToU8::new())),
        (PortType::Json, PortType::U16) => Some(Box::new(N::JsonToU16::new())),
        (PortType::Json, PortType::I8) => Some(Box::new(N::JsonToI8::new())),
        (PortType::Json, PortType::I16) => Some(Box::new(N::JsonToI16::new())),
        (PortType::Json, PortType::F16) => Some(Box::new(N::JsonToF16::new())),
        // f16 → Json panics on non-finite (same as f32 → Json).
        (PortType::F16, PortType::Json) => Some(Box::new(N::F16ToJson::new())),

        // ── 128-bit integers (range-checked / parse / shape) ────
        (PortType::U128, PortType::U64) => Some(Box::new(W::U128ToU64::new())),
        (PortType::I128, PortType::I64) => Some(Box::new(W::I128ToI64::new())),
        (PortType::I64, PortType::U128) => Some(Box::new(W::I64ToU128::new())),
        (PortType::U128, PortType::I128) => Some(Box::new(W::U128ToI128::new())),
        (PortType::I128, PortType::U128) => Some(Box::new(W::I128ToU128::new())),
        (PortType::F64, PortType::U128) => Some(Box::new(W::F64ToU128::new())),
        (PortType::F64, PortType::I128) => Some(Box::new(W::F64ToI128::new())),
        (PortType::Str, PortType::U128) => Some(Box::new(W::StrToU128::new())),
        (PortType::Str, PortType::I128) => Some(Box::new(W::StrToI128::new())),
        (PortType::Bytes, PortType::U128) => Some(Box::new(W::BytesToU128::new())),
        (PortType::Bytes, PortType::I128) => Some(Box::new(W::BytesToI128::new())),
        (PortType::Json, PortType::U128) => Some(Box::new(W::JsonToU128::new())),
        (PortType::Json, PortType::I128) => Some(Box::new(W::JsonToI128::new())),

        // ── Scalar matrix completion (library/polyfill_complete.rs) ──
        // Every remaining scalar→scalar narrowing / cross-sign /
        // float→int / int→narrow-float cell, so the 14×14 scalar
        // block has no `·`. All class B (range-checked, can panic).
        (PortType::U8, PortType::I8) => Some(Box::new(C::U8ToI8::new())),
        (PortType::I8, PortType::U8) => Some(Box::new(C::I8ToU8::new())),
        (PortType::I8, PortType::U16) => Some(Box::new(C::I8ToU16::new())),
        (PortType::I8, PortType::U32) => Some(Box::new(C::I8ToU32::new())),
        (PortType::I8, PortType::U64) => Some(Box::new(C::I8ToU64::new())),
        (PortType::I8, PortType::U128) => Some(Box::new(C::I8ToU128::new())),
        (PortType::U16, PortType::I8) => Some(Box::new(C::U16ToI8::new())),
        (PortType::U16, PortType::I16) => Some(Box::new(C::U16ToI16::new())),
        (PortType::U16, PortType::F16) => Some(Box::new(C::U16ToF16::new())),
        (PortType::I16, PortType::U8) => Some(Box::new(C::I16ToU8::new())),
        (PortType::I16, PortType::I8) => Some(Box::new(C::I16ToI8::new())),
        (PortType::I16, PortType::U16) => Some(Box::new(C::I16ToU16::new())),
        (PortType::I16, PortType::F16) => Some(Box::new(C::I16ToF16::new())),
        (PortType::I16, PortType::U32) => Some(Box::new(C::I16ToU32::new())),
        (PortType::I16, PortType::U64) => Some(Box::new(C::I16ToU64::new())),
        (PortType::I16, PortType::U128) => Some(Box::new(C::I16ToU128::new())),
        (PortType::U32, PortType::I8) => Some(Box::new(C::U32ToI8::new())),
        (PortType::U32, PortType::I16) => Some(Box::new(C::U32ToI16::new())),
        (PortType::U32, PortType::F16) => Some(Box::new(C::U32ToF16::new())),
        (PortType::I32, PortType::U8) => Some(Box::new(C::I32ToU8::new())),
        (PortType::I32, PortType::U16) => Some(Box::new(C::I32ToU16::new())),
        (PortType::I32, PortType::F16) => Some(Box::new(C::I32ToF16::new())),
        (PortType::I32, PortType::U128) => Some(Box::new(C::I32ToU128::new())),
        (PortType::F16, PortType::U8) => Some(Box::new(C::F16ToU8::new())),
        (PortType::F16, PortType::I8) => Some(Box::new(C::F16ToI8::new())),
        (PortType::F16, PortType::U16) => Some(Box::new(C::F16ToU16::new())),
        (PortType::F16, PortType::I16) => Some(Box::new(C::F16ToI16::new())),
        (PortType::F16, PortType::U32) => Some(Box::new(C::F16ToU32::new())),
        (PortType::F16, PortType::I32) => Some(Box::new(C::F16ToI32::new())),
        (PortType::F16, PortType::U64) => Some(Box::new(C::F16ToU64::new())),
        (PortType::F16, PortType::I64) => Some(Box::new(C::F16ToI64::new())),
        (PortType::F16, PortType::U128) => Some(Box::new(C::F16ToU128::new())),
        (PortType::F16, PortType::I128) => Some(Box::new(C::F16ToI128::new())),
        (PortType::F32, PortType::U8) => Some(Box::new(C::F32ToU8::new())),
        (PortType::F32, PortType::I8) => Some(Box::new(C::F32ToI8::new())),
        (PortType::F32, PortType::U16) => Some(Box::new(C::F32ToU16::new())),
        (PortType::F32, PortType::I16) => Some(Box::new(C::F32ToI16::new())),
        (PortType::F32, PortType::U128) => Some(Box::new(C::F32ToU128::new())),
        (PortType::F32, PortType::I128) => Some(Box::new(C::F32ToI128::new())),
        (PortType::I64, PortType::F16) => Some(Box::new(C::I64ToF16::new())),
        (PortType::U128, PortType::U8) => Some(Box::new(C::U128ToU8::new())),
        (PortType::U128, PortType::I8) => Some(Box::new(C::U128ToI8::new())),
        (PortType::U128, PortType::U16) => Some(Box::new(C::U128ToU16::new())),
        (PortType::U128, PortType::I16) => Some(Box::new(C::U128ToI16::new())),
        (PortType::U128, PortType::F16) => Some(Box::new(C::U128ToF16::new())),
        (PortType::U128, PortType::U32) => Some(Box::new(C::U128ToU32::new())),
        (PortType::U128, PortType::I32) => Some(Box::new(C::U128ToI32::new())),
        (PortType::U128, PortType::F32) => Some(Box::new(C::U128ToF32::new())),
        (PortType::U128, PortType::I64) => Some(Box::new(C::U128ToI64::new())),
        (PortType::I128, PortType::U8) => Some(Box::new(C::I128ToU8::new())),
        (PortType::I128, PortType::I8) => Some(Box::new(C::I128ToI8::new())),
        (PortType::I128, PortType::U16) => Some(Box::new(C::I128ToU16::new())),
        (PortType::I128, PortType::I16) => Some(Box::new(C::I128ToI16::new())),
        (PortType::I128, PortType::F16) => Some(Box::new(C::I128ToF16::new())),
        (PortType::I128, PortType::U32) => Some(Box::new(C::I128ToU32::new())),
        (PortType::I128, PortType::I32) => Some(Box::new(C::I128ToI32::new())),
        (PortType::I128, PortType::F32) => Some(Box::new(C::I128ToF32::new())),
        (PortType::I128, PortType::U64) => Some(Box::new(C::I128ToU64::new())),

        // ── Vector lane completion — class B (lossy / checked) ──
        // Inter-lane narrowing + float→int, Bytes/Json/Str decode &
        // parse, float-lane → Json/Str (non-finite panics).
        (PortType::VecI16, PortType::VecI8) => Some(Box::new(C::VecI16ToVecI8::new())),
        (PortType::VecI16, PortType::VecF16) => Some(Box::new(C::VecI16ToVecF16::new())),
        (PortType::VecI32, PortType::VecI8) => Some(Box::new(C::VecI32ToVecI8::new())),
        (PortType::VecI32, PortType::VecI16) => Some(Box::new(C::VecI32ToVecI16::new())),
        (PortType::VecI32, PortType::VecF16) => Some(Box::new(C::VecI32ToVecF16::new())),
        (PortType::VecI64, PortType::VecI8) => Some(Box::new(C::VecI64ToVecI8::new())),
        (PortType::VecI64, PortType::VecI16) => Some(Box::new(C::VecI64ToVecI16::new())),
        (PortType::VecI64, PortType::VecI32) => Some(Box::new(C::VecI64ToVecI32::new())),
        (PortType::VecI64, PortType::VecF16) => Some(Box::new(C::VecI64ToVecF16::new())),
        (PortType::VecI64, PortType::VecF32) => Some(Box::new(C::VecI64ToVecF32::new())),
        (PortType::VecF16, PortType::VecI8) => Some(Box::new(C::VecF16ToVecI8::new())),
        (PortType::VecF16, PortType::VecI16) => Some(Box::new(C::VecF16ToVecI16::new())),
        (PortType::VecF16, PortType::VecI32) => Some(Box::new(C::VecF16ToVecI32::new())),
        (PortType::VecF16, PortType::VecI64) => Some(Box::new(C::VecF16ToVecI64::new())),
        (PortType::VecF32, PortType::VecI8) => Some(Box::new(C::VecF32ToVecI8::new())),
        (PortType::VecF32, PortType::VecI16) => Some(Box::new(C::VecF32ToVecI16::new())),
        (PortType::VecF32, PortType::VecI64) => Some(Box::new(C::VecF32ToVecI64::new())),
        (PortType::VecF32, PortType::VecF16) => Some(Box::new(C::VecF32ToVecF16::new())),
        (PortType::VecF64, PortType::VecI8) => Some(Box::new(C::VecF64ToVecI8::new())),
        (PortType::VecF64, PortType::VecI16) => Some(Box::new(C::VecF64ToVecI16::new())),
        (PortType::VecF64, PortType::VecI32) => Some(Box::new(C::VecF64ToVecI32::new())),
        (PortType::VecF64, PortType::VecI64) => Some(Box::new(C::VecF64ToVecI64::new())),
        (PortType::VecF64, PortType::VecF16) => Some(Box::new(C::VecF64ToVecF16::new())),
        (PortType::VecF64, PortType::VecF32) => Some(Box::new(C::VecF64ToVecF32::new())),
        (PortType::Bytes, PortType::VecF64) => Some(Box::new(C::BytesToVecF64::new())),
        (PortType::Bytes, PortType::VecI64) => Some(Box::new(C::BytesToVecI64::new())),
        (PortType::Bytes, PortType::VecF16) => Some(Box::new(C::BytesToVecF16::new())),
        (PortType::Bytes, PortType::VecI16) => Some(Box::new(C::BytesToVecI16::new())),
        (PortType::Bytes, PortType::VecI8) => Some(Box::new(C::BytesToVecI8::new())),
        (PortType::VecF64, PortType::Json) => Some(Box::new(C::VecF64ToJson::new())),
        (PortType::VecF16, PortType::Json) => Some(Box::new(C::VecF16ToJson::new())),
        (PortType::Json, PortType::VecF64) => Some(Box::new(C::JsonToVecF64::new())),
        (PortType::Json, PortType::VecI64) => Some(Box::new(C::JsonToVecI64::new())),
        (PortType::Json, PortType::VecF16) => Some(Box::new(C::JsonToVecF16::new())),
        (PortType::Json, PortType::VecI16) => Some(Box::new(C::JsonToVecI16::new())),
        (PortType::Json, PortType::VecI8) => Some(Box::new(C::JsonToVecI8::new())),
        (PortType::VecF64, PortType::Str) => Some(Box::new(C::VecF64ToStr::new())),
        (PortType::VecF16, PortType::Str) => Some(Box::new(C::VecF16ToStr::new())),
        (PortType::Str, PortType::VecF64) => Some(Box::new(C::StrToVecF64::new())),
        (PortType::Str, PortType::VecI64) => Some(Box::new(C::StrToVecI64::new())),
        (PortType::Str, PortType::VecF16) => Some(Box::new(C::StrToVecF16::new())),
        (PortType::Str, PortType::VecI16) => Some(Box::new(C::StrToVecI16::new())),
        (PortType::Str, PortType::VecI8) => Some(Box::new(C::StrToVecI8::new())),

        _ => None,
    }
}

// ── The one constructor (engines.md §3.5) ─────────────────

use crate::compile::select::{Engine, KernelError, Provenance};
use crate::kernel::Kernel;

impl PolydatAssembler {
    /// Build a kernel on `engine`: the interpreter, the closure tier,
    /// the hybrid kernel, or pure native code, with the provenance mode
    /// the engine names. Every engine accepts every program the
    /// interpreter accepts, or refuses it with a reason naming the node
    /// or construct ([`KernelError::Refused`]). The older constructors
    /// (`compile`, `try_compile*`, `compile_hybrid`)
    /// remain as aliases of this one for their engine.
    pub fn compile_with(self, engine: Engine) -> Result<Box<dyn Kernel>, KernelError> {
        self.compile_engine_with_log(engine, None)
    }

    /// [`Self::compile_with`] on [`Engine::default`]: compiled code, with
    /// the JIT where the build has it.
    pub fn compile_kernel(self) -> Result<Box<dyn Kernel>, KernelError> {
        self.compile_with(Engine::default())
    }

    /// [`Self::compile_with`] with the compile event log, which
    /// receives the assembly events for every engine.
    pub fn compile_engine_with_log(
        self,
        engine: Engine,
        log: Option<&mut crate::dsl::events::CompileEventLog>,
    ) -> Result<Box<dyn Kernel>, KernelError> {
        match engine {
            // The one engine with no slot surface, and so the one this
            // function builds itself.
            Engine::Interpreter(cones) => {
                let mut asm = self;
                asm.jit_mode = Some(cones);
                Ok(Box::new(asm.compile_with_log(log)?))
            }
            // Every compiled engine is built once, by
            // `compile_slots_with_log`, and upcast for the caller who
            // asked for the ordinary surface. One builder, two views.
            _ => Ok(self.compile_slots_with_log(engine, log)?),
        }
    }

    /// Build on `engine` and keep the slot surface: the same kernel
    /// [`Self::compile_with`] builds, typed as
    /// [`SlotKernel`](crate::compile::SlotKernel) so a caller can read
    /// a buffer slot and evaluate one without boxing a `Value`.
    ///
    /// For testing, measurement and diagnostics, where the layout is
    /// the subject. Normative use is `compile_with`, which returns the
    /// same kernel as `Box<dyn Kernel>`; a `Box<dyn SlotKernel>`
    /// upcasts to one wherever the ordinary surface will do, so a
    /// caller that wants both needs only this call.
    ///
    /// `Engine::Interpreter` is refused, and cannot be anything else:
    /// the interpreter holds typed `Value` buffers and has no slot to
    /// name. Ask for a compiled engine, or use `compile_with` and the
    /// `Kernel` trait, which every engine answers.
    pub fn compile_slots(
        self,
        engine: Engine,
    ) -> Result<Box<dyn crate::compile::SlotKernel>, KernelError> {
        self.compile_slots_with_log(engine, None)
    }

    /// [`Self::compile_slots`] with the compile event log.
    pub fn compile_slots_with_log(
        self,
        engine: Engine,
        mut log: Option<&mut crate::dsl::events::CompileEventLog>,
    ) -> Result<Box<dyn crate::compile::SlotKernel>, KernelError> {
        let refused = |reason: String| KernelError::Refused { engine, reason };
        // A builder names its tier but not the provenance mode it was
        // asked for, which the caller is entitled to see back. Only a
        // refusal is restamped: a fold failure belongs to the program
        // and names no engine at all.
        let asked = |e: KernelError| match e {
            KernelError::Refused { reason, .. } => KernelError::Refused { engine, reason },
            other => other,
        };
        let strict = self.strict;
        match engine {
            Engine::Interpreter(_) => Err(refused(
                "the interpreter has no slot buffer: its buffers are typed `Value`s, so \
                 there is no slot to name. Ask for `closures`, `native` or `pure-native` \
                 for the slot surface, or compile with `compile_with` and drive the \
                 kernel through the `Kernel` trait, which every engine answers."
                    .into(),
            )),
            Engine::Closures(prov) => {
                let resolved = self.resolve_with_log(log.as_deref_mut())?;
                if strict {
                    Self::refuse_strict(&resolved)?;
                }
                let folded = log.is_some().then(|| Self::constant_sites(&resolved));
                let (node_total, output_total) =
                    (resolved.nodes.len(), resolved.output_order.len());
                let kernel = Self::closures_from(resolved, prov).map_err(asked)?;
                Self::log_folded(kernel.as_ref(), folded, log.as_deref_mut());
                Self::log_summary(log, node_total, output_total);
                Ok(kernel)
            }
            // Available in every build. Without the `jit` feature this
            // engine's kernel has no native segment in it and every
            // step is a closure, which `plan()` reports as it reports
            // any other mix; the engine is the kernel architecture, and
            // how much of it got native code is the plan. Refusing here
            // would take a working tier away from an architecture that
            // has no code generator, which is the one place it is most
            // worth keeping every engine that can be built.
            Engine::Native(prov) => {
                {
                    let resolved = self.resolve_with_log(log.as_deref_mut())?;
                    if strict {
                        Self::refuse_strict(&resolved)?;
                    }
                    let folded = log.is_some().then(|| Self::constant_sites(&resolved));
                    let (node_total, output_total) =
                        (resolved.nodes.len(), resolved.output_order.len());
                    // Push without the cone guard has no native kernel
                    // (engines.md §4), so a request for it cannot be
                    // realized. Refuse it rather than build push-pull and
                    // report a mode the caller did not ask for: a kernel's
                    // reported configuration is the one it runs.
                    if prov == Provenance::Push {
                        return Err(refused(
                            "native code has no push-only kernel: push-side invalidation \
                             without the cone guard has no native form. Ask for `pushpull` \
                             for both, `pull` for the guard alone, or `auto` to let the \
                             selector choose; `push` alone is available on the closure tier."
                                .into(),
                        ));
                    }
                    let prov = Self::provenance_for(prov, &resolved);
                    let kernel = Self::hybrid_from(resolved).map_err(asked)?;
                    let kernel: Box<dyn crate::compile::SlotKernel> = match prov {
                        Provenance::Raw => Box::new(kernel.into_raw()),
                        Provenance::Pull => Box::new(kernel.into_pull()),
                        // `provenance_for` resolves `Auto` to `Raw`,
                        // `Pull`, or `PushPull`, and `Push` was refused
                        // above, so this arm is `PushPull` in practice.
                        // It refuses rather than panics if the selector
                        // ever gains a mode with no native kernel.
                        Provenance::PushPull | Provenance::Auto => Box::new(kernel),
                        Provenance::Push => {
                            return Err(refused("native code has no push-only kernel".into()));
                        }
                    };
                    Self::log_folded(kernel.as_ref(), folded, log.as_deref_mut());
                    Self::log_summary(log, node_total, output_total);
                    Ok(kernel)
                }
            }
            Engine::PureNative(prov) => {
                #[cfg(feature = "jit")]
                {
                    let resolved = self.resolve_with_log(log.as_deref_mut())?;
                    if strict {
                        Self::refuse_strict(&resolved)?;
                    }
                    let folded = log.is_some().then(|| Self::constant_sites(&resolved));
                    let (node_total, output_total) =
                        (resolved.nodes.len(), resolved.output_order.len());
                    // Only raw and push+pull have a pure kernel. A named
                    // mode with none is refused rather than silently
                    // answered with another, because a kernel reports
                    // the configuration it runs; `Auto` delegated the
                    // choice, so the selector's pull resolves to
                    // push+pull, whose guard subsumes it.
                    let prov = match prov {
                        Provenance::Auto => match Self::provenance_for(prov, &resolved) {
                            Provenance::Raw => Provenance::Raw,
                            _ => Provenance::PushPull,
                        },
                        named @ (Provenance::Raw | Provenance::PushPull) => named,
                        other => {
                            return Err(refused(format!(
                                "pure native code has no {} kernel: the tier keeps only the \
                                 two forms the differential needs. Ask for `raw` or \
                                 `pushpull`, or `auto` to let the selector choose; every \
                                 mode is available on `native`.",
                                format!("{other:?}").to_lowercase(),
                            )));
                        }
                    };
                    let kernel: Box<dyn crate::compile::SlotKernel> = match prov {
                        Provenance::Raw => Box::new(Self::jit_raw_from(resolved).map_err(asked)?),
                        _ => Box::new(Self::jit_push_pull_from(resolved).map_err(asked)?),
                    };
                    Self::log_folded(kernel.as_ref(), folded, log.as_deref_mut());
                    Self::log_summary(log, node_total, output_total);
                    Ok(kernel)
                }
                #[cfg(not(feature = "jit"))]
                {
                    let _ = (prov, log);
                    Err(refused(
                        "this build has no native code (the `jit` feature is off)".into(),
                    ))
                }
            }
        }
    }

    /// The nodes the compile-constant fold applies to, as the
    /// interpreter's fold selects them: no input reaches the node and it
    /// has one output; with the slot and type to read once the kernel is
    /// built.
    fn constant_sites(resolved: &ResolvedDag) -> Vec<(String, usize, crate::ast::PortType)> {
        let classes = PolydatProgram::classify_lifecycle(
            &resolved.nodes,
            &resolved.wiring,
            &resolved.input_defs,
            &resolved.output_map,
            &resolved.output_modifiers,
        );
        let layout = slot_layout(resolved);
        resolved
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| {
                classes.lifecycle[*i] == crate::kernel::EvalLifecycle::CompileConst
                    && n.meta().outs.len() == 1
            })
            .map(|(i, n)| {
                (
                    n.meta().name.clone(),
                    layout.port_offsets[i][0],
                    n.meta().outs[0].typ,
                )
            })
            .collect()
    }

    /// Record the constants the build folded, as the interpreter's fold
    /// records its own: one event per node, with the value it holds.
    fn log_folded(
        kernel: &dyn Kernel,
        sites: Option<Vec<(String, usize, crate::ast::PortType)>>,
        log: Option<&mut crate::dsl::events::CompileEventLog>,
    ) {
        let (Some(sites), Some(log)) = (sites, log) else {
            return;
        };
        for (node, slot, ty) in sites {
            let value = crate::kernel::KernelInternals::slot_value(kernel, slot, ty);
            if !matches!(value, crate::ast::Value::None) {
                log.push(crate::dsl::events::CompileEvent::ConstantFolded {
                    node,
                    value: value.to_display_string(),
                });
            }
        }
    }

    /// The provenance mode a compiled engine builds for `prov`: `Auto`
    /// is the selector's choice from the resolved graph's shape
    /// ([`select::select_prov_mode`]), on the closure tier and the
    /// native engine alike; a named mode is taken as given.
    fn provenance_for(prov: Provenance, resolved: &ResolvedDag) -> Provenance {
        match prov {
            Provenance::Auto => {
                let analysis =
                    select::analyze_graph(&resolved.nodes, &resolved.wiring, &resolved.output_map);
                match select::select_prov_mode(&analysis) {
                    ProvMode::Raw => Provenance::Raw,
                    ProvMode::Pull => Provenance::Pull,
                    ProvMode::PushPull => Provenance::PushPull,
                }
            }
            p => p,
        }
    }

    /// The closure-tier kernel of a resolved graph in one provenance
    /// mode, or why the closure tier refuses the graph.
    fn closures_from(
        resolved: ResolvedDag,
        prov: Provenance,
    ) -> Result<Box<dyn crate::compile::SlotKernel>, KernelError> {
        let prov = Self::provenance_for(prov, &resolved);
        let (coord_count, total_slots, steps, output_map, ref_slots, extras) =
            Self::build_p2_layout(&resolved).map_err(Self::refused_by_closures)?;
        let dependents = || {
            slot_layout(&resolved).expand_dependents(
                &resolved,
                &PolydatProgram::compute_dependents(
                    &PolydatProgram::compute_provenance(&resolved.nodes, &resolved.wiring),
                    resolved.input_defs.len(),
                ),
            )
        };
        Ok(match prov {
            Provenance::Raw => Box::new(CompiledKernelRaw::new(
                coord_count,
                total_slots,
                steps,
                output_map,
                ref_slots,
                extras,
            )?),
            Provenance::Push => Box::new(CompiledKernelPush::new(
                coord_count,
                total_slots,
                steps,
                output_map,
                dependents(),
                ref_slots,
                extras,
            )?),
            Provenance::Pull => Box::new(CompiledKernelPull::new(
                coord_count,
                total_slots,
                steps,
                output_map,
                &dependents(),
                ref_slots,
                extras,
            )?),
            Provenance::PushPull | Provenance::Auto => Box::new(CompiledKernelPushPull::new(
                coord_count,
                total_slots,
                steps,
                output_map,
                dependents(),
                ref_slots,
                extras,
            )?),
        })
    }
}
