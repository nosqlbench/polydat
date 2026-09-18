// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Polydat compiler diagnostic event stream.
//!
//! The compiler emits typed events for each step: binding
//! resolution, module inlining, type adaptation, constant folding,
//! fusion, and compilation level selection: what `explain` retells.
//!
//! Events are tagged with severity levels:
//! - **Info**: normal compilation steps (parsed, resolved, folded)
//! - **Advisory**: type coercions, widenings, and implicit conversions
//!   that the user should be aware of for module design quality
//! - **Warning**: potential performance or correctness issues
//! - **Error**: compilation failures (surfaced as Result::Err, not events)

/// Severity level for compiler diagnostic events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventLevel {
    /// Normal compilation step — informational only.
    Info,
    /// Design advisory — implicit conversion or coercion that the user
    /// should review for module quality. Surfaced by the binary's
    /// `--stats` and `explain`.
    Advisory,
    /// Potential performance or correctness issue.
    Warning,
}

/// A diagnostic event from the Polydat compilation pipeline.
#[derive(Debug, Clone)]
pub enum CompileEvent {
    /// DSL source parsed into AST.
    Parsed {
        /// Top-level statements in the file.
        statements: usize,
    },
    /// A binding was resolved from DSL to a node.
    BindingResolved {
        /// The binding's name.
        name: String,
        /// The node type it resolved to.
        node_type: String,
    },
    /// A module was loaded and inlined.
    ModuleInlined {
        /// The module's name.
        name: String,
        /// Nodes the inlining added to the graph.
        nodes_added: usize,
    },
    /// Type adapter inserted between mismatched ports.
    TypeAdapterInserted {
        /// The producing node.
        from_node: String,
        /// The consuming node.
        to_node: String,
        /// The adapter node inserted between them.
        adapter: String,
    },
    /// Init-time constant folded (SRD 44).
    ConstantFolded {
        /// The node folded.
        node: String,
        /// The constant's rendered value.
        value: String,
    },
    /// Fusion pattern matched and applied (SRD 36).
    FusionApplied {
        /// The fusion pattern's name.
        pattern: String,
        /// Nodes the fused node replaced.
        nodes_replaced: usize,
    },
    /// Output declared.
    OutputDeclared {
        /// The output's name.
        name: String,
    },
    /// Compilation level selected for a node.
    CompileLevelSelected {
        /// The node.
        node: String,
        /// The level's name.
        level: String,
    },
    /// Config wire connected to a cycle-time source (performance warning).
    ConfigWireCycleWarning {
        /// The consuming node.
        node: String,
        /// The config port fed by a cycle-time source.
        port: String,
    },
    /// Auto-widening type coercion inserted by the compiler.
    TypeWidening {
        /// The source type.
        from: &'static str,
        /// The type widened to.
        to: &'static str,
        /// Where the widening was inserted.
        context: String,
    },
    /// A degenerate composition the comprehension validator warns
    /// about (comprehension_forms.md §5.8): the comprehension
    /// compiles and runs; a strict compile refuses it instead.
    ComprehensionWarning {
        /// The comprehension text as written.
        source: String,
        /// Line of the statement that names it.
        line: usize,
        /// Column of the statement that names it.
        col: usize,
        /// The warning, rendered.
        warning: String,
    },
    /// Warning during compilation.
    Warning {
        /// The warning text.
        message: String,
    },
    /// An extern with no default: `None` until the host sets it, and
    /// every consumer reads `None` through it (engines.md §3.3).
    ExternWithoutDefault {
        /// The extern's name.
        name: String,
        /// Its declared type.
        port_type: String,
    },
    /// Summary of the compiled program.
    Summary {
        /// Nodes in the compiled graph.
        nodes: usize,
        /// Declared outputs.
        outputs: usize,
        /// Init-time constants folded.
        constants_folded: usize,
    },
    /// A module-level pragma was acknowledged. Recorded once per
    /// recognised `// @pragma: <name>` directive at the top of the
    /// source. Lets the binary's `--stats` and `explain` show which
    /// graph transforms the module asked for.
    PragmaAcknowledged {
        /// The pragma's name.
        name: String,
        /// The source line it appears on.
        line: usize,
    },
    /// An unrecognised module-level pragma was seen. Pragmas are
    /// forward-compatible: an old binary parses a newer module
    /// that opts into features it doesn't support, and the only
    /// effect is this advisory.
    UnknownPragma {
        /// The pragma's name.
        name: String,
        /// The source line it appears on.
        line: usize,
    },
    /// Strict-wire mode auto-inserted an assertion node between
    /// `from_node` and `to_node`. SRD 15 §"Strict Wire Mode".
    AssertionInserted {
        /// The producing node.
        from_node: String,
        /// The consuming node.
        to_node: String,
        /// The assertion kind inserted.
        kind: String,
    },
    /// Strict-wire mode considered inserting an assertion but
    /// proved it redundant. The reason field names which skip
    /// rule applied (constant source, upstream assertion, etc.).
    AssertionSkipped {
        /// The producing node.
        from_node: String,
        /// The consuming node.
        to_node: String,
        /// The skip rule that applied.
        reason: String,
    },
    /// A tile hole was typed (SRD 114 §4): its expression, the wire
    /// type the compiler inferred, the declared type if any, the
    /// contextual expectation of its position, the encoder chosen, and
    /// the adapter inserted between wire and declared type if one was.
    TileHoleTyped {
        /// The tile's name.
        tile: String,
        /// The hole's expression text.
        hole: String,
        /// The wire type the compiler inferred.
        wire_type: String,
        /// The declared type, if any.
        declared: Option<String>,
        /// The contextual expectation of the hole's position.
        expectation: String,
        /// The encoder chosen.
        encoder: String,
        /// The adapter inserted between wire and declared type, if any.
        adapter: Option<String>,
    },
    /// A tile's skeleton (SRD 114 §6, §10): how many static runs it
    /// copies and their byte total, its holes, branches, and
    /// projections, and the source of each projection body program.
    TileCompiled {
        /// The tile's name.
        tile: String,
        /// The tile's encoding.
        encoding: String,
        /// Static runs the skeleton copies.
        statics: usize,
        /// Their byte total.
        static_bytes: usize,
        /// Holes.
        holes: usize,
        /// Branches.
        branches: usize,
        /// Projections.
        projections: usize,
        /// The source of each projection body program.
        bodies: Vec<String>,
    },
}

impl CompileEvent {
    /// The severity level of this event.
    pub fn level(&self) -> EventLevel {
        match self {
            // Info: normal steps
            CompileEvent::Parsed { .. } => EventLevel::Info,
            CompileEvent::BindingResolved { .. } => EventLevel::Info,
            CompileEvent::ModuleInlined { .. } => EventLevel::Info,
            CompileEvent::OutputDeclared { .. } => EventLevel::Info,
            CompileEvent::CompileLevelSelected { .. } => EventLevel::Info,
            CompileEvent::ConstantFolded { .. } => EventLevel::Info,
            CompileEvent::FusionApplied { .. } => EventLevel::Info,
            CompileEvent::Summary { .. } => EventLevel::Info,
            CompileEvent::TileHoleTyped { adapter: None, .. } => EventLevel::Info,
            CompileEvent::TileCompiled { .. } => EventLevel::Info,

            // Advisory: implicit conversions the user should review
            CompileEvent::TileHoleTyped {
                adapter: Some(_), ..
            } => EventLevel::Advisory,
            CompileEvent::TypeAdapterInserted { .. } => EventLevel::Advisory,
            CompileEvent::TypeWidening { .. } => EventLevel::Advisory,
            CompileEvent::ComprehensionWarning { .. } => EventLevel::Advisory,
            CompileEvent::PragmaAcknowledged { .. } => EventLevel::Advisory,
            CompileEvent::AssertionInserted { .. } => EventLevel::Advisory,
            CompileEvent::AssertionSkipped { .. } => EventLevel::Advisory,

            // Warning: potential issues
            CompileEvent::ConfigWireCycleWarning { .. } => EventLevel::Warning,
            CompileEvent::Warning { .. } => EventLevel::Warning,
            CompileEvent::ExternWithoutDefault { .. } => EventLevel::Warning,
            CompileEvent::UnknownPragma { .. } => EventLevel::Warning,
        }
    }
}

/// Collects diagnostic events during compilation.
#[derive(Debug, Default)]
pub struct CompileEventLog {
    events: Vec<CompileEvent>,
}

impl CompileEventLog {
    /// An empty log.
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Record an event.
    pub fn push(&mut self, event: CompileEvent) {
        self.events.push(event);
    }

    /// Every event recorded, in order.
    pub fn events(&self) -> &[CompileEvent] {
        &self.events
    }

    /// Whether no event has been recorded.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Return only advisory-level events (type coercions, widenings).
    /// These are the "module design quality" messages surfaced by the
    /// binary's `--stats` and `explain`.
    pub fn advisories(&self) -> Vec<&CompileEvent> {
        self.events
            .iter()
            .filter(|e| e.level() == EventLevel::Advisory)
            .collect()
    }

    /// Return only warning-level events.
    pub fn warnings(&self) -> Vec<&CompileEvent> {
        self.events
            .iter()
            .filter(|e| e.level() == EventLevel::Warning)
            .collect()
    }

    /// Format all events as human-readable diagnostic lines.
    /// Each line is prefixed with the severity tag.
    pub fn format(&self) -> String {
        self.events.iter().map(|e| {
            let tag = match e.level() {
                EventLevel::Info => "info",
                EventLevel::Advisory => "advisory",
                EventLevel::Warning => "warning",
            };
            let msg = match e {
            CompileEvent::Parsed { statements } =>
                format!("parsed {statements} statement(s)"),
            CompileEvent::BindingResolved { name, node_type } =>
                format!("resolved '{name}' → {node_type}"),
            CompileEvent::ModuleInlined { name, nodes_added } =>
                format!("module '{name}' inlined ({nodes_added} nodes)"),
            CompileEvent::TypeAdapterInserted { from_node, to_node, adapter } =>
                format!("type adapter {adapter}: {from_node} → {to_node}"),
            CompileEvent::ConstantFolded { node, value } =>
                format!("constant folded: {node} → {value}"),
            CompileEvent::FusionApplied { pattern, nodes_replaced } =>
                format!("fusion: {pattern} ({nodes_replaced} nodes replaced)"),
            CompileEvent::OutputDeclared { name } =>
                format!("output '{name}'"),
            CompileEvent::CompileLevelSelected { node, level } =>
                format!("{node} → {level}"),
            CompileEvent::ConfigWireCycleWarning { node, port } =>
                format!("config wire '{port}' on '{node}' connected to cycle-time source"),
            CompileEvent::ComprehensionWarning { source, line, col, warning } =>
                format!("`for {source}` at line {line}, col {col}: {warning}"),
            CompileEvent::TypeWidening { from, to, context } =>
                format!("widening {from} → {to} in {context}"),
            CompileEvent::Warning { message } =>
                message.to_string(),
            CompileEvent::ExternWithoutDefault { name, port_type } =>
                format!("extern '{name}' ({port_type}) has no default: it is `None` until the host sets it"),
            CompileEvent::Summary { nodes, outputs, constants_folded } =>
                format!("{nodes} nodes, {outputs} outputs, {constants_folded} constant(s) folded"),
            CompileEvent::PragmaAcknowledged { name, line } =>
                format!("pragma '{name}' acknowledged (line {line})"),
            CompileEvent::UnknownPragma { name, line } =>
                format!("unknown pragma '{name}' at line {line}; ignored"),
            CompileEvent::AssertionInserted { from_node, to_node, kind } =>
                format!("assertion inserted: {from_node} → {to_node} ({kind})"),
            CompileEvent::AssertionSkipped { from_node, to_node, reason } =>
                format!("assertion skipped: {from_node} → {to_node} ({reason})"),
            CompileEvent::TileHoleTyped { tile, hole, wire_type, declared, expectation, encoder, adapter } =>
                format!(
                    "tile '{tile}' hole `{hole}`: wire {wire_type}{} expects {expectation}, encoder {encoder}{}",
                    declared.as_ref().map(|d| format!(", declared {d},")).unwrap_or_else(|| ",".to_string()),
                    adapter.as_ref().map(|a| format!(", adapter {a}")).unwrap_or_default()
                ),
            CompileEvent::TileCompiled { tile, encoding, statics, static_bytes, holes, branches, projections, .. } =>
                format!(
                    "tile '{tile}' ({encoding}): {statics} static run(s), {static_bytes} bytes; {holes} hole(s), {branches} branch(es), {projections} projection(s)"
                ),
            };
            format!("polydat[{tag}]: {msg}")
        }).collect::<Vec<_>>().join("\n")
    }
}
