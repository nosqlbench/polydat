// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Polydat runtime kernel: compiled DAG with pull-through evaluation.
//!
//! ## Architecture
//!
//! ```text
//! PolydatProgram (Arc, immutable, shared across all fibers)
//! ┌──────────────────────────────────────────────────────────────┐
//! │  nodes[]         — Box<dyn PolydatNode> in topological order     │
//! │  wiring[]        — per-node input source tables              │
//! │  input_names[]   — graph input dimension names ("cycle")     │
//! │  output_map      — name → (node_idx, port_idx)               │
//! │  (workload params injected as Polydat constant bindings)          │
//! │  ports           — external port definitions (captures)      │
//! └──────────────────────────────────────────────────────────────┘
//!
//! PolydatState (per-fiber, mutable, private — never shared)
//! ┌──────────────────────────────────────────────────────────────┐
//! │  inputs[]            — current input values (e.g., [cycle])  │
//! │  generation          — advances on set_inputs(), used for    │
//! │                        memoization (skip re-evaluation)      │
//! │  node_generation[]   — last-evaluated generation per node    │
//! │  buffers[][]         — per-node output value slots:          │
//! │    ┌───────────┐                                             │
//! │    │ node 0    │ [Value, Value, ...]  (one per output port)  │
//! │    │ node 1    │ [Value]                                     │
//! │    │ node 2    │ [Value, Value]                              │
//! │    │ ...       │                                             │
//! │    └───────────┘                                             │
//! │  port_values[]       — external port values (captures)        │
//! │  port_defaults[]     — initial values for ports              │
//! │  input_scratch[]     — temp buffer for node input gathering  │
//! └──────────────────────────────────────────────────────────────┘
//!
//! Evaluation:
//!   1. fiber.set_inputs(&[cycle])  → state.inputs = [cycle],
//!                                    dirty affected nodes
//!   2. state.pull(program, "name") → walk topologically, skip nodes
//!                                    already evaluated this generation,
//!                                    return &buffers[node][port]
//!
//! Workload params:
//!   Numeric and string workload params are injected into the GK
//!   source as constant bindings before compilation. They resolve
//!   as normal Polydat outputs — no separate globals mechanism needed.
//! ```
//!
//! Buffer layout in PolydatState:
//! ```text
//! coords[0..C) | ports[0..P) | node_buffers[...]
//! ```

pub mod activation;
mod api;
mod api_impl;
pub(crate) mod engines;
pub mod intern;
pub mod interp;
mod manifest;
mod opt;
mod program;
mod scope;
mod state;
pub mod subcontext;
pub use activation::{Activation, CursorSlice, TraversalStream};

pub(crate) use api::SharedKernel;
pub(crate) use api::internals::KernelInternals;
pub use api::{Construction, Dataflow, Kernel, KernelProgram, Metadata, WireKey, WriteError};
pub use engines::*;
pub use intern::{StaticInterner, static_pair};
pub use manifest::{ManifestEntry, extract_manifest};
pub use opt::KernelOptLevel;
pub use program::*;
pub use scope::{ScopeCoord, format_scope_coordinate_path};
pub use state::*;

use crate::ast::Value;

/// Source of a value for a node input port.
#[derive(Debug, Clone)]
pub enum WireSource {
    /// A named input, by index into the unified input array.
    /// Includes both coordinate inputs and capture inputs.
    Input(usize),
    /// Output of another node: `(node_index, output_port_index)`.
    NodeOutput(usize, usize),
}

/// Classification of a named input by its evaluation lifecycle.
///
/// See [evaluation_model.md](../../docs/design/evaluation_model.md) for the lifecycle classification.
/// The init-binding contract uses this to decide whether a wire
/// to an `Input(idx)` is effectively-const at scope-init time:
/// `IterationExtern` slots count as effectively-const (rebound
/// once per scope activation); `Coordinate` and `ExternalWrite`
/// slots are dynamic and disqualify any init binding that reaches
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    /// Dimensional input declared by `input (cycle: u64, ...: u64)` —
    /// dynamic, written at every coordinate advance.
    Coordinate,
    /// External slot populated by `materialize_wiring_from_outer` from an
    /// enclosing `for_each` / `for_combinations` clause —
    /// effectively-const for the duration of one scope activation.
    IterationExtern,
    /// External port declared by `extern name: type = default` —
    /// written by capture extraction during op execution; dynamic
    /// across cycle boundaries within a stanza.
    ExternalWrite,
}

/// Definition of a named input to the Polydat graph.
///
/// All inputs — coordinates, iteration externs, and external-write ports
/// — are defined uniformly. Coordinates default to `Value::U64(0)`,
/// captures default to `Value::None` (unset until a capture writes
/// to them) or to their declared default. The `kind` field carries
/// the lifecycle classification used by the init-binding contract
/// (see SRD 11 §"Init Binding Contract").
#[derive(Debug, Clone)]
pub struct InputDef {
    /// Input name (e.g., "cycle", "username").
    pub name: String,
    /// Default value. Coordinates default to U64(0), captures
    /// to their declared default (or None if unset).
    pub default: Value,
    /// The declared port type for this input. Used by the assembler
    /// for type checking when wiring nodes to this input.
    pub port_type: crate::ast::PortType,
    /// Lifecycle classification. Defaults to `Coordinate` so
    /// existing call sites that construct `InputDef` directly
    /// (tests, legacy paths) keep their previous semantics.
    /// The DSL compiler sets this explicitly for iteration
    /// externs and external-write ports.
    pub kind: InputKind,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn capture_inputs_persist_across_set_inputs() {
        // Program with 1 coordinate (cycle) + 2 capture inputs
        let program = Arc::new(PolydatProgram::with_inputs(
            vec![],
            vec![],
            vec![
                InputDef {
                    name: "cycle".into(),
                    default: Value::U64(0),
                    port_type: crate::ast::PortType::U64,
                    kind: InputKind::Coordinate,
                },
                InputDef {
                    name: "balance".into(),
                    default: Value::F64(0.0),
                    port_type: crate::ast::PortType::F64,
                    kind: InputKind::ExternalWrite,
                },
                InputDef {
                    name: "auth_token".into(),
                    default: Value::Str("anonymous".into()),
                    port_type: crate::ast::PortType::Str,
                    kind: InputKind::ExternalWrite,
                },
            ],
            1, // coord_count
            HashMap::new(),
            Vec::new(),
            "",
            "(test)",
        ));
        let mut state = program.create_state();

        // Default values for capture inputs
        assert_eq!(state.get_input(1), Value::F64(0.0));
        assert_eq!(state.get_input(2), Value::Str("anonymous".into()));

        // Set capture inputs individually
        state.set_input(1, Value::F64(1234.56));
        state.set_input(2, Value::Str("token_abc".into()));
        assert_eq!(state.get_input(1), Value::F64(1234.56));
        assert_eq!(state.get_input(2), Value::Str("token_abc".into()));

        // Capture inputs persist when coordinates change
        state.set_inputs(&[42]);
        assert_eq!(state.get_input(1), Value::F64(1234.56));
        assert_eq!(state.get_input(2), Value::Str("token_abc".into()));
    }

    #[test]
    fn reset_inputs_restores_capture_defaults() {
        let program = Arc::new(PolydatProgram::with_inputs(
            vec![],
            vec![],
            vec![
                InputDef {
                    name: "cycle".into(),
                    default: Value::U64(0),
                    port_type: crate::ast::PortType::U64,
                    kind: InputKind::Coordinate,
                },
                InputDef {
                    name: "token".into(),
                    default: Value::Str("anon".into()),
                    port_type: crate::ast::PortType::Str,
                    kind: InputKind::ExternalWrite,
                },
            ],
            1,
            HashMap::new(),
            Vec::new(),
            "",
            "(test)",
        ));
        let mut state = program.create_state();

        state.set_input(1, Value::Str("alice".into()));
        assert_eq!(state.get_input(1), Value::Str("alice".into()));

        // Reset only capture inputs (from coord_count onward)
        state.reset_inputs_from(1);
        assert_eq!(state.get_input(1), Value::Str("anon".into()));
    }

    #[test]
    fn invalidate_all_keeps_inputs_and_reset_restores_defaults() {
        let program = Arc::new(PolydatProgram::with_inputs(
            vec![],
            vec![],
            vec![
                InputDef {
                    name: "cycle".into(),
                    default: Value::U64(0),
                    port_type: crate::ast::PortType::U64,
                    kind: InputKind::Coordinate,
                },
                InputDef {
                    name: "token".into(),
                    default: Value::Str("anon".into()),
                    port_type: crate::ast::PortType::Str,
                    kind: InputKind::ExternalWrite,
                },
            ],
            1,
            HashMap::new(),
            Vec::new(),
            "",
            "(test)",
        ));
        let mut state = program.create_state();

        state.set_inputs(&[42]);
        state.set_input(1, Value::Str("alice".into()));

        state.invalidate_all();
        assert_eq!(state.get_input(0), Value::U64(42));
        assert_eq!(state.get_input(1), Value::Str("alice".into()));
        state.reset_inputs_from(0);
        assert_eq!(state.get_input(0), Value::U64(0));
        assert_eq!(state.get_input(1), Value::Str("anon".into()));
    }

    // ---------------------------------------------------------------
    // WireCost tests: config wire warnings for various DAG shapes
    // ---------------------------------------------------------------

    /// A test node with one Config wire input and one Data wire input.
    /// Simulates a node with an expensive LUT that's configured by
    /// the first input and driven by the second.
    struct ConfigWireTestNode {
        meta: crate::ast::NodeMeta,
    }

    impl ConfigWireTestNode {
        fn new() -> Self {
            use crate::ast::{Port, Slot};
            Self {
                meta: crate::ast::NodeMeta {
                    name: "config_test".into(),
                    outs: vec![Port::u64("output")],
                    ins: vec![
                        Slot::Wire(Port::u64("config_param").config()),
                        Slot::Wire(Port::u64("data_input")),
                    ],
                },
            }
        }
    }

    impl crate::ast::PolydatNode for ConfigWireTestNode {
        fn meta(&self) -> &crate::ast::NodeMeta {
            &self.meta
        }
        fn eval(&self, inputs: &[Value], outputs: &mut [Value]) {
            let config = inputs[0].as_u64();
            let data = inputs[1].as_u64();
            outputs[0] = Value::U64(config.wrapping_add(data));
        }
    }

    #[test]
    fn wire_cost_no_warning_when_config_is_init_time() {
        // DAG: constant(42) → config_test.config_param
        //      cycle → hash → config_test.data_input
        // Config wire fed by init-time constant → no warning
        use crate::compile::assembly::{PolydatAssembler, WireRef};
        use crate::dsl::events::CompileEventLog;
        use crate::library::identity::ConstU64;
        use crate::library::identity::Identity;

        let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
        asm.add_node("config_val", Box::new(ConstU64::new(42)), vec![]);
        asm.add_node(
            "hashed",
            Box::new(Identity::new(crate::ast::PortType::U64)),
            vec![WireRef::input("cycle")],
        );
        asm.add_node(
            "test_node",
            Box::new(ConfigWireTestNode::new()),
            vec![WireRef::node("config_val"), WireRef::node("hashed")],
        );
        asm.add_output("result", WireRef::node("test_node"));

        let mut log = CompileEventLog::new();
        let k = asm.compile_with_log(Some(&mut log)).unwrap();
        let _program = k.into_program();

        // Check: no ConfigWireCycleWarning in events
        let warnings: Vec<_> = log
            .events()
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    crate::dsl::events::CompileEvent::ConfigWireCycleWarning { .. }
                )
            })
            .collect();
        assert!(
            warnings.is_empty(),
            "no warning expected when config wire is init-time: {warnings:?}"
        );
    }

    #[test]
    fn wire_cost_warning_when_config_is_cycle_time() {
        // DAG: cycle → hash → config_test.config_param  (BAD: config from cycle)
        //      cycle → config_test.data_input
        // Config wire fed by cycle-time node → should warn
        use crate::compile::assembly::{PolydatAssembler, WireRef};
        use crate::dsl::events::CompileEventLog;
        use crate::library::identity::Identity;

        let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
        asm.add_node(
            "hashed",
            Box::new(Identity::new(crate::ast::PortType::U64)),
            vec![WireRef::input("cycle")],
        );
        asm.add_node(
            "test_node",
            Box::new(ConfigWireTestNode::new()),
            vec![
                WireRef::node("hashed"), // config_param ← cycle-time!
                WireRef::input("cycle"), // data_input ← cycle
            ],
        );
        asm.add_output("result", WireRef::node("test_node"));

        let mut log = CompileEventLog::new();
        let _k = asm.compile_with_log(Some(&mut log)).unwrap();

        let warnings: Vec<_> = log
            .events()
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    crate::dsl::events::CompileEvent::ConfigWireCycleWarning { .. }
                )
            })
            .collect();
        assert_eq!(
            warnings.len(),
            1,
            "expected exactly one config wire warning: {warnings:?}"
        );
    }

    #[test]
    fn wire_cost_warning_when_config_is_coordinate_direct() {
        // DAG: cycle → config_test.config_param  (BAD: coordinate direct to config)
        //      cycle → config_test.data_input
        use crate::compile::assembly::{PolydatAssembler, WireRef};
        use crate::dsl::events::CompileEventLog;

        let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
        asm.add_node(
            "test_node",
            Box::new(ConfigWireTestNode::new()),
            vec![
                WireRef::input("cycle"), // config_param ← coordinate!
                WireRef::input("cycle"), // data_input ← cycle
            ],
        );
        asm.add_output("result", WireRef::node("test_node"));

        let mut log = CompileEventLog::new();
        let _k = asm.compile_with_log(Some(&mut log)).unwrap();

        let warnings: Vec<_> = log
            .events()
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    crate::dsl::events::CompileEvent::ConfigWireCycleWarning { .. }
                )
            })
            .collect();
        assert_eq!(warnings.len(), 1, "config wire from coordinate should warn");
    }

    #[test]
    fn wire_cost_no_warning_data_wire_from_cycle() {
        // DAG: constant(10) → config_test.config_param (init-time, ok)
        //      cycle → config_test.data_input           (cycle-time, ok for Data wire)
        // Only the data wire is cycle-time → no warning
        use crate::compile::assembly::{PolydatAssembler, WireRef};
        use crate::dsl::events::CompileEventLog;
        use crate::library::identity::ConstU64;

        let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
        asm.add_node("config_val", Box::new(ConstU64::new(10)), vec![]);
        asm.add_node(
            "test_node",
            Box::new(ConfigWireTestNode::new()),
            vec![
                WireRef::node("config_val"), // config_param ← constant
                WireRef::input("cycle"),     // data_input ← cycle (Data wire, ok)
            ],
        );
        asm.add_output("result", WireRef::node("test_node"));

        let mut log = CompileEventLog::new();
        let _k = asm.compile_with_log(Some(&mut log)).unwrap();

        let warnings: Vec<_> = log
            .events()
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    crate::dsl::events::CompileEvent::ConfigWireCycleWarning { .. }
                )
            })
            .collect();
        assert!(warnings.is_empty(), "data wire from cycle should not warn");
    }

    #[test]
    fn wire_cost_diamond_config_from_init() {
        // Diamond DAG using two ConfigWireTestNodes:
        //   constant(5) → inner.config_param ─┐
        //   constant(3) → inner.data_input    ─┤→ inner.output → outer.config_param
        //   cycle → hash → outer.data_input
        // inner is fully init-time → its output feeds outer's config wire → no warning
        use crate::compile::assembly::{PolydatAssembler, WireRef};
        use crate::dsl::events::CompileEventLog;
        use crate::library::identity::ConstU64;
        use crate::library::identity::Identity;

        let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
        asm.add_node("a", Box::new(ConstU64::new(5)), vec![]);
        asm.add_node("b", Box::new(ConstU64::new(3)), vec![]);
        asm.add_node(
            "inner",
            Box::new(ConfigWireTestNode::new()),
            vec![WireRef::node("a"), WireRef::node("b")],
        );
        asm.add_node(
            "hashed",
            Box::new(Identity::new(crate::ast::PortType::U64)),
            vec![WireRef::input("cycle")],
        );
        asm.add_node(
            "outer",
            Box::new(ConfigWireTestNode::new()),
            vec![
                WireRef::node("inner"),  // config_param ← init-time (5+3)
                WireRef::node("hashed"), // data_input ← cycle-time
            ],
        );
        asm.add_output("result", WireRef::node("outer"));

        let mut log = CompileEventLog::new();
        let _k = asm.compile_with_log(Some(&mut log)).unwrap();

        // inner's config wire from constant is fine. outer's config wire
        // from init-time inner output is also fine.
        let warnings: Vec<_> = log
            .events()
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    crate::dsl::events::CompileEvent::ConfigWireCycleWarning { .. }
                )
            })
            .collect();
        assert!(
            warnings.is_empty(),
            "init-time derived config should not warn: {warnings:?}"
        );
    }

    #[test]
    fn wire_cost_diamond_config_from_mixed() {
        // Mixed init/cycle feeding config:
        //   constant(5) → mixer.config_param ─┐
        //   cycle → mixer.data_input          ─┤→ mixer.output → outer.config_param
        //   cycle → outer.data_input
        // mixer depends on cycle → its output is cycle-time → outer's config wire warns
        use crate::compile::assembly::{PolydatAssembler, WireRef};
        use crate::dsl::events::CompileEventLog;
        use crate::library::identity::ConstU64;

        let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
        asm.add_node("five", Box::new(ConstU64::new(5)), vec![]);
        asm.add_node(
            "mixer",
            Box::new(ConfigWireTestNode::new()),
            vec![
                WireRef::node("five"),   // config_param ← init
                WireRef::input("cycle"), // data_input ← cycle
            ],
        );
        asm.add_node(
            "outer",
            Box::new(ConfigWireTestNode::new()),
            vec![
                WireRef::node("mixer"),  // config_param ← cycle-tainted!
                WireRef::input("cycle"), // data_input
            ],
        );
        asm.add_output("result", WireRef::node("outer"));

        let mut log = CompileEventLog::new();
        let _k = asm.compile_with_log(Some(&mut log)).unwrap();

        let warnings: Vec<_> = log
            .events()
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    crate::dsl::events::CompileEvent::ConfigWireCycleWarning { .. }
                )
            })
            .collect();
        // outer's config from cycle-tainted mixer should warn.
        // mixer's config from constant should NOT warn.
        assert_eq!(
            warnings.len(),
            1,
            "exactly one warning for outer's config: {warnings:?}"
        );
    }
}
