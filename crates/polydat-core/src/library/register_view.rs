// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The register view retag adapter the assembler inserts between
//! register ports of different lane typings, and the register-port
//! predicate it keys on (type_system_alignment.md §8.4 layer 2).

use crate::ast::{NodeMeta, PolydatNode, Port, PortType, RegLanes, Slot, Value};

/// Pass-through guard that retags a register word's view. The
/// bits are untouched — this is the materialized form of "views
/// are free bitcasts" for intra-graph wires whose producer and
/// consumer declare different lane typings. Auto-inserted by
/// `compile::assembly::auto_adapter` for every reg→reg pair;
/// rarely instantiated by hand.
pub struct RegView {
    meta: NodeMeta,
    to: RegLanes,
}

impl RegView {
    /// A view of a register word as the given register type.
    pub fn new(to: PortType) -> Self {
        let (name, view) = match to {
            PortType::Reg128 => ("__reg_view_raw", RegLanes::Raw),
            PortType::RegI8x16 => ("__reg_view_i8x16", RegLanes::I8x16),
            PortType::RegI16x8 => ("__reg_view_i16x8", RegLanes::I16x8),
            PortType::RegI32x4 => ("__reg_view_i32x4", RegLanes::I32x4),
            PortType::RegI64x2 => ("__reg_view_i64x2", RegLanes::I64x2),
            PortType::RegF16x8 => ("__reg_view_f16x8", RegLanes::F16x8),
            PortType::RegF32x4 => ("__reg_view_f32x4", RegLanes::F32x4),
            PortType::RegF64x2 => ("__reg_view_f64x2", RegLanes::F64x2),
            other => panic!("RegView::new: {other:?} is not a register PortType"),
        };
        Self {
            meta: NodeMeta {
                name: name.into(),
                outs: vec![Port::new("output", to)],
                // The input port type is nominal — any register
                // view satisfies it (free-bitcast rule in
                // `Value::satisfies_slot`).
                ins: vec![Slot::Wire(Port::new("input", PortType::Reg128))],
            },
            to: view,
        }
    }
}

impl PolydatNode for RegView {
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }

    fn eval(&self, inputs: &[Value], outputs: &mut [Value]) {
        outputs[0] = Value::Reg128(inputs[0].as_reg_bits(), self.to);
    }

    /// In compiled buffers a view retag is a two-slot copy — the
    /// lane typing is a static property of the consuming slot, so
    /// the bits pass through verbatim (truly free at P2; at P3
    /// it will be elided entirely).
    fn compiled_u64(&self) -> Option<polydat::ast::CompiledU64Op> {
        Some(Box::new(|inputs: &[u64], outputs: &mut [u64]| {
            outputs[0] = inputs[0];
            outputs[1] = inputs[1];
        }))
    }
}

/// `true` when `t` is any register-plane PortType.
pub fn is_reg_port(t: PortType) -> bool {
    matches!(
        t,
        PortType::Reg128
            | PortType::RegI8x16
            | PortType::RegI16x8
            | PortType::RegI32x4
            | PortType::RegI64x2
            | PortType::RegF16x8
            | PortType::RegF32x4
            | PortType::RegF64x2
    )
}
