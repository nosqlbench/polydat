// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Identity and constant nodes.

use crate::ast::SlotShape;
use crate::ast::{NodeMeta, PolydatNode, Port, PortType, Slot, Value};

/// Passthrough: output equals input. Polymorphic via PolyWire: the
/// runtime port type is resolved by the assembler from the upstream
/// wire's type and passed to `Identity::new(input_type)`.
///
/// JIT is disabled (Value isn't a u64-buffer carrier); for a
/// u64-only fast path, the assembler can synthesize a typed
/// alternative.
#[crate::polydat_node(category = Diagnostic)]
fn identity(input: Value) -> Value {
    input
}

/// Passthrough for external port values (captures).
///
/// Reads a single input (from a `WireSource::Port`) and copies it
/// unchanged to the output. The port type is declared based on the
/// port's default value type at construction time.
///
/// This node is auto-inserted by the compiler for `extern` port
/// declarations, making captured values available as Polydat outputs.
pub struct PortPassthrough {
    meta: NodeMeta,
}

impl PortPassthrough {
    /// Create a port passthrough with the given output type.
    pub fn new(name: &str, port_type: crate::ast::PortType) -> Self {
        Self {
            meta: NodeMeta {
                name: format!("__port_{name}"),
                outs: vec![Port::new("output", port_type)],
                ins: vec![Slot::Wire(Port::new("input", port_type))],
            },
        }
    }
}

impl PolydatNode for PortPassthrough {
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }

    fn eval(&self, inputs: &[Value], outputs: &mut [Value]) {
        outputs[0] = inputs[0].clone();
    }

    /// A passthrough copies its slots, on every color but `Ref2`, whose
    /// pairs may not be forwarded by an identity-style step (axiom S3).
    /// Byte-string handles copy like scalars: a handle is a name (SRD
    /// 115 H1). A table-kind output is not copied here: the builders
    /// give it `assembly::table_copy_op`, which re-enters the value so
    /// the slot names its own entry (H4).
    fn compiled_u64(&self) -> Option<crate::ast::CompiledU64Op> {
        if self.meta.outs[0].typ.slot_color() == crate::ast::SlotColor::Ref2 {
            return None;
        }
        Some(Box::new(|inputs: &[u64], outputs: &mut [u64]| {
            outputs.copy_from_slice(inputs)
        }))
    }
}

/// Emit a fixed u64 value (no inputs).
///
/// Signature: `const_u64(value: u64) -> (u64)`
///
/// Source node that always produces the same u64 regardless of cycle.
/// Use for injecting literal parameters into a DAG, such as a fixed
/// partition key, an epoch timestamp base, or an addend for `add`.
/// Takes no inputs, so it sits at a DAG root.
///
/// JIT level: P2 (compiled_u64 emits a captured constant via the
/// `#[polydat_node]`-emitted body capture).
///
/// Named per the per-type scheme shared with `const_f64` /
/// `const_bool` (see `library::fixed`).
#[crate::polydat_node(category = Math)]
fn const_u64(value: crate::derive_support::Const<u64>) -> u64 {
    *value
}

/// Emit a fixed string value (no inputs).
///
/// Signature: `const_str(value: String) -> (Arc<str>)`
///
/// Source node that always produces the same string regardless of cycle.
/// Use for injecting literal string parameters into a DAG, such as a
/// fixed table name, a static label, or a separator for string
/// concatenation pipelines.
///
/// JIT level: P2 stores the interned static handle; P3 lowers to an
/// immediate static handle (SRD 115 §2.2).
///
/// The `Const<&str>` source captures the owned `String`;
/// `#[poly_const]` derives an `Arc<str>` cache at construction time,
/// so per-cycle eval is a refcount bump on a single heap allocation.
/// The macro emits `ConstStr::new(value: String)`.
fn const_str_arc(s: &str) -> std::sync::Arc<str> {
    std::sync::Arc::from(s)
}

/// The compiled form of a string literal: the text is interned at
/// kernel build, and the step publishes the `(ptr, len)` pair of the
/// interned bytes, which have process lifetime (jit_boundary.md, axiom
/// S7). It owns no scratch and copies nothing.
fn const_str_compiled(
    node: &ConstStr,
    _wire_types: &[crate::ast::PortType],
) -> crate::ast::CompiledSlotKit {
    let (ptr, len) = crate::kernel::static_pair(crate::kernel::StaticInterner::intern(&node.value));
    crate::ast::CompiledSlotKit {
        scratch: Vec::new(),
        op: Box::new(
            move |_inputs: &[u64], outputs: &mut [u64], _scratch: &mut [crate::ast::ScratchBuf]| {
                outputs[0] = ptr;
                outputs[1] = len;
            },
        ),
    }
}

#[crate::polydat_node(category = Diagnostic, compiled_slot = const_str_compiled)]
fn const_str(
    #[poly_default("")] value: crate::derive_support::Const<&str>,
    #[poly_const(const_str_arc, from = value)] cached: &std::sync::Arc<str>,
) -> std::sync::Arc<str> {
    cached.clone()
}

/// Emit a fixed [`Value::Handle`] (no inputs).
///
/// Signature: `const_handle() -> (Handle)`
///
/// Created by the constant-folding pass to replace an `init`
/// binding whose evaluation produced a `Value::Handle` (e.g.
/// `init prebuffered = dataset_prebuffer(...)`). Without this
/// replacement, the original side-effect-bearing node would
/// stay in the program graph with its eval intact, and every
/// fresh fiber's `PolydatState` would re-fire the eval at first
/// downstream pull — producing a per-fiber stampede that, in
/// the prebuffer case, exhausts the per-process thread limit
/// when vectordata's HTTP workers spin up concurrently.
///
/// The handle's `Arc` is cloned per `eval()` call (one atomic
/// refcount bump); the underlying resource is shared.
///
/// JIT level: P1 (Handle output; no compiled_u64 path).
pub struct ConstHandle {
    meta: NodeMeta,
    value: std::sync::Arc<dyn std::any::Any + Send + Sync>,
}

impl ConstHandle {
    /// A constant node holding `value` as a handle.
    pub fn new(value: std::sync::Arc<dyn std::any::Any + Send + Sync>) -> Self {
        Self {
            meta: NodeMeta {
                name: "const_handle".into(),
                outs: vec![Port::new("output", PortType::Handle)],
                // No const slot — the handle is type-erased and
                // doesn't fit the const-slot vocabulary; fold-pass
                // synthesises this node directly with no input wires.
                ins: vec![],
            },
            value,
        }
    }
}

impl PolydatNode for ConstHandle {
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }

    fn eval(&self, _inputs: &[Value], outputs: &mut [Value]) {
        outputs[0] = Value::Handle(self.value.clone());
    }
}

/// SRD 71 — leaf const for [`Value::Ext`]-typed values
/// (Partition, PartitionSpec, PartitionList, …).
///
/// Mirrors [`ConstHandle`]'s shape for `Handle`-typed values:
/// fold-pass synthesises one of these in place of any
/// node-with-wiring whose evaluated output is an `Ext` value,
/// so the post-fold kernel can read the constant via
/// `get_constant` (no input slots, eval just emits the stored
/// value).
pub struct ConstExt {
    meta: NodeMeta,
    value: Box<dyn crate::ast::ReflectedValue>,
}

impl ConstExt {
    /// A constant node holding an extension value.
    pub fn new(value: Box<dyn crate::ast::ReflectedValue>) -> Self {
        Self {
            meta: NodeMeta {
                name: "const_ext".into(),
                outs: vec![Port::new("output", PortType::Ext)],
                ins: vec![],
            },
            value,
        }
    }
}

impl PolydatNode for ConstExt {
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }

    fn eval(&self, _inputs: &[Value], outputs: &mut [Value]) {
        outputs[0] = Value::Ext(self.value.clone());
    }
}
