// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Converting a value to a port type, the one way polydat does it
//! (input_variance.md).
//!
//! A write into a kernel is never converted: a value that does not
//! satisfy the input's declared type is refused. Where a host's values
//! legitimately vary in type, the conversion is either the host's, made
//! with [`to_port`] before the write, or a converter node in the
//! program, which applies the same function when its input changes.
//! Both use the boundary adapter catalog, so a value converts the same
//! way on either side.

use crate::ast::{NodeMeta, PolydatNode, Port, PortType, Slot, Value};

/// Why a value could not be converted.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvertError {
    /// The value's type.
    pub from: PortType,
    /// The type asked for.
    pub to: PortType,
    /// What the conversion said, or that there is none.
    pub reason: String,
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cannot convert {} to {}: {}",
            self.from, self.to, self.reason
        )
    }
}

impl std::error::Error for ConvertError {}

/// `value` as a value of type `to`, by the boundary adapter catalog.
///
/// A value that already satisfies `to` (its own type, or a carrier's
/// bit-stuffed form) and `None` are returned unchanged. Otherwise the
/// catalog's adapter from the value's type to `to` is applied; one that
/// fails, such as a text that is not a number, is an error, as is a pair
/// of types the catalog has no adapter for.
pub fn to_port(value: Value, to: PortType) -> Result<Value, ConvertError> {
    if to == PortType::Dyn || value.satisfies_slot(to) {
        return Ok(value);
    }
    let from = value.port_type();
    let Some(adapter) = crate::compile::assembly::boundary_adapter(from, to) else {
        return Err(ConvertError {
            from,
            to,
            reason: "no conversion between these types".into(),
        });
    };
    // A failing adapter panics, as every node's failure does; caught
    // here quietly and returned as the error it is.
    let capture = crate::kernel::engines::EvalPanicCaptureGuard::arm();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut outputs = vec![Value::None];
        adapter.eval(&[value], &mut outputs);
        outputs.remove(0)
    }));
    drop(capture);
    match outcome {
        Ok(converted) => Ok(converted),
        Err(payload) => Err(ConvertError {
            from,
            to,
            reason: crate::kernel::panic_payload_text(payload.as_ref()),
        }),
    }
}

/// The converter the compiler places in front of an input whose type
/// may vary (input_variance.md §5): a `Dyn` input, the value as the
/// host wrote it, and an output of the type its consumers read.
pub(crate) struct InputConverter {
    meta: NodeMeta,
    input: String,
    to: PortType,
}

impl InputConverter {
    /// The converter of input `input` to `to`, named for both.
    pub(crate) fn new(input: &str, to: PortType) -> Self {
        Self {
            meta: NodeMeta {
                name: converter_name(input, to),
                outs: vec![Port::new("output", to)],
                ins: vec![Slot::Wire(Port::new("input", PortType::Dyn))],
            },
            input: input.to_string(),
            to,
        }
    }
}

/// The node name of input `input`'s converter to `to`.
pub(crate) fn converter_name(input: &str, to: PortType) -> String {
    format!("__convert_{input}_{}", to.to_keyword())
}

/// Convert, or fail as the converter node, naming the input.
fn convert_input(input: &str, value: Value, to: PortType) -> Value {
    match to_port(value, to) {
        Ok(v) => v,
        Err(e) => panic!("input '{input}': {e}"),
    }
}

impl PolydatNode for InputConverter {
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }

    fn eval(&self, inputs: &[Value], outputs: &mut [Value]) {
        outputs[0] = convert_input(&self.input, inputs[0].clone(), self.to);
    }

    /// The same conversion over slots: the input's pair names the value
    /// the kernel stores for it, and the output is written by its type,
    /// a scalar as its bits and a by-reference value into the step's
    /// scratch.
    fn compiled_slot(
        &self,
        _wire_types: &[PortType],
        _engine: crate::compile::select::Engine,
    ) -> Option<crate::ast::CompiledSlotKit> {
        use crate::ast::SlotShape;
        let (input, to) = (self.input.clone(), self.to);
        Some(crate::ast::CompiledSlotKit {
            scratch: to.scratch_elem().into_iter().collect(),
            op: Box::new(
                move |inputs: &[u64],
                      outputs: &mut [u64],
                      scratch: &mut [crate::ast::ScratchBuf]| {
                    let value = crate::derive_support::read_poly(PortType::Dyn, inputs);
                    let converted = convert_input(&input, value, to);
                    crate::derive_support::write_poly(to, converted, scratch, outputs);
                },
            ),
        })
    }
}
