// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Marshalling between typed `Value`s and compiled-tier slots (SRD 115
//! §5). One place owns the rule for every port type, so embedded cones
//! and whole compiled kernels decode alike:
//!
//! - scalars ride as their slot bits;
//! - byte strings (`Str`, `Bytes`) enter as cycle-arena handles and
//!   leave as owned copies;
//! - every other non-scalar value (`Json`, `Ext`, `Handle`) is written
//!   into the entry of the engine's value table that the layout assigned
//!   to its slot, and leaves as a clone of that entry.
//!
//! P1 never holds a handle (axiom H6): decoding always copies out.

use crate::ast::{PortType, Value};
use crate::kernel::ValueTable;

/// A `Value` as the slot bits its port type rides as. A table-kind value
/// is written to `entry` of `table`; `entry` is `None` for every other
/// value. `None` when the value's variant has no compiled representation.
pub(crate) fn encode_slot(v: &Value, table: &mut ValueTable, entry: Option<usize>) -> Option<u64> {
    Some(match v {
        Value::U64(x) => *x,
        Value::I64(x) => *x as u64,
        Value::F64(x) => x.to_bits(),
        Value::Bool(b) => *b as u64,
        Value::Str(s) => crate::kernel::put_thread_str(s),
        Value::Bytes(b) => crate::kernel::put_thread_bytes(b),
        Value::Json(_) | Value::Ext(_) | Value::Handle(_) => {
            let entry = entry.unwrap_or_else(|| {
                panic!("a {:?} boundary value has no value-table entry assigned (SRD 115 §3)", v.port_type())
            });
            table.write(entry, v.clone())
        }
        _ => return None,
    })
}

/// Slot bits as the `Value` their declared port type names, copied out
/// of the arena or of `table` where the bits are a handle.
pub(crate) fn decode_slot(bits: u64, ty: PortType, table: &ValueTable) -> Value {
    match ty {
        PortType::F64 => Value::F64(f64::from_bits(bits)),
        PortType::Bool => Value::Bool(bits != 0),
        PortType::I64 => Value::I64(bits as i64),
        PortType::Str => Value::Str(std::sync::Arc::from(crate::kernel::resolve_thread_str(bits))),
        PortType::Bytes => Value::Bytes(std::sync::Arc::from(crate::kernel::resolve_thread_bytes(bits))),
        PortType::Json | PortType::Ext | PortType::Handle => table.read(bits),
        _ => Value::U64(bits),
    }
}
