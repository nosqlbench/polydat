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
#[cfg(feature = "jit")]
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
                panic!(
                    "a {:?} boundary value has no value-table entry assigned (SRD 115 §3)",
                    v.port_type()
                )
            });
            table.write(entry, v.clone())
        }
        _ => return None,
    })
}

/// The one-byte type code a variadic lowering records per argument
/// (SRD 115 §6): the codes of a node's wires are interned as a static
/// string, and the helper decodes each argument by its code. `None`
/// for a type no helper can take.
#[cfg(feature = "jit")]
pub(crate) fn type_code(ty: PortType) -> Option<u8> {
    Some(match ty {
        PortType::U64 => b'u',
        PortType::I64 => b'i',
        PortType::F64 => b'f',
        PortType::Bool => b'b',
        PortType::Str => b's',
        PortType::Bytes => b'y',
        PortType::Json => b'j',
        PortType::Ext => b'e',
        PortType::Handle => b'h',
        _ => return None,
    })
}

/// The port type a type code names.
#[cfg(feature = "jit")]
pub(crate) fn type_of_code(code: u8) -> PortType {
    match code {
        b'i' => PortType::I64,
        b'f' => PortType::F64,
        b'b' => PortType::Bool,
        b's' => PortType::Str,
        b'y' => PortType::Bytes,
        b'j' => PortType::Json,
        b'e' => PortType::Ext,
        b'h' => PortType::Handle,
        _ => PortType::U64,
    }
}

/// An argument's slot bits as an owned `Value`, decoded by its type
/// code through the table installed for the running native code.
#[cfg(feature = "jit")]
pub(crate) fn arg_value(code: u8, bits: u64) -> Value {
    let ty = type_of_code(code);
    match ty.handle_kind() {
        Some(crate::ast::HandleKind::Table) => {
            crate::kernel::with_current_value_table(|t| t.read(bits))
        }
        _ => decode_slot(bits, ty, &ValueTable::new(0)),
    }
}

/// An argument's slot bits as a borrowed view, decoded by its type
/// code: nothing is copied. Strings are borrowed from the arena or the
/// interner and table kinds from the installed table, both valid for
/// the rest of the current native call.
#[cfg(feature = "jit")]
pub(crate) fn arg_ref(code: u8, bits: u64) -> crate::ast::ValueRef<'static> {
    use crate::ast::ValueRef;
    match code {
        b'u' => ValueRef::U64(bits),
        b'i' => ValueRef::I64(bits as i64),
        b'f' => ValueRef::F64(f64::from_bits(bits)),
        b'b' => ValueRef::Bool(bits != 0),
        b's' => ValueRef::Str(crate::kernel::resolve_thread_str(bits)),
        b'y' => ValueRef::Bytes(crate::kernel::resolve_thread_bytes(bits)),
        _ => ValueRef::from(crate::kernel::current_table_value(bits)),
    }
}

/// An argument as a format argument: strings are borrowed from the
/// arena or the interner rather than copied.
#[cfg(feature = "jit")]
pub(crate) fn fmt_arg(code: u8, bits: u64) -> crate::library::format::FmtArg<'static> {
    use crate::library::format::FmtArg;
    match code {
        b'u' => FmtArg::U64(bits),
        b'f' => FmtArg::F64(f64::from_bits(bits)),
        b'b' => FmtArg::Bool(bits != 0),
        b's' => FmtArg::Str(crate::kernel::resolve_thread_str(bits)),
        _ => FmtArg::Value(arg_value(code, bits)),
    }
}

/// Slot bits as the `Value` their declared port type names, copied out
/// of the arena or of `table` where the bits are a handle.
pub(crate) fn decode_slot(bits: u64, ty: PortType, table: &ValueTable) -> Value {
    match ty {
        PortType::F64 => Value::F64(f64::from_bits(bits)),
        PortType::Bool => Value::Bool(bits != 0),
        PortType::I64 => Value::I64(bits as i64),
        PortType::Str => Value::Str(std::sync::Arc::from(crate::kernel::resolve_thread_str(
            bits,
        ))),
        PortType::Bytes => Value::Bytes(std::sync::Arc::from(crate::kernel::resolve_thread_bytes(
            bits,
        ))),
        PortType::Json | PortType::Ext | PortType::Handle => table.read(bits),
        // A signed narrow carrier rides sign-extended (alignment §8.1)
        // and is the `I64` value its `Wire` impl injects.
        PortType::I8 | PortType::I16 | PortType::I32 => Value::I64(bits as i64),
        _ => Value::U64(bits),
    }
}

/// An output at `slot` of `buffer` as the `Value` its port type names:
/// [`decode_slot`] for a one-slot carrier and the table kinds, and the
/// two-limb reassembly for a 128-bit integer or a register word, which
/// ride two consecutive slots (alignment §8.4 layer 1). This is the
/// typed read every compiled kernel's `get_value` makes; a vector
/// output is read from the kernel's scratch before it reaches here.
pub(crate) fn decode_output(
    buffer: &[u64],
    slot: usize,
    ty: PortType,
    table: &ValueTable,
) -> Value {
    use crate::ast::{Bits128, RegLanes};
    if ty.slot_width() == 2 {
        let limbs = Bits128([buffer[slot], buffer[slot + 1]]);
        return match ty {
            PortType::U128 => Value::U128(limbs),
            PortType::I128 => Value::I128(limbs),
            PortType::RegI8x16 => Value::Reg128(limbs, RegLanes::I8x16),
            PortType::RegI16x8 => Value::Reg128(limbs, RegLanes::I16x8),
            PortType::RegI32x4 => Value::Reg128(limbs, RegLanes::I32x4),
            PortType::RegI64x2 => Value::Reg128(limbs, RegLanes::I64x2),
            PortType::RegF16x8 => Value::Reg128(limbs, RegLanes::F16x8),
            PortType::RegF32x4 => Value::Reg128(limbs, RegLanes::F32x4),
            PortType::RegF64x2 => Value::Reg128(limbs, RegLanes::F64x2),
            _ => Value::Reg128(limbs, RegLanes::Raw),
        };
    }
    decode_slot(buffer[slot], ty, table)
}
