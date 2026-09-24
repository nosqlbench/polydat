// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Marshalling between typed `Value`s and compiled-tier slots. One
//! place owns the rule for every port type, so embedded cones and
//! whole compiled kernels decode alike:
//!
//! - scalars ride as their slot bits, one slot or two;
//! - every `Ref2` kind rides as a `(ptr, len)` pair (jit_boundary.md,
//!   axioms S1–S10): a typed vector, a string, or a byte string as a
//!   slice of its elements, and a JSON, extension, or handle value as
//!   a one-element slice holding the `Value`.
//!
//! Entering the compiled tier, a `Value` is borrowed: the pair points
//! into the value the caller holds for the duration of the call.
//! Leaving it, the pair is copied out to an owned `Value`: the
//! interpreter and the host never hold a reference into a state's
//! buffers.

use crate::ast::SlotShape;
use crate::ast::{PortType, Value};

/// The `(ptr, len)` pair a `Ref2`-kind value is borrowed as: into the
/// value's own bytes or element storage, or the value itself for a
/// JSON, extension, or handle value. The pair is valid while `v` is.
/// `None` for a scalar or for a variant the port type does not admit.
pub(crate) fn borrow_pair(v: &Value) -> Option<(u64, u64)> {
    Some(match v {
        Value::Str(s) => (s.as_ptr() as usize as u64, s.len() as u64),
        Value::Bytes(b) => (b.as_ptr() as usize as u64, b.len() as u64),
        Value::Json(_) | Value::Ext(_) | Value::Handle(_) => (v as *const Value as usize as u64, 1),
        Value::VecF32(v) => slice_pair(v.as_slice()),
        Value::VecF64(v) => slice_pair(v.as_slice()),
        Value::VecF16(v) => slice_pair(v.as_slice()),
        Value::VecI8(v) => slice_pair(v.as_slice()),
        Value::VecI16(v) => slice_pair(v.as_slice()),
        Value::VecI32(v) => slice_pair(v.as_slice()),
        Value::VecI64(v) => slice_pair(v.as_slice()),
        _ => return None,
    })
}

/// The `(ptr, len)` pair of a slice.
#[inline]
pub(crate) fn slice_pair<T>(s: &[T]) -> (u64, u64) {
    (s.as_ptr() as usize as u64, s.len() as u64)
}

/// An empty pair for an unset `Ref2` slot: a dangling, non-null
/// pointer with length zero, which every slice reader accepts.
#[inline]
pub(crate) fn empty_pair() -> (u64, u64) {
    (
        std::ptr::NonNull::<u8>::dangling().as_ptr() as usize as u64,
        0,
    )
}

/// A `Value` as the slot bits its port type `ty` rides as: the bits of a
/// scalar, or the borrowed pair of a `Ref2` kind. A `Dyn` port names the
/// value itself, whatever its variant, for its converter to read. `None`
/// when the value's variant has no compiled representation.
#[cfg(feature = "jit")]
pub(crate) fn encode_slots(v: &Value, ty: PortType, out: &mut [u64]) -> Option<()> {
    if ty == PortType::Dyn {
        (out[0], out[1]) = match v {
            Value::None => empty_pair(),
            v => (v as *const Value as usize as u64, 1),
        };
        return Some(());
    }
    match v {
        Value::U64(x) => out[0] = *x,
        Value::I64(x) => out[0] = *x as u64,
        Value::F64(x) => out[0] = x.to_bits(),
        Value::Bool(b) => out[0] = *b as u64,
        _ => {
            let (p, l) = borrow_pair(v)?;
            out[0] = p;
            out[1] = l;
        }
    }
    Some(())
}

/// The value a `Ref2` pair names, copied out as the `Value` its port
/// type declares. Every read that leaves the compiled tier goes
/// through here.
///
/// # Safety
///
/// The pair was published by a producer whose storage is alive: its
/// own scratch, an extern's stored value, an interned constant, or a
/// boundary value alive for the call (axioms S3, S4).
pub(crate) unsafe fn decode_pair(ty: PortType, ptr: u64, len: u64) -> Value {
    use crate::ast::SliceArc;
    let (p, n) = (ptr as usize, len as usize);
    // SAFETY: as documented on the function.
    unsafe {
        match ty {
            PortType::Str => Value::Str(std::sync::Arc::from(std::str::from_utf8_unchecked(
                std::slice::from_raw_parts(p as *const u8, n),
            ))),
            PortType::Bytes => Value::Bytes(std::sync::Arc::from(std::slice::from_raw_parts(
                p as *const u8,
                n,
            ))),
            PortType::Json | PortType::Ext | PortType::Handle | PortType::Dyn => {
                if n == 0 {
                    Value::None
                } else {
                    (*(p as *const Value)).clone()
                }
            }
            PortType::VecF32 => Value::VecF32(SliceArc::from_vec(
                std::slice::from_raw_parts(p as *const f32, n).to_vec(),
            )),
            PortType::VecF64 => Value::VecF64(SliceArc::from_vec(
                std::slice::from_raw_parts(p as *const f64, n).to_vec(),
            )),
            PortType::VecF16 => Value::VecF16(SliceArc::from_vec(
                std::slice::from_raw_parts(p as *const half::f16, n).to_vec(),
            )),
            PortType::VecI8 => Value::VecI8(SliceArc::from_vec(
                std::slice::from_raw_parts(p as *const i8, n).to_vec(),
            )),
            PortType::VecI16 => Value::VecI16(SliceArc::from_vec(
                std::slice::from_raw_parts(p as *const i16, n).to_vec(),
            )),
            PortType::VecI32 => Value::VecI32(SliceArc::from_vec(
                std::slice::from_raw_parts(p as *const i32, n).to_vec(),
            )),
            PortType::VecI64 => Value::VecI64(SliceArc::from_vec(
                std::slice::from_raw_parts(p as *const i64, n).to_vec(),
            )),
            other => panic!("{other:?} is not a Ref2-colored port type"),
        }
    }
}

/// The port type of the `Value` a port of type `ty` carries, which is
/// what [`decode_slot`] reads back. A narrow integer rides the 64-bit
/// carrier of its sign, and `f32` and `f16` ride their bit patterns in
/// a `U64`, as their `Wire` impls inject them; every other type is its
/// own carrier.
pub(crate) fn carrier_port(ty: PortType) -> PortType {
    match ty {
        PortType::I8 | PortType::I16 | PortType::I32 => PortType::I64,
        PortType::U32 | PortType::U16 | PortType::U8 | PortType::F32 | PortType::F16 => {
            PortType::U64
        }
        other => other,
    }
}

/// Slot bits as the `Value` their declared port type names, copied
/// out where they are a pair.
///
/// The read is chosen by the type's slot color, as `write_poly`'s
/// write is, so every compiled read of a value (an output a host
/// pulls, and an input a polymorphic node takes) is the inverse of
/// the one write. This used to match on the type with a `U64`
/// fallback. The `Imm2` limb reassembly lived only in
/// `decode_output`, so a register or 128-bit value reaching a
/// polymorphic node's input fell through to the fallback and arrived
/// as its low limb, typed `U64`. The type system knew better at every
/// step; only the fallback arm did not.
pub(crate) fn decode_slot(slots: &[u64], ty: PortType) -> Value {
    use crate::ast::{Bits128, RegLanes, SlotColor};
    match ty.slot_color() {
        SlotColor::Imm1 => match ty {
            PortType::F64 => Value::F64(f64::from_bits(slots[0])),
            PortType::Bool => Value::Bool(slots[0] != 0),
            // A signed narrow carrier rides sign-extended (alignment
            // §2) and is the `I64` value its `Wire` impl injects.
            PortType::I64 | PortType::I8 | PortType::I16 | PortType::I32 => {
                Value::I64(slots[0] as i64)
            }
            // Unsigned narrow carriers ride zero-extended, and `f32`
            // and `f16` ride their bit patterns, all in the `U64`
            // carrier their `Wire` impls inject.
            PortType::U64
            | PortType::U32
            | PortType::U16
            | PortType::U8
            | PortType::F32
            | PortType::F16 => Value::U64(slots[0]),
            other => unreachable!("{other:?} is colored Imm1 but has no one-slot carrier"),
        },
        // Two consecutive slots of limb data, low word first
        // (alignment §6).
        SlotColor::Imm2 => {
            let limbs = Bits128([slots[0], slots[1]]);
            match ty {
                PortType::U128 => Value::U128(limbs),
                PortType::I128 => Value::I128(limbs),
                PortType::Reg128 => Value::Reg128(limbs, RegLanes::Raw),
                PortType::RegI8x16 => Value::Reg128(limbs, RegLanes::I8x16),
                PortType::RegI16x8 => Value::Reg128(limbs, RegLanes::I16x8),
                PortType::RegI32x4 => Value::Reg128(limbs, RegLanes::I32x4),
                PortType::RegI64x2 => Value::Reg128(limbs, RegLanes::I64x2),
                PortType::RegF16x8 => Value::Reg128(limbs, RegLanes::F16x8),
                PortType::RegF32x4 => Value::Reg128(limbs, RegLanes::F32x4),
                PortType::RegF64x2 => Value::Reg128(limbs, RegLanes::F64x2),
                other => unreachable!("{other:?} is colored Imm2 but is not a 128-bit value"),
            }
        }
        // SAFETY: the pair in a kernel's buffer was published by a
        // producer whose storage is alive (axioms S3, S4).
        SlotColor::Ref2 => unsafe { decode_pair(ty, slots[0], slots[1]) },
    }
}

/// An output at `slot` of `buffer` as the `Value` its port type names.
/// This is the typed read every compiled kernel's `get_value` makes.
pub fn decode_output(buffer: &[u64], slot: usize, ty: PortType) -> Value {
    decode_slot(&buffer[slot..], ty)
}

/// An argument's slots as a borrowed view of the value, decoded by its
/// port type: nothing is copied. A string, byte string, or value is
/// borrowed through its pair, valid while its producer's storage is.
///
/// # Safety
///
/// As for `decode_pair`.
pub unsafe fn arg_ref<'a>(ty: PortType, slots: &'a [u64]) -> crate::ast::ValueRef<'a> {
    use crate::ast::ValueRef;
    let (p, n) = (
        slots.first().copied().unwrap_or(0) as usize,
        slots.get(1).copied().unwrap_or(0) as usize,
    );
    // SAFETY: as documented on the function.
    unsafe {
        match ty {
            PortType::U64 => ValueRef::U64(slots[0]),
            PortType::I64 | PortType::I8 | PortType::I16 | PortType::I32 => {
                ValueRef::I64(slots[0] as i64)
            }
            PortType::F64 => ValueRef::F64(f64::from_bits(slots[0])),
            PortType::Bool => ValueRef::Bool(slots[0] != 0),
            PortType::Str => ValueRef::Str(std::str::from_utf8_unchecked(
                std::slice::from_raw_parts(p as *const u8, n),
            )),
            PortType::Bytes => ValueRef::Bytes(std::slice::from_raw_parts(p as *const u8, n)),
            PortType::Json | PortType::Ext | PortType::Handle | PortType::Dyn => {
                if n == 0 {
                    ValueRef::None
                } else {
                    ValueRef::from(&*(p as *const Value))
                }
            }
            _ => ValueRef::U64(slots[0]),
        }
    }
}
