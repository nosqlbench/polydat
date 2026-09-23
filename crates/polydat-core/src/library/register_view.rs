// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The register view retags the assembler inserts between register
//! ports of different lane typings, and the register-port predicate it
//! keys on (type_system_alignment.md §8.4 layer 2).
//!
//! A view is a free bitcast: the 128 bits are untouched and only the
//! lane typing changes. Each view is an ordinary registered node, like
//! every other entry of the conversion table, so a program can call it
//! and the conversion fuzzer can reach it by the name the table gives.
//! It used to be one hand-written node parameterized by its target
//! type, which the assembler could insert but no program could name.
//!
//! Each takes the raw word: any register view satisfies a register
//! slot (the free-bitcast rule in `Value::satisfies_slot`).

use crate::ast::{Bits128, PolydatNode, PortType};

/// A register word viewed raw, without a lane typing.
#[crate::polydat_node(category = Conversions)]
fn __reg_view_raw(r: Bits128) -> Bits128 {
    r
}

/// A register word viewed as sixteen `i8` lanes.
#[crate::polydat_node(category = Conversions)]
fn __reg_view_i8x16(r: Bits128) -> [i8; 16] {
    r.lanes_i8()
}

/// A register word viewed as eight `i16` lanes.
#[crate::polydat_node(category = Conversions)]
fn __reg_view_i16x8(r: Bits128) -> [i16; 8] {
    r.lanes_i16()
}

/// A register word viewed as four `i32` lanes.
#[crate::polydat_node(category = Conversions)]
fn __reg_view_i32x4(r: Bits128) -> [i32; 4] {
    r.lanes_i32()
}

/// A register word viewed as two `i64` lanes.
#[crate::polydat_node(category = Conversions)]
fn __reg_view_i64x2(r: Bits128) -> [i64; 2] {
    r.lanes_i64()
}

/// A register word viewed as eight `f16` lanes.
#[crate::polydat_node(category = Conversions)]
fn __reg_view_f16x8(r: Bits128) -> [half::f16; 8] {
    r.lanes_f16()
}

/// A register word viewed as four `f32` lanes.
#[crate::polydat_node(category = Conversions)]
fn __reg_view_f32x4(r: Bits128) -> [f32; 4] {
    r.lanes_f32()
}

/// A register word viewed as two `f64` lanes.
#[crate::polydat_node(category = Conversions)]
fn __reg_view_f64x2(r: Bits128) -> [f64; 2] {
    r.lanes_f64()
}

/// The view of a register word as `to`, the retag the assembler
/// inserts between register ports of different lane typings. `None`
/// when `to` is not a register type.
pub fn reg_view(to: PortType) -> Option<Box<dyn PolydatNode>> {
    Some(match to {
        PortType::Reg128 => Box::new(RegViewRaw::new()),
        PortType::RegI8x16 => Box::new(RegViewI8x16::new()),
        PortType::RegI16x8 => Box::new(RegViewI16x8::new()),
        PortType::RegI32x4 => Box::new(RegViewI32x4::new()),
        PortType::RegI64x2 => Box::new(RegViewI64x2::new()),
        PortType::RegF16x8 => Box::new(RegViewF16x8::new()),
        PortType::RegF32x4 => Box::new(RegViewF32x4::new()),
        PortType::RegF64x2 => Box::new(RegViewF64x2::new()),
        _ => return None,
    })
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
