// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The register-lane bodies: the bounds-checked gathers, lane reads,
//! and lane writes the `reg_*` nodes and their native lowerings share,
//! on the word as the slots hold it, so both produce the same bits and
//! the same message on a bad lane or window.

use crate::ast::Bits128;

/// The word of lanes `[offset, offset+4)` of `v`; panics past the end.
pub fn gather_f32(v: &[f32], offset: u64) -> Bits128 {
    let o = offset as usize;
    if o + 4 > v.len() {
        panic!(
            "reg_gather_f32: window [{o}, {}) exceeds slice length {}",
            o + 4,
            v.len()
        );
    }
    Bits128::from_lanes_f32([v[o], v[o + 1], v[o + 2], v[o + 3]])
}

/// The word of a four-element `v`.
pub fn to_reg_f32(v: &[f32]) -> Bits128 {
    if v.len() != 4 {
        panic!(
            "vec_to_reg_f32: expected exactly 4 elements, got {}",
            v.len()
        );
    }
    Bits128::from_lanes_f32([v[0], v[1], v[2], v[3]])
}

/// Lane `i` of `r` as f32 lanes, widened.
pub fn lane_f32(r: Bits128, i: u64) -> f64 {
    if i >= 4 {
        panic!("reg_lane_f32: lane {i} out of range 0..4");
    }
    r.lanes_f32()[i as usize] as f64
}

/// `r` with f32 lane `i` replaced by `v`.
pub fn with_lane_f32(r: Bits128, i: u64, v: f64) -> Bits128 {
    if i >= 4 {
        panic!("reg_with_lane_f32: lane {i} out of range 0..4");
    }
    let mut out = r.lanes_f32();
    out[i as usize] = v as f32;
    Bits128::from_lanes_f32(out)
}

/// Lane `i` of `r` as i16 lanes.
pub fn lane_i16(r: Bits128, i: u64) -> i16 {
    if i >= 8 {
        panic!("reg_lane_i16: lane {i} out of range 0..8");
    }
    r.lanes_i16()[i as usize]
}

/// Lane `i` of `r` as i64 lanes.
pub fn lane_i64(r: Bits128, i: u64) -> i64 {
    if i >= 2 {
        panic!("reg_lane_i64: lane {i} out of range 0..2");
    }
    r.lanes_i64()[i as usize]
}

/// The wrapping product of `a` and `b` as i8 lanes.
pub fn mul_i8(a: Bits128, b: Bits128) -> Bits128 {
    let (a, b) = (a.lanes_i8(), b.lanes_i8());
    Bits128::from_lanes_i8(core::array::from_fn(|i| a[i].wrapping_mul(b[i])))
}
