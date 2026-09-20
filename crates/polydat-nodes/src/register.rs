// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Register-plane nodes — 128-bit SIMD words as values
//! (type_system_alignment.md §8.4 layer 2).
//!
//! A register word is a plain 16-byte value with a lane-typed or
//! raw view ([`polydat::ast::RegLanes`]). Views are free bitcasts:
//! the [`RegView`] adapter (auto-inserted by the assembler for
//! any reg→reg wire) retags without touching bits, so a word can
//! be `[i64; 2]` for one op, `[u8-ish bytes]` for a shuffle, and
//! algorithm-defined raw state for a third — at zero cost.
//!
//! Families here:
//!
//! - **Splats** — float lanes from `f64` (exact for f32 lanes in
//!   range), integer lanes from a `u64` **bit-level** constructor
//!   (`k as iN` per lane; range semantics belong to the adapter
//!   system, constructors are bit tools).
//! - **Gather / conversions** — `[f32; 4]` ↔ `vec_f32` and a
//!   bounds-checked window gather from a slice.
//! - **Lane access** — get/set for f32 lanes, reads for i16/i64.
//! - **Element-wise arithmetic** — add/sub/mul across every lane
//!   family; integer ops wrap (lane arithmetic is modular, the
//!   range-checked story lives in scalar adapters).
//! - **`reg_dot_f32`** — horizontal dot with a FIXED reduction
//!   tree `((l0+l1) + (l2+l3))`, part of the op contract so
//!   results are reproducible everywhere (determinism D2).
//! - **`reg_shuffle_bytes`** — arbitrary byte permutation of the
//!   raw word from a 16-entry const mask (the SWAR/state-word
//!   workhorse).

use polydat::ast::Bits128;
#[cfg(test)]
use polydat::ast::{PolydatNode, PortType, RegLanes, Value};
pub use polydat::library::register_view::{RegView, is_reg_port};
pub use polydat::numeric::register::{
    gather_f32, lane_f32, lane_i16, lane_i64, mul_i8, to_reg_f32, with_lane_f32,
};
// =================================================================
// Splats
// =================================================================

/// `reg_splat_f32(k)` — broadcast `k` (applied at f32 precision)
/// into all four f32 lanes.
#[polydat::polydat_node(category = Arithmetic)]
fn reg_splat_f32(k: f64) -> [f32; 4] {
    [k as f32; 4]
}

/// `reg_splat_f64(k)` — broadcast `k` into both f64 lanes.
#[polydat::polydat_node(category = Arithmetic)]
fn reg_splat_f64(k: f64) -> [f64; 2] {
    [k; 2]
}

/// `reg_splat_i8(k)` — bit-level broadcast: each lane is `k as i8`.
#[polydat::polydat_node(category = Arithmetic)]
fn reg_splat_i8(k: u64) -> [i8; 16] {
    [k as i8; 16]
}

/// `reg_splat_i16(k)` — bit-level broadcast: each lane is `k as i16`.
#[polydat::polydat_node(category = Arithmetic)]
fn reg_splat_i16(k: u64) -> [i16; 8] {
    [k as i16; 8]
}

/// `reg_splat_i32(k)` — bit-level broadcast: each lane is `k as i32`.
#[polydat::polydat_node(category = Arithmetic)]
fn reg_splat_i32(k: u64) -> [i32; 4] {
    [k as i32; 4]
}

/// `reg_splat_i64(k)` — bit-level broadcast: each lane is `k as i64`.
#[polydat::polydat_node(category = Arithmetic)]
fn reg_splat_i64(k: u64) -> [i64; 2] {
    [k as i64; 2]
}

// =================================================================
// Gather / conversions (f32 lanes — the embedding workhorse)
// =================================================================

/// `reg_gather_f32(v, offset)` — load lanes `[offset, offset+4)`
/// of an f32 slice into a register word. Panics when the window
/// runs past the end (silent zero-fill would corrupt distance
/// math downstream).
#[polydat::polydat_node(category = Arithmetic)]
fn reg_gather_f32(v: &[f32], offset: u64) -> [f32; 4] {
    gather_f32(v, offset).lanes_f32()
}

/// `vec_to_reg_f32(v)` — a length-4 `vec_f32` IS a `reg_f32x4`;
/// panics on any other length.
#[polydat::polydat_node(category = Conversions)]
fn vec_to_reg_f32(v: &[f32]) -> [f32; 4] {
    to_reg_f32(v).lanes_f32()
}

// ── The bodies the nodes and their native lowerings share ──────

/// `reg_to_vec_f32(r)` — the inverse projection.
#[polydat::polydat_node(category = Conversions)]
fn reg_to_vec_f32(r: [f32; 4]) -> Vec<f32> {
    r.to_vec()
}

// =================================================================
// Lane access
// =================================================================

/// `reg_lane_f32(r, i)` — read lane `i` (0..4), widened to f64.
#[polydat::polydat_node(category = Arithmetic)]
fn reg_lane_f32(r: [f32; 4], i: u64) -> f64 {
    lane_f32(Bits128::from_lanes_f32(r), i)
}

/// `reg_with_lane_f32(r, i, v)` — copy of `r` with lane `i`
/// replaced by `v` (at f32 precision).
#[polydat::polydat_node(category = Arithmetic)]
fn reg_with_lane_f32(r: [f32; 4], i: u64, v: f64) -> [f32; 4] {
    with_lane_f32(Bits128::from_lanes_f32(r), i, v).lanes_f32()
}

/// `reg_lane_i16(r, i)` — read lane `i` (0..8).
#[polydat::polydat_node(category = Arithmetic)]
fn reg_lane_i16(r: [i16; 8], i: u64) -> i16 {
    lane_i16(Bits128::from_lanes_i16(r), i)
}

/// `reg_lane_i64(r, i)` — read lane `i` (0..2).
#[polydat::polydat_node(category = Arithmetic)]
fn reg_lane_i64(r: [i64; 2], i: u64) -> i64 {
    lane_i64(Bits128::from_lanes_i64(r), i)
}

// =================================================================
// Element-wise arithmetic (integer ops wrap; floats are IEEE)
// =================================================================

#[polydat::polydat_node(category = Arithmetic)]
fn reg_add_f32(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    core::array::from_fn(|i| a[i] + b[i])
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_sub_f32(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    core::array::from_fn(|i| a[i] - b[i])
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_mul_f32(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    core::array::from_fn(|i| a[i] * b[i])
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_add_f64(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    core::array::from_fn(|i| a[i] + b[i])
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_sub_f64(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    core::array::from_fn(|i| a[i] - b[i])
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_mul_f64(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    core::array::from_fn(|i| a[i] * b[i])
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_add_i8(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
    core::array::from_fn(|i| a[i].wrapping_add(b[i]))
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_sub_i8(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
    core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
}

/// `reg_mul_i8(a, b)` — the wrapping product per lane. x86 has no
/// byte-lane multiply short of AVX-512 and Cranelift lowers none, so
/// native code runs this body through a helper.
#[polydat::polydat_node(category = Arithmetic)]
fn reg_mul_i8(a: [i8; 16], b: [i8; 16]) -> [i8; 16] {
    mul_i8(Bits128::from_lanes_i8(a), Bits128::from_lanes_i8(b)).lanes_i8()
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_add_i16(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
    core::array::from_fn(|i| a[i].wrapping_add(b[i]))
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_sub_i16(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
    core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_mul_i16(a: [i16; 8], b: [i16; 8]) -> [i16; 8] {
    core::array::from_fn(|i| a[i].wrapping_mul(b[i]))
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_add_i32(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
    core::array::from_fn(|i| a[i].wrapping_add(b[i]))
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_sub_i32(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
    core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_mul_i32(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
    core::array::from_fn(|i| a[i].wrapping_mul(b[i]))
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_add_i64(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
    core::array::from_fn(|i| a[i].wrapping_add(b[i]))
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_sub_i64(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
    core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
}

#[polydat::polydat_node(category = Arithmetic)]
fn reg_mul_i64(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
    core::array::from_fn(|i| a[i].wrapping_mul(b[i]))
}

// =================================================================
// Horizontal + raw-word ops
// =================================================================

/// `reg_dot_f32(a, b)` — horizontal dot product with the FIXED
/// reduction tree `(l0*r0 + l1*r1) + (l2*r2 + l3*r3)`. The tree
/// shape is part of the contract (determinism D2): every host and
/// every execution tier produces bit-identical results.
#[polydat::polydat_node(category = Arithmetic)]
fn reg_dot_f32(a: [f32; 4], b: [f32; 4]) -> f64 {
    let p0 = a[0] * b[0];
    let p1 = a[1] * b[1];
    let p2 = a[2] * b[2];
    let p3 = a[3] * b[3];
    ((p0 + p1) + (p2 + p3)) as f64
}

/// `reg_shuffle_bytes(x, mask)` — arbitrary byte permutation of
/// the raw word: output byte `i` is input byte `mask[i]`. The
/// mask is a 16-entry const list, each entry < 16 (panic
/// otherwise, at build time). Duplicate indices broadcast; this
/// is the SWAR / state-word workhorse for lane rearrangement
/// under any view. Native code bakes the mask into one `shuffle`
/// instruction, so the node exposes it as its constants.
#[polydat::polydat_node(category = Arithmetic, jit_constants = reg_shuffle_bytes_jit_constants)]
fn reg_shuffle_bytes(x: Bits128, mask: polydat::derive_support::Const<&[u64]>) -> Bits128 {
    let m = &*mask;
    if m.len() != 16 {
        panic!(
            "reg_shuffle_bytes: mask must have exactly 16 entries, got {}",
            m.len()
        );
    }
    let src = x.to_le_bytes();
    let mut out = [0u8; 16];
    for (i, &idx) in m.iter().enumerate() {
        if idx >= 16 {
            panic!("reg_shuffle_bytes: mask[{i}] = {idx} out of range 0..16");
        }
        out[i] = src[idx as usize];
    }
    Bits128::from_le_bytes(out)
}

fn reg_shuffle_bytes_jit_constants(node: &RegShuffleBytes) -> Vec<u64> {
    node.mask.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval1<N: PolydatNode>(node: &N, a: Value) -> Value {
        let mut out = [Value::None];
        node.eval(&[a], &mut out);
        out[0].clone()
    }

    fn eval2<N: PolydatNode>(node: &N, a: Value, b: Value) -> Value {
        let mut out = [Value::None];
        node.eval(&[a, b], &mut out);
        out[0].clone()
    }

    fn f32x4(l: [f32; 4]) -> Value {
        Value::Reg128(Bits128::from_lanes_f32(l), RegLanes::F32x4)
    }

    #[test]
    fn lane_codecs_round_trip_and_share_bits() {
        let b = Bits128::from_lanes_f32([1.0, -2.5, 0.0, 4.0]);
        assert_eq!(b.lanes_f32(), [1.0, -2.5, 0.0, 4.0]);
        // The same bits under another view round-trip unchanged —
        // views really are bitcasts.
        let as_i16 = b.lanes_i16();
        assert_eq!(Bits128::from_lanes_i16(as_i16), b);
        assert_eq!(Bits128::from_lanes_i8(b.lanes_i8()), b);
        assert_eq!(Bits128::from_lanes_i64(b.lanes_i64()), b);
        assert_eq!(Bits128::from_lanes_f64(b.lanes_f64()), b);
        assert_eq!(Bits128::from_lanes_f16(b.lanes_f16()), b);
    }

    #[test]
    fn reg_view_retags_without_touching_bits() {
        let word = f32x4([1.0, 2.0, 3.0, 4.0]);
        let raw = eval1(&RegView::new(PortType::Reg128), word.clone());
        assert_eq!(raw.as_reg_bits(), word.as_reg_bits());
        assert!(matches!(raw, Value::Reg128(_, RegLanes::Raw)));
        let back = eval1(&RegView::new(PortType::RegI16x8), raw);
        assert!(matches!(back, Value::Reg128(_, RegLanes::I16x8)));
        assert_eq!(back.as_reg_bits(), word.as_reg_bits());
    }

    #[test]
    fn splats_and_lane_access() {
        let r = eval1(&RegSplatF32::new(), Value::F64(2.5));
        assert_eq!(r.as_reg_bits().lanes_f32(), [2.5; 4]);

        let r = eval1(&RegSplatI16::new(), Value::U64(0xFFFF));
        // bit-level constructor: 0xFFFF as i16 = -1 in every lane
        assert_eq!(r.as_reg_bits().lanes_i16(), [-1; 8]);

        let lane = eval2(
            &RegLaneF32::new(),
            f32x4([1.0, 2.0, 3.0, 4.0]),
            Value::U64(2),
        );
        assert_eq!(lane, Value::F64(3.0));

        let mut out = [Value::None];
        RegWithLaneF32::new().eval(
            &[f32x4([1.0, 2.0, 3.0, 4.0]), Value::U64(1), Value::F64(9.0)],
            &mut out,
        );
        assert_eq!(out[0].as_reg_bits().lanes_f32(), [1.0, 9.0, 3.0, 4.0]);
    }

    #[test]
    fn elementwise_arithmetic_and_wrapping() {
        let sum = eval2(
            &RegAddF32::new(),
            f32x4([1.0, 2.0, 3.0, 4.0]),
            f32x4([10.0, 20.0, 30.0, 40.0]),
        );
        assert_eq!(sum.as_reg_bits().lanes_f32(), [11.0, 22.0, 33.0, 44.0]);

        // i16 lanes wrap (modular lane arithmetic).
        let a = Value::Reg128(Bits128::from_lanes_i16([i16::MAX; 8]), RegLanes::I16x8);
        let b = Value::Reg128(Bits128::from_lanes_i16([1; 8]), RegLanes::I16x8);
        let wrapped = eval2(&RegAddI16::new(), a, b);
        assert_eq!(wrapped.as_reg_bits().lanes_i16(), [i16::MIN; 8]);
    }

    #[test]
    fn dot_uses_fixed_reduction_tree() {
        let d = eval2(
            &RegDotF32::new(),
            f32x4([1.0, 2.0, 3.0, 4.0]),
            f32x4([5.0, 6.0, 7.0, 8.0]),
        );
        // ((5 + 12) + (21 + 32)) = 70 — exact in f32.
        assert_eq!(d, Value::F64(70.0));
    }

    #[test]
    fn gather_and_vec_round_trip() {
        use polydat::ast::SliceArc;
        let v = Value::VecF32(SliceArc::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]));
        let r = eval2(&RegGatherF32::new(), v, Value::U64(2));
        assert_eq!(r.as_reg_bits().lanes_f32(), [2.0, 3.0, 4.0, 5.0]);

        let back = eval1(&RegToVecF32::new(), r);
        assert_eq!(back.as_vec_f32(), &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn raw_state_word_byte_shuffle() {
        // Reverse all 16 bytes via the const mask — exercising the
        // raw view as algorithm-defined buffer state.
        let word = Bits128::from_le_bytes(core::array::from_fn(|i| i as u8));
        let node = RegShuffleBytes::new((0..16).rev().collect());
        let mut out = [Value::None];
        node.eval(&[Value::Reg128(word, RegLanes::Raw)], &mut out);
        let shuffled = out[0].as_reg_bits().to_le_bytes();
        assert_eq!(shuffled, core::array::from_fn(|i| 15 - i as u8));
    }

    #[test]
    fn reg_flow_p1_p2_equivalence() {
        // A register dataflow through splat → add → view-retag →
        // lane read, compiled both ways. P2 rides the two-slot
        // limb protocol (§8.4 layer 1); results must be
        // bit-identical with typed eval.
        let src = r#"
            input cycle: u64
            a := reg_splat_i16(cycle)
            b := reg_splat_i16(3)
            s := reg_add_i16(a, b)
            out := reg_lane_i16(s, 7)
        "#;
        let p1 = polydat::dsl::compile_polydat(src).unwrap();
        let asm = polydat::dsl::compile::compile_polydat_to_assembler(src).unwrap();
        let mut p2 = asm
            .compile_slots(polydat_core::Engine::Closures(
                polydat_core::Provenance::Raw,
            ))
            .expect("reg nodes are P2-eligible");

        let mut k1 = p1;
        for cycle in [0u64, 5, 0xFFFF, 0x1_0005] {
            k1.set_inputs(&[cycle]);
            let want = k1.pull("out").as_i64();
            let got = p2.eval_for_slot(&[cycle], p2.resolve_output("out").unwrap());
            assert_eq!(got as i64, want, "cycle={cycle}");
        }
    }

    #[test]
    fn u128_rides_p2_limb_pairs() {
        // 128-bit integers ride the same two-slot limb protocol
        // (§8.4 layer 1): widen u64 → u128, then range-narrow
        // back, through compiled u64 buffers. Built via the
        // programmatic API because the widening adapters are
        // assembler-inserted (`__`-prefixed), not DSL-callable.
        use polydat::compile::assembly::{PolydatAssembler, WireRef};
        let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
        asm.add_node(
            "wide",
            Box::new(polydat::library::polyfill_128::U64ToU128::new()),
            vec![WireRef::input("cycle")],
        );
        asm.add_node(
            "back",
            Box::new(polydat::library::polyfill_128::U128ToU64::new()),
            vec![WireRef::node("wide")],
        );
        asm.add_output("back", WireRef::node("back"));
        let mut p2 = asm
            .compile_slots(polydat_core::Engine::Closures(
                polydat_core::Provenance::Raw,
            ))
            .expect("u128 nodes are P2-eligible");
        for v in [0u64, 1, u64::MAX] {
            let slot = p2.resolve_output("back").unwrap();
            assert_eq!(p2.eval_for_slot(&[v], slot), v, "u128 round trip of {v}");
        }
    }

    /// P1 ↔ P3 equivalence for every register lane family × op:
    /// splat → binop kernels compiled to native cranelift SIMD
    /// must be bit-identical with typed eval. Families whose
    /// vector lowering the host cranelift declines (e.g. i64x2 /
    /// i8x16 multiplies on pre-AVX512 x86) fall back to the
    /// hybrid kernel — which still runs the vector ops as native
    /// JIT segments where supported — and must match there.
    #[cfg(feature = "jit")]
    #[test]
    fn reg_ops_p1_p3_equivalence_all_lane_families() {
        // Operands route through `hash(cycle)` rather than
        // `cycle + N`: a literal operand lowers to a const node
        // that (pre-existing) classifies Fallback, which would
        // decline the whole pure-P3 kernel and mask the SIMD
        // path under test.
        let cases = [
            ("i8", "cycle", "hash(cycle)"),
            ("i16", "cycle", "hash(cycle)"),
            ("i32", "cycle", "hash(cycle)"),
            ("i64", "cycle", "hash(cycle)"),
            ("f32", "unit_interval(cycle)", "unit_interval(hash(cycle))"),
            ("f64", "unit_interval(cycle)", "unit_interval(hash(cycle))"),
        ];
        for (fam, ea, eb) in cases {
            for op in ["add", "sub", "mul"] {
                let src = format!(
                    "input cycle: u64
                     a := reg_splat_{fam}({ea})
                     b := reg_splat_{fam}({eb})
                     out := reg_{op}_{fam}(a, b)"
                );
                let mut p1 = polydat::dsl::compile_polydat(&src).unwrap();

                for cycle in [0u64, 5, 0xFFFF, 0xDEAD_BEEF] {
                    p1.set_inputs(&[cycle]);
                    let want = p1.pull("out").as_reg_bits();

                    let asm = polydat::dsl::compile::compile_polydat_to_assembler(&src).unwrap();
                    match asm.compile_slots(polydat_core::Engine::PureNative(
                        polydat_core::Provenance::Raw,
                    )) {
                        Ok(mut p3) => {
                            let slot = p3.resolve_output("out").unwrap();
                            p3.eval_at(&[cycle]);
                            let got = Bits128([p3.get_slot(slot), p3.get_slot(slot + 1)]);
                            assert_eq!(got, want, "P3 reg_{op}_{fam} mismatch at cycle={cycle}");
                        }
                        Err(e) => {
                            // Host cranelift declined the vector
                            // lowering — hybrid must still agree.
                            eprintln!("reg_{op}_{fam}: pure-P3 declined ({e}); checking hybrid");
                            let asm =
                                polydat::dsl::compile::compile_polydat_to_assembler(&src).unwrap();
                            let mut hy = asm
                                .compile_slots(polydat_core::Engine::Native(
                                    polydat_core::Provenance::PushPull,
                                ))
                                .unwrap();
                            let slot = hy.resolve_output("out").unwrap();
                            hy.eval_at(&[cycle]);
                            let got = Bits128([hy.get_slot(slot), hy.get_slot(slot + 1)]);
                            assert_eq!(
                                got, want,
                                "hybrid reg_{op}_{fam} mismatch at cycle={cycle}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn display_and_json_forms() {
        let word = f32x4([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(word.to_display_string(), "[1.0, 2.0, 3.0, 4.0]");
        assert_eq!(
            word.to_json_value(),
            serde_json::json!([1.0, 2.0, 3.0, 4.0])
        );
        let raw = Value::Reg128(Bits128::from_u128(0xDEAD), RegLanes::Raw);
        assert_eq!(raw.to_display_string(), format!("{:032x}", 0xDEADu128));
    }
}
