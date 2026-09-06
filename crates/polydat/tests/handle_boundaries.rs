// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 115 §5, step 4: byte-string handles cross cone boundaries. A
//! `Str` boundary input is copied into the cycle arena, a `Str`
//! boundary output is copied out to an owned value, and so the string
//! lowerings that existed but never ran now run, bit-identical to P1
//! (axioms H6 and H7).

#![cfg(feature = "jit")]

use polydat::dsl::compile::compile_polydat_to_assembler;
use polydat::kernel::{cycle_arena_used, PolydatKernel};
use polydat::JitMode;

fn kernel(src: &str, mode: JitMode) -> PolydatKernel {
    let mut asm = compile_polydat_to_assembler(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    asm.set_jit_mode(mode);
    asm.compile().unwrap_or_else(|e| panic!("{e}\n{src}"))
}

/// The names of the fused cones in a program, each listing its members.
fn cones(k: &PolydatKernel) -> Vec<String> {
    let p = k.program();
    (0..p.node_count()).map(|i| p.node_meta(i).name.clone()).filter(|n| n.starts_with("jit_cone[")).collect()
}

/// P1 and forced-cone renderings of `outputs` agree over `cycles`, and
/// every name in `fused` sits inside a cone.
fn agree(src: &str, outputs: &[&str], cycles: u64, fused: &[&str]) {
    let mut p1 = kernel(src, JitMode::Off);
    let mut p3 = kernel(src, JitMode::Force);
    let names = cones(&p3);
    for member in fused {
        assert!(names.iter().any(|c| c.contains(member)), "`{member}` was not fused; cones: {names:?}\n{src}");
    }
    for c in 0..cycles {
        p1.set_inputs(&[c]);
        p3.set_inputs(&[c]);
        for out in outputs {
            let a = p1.pull(out).clone();
            let b = p3.pull(out).clone();
            assert_eq!(a.port_type(), b.port_type(), "{out} at cycle {c}: type");
            assert_eq!(a.to_display_string(), b.to_display_string(), "{out} at cycle {c}\n{src}");
        }
    }
}

#[test]
fn scalar_to_string_conversions_run_in_cones() {
    agree("input cycle: u64\nh := hash(cycle)\ns := __u64_to_string(h)\n", &["s"], 6, &["__u64_to_string"]);
    agree("input cycle: u64\nf := to_f64(hash(cycle)) / 7.0\ns := __f64_to_string(f)\n", &["s"], 6, &["__f64_to_string"]);
    agree("input cycle: u64\nb := u64_gt(hash(cycle), 1000)\ns := __bool_to_str(b)\n", &["s"], 6, &["__bool_to_str"]);
}

#[test]
fn string_operations_chain_inside_one_cone() {
    // hash -> to string -> upper -> lower: one cone, one boundary in
    // (cycle) and one boundary out (a Str copied out).
    let src = "input cycle: u64\ns := str_upper(__u64_to_string(hash(cycle)))\nt := str_lower(s)\n";
    agree(src, &["s", "t"], 6, &["str_upper", "str_lower"]);
    let src = "input cycle: u64\na := __u64_to_string(hash(cycle))\nb := __u64_to_string(cycle)\nc := str_concat(a, b)\n";
    agree(src, &["c"], 6, &["str_concat"]);
}

#[test]
fn a_string_literal_feeds_a_cone_as_a_boundary_handle() {
    // The literal is a compile-time constant, which the fold passes
    // own (a cone with no inputs is never planned), so it reaches the
    // cone as a boundary input: a Str copied into the arena once per
    // cycle, concatenated natively, and copied out.
    let src = "input cycle: u64\nlabel := \"row-\"\nn := __u64_to_string(hash(cycle))\nline := str_concat(label, n)\n";
    agree(src, &["line"], 6, &["str_concat"]);
    let mut p3 = kernel(src, JitMode::Force);
    p3.set_inputs(&[2]);
    assert!(p3.pull("line").as_str().starts_with("row-"));
}

#[test]
fn a_string_round_trips_through_parse_and_widening() {
    let src = "input cycle: u64\ns := __u64_to_string(hash(cycle))\nn := __str_to_u64(s)\nf := __str_to_f64(s)\n";
    agree(src, &["n", "f"], 6, &["__str_to_u64", "__str_to_f64"]);
    let mut p3 = kernel(src, JitMode::Force);
    p3.set_inputs(&[3]);
    let mut p1 = kernel(src, JitMode::Off);
    p1.set_inputs(&[3]);
    assert_eq!(p3.pull("n").as_u64(), p1.pull("n").as_u64());
}

#[test]
fn a_str_boundary_input_enters_the_arena_and_the_next_cycle_reclaims_it() {
    // `printf` stays on P1 (variadic, untyped), so its Str output is a
    // cone boundary input for the string ops that follow.
    let src = "input cycle: u64\nw := \"w{cycle}\"\nu := str_upper(w)\nl := str_len_or_upper(u)\n";
    let src = src.replace("l := str_len_or_upper(u)\n", "");
    let mut p3 = kernel(&src, JitMode::Force);
    assert!(cones(&p3).iter().any(|c| c.contains("str_upper")), "{:?}", cones(&p3));
    p3.set_inputs(&[7]);
    assert_eq!(p3.pull("u").as_str(), "W7");
    assert!(cycle_arena_used() > 0, "the boundary copy and the helper's result live in the arena");
    p3.set_inputs(&[8]);
    assert_eq!(cycle_arena_used(), 0, "the root cycle advance reclaimed the arena");
    assert_eq!(p3.pull("u").as_str(), "W8");
}

#[test]
fn an_untyped_variadic_edge_keeps_the_node_on_p1() {
    // `str_concat` takes any wire untyped at P1; a cone cannot, so a
    // u64 wire on its Str port keeps it out of the cone.
    let src = "input cycle: u64\nh := hash(cycle)\nc := str_concat(h, h)\n";
    agree(src, &["c"], 4, &[]);
    let p3 = kernel(src, JitMode::Force);
    assert!(!cones(&p3).iter().any(|c| c.contains("str_concat")), "{:?}", cones(&p3));
}

#[test]
fn a_non_decimal_format_u64_stays_on_p1_and_agrees() {
    let src = "input cycle: u64\nh := hash(cycle)\nd := format_u64(h, 10)\nx := format_u64(h, 16)\n";
    agree(src, &["d", "x"], 4, &["format_u64"]);
    let mut p3 = kernel(src, JitMode::Force);
    p3.set_inputs(&[1]);
    assert!(p3.pull("x").as_str().starts_with("0x"));
}

#[test]
fn a_failed_parse_is_a_diagnostic_at_every_tier() {
    let src = "input cycle: u64\ns := \"nope\"\nn := __str_to_u64(s)\n";
    for mode in [JitMode::Off, JitMode::Force] {
        let mut k = kernel(src, mode);
        k.set_inputs(&[0]);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| k.pull("n").as_u64()));
        assert!(r.is_err(), "{mode:?} accepted an unparseable string");
    }
}
