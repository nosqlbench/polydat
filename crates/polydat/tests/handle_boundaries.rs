// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Strings across engine tiers: every string-producing and
//! string-consuming node agrees with the interpreter whether it runs
//! interpreted, as a closure, or natively, and a string read from any
//! engine is an owned value the reader keeps.
//!
//! A string is a `Ref2` value (jit_boundary.md, axioms S1–S10): a
//! `(ptr, len)` pair into its producing step's own scratch. The native
//! tier carries the pair through a slot call of the producing node's
//! kit (compiled_handles.md §6), so a node with a string port joins
//! cones and segments; `REF_NATIVE` names the fusion assertions that
//! rest on it.

#![cfg(feature = "jit")]

use polydat::JitMode;
use polydat::dsl::compile::compile_polydat_to_assembler;
use polydat::kernel::PolydatKernel;

/// Whether the native tier lowers steps with reference-pair ports.
const REF_NATIVE: bool = true;

fn kernel(src: &str, mode: JitMode) -> PolydatKernel {
    let mut asm = compile_polydat_to_assembler(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    asm.set_jit_mode(mode);
    asm.compile().unwrap_or_else(|e| panic!("{e}\n{src}"))
}

/// The names of the fused cones in a program, each listing its members.
fn cones(k: &PolydatKernel) -> Vec<String> {
    let p = k.program();
    (0..p.node_count())
        .map(|i| p.node_meta(i).name.clone())
        .filter(|n| n.starts_with("jit_cone["))
        .collect()
}

/// P1 and forced-cone renderings of `outputs` agree over `cycles`, and
/// every name in `fused` sits inside a cone.
fn agree(src: &str, outputs: &[&str], cycles: u64, fused: &[&str]) {
    let mut p1 = kernel(src, JitMode::Off);
    let mut p3 = kernel(src, JitMode::Force);
    let names = cones(&p3);
    for member in fused {
        assert!(
            !REF_NATIVE || names.iter().any(|c| c.contains(member)),
            "`{member}` was not fused; cones: {names:?}\n{src}"
        );
    }
    for c in 0..cycles {
        p1.set_inputs(&[c]);
        p3.set_inputs(&[c]);
        for out in outputs {
            let a = p1.pull(out).clone();
            let b = p3.pull(out).clone();
            assert_eq!(a.port_type(), b.port_type(), "{out} at cycle {c}: type");
            assert_eq!(
                a.to_display_string(),
                b.to_display_string(),
                "{out} at cycle {c}\n{src}"
            );
        }
    }
}

#[test]
fn scalar_to_string_conversions_run_in_cones() {
    agree(
        "input cycle: u64\nh := hash(cycle)\ns := __u64_to_string(h)\n",
        &["s"],
        6,
        &["__u64_to_string"],
    );
    agree(
        "input cycle: u64\nf := to_f64(hash(cycle)) / 7.0\ns := __f64_to_string(f)\n",
        &["s"],
        6,
        &["__f64_to_string"],
    );
    agree(
        "input cycle: u64\nb := u64_gt(hash(cycle), 1000)\ns := __bool_to_str(b)\n",
        &["s"],
        6,
        &["__bool_to_str"],
    );
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

/// The string producers with a named native lowering write straight
/// into the step's entry (compiled_handles.md §6): the classifier
/// picks the named op over the slot call, and the bytes agree with the
/// interpreter on the values whose formatting has edges (a signed
/// integer, a float past the exponent threshold, a tiny float, an
/// empty string, a nested document).
#[test]
fn named_string_lowerings_are_chosen_and_agree() {
    use polydat::ast::{PolydatNode, PortType};
    use polydat::compile::jit::{JitOp, classify_node_typed};
    let u = polydat::library::convert::U64ToString::new();
    assert!(matches!(
        classify_node_typed(&u, &[PortType::U64]),
        JitOp::U64ToStr { .. }
    ));
    let f = polydat::library::convert::F64ToString::new();
    assert!(matches!(
        classify_node_typed(&f, &[PortType::F64]),
        JitOp::F64ToStr { .. }
    ));
    let c = polydat::library::string::StrConcat::new(2);
    assert!(matches!(
        classify_node_typed(&c, &[PortType::Str, PortType::Str]),
        JitOp::StrConcat { .. }
    ));
    assert!(
        matches!(
            classify_node_typed(&c, &[PortType::Str, PortType::U64]),
            JitOp::SlotCall { .. }
        ),
        "a mixed concatenation takes the kit, which reads each wire as typed"
    );
    let j = polydat::library::json::JsonToStr::new();
    assert!(matches!(
        classify_node_typed(&j, &[PortType::Json]),
        JitOp::JsonToStr { .. }
    ));
    let _ = j.meta();
    let src = "input cycle: u64\n\
        h := hash(cycle)\n\
        neg := __i64_to_string(__u64_to_i64(mod(h, 1000)))\n\
        big := __f64_to_string(f64_mul(to_f64(h), 1000000000000.0))\n\
        small := __f64_to_string(f64_div(1.0, to_f64(u64_add(h, 1))))\n\
        digits := __u64_to_string(h)\n\
        empty := \"\"\n\
        cat := str_concat(digits, empty, neg, \"|\", big)\n\
        doc := json_object(json_with(\"h\", h), json_with(\"cat\", cat), json_with(\"list\", json_array(small, big, 1)))\n\
        text := json_to_str(doc)\n";
    agree(
        src,
        &["neg", "big", "small", "digits", "cat", "text"],
        8,
        &[],
    );
    // A cone's label lists its first members only, so the fusion of
    // these is checked by their absence as nodes of their own.
    let p3 = kernel(src, JitMode::Force);
    let p = p3.program();
    for name in [
        "__i64_to_string",
        "__f64_to_string",
        "str_concat",
        "json_to_str",
    ] {
        assert!(
            !(0..p.node_count()).any(|i| p.node_meta(i).name == name),
            "`{name}` was not fused; cones: {:?}",
            cones(&p3)
        );
    }
}

#[test]
fn a_string_read_is_an_owned_copy_that_outlives_the_next_write() {
    // A read copies out (the reader never holds a reference into the
    // state's buffers), so a value read before a write is intact after
    // it, and the step's output stands until its input is written.
    let src = "input cycle: u64\nw := \"w{cycle}\"\nu := str_upper(w)\n";
    let mut p3 = kernel(src, JitMode::Force);
    p3.set_inputs(&[7]);
    let first = p3.pull("u").clone();
    assert_eq!(first.as_str(), "W7");
    assert_eq!(
        p3.pull("u").as_str(),
        "W7",
        "a second read is the same value"
    );
    p3.set_inputs(&[8]);
    assert_eq!(p3.pull("u").as_str(), "W8");
    assert_eq!(first.as_str(), "W7", "the earlier read is the reader's own");
}

#[test]
fn an_untyped_variadic_edge_lowers_with_its_wire_types() {
    // `str_concat` takes any wire untyped at P1; its kit is built for
    // the wire types the graph fixed, so a u64 wire on its Str port
    // is read as a u64 inside the cone, as P1 reads it.
    let src = "input cycle: u64\nh := hash(cycle)\nc := str_concat(h, h)\n";
    agree(src, &["c"], 4, &["str_concat"]);
}

#[test]
fn a_non_decimal_format_u64_stays_on_p1_and_agrees() {
    let src =
        "input cycle: u64\nh := hash(cycle)\nd := format_u64(h, 10)\nx := format_u64(h, 16)\n";
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
