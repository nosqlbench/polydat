// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 115 §6, step 6: the nodes that inspect `Value` variants at P1
//! (`printf`, the JSON constructors, `to_json`, `json_text`, the tile
//! nodes) lower with their wire types fixed at classification and run
//! the same body natively. Every result agrees with P1 (axiom H7); a
//! wire of a type no helper takes, and a tile with a projection, keep
//! their node on P1.

#![cfg(feature = "jit")]

use polydat::ast::Value;
use polydat::dsl::compile::compile_polydat_to_assembler;
use polydat::kernel::{cycle_arena_used, PolydatKernel};
use polydat::JitMode;

fn kernel(src: &str, mode: JitMode) -> PolydatKernel {
    let mut asm = compile_polydat_to_assembler(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    asm.set_jit_mode(mode);
    asm.compile().unwrap_or_else(|e| panic!("{e}\n{src}"))
}

fn cones(k: &PolydatKernel) -> Vec<String> {
    let p = k.program();
    (0..p.node_count()).map(|i| p.node_meta(i).name.clone()).filter(|n| n.starts_with("jit_cone[")).collect()
}

/// True when a node of this name still stands in the program on its
/// own, that is, was not taken into a cone.
fn standalone(k: &PolydatKernel, name: &str) -> bool {
    let p = k.program();
    (0..p.node_count()).any(|i| p.node_meta(i).name == name)
}

/// P1 and the fused kernel agree on every output over `cycles`; every
/// name in `fused` sits inside a cone and every name in `not_fused`
/// does not.
fn agree(src: &str, outputs: &[&str], cycles: u64, fused: &[&str], not_fused: &[&str]) -> PolydatKernel {
    let mut p1 = kernel(src, JitMode::Off);
    let mut p3 = kernel(src, JitMode::Force);
    let names = cones(&p3);
    for member in fused {
        assert!(!standalone(&p3, member), "`{member}` was not fused; cones: {names:?}\n{src}");
    }
    for member in not_fused {
        assert!(standalone(&p3, member), "`{member}` was fused; cones: {names:?}\n{src}");
    }
    for c in 0..cycles {
        p1.set_inputs(&[c]);
        p3.set_inputs(&[c]);
        for out in outputs {
            let a = p1.pull(out).clone();
            let b = p3.pull(out).clone();
            assert_eq!(a.port_type(), b.port_type(), "{out} at cycle {c}: type\n{src}");
            assert_eq!(a.to_display_string(), b.to_display_string(), "{out} at cycle {c}\n{src}");
        }
    }
    p3
}

const WIRES: &str = "input cycle: u64\nh := hash(cycle)\nf := to_f64(h) / 7.0\nb := u64_gt(h, 5)\ns := __u64_to_string(h)\n";

#[test]
fn printf_lowers_with_its_wire_types_and_every_spec_agrees() {
    let src = format!("{WIRES}out := printf(\"id={{:05}} hex={{:x}} HEX={{:X}} bin={{:b}} oct={{:o}} f={{:.3}} f2={{}} b={{}} s={{:>24}}|{{}} w={{:8}} lit={{{{}}}}\", cycle, h, h, h, h, f, f, b, s, h, cycle)\n");
    agree(&src, &["out"], 8, &["printf"], &[]);
}

#[test]
fn printf_on_a_p1_produced_string_takes_it_as_a_boundary_input() {
    // `str_concat` with a u64 wire stays on P1 (untyped edge), so its
    // Str output enters the printf cone as a boundary handle.
    let src = "input cycle: u64\nh := hash(cycle)\nc := str_concat(h, \"-x\")\nout := printf(\"[{:>30}]\", c)\n";
    agree(src, &["out"], 5, &["printf"], &["str_concat"]);
}

#[test]
fn json_constructors_lower_and_agree() {
    let src = format!("{WIRES}arr := json_array(h, f, b, s, cycle)\ntxt := json_to_str(arr)\n");
    agree(&src, &["arr", "txt"], 6, &["json_array", "json_to_str"], &[]);
    // `json_with` has no lowering, so its Json outputs enter the
    // `json_object` cone as table-kind boundary inputs.
    let src = format!("{WIRES}obj := json_object(json_with(\"id\", h), json_with(\"name\", s), json_with(\"ok\", b))\nt := json_to_str(obj)\n");
    let mut p3 = agree(&src, &["obj", "t"], 6, &["json_object"], &["json_with"]);
    p3.set_inputs(&[3]);
    let obj = p3.pull("obj").clone();
    assert!(matches!(obj, Value::Json(_)));
    assert!(obj.to_display_string().contains("\"name\""));
}

#[test]
fn value_port_nodes_join_a_cone_only_behind_a_member() {
    // `to_json` and `json_text` take a polymorphic `Value` and tolerate
    // None. Fed by a member (`hash`), no None can reach them and they
    // fuse; fed by a kernel input, they stay on P1 (SRD-74) and agree.
    let src = "input cycle: u64\nj := to_json(hash(cycle))\nt := json_text(j)\n";
    agree(src, &["j", "t"], 6, &["to_json", "json_text"], &[]);
    let src = "input cycle: u64\nj := to_json(cycle)\n";
    agree(src, &["j"], 4, &[], &["to_json"]);
    let src = "input cycle: u64\nj := to_json(__u64_to_string(hash(cycle)))\n";
    agree(src, &["j"], 4, &["to_json"], &[]);
}

#[test]
fn pure_p3_kernels_lower_value_port_nodes_by_port_type() {
    let src = "input cycle: u64\nh := hash(cycle)\nj := to_json(h)\nt := json_text(j)\ns := printf(\"{}/{:x}\", h, h)\n";
    let mut k = compile_polydat_to_assembler(src).unwrap().try_compile_jit().expect("every node lowers");
    let mut p1 = kernel(src, JitMode::Off);
    for c in 0..20u64 {
        k.eval(&[c]);
        p1.set_inputs(&[c]);
        assert_eq!(k.get_value("j").to_display_string(), p1.pull("j").to_display_string());
        assert_eq!(k.get_value("t").to_display_string(), p1.pull("t").to_display_string());
        assert_eq!(k.get_value("s").as_str(), p1.pull("s").as_str());
    }
}

#[test]
fn tiles_without_projections_render_natively_and_agree() {
    let json = format!("{WIRES}tile d : json := {{\"n\": ${{h}}, \"f\": ${{f | .2}}, \"s\": ${{s}}, \"t\": ${{b}}, \"in\": \"x-${{h}}-${{s}}\", \"hex\": ${{h | x}}}}\n");
    agree(&json, &["d"], 6, &["tile_render", "tile_encode"], &[]);
    let csv = format!("{WIRES}c := \"x,y\"\ntile r : csv := \"${{h}},${{f | .1}},${{s}},${{c}},${{b: bool}}\"\n");
    agree(&csv, &["r"], 6, &["tile_render"], &[]);
    let text = format!("{WIRES}tile t : text := \"hello ${{s}} #${{h | 06}} @if b {{yes}} @else {{no}}\"\n");
    agree(&text, &["t"], 6, &["tile_render"], &[]);
}

#[test]
fn a_hole_fed_by_a_kernel_input_keeps_its_encoder_on_p1() {
    // `tile_encode` tolerates None (it writes `null`), so fed by the
    // kernel input it cannot join a cone; the render node still does,
    // taking the encoded text as a boundary input.
    let src = "input cycle: u64\ntile d : json := {\"c\": ${cycle}, \"h\": ${hash(cycle)}}\n";
    let p3 = agree(src, &["d"], 5, &["tile_render"], &[]);
    let n = cones(&p3).len();
    assert!(n >= 1, "{:?}", cones(&p3));
}

#[test]
fn a_tile_with_a_projection_renders_natively_and_agrees() {
    // The tile depends on `cycle` so it is per-cycle work rather than a
    // constant the fold passes take. The projection re-runs its body
    // program per tuple through nested kernels inside the helper.
    let src = "input cycle: u64\ntile t : text := \"${cycle}: @for k in 1..4 sep \\\",\\\" {${k}}\"\n";
    agree(src, &["t"], 3, &["tile_render"], &[]);
}

/// A projection body with its own fusable work: the body program's
/// cone evaluates inside the outer cone's helper, so the cone eval
/// must be re-entrant, and the body's strings must not disturb the
/// outer cone's arena mark.
#[test]
fn a_projection_body_with_cones_renders_inside_a_cone() {
    let src = "input cycle: u64\nh := hash(cycle)\ns := __u64_to_string(h)\ntile t : json := {\"h\": ${h}, \"xs\": [@for k in 1..4 sep \",\" { {\"k\": ${k}, \"hk\": ${hash(k)}, \"s\": ${__u64_to_string(hash(k))}, \"outer\": ${s}} }]}\n";
    let mut p3 = agree(src, &["t"], 5, &["tile_render"], &[]);
    p3.set_inputs(&[2]);
    let before = polydat::kernel::cycle_arena_used();
    let text = p3.pull("t").to_display_string();
    assert!(text.contains("\"hk\":"), "{text}");
    assert_eq!(polydat::kernel::cycle_arena_used(), before, "the cone released what its render took");
}

/// The same tile in a pure-P3 kernel: the helper runs the projection's
/// nested kernels inside the kernel's own root cycle.
#[test]
fn a_projection_tile_lays_out_in_a_pure_p3_kernel() {
    let src = "input cycle: u64\nh := hash(cycle)\ntile t : text := \"${h}: @for k in 1..3 sep \\\"-\\\" {${hash(k)}}\"\n";
    let mut k = compile_polydat_to_assembler(src).unwrap().try_compile_jit().expect("every node lowers");
    let mut p1 = kernel(src, JitMode::Off);
    for c in 0..6u64 {
        k.eval(&[c]);
        p1.set_inputs(&[c]);
        assert_eq!(k.get_value("t").as_str(), p1.pull("t").as_str());
    }
}

#[test]
fn a_hybrid_kernel_begins_a_root_cycle_per_run_so_the_arena_stays_bounded() {
    let src = "input cycle: u64\nh := hash(cycle)\ns := __u64_to_string(h)\nn := __str_to_u64(s)\n";
    let mut hy = compile_polydat_to_assembler(src).unwrap().compile_hybrid().expect("hybrid");
    let n = hy.resolve_output("n").unwrap();
    // The oracle runs first: a P1 root kernel resets the arena at its
    // own cycle advance, which would mask what the hybrid kernel does.
    let mut p1 = kernel(src, JitMode::Off);
    let want: Vec<u64> = (1..500u64)
        .map(|c| {
            p1.set_inputs(&[c]);
            p1.pull("n").as_u64()
        })
        .collect();
    hy.eval(&[1]);
    let used = cycle_arena_used();
    assert!(used > 0, "the string result lives in the arena");
    for c in 2..500u64 {
        hy.eval(&[c]);
        assert_eq!(hy.get_slot(n), want[(c - 1) as usize]);
        // The typed reader copies the string out; the raw reader refuses it.
        assert_eq!(hy.get_value("s").as_str(), want[(c - 1) as usize].to_string());
    }
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| hy.get("s"))).is_err());
    assert_eq!(cycle_arena_used(), used, "each run reset the arena before writing");
}
