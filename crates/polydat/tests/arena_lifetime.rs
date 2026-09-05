// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 115 §4 and axiom H5: the thread's cycle arena belongs to the root
//! cycle. A root state's cycle advance resets it and advances the
//! generation; nested states (traversal activations, projection bodies)
//! run inside that cycle and never reset it.

use polydat::dsl::compile_polydat;
use polydat::kernel::{cycle_arena_used, cycle_generation, put_thread_str, resolve_thread_str};

#[test]
fn a_root_cycle_advance_resets_the_arena_and_advances_the_generation() {
    let mut k = compile_polydat("input cycle: u64\nx := cycle * 2\n").unwrap();
    k.set_inputs(&[1]);
    let g0 = cycle_generation();
    let h = put_thread_str("hello arena");
    assert_eq!(resolve_thread_str(h), "hello arena");
    assert!(cycle_arena_used() >= 11);
    // The next cycle reclaims the bytes and moves the generation.
    k.set_inputs(&[2]);
    assert_eq!(cycle_arena_used(), 0);
    assert_eq!(cycle_generation(), g0 + 1);
    assert_eq!(k.pull("x").as_u64(), 4);
    // A single input write is a cycle advance too, as the binary's run
    // loop drives coordinates one slot at a time.
    put_thread_str("again");
    k.state().set_input(0, polydat::ast::Value::U64(3));
    assert_eq!(cycle_arena_used(), 0);
    assert_eq!(cycle_generation(), g0 + 2);
}

#[test]
fn a_nested_state_never_resets_the_arena() {
    // A projection body drives a nested state per tuple with its own
    // input writes; none of them may reset the enclosing cycle's arena.
    let src = "input cycle: u64\ntile t : text := \"@for k in 0..4 sep \\\",\\\" {${k + cycle}}\"\n";
    let mut k = compile_polydat(src).unwrap();
    k.set_inputs(&[10]);
    let g = cycle_generation();
    let h = put_thread_str("held across the render");
    let before = cycle_arena_used();
    assert_eq!(k.pull("t").as_str(), "10,11,12,13");
    assert!(cycle_arena_used() >= before, "a nested state reset the arena");
    assert_eq!(cycle_generation(), g, "a nested state advanced the generation");
    assert_eq!(resolve_thread_str(h), "held across the render");
}

#[test]
fn traversal_activations_run_inside_the_root_cycle() {
    let src = "input cycle: u64\nfor k in 1..3 {\n  y := k * 10\n}\n";
    let mut root = polydat::kernel::PolydatKernel::over(compile_polydat(src).unwrap().into_program());
    root.set_inputs(&[0]);
    let g = cycle_generation();
    let h = put_thread_str("root cycle bytes");
    let streams = root.traverse_all().unwrap();
    let mut seen = Vec::new();
    for s in &streams {
        for i in 0..s.len() {
            let mut act = s.activation(i).unwrap();
            act.kernel.set_inputs(&[0]);
            seen.push(act.kernel.pull("y").as_u64());
        }
    }
    assert_eq!(seen, vec![10, 20]);
    assert_eq!(cycle_generation(), g, "an activation advanced the root's generation");
    assert_eq!(resolve_thread_str(h), "root cycle bytes");
}
