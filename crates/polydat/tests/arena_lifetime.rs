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
    let src =
        "input cycle: u64\ntile t : text := \"@for k in 0..4 sep \\\",\\\" {${k + cycle}}\"\n";
    let mut k = compile_polydat(src).unwrap();
    k.set_inputs(&[10]);
    let g = cycle_generation();
    let h = put_thread_str("held across the render");
    let before = cycle_arena_used();
    assert_eq!(k.pull("t").as_str(), "10,11,12,13");
    assert!(
        cycle_arena_used() >= before,
        "a nested state reset the arena"
    );
    assert_eq!(
        cycle_generation(),
        g,
        "a nested state advanced the generation"
    );
    assert_eq!(resolve_thread_str(h), "held across the render");
}

#[test]
fn traversal_activations_run_inside_the_root_cycle() {
    let src = "input cycle: u64\nfor k in 1..3 {\n  y := k * 10\n}\n";
    let mut root =
        polydat::kernel::PolydatKernel::over(compile_polydat(src).unwrap().into_program());
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
    assert_eq!(
        cycle_generation(),
        g,
        "an activation advanced the root's generation"
    );
    assert_eq!(resolve_thread_str(h), "root cycle bytes");
}

/// Compiling a program and constructing a kernel open no cycle: both
/// happen inside whatever cycle the thread has open, as when a compiled
/// root opens a traversal whose body compiles on first activation. A
/// build's constant fold once seeded a fresh state through `set_inputs`,
/// which reset the root's arena under it.
#[test]
fn compiling_and_constructing_inside_a_cycle_leave_the_arena_alone() {
    use polydat::Engine;
    use polydat::dsl::compile::compile_polydat_with;
    let mut root = compile_polydat_with(
        "input cycle: u64\ns := \"row-{cycle}\"\n",
        Engine::default(),
    )
    .unwrap();
    root.set_inputs(&[7]);
    assert_eq!(root.pull("s").as_str(), "row-7");
    let g = cycle_generation();
    let h = put_thread_str("held across construction");
    let used = cycle_arena_used();
    // Every way a kernel comes to be, inside the root's cycle: the
    // interpreter with folded constants, the default engine with a
    // traversal whose body is an interpreter program, a kernel created
    // from a program, and a nested one.
    let folded = compile_polydat("input cycle: u64\nk := 41 + 1\nname := \"c-{k}\"\n").unwrap();
    assert_eq!(folded.get_constant("name").unwrap().as_str(), "c-42");
    assert_eq!(
        cycle_generation(),
        g,
        "an interpreter compile advanced the generation"
    );
    let with_body = compile_polydat_with(
        "input cycle: u64\nfor k in 1..3 {\n  y := k * 10\n}\n",
        Engine::default(),
    )
    .unwrap();
    assert_eq!(
        cycle_generation(),
        g,
        "a default-engine compile advanced the generation"
    );
    let program = with_body.into_program();
    assert_eq!(
        cycle_generation(),
        g,
        "into_program advanced the generation"
    );
    let _created = std::sync::Arc::clone(&program).create_kernel();
    assert_eq!(
        cycle_generation(),
        g,
        "create_kernel advanced the generation"
    );
    let _nested = program.create_nested_kernel();
    assert_eq!(
        cycle_generation(),
        g,
        "create_nested_kernel advanced the generation"
    );
    assert!(cycle_arena_used() >= used, "construction reset the arena");
    assert_eq!(resolve_thread_str(h), "held across construction");
    assert_eq!(root.pull("s").as_str(), "row-7");
}
