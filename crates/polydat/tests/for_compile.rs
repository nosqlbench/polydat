// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 113 step 2: element typing and body compilation. Each `for`
//! body compiles once into a child program keyed by its lexical
//! position, with element names typed from the comprehension and outer
//! wires cascaded from the parent.

use polydat::ast::PortType;
use polydat::dsl::compile_polydat_interpreter;
use polydat::kernel::{InputKind, PolydatProgram};

fn compile(src: &str) -> polydat::kernel::PolydatKernel {
    compile_polydat_interpreter(src).unwrap_or_else(|e| panic!("compile failed: {e}\n{src}"))
}

fn elements(p: &PolydatProgram, idx: usize) -> Vec<(String, PortType)> {
    p.traversals()[idx].elements.clone()
}

#[test]
fn element_types_follow_the_source_table() {
    let k = compile(
        "input cycle: u64\n\
         for a in 1..4, b in 10,20,30, c in 1.5,2.5, d in load,verify, e in true,false, p in partitions(\"*/2\", 100) {\n\
             x := hash(a)\n\
         }\n",
    );
    assert_eq!(
        elements(k.program(), 0),
        vec![
            ("a".to_string(), PortType::U64),
            ("b".to_string(), PortType::U64),
            ("c".to_string(), PortType::F64),
            ("d".to_string(), PortType::Str),
            ("e".to_string(), PortType::Bool),
            ("p".to_string(), PortType::Ext),
        ]
    );
}

#[test]
fn int_among_floats_widens_and_mixed_types_are_an_error() {
    let k = compile("input cycle: u64\nfor m in 1,2.5,3 {\n    x := m\n}\n");
    assert_eq!(elements(k.program(), 0)[0].1, PortType::F64);
    let err = compile_polydat_interpreter("input cycle: u64\nfor m in 1,load {\n    x := m\n}\n")
        .unwrap_err();
    assert!(err.to_string().contains("mixes"), "{err}");
}

#[test]
fn generator_call_sources_take_the_node_return_type() {
    let k = compile("input cycle: u64\nfor g in hash_range(cycle, 10) {\n    x := g\n}\n");
    assert_eq!(elements(k.program(), 0)[0].1, PortType::U64);
}

#[test]
fn body_is_a_child_program_with_iteration_externs() {
    let k = compile(
        "input cycle: u64\nfor k in 1..4, limit in 10,20,30 {\n    f := hash(k)\n    g := u64_add(limit, k)\n}\n",
    );
    let parent = k.program();
    assert_eq!(parent.traversals().len(), 1);
    let t = &parent.traversals()[0];
    assert_eq!(t.span.line, 2);
    let child = &t.program;
    // Elements are typed externs of iteration kind; `cycle` is the coordinate.
    for name in ["k", "limit"] {
        let idx = child
            .find_input(name)
            .unwrap_or_else(|| panic!("child lacks {name}"));
        assert_eq!(child.input_kind(idx), Some(InputKind::IterationExtern));
        assert_eq!(child.input_port_type(name), Some(PortType::U64));
    }
    assert_eq!(child.coord_count(), 1);
    assert_eq!(child.input_name_by_idx(0), Some("cycle"));
    assert!(child.output_names().contains(&"f"));
    assert!(child.output_names().contains(&"g"));
    // The parent compiled without the body's wires.
    assert!(!parent.output_names().contains(&"f"));
}

#[test]
fn outer_wires_cascade_with_the_parents_types() {
    let k = compile(
        "input cycle: u64\nextern scale: f64 = 2.0\nbase := hash(cycle)\nlabel := \"run\"\n\
         for k in 1..4 {\n    f := u64_add(base, k)\n    s := \"{label}-{k}\"\n    z := f64_mul(scale, 1.5)\n}\n",
    );
    let t = &k.program().traversals()[0];
    let mut cascade = t.cascade.clone();
    cascade.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        cascade,
        vec![
            ("base".to_string(), PortType::U64),
            ("label".to_string(), PortType::Str),
            ("scale".to_string(), PortType::F64),
        ]
    );
    for (name, ty) in &cascade {
        assert_eq!(t.program.input_port_type(name), Some(*ty));
    }
}

#[test]
fn traversal_over_a_producer_resolves_its_comprehension() {
    let k = compile(
        "input cycle: u64\nsweep := for k in 1..4, limit in 10,20,30 order halton/5\nfor sweep {\n    f := hash(k)\n    g := u64_add(limit, k)\n}\n",
    );
    let p = k.program();
    assert_eq!(p.producers().len(), 1);
    assert_eq!(p.producers()[0].name, "sweep");
    assert_eq!(
        elements(p, 0),
        vec![
            ("k".to_string(), PortType::U64),
            ("limit".to_string(), PortType::U64)
        ]
    );
    let err =
        compile_polydat_interpreter("input cycle: u64\nfor nowhere {\n    f := hash(cycle)\n}\n")
            .unwrap_err();
    assert!(
        err.to_string().contains("no producer named 'nowhere'"),
        "{err}"
    );
}

#[test]
fn nested_bodies_compile_to_nested_programs_once_each() {
    let k = compile(
        "input cycle: u64\nfor p in partitions(\"*/4\", 1000) {\n    outer := cardinality(p)\n\
         for t in 0..20 {\n        mid := u64_add(outer, t)\n        for d in 0..50 {\n            leaf := u64_add(mid, d)\n        }\n    }\n}\n",
    );
    let top = k.program();
    assert_eq!(top.traversals().len(), 1);
    let level1 = &top.traversals()[0].program;
    assert_eq!(level1.traversals().len(), 1);
    let level2 = &level1.traversals()[0].program;
    assert_eq!(level2.traversals().len(), 1);
    let level3 = &level2.traversals()[0].program;
    assert!(level3.traversals().is_empty());
    // Cascade reaches through levels: `mid` from level 2, `outer` from level 1.
    assert_eq!(
        level2.traversals()[0].cascade,
        vec![("mid".to_string(), PortType::U64)]
    );
    assert_eq!(
        level1.traversals()[0].cascade,
        vec![("outer".to_string(), PortType::U64)]
    );
    // Three programs exist however many tuples the traversal would dispense.
    assert!(level3.output_names().contains(&"leaf"));
}

#[test]
fn body_type_errors_are_reported_at_compile_time() {
    let err = compile_polydat_interpreter(
        "input cycle: u64\nfor name in load,verify {\n    x := u64_add(name, 1)\n}\n",
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("for name in load, verify"),
        "{err}"
    );
    assert!(err.to_string().contains("body failed to compile"), "{err}");
}

#[test]
fn unknown_outer_name_is_an_error_in_the_body() {
    let err = compile_polydat_interpreter(
        "input cycle: u64\nfor k in 1..4 {\n    x := hash(missing)\n}\n",
    )
    .unwrap_err();
    assert!(err.to_string().contains("missing"), "{err}");
}

#[test]
fn body_may_not_declare_another_coordinate() {
    let err = compile_polydat_interpreter(
        "input cycle: u64\nfor k in 1..4 {\n    input other: u64\n    x := hash(other)\n}\n",
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("cannot declare input 'other'"),
        "{err}"
    );
}

#[test]
fn a_body_with_a_cursor_over_an_element_compiles() {
    let k = compile(
        "input cycle: u64\nfor p in partitions(\"*/4\", 1000) {\n    cursor rows = range(0, 1000) over p\n    row := mod_in(cycle, rows.cursor)\n}\n",
    );
    let child = &k.program().traversals()[0].program;
    assert_eq!(child.cursor_schemas().len(), 1);
    assert!(child.output_names().contains(&"row"));
}

/// Validation is a stage of the compile (comprehension_forms.md §5):
/// a comprehension that violates a V-axiom is refused at the statement
/// that names it, with the axiom's error. Here V4: `extrema` needs an
/// index-addressable input, and a filter of a filter is not one.
#[test]
fn a_comprehension_that_violates_a_v_axiom_is_a_compile_error() {
    let err = compile_polydat_interpreter(
        "input cycle: u64\nbase := for k in 1..10\nlow := for base where {k} < 8\ncorner := for low where {k} > 2 order extrema/1\n",
    )
    .unwrap_err();
    assert!(err.to_string().contains("V4"), "{err}");
    assert!(err.to_string().contains("line 4"), "{err}");
}

/// A degenerate composition (comprehension_forms.md §5.8) compiles
/// under a lax compile with a `ComprehensionWarning` in the event log,
/// and a strict compile refuses it with the warning as the error.
#[test]
fn a_degenerate_composition_warns_lax_and_fails_strict() {
    use polydat::dsl::compile::{CompileOptions, compile_polydat_interpreter_with_options};
    use polydat::dsl::events::{CompileEvent, CompileEventLog};
    // `extrema` over a one-axis input collapses to its ends.
    let src = "input cycle: u64\nends := for k in 1..10 order extrema/1\n";
    let mut log = CompileEventLog::new();
    compile_polydat_interpreter_with_options(src, &CompileOptions::default(), Some(&mut log))
        .unwrap();
    let warned = log.events().iter().any(|e| {
        matches!(e, CompileEvent::ComprehensionWarning { line, warning, .. }
            if *line == 2 && warning.contains("one-axis"))
    });
    assert!(warned, "{:?}", log.events());
    let strict = CompileOptions {
        strict: true,
        ..CompileOptions::default()
    };
    let err = compile_polydat_interpreter_with_options(src, &strict, None).unwrap_err();
    assert!(err.to_string().contains("strict mode"), "{err}");
    assert!(err.to_string().contains("line 2"), "{err}");
}

/// `over` naming a wire that is not a spec string or partition-typed
/// is a compile error (for_traversal.md §7), not the first
/// activation's.
#[test]
fn over_naming_a_non_partition_wire_is_a_compile_error() {
    let err = compile_polydat_interpreter(
        "input cycle: u64\nfor k in 1..3 {\n    cursor rows = range(0, 100) over k\n    v := rows.ordinal\n}\n",
    )
    .unwrap_err();
    assert!(err.to_string().contains("`over` names a U64 wire"), "{err}");
    let err = compile_polydat_interpreter(
        "input cycle: u64\ncursor rows = range(0, 100) over 42\nv := rows.ordinal\n",
    )
    .unwrap_err();
    assert!(err.to_string().contains("`over` names a U64 wire"), "{err}");
}
