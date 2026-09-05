// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Modules defined in the program itself resolve by name in that
//! compile, ahead of the filesystem and the library: `compile_polydat`
//! on a string sees them, and an author's definition shadows a library
//! node of the same name.

use polydat::dsl::compile_polydat;

#[test]
fn a_module_defined_in_the_same_source_resolves_without_a_source_directory() {
    let src = "input cycle: u64\n\
        pair_sum(a: u64, b: u64) -> (total: u64) := {\n\
            total := a + b\n\
        }\n\
        s := pair_sum(cycle, 10)\n\
        t := pair_sum(s, s)\n";
    let mut k = compile_polydat(src).unwrap();
    k.set_inputs(&[5]);
    assert_eq!(k.pull("s").as_u64(), 15);
    assert_eq!(k.pull("t").as_u64(), 30);
}

#[test]
fn a_local_module_shadows_a_library_node_of_the_same_name() {
    // `pick` is a library node; the program's own `pick` wins.
    let src = "input cycle: u64\n\
        pick(n: u64) -> (chosen: u64) := {\n\
            chosen := n * 2\n\
        }\n\
        r := pick(cycle)\n";
    let mut k = compile_polydat(src).unwrap();
    k.set_inputs(&[21]);
    assert_eq!(k.pull("r").as_u64(), 42);
    // Without the definition the library node is still the one called.
    let e = compile_polydat("input cycle: u64\nr := pick(cycle)\n").unwrap_err();
    assert!(e.contains("pick") || e.contains("inputs"), "{e}");
}

#[test]
fn a_local_module_with_a_tile_resolves_too() {
    let src = "input cycle: u64\n\
        card(n: u64) -> (doc: String) := {\n\
            tile doc : json := {\"n\": ${n}, \"twice\": ${n * 2}}\n\
        }\n\
        d := card(cycle + 1)\n";
    let mut k = compile_polydat(src).unwrap();
    k.set_inputs(&[4]);
    assert_eq!(k.pull("d").as_str(), "{\"n\": 5, \"twice\": 10}");
}

#[test]
fn variadic_value_nodes_receive_typed_values() {
    // `json_array` sees numbers as numbers, not as their text.
    let src = "input cycle: u64\nf := to_f64(cycle) / 2.0\n\
        arr := json_array(cycle, f, \"s\")\n";
    let mut k = compile_polydat(src).unwrap();
    k.set_inputs(&[7]);
    let arr: serde_json::Value = serde_json::from_str(&k.pull("arr").to_display_string()).unwrap();
    assert_eq!(arr, serde_json::json!([7, 3.5, "s"]));
}

#[test]
fn as_casts_through_the_adapter_catalog() {
    // Every lossless catalog adapter is reachable through `as`, not only
    // the u64 -> f64 widening.
    let src = "input cycle: u64\nb := u64_gt(cycle, 1)\n\
        j := cycle as json\n\
        s := cycle as str\n\
        t := b as bool\n";
    let mut k = compile_polydat(src).unwrap();
    k.set_inputs(&[3]);
    assert_eq!(k.pull("j").to_display_string(), "3");
    assert_eq!(k.pull("s").as_str(), "3");
    assert!(k.pull("t").as_bool());
    // Narrowing is still refused with the explicit alternatives.
    let e = compile_polydat("input cycle: u64\nf := to_f64(cycle)\nn := f as u64\n").unwrap_err();
    assert!(e.contains("floor_to_u64"), "{e}");
}

#[test]
fn a_producer_bound_inside_a_module_projects_in_its_tiles() {
    let src = "input cycle: u64\n\
        grid(n: u64) -> (cells: String) := {\n\
            axes := for r in 0..2, c in 0..2\n\
            tile cells : text := \"@for axes sep \\\",\\\" {${r * n + c}} | @for axes where {c} > 0 sep \\\",\\\" {${r}}\"\n\
        }\n\
        g := grid(cycle + 10)\n";
    let mut k = compile_polydat(src).unwrap();
    k.set_inputs(&[0]);
    assert_eq!(k.pull("g").as_str(), "0,1,10,11 | 0,1");
    // Two calls bind two producers under their own prefixes.
    let src = format!("{src}h := grid(cycle + 100)\n");
    let mut k = compile_polydat(&src).unwrap();
    k.set_inputs(&[0]);
    assert_eq!(k.pull("h").as_str(), "0,1,100,101 | 0,1");
}
