// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 114 §4: every hole is typed at compile time. The declared type
//! wins and must be reachable through the adapter catalog; otherwise the
//! wire's inferred type stands; the position decides how the encoding
//! writes the value. One test per row of the §4.3 table, one per error
//! case, and one for the `explain tiles` events.

use polydat::dsl::compile_polydat;
use polydat::dsl::events::{CompileEvent, CompileEventLog};

const WIRES: &str = "input cycle: u64\n\
    n := cycle * 3\n\
    f := to_f64(cycle) / 4.0\n\
    s := \"a \\\"q\\\" b\"\n\
    t := u64_gt(cycle, 1)\n\
    j := str_to_json(\"{\\\"k\\\": [1, 2]}\")\n";

fn render(src: &str, cycle: u64, name: &str) -> String {
    let mut k = compile_polydat(src).unwrap_or_else(|e| panic!("compile failed: {e}\n{src}"));
    k.set_inputs(&[cycle]);
    k.pull(name).as_str().to_string()
}

fn err(src: &str) -> String {
    compile_polydat(src)
        .err()
        .unwrap_or_else(|| panic!("expected a compile error\n{src}"))
        .to_string()
}

// §4.3 row: json, value position. The wire's type picks the JSON form.
#[test]
fn json_value_position_by_wire_type() {
    let src = format!(
        "{WIRES}tile d : json := {{\"n\": ${{n}}, \"f\": ${{f | .2}}, \"s\": ${{s}}, \"t\": ${{t}}, \"j\": ${{j}}}}\n"
    );
    assert_eq!(
        render(&src, 2, "d"),
        "{\"n\": 6, \"f\": 0.50, \"s\": \"a \\\"q\\\" b\", \"t\": 1, \"j\": {\"k\":[1,2]}}"
    );
    // A `u64` truth value stays a number unless declared `bool`.
    let src = format!("{WIRES}tile d : json := {{\"t\": ${{t: bool}}}}\n");
    assert_eq!(render(&src, 2, "d"), "{\"t\": true}");
    assert_eq!(render(&src, 0, "d"), "{\"t\": false}");
}

// §4.3 row: json, inside a string literal. Any type renders as escaped text.
#[test]
fn json_string_position_renders_any_type_as_escaped_text() {
    let src = format!(
        "{WIRES}tile d : json := {{\"all\": \"${{n}}|${{f | .1}}|${{s}}|${{t}}|${{j}}\"}}\n"
    );
    assert_eq!(
        render(&src, 2, "d"),
        "{\"all\": \"6|0.5|a \\\"q\\\" b|1|{\\\"k\\\":[1,2]}\"}"
    );
}

// §4.3 row: json, object key. Text, escaped.
#[test]
fn json_object_key_is_text() {
    let src = format!("{WIRES}tile d : json := {{\"${{s}}\": ${{n}}, \"k-${{n}}\": true}}\n");
    assert_eq!(
        render(&src, 1, "d"),
        "{\"a \\\"q\\\" b\": 3, \"k-3\": true}"
    );
}

// §4.3 row: csv field. Text, quoted when needed.
#[test]
fn csv_fields_are_text_quoted_when_needed() {
    let src = format!(
        "{WIRES}c := \"x,y\"\ntile r : csv := \"${{n}},${{f | .1}},${{s}},${{c}},${{t: bool}}\"\n"
    );
    assert_eq!(render(&src, 2, "r"), "6,0.5,\"a \"\"q\"\" b\",\"x,y\",true");
}

// §4.3 row: text anywhere. Display form.
#[test]
fn text_encoding_is_display_form() {
    let src = format!(
        "{WIRES}tile l : text := \"${{n}} ${{f | .2}} ${{s}} ${{t}} ${{t: bool}} ${{j}}\"\n"
    );
    assert_eq!(
        render(&src, 2, "l"),
        "6 0.50 a \"q\" b 1 true {\"k\":[1,2]}"
    );
}

// §4.1: a declared type inserts the catalog adapter. u64 -> f64 widens
// and the encoder sees a float, so a precision format applies.
#[test]
fn declared_type_widens_through_the_adapter_catalog() {
    let src = "input cycle: u64\ntile d : json := {\"w\": ${cycle: f64 | .1}, \"n\": ${cycle}}\n";
    assert_eq!(render(src, 7, "d"), "{\"w\": 7.0, \"n\": 7}");
    // Declared str on a number is the display text, quoted in json.
    let src = "input cycle: u64\ntile d : json := {\"s\": ${cycle: str}}\n";
    assert_eq!(render(src, 7, "d"), "{\"s\": \"7\"}");
    // Declared json on a number serializes it as a JSON value.
    let src = "input cycle: u64\ntile d : json := {\"v\": ${cycle: json}}\n";
    assert_eq!(render(src, 7, "d"), "{\"v\": 7}");
}

// §4.1: string-to-number must be written explicitly.
#[test]
fn declared_type_unreachable_from_wire_type_is_an_error_naming_both() {
    let e = err("input cycle: u64\ns := \"12\"\ntile d : json := {\"n\": ${s: u64}}\n");
    assert!(e.contains("tile 'd'"), "{e}");
    assert!(e.contains("hole `s: u64`"), "{e}");
    assert!(e.contains("str") && e.contains("u64"), "{e}");
    assert!(e.contains("explicitly"), "{e}");
    // Narrowing f64 -> u64 is not an adapter either.
    let e = err("input cycle: u64\nf := to_f64(cycle)\ntile d : text := \"${f: u64}\"\n");
    assert!(e.contains("f64") && e.contains("u64"), "{e}");
    // The explicit form compiles.
    let src = "input cycle: u64\ns := \"12\"\ntile d : json := {\"n\": ${s as u64}}\n";
    assert_eq!(render(src, 0, "d"), "{\"n\": 12}");
}

#[test]
fn unknown_declared_type_is_an_error() {
    let e = err("input cycle: u64\ntile d : text := \"${cycle: integer}\"\n");
    assert!(e.contains("unknown type 'integer'"), "{e}");
}

// §4.4: a wire with no text form cannot fill a hole.
#[test]
fn a_producer_wire_cannot_fill_a_hole() {
    let e = err("input cycle: u64\naxes := for k in 1..3\ntile d : text := \"${axes}\"\n");
    assert!(e.contains("no text form"), "{e}");
}

// §4.4: strict mode rejects the implicit adapter a declaration needs.
#[test]
fn strict_mode_rejects_implicit_adapters_at_holes() {
    let e = err("input cycle: u64\ntile d : json (strict) := {\"w\": ${cycle: f64}}\n");
    assert!(e.contains("strict mode rejects"), "{e}");
    assert!(e.contains("u64 -> f64"), "{e}");
    // The explicit conversion is accepted under strict.
    let src = "input cycle: u64\ntile d : json (strict) := {\"w\": ${to_f64(cycle) | .1}}\n";
    assert_eq!(render(src, 3, "d"), "{\"w\": 3.0}");
    // A declaration that matches the wire type needs no adapter.
    let src = "input cycle: u64\ntile d : json (strict) := {\"n\": ${cycle: u64}}\n";
    assert_eq!(render(src, 3, "d"), "{\"n\": 3}");
}

// §4.2: projection elements and cascaded outer wires carry their types.
#[test]
fn projection_body_holes_are_typed_from_elements_and_outer_wires() {
    let src = "input cycle: u64\nlabel := \"L\"\n\
        tile d : json := {\"xs\": [@for k in 1..3, w in a,b { {\"k\": ${k}, \"w\": ${w}, \"l\": ${label}, \"kf\": ${k: f64 | .1}} }]}\n";
    assert_eq!(
        render(src, 0, "d"),
        "{\"xs\": [{\"k\": 1, \"w\": \"a\", \"l\": \"L\", \"kf\": 1.0},{\"k\": 1, \"w\": \"b\", \"l\": \"L\", \"kf\": 1.0},{\"k\": 2, \"w\": \"a\", \"l\": \"L\", \"kf\": 2.0},{\"k\": 2, \"w\": \"b\", \"l\": \"L\", \"kf\": 2.0}]}"
    );
}

// §4.4: `explain tiles` sees one typing event per hole.
#[test]
fn explain_events_report_wire_type_declaration_expectation_and_encoder() {
    let mut log = CompileEventLog::default();
    let src = format!(
        "{WIRES}tile d : json := {{\"n\": ${{n}}, \"s\": \"x-${{s}}\", \"w\": ${{cycle: f64}}, \"b\": @if t {{ 1 }} @else {{ 0 }}}}\n"
    );
    polydat::dsl::compile::compile_polydat_interpreter_with_log(&src, &mut log).unwrap();
    let holes: Vec<&CompileEvent> = log
        .events()
        .iter()
        .filter(|e| matches!(e, CompileEvent::TileHoleTyped { .. }))
        .collect();
    assert_eq!(holes.len(), 4, "{holes:?}");
    let find = |h: &str| {
        holes
            .iter()
            .find_map(|e| match e {
                CompileEvent::TileHoleTyped {
                    hole,
                    wire_type,
                    declared,
                    expectation,
                    encoder,
                    adapter,
                    tile,
                } if hole == h => Some((
                    tile.clone(),
                    wire_type.clone(),
                    declared.clone(),
                    expectation.clone(),
                    encoder.clone(),
                    adapter.clone(),
                )),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no event for hole {h}: {holes:?}"))
    };
    let (tile, wire, declared, expect, encoder, adapter) = find("n");
    assert_eq!(tile, "d");
    assert_eq!(wire, "u64");
    assert!(declared.is_none());
    assert!(expect.starts_with("any JSON value"), "{expect}");
    assert_eq!(encoder, "json number");
    assert!(adapter.is_none());
    let (_, wire, _, expect, encoder, _) = find("s");
    assert_eq!(wire, "str");
    assert!(expect.starts_with("text inside a JSON string"), "{expect}");
    assert_eq!(encoder, "json escaped text");
    let (_, wire, declared, _, encoder, adapter) = find("cycle: f64");
    assert_eq!(wire, "u64");
    assert_eq!(declared.as_deref(), Some("f64"));
    assert_eq!(encoder, "json number");
    assert_eq!(adapter.as_deref(), Some("u64 -> f64"));
    let (_, _, _, expect, encoder, _) = find("if t");
    assert!(expect.starts_with("a truth value"), "{expect}");
    assert_eq!(encoder, "truth value as 1 or 0");
    // The log's text form names the tile and the hole.
    let text = log.format();
    assert!(text.contains("tile 'd' hole `n`"), "{text}");
}

// §4.1 inside a projection body: every catalog adapter is available,
// through the `as` fusion the body's source text can express.
#[test]
fn body_holes_reach_every_catalog_adapter() {
    let src = "input cycle: u64\ntile t : json := {\"xs\": [@for k in 1..3 { {\"j\": ${k: json}, \"f\": ${k: f64 | .1}, \"s\": ${k: str}} }]}\n";
    assert_eq!(
        render(src, 0, "t"),
        "{\"xs\": [{\"j\": 1, \"f\": 1.0, \"s\": \"1\"},{\"j\": 2, \"f\": 2.0, \"s\": \"2\"}]}"
    );
}
