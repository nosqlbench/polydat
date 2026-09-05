// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 114: tiles compile to wires and render byte-exactly. Holes encode
//! by type and position, formats apply first, branches select,
//! projections repeat over comprehensions with element and outer wires,
//! and splices inline earlier tiles.

use polydat::dsl::compile_polydat;

fn render(src: &str, cycle: u64, name: &str) -> String {
    let mut k = compile_polydat(src).unwrap_or_else(|e| panic!("compile failed: {e}\n{src}"));
    k.set_inputs(&[cycle]);
    k.pull(name).as_str().to_string()
}

#[test]
fn text_tile_with_holes_and_a_raw_hole() {
    let src = "input cycle: u64\nname := \"row-{cycle}\"\ntile line : text := \"id=${cycle} name=${name} twice=${cycle * 2} raw=${name!}\"\n";
    assert_eq!(render(src, 7, "line"), "id=7 name=row-7 twice=14 raw=row-7");
}

#[test]
fn json_tile_encodes_by_wire_type_and_position() {
    let src = "input cycle: u64\nlabel := \"q\\\"uote\"\nflag := u64_gt(cycle, 3)\nratio := to_f64(cycle) / 3.0\n\
               tile doc : json := {\"id\": ${cycle}, \"label\": ${label}, \"inline\": \"v-${label}\", \"ratio\": ${ratio | .2}, \"flag\": ${flag: bool}, \"meta\": {\"schema\": 3, \"units\": {\"t\": \"C\"}}}\n";
    assert_eq!(
        render(src, 5, "doc"),
        "{\"id\": 5, \"label\": \"q\\\"uote\", \"inline\": \"v-q\\\"uote\", \"ratio\": 1.67, \"flag\": true, \"meta\": {\"schema\": 3, \"units\": {\"t\": \"C\"}}}"
    );
    // The rendered document is valid JSON.
    let parsed: serde_json::Value = serde_json::from_str(&render(src, 5, "doc")).unwrap();
    assert_eq!(parsed["meta"]["units"]["t"], "C");
    assert_eq!(parsed["id"], 5);
}

#[test]
fn declared_type_wins_over_wire_type() {
    let src = "input cycle: u64\ntile t : json := {\"as_text\": ${cycle: str}, \"as_num\": ${cycle}}\n";
    assert_eq!(render(src, 9, "t"), "{\"as_text\": \"9\", \"as_num\": 9}");
}

#[test]
fn csv_tile_quotes_only_when_needed() {
    let src = "input cycle: u64\nwords := \"a,b\"\ntile row : csv := \"${cycle},${words},plain\"\n";
    assert_eq!(render(src, 1, "row"), "1,\"a,b\",plain");
}

#[test]
fn formats_apply_before_encoding() {
    let src = "input cycle: u64\ntile t : text := \"${cycle | 04} ${cycle | x} ${to_f64(cycle) / 4.0 | .3}\"\n";
    assert_eq!(render(src, 255, "t"), "0255 ff 63.750");
}

#[test]
fn branches_select_by_condition() {
    let src = "input cycle: u64\nbig := u64_gt(cycle, 10)\ntile t : text := \"n=${cycle} @if big { large } @else { small } end\"\n";
    assert_eq!(render(src, 20, "t"), "n=20 large end");
    assert_eq!(render(src, 2, "t"), "n=2 small end");
}

#[test]
fn projection_over_an_inline_comprehension_with_element_and_outer_wires() {
    let src = "input cycle: u64\nbase := cycle * 100\ntile t : json := {\"samples\": [@for s in 0..3 {{\"n\": ${s}, \"v\": ${base + s}}}]}\n";
    assert_eq!(render(src, 2, "t"), "{\"samples\": [{\"n\": 0, \"v\": 200},{\"n\": 1, \"v\": 201},{\"n\": 2, \"v\": 202}]}");
    let parsed: serde_json::Value = serde_json::from_str(&render(src, 2, "t")).unwrap();
    assert_eq!(parsed["samples"].as_array().unwrap().len(), 3);
}

#[test]
fn projection_over_a_producer_with_a_separator_and_two_elements() {
    let src = "input cycle: u64\ngrid := for k in 1..3, name in a,b\ntile t : text := \"@for grid sep \\\"; \\\" {${k}:${name}}\"\n";
    assert_eq!(render(src, 0, "t"), "1:a; 1:b; 2:a; 2:b");
}

#[test]
fn projection_inside_a_string_position_escapes() {
    let src = "input cycle: u64\ntile t : json := {\"list\": \"@for w in x,y sep \\\"|\\\" {${w}}\"}\n";
    assert_eq!(render(src, 0, "t"), "{\"list\": \"x|y\"}");
}

#[test]
fn splicing_inlines_an_earlier_tile() {
    let src = "input cycle: u64\ntile inner : json := {\"n\": ${cycle}}\ntile outer : json := {\"wrapped\": ${inner}, \"again\": ${inner}}\n";
    assert_eq!(render(src, 4, "outer"), "{\"wrapped\": {\"n\": 4}, \"again\": {\"n\": 4}}");
}

#[test]
fn a_tile_is_an_ordinary_wire() {
    let src = "input cycle: u64\ntile doc : json := {\"n\": ${cycle}}\nstmt := \"INSERT doc '{doc}'\"\nup := str_upper(doc)\n";
    let mut k = compile_polydat(src).unwrap();
    k.set_inputs(&[3]);
    assert_eq!(k.pull("stmt").as_str(), "INSERT doc '{\"n\": 3}'");
    assert_eq!(k.pull("up").as_str(), "{\"N\": 3}");
    assert!(k.program().output_names().contains(&"doc"));
}

#[test]
fn a_tile_without_holes_is_a_constant() {
    let src = "input cycle: u64\ntile fixed : json := {\"schema\": 3}\n";
    let k = compile_polydat(src).unwrap();
    assert!(k.program().const_outputs().contains(&"fixed"));
}

#[test]
fn doubled_open_delimiter_renders_literally() {
    let src = "input cycle: u64\ntile t : text := \"cost ${${amount} is ${cycle}\"\n";
    assert_eq!(render(src, 3, "t"), "cost ${amount} is 3");
}

#[test]
fn custom_delimiters_render_the_same() {
    let src = "input cycle: u64\ntile t : text (delims \"<%\" \"%>\", sigil \"#\") := \"${keep} <%cycle%> #if cycle { on } #else { off }\"\n";
    assert_eq!(render(src, 1, "t"), "${keep} 1 on");
}

#[test]
fn unknown_wire_in_a_hole_is_a_compile_error_naming_the_tile() {
    let err = compile_polydat("input cycle: u64\ntile t : text := \"${missing}\"\n").unwrap_err();
    assert!(err.contains("missing"), "{err}");
}

#[test]
fn splicing_across_encodings_keeps_the_inner_encoding() {
    // A json tile carried inside a text tile renders as json: its
    // strings stay quoted and its projection keeps the `,` separator.
    let src = "input cycle: u64\nword := \"ok\"\ntile doc : json := {\"w\": ${word}, \"xs\": [@for i in 0..2 { ${i} }]}\ntile stmt : text := \"INSERT ${cycle} '${doc!}'\"\n";
    assert_eq!(render(src, 3, "stmt"), "INSERT 3 '{\"w\": \"ok\", \"xs\": [0,1]}'");
    // Without `!` the inner text is a string value of the outer encoding.
    let src = "input cycle: u64\ntile inner : text := \"a \\\"b\\\"\"\ntile outer : json := {\"body\": ${inner}}\n";
    assert_eq!(render(src, 3, "outer"), "{\"body\": \"a \\\"b\\\"\"}");
}

#[test]
fn directive_blocks_trim_padding_and_bare_word_blocks_are_blocks() {
    let src = "input cycle: u64\nflag := u64_gt(cycle, 1)\ntile t : text := \"[@if flag {yes} @else {no}] [@for k in 1..3 sep \\\"-\\\" { ${k} }] @for k in 1..4 where {k} > 2 {${k}}\"\n";
    assert_eq!(render(src, 5, "t"), "[yes] [1-2] 3");
    assert_eq!(render(src, 0, "t"), "[no] [1-2] 3");
}

#[test]
fn cycle_inside_a_projection_body_is_the_program_cycle() {
    let src = "input cycle: u64\ntile t : text := \"@for s in 0..3 sep \\\",\\\" {${cycle + s}}\"\n";
    assert_eq!(render(src, 4, "t"), "4,5,6");
    assert_eq!(render(src, 10, "t"), "10,11,12");
}
