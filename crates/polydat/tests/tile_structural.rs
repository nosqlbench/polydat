// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 114 §3 and §5.6: the structural JSON form renders exactly what
//! the equivalent textual template renders (L5), and a host can hand a
//! tile in as template text, JSON text, a parsed JSON value, or a
//! `polytile` binding in source.

use polydat::dsl::compile_polydat_interpreter;
use polydat::tile::{
    Span, TileOptions, compile_polydat_with_tiles, tile_from_json_text, tile_from_json_value,
    tile_from_text,
};

const PROGRAM: &str = "input cycle: u64\n\
    tenant_id := mod(hash(cycle), 1000)\n\
    device_id := \"dev-{cycle}\"\n\
    temp_c := to_f64(cycle) + 0.5\n\
    verbose := u64_gt(cycle, 2)\n\
    operator := \"ops\"\n\
    tags := for t in a,b\n";

fn render(src: &str, cycle: u64, name: &str) -> String {
    let mut k =
        compile_polydat_interpreter(src).unwrap_or_else(|e| panic!("compile failed: {e}\n{src}"));
    k.set_inputs(&[cycle]);
    k.pull(name).as_str().to_string()
}

const STRUCTURAL: &str = r#"{
  "meta": { "schema": 3, "units": { "temp": "C" } },
  "tenant": "${tenant_id}",
  "device": "${device_id}",
  "label": "row-${cycle}",
  "as_text": "${tenant_id: str}",
  "samples": [ "@for s in 0..3", { "n": "${s}", "temp": "${temp_c + to_f64(s) | .1}" } ],
  "audit": [ "@if verbose", { "by": "${operator}" }, "@else", null ],
  "tags": { "@for tags": { "${t}": true } }
}"#;

const TEXTUAL: &str = r#"{
  "meta": { "schema": 3, "units": { "temp": "C" } },
  "tenant": ${tenant_id},
  "device": ${device_id},
  "label": "row-${cycle}",
  "as_text": ${tenant_id: str},
  "samples": [@for s in 0..3 { {"n": ${s}, "temp": ${temp_c + to_f64(s) | .1}} }],
  "audit": [@if verbose { {"by": ${operator}} } @else { null }],
  "tags": {@for tags { "${t}": true }}
}"#;

fn canonical(s: &str) -> serde_json::Value {
    serde_json::from_str(s).unwrap_or_else(|e| panic!("not JSON: {e}\n{s}"))
}

#[test]
fn structural_and_textual_forms_render_the_same_document() {
    let src_text = format!("{PROGRAM}tile doc : json := {TEXTUAL}\n");
    let src_struct = format!("{PROGRAM}doc := polytile_json(<<<\n{STRUCTURAL}\n>>>)\n");
    for cycle in [0u64, 3] {
        let a = render(&src_text, cycle, "doc");
        let b = render(&src_struct, cycle, "doc");
        assert_eq!(canonical(&a), canonical(&b), "cycle {cycle}\n{a}\n{b}");
    }
    let doc = canonical(&render(&src_struct, 3, "doc"));
    let mut k = compile_polydat_interpreter(&src_struct).unwrap();
    k.set_inputs(&[3]);
    assert_eq!(doc["tenant"], k.pull("tenant_id").as_u64());
    assert_eq!(doc["device"], "dev-3");
    assert_eq!(doc["label"], "row-3");
    assert_eq!(doc["as_text"], doc["tenant"].to_string());
    assert_eq!(doc["samples"].as_array().unwrap().len(), 3);
    assert_eq!(doc["samples"][2]["temp"], 5.5);
    assert_eq!(doc["audit"][0]["by"], "ops");
    assert_eq!(doc["tags"]["a"], true);
    assert_eq!(doc["tags"]["b"], true);
    let quiet = canonical(&render(&src_struct, 0, "doc"));
    assert!(quiet["audit"][0].is_null());
}

#[test]
fn host_builds_a_tile_from_a_parsed_json_value() {
    let value: serde_json::Value = serde_json::from_str(STRUCTURAL).unwrap();
    let tile = tile_from_json_value(
        "doc",
        &value,
        &TileOptions::default(),
        Span { line: 0, col: 0 },
    )
    .unwrap();
    assert_eq!(tile.encoding.as_deref(), Some("json"));
    let mut k = compile_polydat_with_tiles(PROGRAM, vec![tile]).unwrap();
    k.set_inputs(&[3]);
    let doc = canonical(k.pull("doc").as_str());
    assert_eq!(doc["device"], "dev-3");
    assert_eq!(doc["samples"][0]["n"], 0);
    // The tile is a wire like any other.
    assert!(k.program().output_names().contains(&"doc"));
}

#[test]
fn host_builds_tiles_from_json_text_and_template_text() {
    let json = tile_from_json_text(
        "doc",
        r#"{"n": "${cycle}", "s": "${cycle: str}"}"#,
        &TileOptions::default(),
        Span { line: 0, col: 0 },
    )
    .unwrap();
    let text = tile_from_text(
        "line",
        "text",
        "n=${cycle} doc=${doc!}",
        &TileOptions::default(),
        Span { line: 0, col: 0 },
    )
    .unwrap();
    let mut k = compile_polydat_with_tiles("input cycle: u64\n", vec![json, text]).unwrap();
    k.set_inputs(&[7]);
    assert_eq!(k.pull("doc").as_str(), "{\"n\": 7, \"s\": \"7\"}");
    assert_eq!(k.pull("line").as_str(), "n=7 doc={\"n\": 7, \"s\": \"7\"}");
}

#[test]
fn polytile_bindings_in_source_with_string_and_heredoc_bodies() {
    let src = "input cycle: u64\n\
        greeting := polytile(\"text\", \"hello ${cycle}\")\n\
        doc := polytile(\"json\", <<<\n{\"n\": ${cycle}, \"g\": ${greeting}}\n>>>)\n\
        page := polytile(\"text\", \"<%cycle%> {{ keep }} #if cycle { on } #else { off }\", open: \"<%\", close: \"%>\", sigil: \"#\")\n\
        up := str_upper(greeting)\n";
    assert_eq!(render(src, 2, "greeting"), "hello 2");
    assert_eq!(render(src, 2, "doc"), "{\"n\": 2, \"g\": \"hello 2\"}");
    assert_eq!(render(src, 2, "page"), "2 {{ keep }} on");
    assert_eq!(render(src, 2, "up"), "HELLO 2");
}

#[test]
fn heredoc_string_literals_work_as_ordinary_strings_too() {
    let src = "input cycle: u64\nnote := <<<\nline one\nline two {cycle}\n>>>\n";
    assert_eq!(render(src, 4, "note"), "line one\nline two 4");
}

#[test]
fn structural_errors_name_the_problem() {
    let err = compile_polydat_interpreter(
        "input cycle: u64\ndoc := polytile_json(\"{\\\"a\\\": \\\"@for s in 0..4\\\"}\")\n",
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("outside a directive position"),
        "{err}"
    );
    let err = compile_polydat_interpreter(
        "input cycle: u64\ndoc := polytile_json(\"{\\\"a\\\": [\\\"@else\\\", 1]}\")\n",
    )
    .unwrap_err();
    assert!(err.to_string().contains("without a leading"), "{err}");
    let err = compile_polydat_interpreter("input cycle: u64\ndoc := polytile_json(\"not json\")\n")
        .unwrap_err();
    assert!(err.to_string().contains("not JSON"), "{err}");
    let err = compile_polydat_interpreter("input cycle: u64\ndoc := polytile(\"yaml\", \"x\")\n")
        .unwrap_err();
    assert!(err.to_string().contains("unknown encoding 'yaml'"), "{err}");
    let err = compile_polydat_interpreter("input cycle: u64\ndoc := polytile(\"json\", cycle)\n")
        .unwrap_err();
    assert!(err.to_string().contains("template body"), "{err}");
    let err = compile_polydat_interpreter(
        "input cycle: u64\ndoc := polytile(\"json\", \"${missing}\")\n",
    )
    .unwrap_err();
    assert!(err.to_string().contains("missing"), "{err}");
}

#[test]
fn a_hole_in_a_key_and_static_members_beside_a_projection() {
    let src = "input cycle: u64\nk := \"key-{cycle}\"\nns := for n in 1..3\n\
        doc := polytile_json(\"{\\\"fixed\\\": 1, \\\"@for ns\\\": {\\\"${k}-${n}\\\": \\\"${n}\\\"}}\")\n";
    let doc = canonical(&render(src, 5, "doc"));
    assert_eq!(doc["fixed"], 1);
    assert_eq!(doc["key-5-1"], 1);
    assert_eq!(doc["key-5-2"], 2);
}

#[test]
fn pretty_printer_reproduces_a_host_tile_as_a_tile_statement() {
    let tokens = polydat::dsl::lexer::lex(
        "input cycle: u64\ndoc := polytile_json(\"{\\\"n\\\": \\\"${cycle}\\\"}\")\n",
    )
    .unwrap();
    let file = polydat::dsl::parser::parse(tokens).unwrap();
    let printed = polydat::dsl::pprint::pp_file(&file);
    assert!(
        printed.contains("tile doc : json := {\"n\": ${cycle}}"),
        "{printed}"
    );
    // And the printed form compiles to the same output.
    assert_eq!(render(&printed, 9, "doc"), "{\"n\": 9}");
}

#[test]
fn directive_members_beside_static_members_carry_their_own_commas() {
    // Zero tuples beside a static member leaves valid JSON; so does a
    // leading projection followed by static members, and a branch.
    let src = "input cycle: u64\nnone := for k in 1..1\nsome := for k in 1..3\nflag := u64_gt(cycle, 0)\n\
        a := polytile_json(\"{\\\"fixed\\\": 1, \\\"@for none\\\": {\\\"k${k}\\\": \\\"${k}\\\"}}\")\n\
        b := polytile_json(\"{\\\"@for some\\\": {\\\"k${k}\\\": \\\"${k}\\\"}, \\\"fixed\\\": 1}\")\n\
        c := polytile_json(\"{\\\"@for none\\\": {\\\"k${k}\\\": \\\"${k}\\\"}, \\\"fixed\\\": 1}\")\n\
        d := polytile_json(\"{\\\"fixed\\\": 1, \\\"@if flag\\\": {\\\"on\\\": true}, \\\"last\\\": 2}\")\n";
    for (name, cycle, expect) in [
        ("a", 0u64, serde_json::json!({"fixed": 1})),
        ("b", 0, serde_json::json!({"k1": 1, "k2": 2, "fixed": 1})),
        ("c", 0, serde_json::json!({"fixed": 1})),
        ("d", 0, serde_json::json!({"fixed": 1, "last": 2})),
        (
            "d",
            1,
            serde_json::json!({"fixed": 1, "on": true, "last": 2}),
        ),
    ] {
        let text = render(src, cycle, name);
        let doc: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("{name} at cycle {cycle} is not JSON: {e}\n{text}"));
        assert_eq!(doc, expect, "{name} at cycle {cycle}: {text}");
    }
}

/// A tile is its pieces, and pieces compose by concatenation whatever
/// admitted them. Text, a parsed fragment, and hand-built pieces are
/// three input forms for one thing, so a template assembled from any
/// mixture in any order is the same tile as the template written whole.
#[test]
fn text_fragments_and_built_pieces_compose_into_one_tile() {
    use polydat::dsl::ast::{TileBodyKind, TileDef, TilePiece};
    use polydat::dsl::lexer::Span as S;
    use polydat::dsl::tile::parse_template;

    let sp = S { line: 1, col: 1 };
    let opts = TileOptions::default();

    // The same body, written whole.
    let whole = TileDef::from_body(
        "t",
        Some("text".into()),
        opts.clone(),
        TileBodyKind::Literal,
        "id=${tenant_id} t=${temp_c | .2}",
        sp,
    )
    .expect("whole body parses");

    // The same body, assembled: a parsed fragment, then a static run
    // built by hand, then another parsed fragment.
    let mut pieces = parse_template("id=${tenant_id}", &opts, sp).expect("fragment one");
    pieces.push(TilePiece::Static(" t=".into()));
    pieces.extend(parse_template("${temp_c | .2}", &opts, sp).expect("fragment two"));
    let assembled = TileDef::from_pieces(
        "t",
        Some("text".into()),
        opts.clone(),
        TileBodyKind::Literal,
        pieces,
        sp,
    );

    // Same tile: same text, and the text re-reads to the same pieces.
    assert_eq!(assembled.body_text(), whole.body_text());
    assert_eq!(
        parse_template(&assembled.body_text(), &opts, sp)
            .unwrap()
            .len(),
        assembled.pieces.len()
    );

    // And the same rendered bytes, which is what a tile is for.
    let render = |t: &TileDef| {
        let mut k = compile_polydat_with_tiles(PROGRAM, vec![t.clone()]).expect("compiles");
        k.set_inputs(&[3]);
        k.pull("t").to_display_string()
    };
    assert_eq!(render(&assembled), render(&whole));

    // Admitting the pieces in a different order builds a different
    // template, as concatenation should: composition is associative,
    // not commutative.
    let mut swapped = parse_template("${temp_c | .2}", &opts, sp).unwrap();
    swapped.extend(parse_template("id=${tenant_id}", &opts, sp).unwrap());
    let other = TileDef::from_pieces(
        "t",
        Some("text".into()),
        opts.clone(),
        TileBodyKind::Literal,
        swapped,
        sp,
    );
    assert_ne!(other.body_text(), whole.body_text());
}
