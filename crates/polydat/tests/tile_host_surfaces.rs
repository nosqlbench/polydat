// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 114 §5.6 and §10, step 6: the host surfaces. Tile defaults as a
//! program transform, tiles inside module bodies, the skeleton event
//! `explain tiles` prints, the text emit format, and the binary's
//! `--emit tile:<name>`, `--tile-delims`, and `--tile-sigil`.

use std::process::Command;

use polydat::dsl::ast::TileOptions;
use polydat::dsl::compile::{CompileOptions, compile_ast_with_options, compile_polydat_with_log};
use polydat::dsl::events::{CompileEvent, CompileEventLog};
use polydat::dsl::transform::apply_tile_defaults;

/// The interpreter kernel of a parsed program under the default options.
fn compile_ast(
    ast: &polydat::dsl::ast::PolydatFile,
) -> Result<polydat::kernel::PolydatKernel, String> {
    compile_ast_with_options(ast, "", &CompileOptions::default(), None)
}

#[test]
fn tile_defaults_transform_rereads_untouched_tiles_only() {
    let src = "input cycle: u64\n\
        tile a : text := \"<%cycle%> #if cycle { on } #else { off } ${keep}\"\n\
        tile b : text (delims \"[[\" \"]]\") := \"[[cycle]] <%cycle%>\"\n";
    let tokens = polydat::dsl::lexer::lex(src).unwrap();
    let mut ast = polydat::dsl::parser::parse(tokens).unwrap();
    // Before the transform, `a` reads `${keep}` as a hole and fails.
    assert!(compile_ast(&ast).is_err());
    let defaults = TileOptions {
        open: "<%".into(),
        close: "%>".into(),
        sigil: "#".into(),
        strict: false,
        in_string: false,
    };
    apply_tile_defaults(&mut ast, &defaults).unwrap();
    let mut k = compile_ast(&ast).unwrap();
    k.set_inputs(&[3]);
    assert_eq!(k.pull("a").as_str(), "3 on ${keep}");
    // `b` declared its own delimiters and keeps them.
    assert_eq!(k.pull("b").as_str(), "3 <%cycle%>");
}

// These run through the binary on a temp file, so they also cover the
// binary's module resolution from the program's own directory.
#[test]
fn a_tile_inside_a_module_body_inlines_with_the_call() {
    let path = write_temp(
        "module",
        "input cycle: u64\n\
         make_card(n: u64, label: str) -> (doc: str) := {\n\
             twice := n * 2\n\
             tile doc : json := {\"n\": ${n}, \"twice\": ${twice}, \"label\": ${label}, \"big\": @if n > 5 { true } @else { false }}\n\
         }\n\
         first := make_card(cycle, \"one\")\n\
         second := make_card(cycle + 10, \"two\")\n",
    );
    let (ok, stdout, stderr) = run_binary(&[
        "run",
        path.to_str().unwrap(),
        "--cycles",
        "1",
        "--emit",
        "jsonl",
        "--outputs",
        "first,second",
        "-q",
    ]);
    assert!(ok, "{stderr}");
    let row: serde_json::Value =
        serde_json::from_str(stdout.lines().find(|l| l.starts_with('{')).unwrap()).unwrap();
    assert_eq!(
        row["first"],
        "{\"n\": 0, \"twice\": 0, \"label\": \"one\", \"big\": false}"
    );
    assert_eq!(
        row["second"],
        "{\"n\": 10, \"twice\": 20, \"label\": \"two\", \"big\": true}"
    );
}

#[test]
fn a_module_tile_projection_reads_module_wires() {
    // A generator expression in the projection source is rewritten
    // against the caller's arguments like any module expression.
    let path = write_temp(
        "module-proj",
        "input cycle: u64\n\
         make_row(top: u64) -> (chosen: str) := {\n\
             limit := top + 1\n\
             tile chosen : text := \"@for g in hash_range(limit, 50), k in 1..3 sep \\\",\\\" {${g + k * top}}\"\n\
         }\n\
         top := cycle + 2\n\
         r := make_row(top)\n\
         expect := hash_range(top + 1, 50)\n",
    );
    let (ok, stdout, stderr) = run_binary(&[
        "run",
        path.to_str().unwrap(),
        "--cycles",
        "1",
        "--emit",
        "jsonl",
        "--outputs",
        "r,expect",
        "-q",
    ]);
    assert!(ok, "{stderr}");
    let row: serde_json::Value =
        serde_json::from_str(stdout.lines().find(|l| l.starts_with('{')).unwrap()).unwrap();
    let g: u64 = row["expect"]
        .as_str()
        .map(|s| s.parse().unwrap())
        .unwrap_or_else(|| row["expect"].as_u64().unwrap());
    assert_eq!(row["r"], format!("{},{}", g + 2, g + 4));
    // A `{name}` placeholder for a module input needs a wire, not an expression.
    let path = write_temp(
        "module-ph",
        "input cycle: u64\n\
         make_parts(n: u64) -> (out: str) := {\n\
             tile out : text := \"@for p in partitions(\\\"*/4\\\", {n}) {x}\"\n\
         }\n\
         r := make_parts(cycle + 2)\n",
    );
    let (ok, _, stderr) = run_binary(&["check", path.to_str().unwrap()]);
    assert!(!ok);
    assert!(stderr.contains("must pass a wire"), "{stderr}");
}

#[test]
fn the_compile_log_describes_each_tile_skeleton() {
    let mut log = CompileEventLog::default();
    let src = "input cycle: u64\n\
        tile doc : json := {\"meta\": {\"schema\": 3}, \"n\": ${cycle}, \"xs\": [@for i in 0..2 { ${i} }], \"f\": @if cycle { 1 } @else { 0 }}\n";
    compile_polydat_with_log(src, &mut log).unwrap();
    let shape = log
        .events()
        .iter()
        .find_map(|e| match e {
            CompileEvent::TileCompiled {
                tile,
                encoding,
                statics,
                static_bytes,
                holes,
                branches,
                projections,
                bodies,
            } if tile == "doc" => Some((
                encoding.clone(),
                *statics,
                *static_bytes,
                *holes,
                *branches,
                *projections,
                bodies.clone(),
            )),
            _ => None,
        })
        .expect("a TileCompiled event");
    assert_eq!(shape.0, "json");
    assert!(shape.1 >= 3, "{shape:?}");
    assert!(shape.2 > 20, "{shape:?}");
    assert_eq!(shape.3, 2, "holes include the projection body hole");
    assert_eq!(shape.4, 1);
    assert_eq!(shape.5, 1);
    assert_eq!(shape.6.len(), 1);
    assert!(shape.6[0].contains("extern i: u64"), "{}", shape.6[0]);
    assert!(
        log.format().contains("tile 'doc' (json)"),
        "{}",
        log.format()
    );
}

#[test]
fn the_text_emit_format_writes_values_as_they_are() {
    use polydat::ast::Value;
    use polydat::library::emit::{EmitFormat, header, render_row};
    assert_eq!(EmitFormat::parse("text"), Some(EmitFormat::Text));
    assert!(header(EmitFormat::Text, &["doc"]).is_none());
    assert_eq!(
        render_row(
            EmitFormat::Text,
            &["doc"],
            &[Value::Str("{\"a\": 1}".into())]
        ),
        "{\"a\": 1}"
    );
}

fn write_temp(name: &str, contents: &str) -> std::path::PathBuf {
    let dir =
        std::env::temp_dir().join(format!("polydat-tile-host-{}-{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("program.polydat");
    std::fs::write(&path, contents).unwrap();
    path
}

fn run_binary(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_polydat"))
        .args(args)
        .output()
        .expect("polydat binary runs");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn the_binary_emits_a_tile_per_cycle() {
    let path = write_temp(
        "emit",
        "input cycle: u64\nn := cycle * 2\ntile doc : json := {\"cycle\": ${cycle}, \"n\": ${n}}\n",
    );
    let (ok, stdout, stderr) = run_binary(&[
        "run",
        path.to_str().unwrap(),
        "--cycles",
        "3",
        "--emit",
        "tile:doc",
        "-q",
    ]);
    assert!(ok, "{stderr}");
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.starts_with("DBG")).collect();
    assert_eq!(
        lines,
        vec![
            "{\"cycle\": 0, \"n\": 0}",
            "{\"cycle\": 1, \"n\": 2}",
            "{\"cycle\": 2, \"n\": 4}"
        ],
        "{stdout}"
    );
    let (ok, _, stderr) = run_binary(&["run", path.to_str().unwrap(), "--emit", "tile:nope", "-q"]);
    assert!(!ok);
    assert!(
        stderr.contains("no tile or output named 'nope'"),
        "{stderr}"
    );
    let (ok, _, stderr) = run_binary(&["run", path.to_str().unwrap(), "--emit", "yaml", "-q"]);
    assert!(!ok);
    assert!(stderr.contains("tile:<name>"), "{stderr}");
}

#[test]
fn the_binary_applies_tile_delimiters_and_sigil_as_a_transform() {
    let path = write_temp(
        "delims",
        "input cycle: u64\ntile page : text := \"cycle <%cycle%> #if cycle { live } #else { zero } {{ keep }} ${keep}\"\n",
    );
    let (ok, stdout, stderr) = run_binary(&[
        "run",
        path.to_str().unwrap(),
        "--cycles",
        "2",
        "--emit",
        "tile:page",
        "-q",
        "--tile-delims",
        "<%",
        "%>",
        "--tile-sigil",
        "#",
    ]);
    assert!(ok, "{stderr}");
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.starts_with("DBG")).collect();
    assert_eq!(
        lines,
        vec![
            "cycle 0 zero {{ keep }} ${keep}",
            "cycle 1 live {{ keep }} ${keep}"
        ],
        "{stdout}"
    );
    // Without the defaults the same program is an error at `${keep}`.
    let (ok, _, stderr) = run_binary(&["run", path.to_str().unwrap(), "--emit", "tile:page", "-q"]);
    assert!(!ok);
    assert!(stderr.contains("keep"), "{stderr}");
    // `explain tiles` narrates the skeleton and the holes.
    let (ok, stdout, stderr) = run_binary(&[
        "explain",
        path.to_str().unwrap(),
        "tiles",
        "--tile-delims",
        "<%",
        "%>",
        "--tile-sigil",
        "#",
    ]);
    assert!(ok, "{stderr}");
    assert!(
        stdout.contains("page") && stdout.contains("static run") && stdout.contains("${cycle}"),
        "{stdout}"
    );
}

#[test]
fn the_toy_definition_emits_a_json_document_per_reading() {
    // SRD 114 step 8: the toy test definition's load statement carries a
    // document rendered by a tile inside the traversal body.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("toy_test_definition.polydat");
    let (ok, stdout, stderr) = run_binary(&[
        "run",
        path.to_str().unwrap(),
        "--cycles",
        "1",
        "--emit",
        "tile:doc",
        "-q",
    ]);
    assert!(ok, "{stderr}");
    let text: String = stdout
        .lines()
        .filter(|l| !l.starts_with("DBG"))
        .collect::<Vec<_>>()
        .join("\n");
    // Sixteen activations, one document each; every one is valid JSON.
    let docs: Vec<serde_json::Value> = serde_json::Deserializer::from_str(&text)
        .into_iter()
        .map(|d| d.unwrap())
        .collect();
    assert_eq!(docs.len(), 16, "{text}");
    let first = &docs[0];
    assert_eq!(first["meta"]["schema"], 3);
    assert_eq!(first["tenant"], 607535);
    assert_eq!(first["device"], "d9ac876f-bb3a-4bc7-b9f8-382893178079");
    assert_eq!(first["ts"], 1700000000000u64);
    assert_eq!(first["reading"]["status"], "ok");
    assert_eq!(first["samples"].as_array().unwrap().len(), 4);
    assert_eq!(first["flagged"], false);
    // The load statement carries the same document raw.
    let (ok, stdout, stderr) = run_binary(&[
        "run",
        path.to_str().unwrap(),
        "--cycles",
        "1",
        "--emit",
        "jsonl",
        "--outputs",
        "phase,stmt",
        "-q",
    ]);
    assert!(ok, "{stderr}");
    let row: serde_json::Value =
        serde_json::from_str(stdout.lines().find(|l| l.starts_with('{')).unwrap()).unwrap();
    assert_eq!(row["phase"], "load");
    let stmt = row["stmt"].as_str().unwrap();
    assert!(stmt.starts_with("INSERT INTO toy.readings (tenant_id, device_id, ts, doc) VALUES (607535, 'd9ac876f-bb3a-4bc7-b9f8-382893178079', 1700000000000, '{"), "{stmt}");
    let start = stmt.find('{').unwrap();
    let end = stmt.rfind('}').unwrap();
    let carried: serde_json::Value = serde_json::from_str(&stmt[start..=end]).unwrap();
    assert_eq!(carried, docs[0]);
    // `explain tiles` sees the body's tiles.
    let (ok, stdout, stderr) = run_binary(&["explain", path.to_str().unwrap(), "tiles"]);
    assert!(ok, "{stderr}");
    assert!(
        stdout.contains("doc") && stdout.contains("${tenant_id}") && stdout.contains("load"),
        "{stdout}"
    );
}
