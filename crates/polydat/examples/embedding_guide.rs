// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Guide companion: every program in docs/guides/embedding.md, run here
//! so the guide's output is real. Each section is one host
//! responsibility or extension point.

use polydat::ast::Value;
use polydat::derive_support::Ext;
use polydat::dsl::compile::{compile_polydat, compile_polydat_to_assembler, compile_polydat_with_libs, compile_polydat_with_log};
use polydat::dsl::events::CompileEventLog;

/// A node the host defines. The attribute registers it at link time
/// under its function name, so DSL text compiled anywhere in this
/// process can call `host_checksum(a, b)` and every engine level can run
/// it: the macro emits the P1 body and the P2 closure, and the P3 tiers
/// fall back to the closure inside a hybrid kernel.
#[polydat::polydat_node(category = Math)]
fn host_checksum(a: u64, b: u64) -> u64 {
    a.rotate_left(7) ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// A string node the host defines, over the same borrowed-argument
/// contract the built-in string nodes use.
#[polydat::polydat_node(category = String)]
fn host_tag(prefix: &str, n: u64) -> String {
    format!("{prefix}-{n:04}")
}

/// A value type the host defines. Implementing `ReflectedValue`
/// lets it ride a wire as `Value::Ext`: nodes take and return it
/// through `Ext<T>`, string interpolation and JSON use `display` and
/// `to_json_value`, and the host downcasts it back through `as_any`.
#[derive(Debug, Clone, PartialEq)]
struct GeoCell {
    lat_deg: f64,
    lon_deg: f64,
    level: u64,
}

impl polydat::ast::ReflectedValue for GeoCell {
    fn type_name(&self) -> &str {
        "GeoCell"
    }
    fn display(&self) -> String {
        format!("cell({:.3}, {:.3}, L{})", self.lat_deg, self.lon_deg, self.level)
    }
    fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({ "lat": self.lat_deg, "lon": self.lon_deg, "level": self.level })
    }
    fn clone_reflected(&self) -> Box<dyn polydat::ast::ReflectedValue> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A node that produces the host type. The return type `Ext<GeoCell>`
/// boxes it into the wire. The generated struct would be named `GeoCell`
/// after the function, which is the value type, so the attribute names
/// it `GeoCellNode` instead; the DSL name stays `geo_cell`.
#[polydat::polydat_node(category = Math, struct_name = GeoCellNode)]
fn geo_cell(lat: f64, lon: f64, level: u64) -> Ext<GeoCell> {
    Ext(GeoCell { lat_deg: lat, lon_deg: lon, level })
}

/// A node that consumes it. The parameter type `Ext<GeoCell>` downcasts
/// the wire value; a wire carrying some other extension type is a
/// panic, not a silent mismatch.
#[polydat::polydat_node(category = String)]
fn cell_token(cell: Ext<GeoCell>) -> String {
    let scale = (1u64 << cell.level) as f64;
    let row = ((cell.lat_deg + 90.0) / 180.0 * scale) as u64;
    let col = ((cell.lon_deg + 180.0) / 360.0 * scale) as u64;
    format!("L{}:{row}:{col}", cell.level)
}

fn main() {
    section_compile_and_drive();
    section_externs();
    section_share_across_threads();
    section_host_nodes();
    section_host_values();
    section_assembler();
    section_modules();
    section_traversal();
    section_tiles();
    section_compiled_kernels();
    section_transforms();
    section_diagnostics();
}

/// Compile from source, drive by coordinate, read typed values.
fn section_compile_and_drive() {
    println!("== 1. Compile and drive ==");
    let mut kernel = compile_polydat(
        r#"
            input cycle: u64
            user_id := mod(hash(cycle), 1000000)
            score   := unit_interval(hash(user_id))
            label   := "user-{user_id}"
        "#,
    )
    .expect("compile");
    println!("inputs: {:?}", kernel.input_names());
    let named: Vec<&str> = kernel.output_names().into_iter().filter(|n| !n.contains("__anon")).collect();
    println!("outputs: {named:?}");
    for cycle in [0u64, 1, 2] {
        kernel.set_inputs(&[cycle]);
        let user_id = kernel.pull("user_id").as_u64();
        let score = kernel.pull("score").as_f64();
        let label = kernel.pull("label").as_str().to_string();
        println!("cycle {cycle}: user_id={user_id} score={score:.3} label={label}");
    }
    println!();
}

/// Externs: typed slots with defaults that the host may overwrite per
/// state, and that a program transform may fix before compilation.
fn section_externs() {
    println!("== 2. Externs ==");
    let src = r#"
        input cycle: u64
        extern region: str = "us-east"
        extern scale: u64 = 10
        id := mod_wire(hash(cycle), scale)
        key := "{region}/{id}"
    "#;
    let kernel = compile_polydat(src).expect("compile");
    let program = kernel.into_program();
    let mut state = program.create_state();
    state.set_inputs(&[7]);
    println!("defaults: {}", state.pull(&program, "key").as_str());
    let region = program.find_input("region").expect("extern is an input slot");
    let scale = program.find_input("scale").expect("extern is an input slot");
    state.set_input(region, Value::Str("eu-west".into()));
    state.set_input(scale, Value::U64(1000));
    state.set_inputs(&[7]);
    println!("overridden: {}", state.pull(&program, "key").as_str());
    // The same assignment as a program transform: rewrite the source
    // before compiling, as the binary does for `name=value` arguments,
    // so nothing is written to a state at run time.
    let transformed = src.replace("extern region: str = \"us-east\"", "extern region: str = \"ap-south\"");
    let mut fixed = compile_polydat(&transformed).expect("compile");
    fixed.set_inputs(&[7]);
    println!("transformed: {}", fixed.pull("key").as_str());
    // The compiled engines carry the same externs: defaults are seeded
    // into the kernel and set_input replaces them between runs.
    let mut compiled = compile_polydat_to_assembler(src).unwrap().try_compile_raw().unwrap_or_else(|_| panic!("P2"));
    compiled.eval(&[7]);
    println!("compiled, defaults: {}", compiled.get_value("key").as_str());
    compiled.set_input("region", Value::Str("eu-west".into())).expect("a str extern");
    compiled.set_input("scale", Value::U64(1000)).expect("a u64 extern");
    compiled.eval(&[7]);
    println!("compiled, overridden: {}", compiled.get_value("key").as_str());
    println!();
}

/// One immutable program, one state per thread, no locks.
fn section_share_across_threads() {
    println!("== 3. Share a program across threads ==");
    let program = compile_polydat("input cycle: u64\nv := mod(hash(cycle), 1000)\n").expect("compile").into_program();
    let threads = 4;
    let per_thread = 25_000u64;
    let sums: Vec<u64> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let program = program.clone();
                s.spawn(move || {
                    let mut state = program.create_state();
                    let mut sum = 0u64;
                    for c in t * per_thread..(t + 1) * per_thread {
                        state.set_inputs(&[c]);
                        sum += state.pull(&program, "v").as_u64();
                    }
                    sum
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let total: u64 = sums.iter().sum();
    // The same cycles on one thread give the same total: determinism is
    // per coordinate, not per thread.
    let mut state = program.create_state();
    let mut serial = 0u64;
    for c in 0..threads * per_thread {
        state.set_inputs(&[c]);
        serial += state.pull(&program, "v").as_u64();
    }
    println!("{threads} threads x {per_thread} cycles: sum {total}; serial sum {serial}; equal: {}", total == serial);
    println!();
}

/// Nodes the host registered are ordinary functions in the DSL.
fn section_host_nodes() {
    println!("== 4. Host-defined nodes ==");
    let mut kernel = compile_polydat(
        r#"
            input cycle: u64
            c := host_checksum(cycle, hash(cycle))
            t := host_tag("job", mod(c, 10000))
        "#,
    )
    .expect("compile");
    for cycle in [1u64, 2] {
        kernel.set_inputs(&[cycle]);
        println!("cycle {cycle}: c={} t={}", kernel.pull("c").as_u64(), kernel.pull("t").as_str());
    }
    println!();
}

/// A host type carried through wires as an extension value.
fn section_host_values() {
    println!("== 5. Host-defined value types ==");
    let mut kernel = compile_polydat(
        r#"
            input cycle: u64
            lat  := unit_interval(hash(cycle)) * 180.0 - 90.0
            lon  := unit_interval(hash(cycle + 1000)) * 360.0 - 180.0
            cell := geo_cell(lat, lon, 6)
            tok  := cell_token(cell)
            line := "{tok} is {cell}"
        "#,
    )
    .expect("compile");
    for cycle in [0u64, 1] {
        kernel.set_inputs(&[cycle]);
        println!("cycle {cycle}: {}", kernel.pull("line").as_str());
        // The host reads the wire as its own type again.
        let cell = kernel.pull("cell").clone();
        let Value::Ext(boxed) = cell else { panic!("cell is an Ext wire") };
        let cell = boxed.as_any().downcast_ref::<GeoCell>().expect("a GeoCell");
        println!("cycle {cycle}: level {} at ({:.1}, {:.1}); json {}", cell.level, cell.lat_deg, cell.lon_deg, boxed.to_json_value());
    }
    println!("cell wire type: {:?}", kernel.program().output_port_type("cell"));
    println!();
}

/// The assembler API: the same graph without source text.
fn section_assembler() {
    println!("== 6. The assembler API ==");
    use polydat::compile::assembly::{PolydatAssembler, WireRef};
    use polydat::library::arithmetic::Mod;
    use polydat::library::hash::Hash;
    let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
    asm.add_node("hashed", Box::new(Hash::new()), vec![WireRef::input("cycle")]);
    asm.add_node("user_id", Box::new(Mod::new(1_000_000)), vec![WireRef::node("hashed")]);
    asm.add_output("user_id", WireRef::node("user_id"));
    let mut kernel = asm.compile().expect("compile");
    kernel.set_inputs(&[42]);
    println!("user_id at cycle 42: {}", kernel.pull("user_id").as_u64());
    println!();
}

/// Modules from files: a library directory the host controls.
fn section_modules() {
    println!("== 7. Modules from a library directory ==");
    let dir = std::env::temp_dir().join(format!("polydat-embedding-guide-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    std::fs::write(
        dir.join("bucketed.polydat"),
        "bucketed(input: u64, buckets: u64) -> (bucket: u64, label: str) := {\n    bucket := mod(hash(input), buckets)\n    label := \"b{bucket}\"\n}\n",
    )
    .expect("write module");
    let mut kernel = compile_polydat_with_libs(
        "input cycle: u64\n(b, l) := bucketed(input: cycle, buckets: 8)\n",
        None,
        vec![dir.clone()],
        &[],
        false,
        "embedding guide",
    )
    .expect("compile with libs");
    for cycle in [0u64, 1, 2] {
        kernel.set_inputs(&[cycle]);
        println!("cycle {cycle}: bucket={} label={}", kernel.pull("b").as_u64(), kernel.pull("l").as_str());
    }
    let _ = std::fs::remove_dir_all(&dir);
    println!();
}

/// A traversal: the host opens the comprehension the program declares
/// and drives each activation itself.
fn section_traversal() {
    println!("== 8. Traversal ==");
    let mut kernel = compile_polydat(
        r#"
            input cycle: u64
            for shard in 0..3, phase in load,verify {
                row := mod(hash(cycle), 100) + shard * 100
                stmt := "{phase} shard {shard} row {row}"
            }
        "#,
    )
    .expect("compile");
    kernel.set_inputs(&[0]);
    let mut stream = kernel.traverse(0).expect("open traversal");
    println!("{} activations of `{}`", stream.len(), stream.traversal().source_text);
    while let Some(mut act) = stream.advance().expect("activation") {
        let index = act.index;
        let k = act.cycle(1);
        println!("  activation {index}: {}", k.pull("stmt").as_str());
    }
    println!();
}

/// Tiles at the host boundary: a template that arrives as a parsed
/// JSON value becomes a tile statement.
fn section_tiles() {
    println!("== 9. Tiles from host data ==");
    use polydat::tile::{compile_polydat_with_tiles, tile_from_json_value, Span, TileOptions};
    let template = serde_json::json!({
        "id": "${cycle}",
        "label": "row-${cycle}",
        "points": [ "@for s in 0..2", { "n": "${s}", "v": "${cycle + s}" } ]
    });
    let tile = tile_from_json_value("doc", &template, &TileOptions::default(), Span { line: 0, col: 0 }).expect("tile");
    let mut kernel = compile_polydat_with_tiles("input cycle: u64\n", vec![tile]).expect("compile");
    kernel.set_inputs(&[4]);
    println!("doc: {}", kernel.pull("doc").as_str());
    println!();
}

/// Driving a compiled kernel directly: the closure tier and native code,
/// with typed reads and the one host rule they carry.
fn section_compiled_kernels() {
    println!("== 10. Compiled kernels ==");
    // host_tag and the two extension nodes have closure forms but no
    // native one, so the hybrid kernel must mix engines to run this.
    let src = "input cycle: u64\nh := hash(cycle)\nname := \"user-{h}\"\ntag := host_tag(\"job\", mod(h, 10000))\ncell := geo_cell(to_f64(mod(h, 180)) - 90.0, to_f64(mod(h, 360)) - 180.0, 4)\ntok := cell_token(cell)\ntile j : json := {\"h\": ${h}, \"name\": ${name}, \"tag\": ${tag}, \"cell\": ${tok}}\n";
    let mut p2 = compile_polydat_to_assembler(src).unwrap().try_compile_raw().unwrap_or_else(|_| panic!("P2"));
    // Pure native code needs every node to have a native form.
    match compile_polydat_to_assembler(src).unwrap().try_compile_jit() {
        Ok(_) => println!("pure P3: compiled"),
        Err(e) => println!("pure P3 refused: {e}"),
    }
    // The hybrid kernel lowers what it can and runs the rest as closures.
    let mut hybrid = compile_polydat_to_assembler(src).unwrap().compile_hybrid().expect("hybrid");
    let (native, closures) = hybrid.engine_counts();
    println!("hybrid plan: {native} native segment(s), {closures} closure step(s)");
    for cycle in [0u64, 1] {
        p2.eval(&[cycle]);
        // Handle outputs are read through get_value, which copies out;
        // do that before running another root kernel on this thread.
        let p2_h = p2.get("h");
        let p2_j = p2.get_value("j").to_display_string();
        hybrid.eval(&[cycle]);
        let hy_h = hybrid.get("h");
        let hy_j = hybrid.get_value("j").to_display_string();
        println!("cycle {cycle}: P2 h={p2_h} j={p2_j}");
        println!("cycle {cycle}: hybrid agrees: {}", p2_h == hy_h && p2_j == hy_j);
    }
    // The production kernel mixes engines itself; the host never picks.
    let mut mixed = compile_polydat_to_assembler(src).unwrap().compile().expect("mixed");
    mixed.set_inputs(&[1]);
    println!("production kernel j: {}", mixed.pull("j").to_display_string());
    println!();
}

/// Host features are program transforms: append a binding, compile the
/// result, and drain what the node buffered.
fn section_transforms() {
    println!("== 11. Program transforms ==");
    let src = "input cycle: u64\nid := mod(hash(cycle), 1000)\nname := \"user-{id}\"\n";
    let with_emit = format!("{src}__emit := emit_row(\"jsonl\", \"cycle,id,name\", cycle, id, name)\n");
    let mut kernel = compile_polydat(&with_emit).expect("compile");
    for cycle in [0u64, 1, 2] {
        kernel.set_inputs(&[cycle]);
        kernel.pull("__emit"); // the pull is what emits
    }
    // Drain this thread's buffer; nothing here refers to the kernel.
    for row in polydat::library::emit::take_rows() {
        println!("{row}");
    }
    println!();
}

/// Diagnostics: the compile log the binary's `explain` narrates, and the
/// program's own introspection.
fn section_diagnostics() {
    println!("== 12. Diagnostics ==");
    let mut log = CompileEventLog::new();
    let kernel = compile_polydat_with_log("input cycle: u64\nh := hash(cycle)\nf := to_f64(h) / 3.0\ntile t : json := {\"h\": ${h}, \"f\": ${f | .2}}\n", &mut log)
        .expect("compile");
    println!("compile events: {}", log.events().len());
    for event in log.events() {
        println!("  {:?}: {event:?}", event.level());
    }
    let program = kernel.program();
    println!("nodes: {}, deterministic: {}", program.node_count(), program.is_deterministic());
    let names: Vec<String> = (0..program.node_count()).map(|i| program.node_meta(i).name.clone()).collect();
    println!("node names: {names:?}");
    println!();
}
