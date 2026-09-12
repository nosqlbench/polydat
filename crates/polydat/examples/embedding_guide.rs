// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Guide companion: every program in docs/guides/embedding.md, run here
//! so the guide's output is real. Each section is one host
//! responsibility or extension point.

use polydat::ast::Value;
use polydat::derive_support::Ext;
use polydat::dsl::compile::{
    CompileOptions, compile_polydat_kernel, compile_polydat_kernel_with_options,
    compile_polydat_with, compile_polydat_with_log,
};
use polydat::dsl::events::CompileEventLog;
use polydat::{Engine, JitMode, Kernel, Provenance};

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
        format!(
            "cell({:.3}, {:.3}, L{})",
            self.lat_deg, self.lon_deg, self.level
        )
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
    Ext(GeoCell {
        lat_deg: lat,
        lon_deg: lon,
        level,
    })
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
    section_engines();
    section_transforms();
    section_diagnostics();
}

/// Compile from source, drive by coordinate, read typed values.
fn section_compile_and_drive() {
    println!("== 1. Compile and drive ==");
    let mut kernel = compile_polydat_kernel(
        r#"
            input cycle: u64
            user_id := mod(hash(cycle), 1000000)
            score   := unit_interval(hash(user_id))
            label   := "user-{user_id}"
        "#,
    )
    .expect("compile");
    println!("engine: {}", kernel.engine());
    println!("inputs: {:?}", kernel.input_names());
    let named: Vec<String> = kernel
        .output_names()
        .into_iter()
        .filter(|n| !n.contains("__anon"))
        .collect();
    println!("outputs: {named:?}");
    for cycle in [0u64, 1, 2] {
        kernel.set_inputs(&[cycle]);
        let user_id = kernel.pull("user_id").as_u64();
        let score = kernel.pull("score").as_f64();
        let label = kernel.pull("label").as_str().to_string();
        println!("cycle {cycle}: user_id={user_id} score={score:.3} label={label}");
    }
    // A host that reads the same outputs every cycle resolves each name
    // once and pulls by index, as the binary's fibers do.
    let at: Vec<usize> = ["user_id", "score", "label"]
        .iter()
        .map(|n| kernel.output_index(n).expect("a named output"))
        .collect();
    println!("output indices: {at:?}");
    for cycle in [3u64, 4] {
        kernel.set_inputs(&[cycle]);
        let user_id = kernel.pull_at(at[0]).as_u64();
        let score = kernel.pull_at(at[1]).as_f64();
        let label = kernel.pull_at(at[2]).as_str().to_string();
        println!("cycle {cycle} by index: user_id={user_id} score={score:.3} label={label}");
    }
    println!();
}

/// Externs: typed slots with defaults that the host may overwrite per
/// kernel, and that a program transform may fix before compilation.
fn section_externs() {
    println!("== 2. Externs ==");
    let src = r#"
        input cycle: u64
        extern region: str = "us-east"
        extern scale: u64 = 10
        id := mod_wire(hash(cycle), scale)
        key := "{region}/{id}"
    "#;
    let mut kernel = compile_polydat_kernel(src).expect("compile");
    kernel.set_inputs(&[7]);
    println!("defaults: {}", kernel.pull("key").as_str());
    kernel
        .set_input("region", Value::Str("eu-west".into()))
        .expect("a str extern");
    kernel
        .set_input("scale", Value::U64(1000))
        .expect("a u64 extern");
    kernel.set_inputs(&[7]);
    println!("overridden: {}", kernel.pull("key").as_str());
    // The same assignment as a program transform: rewrite the source
    // before compiling, as the binary does for `name=value` arguments,
    // so nothing is written to a kernel at run time.
    let transformed = src.replace(
        "extern region: str = \"us-east\"",
        "extern region: str = \"ap-south\"",
    );
    let mut fixed = compile_polydat_kernel(&transformed).expect("compile");
    fixed.set_inputs(&[7]);
    println!("transformed: {}", fixed.pull("key").as_str());
    // The interpreter has the same slots behind the same calls.
    let mut p1 = compile_polydat_with(src, Engine::Interpreter(JitMode::Auto)).expect("compile");
    p1.set_input("region", Value::Str("eu-west".into()))
        .expect("a str extern");
    p1.set_input("scale", Value::U64(1000))
        .expect("a u64 extern");
    p1.set_inputs(&[7]);
    println!("interpreter, overridden: {}", p1.pull("key").as_str());
    // An extern the host rewrites every cycle: resolve its slot once
    // and write by index. The coordinates come first in `input_names`,
    // so the extern's index follows them.
    let region = kernel.input_index("region").expect("a declared extern");
    let key = kernel.output_index("key").expect("a named output");
    println!("input index of region: {region}");
    for (cycle, name) in [(8u64, "us-east"), (9, "eu-west"), (10, "ap-south")] {
        kernel.set_inputs(&[cycle]);
        kernel
            .set_input_at(region, Value::Str(name.into()))
            .expect("a str extern");
        println!("cycle {cycle} by index: {}", kernel.pull_at(key).as_str());
    }
    println!();
}

/// One immutable program, one kernel per thread, no locks.
fn section_share_across_threads() {
    println!("== 3. Share a program across threads ==");
    let program = compile_polydat_kernel("input cycle: u64\nv := mod(hash(cycle), 1000)\n")
        .expect("compile")
        .into_program();
    let threads = 4;
    let per_thread = 25_000u64;
    let sums: Vec<u64> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let program = program.clone();
                s.spawn(move || {
                    let mut kernel = program.create_kernel();
                    let mut sum = 0u64;
                    for c in t * per_thread..(t + 1) * per_thread {
                        kernel.set_inputs(&[c]);
                        sum += kernel.pull("v").as_u64();
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
    let mut kernel = program.create_kernel();
    let mut serial = 0u64;
    for c in 0..threads * per_thread {
        kernel.set_inputs(&[c]);
        serial += kernel.pull("v").as_u64();
    }
    println!(
        "{threads} threads x {per_thread} cycles: sum {total}; serial sum {serial}; equal: {}",
        total == serial
    );
    println!();
}

/// Nodes the host registered are ordinary functions in the DSL.
fn section_host_nodes() {
    println!("== 4. Host-defined nodes ==");
    let mut kernel = compile_polydat_kernel(
        r#"
            input cycle: u64
            c := host_checksum(cycle, hash(cycle))
            t := host_tag("job", mod(c, 10000))
        "#,
    )
    .expect("compile");
    for cycle in [1u64, 2] {
        kernel.set_inputs(&[cycle]);
        println!(
            "cycle {cycle}: c={} t={}",
            kernel.pull("c").as_u64(),
            kernel.pull("t").as_str()
        );
    }
    println!();
}

/// A host type carried through wires as an extension value.
fn section_host_values() {
    println!("== 5. Host-defined value types ==");
    let mut kernel = compile_polydat_kernel(
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
        let Value::Ext(boxed) = kernel.pull("cell") else {
            panic!("cell is an Ext wire")
        };
        let cell = boxed.as_any().downcast_ref::<GeoCell>().expect("a GeoCell");
        println!(
            "cycle {cycle}: level {} at ({:.1}, {:.1}); json {}",
            cell.level,
            cell.lat_deg,
            cell.lon_deg,
            boxed.to_json_value()
        );
    }
    println!("cell wire type: {:?}", kernel.output_type("cell"));
    println!();
}

/// The assembler API: the same graph without source text.
fn section_assembler() {
    println!("== 6. The assembler API ==");
    use polydat::compile::assembly::{PolydatAssembler, WireRef};
    use polydat::library::arithmetic::Mod;
    use polydat::library::hash::Hash;
    let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
    asm.add_node(
        "hashed",
        Box::new(Hash::new()),
        vec![WireRef::input("cycle")],
    );
    asm.add_node(
        "user_id",
        Box::new(Mod::new(1_000_000)),
        vec![WireRef::node("hashed")],
    );
    asm.add_output("user_id", WireRef::node("user_id"));
    let mut kernel = asm.compile_kernel().expect("compile");
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
    let mut kernel = compile_polydat_kernel_with_options(
        "input cycle: u64\n(b, l) := bucketed(input: cycle, buckets: 8)\n",
        &CompileOptions {
            lib_paths: vec![dir.clone()],
            context: "embedding guide".into(),
            ..CompileOptions::default()
        },
        None,
    )
    .expect("compile with libs");
    for cycle in [0u64, 1, 2] {
        kernel.set_inputs(&[cycle]);
        println!(
            "cycle {cycle}: bucket={} label={}",
            kernel.pull("b").as_u64(),
            kernel.pull("l").as_str()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
    println!();
}

/// A traversal: the host opens the comprehension the program declares
/// and drives each activation itself.
fn section_traversal() {
    println!("== 8. Traversal ==");
    let mut kernel = compile_polydat_kernel(
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
    let stream = kernel.traverse(0).expect("open traversal");
    println!(
        "{} activations of `{}`",
        stream.len(),
        stream.traversal().source_text
    );
    for index in 0..stream.len() {
        let mut act = stream.activate(index).expect("activation");
        let k = act.cycle(1);
        println!("  activation {index}: {}", k.pull("stmt").as_str());
    }
    println!();
}

/// Tiles at the host boundary: a template that arrives as a parsed
/// JSON value becomes a tile statement.
fn section_tiles() {
    println!("== 9. Tiles from host data ==");
    use polydat::tile::{
        Span, TileOptions, compile_polydat_kernel_with_tiles, tile_from_json_value,
    };
    let template = serde_json::json!({
        "id": "${cycle}",
        "label": "row-${cycle}",
        "points": [ "@for s in 0..2", { "n": "${s}", "v": "${cycle + s}" } ]
    });
    let tile = tile_from_json_value(
        "doc",
        &template,
        &TileOptions::default(),
        Span { line: 0, col: 0 },
    )
    .expect("tile");
    let mut kernel =
        compile_polydat_kernel_with_tiles("input cycle: u64\n", vec![tile]).expect("compile");
    kernel.set_inputs(&[4]);
    println!("doc: {}", kernel.pull("doc").as_str());
    println!();
}

/// Naming an engine: the closure tier, native code, and the
/// interpreter, driven through one trait, with the interpreter as the
/// oracle.
fn section_engines() {
    println!("== 10. Engines ==");
    // host_tag and the two extension nodes have closure forms but no
    // native one, so P3 mixes native segments and closure steps.
    let src = "input cycle: u64\nh := hash(cycle)\nname := \"user-{h}\"\ntag := host_tag(\"job\", mod(h, 10000))\ncell := geo_cell(to_f64(mod(h, 180)) - 90.0, to_f64(mod(h, 360)) - 180.0, 4)\ntok := cell_token(cell)\ntile j : json := {\"h\": ${h}, \"name\": ${name}, \"tag\": ${tag}, \"cell\": ${tok}}\n";
    // One constructor names the engine; one trait drives whatever it
    // built. An engine that cannot run the program says so by name.
    let mut kernels: Vec<Box<dyn Kernel>> = Vec::new();
    for engine in [
        Engine::Closures(Provenance::Auto),
        Engine::Native(Provenance::Auto),
    ] {
        match compile_polydat_with(src, engine) {
            Ok(k) => kernels.push(k),
            Err(e) => println!("{e}"),
        }
    }
    // The default engine's kernel and its plan, the one planning detail
    // a kernel exposes.
    let p3 = compile_polydat_kernel(src).expect("default");
    println!("Engine::default() is {}", p3.engine());
    println!("P3 plan: {}", p3.plan());
    // The interpreter is the oracle; every engine that accepted the
    // program computes the same values through the same calls.
    let mut p1 = compile_polydat_with(src, Engine::Interpreter(JitMode::Auto)).unwrap();
    for cycle in [0u64, 1] {
        p1.set_inputs(&[cycle]);
        let want = p1.pull("j").to_display_string();
        println!("cycle {cycle}: j={want}");
        for k in kernels.iter_mut() {
            k.set_inputs(&[cycle]);
            let got = k.pull("j").to_display_string();
            println!("  {} agrees: {}", k.engine(), got == want);
        }
    }
    println!();
}

/// Host features are program transforms: append a binding, compile the
/// result, and drain what the node buffered.
fn section_transforms() {
    println!("== 11. Program transforms ==");
    let src = "input cycle: u64\nid := mod(hash(cycle), 1000)\nname := \"user-{id}\"\n";
    let with_emit =
        format!("{src}__emit := emit_row(\"jsonl\", \"cycle,id,name\", cycle, id, name)\n");
    let mut kernel = compile_polydat_kernel(&with_emit).expect("compile");
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
/// interpreter program's own introspection.
fn section_diagnostics() {
    println!("== 12. Diagnostics ==");
    let src = "input cycle: u64\nh := hash(cycle)\nf := to_f64(h) / 3.0\ntile t : json := {\"h\": ${h}, \"f\": ${f | .2}}\n";
    let mut log = CompileEventLog::new();
    let kernel = compile_polydat_with_log(src, &mut log).expect("compile");
    println!("compile events: {}", log.events().len());
    for event in log.events() {
        println!("  {:?}: {event:?}", event.level());
    }
    let program = kernel.program();
    println!(
        "nodes: {}, deterministic: {}",
        program.node_count(),
        program.is_deterministic()
    );
    let names: Vec<String> = (0..program.node_count())
        .map(|i| program.node_meta(i).name.clone())
        .collect();
    println!("node names: {names:?}");
    // The same log from the default engine.
    let mut compiled_log = CompileEventLog::new();
    compile_polydat_kernel_with_options(src, &CompileOptions::default(), Some(&mut compiled_log))
        .expect("compile");
    println!(
        "compile events on the default engine: {}",
        compiled_log.events().len()
    );
    println!();
}
