// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 114 §5.4, step 5: projections over every source the `for`
//! construct accepts (inline text, producers, derivations, generator
//! calls), nested projections inside a body, the bounded-cardinality
//! check, and member projections through the structural form.

use polydat::dsl::compile_polydat;

fn render(src: &str, cycle: u64, name: &str) -> String {
    let mut k = compile_polydat(src).unwrap_or_else(|e| panic!("compile failed: {e}\n{src}"));
    k.set_inputs(&[cycle]);
    k.pull(name).as_str().to_string()
}

fn err(src: &str) -> String {
    compile_polydat(src)
        .err()
        .unwrap_or_else(|| panic!("expected a compile error\n{src}"))
}

#[test]
fn derived_sources_filter_and_order_a_producer() {
    let src = "input cycle: u64\nks := for k in 1..7\n\
        tile t : text := \"[@for ks where {k} > 4 sep \\\",\\\" {${k}}] [@for ks where {k} < 3 sep \\\",\\\" {${k}}]\"\n";
    assert_eq!(render(src, 0, "t"), "[5,6] [1,2]");
    // Ordering with truncation bounds a larger sweep.
    let src = "input cycle: u64\nsweep := for k in 1..100\n\
        tile t : text := \"@for sweep order lex/3 sep \\\",\\\" {${k}}\"\n";
    assert_eq!(render(src, 0, "t"), "1,2,3");
}

#[test]
fn a_predicate_placeholder_for_an_outer_wire_is_a_clear_error() {
    // Predicates see the comprehension's elements, as in the `for`
    // construct; an outer wire has no value inside the stream.
    let e = err("input cycle: u64\nlimit := cycle + 2\n\
        tile t : text := \"@for k in 1..9 where {k} < {limit} sep \\\" \\\" {${k}}\"\n");
    assert!(e.contains("tile 't'"), "{e}");
    assert!(e.contains("`{limit}`"), "{e}");
    assert!(e.contains("outside the comprehension"), "{e}");
    // The header still parsed as a whole: the block is `${k}`, not `{limit}`.
    let src = "input cycle: u64\ntile t : text := \"@for k in 1..9 where {k} < {k} {x}\"\n";
    assert_eq!(render(src, 0, "t"), "");
}

#[test]
fn generator_call_sources_compile_to_wires_of_the_scope() {
    // A scalar generator is one tuple whose element is the wire's value.
    let src = "input cycle: u64\nexpect := hash_range(cycle, 1000)\n\
        tile t : text := \"@for g in hash_range(cycle, 1000) sep \\\",\\\" {${g}}\"\n";
    let mut k = compile_polydat(src).unwrap();
    for cycle in [1u64, 2, 3] {
        k.set_inputs(&[cycle]);
        let expect = k.pull("expect").as_u64().to_string();
        assert_eq!(k.pull("t").as_str(), expect);
    }
    // The element type reached the encoder: numbers are bare in json,
    // and the generator combines with other clauses.
    let src = "input cycle: u64\ntile t : json := {\"g\": [@for g in hash_range(cycle, 10), i in 0..2 { {\"g\": ${g}, \"i\": ${i}} }]}\n";
    let doc: serde_json::Value = serde_json::from_str(&render(src, 1, "t")).unwrap();
    assert_eq!(doc["g"].as_array().unwrap().len(), 2, "{doc}");
    assert!(doc["g"][0]["g"].is_number(), "{doc}");
    assert_eq!(doc["g"][1]["i"], 1);
    // A generator reading an outer wire inside a nested projection.
    let src = "input cycle: u64\nseed := cycle + 1\n\
        tile t : text := \"@for a in 0..2 sep \\\" \\\" {@for g in hash_range(seed, 100) {${a}:${g}}}\"\n";
    let mut k = compile_polydat(src).unwrap();
    k.set_inputs(&[4]);
    let text = k.pull("t").as_str().to_string();
    let parts: Vec<&str> = text.split(' ').collect();
    assert_eq!(parts.len(), 2, "{text}");
    assert!(
        parts[0].starts_with("0:") && parts[1].starts_with("1:"),
        "{text}"
    );
    assert_eq!(parts[0][2..], parts[1][2..]);
}

#[test]
fn continuous_sources_sample_with_an_order_count() {
    let e = err("input cycle: u64\ntile t : text := \"@for x in 0.0..1.0 {${x}}\"\n");
    assert!(e.contains("continuous source"), "{e}");
    assert!(e.contains("order <strategy>/<count>"), "{e}");
    // With a strategy and a count, that many points come from the interval.
    let src = "input cycle: u64\ntile t : text := \"@for x in 2.0..4.0 order halton/4 sep \\\" \\\" {${x | .3}}\"\n";
    let text = render(src, 0, "t");
    let xs: Vec<f64> = text.split(' ').map(|s| s.parse().unwrap()).collect();
    assert_eq!(xs.len(), 4, "{text}");
    assert!(xs.iter().all(|x| (2.0..4.0).contains(x)), "{text}");
    assert_eq!(text, "3.000 2.500 3.500 2.250");
    // Two continuous axes sample jointly; sobol and lhs work too.
    let src = "input cycle: u64\ntile t : text := \"@for x in 0.0..1.0, y in 10.0..20.0 order sobol/3 sep \\\";\\\" {${x | .2},${y | .1}}\"\n";
    let text = render(src, 0, "t");
    assert_eq!(text.split(';').count(), 3, "{text}");
    for pair in text.split(';') {
        let (x, y) = pair.split_once(',').unwrap();
        assert!((0.0..1.0).contains(&x.parse::<f64>().unwrap()), "{text}");
        assert!((10.0..20.0).contains(&y.parse::<f64>().unwrap()), "{text}");
    }
    let src = "input cycle: u64\ntile t : text := \"@for x in 0.0..1.0 order lhs/5 sep \\\" \\\" {${x | .2}}\"\n";
    assert_eq!(render(src, 0, "t").split(' ').count(), 5);
    // A non-sampling strategy over a continuous source is a compile error.
    let e = err("input cycle: u64\ntile t : text := \"@for x in 0.0..1.0 order lex/4 {${x}}\"\n");
    assert!(e.contains("sampling strategy"), "{e}");
    assert!(e.contains("tile 't'"), "{e}");
    // An order strategy with truncation over a discrete range samples
    // exactly that many tuples.
    let src = "input cycle: u64\ntile t : text := \"@for k in 1..100 order halton/4 sep \\\" \\\" {${k}}\"\n";
    assert_eq!(render(src, 0, "t"), "50 25 75 13");
}

#[test]
fn nested_projection_in_a_json_value_position() {
    let src = "input cycle: u64\nbase := cycle * 10\n\
        tile grid : json := {\"rows\": [@for r in 0..2 { {\"r\": ${r}, \"cells\": [@for c in 0..3 { ${base + r * 10 + c} }]} }]}\n";
    assert_eq!(
        render(src, 1, "grid"),
        "{\"rows\": [{\"r\": 0, \"cells\": [10,11,12]},{\"r\": 1, \"cells\": [20,21,22]}]}"
    );
    let doc: serde_json::Value = serde_json::from_str(&render(src, 1, "grid")).unwrap();
    assert_eq!(doc["rows"][1]["cells"][2], 22);
}

#[test]
fn nested_projection_over_producers_and_derivations() {
    let src = "input cycle: u64\nsides := for s in left,right\n\
        tile t : text := \"@for k in 1..3 sep \\\"; \\\" {${k}: @for sides sep \\\",\\\" {${s}${k}}}\"\n";
    assert_eq!(render(src, 0, "t"), "1: left1,right1; 2: left2,right2");
    let src = "input cycle: u64\nks := for k in 1..5\n\
        tile t : text := \"@for a in 1..3 sep \\\"|\\\" {${a}=@for ks where {k} > 2 sep \\\",\\\" {${k}}}\"\n";
    assert_eq!(render(src, 0, "t"), "1=3,4|2=3,4");
}

#[test]
fn nested_projection_reads_outer_elements_and_program_wires() {
    let src = "input cycle: u64\nlabel := \"L{cycle}\"\n\
        tile t : text := \"@for a in 0..2 sep \\\" \\\" {@for b in 0..2 sep \\\",\\\" {${label}:${a}${b}:${cycle}}}\"\n";
    assert_eq!(render(src, 7, "t"), "L7:00:7,L7:01:7 L7:10:7,L7:11:7");
}

#[test]
fn nested_projection_keeps_custom_delimiters_and_branches() {
    let src = "input cycle: u64\n\
        tile t : text (delims \"<%\" \"%>\", sigil \"#\") := \"#for a in 1..3 sep \\\" \\\" {#for b in 1..3 sep \\\",\\\" {<%a * b%>#if a == b { = } #else { ~ }}}\"\n";
    assert_eq!(render(src, 0, "t"), "1=,2~ 2~,4=");
}

#[test]
fn nested_projection_three_levels_deep() {
    let src = "input cycle: u64\n\
        tile t : text := \"@for a in 0..2 sep \\\"|\\\" {@for b in 0..2 sep \\\"/\\\" {@for c in 0..2 {${a}${b}${c}}}}\"\n";
    assert_eq!(render(src, 0, "t"), "000001/010011|100101/110111");
}

#[test]
fn nested_projection_inside_a_string_position_escapes_as_text() {
    // The inner projection compiles as a tile that starts inside a JSON
    // string, so its holes escape as text and the outer string stays one
    // string value.
    let src = "input cycle: u64\nq := \"say \\\"hi\\\"\"\n\
        tile t : json := {\"rows\": [@for r in 0..2 { {\"s\": \"r${r}: @for c in 0..2 sep \\\"; \\\" {${c}=${q}}\"} }]}\n";
    assert_eq!(
        render(src, 0, "t"),
        "{\"rows\": [{\"s\": \"r0: 0=say \\\"hi\\\"; 1=say \\\"hi\\\"\"},{\"s\": \"r1: 0=say \\\"hi\\\"; 1=say \\\"hi\\\"\"}]}"
    );
    let doc: serde_json::Value = serde_json::from_str(&render(src, 0, "t")).unwrap();
    assert_eq!(doc["rows"][1]["s"], "r1: 0=say \"hi\"; 1=say \"hi\"");
}

#[test]
fn errors_in_nested_sources_name_the_tile() {
    let e = err("input cycle: u64\ntile t : text := \"@for a in 0..2 {@for nowhere {${a}}}\"\n");
    assert!(e.contains("tile 't'"), "{e}");
    assert!(e.contains("nowhere"), "{e}");
}

#[test]
fn member_projections_through_the_structural_form() {
    let src = "input cycle: u64\nns := for n in 1..4\n\
        doc := polytile_json(\"{\\\"fixed\\\": true, \\\"@for ns\\\": {\\\"k${n}\\\": \\\"${n * n}\\\"}}\")\n";
    let doc: serde_json::Value = serde_json::from_str(&render(src, 0, "doc")).unwrap();
    assert_eq!(doc["fixed"], true);
    assert_eq!(doc["k1"], 1);
    assert_eq!(doc["k2"], 4);
    assert_eq!(doc["k3"], 9);
}

#[test]
fn nested_projection_through_the_structural_form() {
    let src = "input cycle: u64\n\
        doc := polytile_json(\"[\\\"@for r in 0..2\\\", [\\\"@for c in 0..2\\\", {\\\"r\\\": \\\"${r}\\\", \\\"c\\\": \\\"${c}\\\"}]]\")\n";
    let doc: serde_json::Value = serde_json::from_str(&render(src, 0, "doc")).unwrap();
    assert_eq!(doc.as_array().unwrap().len(), 2);
    assert_eq!(doc[1][0]["r"], 1);
    assert_eq!(doc[1][1]["c"], 1);
}

#[test]
fn values_cross_into_bodies_as_themselves() {
    // A JSON wire cascaded into a body renders serialized, not as a
    // quoted string of its text; an f64 keeps its value.
    let src = "input cycle: u64\nj := str_to_json(\"{\\\"k\\\": [1, 2]}\")\nf := to_f64(cycle) / 3.0\n\
        tile t : json := {\"xs\": [@for i in 0..2 { {\"i\": ${i}, \"j\": ${j}, \"f\": ${f | .4}} }]}\n";
    assert_eq!(
        render(src, 1, "t"),
        "{\"xs\": [{\"i\": 0, \"j\": {\"k\":[1,2]}, \"f\": 0.3333},{\"i\": 1, \"j\": {\"k\":[1,2]}, \"f\": 0.3333}]}"
    );
}

#[test]
fn list_valued_generators_contribute_one_tuple_per_item() {
    // A JSON array from a node call is a list source: one tuple per
    // element, each carrying its own JSON kind.
    let src = "input cycle: u64\nlist := str_to_json(\"[3, 4.5, true, \\\"x\\\"]\")\ntile t : json := {\"g\": [@for g in list { ${g} }]}\n";
    assert_eq!(render(src, 3, "t"), "{\"g\": [3,4.5,true,\"x\"]}");
    // `json_array` receives its arguments as text, so its items are strings.
    let src = "input cycle: u64\ntile t : text := \"@for g in json_array(\\\"a\\\", \\\"b\\\") sep \\\"-\\\" {${g}}\"\n";
    assert_eq!(render(src, 0, "t"), "a-b");
    // Combined with a range, the list is one axis of the product.
    let src = "input cycle: u64\npair := str_to_json(\"[1, 2]\")\ntile t : text := \"@for g in pair, k in 0..2 sep \\\",\\\" {${g}${k}}\"\n";
    assert_eq!(render(src, 0, "t"), "10,11,20,21");
}

#[test]
fn a_projection_body_runs_on_the_engine_of_the_kernel_rendering() {
    // SRD 117 step 2: the body of a projection is a nested kernel over
    // the body's program for the engine rendering, compiled once per
    // engine; the tuples of a constant comprehension are evaluated once.
    use polydat::dsl::compile::compile_polydat_to_assembler;
    use polydat::library::tile_render::body_kernels_created;
    use polydat::{Engine, JitMode, Kernel, Provenance};
    let src = r#"
        input cycle: u64
        base := cycle * 10
        tile t : json := { "rows": [ @for r in 0..3 { {"r": ${r}, "v": ${base + r}} } ] }
    "#;
    let want = |cycle: u64| {
        format!(
            r#"{{ "rows": [ {{"r": 0, "v": {}}},{{"r": 1, "v": {}}},{{"r": 2, "v": {}}} ] }}"#,
            cycle * 10,
            cycle * 10 + 1,
            cycle * 10 + 2
        )
    };
    let mut engines = vec![Engine::Interpreter, Engine::Closures(Provenance::Auto)];
    if cfg!(feature = "jit") {
        engines.push(Engine::Native(Provenance::Auto));
    }
    for engine in engines {
        // The interpreter with its cones off, so the render is the
        // interpreter's own; with cones the render node fuses and its
        // bodies are the helper's, on the default engine.
        let mut k: Box<dyn Kernel> = match engine {
            Engine::Interpreter => {
                let mut asm = compile_polydat_to_assembler(src).unwrap();
                asm.set_jit_mode(JitMode::Off);
                Box::new(asm.compile().unwrap())
            }
            _ => polydat::dsl::compile_polydat_with(src, engine).unwrap(),
        };
        // A compiled kernel renders its projection bodies on the default
        // engine, whichever compiled engine it is on itself.
        let body_engine = match engine {
            Engine::Interpreter => Engine::Interpreter,
            _ => Engine::default(),
        };
        let before = body_kernels_created(body_engine);
        for cycle in 0..3u64 {
            k.set_inputs(&[cycle]);
            assert_eq!(k.pull("t").as_str(), want(cycle), "{engine} cycle {cycle}");
        }
        // One kernel per thread, body program, and engine, reused across
        // renders and across the compiled engines, which share the default
        // engine's body: a kernel created by the previous engine counts.
        let created = body_kernels_created(body_engine);
        assert!(
            created > before || (engine != Engine::Interpreter && created >= 1),
            "{engine}: no body kernel on {body_engine}"
        );
        k.set_inputs(&[7]);
        k.pull("t");
        assert_eq!(body_kernels_created(body_engine), created, "{engine}");
    }
}
