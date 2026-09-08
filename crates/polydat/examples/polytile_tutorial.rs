// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Tutorial companion: every program in docs/tutorials/polytile_tutorial.md,
//! compiled and rendered here so the document's output is real.

use polydat::dsl::compile_polydat;

fn show(title: &str, src: &str, cycles: &[u64], outputs: &[&str]) {
    println!("== {title} ==");
    let mut kernel = compile_polydat(src).unwrap_or_else(|e| panic!("{title}: {e}"));
    for &cycle in cycles {
        kernel.set_inputs(&[cycle]);
        for name in outputs {
            let value = kernel.pull(name).clone();
            println!("cycle {cycle} {name}: {}", value.to_display_string());
        }
    }
    println!();
}

fn main() {
    show(
        "1. A text tile",
        r#"
            input cycle: u64
            user_id := mod(hash(cycle), 1000000)
            tile greeting := "user ${user_id} on cycle ${cycle}"
        "#,
        &[0, 1],
        &["greeting"],
    );

    show(
        "2. A JSON tile with a static arm",
        r#"
            input cycle: u64
            user_id := mod(hash(cycle), 1000000)
            name    := "user-{user_id}"
            score   := unit_interval(hash(cycle)) * 100.0
            tile doc : json := {
                "meta": { "schema": 3, "source": "polydat", "units": { "score": "pct" } },
                "id": ${user_id},
                "name": ${name},
                "label": "id-${user_id}",
                "score": ${score | .1}
            }
        "#,
        &[0, 1],
        &["doc"],
    );

    show(
        "3. Declared types and formats",
        r#"
            input cycle: u64
            flag := u64_gt(mod(cycle, 2), 0)
            tile typed : json := {"n": ${cycle}, "as_text": ${cycle: str}, "hex": "${cycle | x}", "odd": ${flag: bool}, "padded": "${cycle | 04}"}
        "#,
        &[10, 255],
        &["typed"],
    );

    show(
        "4. A CSV row and a raw hole",
        r#"
            input cycle: u64
            note := "has, comma"
            tile row : csv := "${cycle},${note},${note!}"
        "#,
        &[7],
        &["row"],
    );

    show(
        "5. Branches",
        r#"
            input cycle: u64
            hot := u64_gt(mod(cycle, 3), 1)
            tile status : json := {"cycle": ${cycle}, "state": @if hot { "hot" } @else { "cold" }}
        "#,
        &[0, 1, 2],
        &["status"],
    );

    show(
        "6. Projections over a comprehension and a producer",
        r#"
            input cycle: u64
            base := cycle * 100
            axes := for k in 1..3, side in left,right
            tile samples : json := {
                "base": ${base},
                "points": [ @for i in 0..3 { {"i": ${i}, "v": ${base + i}} } ],
                "grid": "@for axes sep \"; \" {${k}-${side}}"
            }
        "#,
        &[2],
        &["samples"],
    );

    show(
        "6b. Nested projections, derivations, and a generator source",
        r#"
            input cycle: u64
            base := cycle * 10
            ks := for k in 1..7
            tile grid : json := {
                "rows": [ @for r in 0..2 { {"r": ${r}, "cells": [ @for c in 0..3 { ${base + r * 10 + c} } ]} } ],
                "big": [ @for ks where {k} > 4 { ${k} } ],
                "sampled": [ @for k in 1..100 order halton/4 { ${k} } ],
                "pick": [ @for g in hash_range(cycle, 1000) { ${g} } ]
            }
        "#,
        &[1],
        &["grid"],
    );

    show(
        "7. Splicing one tile into another",
        r#"
            input cycle: u64
            tile inner : json := {"n": ${cycle}, "double": ${cycle * 2}}
            tile outer : json := {"first": ${inner}, "second": ${inner}, "wrapped": true}
            tile stmt := "INSERT INTO docs (id, body) VALUES (${cycle}, '${outer!}')"
        "#,
        &[5],
        &["outer", "stmt"],
    );

    show(
        "8. Custom delimiters for a template that lives inside another template",
        r##"
            input cycle: u64
            tile page (delims "<%" "%>", sigil "#") := "Hello {{ user.name }}, cycle <%cycle%> #if cycle { is live } #else { is zero }"
        "##,
        &[0, 3],
        &["page"],
    );

    show(
        "9. Templates handed in as data: polytile_json and polytile",
        r##"
            input cycle: u64
            tags := for t in a,b
            verbose := u64_gt(cycle, 0)
            doc := polytile_json(<<<
{
  "id": "${cycle}",
  "label": "row-${cycle}",
  "samples": [ "@for s in 0..2", { "n": "${s}", "twice": "${s * 2}" } ],
  "audit": [ "@if verbose", { "by": "ops" }, "@else", null ],
  "tags": { "@for tags": { "${t}": true } }
}
>>>)
            line := polytile("text", "cycle ${cycle}: ${doc!}")
        "##,
        &[0, 1],
        &["doc", "line"],
    );

    host_boundary();

    show(
        "11. Escapes, braces, and block layout",
        r#"
            input cycle: u64
            tile lit := "a literal ${${cycle} and braces {ok} around cycle ${cycle}"
            tile block : json := {
                "a": ${cycle},
                    "b": { "deep": ${cycle} }
            }
        "#,
        &[3],
        &["lit", "block"],
    );

    show(
        "12. Encoding corner cases",
        r##"
            input cycle: u64
            line := printf("quote {} and comma, then newline{}end", "\"", "\n")
            tile injson : json := {"s": ${line}, "in": "x-${line}-y", "n": ${cycle: str}}
            tile row : csv := "${line},${cycle},${cycle | 04}"
            tile txt := "[${line}]"
            tile raw : json := {"raw": "${line!}"}
        "##,
        &[1],
        &["injson", "row", "txt", "raw"],
    );

    show(
        "13. Empty projections and separators",
        r#"
            input cycle: u64
            tile doc : json := {"before": 1, "none": [@for k in 1..1 { ${k} }], "after": 2, "ys": [@for k in 0..3 sep " | " { ${k + cycle} }]}
            tile line := "@for k in 0..3 {${k}}|@for k in 0..3 sep \", \" {${k}}"
            tile cells : csv := "@for k in 0..3 {${k + cycle}}"
        "#,
        &[1],
        &["doc", "line", "cells"],
    );

    show(
        "14. A tile inside a module",
        r#"
            input cycle: u64
            label(n: u64, tag: str) -> (out: str) := {
                tile t : json := {"n": ${n}, "twice": ${n * 2}, "tag": ${tag}}
                out := t
            }
            a := label(n: cycle, tag: "first")
            b := label(n: cycle + 100, tag: "second")
        "#,
        &[2],
        &["a", "b"],
    );

    println!("== 15. What the compiler refuses ==");
    fail(
        "a declared type the catalog cannot reach",
        "input cycle: u64\ns := __u64_to_string(cycle)\ntile t : json := {\"n\": ${s: u64}}\n",
    );
    fail(
        "a predicate over a per-cycle outer wire",
        "input cycle: u64\nlimit := cycle\ntile t : text := \"@for k in 1..9 where {k} < {limit} {${k}}\"\n",
    );
    fail(
        "a continuous source without a sampling order",
        "input cycle: u64\ntile t : text := \"@for x in 0.0..1.0 {${x}}\"\n",
    );
    fail(
        "strict mode and an implicit adapter",
        "input cycle: u64\ntile t : json (strict) := {\"f\": ${cycle: f64}}\n",
    );
    fail(
        "a json body that is not valid JSON once the holes are typed",
        "input cycle: u64\ntile t : json := {\"n\": ${cycle} \"x\": 1}\n",
    );
    println!();

    tiers();
}

/// Print the compiler's diagnostic for a program it refuses.
fn fail(title: &str, src: &str) {
    match compile_polydat(src) {
        Ok(_) => println!("{title}: compiled (unexpected)"),
        Err(e) => println!("{title}:\n  {}", e.to_string().replace('\n', "\n  ")),
    }
}

/// The same tile at every engine level: the interpreter, the production
/// kernel with fused native cones, the P2 closure kernel, and the
/// pure-P3 native kernel produce identical bytes.
fn tiers() {
    use polydat::dsl::compile::compile_polydat_to_assembler;
    let src = r#"
        input cycle: u64
        user_id := mod(hash(cycle), 1000000)
        name    := "user-{user_id}"
        tile doc : json := {"id": ${user_id}, "name": ${name}, "label": "id-${user_id}", "tags": [@for k in 0..2 { ${k + user_id} }]}
    "#;
    println!("== 16. One tile, every engine ==");
    let mut p1 = compile_polydat_to_assembler(src).unwrap();
    p1.set_jit_mode(polydat::JitMode::Off);
    let mut p1 = p1.compile().unwrap();
    let mut cones = compile_polydat_to_assembler(src).unwrap();
    cones.set_jit_mode(polydat::JitMode::Force);
    let mut cones = cones.compile().unwrap();
    let mut p2 = compile_polydat_to_assembler(src).unwrap().try_compile_raw().unwrap_or_else(|_| panic!("P2 closures"));
    let mut p3 = compile_polydat_to_assembler(src).unwrap().try_compile_jit().expect("pure P3");
    for cycle in [0u64, 1] {
        p1.set_inputs(&[cycle]);
        let a = p1.pull("doc").to_display_string();
        cones.set_inputs(&[cycle]);
        let b = cones.pull("doc").to_display_string();
        p2.eval(&[cycle]);
        let c = p2.get_value("doc").to_display_string();
        p3.eval(&[cycle]);
        let d = p3.get_value("doc").to_display_string();
        println!("cycle {cycle} P1:    {a}");
        println!("cycle {cycle} cones: {}", if b == a { "identical" } else { &b });
        println!("cycle {cycle} P2:    {}", if c == a { "identical" } else { &c });
        println!("cycle {cycle} P3:    {}", if d == a { "identical" } else { &d });
    }
    println!();
}

/// A host that already holds the template as a parsed JSON value hands
/// it in without going through source text at all.
fn host_boundary() {
    use polydat::tile::{compile_polydat_with_tiles, tile_from_json_value, Span, TileOptions};

    let template = serde_json::json!({
        "id": "${cycle}",
        "points": [ "@for s in 0..3", { "n": "${s}", "v": "${cycle + s}" } ]
    });
    let tile = tile_from_json_value("doc", &template, &TileOptions::default(), Span { line: 0, col: 0 })
        .expect("structural template");
    let mut kernel = compile_polydat_with_tiles("input cycle: u64\n", vec![tile]).expect("compile");
    kernel.set_inputs(&[4]);
    println!("== 10. A tile from a parsed JSON value ==");
    println!("cycle 4 doc: {}", kernel.pull("doc").as_str());
    println!();
}
