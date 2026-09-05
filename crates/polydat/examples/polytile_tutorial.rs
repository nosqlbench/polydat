// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Tutorial companion: every program in docs/polytile_tutorial.md,
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
