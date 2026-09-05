// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Illustration: `name := for ...` binds a comprehension as a value. The
//! wire carries a Streamer; derivations filter and order it; streams
//! from one wire advance independently.

use polydat::iteration::comprehension::strategies::TupleValue;

fn show(label: &str, s: &polydat::iteration::comprehension::StreamerValue) {
    let tuples: Vec<String> = s
        .coordinate_stream()
        .map(|t| {
            let cells: Vec<String> = t
                .bindings
                .iter()
                .map(|(_, v)| match v {
                    TupleValue::I64(n) => n.to_string(),
                    TupleValue::U64(n) => n.to_string(),
                    other => format!("{other:?}"),
                })
                .collect();
            format!("({})", cells.join(","))
        })
        .collect();
    println!("{label:<8} {:?}  {:>2} tuples  {}", s.cardinality(), tuples.len(), tuples.join(" "));
}

fn main() {
    let mut kernel = polydat::dsl::compile_polydat(r#"
        input cycle: u64

        base    := for k in 1..4, limit in 10,20,30
        corners := for base where {k} == 1 || {k} == 3
        sampled := for base order halton/4
        label   := "plan: {base}"
    "#).expect("compile failed");

    kernel.set_inputs(&[0]);
    println!("{}", kernel.pull("label").as_str());
    for name in ["base", "corners", "sampled"] {
        let value = kernel.pull(name).clone();
        show(name, value.as_streamer().expect("streamer"));
    }

    // Two streams from one wire never share a cursor.
    let value = kernel.pull("base").clone();
    let base = value.as_streamer().unwrap();
    let mut a = base.coordinate_stream();
    let b = base.coordinate_stream();
    a.next();
    a.next();
    println!("after two pulls on a: a has {} left, b has {}", a.count(), b.count());
}
