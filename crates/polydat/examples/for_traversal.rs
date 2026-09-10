// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Illustration: `for <comprehension> { body }` activates one child scope
//! per tuple. The body compiles once; each activation is a fresh state
//! over that program with the tuple's elements bound. A cursor declared
//! `over` an element is narrowed per activation, and its slice sets the
//! activation's cycles.

use polydat::kernel::programs_built;

fn main() {
    let mut kernel = polydat::dsl::compile_polydat(
        r#"
        input cycle: u64
        extern total: u64 = 1000
        base := hash(cycle)

        for p in partitions("*/4", {total}), scale in 1,100 {
            cursor rows = range(0, 1000) over p
            row  := mod_in(cycle, rows.cursor)
            v    := u64_add(u64_mul(row, scale), base)
        }
    "#,
    )
    .expect("compile failed");
    kernel.set_inputs(&[7]);

    let program = kernel.program().traversals()[0].program.clone();
    println!(
        "body program: {} nodes, compiled once",
        program.node_count()
    );

    let built_before = programs_built();
    let mut stream = kernel.traverse(0).expect("open traversal");
    println!(
        "{} activations from `{}`",
        stream.len(),
        stream.traversal().source_text
    );
    println!();
    println!("act  p          scale  cycles  first row  first v");
    while let Some(mut act) = stream.advance().expect("activation") {
        let slice = act.cursor.clone().expect("cursor slice");
        let scale = act.coord("scale").unwrap().as_u64();
        let kernel = act.cycle(0);
        let row = kernel.pull("row").as_u64();
        let v = kernel.pull("v").as_u64();
        println!(
            "{:>3}  [{:>3},{:>4})  {:>5}  {:>6}  {:>9}  {}",
            act.index,
            slice.start,
            slice.end,
            scale,
            act.cycle_count(),
            row,
            v
        );
    }
    println!();
    println!(
        "programs built while activating: {}",
        programs_built() - built_before
    );
}
