// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Illustration: resolve a partition spec against a domain, then let one
//! fiber narrow its cursor to a single partition. The host plays its
//! scope-setup role explicitly here by writing the cursor slots.

use polydat::dsl::compile_polydat_kernel;
use polydat::iteration::cursor_partition::cursor_over_partitions_on;

fn main() {
    // 1. Resolve a partition spec against a domain.
    let mut k = compile_polydat_kernel(
        r#"
        input cycle: u64
        parts := partitions("20%,30%,*", 1000000)
    "#,
    )
    .expect("compile");
    k.set_inputs(&[0]);
    let list = k.pull("parts").as_partition_list().expect("list").clone();
    for p in list.0.iter() {
        println!(
            "p{}  [{:>7}, {:>7})  {:>6} ordinals",
            p.idx,
            p.start_ord,
            p.end_ord,
            p.end_ord - p.start_ord
        );
    }
    println!();

    // 2. One fiber narrows its cursor to one partition.
    let mut k = compile_polydat_kernel(
        r#"
        input cycle: u64
        cursor q = range(0, 1000000) over "20%,30%,*"
        start := q.cursor.start_ordinal
        end   := q.cursor.end_ordinal
        size  := cardinality(q.cursor)
        slot  := mod_in(cycle, q.cursor)
        row   := mod(hash(slot), 1000000)
        sub   := subdivide(q.cursor, 4)
    "#,
    )
    .expect("compile2");

    // The host resolves the `over` spec and hands fiber 1 its partition.
    let schema = k.cursor_schemas()[0].clone();
    let parts = cursor_over_partitions_on(k.as_mut(), &schema).expect("resolve");
    k.set_cursor("q", &parts[1]).expect("narrow");

    for cycle in [0u64, 1, 299_999, 300_000] {
        k.set_inputs(&[cycle]);
        let s = k.pull("start").as_u64();
        let e = k.pull("end").as_u64();
        let n = k.pull("size").as_u64();
        let slot = k.pull("slot").as_u64();
        let row = k.pull("row").as_u64();
        println!("cycle={cycle:<6} window=[{s}, {e}) size={n} slot={slot:<6} row={row}");
    }
    println!();
    k.set_inputs(&[0]);
    let sub = k.pull("sub").as_partition_list().expect("sub").clone();
    for p in sub.0.iter() {
        println!("sub{}  [{}, {})", p.idx, p.start_ord, p.end_ord);
    }
}
