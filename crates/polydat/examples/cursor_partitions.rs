// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Illustration: resolve a partition spec against a domain, then let one
//! fiber narrow its cursor to a single partition. The host plays its
//! scope-setup role explicitly here by writing the cursor slots.

use polydat::ast::Value;
use polydat::dsl::compile_polydat;
use polydat::iteration::cursor_partition::{parse, resolve};

fn main() {
    // 1. Resolve a partition spec against a domain.
    let mut k = compile_polydat(r#"
        input cycle: u64
        parts := partitions("20%,30%,*", 1000000)
    "#).expect("compile");
    k.set_inputs(&[0]);
    let list = k.pull("parts").as_partition_list().expect("list").clone();
    for p in list.0.iter() {
        println!("p{}  [{:>7}, {:>7})  {:>6} ordinals", p.idx, p.start_ord, p.end_ord, p.end_ord - p.start_ord);
    }
    println!();

    // 2. One fiber narrows its cursor to one partition.
    let mut k = compile_polydat(r#"
        input cycle: u64
        cursor q = range(0, 1000000) over "20%,30%,*"
        start := q.cursor.start_ordinal
        end   := q.cursor.end_ordinal
        size  := cardinality(q.cursor)
        slot  := mod_in(cycle, q.cursor)
        row   := mod(hash(slot), 1000000)
        sub   := subdivide(q.cursor, 4)
    "#).expect("compile2");

    // The host resolves the `over` spec and hands fiber 1 its partition.
    let spec = parse("20%,30%,*").expect("spec");
    let mine = resolve(&spec, 0, 1_000_000).expect("resolve")[1].clone();
    let writes = [
        ("q__cursor", Value::from_partition(mine.clone())),
        ("q__cursor__idx", Value::U64(mine.idx)),
        ("q__cursor__partition_count", Value::U64(mine.count)),
        ("q__cursor__start_ordinal", Value::U64(mine.start_ord)),
        ("q__cursor__end_ordinal", Value::U64(mine.end_ord)),
    ];
    for (name, value) in writes {
        let idx = k.program().find_input(name).expect(name);
        k.state().set_input(idx, value);
    }

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
