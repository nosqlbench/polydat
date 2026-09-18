// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Cursors on every engine (engines.md §3.5). A cursor's
//! `over` clause resolves to partitions at build when the clause and
//! the extent are constant; a clause that denotes one partition seeds
//! the cursor so the program runs on every engine with no host call,
//! and a clause that denotes several is narrowed through the same
//! `set_cursor` on the interpreter, the closure tier, and the hybrid
//! kernel. The partition family reads the cursor through its `Ext`
//! slot and the scalar projections through their own.

#![cfg(feature = "jit")]

use polydat::ast::Value;
use polydat::dsl::compile::compile_polydat_to_assembler;
use polydat::iteration::cursor_partition::Partition;

/// The partition family and the scalar projections over one cursor.
const CONSUMERS: &str = "\
n := cardinality(q.cursor)
s := start_of(q.cursor)
e := end_of(q.cursor)
i := idx_of(q.cursor)
c := count_of(q.cursor)
m := mod_in(cycle, q.cursor)
k := clamp_in(cycle, q.cursor)
r := random_in(q.cursor, cycle)
a := at(q.cursor, u64_mod(cycle, cardinality(q.cursor)))
h := partition_count(subdivide(q.cursor, 2))
pi := q.cursor.idx
ps := q.cursor.start_ordinal
pe := q.cursor.end_ordinal
pc := q.cursor.partition_count
";

const OUTPUTS: [&str; 14] = [
    "n", "s", "e", "i", "c", "m", "k", "r", "a", "h", "pi", "ps", "pe", "pc",
];

fn program(over: &str) -> String {
    format!("input cycle: u64\ncursor q = range(0, 1000) over \"{over}\"\n{CONSUMERS}")
}

/// Every output on every engine, for a run of cycles, after `narrow`
/// has been applied to each kernel.
fn agree(src: &str, narrow: Option<&Partition>) {
    let mut p1 = compile_polydat_to_assembler(src)
        .unwrap()
        .compile()
        .unwrap();
    let mut p2 = compile_polydat_to_assembler(src)
        .unwrap()
        .try_compile_raw()
        .unwrap_or_else(|_| panic!("the partition family has a closure form\n{src}"));
    let mut p2pp = compile_polydat_to_assembler(src)
        .unwrap()
        .try_compile()
        .unwrap_or_else(|_| panic!("P2 push-pull\n{src}"));
    let mut hybrid = compile_polydat_to_assembler(src)
        .unwrap()
        .compile_hybrid()
        .unwrap_or_else(|e| panic!("hybrid: {e}\n{src}"));
    if let Some(p) = narrow {
        p1.set_cursor("q", p).unwrap();
        p2.set_cursor("q", p).unwrap();
        p2pp.set_cursor("q", p).unwrap();
        hybrid.set_cursor("q", p).unwrap();
    }
    for c in [0u64, 1, 7, 250, 999, 1000, 4096] {
        p1.set_inputs(&[c]);
        let want: Vec<Value> = OUTPUTS.iter().map(|o| p1.pull(o).clone()).collect();
        p2.eval(&[c]);
        let got_p2: Vec<Value> = OUTPUTS.iter().map(|o| p2.get_value(o)).collect();
        p2pp.eval(&[c]);
        let got_p2pp: Vec<Value> = OUTPUTS.iter().map(|o| p2pp.get_value(o)).collect();
        hybrid.eval(&[c]);
        let got_hybrid: Vec<Value> = OUTPUTS.iter().map(|o| hybrid.get_value(o)).collect();
        for (i, out) in OUTPUTS.iter().enumerate() {
            for (tier, got) in [
                ("P2", &got_p2[i]),
                ("P2 push-pull", &got_p2pp[i]),
                ("hybrid", &got_hybrid[i]),
            ] {
                assert_eq!(
                    want[i].port_type(),
                    got.port_type(),
                    "{tier}: `{out}` at cycle {c}: type\n{src}"
                );
                assert_eq!(
                    want[i].to_display_string(),
                    got.to_display_string(),
                    "{tier}: `{out}` at cycle {c}\n{src}"
                );
            }
        }
    }
}

#[test]
fn a_single_partition_clause_needs_no_host_call_on_any_engine() {
    for over in ["0..50%", "25%..75%", "*"] {
        let src = program(over);
        let asm = compile_polydat_to_assembler(&src).unwrap();
        let schema = &asm.cursor_schemas()[0];
        let parts = schema.partitions.as_ref().expect("resolved at build");
        assert_eq!(parts.len(), 1, "`over \"{over}\"` denotes one partition");
        agree(&src, None);
        // What the interpreter reads is the one partition.
        let mut p1 = compile_polydat_to_assembler(&src)
            .unwrap()
            .compile()
            .unwrap();
        p1.set_inputs(&[0]);
        assert_eq!(p1.pull("n").as_u64(), parts[0].end_ord - parts[0].start_ord);
        assert_eq!(p1.pull("ps").as_u64(), parts[0].start_ord);
    }
}

#[test]
fn every_partition_of_a_list_narrows_every_engine_alike() {
    for over in ["*/4", "20%,30%,*", "10%,*"] {
        let src = program(over);
        let asm = compile_polydat_to_assembler(&src).unwrap();
        let parts = asm.cursor_schemas()[0]
            .partitions
            .clone()
            .expect("resolved at build");
        assert!(
            parts.len() > 1,
            "`over \"{over}\"` denotes several partitions"
        );
        for p in &parts {
            agree(&src, Some(p));
        }
    }
}

#[test]
fn the_compiled_kernels_report_the_cursors_the_interpreter_does() {
    let src = program("*/4");
    let p1 = compile_polydat_to_assembler(&src)
        .unwrap()
        .compile()
        .unwrap();
    let p2 = compile_polydat_to_assembler(&src)
        .unwrap()
        .try_compile_raw()
        .ok()
        .unwrap();
    let hybrid = compile_polydat_to_assembler(&src)
        .unwrap()
        .compile_hybrid()
        .unwrap();
    let names = |s: &[polydat::iteration::source::SourceSchema]| -> Vec<(String, usize)> {
        s.iter()
            .map(|c| (c.name.clone(), c.partitions.as_ref().map_or(0, Vec::len)))
            .collect()
    };
    assert_eq!(
        names(p1.program().cursor_schemas()),
        vec![("q".to_string(), 4)]
    );
    assert_eq!(names(p2.cursor_schemas()), vec![("q".to_string(), 4)]);
    assert_eq!(names(hybrid.cursor_schemas()), vec![("q".to_string(), 4)]);
    // An unknown cursor is refused by name on every engine.
    let p = p2.cursor_schemas()[0].partitions.as_ref().unwrap()[0];
    let mut p2 = p2;
    for (engine, err) in [
        ("closures", p2.set_cursor("zz", &p).unwrap_err()),
        ("interpreter", {
            let mut p1 = p1;
            p1.set_cursor("zz", &p).unwrap_err()
        }),
    ] {
        match err {
            polydat::kernel::WriteError::UnknownWire { key, .. } => {
                assert!(key.contains("zz"), "{engine}: {key}")
            }
            other => panic!("{engine}: expected an unknown-wire error, got {other:?}"),
        }
    }
}

#[test]
fn a_clause_denoting_several_partitions_is_unset_until_narrowed() {
    // The cursor is `None` until the host narrows it, and every
    // consumer reads `None` through it, on the interpreter and on the
    // closure tier alike (engines.md §3.3, §3.5). Native code
    // cannot carry `None` and refuses to run.
    let src = program("*/4");
    let mut p1 = compile_polydat_to_assembler(&src)
        .unwrap()
        .compile()
        .unwrap();
    p1.set_inputs(&[0]);
    assert_eq!(*p1.pull("n"), Value::None);
    let mut p2 = compile_polydat_to_assembler(&src)
        .unwrap()
        .try_compile_raw()
        .ok()
        .unwrap();
    p2.eval(&[0]);
    for o in OUTPUTS {
        // The partition consumers read `None`; the scalar projections
        // read their declared defaults, on both engines.
        assert_eq!(p2.get_value(o), *p1.pull(o), "`{o}` on the closure tier");
    }
    // Narrowed, the same kernel computes; the outputs stop being `None`.
    let parts = compile_polydat_to_assembler(&src).unwrap().cursor_schemas()[0]
        .partitions
        .clone()
        .unwrap();
    p2.set_cursor("q", &parts[1]).unwrap();
    p2.eval(&[0]);
    assert_eq!(
        p2.get_value("n").as_u64(),
        parts[1].end_ord - parts[1].start_ord
    );
}
