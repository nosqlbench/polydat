// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Compile the toy test definition grammar and print a few rows of each phase.

use polydat::ast::Value;
use polydat::iteration::cursor_partition::{parse, resolve};

fn main() {
    // The module defined inside the grammar file is resolved by the
    // module loader, which searches the library paths given here.
    let examples_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples");
    let src = include_str!("toy_test_definition.polydat");
    let mut k = polydat::dsl::compile_polydat_with_libs(
        src, None, vec![examples_dir], &[], false, "toy_test_definition",
    ).expect("compile failed");

    // Play the host's role for the partitioned cursor: resolve the
    // spec, take fiber 1's slice, and write the cursor slots.
    let spec = parse("*/4").expect("spec");
    let mine = resolve(&spec, 0, 1_000_000).expect("resolve")[1].clone();
    let writes = [
        ("rows__cursor", Value::from_partition(mine.clone())),
        ("rows__cursor__idx", Value::U64(mine.idx)),
        ("rows__cursor__partition_count", Value::U64(mine.count)),
        ("rows__cursor__start_ordinal", Value::U64(mine.start_ord)),
        ("rows__cursor__end_ordinal", Value::U64(mine.end_ord)),
    ];
    for (name, value) in writes {
        let idx = k.program().find_input(name).expect(name);
        k.state().set_input(idx, value);
    }

    // Override a runtime parameter: one reading per minute instead of
    // the declared one-per-second default.
    let idx = k.program().find_input("interval_ms").expect("interval_ms");
    k.state().set_input(idx, Value::U64(60_000));

    k.set_inputs(&[0]);
    println!("dataset: {}", k.pull("dataset").as_str());
    println!("shape:   {}", k.pull("shape").as_str());
    println!("fiber:   partition {} of {} = [{}, {})", mine.idx, mine.count, mine.start_ord, mine.end_ord);
    println!("schema:  {}", k.pull("schema_stmt").as_str());
    println!();
    for cycle in [0u64, 1, 1000, 12_345] {
        k.set_inputs(&[cycle]);
        let t = k.pull("tenant").as_u64();
        let d = k.pull("device").as_u64();
        let r = k.pull("reading").as_u64();
        let kind = k.pull("device_kind").as_str().to_string();
        let row = k.pull("row").as_u64();
        println!("cycle {cycle}: row={row} tenant={t} device={d} reading={r} kind={kind}");
        let load = k.pull("load_stmt").as_str().to_string();
        let read = k.pull("read_stmt").as_str().to_string();
        let verify = k.pull("verify_stmt").as_str().to_string();
        let flagged = k.pull("flagged").as_u64();
        println!("  load:   {load}");
        println!("  read:   {read}");
        println!("  verify: {verify}  flagged={flagged}");
    }
}
