// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The Miri lane for the arena and value-table paths (SRD 115 §10,
//! the handle analogue of S9(b)). Everything here is pure Rust: the
//! chunked cycle arena, the arena writer, the value table and its
//! installation, and the P2 closure kernels that run string, JSON, and
//! tile nodes over handle slots without native code. Miri adjudicates
//! the aliasing arguments the code makes in words: that arena bytes
//! never move within a cycle, so a resolved `&str` stays valid while
//! more is allocated; that a source inside the arena can be copied into
//! it; that a table entry borrowed for a helper's duration is not
//! written under it; and that the installation pointer is restored on
//! every exit path.
//!
//! Lane command (no JIT, since Miri cannot execute native code):
//!
//! ```sh
//! cargo +nightly miri test -p polydat --no-default-features --test handle_miri
//! ```
//!
//! The same tests run natively in the ordinary suite.

use polydat::ast::Value;
use polydat::dsl::compile::compile_polydat_to_assembler;
use polydat::kernel::{
    begin_root_cycle, cycle_arena_mark, cycle_arena_release, cycle_arena_used, put_thread_bytes, put_thread_str,
    resolve_thread_bytes, resolve_thread_str, with_current_value_table, with_value_table, ArenaWriter, ValueTable,
};
use polydat::JitMode;
use std::fmt::Write as _;

/// A resolved string stays valid, at the same address, while the arena
/// grows across chunks and takes a large allocation.
#[test]
fn arena_bytes_never_move_within_a_cycle() {
    begin_root_cycle();
    let h = put_thread_str("anchor");
    let s: &'static str = resolve_thread_str(h);
    let p = s.as_ptr();
    for _ in 0..40 {
        put_thread_bytes(&[7u8; 5000]);
    }
    let big = put_thread_bytes(&vec![9u8; 3 << 16]);
    assert_eq!(s, "anchor");
    assert_eq!(s.as_ptr(), p);
    assert_eq!(resolve_thread_bytes(big).len(), 3 << 16);
    assert_eq!(resolve_thread_str(put_thread_str("after")), "after");
}

/// A string already in the arena is a valid source for a copy into it.
#[test]
fn a_source_inside_the_arena_copies_into_it() {
    begin_root_cycle();
    let h = put_thread_str("copy me");
    let h2 = put_thread_str(resolve_thread_str(h));
    assert_eq!(resolve_thread_str(h2), "copy me");
    assert_ne!(h, h2);
}

/// The writer builds one contiguous result, relocating when something
/// else allocates between its pushes and when it crosses a chunk.
#[test]
fn the_arena_writer_is_contiguous_under_interleaving_and_chunk_ends() {
    begin_root_cycle();
    let mut w = ArenaWriter::new();
    w.push(b"head");
    let other = put_thread_str("interleaved");
    let n = 42;
    write!(w, "-{n}-").unwrap();
    let h = w.finish();
    assert_eq!(resolve_thread_str(h), "head-42-");
    assert_eq!(resolve_thread_str(other), "interleaved");

    put_thread_bytes(&vec![0u8; (1 << 16) - 10]);
    let mut w = ArenaWriter::new();
    w.push(b"12345");
    w.push(&[b'x'; 20]);
    let s = resolve_thread_str(w.finish());
    assert_eq!(s.len(), 25);
    assert!(s.starts_with("12345") && s.ends_with("xxxxx"));
}

/// Mark and release scope a region to one eval; the reset at the next
/// root cycle reclaims everything.
#[test]
fn mark_release_and_reset_bound_the_arena() {
    begin_root_cycle();
    put_thread_str("kept");
    let mark = cycle_arena_mark();
    let used = cycle_arena_used();
    for _ in 0..30 {
        put_thread_bytes(&[1u8; 4000]);
    }
    cycle_arena_release(mark);
    assert_eq!(cycle_arena_used(), used);
    begin_root_cycle();
    assert_eq!(cycle_arena_used(), 0);
}

/// The value table: entries are replaced in place, a stale generation
/// is refused, and the installation nests and is restored on unwind.
#[test]
fn the_value_table_installation_nests_and_restores() {
    let mut outer = ValueTable::new(2);
    outer.set_generation(7);
    let h = outer.write(0, Value::Json(std::sync::Arc::new(serde_json::json!({"k": 1}))));
    with_value_table(&mut outer, || {
        assert_eq!(with_current_value_table(|t| t.read(h)).to_display_string(), "{\"k\":1}");
        let mut inner = ValueTable::new(1);
        inner.set_generation(9);
        let hi = inner.write(0, Value::U64(5));
        with_value_table(&mut inner, || {
            assert_eq!(with_current_value_table(|t| t.read(hi)).as_u64(), 5);
        });
        // Back to the outer table after the inner installation ends.
        assert_eq!(with_current_value_table(|t| t.len()), 2);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut failing = ValueTable::new(1);
            with_value_table(&mut failing, || panic!("unwind through the installation"));
        }));
        assert!(r.is_err());
        assert_eq!(with_current_value_table(|t| t.len()), 2, "restored on unwind");
    });
    assert!(std::panic::catch_unwind(|| with_current_value_table(|t| t.len())).is_err(), "nothing installed after");
    outer.set_generation(8);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| outer.read(h))).is_err());
}

/// The P2 closure kernels run string, JSON, and tile nodes over handle
/// slots and agree with the interpreter; every run installs the table,
/// begins a root cycle, and validates H4.
#[test]
fn p2_handle_closures_agree_with_the_interpreter() {
    let src = "input cycle: u64\nh := hash(cycle)\ns := __u64_to_string(h)\nu := str_upper(s)\np := printf(\"{}-{:x}-{:>6}\", cycle, h, s)\nj := json_array(h, s, cycle)\no := json_object(json_with(\"id\", h), json_with(\"s\", u))\nt := json_to_str(o)\nq := to_json(s)\nx := json_text(j)\ntile d : json := {\"h\": ${h}, \"s\": ${s}, \"in\": \"x-${h}-y\", \"b\": @if h { true } @else { false }}\n";
    let mut p1 = compile_polydat_to_assembler(src).unwrap();
    p1.set_jit_mode(JitMode::Off);
    let mut p1 = p1.compile().unwrap();
    let mut p2 = compile_polydat_to_assembler(src)
        .unwrap()
        .try_compile_raw()
        .unwrap_or_else(|_| panic!("every node here has a P2 form"));
    let outputs = ["s", "u", "p", "j", "o", "t", "q", "x", "d"];
    for c in 0..3u64 {
        p1.set_inputs(&[c]);
        let want: Vec<Value> = outputs.iter().map(|o| p1.pull(o).clone()).collect();
        p2.eval(&[c]);
        for (i, out) in outputs.iter().enumerate() {
            let got = p2.get_value(out);
            assert_eq!(got.port_type(), want[i].port_type(), "{out} at {c}");
            assert_eq!(got.to_display_string(), want[i].to_display_string(), "{out} at {c}");
        }
    }
}

/// A tile with a projection at P2: the closure runs nested kernels
/// inside the kernel's own root cycle.
#[test]
fn p2_projection_tiles_run_nested_kernels_inside_the_cycle() {
    let src = "input cycle: u64\nh := hash(cycle)\ntile t : text := \"${h}: @for k in 1..3 sep \\\"-\\\" {${k}:${h}}\"\n";
    let mut p1 = compile_polydat_to_assembler(src).unwrap();
    p1.set_jit_mode(JitMode::Off);
    let mut p1 = p1.compile().unwrap();
    let mut p2 = compile_polydat_to_assembler(src).unwrap().try_compile_raw().unwrap_or_else(|_| panic!("P2"));
    for c in 0..2u64 {
        p1.set_inputs(&[c]);
        let want = p1.pull("t").to_display_string();
        p2.eval(&[c]);
        assert_eq!(p2.get_value("t").as_str(), want);
    }
}
