// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 115 §3, step 5: the value table carries `Json`, `Ext`, and
//! `Handle` values through native code as table handles. A table is
//! owned by the engine that runs the code, sized at compile time to one
//! entry per table-kind slot, and written in place (axiom H4). Values
//! enter and leave through it, JSON conversions run natively and agree
//! with P1 bit for bit, a cone releases what it took at the end of its
//! eval, a whole kernel keeps its table for its lifetime, and a stale
//! handle is refused by generation (axioms H3, H6, H7).

#![cfg(feature = "jit")]

use polydat::JitMode;
use polydat::ast::Value;
use polydat::dsl::compile::compile_polydat_to_assembler;
use polydat::kernel::{
    PolydatKernel, ValueTable, cycle_arena_used, with_current_value_table, with_value_table,
};

fn kernel(src: &str, mode: JitMode) -> PolydatKernel {
    let mut asm = compile_polydat_to_assembler(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    asm.set_jit_mode(mode);
    asm.compile().unwrap_or_else(|e| panic!("{e}\n{src}"))
}

fn cones(k: &PolydatKernel) -> Vec<String> {
    let p = k.program();
    (0..p.node_count())
        .map(|i| p.node_meta(i).name.clone())
        .filter(|n| n.starts_with("jit_cone["))
        .collect()
}

fn agree(src: &str, outputs: &[&str], cycles: u64, fused: &[&str]) {
    let mut p1 = kernel(src, JitMode::Off);
    let mut p3 = kernel(src, JitMode::Force);
    let names = cones(&p3);
    for member in fused {
        assert!(
            names.iter().any(|c| c.contains(member)),
            "`{member}` was not fused; cones: {names:?}\n{src}"
        );
    }
    for c in 0..cycles {
        p1.set_inputs(&[c]);
        p3.set_inputs(&[c]);
        for out in outputs {
            let a = p1.pull(out).clone();
            let b = p3.pull(out).clone();
            assert_eq!(
                a.port_type(),
                b.port_type(),
                "{out} at cycle {c}: type\n{src}"
            );
            assert_eq!(
                a.to_display_string(),
                b.to_display_string(),
                "{out} at cycle {c}\n{src}"
            );
        }
    }
}

#[test]
fn json_conversions_run_natively_through_the_table() {
    agree(
        "input cycle: u64\nh := hash(cycle)\nj := __u64_to_json(h)\n",
        &["j"],
        6,
        &["__u64_to_json"],
    );
    agree(
        "input cycle: u64\nb := u64_gt(hash(cycle), 1000)\nj := __bool_to_json(b)\n",
        &["j"],
        6,
        &["__bool_to_json"],
    );
    agree(
        "input cycle: u64\nf := to_f64(hash(cycle)) / 3.0\nj := __f64_to_json(f)\n",
        &["j"],
        6,
        &["__f64_to_json"],
    );
    // `to_json` takes a polymorphic `Value` port, which the macro marks
    // None-tolerant, and a cone excludes None-tolerant nodes so the
    // kernel's None short-circuit stays uniform (SRD-74). It stays on
    // P1 and still agrees.
    agree(
        "input cycle: u64\nh := hash(cycle)\nj := to_json(h)\n",
        &["j"],
        6,
        &[],
    );
}

#[test]
fn a_json_value_chains_natively_to_text_and_back_to_p1() {
    // hash -> json (table handle, interior) -> text (arena handle,
    // boundary out): one cone, the Json wire never leaves native code.
    let src = "input cycle: u64\nh := hash(cycle)\nj := __u64_to_json(h)\ns := json_to_str(j)\n";
    agree(src, &["j", "s"], 6, &["__u64_to_json", "json_to_str"]);
    let mut p3 = kernel(src, JitMode::Force);
    p3.set_inputs(&[2]);
    let j = p3.pull("j").clone();
    assert!(
        matches!(j, Value::Json(_)),
        "the boundary decoded a Json value, got {j:?}"
    );
    assert_eq!(p3.pull("s").as_str(), j.to_display_string());
}

#[test]
fn parsed_json_text_agrees_including_the_error_wrapping() {
    let src = "input cycle: u64\nt := \"{{\\\"n\\\": {cycle}}}\"\nj := __str_to_json(t)\nback := json_to_str(j)\n";
    agree(src, &["j", "back"], 4, &["__str_to_json"]);
    let bad = "input cycle: u64\nt := \"not json {cycle}\"\nj := __str_to_json(t)\n";
    agree(bad, &["j"], 3, &["__str_to_json"]);
}

/// A cone's eval takes table entries and arena bytes for itself and
/// releases both before it returns: the arena cursor is where it was,
/// and no table is left installed on the thread.
#[test]
fn a_cone_eval_releases_its_table_entries_and_arena_bytes() {
    let src = "input cycle: u64\nh := hash(cycle)\nj := __u64_to_json(h)\ns := json_to_str(j)\ntxt := __u64_to_string(h)\n";
    let mut k = kernel(src, JitMode::Force);
    assert!(!cones(&k).is_empty());
    k.set_inputs(&[1]);
    let before = cycle_arena_used();
    let s = k.pull("s").clone();
    let txt = k.pull("txt").clone();
    let j_text = k.pull("j").to_display_string();
    assert_eq!(
        cycle_arena_used(),
        before,
        "the cone released the arena bytes it took"
    );
    assert!(
        std::panic::catch_unwind(|| with_current_value_table(|t| t.len())).is_err(),
        "no table stays installed"
    );
    // The copied-out values are whole and stay valid after the next
    // cycle has reused the storage they came from (axiom H6).
    k.set_inputs(&[2]);
    let _ = k.pull("s");
    assert_eq!(s.as_str(), j_text);
    assert_eq!(txt.as_str(), j_text);
    assert_eq!(txt.port_type(), polydat::ast::PortType::Str);
    // The cycle's arena use is bounded by the largest cone eval, not
    // the sum: pulling many times leaves the cursor where it started.
    let mark = cycle_arena_used();
    for _ in 0..1000 {
        let _ = k.pull("txt");
    }
    assert_eq!(cycle_arena_used(), mark);
}

/// A whole compiled kernel owns its table for its lifetime, sized to
/// its table-kind slots, and each eval replaces the entries in place,
/// so a long run allocates no table growth.
#[test]
fn a_pure_p3_kernel_owns_a_fixed_table_and_replaces_entries_in_place() {
    let src = "input cycle: u64\nh := hash(cycle)\nj := __u64_to_json(h)\nk := __bool_to_json(u64_gt(h, 7))\ns := __u64_to_string(h)\n";
    let asm = compile_polydat_to_assembler(src).unwrap();
    let mut k = asm.try_compile_jit().expect("every node lowers");
    assert_eq!(k.table_len(), 2, "one entry per table-kind slot");
    for c in 0..200u64 {
        k.eval(&[c]);
        let h = k.get("h");
        assert_eq!(k.get_value("j").to_display_string(), h.to_string());
        assert_eq!(
            k.get_value("k").to_display_string(),
            if h > 7 { "true" } else { "false" }
        );
        assert_eq!(k.get_value("s").as_str(), h.to_string());
    }
    // Repeating the same coordinates keeps the handle-producing steps
    // live: their storage belongs to the cycle, so they recompute
    // rather than leaving a slot pointing into a reset arena.
    k.eval(&[5]);
    let first = k.get_value("s");
    k.eval(&[5]);
    assert_eq!(k.get_value("s"), first);
    assert_eq!(k.table_len(), 2);
}

#[test]
fn an_ext_value_round_trips_through_a_table_as_itself() {
    // No non-polymorphic node consumes or produces an Ext natively yet
    // (`identity` is None-tolerant and stays on P1), so the table is
    // exercised at its API: an Ext from a P1 kernel is written to an
    // entry and read back as an equal value, opaque throughout.
    let src = "input cycle: u64\nspec := \"*/{cycle + 2}\"\np := partitions(spec, 100)\n";
    let mut k = kernel(src, JitMode::Off);
    k.set_inputs(&[1]);
    let p = k.pull("p").clone();
    assert!(matches!(p, Value::Ext(_)), "got {p:?}");
    let mut table = ValueTable::new(1);
    let h = table.write(0, p.clone());
    let back = table.read(h);
    assert!(matches!(back, Value::Ext(_)));
    assert_eq!(back.to_display_string(), p.to_display_string());
    assert_eq!(back.port_type(), p.port_type());
}

#[test]
fn every_handle_kind_round_trips_through_its_store() {
    use polydat::kernel::{
        put_thread_bytes, put_thread_str, resolve_thread_bytes, resolve_thread_str,
    };
    let s = put_thread_str("copied into the arena");
    assert_eq!(resolve_thread_str(s), "copied into the arena");
    let b = put_thread_bytes(&[1, 2, 3]);
    assert_eq!(resolve_thread_bytes(b), &[1, 2, 3]);
    let mut table = ValueTable::new(2);
    let j = table.write(
        0,
        Value::Json(std::sync::Arc::new(serde_json::json!({"k": [1, 2]}))),
    );
    assert_eq!(table.read(j).to_display_string(), "{\"k\":[1,2]}");
    let h = table.write(1, Value::Handle(std::sync::Arc::new(42u32)));
    assert!(matches!(table.read(h), Value::Handle(_)));
    // Native helpers reach the table only through the installation
    // the engine makes around its call.
    let n = with_value_table(&mut table, || with_current_value_table(|t| t.len()));
    assert_eq!(n, 2);
}

#[test]
fn a_handle_from_another_generation_is_refused() {
    let mut table = ValueTable::new(1);
    table.set_generation(1);
    let h = table.write(0, Value::U64(9));
    assert_eq!(table.read(h).as_u64(), 9);
    table.set_generation(2);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| table.read(h)));
    assert!(
        r.is_err(),
        "a handle written in generation 1 must not read in generation 2"
    );
    // An entry outside the table is refused too: the layout fixed the
    // count, and native code cannot grow it.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        table.write(1, Value::U64(0))
    }));
    assert!(r.is_err());
}
