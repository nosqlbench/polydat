// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A host's own node, on every engine.
//!
//! `engine_parity` pins the node *library* per engine, and
//! `volatile_is_never_current` pins the `volatile` modifier on the
//! interpreter. Between them sits the shape a host actually registers:
//! a node of its own, declaring `Purity::Nondeterministic` because it
//! reads something polydat cannot see, taking its configuration as
//! `Const<&str>` and handing back a value.
//!
//! Two things have to hold for such a node, and the second is the one
//! that can break quietly:
//!
//! 1. Every engine computes what the interpreter computes.
//! 2. Every engine keeps re-reading it. A nondeterministic node is
//!    never current (runtime_model.md, R1.v), so a compiled kernel must
//!    not fold it at build nor cache it across cycles — if it did, a
//!    host would read one value forever and nothing would say so.
//!
//! Written 2026-09-22, when `compile_polydat` moved from the
//! interpreter to `Engine::default()`. That put every host node on a
//! compiled tier for the first time, and the hosts whose nodes these
//! stand in for had no test that they survived it.

use polydat::dsl::compile::compile_polydat_with;
use polydat::{Engine, JitMode, Kernel, KernelError, Provenance};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// What the node reads and polydat cannot: values that move under it
/// between cycles, the way a metrics reader or a clock does.
///
/// Keyed, so the tests in this binary — which libtest runs as threads
/// of one process — do not move each other's world. Each test owns a
/// prefix.
fn outside() -> &'static Mutex<HashMap<String, f64>> {
    static OUTSIDE: OnceLock<Mutex<HashMap<String, f64>>> = OnceLock::new();
    OUTSIDE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn set(key: &str, v: f64) {
    outside()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key.to_string(), v);
}

/// Reads the outside world, choosing *what* to read from a constant —
/// the shape of a host node that is configured at build and reads at
/// every pull (`metric(label_pattern, stat)` is this).
#[polydat::polydat_node(
    category = Context,
    purity = Nondeterministic("reads state outside the graph; changes between cycles"),
)]
fn host_reading(which: Const<&str>) -> f64 {
    outside()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(which.0)
        .copied()
        .unwrap_or(0.0)
}

/// Every engine a build can name.
fn tiers() -> Vec<Engine> {
    let mut all = vec![
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Auto),
        Engine::Closures(Provenance::Raw),
    ];
    if cfg!(feature = "jit") {
        all.push(Engine::Native(Provenance::Auto));
        all.push(Engine::Native(Provenance::Raw));
        all.push(Engine::PureNative(Provenance::Auto));
        all.push(Engine::PureNative(Provenance::Raw));
    }
    all
}

fn src(prefix: &str) -> String {
    format!("a := host_reading(\"{prefix}.a\")\nb := host_reading(\"{prefix}.b\")\n")
}

fn read(k: &mut dyn Kernel, cycle: u64) -> Vec<polydat::ast::Value> {
    k.set_inputs(&[cycle]);
    ["a", "b"].iter().map(|n| k.pull(n)).collect()
}

/// One kernel per engine, built once and driven together, so every
/// assertion below is the same program at the same moment on each.
fn kernels(prefix: &str) -> Vec<(String, Box<dyn Kernel>)> {
    kernels_of(&src(prefix))
}

/// [`kernels`] over any source.
fn kernels_of(source: &str) -> Vec<(String, Box<dyn Kernel>)> {
    let mut out: Vec<(String, Box<dyn Kernel>)> = Vec::new();
    for tier in tiers() {
        match compile_polydat_with(source, tier) {
            Ok(k) => out.push((format!("{tier}"), k)),
            // A tier that refuses says so; it is not a silent skip.
            Err(KernelError::Refused { .. }) => {
                eprintln!("{tier} refuses a host node of this shape");
            }
            Err(e) => panic!("{tier}: {e}"),
        }
    }
    assert!(
        out.len() >= 3,
        "expected the interpreter and both closure modes at least, got {}",
        out.len()
    );
    out
}

/// Every engine computes what the interpreter computes, and the
/// constant argument reaches the body on each.
#[test]
fn a_host_node_agrees_on_every_engine() {
    set("agree.a", 21.0);
    set("agree.b", 42.0);
    let mut ks = kernels("agree");

    let want = read(ks[0].1.as_mut(), 1);
    assert_eq!(
        want,
        vec![
            polydat::ast::Value::F64(21.0),
            polydat::ast::Value::F64(42.0)
        ],
        "the interpreter itself is wrong; the rest of this test means nothing"
    );

    for (name, k) in ks.iter_mut() {
        assert_eq!(read(k.as_mut(), 1), want, "{name}");
    }
}

/// The node is never current: when what it reads moves, every engine
/// sees the new value on the next cycle.
///
/// This is the half a compiled tier can lose on its own — by folding
/// the step at build, or by treating it as clean because its inputs did
/// not change. Its inputs never change; that is the point of the
/// declaration.
#[test]
fn a_host_node_stays_live_on_every_engine() {
    set("live.a", 1.0);
    set("live.b", 2.0);
    let mut ks = kernels("live");

    let first: Vec<_> = ks
        .iter_mut()
        .map(|(n, k)| (n.clone(), read(k.as_mut(), 1)))
        .collect();

    set("live.a", 7.0);
    set("live.b", 9.0);

    let second: Vec<_> = ks
        .iter_mut()
        .map(|(n, k)| (n.clone(), read(k.as_mut(), 2)))
        .collect();

    let before = &first[0].1;
    let after = &second[0].1;
    assert_ne!(
        before, after,
        "the interpreter did not see the change either; the test proves nothing"
    );
    for (name, v) in &first {
        assert_eq!(v, before, "{name} disagreed before the change");
    }
    for (name, v) in &second {
        assert_eq!(
            v, after,
            "{name} did not see the change — the step was folded or cached"
        );
    }
}

/// Read granularity is the step's, and the step is the engine's
/// (runtime_model.md, R1.v "Read granularity").
///
/// Two volatile wires that share nothing are two steps on the
/// interpreter and the closure tier, and two fusion units on the native
/// tier and pure native code, which fuse only nodes a wire connects. So
/// a change made between two pulls of one write is visible to the
/// second read on every engine: each is read when its own output is
/// pulled.
///
/// What every engine owes regardless: the value is not carried across
/// the write. That is asserted for all of them at the end.
#[test]
fn read_granularity_is_the_engines_step() {
    for (name, k) in kernels("midcycle").iter_mut() {
        set("midcycle.a", 3.0);
        set("midcycle.b", 3.0);
        k.set_inputs(&[1]);
        let a = k.pull("a");
        set("midcycle.b", 9.0);
        let b = k.pull("b");

        assert_eq!(a, polydat::ast::Value::F64(3.0), "{name}: first read");
        assert_eq!(
            b,
            polydat::ast::Value::F64(9.0),
            "{name}: a second volatile read within one write is its own \
             step's, read when its own output was pulled"
        );

        // The guarantee that does not vary: the next write re-reads.
        k.set_inputs(&[2]);
        assert_eq!(
            k.pull("b"),
            polydat::ast::Value::F64(9.0),
            "{name}: no engine may carry a volatile value across a write"
        );
    }
}

/// Two volatile reads that a wire connects are one fusion unit on the
/// native engines, read together at the first pull of either; the
/// interpreter and the closure tier read each at its own step's pull
/// (runtime_model.md R1.v, "Read granularity").
#[test]
fn connected_volatile_reads_are_one_unit_on_native_code() {
    let source = "a := host_reading(\"joined.a\")\n\
                  b := host_reading(\"joined.b\") + a\n";
    for (name, k) in kernels_of(source).iter_mut() {
        set("joined.a", 3.0);
        set("joined.b", 3.0);
        k.set_inputs(&[1]);
        let a = k.pull("a");
        set("joined.b", 9.0);
        let b = k.pull("b");

        assert_eq!(a, polydat::ast::Value::F64(3.0), "{name}: first read");
        let fused = name.starts_with("native") || name.starts_with("pure native");
        let expected = if fused {
            // Read with `a`, whose pull ran the unit they share.
            polydat::ast::Value::F64(6.0)
        } else {
            // Its own step, read when `b` was pulled.
            polydat::ast::Value::F64(12.0)
        };
        assert_eq!(
            b, expected,
            "{name}: connected volatile reads follow the engine's unit"
        );

        k.set_inputs(&[2]);
        assert_eq!(
            k.pull("b"),
            polydat::ast::Value::F64(12.0),
            "{name}: no engine may carry a volatile value across a write"
        );
    }
}
