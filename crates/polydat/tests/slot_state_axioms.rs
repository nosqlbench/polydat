// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! CI gates for the slot-state axioms (S1–S10) —
//! `polydat/docs/design/jit_boundary.md` §"Slot-state axioms".
//!
//! - **S2** — pointer containment: raw u64 readers refuse
//!   Ref2-colored slots; the typed borrow-checked accessors are
//!   the sanctioned path.
//! - **S5** — skip coherence: the Raw (never-skip) engine is the
//!   oracle; the Push / Pull / PushPull (skip) engines and the
//!   hybrid kernel must agree with it for arbitrary input-change
//!   sequences, including repeats that exercise the clean-skip
//!   paths over Ref state.
//! - **S9(a)** runs implicitly throughout: every eval in these
//!   debug-built tests executes the ref validator.
//! - **No per-thread storage** — a `thread_local!` tripwire: values
//!   belong to kernel states, never to threads.
//! - **S10** — the `from_raw_parts` tripwire: Ref-deref unsafe
//!   stays inside the enumerated allowlist.

use polydat::dsl::compile::compile_polydat_to_assembler;

/// Mixed scalar / register / slice flow — every slot color in one
/// kernel, with a Ref-colored named output (`vsum`) to exercise
/// the typed accessors and a scalar output (`out`) for the raw
/// readers.
const MIXED_SRC: &str = r#"
    input cycle: u64
    a := hash_vec(cycle, 19)
    b := hash_vec(hash(cycle), 19)
    vsum := vec_add(a, b)
    r := reg_splat_i16(cycle)
    r2 := reg_add_i16(r, r)
    lane := reg_lane_i16(r2, 3)
    out := vec_dot(vsum, b) * unit_interval(hash(cycle))
"#;

/// Deterministic input sequence with repeats (repeats are what
/// drive the skip paths the S5 oracle exists to check).
fn input_seq(n: usize) -> Vec<u64> {
    let mut seq = Vec::with_capacity(n);
    let mut last = 7u64;
    for i in 0..n {
        let h = xxhash_rust::xxh3::xxh3_64(&(i as u64).to_le_bytes());
        if h.is_multiple_of(3) {
            seq.push(last); // repeat → skip machinery engages
        } else {
            last = h;
            seq.push(h);
        }
    }
    seq
}

/// S5 — every skip engine must match the Raw oracle on both the
/// scalar output (raw reader) and the Ref output (typed
/// accessor), across a repeat-laden input sequence.
#[test]
fn s5_skip_engines_match_raw_oracle() {
    let mk = || compile_polydat_to_assembler(MIXED_SRC).unwrap();

    let mut raw = mk().try_compile_raw().expect("raw P2");
    let mut push = mk().try_compile_push().expect("push P2");
    let mut pull = mk().try_compile_pull().expect("pull P2");
    let mut pushpull = mk().try_compile().expect("pushpull P2");
    let mut hybrid = mk().compile_hybrid().expect("hybrid");

    let out = raw.resolve_output("out").unwrap();
    let vsum = raw.resolve_output("vsum").unwrap();

    for (i, &cycle) in input_seq(120).iter().enumerate() {
        raw.eval(&[cycle]);
        push.eval(&[cycle]);
        pull.eval(&[cycle]);
        pushpull.eval(&[cycle]);
        hybrid.eval(&[cycle]);

        let want_out = raw.get_slot(out);
        let want_vsum: Vec<f32> = raw.read_vec_f32(vsum).to_vec();

        assert_eq!(push.get_slot(out), want_out, "push scalar @ {i}");
        assert_eq!(pull.get_slot(out), want_out, "pull scalar @ {i}");
        assert_eq!(pushpull.get_slot(out), want_out, "pushpull scalar @ {i}");
        assert_eq!(hybrid.get_slot(out), want_out, "hybrid scalar @ {i}");

        assert_eq!(push.read_vec_f32(vsum), &want_vsum[..], "push ref @ {i}");
        assert_eq!(pull.read_vec_f32(vsum), &want_vsum[..], "pull ref @ {i}");
        assert_eq!(
            pushpull.read_vec_f32(vsum),
            &want_vsum[..],
            "pushpull ref @ {i}"
        );
        assert_eq!(
            hybrid.read_vec_f32(vsum),
            &want_vsum[..],
            "hybrid ref @ {i}"
        );
    }
}

/// S2 — the raw u64 readers refuse Ref2-colored slots (they would
/// leak an interior address); the typed accessor is the sanctioned
/// path and returns the actual contents.
#[test]
fn s2_raw_readers_refuse_ref_slots() {
    let mut p2 = compile_polydat_to_assembler(MIXED_SRC)
        .unwrap()
        .try_compile_raw()
        .expect("P2");
    p2.eval(&[42]);
    let vsum = p2.resolve_output("vsum").unwrap();

    // Typed accessor works and sees real data.
    assert_eq!(p2.read_vec_f32(vsum).len(), 19);

    // Raw readers panic, citing S2.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = p2.get_slot(vsum);
    }));
    assert!(r.is_err(), "get_slot must refuse a Ref2-colored slot");

    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = p2.get("vsum");
    }));
    assert!(r.is_err(), "get must refuse a Ref2-colored output");

    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = p2.eval_for_slot(&[43], vsum);
    }));
    assert!(r.is_err(), "eval_for_slot must refuse a Ref2-colored slot");
}

/// S2 — wrong-typed accessor reads fail loudly instead of
/// reinterpreting scratch.
#[test]
fn s2_typed_accessor_rejects_wrong_lane_type() {
    let mut p2 = compile_polydat_to_assembler(MIXED_SRC)
        .unwrap()
        .try_compile_raw()
        .expect("P2");
    p2.eval(&[42]);
    let vsum = p2.resolve_output("vsum").unwrap();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = p2.read_vec_i32(vsum);
    }));
    assert!(r.is_err(), "f32 scratch must not read as i32");
}

/// S10 — every `from_raw_parts` in the workspace's polydat crates
/// sits in the enumerated allowlist. Anything else is a new
/// Ref-deref unsafe site that must be brought under the axioms
/// (SAFETY comment citing S3/S4) and added here deliberately.
// Miri runs with FS isolation; the tripwire is a source scan, not
// an aliasing concern — native test runs cover it.
#[cfg_attr(miri, ignore)]
#[test]
fn s10_from_raw_parts_tripwire() {
    // (file, why it's allowed)
    let allow: &[(&str, &str)] = &[
        // SliceArc's own borrow projection — predates the axioms,
        // governed by SliceArc's owner-lifetime contract.
        ("src/ast.rs", "SliceArc as_slice"),
        // JIT extern helpers reading node-owned const tables baked
        // at compile time (retain_nodes keeps them alive), and the
        // slot-call helper's view of the native frame and the state's
        // scratch (compiled_handles.md §6).
        (
            "src/compile/jit/codegen.rs",
            "extern const-table reads; slot-call frames",
        ),
        // Dataset accessor reading an mmap-backed uniform facet
        // (vectordata owner-lifetime contract).
        ("src/library/vectors.rs", "dataset facet view"),
        // The one place a `Ref2` pair is dereferenced on the way out of
        // the compiled tier (S7): the boundary decode, under S3/S4.
        ("src/compile/marshal.rs", "reference pair decode"),
        // A copy step reads its producer's pair into its own scratch
        // (S3: pairs are never forwarded).
        ("src/compile/assembly.rs", "reference copy into own scratch"),
        // A string assertion reads its producer's pair before copying
        // it into its own scratch (S3).
        ("src/library/assertions.rs", "string assertion read"),
    ];
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offending = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read_dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let text = std::fs::read_to_string(&path).expect("read");
                if text.contains("from_raw_parts") {
                    let rel = path
                        .strip_prefix(root.parent().unwrap())
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/");
                    if !allow.iter().any(|(f, _)| rel == *f) {
                        offending.push(rel);
                    }
                }
            }
        }
    }
    assert!(
        offending.is_empty(),
        "S10 tripwire: from_raw_parts outside the allowlist (bring the \
         site under the slot-state axioms and add it here deliberately):\n{}",
        offending.join("\n"),
    );
}

/// No value is stored per thread. Storage for a kernel's values
/// belongs to a kernel state: its slot buffer, its scratch entries,
/// its externs (compiled_handles.md §3); the program is shared and
/// holds nothing that changes. A `thread_local!` that held a value,
/// a pointer to one, or a state would tie an output's lifetime to a
/// thread instead of to its provenance. The thread-locals that exist
/// are enumerated here with why each is not value storage; a new one
/// is brought under this rule and added deliberately.
#[cfg_attr(miri, ignore)]
#[test]
fn no_thread_local_value_storage_tripwire() {
    // (file, why it's allowed)
    let allow: &[(&str, &str)] = &[
        // The longjmp target while native code runs: control flow for
        // the panic path, holding no value.
        ("src/compile/jit/codegen.rs", "native panic return target"),
        // The binding name a node is built under, for attribution
        // during one synchronous build call.
        ("src/dsl/factory.rs", "build-time attribution context"),
        // A flag that a node eval runs under the enrichment catch, so
        // the panic hook stays quiet.
        ("src/kernel/engines.rs", "panic-capture flag"),
        // The directory relative data-file paths resolve against, set
        // for the duration of one compile.
        ("src/library/datafile.rs", "compile-time base directory"),
        // The rows a side-channel node emitted, a sink the harness
        // drains; no kernel reads them back.
        ("src/library/emit.rs", "side-channel row sink"),
        // The entropy state of the nondeterministic random nodes.
        ("src/library/random.rs", "entropy source"),
        // The calling thread's own numeric id, a fact about the
        // thread that `thread_id` reports, extracted once per thread.
        ("src/library/context.rs", "the thread's own id"),
    ];
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offending = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read_dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let text = std::fs::read_to_string(&path).expect("read");
                if text.contains("thread_local!") || text.contains("#[thread_local]") {
                    let rel = path
                        .strip_prefix(root.parent().unwrap())
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/");
                    if !allow.iter().any(|(f, _)| rel == *f) {
                        offending.push(rel);
                    }
                }
            }
        }
    }
    assert!(
        offending.is_empty(),
        "thread-local tripwire: a thread_local! outside the allowlist (no value, \
         pointer, or state is stored per thread; see compiled_handles.md §3):\n{}",
        offending.join("\n"),
    );
}

/// Registry `lookup` returns a real `&'static` into the link-time
/// inventory — same address across calls — proving it does not
/// allocate (and leak) per call. Regression guard for the Miri
/// leak finding (2026-06-12): the former impl `Box::leak`'d a
/// clone, leaking ~200 bytes every call.
#[test]
fn lookup_does_not_allocate_per_call() {
    use polydat::dsl::registry::lookup;
    let a = lookup("hash").expect("hash registered") as *const _;
    let b = lookup("hash").expect("hash registered") as *const _;
    assert_eq!(
        a, b,
        "lookup must return a stable &'static, not a fresh leak"
    );
}

/// Axiom S1: the slot color is static, total, and three-valued. The
/// by-reference types (strings, byte strings, JSON, extension values,
/// handles) are `Ref2` like the typed vectors, each with the scratch
/// element its producer owns; the scalar colors are untouched.
#[test]
fn ref2_is_the_color_of_by_reference_types() {
    use polydat::ast::{PortType, ScratchElem, SlotColor};
    for (ty, elem) in [
        (PortType::Str, ScratchElem::Str),
        (PortType::Bytes, ScratchElem::Bytes),
        (PortType::Json, ScratchElem::Value),
        (PortType::Ext, ScratchElem::Value),
        (PortType::Handle, ScratchElem::Value),
        (PortType::VecF32, ScratchElem::F32),
        (PortType::VecI64, ScratchElem::I64),
    ] {
        assert_eq!(ty.slot_color(), SlotColor::Ref2, "{ty:?}");
        assert_eq!(ty.slot_width(), 2, "{ty:?}");
        assert_eq!(ty.scratch_elem(), Some(elem), "{ty:?}");
    }
    for ty in [
        PortType::U64,
        PortType::F64,
        PortType::Bool,
        PortType::I64,
        PortType::U8,
        PortType::F32,
    ] {
        assert_eq!(ty.slot_color(), SlotColor::Imm1, "{ty:?}");
        assert_eq!(ty.scratch_elem(), None, "{ty:?}");
    }
    assert_eq!(PortType::U128.slot_color(), SlotColor::Imm2);
    assert_eq!(PortType::U128.scratch_elem(), None);
}

/// Axiom S2 for by-reference outputs on the hybrid kernel and on pure
/// native code: the raw readers refuse a `Ref2` slot, and the typed
/// reader copies the value out. Native code carries the pair through
/// a slot call of the producing node's kit (compiled_handles.md §6).
#[cfg(feature = "jit")]
#[test]
fn native_kernels_guard_reference_slots_and_copy_them_out() {
    let src =
        "input cycle: u64\nh := hash(cycle)\nj := __u64_to_json(h)\ns := __u64_to_string(h)\n";
    let mut k = compile_polydat_to_assembler(src)
        .unwrap()
        .compile_hybrid()
        .expect("hybrid");
    k.eval(&[3]);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| k.get("s")));
    assert!(r.is_err(), "a raw read of a reference slot must be refused");
    // The typed reader decodes by port type and copies out.
    let h = k.get("h");
    assert_eq!(k.get_value("s").as_str(), h.to_string());
    assert_eq!(k.get_value("j").to_display_string(), h.to_string());
    let mut pure = compile_polydat_to_assembler(src)
        .unwrap()
        .try_compile_pure_jit()
        .expect("pure native code carries reference pairs through slot calls");
    pure.eval(&[3]);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pure.get("s")));
    assert!(
        r.is_err(),
        "a raw read of a reference slot must be refused natively too"
    );
    assert_eq!(pure.get_value("s").as_str(), h.to_string());
    assert_eq!(pure.get_value("j").to_display_string(), h.to_string());
}

/// A program with a string extern and string and JSON outputs, for the
/// created-kernel checks.
const OWNED_PAIRS_SRC: &str = "input cycle: u64\n\
    extern label: str = \"lbl\"\n\
    h := hash(cycle)\n\
    s := __u64_to_string(h)\n\
    j := __u64_to_json(h)\n\
    line := str_concat(label, \"-\", s)\n";

/// Axiom S3 across states: a kernel created from a shared program is a
/// new state whose reference pairs point into its own scratch and
/// extern storage, never into the state it was made from. The source
/// state has run every step before it is shared, and the shared program
/// (which holds it) is gone before the created ones are read, so a pair
/// left pointing into it would read freed memory; the debug
/// ref-validator on every run checks each pair against the state's own
/// entry.
fn created_kernels_own_their_pairs(mut source: Box<dyn polydat::Kernel>) {
    use polydat::ast::Value;
    let mut p1 = polydat::dsl::compile::compile_polydat(OWNED_PAIRS_SRC).unwrap();
    source.set_inputs(&[3]);
    source.pull("line");
    source.pull("j");
    let program = source.into_program();
    let mut a = std::sync::Arc::clone(&program).create_kernel();
    let mut b = program.create_kernel();
    b.set_input("label", Value::Str("other".into())).unwrap();
    for cycle in [3u64, 4, 3] {
        for (kernel, label) in [(&mut a, "lbl"), (&mut b, "other")] {
            p1.set_input("label", Value::Str(label.into())).unwrap();
            p1.set_inputs(&[cycle]);
            kernel.set_inputs(&[cycle]);
            for name in ["s", "j", "line"] {
                assert_eq!(
                    kernel.pull(name).to_display_string(),
                    p1.pull(name).to_display_string(),
                    "cycle {cycle}, label {label}: `{name}`"
                );
            }
        }
    }
}

#[test]
fn a_created_closure_kernel_owns_its_reference_pairs() {
    let kernel = compile_polydat_to_assembler(OWNED_PAIRS_SRC)
        .unwrap()
        .try_compile()
        .unwrap_or_else(|_| panic!("the closure tier refused the program"));
    created_kernels_own_their_pairs(Box::new(kernel));
}

#[cfg(feature = "jit")]
#[test]
fn a_created_hybrid_kernel_owns_its_reference_pairs() {
    let kernel = compile_polydat_to_assembler(OWNED_PAIRS_SRC)
        .unwrap()
        .compile_hybrid()
        .expect("hybrid");
    created_kernels_own_their_pairs(Box::new(kernel));
}
