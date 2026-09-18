// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Extension values on every engine. A host type that implements
//! `ReflectedValue` rides a wire as `Value::Ext`; nodes take and return
//! it through `Ext<T>`. Such a node has a slot kit (SRD 115 §5) that
//! reads the value through its pair and writes its own into the step's
//! scratch, which the closure tier runs as a step and native code calls
//! in place (SRD 115 §6).

use polydat::ast::{ReflectedValue, Value};
use polydat::derive_support::Ext;
use polydat::dsl::compile::compile_polydat_to_assembler;

#[derive(Debug, Clone, PartialEq)]
struct Span {
    lo: u64,
    hi: u64,
}

impl ReflectedValue for Span {
    fn type_name(&self) -> &str {
        "Span"
    }
    fn display(&self) -> String {
        format!("[{}, {})", self.lo, self.hi)
    }
    fn clone_reflected(&self) -> Box<dyn ReflectedValue> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[polydat::polydat_node(category = Math, struct_name = SpanOf)]
fn span_of(center: u64, width: u64) -> Ext<Span> {
    Ext(Span {
        lo: center.saturating_sub(width),
        hi: center.saturating_add(width),
    })
}

#[polydat::polydat_node(category = Math)]
fn span_len(span: Ext<Span>) -> u64 {
    span.hi - span.lo
}

#[polydat::polydat_node(category = Math)]
fn span_shift(span: Ext<Span>, by: u64) -> Ext<Span> {
    Ext(Span {
        lo: span.lo + by,
        hi: span.hi + by,
    })
}

#[polydat::polydat_node(category = String)]
fn span_text(span: Ext<Span>) -> String {
    span.display()
}

const SRC: &str = "input cycle: u64\n\
    h := hash(cycle)\n\
    s := span_of(mod(h, 1000), mod(cycle, 7))\n\
    t := span_shift(s, 10)\n\
    n := span_len(t)\n\
    label := span_text(t)\n\
    doc := json_object(json_with(\"n\", n), json_with(\"label\", label))\n";

#[test]
fn extension_nodes_agree_between_interpreter_closures_and_hybrid() {
    let mut p1 = compile_polydat_to_assembler(SRC).unwrap();
    p1.set_jit_mode(polydat::JitMode::Off);
    let mut p1 = p1.compile().expect("P1");
    let mut p2 = compile_polydat_to_assembler(SRC)
        .unwrap()
        .try_compile_raw()
        .unwrap_or_else(|_| panic!("the closure tier refused a program with extension nodes"));
    let mut hybrid = compile_polydat_to_assembler(SRC)
        .unwrap()
        .compile_hybrid()
        .expect("hybrid");
    for cycle in 0..64u64 {
        p1.set_inputs(&[cycle]);
        let n = p1.pull("n").as_u64();
        let label = p1.pull("label").as_str().to_string();
        let doc = p1.pull("doc").to_display_string();
        let Value::Ext(t) = p1.pull("t").clone() else {
            panic!("t is an Ext wire")
        };
        let t = t.as_any().downcast_ref::<Span>().expect("a Span").clone();

        p2.eval(&[cycle]);
        assert_eq!(p2.get("n"), n, "cycle {cycle}: P2 n");
        assert_eq!(
            p2.get_value("label").as_str(),
            label,
            "cycle {cycle}: P2 label"
        );
        assert_eq!(
            p2.get_value("doc").to_display_string(),
            doc,
            "cycle {cycle}: P2 doc"
        );
        let Value::Ext(t2) = p2.get_value("t") else {
            panic!("P2 t is an Ext wire")
        };
        assert_eq!(
            t2.as_any().downcast_ref::<Span>(),
            Some(&t),
            "cycle {cycle}: P2 t"
        );

        hybrid.eval(&[cycle]);
        assert_eq!(hybrid.get("n"), n, "cycle {cycle}: hybrid n");
        assert_eq!(
            hybrid.get_value("label").as_str(),
            label,
            "cycle {cycle}: hybrid label"
        );
        assert_eq!(
            hybrid.get_value("doc").to_display_string(),
            doc,
            "cycle {cycle}: hybrid doc"
        );
        let Value::Ext(t3) = hybrid.get_value("t") else {
            panic!("hybrid t is an Ext wire")
        };
        assert_eq!(
            t3.as_any().downcast_ref::<Span>(),
            Some(&t),
            "cycle {cycle}: hybrid t"
        );
    }
}

/// An extension node runs as a slot call of its kit inside native
/// code (compiled_handles.md §6): the whole program is one segment on
/// the hybrid kernel and one function on pure native code, and the
/// extension value crosses nowhere as anything but the pair into its
/// producing step's scratch.
#[test]
fn extension_nodes_run_as_slot_calls_in_native_code() {
    let hybrid = compile_polydat_to_assembler(SRC)
        .unwrap()
        .compile_hybrid()
        .expect("hybrid");
    let (native, closures) = hybrid.engine_counts();
    #[cfg(feature = "jit")]
    {
        assert_eq!(
            closures, 0,
            "every node has a kit, so none is a closure step"
        );
        assert!(
            native >= 1,
            "the program runs as native segments, got {native}"
        );
        let mut p1 = compile_polydat_to_assembler(SRC).unwrap();
        p1.set_jit_mode(polydat::JitMode::Off);
        let mut p1 = p1.compile().expect("P1");
        let mut pure = compile_polydat_to_assembler(SRC)
            .unwrap()
            .try_compile_pure_jit()
            .expect("pure native code calls the extension nodes' kits");
        for cycle in [0u64, 1, 1, 9] {
            p1.set_inputs(&[cycle]);
            pure.eval(&[cycle]);
            for name in ["n", "label", "doc"] {
                assert_eq!(
                    pure.get_value(name).to_display_string(),
                    p1.pull(name).to_display_string(),
                    "cycle {cycle}: pure native `{name}`"
                );
            }
        }
    }
    #[cfg(not(feature = "jit"))]
    {
        assert_eq!(native, 0, "nothing is native without the jit feature");
        assert!(closures >= 3, "the nodes are closure steps, got {closures}");
    }
}

// ── Fallible construction and tuple returns on the closure tier ──

/// A fallible node runs its body once at construction and replays the
/// cached value; the closure kits write that value every run.
#[polydat::polydat_node(category = Math)]
fn fixed_seed(base: polydat::derive_support::Const<u64>) -> Result<u64, String> {
    if *base == 0 {
        Err("fixed_seed: base must be non-zero".into())
    } else {
        Ok(base.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }
}

#[polydat::polydat_node(category = String)]
fn fixed_label(prefix: polydat::derive_support::Const<&str>) -> Result<String, String> {
    if prefix.is_empty() {
        Err("fixed_label: empty prefix".into())
    } else {
        Ok(format!("{}-fixed", *prefix))
    }
}

#[polydat::polydat_node(category = Math, struct_name = FixedSpanNode)]
fn fixed_span(
    lo: polydat::derive_support::Const<u64>,
    hi: polydat::derive_support::Const<u64>,
) -> Result<Ext<Span>, String> {
    if *lo >= *hi {
        Err("fixed_span: empty".into())
    } else {
        Ok(Ext(Span { lo: *lo, hi: *hi }))
    }
}

#[polydat::polydat_node(category = Math)]
fn fixed_doc(
    n: polydat::derive_support::Const<u64>,
) -> Result<std::sync::Arc<serde_json::Value>, String> {
    Ok(std::sync::Arc::new(
        serde_json::json!({ "n": *n, "twice": *n * 2 }),
    ))
}

/// A tuple return with one element of every shape the handle kit
/// writes: a carrier, a string, an extension value, and JSON.
#[polydat::polydat_node(category = Math, output_names(len, text, moved, doc))]
fn span_parts(
    span: Ext<Span>,
    by: u64,
) -> (u64, String, Ext<Span>, std::sync::Arc<serde_json::Value>) {
    let moved = Span {
        lo: span.lo + by,
        hi: span.hi + by,
    };
    let doc = std::sync::Arc::new(serde_json::json!({ "lo": moved.lo, "hi": moved.hi }));
    (span.hi - span.lo, span.display(), Ext(moved), doc)
}

const SHAPES: &str = "input cycle: u64\n\
    h := hash(cycle)\n\
    seed := fixed_seed(7)\n\
    label := fixed_label(\"job\")\n\
    fs := fixed_span(10, 20)\n\
    fd := fixed_doc(3)\n\
    s := span_of(mod(h, 1000), mod(cycle, 7))\n\
    (len, text, moved, doc) := span_parts(s, mod(seed, 100))\n\
    tail := span_len(moved)\n\
    line := \"{label}/{len}/{text}/{tail}/{fs}/{fd}/{doc}\"\n";

#[test]
fn fallible_and_tuple_nodes_agree_between_interpreter_closures_and_hybrid() {
    let mut p1 = compile_polydat_to_assembler(SHAPES).unwrap();
    p1.set_jit_mode(polydat::JitMode::Off);
    let mut p1 = p1.compile().expect("P1");
    let mut p2 = compile_polydat_to_assembler(SHAPES)
        .unwrap()
        .try_compile_raw()
        .unwrap_or_else(|e| panic!("the closure tier refused a fallible or tuple node: {e}"));
    let mut hybrid = compile_polydat_to_assembler(SHAPES)
        .unwrap()
        .compile_hybrid()
        .expect("hybrid");
    let outputs = [
        "seed", "label", "fs", "fd", "len", "text", "moved", "doc", "tail", "line",
    ];
    for cycle in 0..64u64 {
        p1.set_inputs(&[cycle]);
        let want: Vec<(polydat::ast::PortType, String)> = outputs
            .iter()
            .map(|o| {
                let v = p1.pull(o);
                (v.port_type(), v.to_display_string())
            })
            .collect();
        p2.eval(&[cycle]);
        for (i, o) in outputs.iter().enumerate() {
            let v = p2.get_value(o);
            assert_eq!(
                (v.port_type(), v.to_display_string()),
                want[i],
                "cycle {cycle}: P2 `{o}`"
            );
        }
        hybrid.eval(&[cycle]);
        for (i, o) in outputs.iter().enumerate() {
            let v = hybrid.get_value(o);
            assert_eq!(
                (v.port_type(), v.to_display_string()),
                want[i],
                "cycle {cycle}: hybrid `{o}`"
            );
        }
    }
    let (native, closures) = hybrid.engine_counts();
    #[cfg(feature = "jit")]
    assert!(
        native >= 1 && closures == 0,
        "the fallible and tuple nodes run as slot calls inside native segments, got \
         {native} native and {closures} closure steps"
    );
    #[cfg(not(feature = "jit"))]
    assert!(
        native == 0 && closures >= 5,
        "the fallible and tuple nodes run as closure steps, got {closures}"
    );
}

#[test]
fn a_failing_construction_is_a_compile_error_on_every_tier() {
    let src = "input cycle: u64\nseed := fixed_seed(0)\n";
    let err = compile_polydat_to_assembler(src)
        .err()
        .expect("construction fails at compile time");
    assert!(err.contains("base must be non-zero"), "{err}");
}

// ── Externs on every engine ──

/// A type the host defines, to carry through an extern.
#[derive(Debug, Clone, PartialEq)]
struct Region(String);

impl ReflectedValue for Region {
    fn type_name(&self) -> &str {
        "Region"
    }
    fn display(&self) -> String {
        self.0.clone()
    }
    fn clone_reflected(&self) -> Box<dyn ReflectedValue> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[polydat::polydat_node(category = String)]
fn region_code(region: Ext<Region>, n: u64) -> String {
    format!("{}-{n}", (*region).0.to_uppercase())
}

/// Externs with defaults and externs the host sets, of every kind the
/// compiled engines carry: a carrier, a string, JSON, and an extension
/// value. Every engine seeds the defaults, takes a host value through
/// `set_input`, and agrees with the interpreter afterwards.
const EXTERNS: &str = "input cycle: u64\n\
    extern scale: u64 = 10\n\
    extern label: str = \"lbl\"\n\
    extern doc: json\n\
    extern region: ext\n\
    h := hash(cycle)\n\
    id := mod_wire(h, scale)\n\
    tag := str_concat(label, \"-\", id)\n\
    text := json_to_str(doc)\n\
    code := region_code(region, id)\n\
    line := \"{tag}/{text}/{code}\"\n";

fn host_values(doc_n: u64, region: &str) -> [(&'static str, Value); 4] {
    [
        ("scale", Value::U64(1000)),
        ("label", Value::Str("host".into())),
        (
            "doc",
            Value::Json(std::sync::Arc::new(serde_json::json!({ "n": doc_n }))),
        ),
        ("region", Value::Ext(Box::new(Region(region.to_string())))),
    ]
}

#[test]
fn externs_agree_between_interpreter_closures_hybrid_and_native() {
    let mut p1 = polydat::dsl::compile::compile_polydat(EXTERNS).expect("interpreter");
    let mut p2 = compile_polydat_to_assembler(EXTERNS)
        .unwrap()
        .try_compile_raw()
        .unwrap_or_else(|_| panic!("the closure tier refused externs"));
    let mut p2pp = compile_polydat_to_assembler(EXTERNS)
        .unwrap()
        .try_compile()
        .unwrap_or_else(|_| panic!("the push-pull closure tier refused externs"));
    let mut hybrid = compile_polydat_to_assembler(EXTERNS)
        .unwrap()
        .compile_hybrid()
        .expect("hybrid");
    // `doc` and `region` have no default, so the host sets every extern
    // before the first run and again between rounds; the compiled
    // engines evaluate eagerly and refuse to start with an unset JSON
    // or extension extern.
    let read = ["id", "tag", "text", "code", "line"];
    for round in 0..3u64 {
        let (doc_n, region) = (round * 7, ["east", "west", "north"][round as usize]);
        for (name, value) in host_values(doc_n, region) {
            p1.set_input(name, value.clone()).expect("P1 set_input");
            p2.set_input(name, value.clone()).expect("P2 set_input");
            p2pp.set_input(name, value.clone())
                .expect("P2 push-pull set_input");
            hybrid.set_input(name, value).expect("hybrid set_input");
        }
        for cycle in [0u64, 1, 1, 2, 2, 0] {
            p1.set_inputs(&[cycle]);
            let want: Vec<(polydat::ast::PortType, String)> = read
                .iter()
                .map(|o| {
                    let v = p1.pull(o);
                    (v.port_type(), v.to_display_string())
                })
                .collect();
            p2.eval(&[cycle]);
            for (i, o) in read.iter().enumerate() {
                let v = p2.get_value(o);
                assert_eq!(
                    (v.port_type(), v.to_display_string()),
                    want[i],
                    "round {round} cycle {cycle}: P2 `{o}`"
                );
            }
            p2pp.eval(&[cycle]);
            for (i, o) in read.iter().enumerate() {
                let v = p2pp.get_value(o);
                assert_eq!(
                    (v.port_type(), v.to_display_string()),
                    want[i],
                    "round {round} cycle {cycle}: P2 push-pull `{o}`"
                );
            }
            hybrid.eval(&[cycle]);
            for (i, o) in read.iter().enumerate() {
                let v = hybrid.get_value(o);
                assert_eq!(
                    (v.port_type(), v.to_display_string()),
                    want[i],
                    "round {round} cycle {cycle}: hybrid `{o}`"
                );
            }
        }
    }
}

#[test]
fn an_extern_set_to_the_wrong_type_is_refused_by_name() {
    let mut p2 = compile_polydat_to_assembler(EXTERNS)
        .unwrap()
        .try_compile_raw()
        .unwrap_or_else(|_| panic!("P2"));
    // The write rule refuses by variant, not by message text: the
    // structured error names the slot, what it wanted, and what it got.
    let err = p2
        .set_input("scale", Value::Str("ten".into()))
        .expect_err("a string is not a u64");
    assert_eq!(
        err,
        polydat::kernel::WriteError::TypeMismatch {
            slot: "scale".into(),
            expected: polydat::ast::PortType::U64,
            got: polydat::ast::PortType::Str,
        },
        "{err}"
    );
    let err = p2
        .set_input("nope", Value::U64(1))
        .expect_err("no such extern");
    match &err {
        polydat::kernel::WriteError::UnknownWire { key, known } => {
            assert_eq!(key, "nope");
            // The error carries what the caller could have meant.
            assert!(known.iter().any(|n| n == "scale"), "{known:?}");
        }
        other => panic!("expected an unknown-wire error, got {other:?}"),
    }
    assert_eq!(p2.externs().len(), 4);
}

/// Pure native code with externs. Carrier externs reach native code
/// through their slots; a string, JSON, or extension extern reaches it
/// as the pair into the value the state stores, and the nodes over
/// them run as slot calls inside the one native function
/// (compiled_handles.md §6). Every extern is set by the host between
/// runs.
#[cfg(feature = "jit")]
#[test]
fn externs_agree_between_interpreter_and_pure_native_code() {
    const SRC: &str = "input cycle: u64\n\
        extern scale: u64 = 10\n\
        extern offset: u64 = 3\n\
        h := hash(cycle)\n\
        id := mod_wire(h, scale)\n\
        n := u64_add(id, offset)\n\
        f := to_f64(n) / 2.0\n";
    let mut p1 = polydat::dsl::compile::compile_polydat(SRC).expect("interpreter");
    let mut p3 = compile_polydat_to_assembler(SRC)
        .unwrap()
        .try_compile_pure_jit()
        .expect("pure native code with carrier externs");
    let all = ["id", "n", "f"];
    for round in 0..3u64 {
        let host = [
            ("scale", Value::U64(100 + round)),
            ("offset", Value::U64(round * 7)),
        ];
        for (name, value) in host {
            p1.set_input(name, value.clone()).expect("P1 set_input");
            p3.set_input(name, value).expect("P3 set_input");
        }
        for cycle in [0u64, 1, 1, 2, 2, 0] {
            p1.set_inputs(&[cycle]);
            let want: Vec<(polydat::ast::PortType, String)> = all
                .iter()
                .map(|o| {
                    let v = p1.pull(o);
                    (v.port_type(), v.to_display_string())
                })
                .collect();
            p3.eval(&[cycle]);
            for (i, o) in all.iter().enumerate() {
                let v = p3.get_value(o);
                assert_eq!(
                    (v.port_type(), v.to_display_string()),
                    want[i],
                    "round {round} cycle {cycle}: P3 `{o}`"
                );
            }
        }
    }
    // Every kind of extern, read through by-reference nodes.
    let mut p1 = polydat::dsl::compile::compile_polydat(EXTERNS).expect("interpreter");
    let mut p3 = compile_polydat_to_assembler(EXTERNS)
        .unwrap()
        .try_compile_pure_jit()
        .expect("pure native code with string, JSON, and extension externs");
    let read = ["id", "tag", "text", "code", "line"];
    for round in 0..3u64 {
        let (doc_n, region) = (round * 7, ["east", "west", "north"][round as usize]);
        for (name, value) in host_values(doc_n, region) {
            p1.set_input(name, value.clone()).expect("P1 set_input");
            p3.set_input(name, value).expect("P3 set_input");
        }
        for cycle in [0u64, 1, 1, 2, 2, 0] {
            p1.set_inputs(&[cycle]);
            let want: Vec<(polydat::ast::PortType, String)> = read
                .iter()
                .map(|o| {
                    let v = p1.pull(o);
                    (v.port_type(), v.to_display_string())
                })
                .collect();
            p3.eval(&[cycle]);
            for (i, o) in read.iter().enumerate() {
                let v = p3.get_value(o);
                assert_eq!(
                    (v.port_type(), v.to_display_string()),
                    want[i],
                    "round {round} cycle {cycle}: pure native `{o}`"
                );
            }
        }
    }
}
