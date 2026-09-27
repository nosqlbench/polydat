// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A name a comprehension reads that nothing binds (comprehension_forms.md
//! §5 V3, polydat_grammar.md §14). Under `pragma strict` in the scope the
//! name is read in, the statement is refused. Outside it the statement
//! compiles, the tree's ledger records a V3 warning naming each such name
//! and where it is read, and the name reads None: a source reading it
//! yields nothing and a predicate that reads it for a tuple keeps it not.
//! Traversals, producer bindings, and the streams of a producer's wire
//! agree, on every engine.

use polydat::dsl::compile::{CompileOptions, compile_polydat_with_engine};
use polydat::dsl::events::{CompileEvent, CompileEventLog};
use polydat::iteration::comprehension::strategies::TupleValue;
use polydat::iteration::comprehension::{ReadSite, ValidationError};
use polydat::kernel::{CompileLedger, UnresolvedNameWarning};
use polydat::{Engine, JitMode, Provenance};

fn engines() -> Vec<Engine> {
    let mut engines = vec![
        Engine::Interpreter(JitMode::Off),
        Engine::Closures(Provenance::Auto),
    ];
    if cfg!(feature = "jit") {
        engines.push(Engine::Native(Provenance::Auto));
        engines.push(Engine::PureNative(Provenance::PushPull));
    }
    engines
}

/// What compiling and traversing a program found.
struct Run {
    /// Each activation's `k`, for every traversal in order.
    traversals: Vec<Vec<u64>>,
    /// The V3 warnings on the tree's ledger.
    warnings: Vec<UnresolvedNameWarning>,
    /// The compile's warning events.
    events: Vec<String>,
}

/// Compile `src` on `engine` and open each of its traversals, or the
/// compile error.
fn run(src: &str, engine: Engine) -> Result<Run, String> {
    run_with(src, engine, false)
}

/// [`run`] with the host's `CompileOptions::strict` set to `host_strict`.
fn run_with(src: &str, engine: Engine, host_strict: bool) -> Result<Run, String> {
    let ledger = CompileLedger::new();
    let options = CompileOptions {
        ledger: Some(ledger.clone()),
        strict: host_strict,
        ..CompileOptions::default()
    };
    let mut log = CompileEventLog::new();
    let mut kernel = compile_polydat_with_engine(src, engine, &options, Some(&mut log))
        .map_err(|e| e.to_string())?;
    kernel.set_inputs(&[2]);
    let mut traversals = Vec::new();
    for stream in kernel.traverse_all()? {
        let mut stream = stream;
        let mut ks = Vec::new();
        while let Some(activation) = stream.advance()? {
            let k = activation
                .coords
                .iter()
                .find(|(n, _)| n == "k")
                .map(|(_, v)| v.as_u64())
                .expect("every tuple binds k");
            ks.push(k);
        }
        traversals.push(ks);
    }
    let events = log
        .events()
        .iter()
        .filter_map(|e| match e {
            CompileEvent::Warning { message } => Some(message.clone()),
            _ => None,
        })
        .collect();
    Ok(Run {
        traversals,
        warnings: ledger.unresolved_names(),
        events,
    })
}

fn lax(src: &str, engine: Engine) -> Run {
    run(src, engine).unwrap_or_else(|e| panic!("{engine}: {e}\n{src}"))
}

fn strict(src: &str, engine: Engine, name: &str) {
    let Err(err) = run(src, engine) else {
        panic!("{engine}: compiles under pragma strict\n{src}")
    };
    assert!(err.contains("V3:"), "{engine}: {err}");
    assert!(err.contains(&format!("`{name}`")), "{engine}: {err}");
}

/// Each statement reads `zz`, which nothing binds, in a predicate or a
/// source, and the traversal dispenses what reading None leaves.
const CASES: &[(&str, &[u64])] = &[
    // A predicate over None keeps no tuple.
    ("k in 1..6 where {k} > {zz}", &[]),
    // `||` stops at the operand that decides it, so `zz` is read only
    // for the tuples it does not decide.
    ("k in 1..6 where {k} == 1 || {k} > {zz}", &[1]),
    // A bare word is a name, which nothing binds.
    ("k in 1..4 where {k} != s1", &[]),
    // A source over None yields nothing, and so does the product.
    ("k in pow2({zz})", &[]),
    ("k in 1..4, j in pow2({zz})", &[]),
];

/// The name a case reads that nothing binds.
fn unbound(text: &str) -> &'static str {
    if text.contains("s1") { "s1" } else { "zz" }
}

/// Outside `pragma strict` each statement compiles, reads `zz` as None,
/// and dispenses the same activations on every engine.
#[test]
fn an_unbound_name_reads_none_outside_strict_on_every_engine() {
    for engine in engines() {
        for (text, expected) in CASES {
            let src = format!("input cycle: u64\nfor {text} {{\n    s := u64_add(k, 1)\n}}\n");
            let found = lax(&src, engine);
            assert_eq!(
                found.traversals,
                vec![expected.to_vec()],
                "{engine}: {text}"
            );
            assert_eq!(found.warnings.len(), 1, "{engine}: {text}");
            assert_eq!(
                found.events.len(),
                1,
                "{engine}: {text}: {:?}",
                found.events
            );
        }
    }
}

/// `pragma strict` refuses each statement, and `strict_values` alone does
/// not.
#[test]
fn pragma_strict_refuses_an_unbound_name_and_strict_values_does_not() {
    for engine in engines() {
        for (text, _) in CASES {
            let body = format!("input cycle: u64\nfor {text} {{\n    s := u64_add(k, 1)\n}}\n");
            strict(&format!("pragma strict\n{body}"), engine, unbound(text));
            let found = lax(&format!("pragma strict_values\n{body}"), engine);
            assert_eq!(found.warnings.len(), 1, "{engine}: {text}");
        }
    }
}

/// The ledger's warning names the statement, where it is, each unbound
/// name, and where the comprehension reads it; the event carries the same
/// text.
#[test]
fn the_ledger_warning_names_each_unbound_name_and_its_read_site() {
    let src = "input cycle: u64\n\
               for k in 1..4, j in pow2({zz}) where {k} > {limit} && {k} != s1 {\n    \
               s := u64_add(k, 1)\n}\n";
    for engine in engines() {
        let found = lax(src, engine);
        let [warning] = found.warnings.as_slice() else {
            panic!("{engine}: {:?}", found.warnings)
        };
        assert_eq!(
            warning.statement,
            "for k in 1..4, j in pow2({zz}) where {k} > {limit} && {k} != s1"
        );
        assert_eq!((warning.line, warning.col), (2, 1));
        let reads: Vec<(&str, bool)> = warning
            .reads
            .iter()
            .map(|r| (r.name.as_str(), r.bare))
            .collect();
        assert_eq!(reads, [("zz", false), ("limit", false), ("s1", true)]);
        assert!(matches!(
            &warning.reads[0].site,
            ReadSite::Source { clause, .. } if clause == "j"
        ));
        assert!(matches!(&warning.reads[1].site, ReadSite::Predicate { .. }));
        let text = warning.to_string();
        for part in [
            "`for k in 1..4, j in pow2({zz}) where {k} > {limit} && {k} != s1` at line 2, col 1",
            "V3: `zz` read by clause 'j' in `pow2({zz})`",
            "`limit` read by predicate `{k} > {limit} && {k} != s1`",
            "so each reads None",
            "as in `\"s1\"`",
        ] {
            assert!(text.contains(part), "{engine}: {part} not in {text}");
        }
        assert_eq!(found.events, vec![text], "{engine}");
    }
}

/// A producer binding reads `zz` as a traversal does: its traversal and
/// the stream of its wire dispense what reading None leaves, and a name
/// the scope has stays a name the stream refuses to read.
#[test]
fn a_producer_and_its_stream_read_an_unbound_name_as_none() {
    for engine in engines() {
        for (text, expected) in CASES {
            let src = format!(
                "input cycle: u64\nsweep := for {text}\nfor sweep {{\n    s := u64_add(k, 1)\n}}\n"
            );
            let found = lax(&src, engine);
            assert_eq!(
                found.traversals,
                vec![expected.to_vec()],
                "{engine}: {text}"
            );
            // The producer binding and the traversal over it each read it.
            assert_eq!(found.warnings.len(), 2, "{engine}: {text}");
            strict(&format!("pragma strict\n{src}"), engine, unbound(text));
        }
    }
    for (text, expected) in CASES {
        let mut kernel = polydat::dsl::compile_polydat_interpreter(&format!(
            "input cycle: u64\nsweep := for {text}\n"
        ))
        .unwrap_or_else(|e| panic!("{text}: {e}"));
        let sweep = kernel.pull_ref("sweep").clone();
        let streamed: Vec<u64> = sweep
            .as_streamer()
            .unwrap()
            .coordinate_stream()
            .unwrap_or_else(|e| panic!("{text}: {e}"))
            .map(|t| {
                let t = t.unwrap();
                match t.bindings.iter().find(|(n, _)| n == "k").unwrap().1 {
                    TupleValue::I64(n) => n as u64,
                    TupleValue::U64(n) => n,
                    ref other => panic!("{text}: k is {other:?}"),
                }
            })
            .collect();
        assert_eq!(streamed, expected.to_vec(), "{text}");
    }
    let mut kernel = polydat::dsl::compile_polydat_interpreter(
        "input cycle: u64\nsweep := for k in 1..6 where {k} > {cycle} || {k} > {zz}\n",
    )
    .unwrap();
    let sweep = kernel.pull_ref("sweep").clone();
    let Err(err) = sweep.as_streamer().unwrap().coordinate_stream() else {
        panic!("a predicate over a scope's name streams")
    };
    assert!(
        matches!(err, ValidationError::PredicateContextRequired { ref references, .. }
            if references == &["cycle".to_string()]),
        "{err}"
    );
}

/// Strictness is lexical: a `for` body's `pragma strict` refuses a name
/// read in the body and leaves the enclosing scope lax, and a program's
/// reaches its bodies.
#[test]
fn a_for_body_is_checked_under_its_scope() {
    let nested = |outer: &str, inner: &str| {
        format!(
            "{outer}input cycle: u64\n\
             for k in 1..3 where {{k}} > 0 || {{k}} > {{zz}} {{\n    \
             {inner}for j in pow2({{zz}}) {{\n        t := j\n    }}\n    \
             s := k\n}}\n"
        )
    };
    for engine in engines() {
        strict(&nested("", "pragma strict\n    "), engine, "zz");
        strict(&nested("pragma strict\n", ""), engine, "zz");
        let found = lax(&nested("", ""), engine);
        assert_eq!(found.traversals, vec![vec![1, 2]], "{engine}");
        assert_eq!(found.warnings.len(), 2, "{engine}: {:?}", found.warnings);
    }
}

/// A module body is checked under its own pragmas: a strict module in a
/// lax host refuses its producer's unbound name, and a lax module in a
/// strict host warns about it.
#[test]
fn a_module_is_checked_under_its_own_pragmas() {
    let program = |host: &str, module: &str| {
        format!(
            "{host}input cycle: u64\n\
             m(a: u64) -> (out: u64) := {{\n\
                 {module}p := for k in 1..4 where {{k}} > {{zz}}\n\
                 out := a\n\
             }}\n\
             s := m(cycle)\n"
        )
    };
    for engine in engines() {
        strict(&program("", "pragma strict\n"), engine, "zz");
        let found = lax(&program("pragma strict\n", ""), engine);
        assert_eq!(found.warnings.len(), 1, "{engine}: {:?}", found.warnings);
        assert!(
            found.warnings[0].statement.starts_with("p := for"),
            "{engine}: {}",
            found.warnings[0]
        );
    }
}

/// The host's `CompileOptions::strict` is `pragma strict` at the
/// program's top scope: an unbound name that compiles with a warning
/// without it is refused with it, in the program and in a `for` body, and
/// a module without pragmas stays lax under it.
#[test]
fn host_strict_is_pragma_strict_at_the_top_scope() {
    let refused = |src: &str, engine: Engine| {
        let Err(err) = run_with(src, engine, true) else {
            panic!("{engine}: compiles under CompileOptions::strict\n{src}")
        };
        assert!(
            err.contains("V3:") && err.contains("`zz`"),
            "{engine}: {err}"
        );
    };
    let top = "input cycle: u64\nfor k in 1..4 where {k} > {zz} {\n    s := u64_add(k, 1)\n}\n";
    let body = "input cycle: u64\n\
                for k in 1..3 {\n    \
                for j in pow2({zz}) {\n        t := j\n    }\n    \
                s := k\n}\n";
    let module = "input cycle: u64\n\
                  m(a: u64) -> (out: u64) := {\n\
                      p := for k in 1..4 where {k} > {zz}\n\
                      out := a\n\
                  }\n\
                  s := m(a: cycle)\n";
    for engine in engines() {
        for src in [top, body] {
            let found = lax(src, engine);
            assert_eq!(found.warnings.len(), 1, "{engine}: {:?}", found.warnings);
            refused(src, engine);
        }
        let found = run_with(module, engine, true).unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert_eq!(found.warnings.len(), 1, "{engine}: {:?}", found.warnings);
    }
}

/// A value bound to None reads as a name nothing binds does: an extern
/// with no default is None until the host sets it.
#[test]
fn a_scope_name_bound_to_none_reads_none() {
    let src = "input cycle: u64\nextern limit: u64\n\
               for k in 1..6 where {k} == 1 || {k} > {limit} {\n    s := u64_add(k, 1)\n}\n";
    let found = lax(src, Engine::Interpreter(JitMode::Off));
    assert!(found.warnings.is_empty());
    assert_eq!(found.traversals, vec![vec![1]]);
}
