// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A source that reads None yields nothing, decided on the names it
//! reads after composition (comprehension_forms.md §5 V3, §10.9.1;
//! none_semantics.md Rule 1).
//!
//! A host evaluates a comprehension against a kernel of any engine. A
//! composed name (`{k_{k}_limits}`) reads the name its leaves compose to
//! with the earlier axis bound, and `all(<cursor>)` reads the cursor's
//! extent: both yield their values, V3 finds no unbound name in them,
//! and a strict name check accepts them. A composed name whose target
//! nothing binds, or whose target is bound to None, yields nothing for
//! that tuple, and the clause's yield names the target and how it read
//! None. The two traversal evaluators agree on every case, and a stream
//! refuses the forms that need a scope.

use polydat::ast::Value;
use polydat::iteration::comprehension::runtime::{
    EvaluatedIteration, NoneReads, evaluate_for_iteration_materialized_with_none_reads,
    evaluate_for_iteration_reported, evaluate_for_iteration_with_none_reads, evaluate_indexed,
};
use polydat::iteration::comprehension::spec::parse_comprehension_algebra;
use polydat::iteration::comprehension::surfaces::CompiledComprehension;
use polydat::iteration::comprehension::{
    ClauseYield, Comprehension, Mode, NoneRead, Surface, ValidationError, ValidationWarning,
    check_names,
};
use polydat::kernel::interp::{KernelLookup, Layered, Lookup};
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

fn kernel(src: &str, engine: Engine) -> Box<dyn polydat::Kernel> {
    let options = polydat::dsl::compile::CompileOptions::default();
    polydat::dsl::compile::compile_polydat_with_engine(src, engine, &options, None)
        .unwrap_or_else(|e| panic!("{engine}: {e}\n{src}"))
}

fn parse(text: &str) -> Comprehension {
    parse_comprehension_algebra(text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

/// Each tuple rendered `name=value`, in binding order.
fn rows(evaluated: &EvaluatedIteration) -> Vec<Vec<String>> {
    evaluated
        .tuples
        .iter()
        .map(|t| {
            t.iter()
                .map(|(n, v)| format!("{n}={}", v.to_display_string()))
                .collect()
        })
        .collect()
}

/// What a traversal yielded, and each clause's None reads beside it.
struct Traversed {
    evaluated: EvaluatedIteration,
    none_reads: NoneReads,
}

impl Traversed {
    fn clause(&self, var: &str) -> &ClauseYield {
        &self.evaluated.clauses[self.index(var)]
    }

    fn reads_none(&self, var: &str) -> &[NoneRead] {
        self.none_reads.clause(self.index(var))
    }

    fn index(&self, var: &str) -> usize {
        self.evaluated
            .clauses
            .iter()
            .position(|c| c.var == var)
            .unwrap_or_else(|| panic!("no clause {var}"))
    }
}

/// Evaluate `ast` in `scope` with both traversal evaluators, held to
/// each other, and report what each clause yielded and read as None.
fn traverse(ast: &Comprehension, scope: &dyn Lookup, what: &str) -> Traversed {
    let (reported, none_reads) = evaluate_for_iteration_with_none_reads(ast, scope)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let (materialized, materialized_none) =
        evaluate_for_iteration_materialized_with_none_reads(ast, scope)
            .unwrap_or_else(|e| panic!("{what}: {e}"));
    assert_eq!(reported.tuples, materialized.tuples, "{what}");
    assert_eq!(reported.clauses, materialized.clauses, "{what}");
    assert_eq!(none_reads, materialized_none, "{what}");
    let plain =
        evaluate_for_iteration_reported(ast, scope).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert_eq!(plain.clauses, reported.clauses, "{what}");
    let indexed = evaluate_indexed(ast, scope).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert_eq!(indexed.to_vec(), reported.tuples, "{what}");
    Traversed {
        evaluated: reported,
        none_reads,
    }
}

/// Whether the kernel's scope has `name`.
fn has(scope: &dyn Lookup) -> impl Fn(&str) -> bool + '_ {
    move |name| scope.lookup(name).is_some()
}

/// `all(<cursor>)` over a literal range enumerates the cursor's extent on
/// every engine; V3 finds no unbound name, strict or not. Over a scope
/// with no such cursor it reads the extent as unbound: it yields nothing,
/// its yield names the extent, and V3 names the cursor.
#[test]
fn all_over_a_cursor_reads_its_extent() {
    let ast = parse("xval in all(row)");
    let expected: Vec<Vec<String>> = (0..50).map(|n| vec![format!("xval={n}")]).collect();
    for engine in engines() {
        let k = kernel("input cycle: u64\ncursor row = range(0, 50)\n", engine);
        let scope = KernelLookup::new(k.as_ref());
        let found = traverse(&ast, &scope, &format!("{engine}"));
        assert_eq!(rows(&found.evaluated), expected, "{engine}");
        assert!(found.reads_none("xval").is_empty(), "{engine}");
        check_names(&ast, Surface::Traversal(&has(&scope)))
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        // A stream has no scope to read the extent in.
        let err = CompiledComprehension::from_ast_in(&ast, Mode::Strict, &has(&scope))
            .err()
            .unwrap_or_else(|| panic!("{engine}: a stream reads a cursor's extent"));
        assert!(
            matches!(err, ValidationError::ContextRequired { .. }),
            "{engine}: {err}"
        );

        let bare = kernel("input cycle: u64\n", engine);
        let scope = KernelLookup::new(bare.as_ref());
        let found = traverse(&ast, &scope, &format!("{engine}: no cursor"));
        assert!(found.evaluated.tuples.is_empty(), "{engine}");
        assert_eq!(
            found.reads_none("xval"),
            [NoneRead::Unbound("__cursor_extent_row_start".into())],
            "{engine}"
        );
        let err = check_names(&ast, Surface::Traversal(&has(&scope))).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("V3: the extent of cursor `row` read by clause 'xval'")
                && text.contains(" is bound neither"),
            "{engine}: {text}"
        );
    }
    // The scope-less stream reads the extent as None and yields nothing.
    let (compiled, report) = CompiledComprehension::from_ast_with(&ast, Mode::Permissive)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(
        report.warnings.first(),
        Some(ValidationWarning::UnresolvedNames { .. })
    ));
    assert_eq!(compiled.coordinate_stream().count(), 0);
}

const LIMITS: &str = "input cycle: u64\nconst k_values := \"1, 10\"\n\
                      const k_1_limits := \"1, 2\"\nconst k_10_limits := \"10, 20, 30\"\n";

/// A dependent clause's composed name reads the name its leaves compose
/// to with the earlier axis bound, in the braced and the bare-prior
/// forms, on every engine. The composed name reads its leaf, which the
/// earlier axis binds, so V3 finds no unbound name; a stream refuses the
/// sources that read the scope or an earlier axis, which only a
/// traversal evaluates.
#[test]
fn a_composed_source_name_reads_its_composed_target() {
    let expected: Vec<Vec<String>> = [(1, 1), (1, 2), (10, 10), (10, 20), (10, 30)]
        .iter()
        .map(|(k, l)| vec![format!("k={k}"), format!("limit={l}")])
        .collect();
    for engine in engines() {
        let k = kernel(LIMITS, engine);
        let scope = KernelLookup::new(k.as_ref());
        for text in [
            "k in {k_values}, limit in {k_{k}_limits}",
            "k in k_values, limit in {k_{k}_limits}",
        ] {
            let ast = parse(text);
            let found = traverse(&ast, &scope, &format!("{engine}: {text}"));
            assert_eq!(rows(&found.evaluated), expected, "{engine}: {text}");
            assert!(
                found.none_reads.is_empty(),
                "{engine}: {text}: {:?}",
                found.none_reads
            );
            check_names(&ast, Surface::Traversal(&has(&scope)))
                .unwrap_or_else(|e| panic!("{engine}: {text}: {e}"));
            let err = CompiledComprehension::from_ast_in(&ast, Mode::Strict, &has(&scope))
                .err()
                .unwrap_or_else(|| panic!("{engine}: {text}: a stream reads an earlier axis"));
            assert!(
                matches!(err, ValidationError::ContextRequired { .. }),
                "{engine}: {text}: {err}"
            );
        }
    }
}

/// A composed name whose target nothing binds yields nothing for that
/// tuple, and one whose target is bound to None does too; the clause's
/// yield names each target read and how it read None. Which target a
/// composition reads is known only once its leaves are bound, so V3 at
/// compile checks the leaves, and a strict check accepts the source.
#[test]
fn a_composed_target_that_reads_none_yields_nothing_and_is_reported() {
    let text = "k in {k_values}, limit in {k_{k}_limits}";
    let ast = parse(text);
    let src = "input cycle: u64\nconst k_values := \"1, 7, 9\"\nconst k_1_limits := \"1, 2\"\n";
    for engine in engines() {
        let k = kernel(src, engine);
        let kernel_scope = KernelLookup::new(k.as_ref());
        let prefix = [("k_9_limits".to_string(), Value::None)];
        let scope = Layered {
            prefix: &prefix,
            inner: &kernel_scope,
        };
        let found = traverse(&ast, &scope, &format!("{engine}"));
        assert_eq!(
            rows(&found.evaluated),
            [["k=1", "limit=1"], ["k=1", "limit=2"]],
            "{engine}"
        );
        let limit = found.clause("limit");
        assert_eq!((limit.evaluations, limit.values), (3, 2), "{engine}");
        assert_eq!(
            found.reads_none("limit"),
            [
                NoneRead::Unbound("k_7_limits".into()),
                NoneRead::BoundNone("k_9_limits".into()),
            ],
            "{engine}"
        );
        assert!(found.reads_none("k").is_empty(), "{engine}");
        check_names(&ast, Surface::Traversal(&has(&scope)))
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
    }
    // A leaf nothing binds is a name V3 knows at compile: it is refused
    // under strictness and read as None otherwise.
    let unbound_leaf = parse("limit in {k_{zz}_limits}");
    let err = check_names(&unbound_leaf, Surface::Stream).unwrap_err();
    assert!(
        matches!(&err, ValidationError::V3UnresolvedNames { reads }
            if reads.len() == 1 && reads[0].name == "zz"),
        "{err}"
    );
    let k = kernel(LIMITS, Engine::Interpreter(JitMode::Off));
    let scope = KernelLookup::new(k.as_ref());
    let found = traverse(&unbound_leaf, &scope, "unbound leaf");
    assert!(found.evaluated.tuples.is_empty());
    assert_eq!(found.reads_none("limit"), [NoneRead::Unbound("zz".into())]);
}
