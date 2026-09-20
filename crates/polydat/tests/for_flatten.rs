// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Context-free generators are flattened at compile
//! (comprehension_forms.md §10.7.0, §10.7.7): a call that references
//! no name is evaluated once by the compile and traversed as the
//! literal of its values, so its cardinality is exact and an order
//! over it validates; a call that references a name is left to the
//! traversal.

use polydat::dsl::compile_polydat_interpreter;
use polydat::iteration::comprehension::cardinality::CardinalityClass;

fn traversed(src: &str) -> Vec<u64> {
    let mut k = compile_polydat_interpreter(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    k.set_inputs(&[0]);
    let mut stream = k.traverse(0).unwrap();
    let mut seen = Vec::new();
    while let Some(mut a) = stream.advance().unwrap() {
        seen.push(a.cycle(0).pull_ref("v").as_u64());
    }
    seen
}

/// An order over a context-free generator validates and samples: the
/// call is a literal by the time V6 looks at it.
#[test]
fn an_order_over_a_context_free_generator_validates_and_samples() {
    let seen = traversed(
        "input cycle: u64\nfor k in fib(8) order halton/4 {\n    v := u64_add(k, 0)\n}\n",
    );
    assert_eq!(seen.len(), 4);
    assert!(
        seen.iter().all(|v| [1, 2, 3, 5, 8, 13, 21].contains(v)),
        "{seen:?}"
    );
    let seen = traversed(
        "input cycle: u64\nfor k in pow2(5) order extrema/1 {\n    v := u64_add(k, 0)\n}\n",
    );
    assert_eq!(seen, vec![1, 16]);
}

/// A producer over a context-free generator has an exact cardinality,
/// and its stream is the generator's values.
#[test]
fn a_producer_over_a_context_free_generator_is_bounded_by_its_values() {
    let mut k = compile_polydat_interpreter("input cycle: u64\nfibs := for k in fib(8)\n").unwrap();
    k.set_inputs(&[0]);
    let streamer = k.pull_ref("fibs").clone();
    let streamer = streamer.as_streamer().unwrap();
    assert!(
        matches!(streamer.cardinality(), CardinalityClass::Bounded(8)),
        "{:?}",
        streamer.cardinality()
    );
    let mut stream = streamer.coordinate_stream().unwrap();
    let mut n = 0;
    while stream.advance().is_some() {
        n += 1;
    }
    assert_eq!(n, 8);
}

/// A generator that references a name the compile cannot bind is left
/// to the traversal, so an order over it is still the validator's to
/// refuse: unbounded and without a closed-form index function (V4).
#[test]
fn a_generator_over_a_runtime_name_stays_unbounded_at_compile() {
    let err = compile_polydat_interpreter(
        "input cycle: u64\ninput n: u64\nfor k in pow2({n}) order halton/2 {\n    v := u64_add(k, 0)\n}\n",
    )
    .unwrap_err();
    assert!(err.to_string().contains("V4"), "{err}");
    // Without an order the traversal evaluates it with the wire bound.
    let mut k = compile_polydat_interpreter(
        "input cycle: u64\nextern total: u64 = 100\nfor p in partitions(\"*/2\", {total}) {\n    v := cardinality(p)\n}\n",
    )
    .unwrap();
    k.set_inputs(&[0]);
    let mut stream = k.traverse(0).unwrap();
    let mut seen = Vec::new();
    while let Some(mut a) = stream.advance().unwrap() {
        seen.push(a.cycle(0).pull_ref("v").as_u64());
    }
    assert_eq!(seen, vec![50, 50]);
}

/// A traversal source is its comprehension: the text is rendered from
/// the tree rather than stored beside it (comprehension_forms.md §8), so
/// what a program projects is what it traverses by construction, and a
/// source whose halves disagree is not a thing that can be built.
#[test]
fn a_built_source_projects_the_comprehension_it_traverses() {
    use polydat::dsl::ast::{
        Binding, BindingModifier, Expr, ForSource, InputDecl, PolydatFile, Statement,
    };
    use polydat::dsl::compile::{CompileOptions, compile_ast_interpreter_with_options};
    use polydat::dsl::lexer::Span;
    use polydat::iteration::comprehension::ast::Comprehension;
    use polydat::iteration::comprehension::source::Source;

    let sp = Span { line: 0, col: 0 };
    let tree = Comprehension::cartesian(vec![
        Comprehension::clause(
            "k",
            Source::IntRange {
                lo: 1,
                hi: 4,
                step: 1,
            },
        ),
        Comprehension::clause(
            "limit",
            Source::IntRange {
                lo: 10,
                hi: 12,
                step: 1,
            },
        ),
    ]);
    let built = ForSource::comprehension(tree.clone(), sp).expect("the text writes this tree");
    assert_eq!(built.to_text(), "k in 1..4, limit in 10..12");

    let file = |source: ForSource| PolydatFile {
        statements: vec![
            Statement::InputDecl(InputDecl {
                name: "cycle".into(),
                ty: Some("u64".into()),
                span: sp,
            }),
            Statement::Binding(Binding {
                targets: vec!["sweep".into()],
                value: Expr::For(Box::new(source)),
                modifier: BindingModifier::NONE,
                type_annotation: None,
                span: sp,
            }),
        ],
    };

    // What the program projects is what it traverses.
    let projected = polydat::dsl::pprint::pp_file(&file(built.clone()));
    assert!(
        projected.contains("sweep := for k in 1..4, limit in 10..12"),
        "{projected}"
    );
    let mut k =
        compile_ast_interpreter_with_options(&file(built), "", &CompileOptions::default(), None)
            .unwrap();
    k.set_inputs(&[0]);
    let streamer = k.pull_ref("sweep").clone();
    assert!(
        matches!(
            streamer.as_streamer().unwrap().cardinality(),
            CardinalityClass::Bounded(6)
        ),
        "{:?}",
        streamer.as_streamer().unwrap().cardinality()
    );

    // A source's text is rendered from its comprehension, so a text
    // that disagrees with the tree is not a thing that can be built:
    // there is no text to set. The forged source this test used to
    // construct no longer compiles, and the property it was checking
    // holds by construction instead.
    let built = ForSource::comprehension(tree.clone(), sp).expect("writable tree");
    assert_eq!(
        built.to_text(),
        tree.to_text().expect("writable tree"),
        "a source's text is its comprehension's text"
    );
    // And the pair that makes that trustworthy: written and read back,
    // the comprehension is the same comprehension.
    let reread =
        polydat::iteration::comprehension::spec::parse_comprehension_algebra(&built.to_text())
            .expect("its own text parses");
    assert_eq!(reread, tree, "writing and reading back is the identity");
}

/// A tile built from its body text carries pieces parsed from that
/// text, so the body it projects is the body it renders (polytile.md
/// §3): the built tile behaves as the written one.
#[test]
fn a_built_tile_projects_the_body_it_renders() {
    use polydat::dsl::ast::{
        InputDecl, PolydatFile, Statement, TileBodyKind, TileDef, TileOptions,
    };
    use polydat::dsl::compile::{CompileOptions, compile_ast_interpreter_with_options};
    use polydat::dsl::lexer::Span;

    let sp = Span { line: 0, col: 0 };
    let tile = TileDef::from_body(
        "t",
        None,
        TileOptions::default(),
        TileBodyKind::Literal,
        "n=${u64_add(cycle, 1)}",
        sp,
    )
    .expect("the body parses");
    let file = PolydatFile {
        statements: vec![
            Statement::InputDecl(InputDecl {
                name: "cycle".into(),
                ty: Some("u64".into()),
                span: sp,
            }),
            Statement::Tile(tile),
        ],
    };
    let projected = polydat::dsl::pprint::pp_file(&file);
    assert!(
        projected.contains("tile t := \"n=${u64_add(cycle, 1)}\""),
        "{projected}"
    );
    let mut k =
        compile_ast_interpreter_with_options(&file, "", &CompileOptions::default(), None).unwrap();
    k.set_inputs(&[41]);
    assert_eq!(k.pull_ref("t").as_str(), "n=42");
    // The same text written as source gives the same program.
    let mut written =
        compile_polydat_interpreter("input cycle: u64\ntile t := \"n=${u64_add(cycle, 1)}\"\n")
            .unwrap();
    written.set_inputs(&[41]);
    assert_eq!(written.pull_ref("t").as_str(), "n=42");
}
