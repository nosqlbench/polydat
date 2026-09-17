// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 113 step 1: the `for` construct parses in both readings and
//! round trips through the pretty-printer. Step 2 lowers both forms.

use polydat::dsl::ast::{Expr, ForSourceKind, PolydatFile, Statement};
use polydat::dsl::pprint::pp_file;

fn parse(src: &str) -> PolydatFile {
    let tokens = polydat::dsl::lexer::lex(src).unwrap_or_else(|e| panic!("lex: {e}"));
    polydat::dsl::parser::parse(tokens).unwrap_or_else(|e| panic!("parse: {e}"))
}

fn parse_err(src: &str) -> String {
    let tokens = match polydat::dsl::lexer::lex(src) {
        Ok(t) => t,
        Err(e) => return e,
    };
    polydat::dsl::parser::parse(tokens).expect_err("expected a parse error")
}

#[test]
fn producer_binding_parses_to_a_comprehension() {
    let f = parse("sweep := for k in 1..4, limit in 10,20,30 order halton/5");
    let Statement::Binding(b) = &f.statements[0] else {
        panic!("expected binding")
    };
    assert_eq!(b.targets, vec!["sweep"]);
    let Expr::For(source) = &b.value else {
        panic!("expected for expression, got {:?}", b.value)
    };
    assert_eq!(source.text, "k in 1..4, limit in 10,20,30 order halton/5");
    assert!(matches!(source.kind, ForSourceKind::Comprehension(_)));
    assert_eq!(source.element_names(), vec!["k", "limit"]);
}

#[test]
fn traversal_over_inline_comprehension_with_body() {
    let src = "for phase in load,verify, p in partitions(\"*/4\", 1000000) {\n    row := mod_in(cycle, p)\n    stmt := hash(row)\n}\n";
    let f = parse(src);
    let Statement::For(fs) = &f.statements[0] else {
        panic!("expected for statement")
    };
    assert_eq!(
        fs.source.text,
        "phase in load,verify, p in partitions(\"*/4\", 1000000)"
    );
    assert_eq!(fs.source.element_names(), vec!["phase", "p"]);
    assert_eq!(fs.body.len(), 2);
    assert!(matches!(&fs.body[0], Statement::Binding(b) if b.targets == vec!["row"]));
}

#[test]
fn traversal_over_bound_producer() {
    let f = parse("sweep := for k in 1..4\nfor sweep {\n    f := hash(k)\n}\n");
    let Statement::For(fs) = &f.statements[1] else {
        panic!("expected for statement")
    };
    assert!(matches!(&fs.source.kind, ForSourceKind::Producer(name) if name == "sweep"));
    assert_eq!(fs.body.len(), 1);
}

#[test]
fn nested_traversals_and_every_statement_kind_in_a_body() {
    let src = "for p in partitions(\"*/2\", 100) {\n    extern base: u64 = 5\n    const label := \"x\"\n    cursor rows = range(0, 100) over p\n    input cycle: u64\n    for k in 1..3 {\n        v := hash(k)\n        inner := for j in 1..2\n    }\n}\n";
    let f = parse(src);
    let Statement::For(outer) = &f.statements[0] else {
        panic!()
    };
    assert_eq!(outer.body.len(), 5);
    let Statement::For(inner) = &outer.body[4] else {
        panic!("expected nested for")
    };
    assert_eq!(inner.body.len(), 2);
    assert!(matches!(&inner.body[1], Statement::Binding(b) if matches!(b.value, Expr::For(_))));
}

#[test]
fn pretty_printer_round_trips_both_forms() {
    let src = "sweep := for k in 1..4, limit in 10,20,30 order halton/5\nfor sweep {\n    f := myfunc(k)\n    g := otherfunc(limit, k)\n}\nfor phase in load,verify, p in partitions(\"*/4\", 1000000) {\n    row := mod_in(cycle, p)\n    for q in 1..2 {\n        z := hash(q)\n    }\n}\n";
    let printed = pp_file(&parse(src));
    assert_eq!(printed, src);
    // And the printed form parses back to the same shape.
    let again = pp_file(&parse(&printed));
    assert_eq!(again, printed);
}

#[test]
fn where_and_order_survive_in_the_captured_text() {
    let f = parse(
        "for k in 1..4, limit in 10,20,30 where {k} >= 2 && {limit} != 20 order extrema/1 {\n    x := hash(k)\n}\n",
    );
    let Statement::For(fs) = &f.statements[0] else {
        panic!()
    };
    assert!(fs.source.text.ends_with("order extrema/1"));
    assert!(fs.source.text.contains("where {k} >= 2"));
}

#[test]
fn comments_after_the_comprehension_are_not_captured() {
    let f = parse("sweep := for k in 1..4 // the sweep\nx := hash(cycle)\n");
    let Statement::Binding(b) = &f.statements[0] else {
        panic!()
    };
    let Expr::For(source) = &b.value else {
        panic!()
    };
    assert_eq!(source.text, "k in 1..4");
    assert_eq!(f.statements.len(), 2);
}

#[test]
fn statement_form_requires_a_block_on_the_same_line() {
    let err = parse_err("for k in 1..4\nx := hash(k)\n");
    assert!(err.contains("needs a `{` block"), "{err}");
}

#[test]
fn producer_expression_rejects_a_bare_identifier() {
    let err = parse_err("x := for sweep\n");
    assert!(err.contains("comprehension text"), "{err}");
}

#[test]
fn bad_comprehension_text_reports_position() {
    let err = parse_err("for k 1..4 {\n    x := hash(k)\n}\n");
    assert!(err.contains("line 1"), "{err}");
}

#[test]
fn compiler_lowers_both_forms() {
    // Step 2: a traversal body compiles to a child program and a
    // producer binding becomes program metadata.
    let k =
        polydat::dsl::compile_polydat("input cycle: u64\nfor k in 1..4 {\n    x := hash(k)\n}\n")
            .unwrap();
    assert_eq!(k.program().traversals().len(), 1);
    let k = polydat::dsl::compile_polydat("input cycle: u64\nsweep := for k in 1..4\n").unwrap();
    assert_eq!(k.program().producers().len(), 1);
}

/// A bracketed union runs across lines, and the positions of what
/// follows it stay right.
#[test]
fn a_bracketed_union_keeps_line_numbers_after_it() {
    let err = parse_err(
        "both := for [\n    for k in 1..3,\n    for k in 10..12,\n]\nfor k 1..4 {\n    x := hash(k)\n}\n",
    );
    assert!(err.contains("line 5"), "{err}");
}
