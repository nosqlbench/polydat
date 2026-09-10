// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 114 step 1: the `tile` statement parses in every body form, the
//! template grammar yields the expected pieces under default and custom
//! delimiters, the printer round trips, and the compiler declines with a
//! pointer to the SRD until step 4 lands.

use polydat::dsl::ast::{Expr, PolydatFile, Statement, TileBodyKind, TileDef, TilePiece};
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

fn tile(src: &str) -> TileDef {
    let f = parse(src);
    match &f.statements[0] {
        Statement::Tile(t) => t.clone(),
        other => panic!("expected tile, got {other:?}"),
    }
}

fn shape(pieces: &[TilePiece]) -> String {
    pieces
        .iter()
        .map(|p| match p {
            TilePiece::Static(s) => format!("S({s:?})"),
            TilePiece::Hole(h) => format!(
                "H({}{}{}{})",
                h.text,
                h.decl_type
                    .as_ref()
                    .map(|t| format!(" :{t}"))
                    .unwrap_or_default(),
                h.format
                    .as_ref()
                    .map(|f| format!(" |{f}"))
                    .unwrap_or_default(),
                if h.raw { " !" } else { "" }
            ),
            TilePiece::Projection {
                source, sep, body, ..
            } => format!(
                "P({}{} [{}])",
                source.text,
                sep.as_ref()
                    .map(|s| format!(" sep {s:?}"))
                    .unwrap_or_default(),
                shape(body)
            ),
            TilePiece::Branch {
                then, otherwise, ..
            } => format!(
                "B([{}]{})",
                shape(then),
                otherwise
                    .as_ref()
                    .map(|o| format!(" else [{}]", shape(o)))
                    .unwrap_or_default()
            ),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn json_block_body_with_holes_and_a_static_arm() {
    let t = tile(
        "tile doc : json := {\n    \"meta\": { \"schema\": 3, \"units\": { \"temp\": \"C\" } },\n    \"tenant\": ${tenant_id},\n    \"device\": \"${device_id}\"\n}\n",
    );
    assert_eq!(t.name, "doc");
    assert_eq!(t.encoding.as_deref(), Some("json"));
    assert_eq!(t.body_kind, TileBodyKind::Block);
    assert!(t.body.starts_with('{') && t.body.ends_with('}'));
    let holes: Vec<&str> = t
        .pieces
        .iter()
        .filter_map(|p| {
            if let TilePiece::Hole(h) = p {
                Some(h.text.as_str())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(holes, vec!["tenant_id", "device_id"]);
    // The static arm before the first hole is one run.
    assert!(
        matches!(&t.pieces[0], TilePiece::Static(s) if s.contains("\"units\": { \"temp\": \"C\" }"))
    );
}

#[test]
fn heredoc_body_trims_one_newline_each_side() {
    let t = tile(
        "tile load : text := <<<\nINSERT INTO ${keyspace}.${table} VALUES (${tenant_id}, '${doc!}')\n>>>\nx := hash(cycle)\n",
    );
    assert_eq!(t.body_kind, TileBodyKind::Heredoc);
    assert_eq!(
        t.body,
        "INSERT INTO ${keyspace}.${table} VALUES (${tenant_id}, '${doc!}')"
    );
    assert_eq!(
        shape(&t.pieces),
        "S(\"INSERT INTO \") H(keyspace) S(\".\") H(table) S(\" VALUES (\") H(tenant_id) S(\", '\") H(doc! !) S(\"')\")"
    );
    // The statement after the heredoc still parses.
    assert_eq!(
        parse("tile load : text := <<<\nINSERT\n>>>\nx := hash(cycle)\n")
            .statements
            .len(),
        2
    );
}

#[test]
fn string_literal_body_and_csv_encoding() {
    let t = tile("tile row : csv := \"${tenant_id},${device_id},${ts}\"\n");
    assert_eq!(t.body_kind, TileBodyKind::Literal);
    assert_eq!(
        shape(&t.pieces),
        "H(tenant_id) S(\",\") H(device_id) S(\",\") H(ts)"
    );
}

#[test]
fn hole_modifiers_type_format_and_raw() {
    let t =
        tile("tile t := \"${ts: u64} ${temp_c + s | .2} ${doc!} ${a | 05} ${b: str | >8 !}\"\n");
    assert_eq!(
        shape(&t.pieces),
        "H(ts: u64 :u64) S(\" \") H(temp_c + s | .2 |.2) S(\" \") H(doc! !) S(\" \") H(a | 05 |05) S(\" \") H(b: str | >8 ! :str |>8 !)"
    );
    let TilePiece::Hole(h) = &t.pieces[2] else {
        panic!()
    };
    assert!(matches!(h.expr, Expr::BinOp(..)));
    // Named arguments inside the expression are not type declarations.
    let t = tile("tile u := \"${hashed_id(input: cycle, bound: 10)}\"\n");
    let TilePiece::Hole(h) = &t.pieces[0] else {
        panic!()
    };
    assert!(h.decl_type.is_none());
    assert!(matches!(h.expr, Expr::Call(_)));
}

#[test]
fn projections_with_default_and_explicit_separators_and_nesting() {
    let t = tile(
        "tile doc : json := {\"samples\": [ @for s in 0..4 { { \"n\": ${s}, \"t\": ${temp + s} } } ], \"tags\": [ @for t in a,b sep \"; \" { \"${t}\" } ]}\n",
    );
    assert_eq!(
        shape(&t.pieces),
        "S(\"{\\\"samples\\\": [ \") P(s in 0..4 [S(\"{ \\\"n\\\": \") H(s) S(\", \\\"t\\\": \") H(temp + s) S(\" }\")]) S(\" ], \\\"tags\\\": [ \") P(t in a,b sep \"; \" [S(\"\\\"\") H(t) S(\"\\\"\")]) S(\" ]}\")"
    );
    let TilePiece::Projection { source, .. } = &t.pieces[1] else {
        panic!()
    };
    assert_eq!(source.element_names(), vec!["s"]);
    // Nested projection inside a projection body.
    let t = tile("tile n := \"@for a in 1..3 { @for b in 1..3 { ${a}:${b}; } }\"\n");
    assert!(
        matches!(&t.pieces[0], TilePiece::Projection { body, .. } if matches!(&body[0], TilePiece::Projection { .. }))
    );
}

#[test]
fn projection_over_a_bound_producer_and_a_where_predicate() {
    let t = tile("tile n := \"@for sweep { ${k} } @for k in 1..9 where {k} > 3 { ${k} }\"\n");
    let TilePiece::Projection { source, .. } = &t.pieces[0] else {
        panic!()
    };
    assert!(matches!(&source.kind, polydat::dsl::ast::ForSourceKind::Producer(p) if p == "sweep"));
    let TilePiece::Projection { source, .. } = &t.pieces[2] else {
        panic!()
    };
    assert!(source.text.contains("where {k} > 3"));
}

#[test]
fn branches_with_and_without_else() {
    let t = tile(
        "tile b := \"@if verbose { , \\\"audit\\\": \\\"${operator}\\\" } tail @if x > 1 { a } @else { b }\"\n",
    );
    let s = shape(&t.pieces);
    // Block padding is not body: `{ a }` is the text `a`.
    assert!(s.starts_with("B([S(\", \\\"audit\\\": \\\"\") H(operator) S(\"\\\"\")]) S(\" tail \") B([S(\"a\")] else [S(\"b\")])"), "{s}");
}

#[test]
fn custom_delimiters_and_sigil() {
    let t = tile(
        "tile t : text (delims \"<%\" \"%>\", sigil \"#\") := \"a <%x%> b #for k in 1..3 { <%k%> } ${literal} #if c { y }\"\n",
    );
    assert_eq!(t.options.open, "<%");
    assert_eq!(t.options.close, "%>");
    assert_eq!(t.options.sigil, "#");
    assert_eq!(
        shape(&t.pieces),
        "S(\"a \") H(x) S(\" b \") P(k in 1..3 [H(k)]) S(\" ${literal} \") B([S(\"y\")])"
    );
}

#[test]
fn doubled_open_delimiter_is_a_literal() {
    let t = tile("tile t := \"cost ${${amount} is ${amount}\"\n");
    assert_eq!(shape(&t.pieces), "S(\"cost ${amount} is \") H(amount)");
}

#[test]
fn braces_inside_hole_expressions_and_strings_do_not_close_holes() {
    let t = tile("tile t := \"${if x > 1 { 1 } else { 2 }} ${str_concat(\\\"}\\\", y)}\"\n");
    assert_eq!(
        shape(&t.pieces),
        "H(if x > 1 { 1 } else { 2 }) S(\" \") H(str_concat(\"}\", y))"
    );
}

#[test]
fn pretty_printer_round_trips_every_body_form() {
    let src = "tile doc : json := {\n    \"tenant\": ${tenant_id},\n    \"samples\": [ @for s in 0..4 { { \"n\": ${s} } } ]\n}\ntile load : text := <<<\nINSERT ${keyspace} '${doc!}'\n>>>\ntile row : csv := \"${a},${b}\"\ntile odd : text (delims \"<%\" \"%>\", sigil \"#\", strict) := \"<%x%> #if c { y }\"\n";
    let printed = pp_file(&parse(src));
    assert_eq!(printed, src);
    assert_eq!(pp_file(&parse(&printed)), printed);
}

#[test]
fn template_render_is_the_inverse_of_template_parse() {
    use polydat::dsl::ast::TileOptions;
    use polydat::dsl::tile::{parse_template, render_template};
    let opts = TileOptions::default();
    let span = polydat::dsl::lexer::Span { line: 1, col: 1 };
    for text in [
        "plain",
        "a ${x} b",
        "${${x} ${y: u64 | .2 !}",
        "@for k in 1..3 sep \", \" { ${k} } @if z { p } @else { q }",
    ] {
        let pieces = parse_template(text, &opts, span).unwrap();
        let rendered = render_template(&pieces, &opts);
        let again = parse_template(&rendered, &opts, span).unwrap();
        assert_eq!(render_template(&again, &opts), rendered, "{text}");
    }
}

#[test]
fn errors_name_the_tile_and_the_problem() {
    let e = parse_err("tile t := \"${x\"\n");
    assert!(
        e.contains("tile 't'") && e.contains("unterminated hole"),
        "{e}"
    );
    let e = parse_err("tile t := \"${}\"\n");
    assert!(e.contains("empty hole"), "{e}");
    let e = parse_err("tile t : json := {\n \"a\": ${x}\n");
    assert!(e.contains("never closed"), "{e}");
    let e = parse_err("tile t : text := <<<\nabc\n");
    assert!(e.contains("heredoc never closed"), "{e}");
    let e = parse_err("tile t : yaml := \"x\"\n");
    assert!(e.contains("unknown encoding 'yaml'"), "{e}");
    let e = parse_err("tile t (delims \"<%\") := \"x\"\n");
    assert!(e.contains("delims close"), "{e}");
    let e = parse_err("tile t := \"@for k in 1..3 ${k}\"\n");
    assert!(e.contains("no `{` block"), "{e}");
    let e = parse_err("tile t := \"@if a { b\"\n");
    assert!(e.contains("unterminated block"), "{e}");
    let e = parse_err("tile t := \"${1 +}\"\n");
    assert!(e.contains("hole `1 +`"), "{e}");
}

#[test]
fn compiler_lowers_a_tile_to_a_wire() {
    let mut k =
        polydat::dsl::compile_polydat("input cycle: u64\ntile t := \"n=${cycle}\"\n").unwrap();
    k.set_inputs(&[4]);
    assert_eq!(k.pull("t").as_str(), "n=4");
}

#[test]
fn a_tile_body_without_the_binding_operator_is_a_clear_error() {
    // A tile binds a wire, so every body form follows `:=`.
    let e = parse_err("tile doc : json {\"n\": 1}\n");
    assert!(e.contains("tile 'doc'"), "{e}");
    assert!(e.contains("expected `:=`"), "{e}");
    let e = parse_err("tile load <<<\nx\n>>>\n");
    assert!(e.contains("expected `:=`"), "{e}");
    // The header that lacks it does not swallow the next binding's value.
    let e = parse_err("tile doc : json\nx := 1\n");
    assert!(e.contains("tile 'doc'"), "{e}");
}
