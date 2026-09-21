// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Workspace-level verification for the definitive grammar spec.
//!
//! Guards three properties of
//! [`docs/design/polydat_grammar.md`](../docs/design/polydat_grammar.md)
//! and its programmatic companion
//! [`polydat_grammar_programmatic.md`](../docs/design/polydat_grammar_programmatic.md):
//!
//! 1. **Every ` ```polydat ` block parses and round-trips idempotently.**
//!    `pp_file(parse(src))` re-parses and re-projects to itself — the
//!    documented projection contract (`pprint.rs`). A doc example that
//!    is not real, parseable, canonical Polydat fails here.
//! 2. **Every ` ```polydat compile ` block compiles.** The
//!    `compile`-tagged subset is run through `compile_polydat`. Blocks
//!    additionally tagged `vectordata` are compiled only when that
//!    feature is on (parsing is feature-independent; compiling the
//!    vectordata cursor sugar is not).
//! 3. **The programmatic examples build the same kernel.** For each
//!    paired example, a hand-assembled `PolydatFile` projects to exactly
//!    the same canonical syntax as the spec's grammar source
//!    (`pp_file(builder) == pp_file(parse(grammar_src))`), the example
//!    appears as a ` ```polydat ` block in the spec, and the builder
//!    code appears verbatim in the programmatic doc (so the doc cannot
//!    silently drift from the verified code).
//!
//! The docs are read with `include_str!` (cwd-independent), mirroring
//! `adapter_catalog_invariants.rs`'s doc-as-spec pattern.

use polydat::ast::PortType;
use polydat::dsl::ast::{
    Arg, BinOpKind, Binding, BindingModifier, CallExpr, CursorDecl, Expr, ExternPort, ForSource,
    ForSourceKind, ForStmt, InputDecl, ModuleDef, PolydatFile, Statement, TileBodyKind, TileDef,
    TileOptions, TilePiece, TypedParam,
};
use polydat::dsl::compile_polydat;
use polydat::dsl::lexer::Span;
use polydat::dsl::pprint::pp_file;
use polydat::dsl::{lexer, parser};
use polydat::iteration::comprehension::spec::parse_comprehension_algebra;

const GRAMMAR_DOC: &str = include_str!("../docs/design/polydat_grammar.md");
const PROG_DOC: &str = include_str!("../docs/design/polydat_grammar_programmatic.md");

// ── Fenced-block extraction ───────────────────────────────────────

struct Block {
    /// Info-string directives after the language token (e.g. `compile`).
    directives: Vec<String>,
    /// The block body (fence lines excluded), newline-terminated.
    body: String,
    /// 1-based line where the opening fence sits, for diagnostics.
    start_line: usize,
}

/// Extract every fenced block whose info string's first token equals
/// `lang`. No line of block *content* in these docs begins with a fence,
/// so a `trim_start` test on ```` ``` ```` reliably toggles in/out.
fn extract_blocks(md: &str, lang: &str) -> Vec<Block> {
    let mut out = Vec::new();
    let mut lines = md.lines().enumerate();
    while let Some((idx, line)) = lines.next() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("```") {
            continue;
        }
        let info = trimmed.trim_start_matches('`');
        let mut tokens = info.split([' ', ',']).filter(|t| !t.is_empty());
        let is_match = tokens.next() == Some(lang);
        let directives: Vec<String> = tokens.map(|t| t.to_string()).collect();
        // Consume the body up to the closing fence.
        let mut body = String::new();
        let mut closed = false;
        for (_, body_line) in lines.by_ref() {
            if body_line.trim_start().starts_with("```") {
                closed = true;
                break;
            }
            body.push_str(body_line);
            body.push('\n');
        }
        debug_assert!(closed, "unterminated fence at line {}", idx + 1);
        if is_match {
            out.push(Block {
                directives,
                body,
                start_line: idx + 1,
            });
        }
    }
    out
}

fn parse(src: &str) -> Result<PolydatFile, String> {
    let tokens = lexer::lex(src)?;
    parser::parse(tokens)
}

fn project(src: &str, label: &str) -> String {
    let ast = parse(src)
        .unwrap_or_else(|e| panic!("{label} failed to parse: {e}\n--- source ---\n{src}"));
    pp_file(&ast)
}

// ── Property 1: idempotent round-trip ─────────────────────────────

#[test]
fn polydat_blocks_round_trip_idempotently() {
    let blocks = extract_blocks(GRAMMAR_DOC, "polydat");
    assert!(
        !blocks.is_empty(),
        "no ```polydat blocks found in polydat_grammar.md — the spec must \
         carry at least one verifiable example"
    );
    for b in &blocks {
        let loc = format!("```polydat block at line {}", b.start_line);
        let p1 = project(&b.body, &loc);
        let p2 = project(&p1, &format!("re-projection of {loc}"));
        assert_eq!(
            p1, p2,
            "{loc} is not idempotent under projection.\n\
             --- original ---\n{}\n--- first projection ---\n{p1}\n\
             --- second projection ---\n{p2}",
            b.body
        );
    }
}

// ── Property 2: compile-tagged blocks compile ─────────────────────

#[test]
fn polydat_compile_blocks_compile() {
    let mut compiled = 0;
    for b in extract_blocks(GRAMMAR_DOC, "polydat") {
        if !b.directives.iter().any(|d| d == "compile") {
            continue;
        }
        if b.directives.iter().any(|d| d == "vectordata") && !cfg!(feature = "vectordata") {
            continue;
        }
        compile_polydat(&b.body).unwrap_or_else(|e| {
            panic!(
                "```polydat compile block at line {} failed to compile: {e}\n\
                 --- source ---\n{}",
                b.start_line, b.body
            )
        });
        compiled += 1;
    }
    assert!(
        compiled > 0,
        "expected at least one ```polydat compile block to compile"
    );
}

// ── Property 3: programmatic builders match the grammar ───────────

/// A paired example: an anchor, the spec grammar source, a verbatim
/// snippet that must appear in the programmatic doc, and the builder.
struct Paired {
    anchor: &'static str,
    grammar_src: &'static str,
    /// A distinctive line of the builder that must appear verbatim in
    /// `polydat_grammar_programmatic.md` (drift guard).
    doc_snippet: &'static str,
    build: fn() -> PolydatFile,
}

#[test]
fn programmatic_examples_match_grammar() {
    // Canonical projections of every spec ```polydat block, so we can
    // confirm each paired example actually appears in the spec.
    let spec_projections: Vec<String> = extract_blocks(GRAMMAR_DOC, "polydat")
        .iter()
        .map(|b| project(&b.body, "spec block"))
        .collect();

    let examples = paired_examples();
    assert!(!examples.is_empty());

    for p in &examples {
        let canonical = project(p.grammar_src, p.anchor);

        // (a) The builder converges on the same canonical syntax.
        let built = pp_file(&(p.build)());
        assert_eq!(
            built, canonical,
            "builder `{}` does not project to its grammar source.\n\
             --- builder projection ---\n{built}\n--- grammar projection ---\n{canonical}",
            p.anchor
        );

        // (b) The example is actually present in the spec.
        assert!(
            spec_projections.iter().any(|s| s == &canonical),
            "paired example `{}` is not present as a ```polydat block in \
             polydat_grammar.md",
            p.anchor
        );

        // (c) The programmatic doc anchors and shows the builder.
        assert!(
            PROG_DOC.contains(p.anchor),
            "anchor `{}` missing from polydat_grammar_programmatic.md",
            p.anchor
        );
        assert!(
            PROG_DOC.contains(p.doc_snippet),
            "builder snippet for `{}` not found verbatim in \
             polydat_grammar_programmatic.md — doc and code have drifted.\n\
             expected substring:\n{}",
            p.anchor,
            p.doc_snippet
        );
    }
}

// ── Builders (must stay identical to the snippets in the prog doc) ──

fn sp() -> Span {
    Span { line: 0, col: 0 }
}
fn input(name: &str, ty: &str) -> Statement {
    Statement::InputDecl(InputDecl {
        name: name.into(),
        ty: Some(ty.into()),
        span: sp(),
    })
}
fn id(n: &str) -> Expr {
    Expr::Ident(n.into(), sp())
}
fn int(n: u64) -> Expr {
    Expr::IntLit(n, sp())
}
fn flt(n: f64) -> Expr {
    Expr::FloatLit(n, sp())
}
fn call(func: &str, args: Vec<Expr>) -> Expr {
    Expr::Call(CallExpr {
        func: func.into(),
        args: args.into_iter().map(Arg::Positional).collect(),
        span: sp(),
    })
}
fn binop(l: Expr, op: BinOpKind, r: Expr) -> Expr {
    Expr::BinOp(Box::new(l), op, Box::new(r))
}
fn bind(target: &str, value: Expr) -> Statement {
    Statement::Binding(Binding {
        targets: vec![target.into()],
        value,
        modifier: BindingModifier::NONE,
        type_annotation: None,
        span: sp(),
    })
}
fn bind_multi(targets: &[&str], value: Expr) -> Statement {
    Statement::Binding(Binding {
        targets: targets.iter().map(|s| s.to_string()).collect(),
        value,
        modifier: BindingModifier::NONE,
        type_annotation: None,
        span: sp(),
    })
}
fn file(statements: Vec<Statement>) -> PolydatFile {
    PolydatFile { statements }
}

fn build_minimal() -> PolydatFile {
    file(vec![
        input("cycle", "u64"),
        bind("hashed", call("hash", vec![id("cycle")])),
        bind("user_id", call("mod", vec![id("hashed"), int(1_000_000)])),
    ])
}

fn build_destructure() -> PolydatFile {
    file(vec![
        input("cycle", "u64"),
        bind_multi(
            &["region", "store", "tx"],
            call("mixed_radix", vec![id("cycle"), int(50), int(200), int(0)]),
        ),
        bind(
            "region_id",
            call("mod", vec![call("hash", vec![id("region")]), int(10_000)]),
        ),
        bind(
            "store_id",
            call(
                "mod",
                vec![
                    call(
                        "hash",
                        vec![call("interleave", vec![id("region"), id("store")])],
                    ),
                    int(100_000),
                ],
            ),
        ),
    ])
}

fn build_tuple_input() -> PolydatFile {
    file(vec![
        input("cycle", "u64"),
        input("thread", "u64"),
        bind(
            "combined",
            call("interleave", vec![id("cycle"), id("thread")]),
        ),
        bind(
            "row_key",
            call(
                "mod",
                vec![call("hash", vec![id("combined")]), int(1_000_000)],
            ),
        ),
    ])
}

fn build_extern() -> PolydatFile {
    file(vec![
        input("cycle", "u64"),
        Statement::ExternPort(ExternPort {
            name: "scale".into(),
            typ: "u64".into(),
            default: None,
            span: sp(),
        }),
        bind("result", binop(id("cycle"), BinOpKind::Mul, id("scale"))),
    ])
}

fn build_cursor_over() -> PolydatFile {
    file(vec![
        Statement::Cursor(CursorDecl {
            name: "q".into(),
            constructor: call("range", vec![int(0), int(100)]),
            over: Some(id("p")),
            span: sp(),
        }),
        bind(
            "i",
            Expr::FieldAccess {
                source: "q__cursor".into(),
                field: "idx".into(),
                span: sp(),
            },
        ),
        bind(
            "ratio",
            binop(
                Expr::Cast(Box::new(id("i")), PortType::F64, sp()),
                BinOpKind::Div,
                flt(100.0),
            ),
        ),
    ])
}

fn build_module() -> PolydatFile {
    let body = vec![
        bind(
            "pos",
            call(
                "to_f64",
                vec![binop(id("input"), BinOpKind::Mod, id("period"))],
            ),
        ),
        bind("per", call("to_f64", vec![id("period")])),
        bind(
            "value",
            call(
                "sin",
                vec![binop(
                    binop(id("pos"), BinOpKind::Div, id("per")),
                    BinOpKind::Mul,
                    flt(std::f64::consts::TAU),
                )],
            ),
        ),
    ];
    file(vec![Statement::ModuleDef(ModuleDef {
        name: "sine_wave".into(),
        params: vec![
            TypedParam {
                name: "input".into(),
                typ: "u64".into(),
            },
            TypedParam {
                name: "period".into(),
                typ: "u64".into(),
            },
        ],
        outputs: vec![TypedParam {
            name: "value".into(),
            typ: "f64".into(),
        }],
        body,
        span: sp(),
    })])
}

fn comprehension(text: &str) -> ForSource {
    let algebra = parse_comprehension_algebra(text).expect("comprehension text");
    // The source takes its text from the tree, so the two halves cannot
    // disagree; the `for` lowering refuses one where they do.
    ForSource::comprehension(algebra, sp()).expect("the text writes this tree")
}

fn build_for() -> PolydatFile {
    file(vec![
        bind(
            "sweep",
            Expr::For(Box::new(comprehension(
                "k in 1..4, limit in 10, 20, 30 order halton/5",
            ))),
        ),
        Statement::For(ForStmt {
            source: ForSource::producer("sweep", sp()),
            body: vec![
                bind("f", call("myfunc", vec![id("k")])),
                bind("g", call("otherfunc", vec![id("limit"), id("k")])),
            ],
            span: sp(),
        }),
        Statement::For(ForStmt {
            source: comprehension("phase in load, verify, p in partitions(\"*/4\", 1000000)"),
            body: vec![
                bind("row", call("mod_in", vec![id("cycle"), id("p")])),
                Statement::For(ForStmt {
                    source: comprehension("q in 1..2"),
                    body: vec![bind("z", call("hash", vec![id("q")]))],
                    span: sp(),
                }),
            ],
            span: sp(),
        }),
    ])
}

fn tile(
    name: &str,
    encoding: &str,
    options: TileOptions,
    body_kind: TileBodyKind,
    body: &str,
) -> Statement {
    Statement::Tile(
        TileDef::from_body(name, Some(encoding.into()), options, body_kind, body, sp())
            .expect("template"),
    )
}

fn build_tile() -> PolydatFile {
    let odd_options = TileOptions {
        open: "<%".into(),
        close: "%>".into(),
        sigil: "#".into(),
        strict: true,
        in_string: false,
    };
    file(vec![
        tile(
            "doc",
            "json",
            TileOptions::default(),
            TileBodyKind::Block,
            "{\n    \"tenant\": ${tenant_id},\n    \"samples\": [ @for s in 0..4 { { \"n\": ${s} } } ]\n}",
        ),
        tile(
            "load",
            "text",
            TileOptions::default(),
            TileBodyKind::Heredoc,
            "INSERT ${keyspace} '${doc!}'",
        ),
        tile(
            "row",
            "csv",
            TileOptions::default(),
            TileBodyKind::Literal,
            "${a},${b}",
        ),
        tile(
            "odd",
            "text",
            odd_options,
            TileBodyKind::Literal,
            "<%x%> #if c { y }",
        ),
    ])
}

fn paired_examples() -> Vec<Paired> {
    vec![
        Paired {
            anchor: "p-minimal",
            grammar_src: "input cycle: u64\nhashed := hash(cycle)\nuser_id := mod(hashed, 1000000)\n",
            doc_snippet: "bind(\"user_id\", call(\"mod\", vec![id(\"hashed\"), int(1_000_000)])),",
            build: build_minimal,
        },
        Paired {
            anchor: "p-destructure",
            grammar_src: "input cycle: u64\n(region, store, tx) := mixed_radix(cycle, 50, 200, 0)\nregion_id := mod(hash(region), 10000)\nstore_id := mod(hash(interleave(region, store)), 100000)\n",
            doc_snippet: "bind_multi(&[\"region\", \"store\", \"tx\"], call(\"mixed_radix\", vec![id(\"cycle\"), int(50), int(200), int(0)])),",
            build: build_destructure,
        },
        Paired {
            anchor: "p-tuple-input",
            grammar_src: "input (cycle: u64, thread: u64)\ncombined := interleave(cycle, thread)\nrow_key := mod(hash(combined), 1000000)\n",
            doc_snippet: "bind(\"row_key\", call(\"mod\", vec![call(\"hash\", vec![id(\"combined\")]), int(1_000_000)])),",
            build: build_tuple_input,
        },
        Paired {
            anchor: "p-extern",
            grammar_src: "input cycle: u64\nextern scale: u64\nresult := cycle * scale\n",
            doc_snippet: "ExternPort { name: \"scale\".into(), typ: \"u64\".into(), default: None, span: sp() }",
            build: build_extern,
        },
        Paired {
            anchor: "p-cursor-over",
            grammar_src: "cursor q = range(0, 100) over p\ni := q.cursor.idx\nratio := (i as f64) / 100.0\n",
            doc_snippet: "over: Some(id(\"p\")),",
            build: build_cursor_over,
        },
        Paired {
            anchor: "p-module",
            grammar_src: "sine_wave(input: u64, period: u64) -> (value: f64) := {\n    pos := to_f64(input % period)\n    per := to_f64(period)\n    value := sin((pos / per) * 6.283185307179586)\n}\n",
            doc_snippet: "name: \"sine_wave\".into(),",
            build: build_module,
        },
        Paired {
            anchor: "p-for",
            grammar_src: "sweep := for k in 1..4, limit in 10, 20, 30 order halton/5\nfor sweep {\n    f := myfunc(k)\n    g := otherfunc(limit, k)\n}\nfor phase in load, verify, p in partitions(\"*/4\", 1000000) {\n    row := mod_in(cycle, p)\n    for q in 1..2 {\n        z := hash(q)\n    }\n}\n",
            doc_snippet: "ForSource::comprehension(algebra, sp()).expect(\"the text writes this tree\")",
            build: build_for,
        },
        Paired {
            anchor: "p-tile",
            grammar_src: "tile doc : json := {\n    \"tenant\": ${tenant_id},\n    \"samples\": [ @for s in 0..4 { { \"n\": ${s} } } ]\n}\ntile load : text := <<<\nINSERT ${keyspace} '${doc!}'\n>>>\ntile row : csv := \"${a},${b}\"\ntile odd : text (delims \"<%\" \"%>\", sigil \"#\", strict) := \"<%x%> #if c { y }\"\n",
            doc_snippet: "TileDef::from_body(",
            build: build_tile,
        },
    ]
}

// ── Grammar coverage: every surface form has a worked example ──────

/// The name of a statement's form, as the grammar spells it.
///
/// Exhaustive on purpose: a new `Statement` variant is a compile
/// error here, and then a missing example is a test failure below
/// (F-L7). Coverage used to be whatever someone had pasted into the
/// document, and the document was missing `for`, `tile`, the `if`
/// block, and `shared x: T` — four of the newest forms in the
/// language.
fn statement_form(s: &Statement) -> &'static str {
    match s {
        Statement::InputDecl(_) => "input",
        Statement::Binding(b) => match () {
            _ if b.modifier.is_const() => "const binding",
            _ if b.modifier.has(polydat::dsl::ast::WireModifier::Shared) => "shared binding",
            _ if b.modifier.is_volatile() => "volatile binding",
            _ => "binding",
        },
        Statement::ModuleDef(_) => "module",
        Statement::ExternPort(_) => "extern",
        Statement::Cursor(_) => "cursor",
        Statement::Pragma { .. } => "pragma",
        Statement::For(_) => "for",
        Statement::Tile(_) => "tile",
    }
}

/// The name of an expression's form.
fn expr_form(e: &Expr) -> &'static str {
    match e {
        Expr::Ident(..) => "identifier",
        Expr::IntLit(..) => "integer literal",
        Expr::FloatLit(..) => "float literal",
        Expr::StringLit(..) => "string literal",
        Expr::ArrayLit(..) => "list literal",
        Expr::Call(_) => "call",
        Expr::BinOp(..) => "binary operator",
        Expr::UnaryNeg(..) => "unary minus",
        Expr::UnaryBitNot(..) => "unary bitwise not",
        Expr::FieldAccess { .. } => "field access",
        Expr::Cast(..) => "cast",
        Expr::For(_) => "for expression",
    }
}

/// The name of a `for` source's form.
fn for_source_form(s: &ForSourceKind) -> &'static str {
    match s {
        ForSourceKind::Producer(_) => "for over a producer",
        ForSourceKind::Comprehension(_) => "for over a comprehension",
        ForSourceKind::Derived { .. } => "for over a derived producer",
    }
}

/// The name of a tile template piece's form.
fn tile_piece_form(p: &TilePiece) -> &'static str {
    match p {
        TilePiece::Static(_) => "tile static text",
        TilePiece::Hole(_) => "tile hole",
        TilePiece::Projection { .. } => "tile projection",
        TilePiece::Branch { .. } => "tile branch",
    }
}

/// The name of a tile body's delimiting form.
fn tile_body_form(k: &TileBodyKind) -> &'static str {
    match k {
        TileBodyKind::Block => "tile block body",
        TileBodyKind::Heredoc => "tile heredoc body",
        TileBodyKind::Literal => "tile literal body",
    }
}

/// Every form the four surface enums can take. The test asserts the
/// document's examples cover all of them.
fn every_form() -> Vec<&'static str> {
    vec![
        "input",
        "binding",
        "const binding",
        "shared binding",
        "volatile binding",
        "module",
        "extern",
        "cursor",
        "pragma",
        "for",
        "tile",
        "identifier",
        "integer literal",
        "float literal",
        "string literal",
        "list literal",
        "call",
        "binary operator",
        "unary minus",
        "unary bitwise not",
        "field access",
        "cast",
        "for expression",
        "for over a producer",
        "for over a comprehension",
        "for over a derived producer",
        "tile static text",
        "tile hole",
        "tile projection",
        "tile branch",
        "tile block body",
        "tile heredoc body",
        "tile literal body",
    ]
}

fn walk_expr(e: &Expr, seen: &mut std::collections::BTreeSet<&'static str>) {
    seen.insert(expr_form(e));
    match e {
        Expr::Call(c) => {
            for a in &c.args {
                let inner = match a {
                    Arg::Positional(x) => x,
                    Arg::Named(_, x) => x,
                };
                walk_expr(inner, seen);
            }
        }
        Expr::BinOp(l, _, r) => {
            walk_expr(l, seen);
            walk_expr(r, seen);
        }
        Expr::UnaryNeg(i, _) | Expr::UnaryBitNot(i, _) | Expr::Cast(i, _, _) => walk_expr(i, seen),
        Expr::ArrayLit(items, _) => {
            for i in items {
                walk_expr(i, seen);
            }
        }
        Expr::For(src) => walk_for_source(src, seen),
        Expr::Ident(..)
        | Expr::IntLit(..)
        | Expr::FloatLit(..)
        | Expr::StringLit(..)
        | Expr::FieldAccess { .. } => {}
    }
}

fn walk_for_source(s: &ForSource, seen: &mut std::collections::BTreeSet<&'static str>) {
    seen.insert(for_source_form(&s.kind));
}

fn walk_tile_piece(p: &TilePiece, seen: &mut std::collections::BTreeSet<&'static str>) {
    seen.insert(tile_piece_form(p));
    match p {
        TilePiece::Projection { source, body, .. } => {
            walk_for_source(source, seen);
            for b in body {
                walk_tile_piece(b, seen);
            }
        }
        TilePiece::Branch {
            cond,
            then,
            otherwise,
            ..
        } => {
            walk_expr(cond, seen);
            for b in then.iter().chain(otherwise.iter().flatten()) {
                walk_tile_piece(b, seen);
            }
        }
        TilePiece::Static(_) | TilePiece::Hole(_) => {}
    }
}

fn walk_statement(s: &Statement, seen: &mut std::collections::BTreeSet<&'static str>) {
    seen.insert(statement_form(s));
    match s {
        Statement::Binding(b) => walk_expr(&b.value, seen),
        Statement::ModuleDef(m) => {
            for inner in &m.body {
                walk_statement(inner, seen);
            }
        }
        Statement::For(f) => {
            walk_for_source(&f.source, seen);
            for inner in &f.body {
                walk_statement(inner, seen);
            }
        }
        Statement::Tile(t) => {
            seen.insert(tile_body_form(&t.body_kind));
            for p in &t.pieces {
                walk_tile_piece(p, seen);
            }
        }
        Statement::InputDecl(_)
        | Statement::ExternPort(_)
        | Statement::Cursor(_)
        | Statement::Pragma { .. } => {}
    }
}

/// Every surface form the grammar can express appears in a worked
/// example in the specification.
///
/// The document's coverage used to be whatever examples someone had
/// pasted in, and nothing noticed a form that had none: `for`, `tile`,
/// the tile `if` block, and `shared x: T` were all absent. The form
/// lists above are exhaustive matches, so a new AST variant fails to
/// compile here until it is named, and then fails this test until the
/// specification shows it.
#[test]
fn every_surface_form_has_an_example_in_the_spec() {
    let mut seen: std::collections::BTreeSet<&'static str> = Default::default();
    for block in extract_blocks(GRAMMAR_DOC, "polydat") {
        let Ok(file) = parse(&block.body) else {
            // The round-trip test above reports an unparseable block;
            // this one is about coverage.
            continue;
        };
        for s in &file.statements {
            walk_statement(s, &mut seen);
        }
    }
    let missing: Vec<&str> = every_form()
        .into_iter()
        .filter(|f| !seen.contains(f))
        .collect();
    assert!(
        missing.is_empty(),
        "polydat_grammar.md has no worked example of: {}",
        missing.join(", ")
    );
}
