// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Fuzz coverage for the tile grammar (SRD 114 step 1).
//!
//! A grammar-directed generator emits random tile statements over every
//! body form and encoding, with random delimiters and sigils, holes with
//! random expressions, declared types, formats, and raw flags,
//! projections with separators and nesting, branches, doubled-open
//! escapes, and static text drawn from characters the delimiters could
//! collide with. Invariants: the front end accepts the program, the
//! parsed pieces match what was emitted, the pretty-printer reaches a
//! fixed point, the template renderer inverts the template parser, and
//! the compiler declines by name. A mutation pass checks that no stage
//! panics on damaged input and that every error is a sentence.
//!
//! `FUZZ_SEED`, `FUZZ_ITERATIONS`, and an ignored `SUPERFUZZ_SEEDS`
//! sweep follow the other fuzzers.

use polydat::dsl::ast::{PolydatFile, Statement, TileOptions, TilePiece};
use polydat::dsl::pprint::pp_file;
use polydat::dsl::tile::{parse_template, render_template};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() as usize) % n
        }
    }
    fn coin(&mut self, pct: usize) -> bool {
        self.range(100) < pct
    }
    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.range(items.len())]
    }
}

/// The shape the generator emitted, for comparison against the parse.
#[derive(Debug, Clone, PartialEq)]
enum Shape {
    Static,
    Hole {
        typed: bool,
        formatted: bool,
        raw: bool,
    },
    Projection {
        sep: bool,
        body: Vec<Shape>,
    },
    Branch {
        has_else: bool,
        then: Vec<Shape>,
        otherwise: Vec<Shape>,
    },
}

const DELIMS: &[(&str, &str)] = &[
    ("${", "}"),
    ("<%", "%>"),
    ("{{", "}}"),
    ("[[", "]]"),
    ("<<", ">>"),
    ("$(", ")"),
];
const SIGILS: &[&str] = &["@", "#", "%%", "::"];
const TYPES: &[&str] = &["u64", "f64", "str", "bool", "i64", "json"];
const FORMATS: &[&str] = &[".2", "05", ">8", "<4", ".3", "x"];
const EXPRS: &[&str] = &[
    "cycle",
    "hash(cycle)",
    "a + b",
    "mod(x, 10)",
    "hashed_id(input: cycle, bound: 10)",
    "\"quoted } text\"",
    "if x > 1 { 1 } else { 2 }",
    "temp_c + s",
    "k",
];
const STATIC_WORDS: &[&str] = &[
    "plain",
    "INSERT INTO t",
    "\"key\": ",
    "a,b,c",
    " { nested } ",
    "50% off",
    "x=1;",
    "tail ",
];

struct Gen {
    rng: Rng,
    opts: TileOptions,
    encoding: Option<&'static str>,
}

impl Gen {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let (open, close) = DELIMS[rng.range(DELIMS.len())];
        let sigil = SIGILS[rng.range(SIGILS.len())];
        let strict = rng.coin(20);
        let encoding = match rng.range(4) {
            0 => None,
            1 => Some("json"),
            2 => Some("text"),
            _ => Some("csv"),
        };
        let in_string = rng.coin(10);
        Gen {
            rng,
            opts: TileOptions {
                open: open.into(),
                close: close.into(),
                sigil: sigil.into(),
                strict,
                in_string,
            },
            encoding,
        }
    }

    /// Static text that never contains the open delimiter, the sigil
    /// followed by a directive word, or a bare brace that would break a
    /// block body; braces are allowed only balanced.
    fn static_text(&mut self) -> String {
        let mut s = String::new();
        let n = 1 + self.rng.range(3);
        for _ in 0..n {
            let w = self.rng.pick(STATIC_WORDS);
            s.push_str(w);
            s.push(' ');
        }
        // Add a literal open delimiter via the doubled escape sometimes.
        if self.rng.coin(15) {
            s.push_str(&format!("{}{}", self.opts.open, self.opts.open));
            s.push_str("lit ");
        }
        s.replace(&self.opts.open, "")
            .replace(&format!("{}for", self.opts.sigil), "")
            .replace(&format!("{}if", self.opts.sigil), "")
            + if s.is_empty() { "z" } else { "" }
    }

    fn hole(&mut self) -> (String, Shape) {
        let expr = self.rng.pick(EXPRS);
        let typed = self.rng.coin(30);
        let formatted = self.rng.coin(30);
        let raw = self.rng.coin(20);
        let mut inner = expr.to_string();
        if typed {
            inner.push_str(&format!(": {}", self.rng.pick(TYPES)));
        }
        if formatted {
            inner.push_str(&format!(" | {}", self.rng.pick(FORMATS)));
        }
        if raw {
            inner.push('!');
        }
        (
            format!("{}{}{}", self.opts.open, inner, self.opts.close),
            Shape::Hole {
                typed,
                formatted,
                raw,
            },
        )
    }

    fn comprehension(&mut self) -> String {
        match self.rng.range(4) {
            0 => format!("k in 1..{}", 2 + self.rng.range(5)),
            1 => "s in 0..4, t in 10,20".to_string(),
            2 => format!("k in 1..9 where {{k}} > {}", self.rng.range(5)),
            _ => "sweep".to_string(),
        }
    }

    fn pieces(&mut self, depth: usize) -> (String, Vec<Shape>) {
        let mut text = String::new();
        let mut shapes = Vec::new();
        let n = 1 + self.rng.range(4);
        for _ in 0..n {
            match self.rng.range(if depth > 0 { 6 } else { 4 }) {
                0 | 1 => {
                    text.push_str(&self.static_text());
                    shapes.push(Shape::Static);
                }
                2 | 3 => {
                    let (h, s) = self.hole();
                    text.push_str(&h);
                    shapes.push(s);
                }
                4 => {
                    let comp = self.comprehension();
                    let sep = self.rng.coin(40);
                    let (body, body_shapes) = self.pieces(depth - 1);
                    text.push_str(&format!(
                        "{}for {}{} {{{}}}",
                        self.opts.sigil,
                        comp,
                        if sep { " sep \", \"" } else { "" },
                        body
                    ));
                    shapes.push(Shape::Projection {
                        sep,
                        body: body_shapes,
                    });
                }
                _ => {
                    let (then, then_shapes) = self.pieces(depth - 1);
                    let has_else = self.rng.coin(50);
                    let (otherwise, else_shapes) = if has_else {
                        self.pieces(depth - 1)
                    } else {
                        (String::new(), Vec::new())
                    };
                    text.push_str(&format!("{}if x > 1 {{{}}}", self.opts.sigil, then));
                    if has_else {
                        text.push_str(&format!(" {}else {{{}}}", self.opts.sigil, otherwise));
                    }
                    shapes.push(Shape::Branch {
                        has_else,
                        then: then_shapes,
                        otherwise: else_shapes,
                    });
                }
            }
        }
        (text, shapes)
    }

    /// A whole program with one tile and a few plain statements.
    fn program(&mut self) -> (String, Vec<Shape>) {
        let (body, shapes) = self.pieces(2);
        let mut header = String::from("tile t");
        if let Some(e) = self.encoding {
            header.push_str(&format!(" : {e}"));
        }
        let defaults = TileOptions::default();
        let mut opts = Vec::new();
        if self.opts.open != defaults.open || self.opts.close != defaults.close {
            opts.push(format!(
                "delims \"{}\" \"{}\"",
                self.opts.open, self.opts.close
            ));
        }
        if self.opts.sigil != defaults.sigil {
            opts.push(format!("sigil \"{}\"", self.opts.sigil));
        }
        if self.opts.strict {
            opts.push("strict".to_string());
        }
        if self.opts.in_string {
            opts.push("instring".to_string());
        }
        if !opts.is_empty() {
            header.push_str(&format!(" ({})", opts.join(", ")));
        }
        let body_form = match self.rng.range(3) {
            // A block body must be brace-balanced as a whole; wrap it.
            0 if !body.contains('\n') => format!("{{ {body} }}"),
            1 => format!("<<<\n{body}\n>>>"),
            _ => format!("\"{}\"", body.replace('\\', "\\\\").replace('"', "\\\"")),
        };
        let shapes = if body_form.starts_with('{') {
            // The wrapping braces become static text around the pieces.
            let mut s = vec![Shape::Static];
            s.extend(shapes);
            s.push(Shape::Static);
            s
        } else {
            shapes
        };
        let src = format!(
            "input cycle: u64\nsweep := for k in 1..3\n{header} := {body_form}\nx := hash(cycle)\n"
        );
        (src, shapes)
    }
}

fn parse(src: &str) -> Result<PolydatFile, String> {
    let tokens = polydat::dsl::lexer::lex(src)?;
    polydat::dsl::parser::parse(tokens)
}

/// Collapse adjacent statics, since the generator's static runs and the
/// block wrapper merge into single pieces.
fn normalize(shapes: &[Shape]) -> Vec<Shape> {
    let mut out: Vec<Shape> = Vec::new();
    for s in shapes {
        let s = match s {
            Shape::Projection { sep, body } => Shape::Projection {
                sep: *sep,
                body: normalize(body),
            },
            Shape::Branch {
                has_else,
                then,
                otherwise,
            } => Shape::Branch {
                has_else: *has_else,
                then: normalize(then),
                otherwise: normalize(otherwise),
            },
            other => other.clone(),
        };
        if matches!(s, Shape::Static) && matches!(out.last(), Some(Shape::Static)) {
            continue;
        }
        out.push(s);
    }
    out
}

fn shapes_of(pieces: &[TilePiece]) -> Vec<Shape> {
    pieces
        .iter()
        .map(|p| match p {
            TilePiece::Static(_) => Shape::Static,
            TilePiece::Hole(h) => Shape::Hole {
                typed: h.decl_type.is_some(),
                formatted: h.format.is_some(),
                raw: h.raw,
            },
            TilePiece::Projection { sep, body, .. } => Shape::Projection {
                sep: sep.is_some(),
                body: shapes_of(body),
            },
            TilePiece::Branch {
                then, otherwise, ..
            } => Shape::Branch {
                has_else: otherwise.is_some(),
                then: shapes_of(then),
                otherwise: otherwise.as_ref().map(|o| shapes_of(o)).unwrap_or_default(),
            },
        })
        .collect()
}

/// Statics in the generator are separated from directives by spaces the
/// parser keeps as statics; compare with statics collapsed on both sides.
fn strip_statics(shapes: &[Shape]) -> Vec<Shape> {
    shapes
        .iter()
        .filter(|s| !matches!(s, Shape::Static))
        .map(|s| match s {
            Shape::Projection { sep, body } => Shape::Projection {
                sep: *sep,
                body: strip_statics(body),
            },
            Shape::Branch {
                has_else,
                then,
                otherwise,
            } => Shape::Branch {
                has_else: *has_else,
                then: strip_statics(then),
                otherwise: strip_statics(otherwise),
            },
            other => other.clone(),
        })
        .collect()
}

fn panic_text(p: &Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "<non-string panic>".into())
}

fn cryptic(msg: &str) -> bool {
    let m = msg.to_lowercase();
    msg.trim().is_empty()
        || m.contains("panic")
        || m.contains("index out of bounds")
        || m.contains("unreachable")
}

fn run_wellformed_pass(seed: u64, iterations: usize) -> Vec<String> {
    const MAX: usize = 8;
    let mut failures = Vec::new();
    let repro = |i: usize| {
        format!(
            "reproduce: FUZZ_SEED={seed} FUZZ_ITERATIONS={} cargo test -p polydat --test fuzz_tile_syntax wellformed",
            i + 1
        )
    };
    let mut rng = Rng::new(seed);
    for i in 0..iterations {
        if failures.len() >= MAX {
            failures.push(format!("[seed {seed:#x}] … stopping after {MAX} failures"));
            break;
        }
        let mut g = Gen::new(rng.next_u64());
        let (source, expected) = g.program();
        let parsed = match std::panic::catch_unwind(|| parse(&source)) {
            Ok(Ok(f)) => f,
            Ok(Err(e)) => {
                failures.push(format!("[seed {seed:#x}] iteration {i}: parse rejected a well-formed tile:\n  {e}\n  source:\n{source}\n  {}", repro(i)));
                continue;
            }
            Err(p) => {
                failures.push(format!("[seed {seed:#x}] iteration {i}: front end panicked: {}\n  source:\n{source}\n  {}", panic_text(&p), repro(i)));
                continue;
            }
        };
        let Some(Statement::Tile(t)) = parsed
            .statements
            .iter()
            .find(|s| matches!(s, Statement::Tile(_)))
        else {
            failures.push(format!("[seed {seed:#x}] iteration {i}: no tile statement parsed.\n  source:\n{source}\n  {}", repro(i)));
            continue;
        };
        let got = strip_statics(&shapes_of(&t.pieces));
        let want = strip_statics(&normalize(&expected));
        if got != want {
            failures.push(format!("[seed {seed:#x}] iteration {i}: pieces differ.\n  expected: {want:?}\n  got:      {got:?}\n  source:\n{source}\n  {}", repro(i)));
            continue;
        }
        // Printer fixed point.
        let printed = pp_file(&parsed);
        match parse(&printed) {
            Ok(again) => {
                if pp_file(&again) != printed {
                    failures.push(format!("[seed {seed:#x}] iteration {i}: pretty-printer is not a fixed point.\n  printed:\n{printed}\n  {}", repro(i)));
                    continue;
                }
            }
            Err(e) => {
                failures.push(format!("[seed {seed:#x}] iteration {i}: printed form does not parse: {e}\n  printed:\n{printed}\n  {}", repro(i)));
                continue;
            }
        }
        // Template renderer inverts the template parser.
        let span = polydat::dsl::lexer::Span { line: 1, col: 1 };
        let rendered = render_template(&t.pieces, &t.options);
        match parse_template(&rendered, &t.options, span) {
            Ok(pieces) => {
                if render_template(&pieces, &t.options) != rendered {
                    failures.push(format!("[seed {seed:#x}] iteration {i}: render/parse of template is not a fixed point.\n  rendered:\n{rendered}\n  {}", repro(i)));
                    continue;
                }
            }
            Err(e) => {
                failures.push(format!("[seed {seed:#x}] iteration {i}: rendered template does not parse: {e}\n  rendered:\n{rendered}\n  {}", repro(i)));
                continue;
            }
        }
        // The compiler either lowers the tile to a wire that renders, or
        // errors in a sentence naming the tile. It never panics. Holes
        // reference names the generator does not define, so most
        // programs take the error path; the ones that compile must
        // render without panicking.
        match std::panic::catch_unwind(|| {
            let mut k = polydat::dsl::compile_polydat(&source)?;
            k.set_inputs(&[1]);
            Ok::<String, String>(k.pull("t").to_display_string())
        }) {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                if cryptic(&e) || !e.contains("tile 't'") && !e.contains("unknown wire") && !e.contains("unknown function") {
                    failures.push(format!("[seed {seed:#x}] iteration {i}: compiler error is cryptic or does not name the tile: {e}\n  source:\n{source}\n  {}", repro(i)));
                }
            }
            Err(p) => failures.push(format!("[seed {seed:#x}] iteration {i}: compiler or renderer panicked: {}\n  source:\n{source}\n  {}", panic_text(&p), repro(i))),
        }
    }
    failures
}

fn mutate(rng: &mut Rng, src: &str) -> String {
    let mut chars: Vec<char> = src.chars().collect();
    const INSERT: &[char] = &[
        '{', '}', '$', '%', '<', '>', '@', '#', '"', '\'', '(', ')', '|', ':', '!', ' ', '\n', 't',
        'i', 'l', 'e', 'f', 'o', 'r',
    ];
    for _ in 0..(1 + rng.range(4)) {
        if chars.is_empty() {
            break;
        }
        let at = rng.range(chars.len());
        match rng.range(4) {
            0 => {
                chars.remove(at);
            }
            1 => chars.insert(at, INSERT[rng.range(INSERT.len())]),
            2 => chars[at] = INSERT[rng.range(INSERT.len())],
            _ => {
                let end = (at + 1 + rng.range(10)).min(chars.len());
                let slice: Vec<char> = chars[at..end].to_vec();
                for (k, c) in slice.into_iter().enumerate() {
                    chars.insert(end + k, c);
                }
            }
        }
    }
    chars.into_iter().collect()
}

fn run_mutation_pass(seed: u64, iterations: usize) -> Vec<String> {
    const MAX: usize = 8;
    let mut failures = Vec::new();
    let repro = |i: usize| {
        format!(
            "reproduce: FUZZ_SEED={seed} FUZZ_ITERATIONS={} cargo test -p polydat --test fuzz_tile_syntax mutated",
            i + 1
        )
    };
    let mut rng = Rng::new(seed ^ 0x71E5_71E5);
    for i in 0..iterations {
        if failures.len() >= MAX {
            failures.push(format!("[seed {seed:#x}] … stopping after {MAX} failures"));
            break;
        }
        let mut g = Gen::new(rng.next_u64());
        let (clean, _) = g.program();
        let source = mutate(&mut rng, &clean);
        let outcome = std::panic::catch_unwind(|| {
            let parsed = parse(&source)?;
            let printed = pp_file(&parsed);
            parse(&printed).map_err(|e| {
                format!(
                    "printed form of a parsed mutant does not parse: {e}\n  printed:\n{printed}"
                )
            })?;
            polydat::dsl::compile_polydat(&source)
        });
        match outcome {
            Err(p) => failures.push(format!(
                "[seed {seed:#x}] iteration {i}: panic on mutant: {}\n  source:\n{source}\n  {}",
                panic_text(&p),
                repro(i)
            )),
            Ok(Err(e)) => {
                if cryptic(&e) || e.contains("printed form of a parsed mutant") {
                    failures.push(format!(
                        "[seed {seed:#x}] iteration {i}: {e}\n  source:\n{source}\n  {}",
                        repro(i)
                    ));
                }
            }
            Ok(Ok(_)) => {}
        }
    }
    failures
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[test]
fn wellformed_tiles_parse_print_and_are_declined() {
    let seed = env_u64("FUZZ_SEED", 0x7113_5EED);
    let iterations = env_u64("FUZZ_ITERATIONS", 400) as usize;
    let failures = run_wellformed_pass(seed, iterations);
    assert!(
        failures.is_empty(),
        "fuzz invariants violated ({} failures):\n\n{}",
        failures.len(),
        failures.join("\n---\n")
    );
}

#[test]
fn mutated_tiles_never_panic() {
    let seed = env_u64("FUZZ_SEED", 0x7113_5EED);
    let iterations = env_u64("FUZZ_ITERATIONS", 600) as usize;
    let failures = run_mutation_pass(seed, iterations);
    assert!(
        failures.is_empty(),
        "fuzz invariants violated ({} failures):\n\n{}",
        failures.len(),
        failures.join("\n---\n")
    );
}

#[test]
fn generator_smoke() {
    let mut rng = Rng::new(11);
    let (
        mut holes,
        mut typed,
        mut formatted,
        mut raw,
        mut projections,
        mut branches,
        mut custom,
        mut heredoc,
        mut block,
        mut escapes,
    ) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    for _ in 0..200 {
        let mut g = Gen::new(rng.next_u64());
        let (src, shapes) = g.program();
        fn walk(
            shapes: &[Shape],
            holes: &mut i32,
            typed: &mut i32,
            formatted: &mut i32,
            raw: &mut i32,
            projections: &mut i32,
            branches: &mut i32,
        ) {
            for s in shapes {
                match s {
                    Shape::Static => {}
                    Shape::Hole {
                        typed: t,
                        formatted: f,
                        raw: r,
                    } => {
                        *holes += 1;
                        if *t {
                            *typed += 1
                        };
                        if *f {
                            *formatted += 1
                        };
                        if *r {
                            *raw += 1
                        }
                    }
                    Shape::Projection { body, .. } => {
                        *projections += 1;
                        walk(body, holes, typed, formatted, raw, projections, branches)
                    }
                    Shape::Branch {
                        then, otherwise, ..
                    } => {
                        *branches += 1;
                        walk(then, holes, typed, formatted, raw, projections, branches);
                        walk(
                            otherwise,
                            holes,
                            typed,
                            formatted,
                            raw,
                            projections,
                            branches,
                        )
                    }
                }
            }
        }
        walk(
            &shapes,
            &mut holes,
            &mut typed,
            &mut formatted,
            &mut raw,
            &mut projections,
            &mut branches,
        );
        if src.contains("delims") || src.contains("sigil") {
            custom += 1;
        }
        if src.contains("<<<") {
            heredoc += 1;
        }
        if src.contains(":= {") {
            block += 1;
        }
        if src.contains("lit ") {
            escapes += 1;
        }
    }
    assert!(
        holes > 100
            && typed > 20
            && formatted > 20
            && raw > 10
            && projections > 20
            && branches > 20
            && custom > 50
            && heredoc > 20
            && block > 10
            && escapes > 5,
        "generator coverage too thin: holes={holes} typed={typed} formatted={formatted} raw={raw} projections={projections} branches={branches} custom={custom} heredoc={heredoc} block={block} escapes={escapes}"
    );
}

#[test]
#[ignore = "manual superfuzz — minutes of runtime; run with `-- --ignored`"]
fn superfuzz_tile_syntax() {
    let base = env_u64("FUZZ_SEED", 0x7113_5EED);
    let seeds = env_u64("SUPERFUZZ_SEEDS", 32);
    let iterations = env_u64("FUZZ_ITERATIONS", 2000) as usize;
    let mut all = Vec::new();
    for k in 0..seeds {
        let seed = base.wrapping_add(k);
        all.extend(run_wellformed_pass(seed, iterations));
        all.extend(run_mutation_pass(seed, iterations));
        if k % 8 == 7 {
            eprintln!(
                "superfuzz: {}/{seeds} seeds swept, {} violation(s) so far",
                k + 1,
                all.len()
            );
        }
    }
    let mut report = all.join("\n---\n");
    if report.len() > 30_000 {
        let mut cut = 30_000;
        while !report.is_char_boundary(cut) {
            cut -= 1;
        }
        report.truncate(cut);
        report.push_str("\n… (report truncated)");
    }
    assert!(
        all.is_empty(),
        "superfuzz invariants violated ({} failures):\n\n{report}",
        all.len()
    );
}
