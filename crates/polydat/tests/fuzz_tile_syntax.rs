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
//! `FUZZ_SEED` and `FUZZ_ITERATIONS` follow the other fuzzers.
//!
//! The ignored `superfuzz_tile_syntax` enumerates the generator's
//! dimensions instead of drawing them: the six delimiter pairs, the
//! four sigils, strict and instring each on or off, the four
//! encodings, the three body forms, the directive nesting depth 0 to
//! 2, projection or branch as the nesting directive, the four
//! comprehension shapes, a separator or none, an else or none, the
//! nine hole expressions, no type or each of six, no format or each of
//! six, raw or not, the eight static words, and a doubled-open escape
//! or none. A combination fixes each of those choices everywhere the
//! generator makes it, and the tile nests to exactly the planned
//! depth; the remaining choices (extra static words, piece order,
//! literals) come from the combination's seed. The random generator's
//! escape is stripped with the open delimiter before it reaches the
//! output, so only the planned generator emits it. Each combination's
//! program goes through the well-formed invariants, the engine sweep,
//! and one mutation.
//!
//! The combinations run in the stratified order of `common::superfuzz`,
//! which reaches every value of every dimension, depths included,
//! within the first nine combinations. The run stops at 1000
//! combinations or 60 seconds, whichever comes first;
//! `SUPERFUZZ_MAX_COMBINATIONS` and `SUPERFUZZ_MAX_SECONDS` change the
//! limits (0 keeps the default, a large number lifts the limit).
//! `SUPERFUZZ_SEEDS` is the number of seeds run per combination
//! (default 1), `FUZZ_SEED` is the base seed, and `FUZZ_ENGINE_SWEEP`
//! is the stride over combinations of the engine sweep (default 1,
//! every combination; 0 turns it off).

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

/// Tile encodings: none, and the three the header names.
const ENCODINGS: &[Option<&str>] = &[None, Some("json"), Some("text"), Some("csv")];
/// Body forms: a brace block, a heredoc, a quoted string.
const BODY_FORMS: &[&str] = &["block", "heredoc", "string"];
/// Comprehension shapes a projection draws from.
const COMPREHENSIONS: usize = 4;
/// The deepest directive nesting the generator emits.
const MAX_DEPTH: usize = 2;
/// Directives that nest: a projection and a branch.
const DIRECTIVES: &[&str] = &["projection", "branch"];

struct Gen {
    rng: Rng,
    opts: TileOptions,
    encoding: Option<&'static str>,
    /// The superfuzz's fixed choices, or `None` for the random
    /// generator.
    plan: Option<TilePlan>,
}

/// One combination of the superfuzz's dimensions. Each field fixes a
/// choice the random generator otherwise draws, everywhere it draws
/// it.
#[derive(Clone, Copy, Debug)]
struct TilePlan {
    /// Index into [`DELIMS`].
    delims: usize,
    /// Index into [`SIGILS`].
    sigil: usize,
    strict: bool,
    in_string: bool,
    /// Index into [`ENCODINGS`].
    encoding: usize,
    /// Index into [`BODY_FORMS`].
    body_form: usize,
    /// Directive nesting, `0..=MAX_DEPTH`; the tile nests exactly this
    /// deep.
    depth: usize,
    /// Index into [`DIRECTIVES`], the directive on the nesting spine.
    directive: usize,
    /// Comprehension shape, `0..COMPREHENSIONS`.
    comprehension: usize,
    sep: bool,
    has_else: bool,
    /// Index into [`EXPRS`].
    expr: usize,
    /// 0 untyped, else `TYPES[ty - 1]`.
    ty: usize,
    /// 0 unformatted, else `FORMATS[format - 1]`.
    format: usize,
    raw: bool,
    /// Index into [`STATIC_WORDS`], the first word of every static.
    word: usize,
    /// Whether every static carries a doubled-open escape.
    escape: bool,
}

impl Gen {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let (open, close) = DELIMS[rng.range(DELIMS.len())];
        let sigil = SIGILS[rng.range(SIGILS.len())];
        let strict = rng.coin(20);
        let encoding = ENCODINGS[rng.range(ENCODINGS.len())];
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
            plan: None,
        }
    }

    fn planned(seed: u64, plan: TilePlan) -> Self {
        let (open, close) = DELIMS[plan.delims];
        Gen {
            rng: Rng::new(seed),
            opts: TileOptions {
                open: open.into(),
                close: close.into(),
                sigil: SIGILS[plan.sigil].into(),
                strict: plan.strict,
                in_string: plan.in_string,
            },
            encoding: ENCODINGS[plan.encoding],
            plan: Some(plan),
        }
    }

    /// Static text that never contains the open delimiter, the sigil
    /// followed by a directive word, or a bare brace that would break a
    /// block body; braces are allowed only balanced.
    ///
    /// The random generator appends its doubled-open escape before the
    /// open delimiter is stripped, so the escape never reaches its
    /// output. A planned static strips first and appends the escape
    /// after, so the escape is present whenever the plan asks for it.
    fn static_text(&mut self) -> String {
        let mut s = String::new();
        let n = 1 + self.rng.range(3);
        for i in 0..n {
            let w = match self.plan {
                Some(p) if i == 0 => STATIC_WORDS[p.word],
                _ => self.rng.pick(STATIC_WORDS),
            };
            s.push_str(w);
            s.push(' ');
        }
        let strip = |s: &str, opts: &TileOptions| {
            s.replace(&opts.open, "")
                .replace(&format!("{}for", opts.sigil), "")
                .replace(&format!("{}if", opts.sigil), "")
        };
        if let Some(p) = self.plan {
            let mut out = strip(&s, &self.opts);
            // A block body is captured by balancing its brackets
            // (polytile.md §2.2), so a doubled open delimiter that
            // holds a bracket would unbalance it; the escape of such a
            // delimiter goes only in heredoc and string bodies.
            let bracketed = self.opts.open.contains(['{', '[', '}', ']']);
            if p.escape && !(BODY_FORMS[p.body_form] == "block" && bracketed) {
                out.push_str(&format!("{}{}lit ", self.opts.open, self.opts.open));
            }
            return out;
        }
        // Add a literal open delimiter via the doubled escape sometimes.
        if self.rng.coin(15) {
            s.push_str(&format!("{}{}", self.opts.open, self.opts.open));
            s.push_str("lit ");
        }
        strip(&s, &self.opts) + if s.is_empty() { "z" } else { "" }
    }

    fn hole(&mut self) -> (String, Shape) {
        let (expr, ty, format, raw) = match self.plan {
            Some(p) => (
                EXPRS[p.expr],
                (p.ty > 0).then(|| TYPES[p.ty - 1]),
                (p.format > 0).then(|| FORMATS[p.format - 1]),
                p.raw,
            ),
            None => {
                let expr = self.rng.pick(EXPRS);
                let typed = self.rng.coin(30);
                let formatted = self.rng.coin(30);
                let raw = self.rng.coin(20);
                let ty = typed.then(|| self.rng.pick(TYPES));
                let format = formatted.then(|| self.rng.pick(FORMATS));
                (expr, ty, format, raw)
            }
        };
        let (typed, formatted) = (ty.is_some(), format.is_some());
        let mut inner = expr.to_string();
        if let Some(ty) = ty {
            inner.push_str(&format!(": {ty}"));
        }
        if let Some(format) = format {
            inner.push_str(&format!(" | {format}"));
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
        let kind = match self.plan {
            Some(p) => p.comprehension,
            None => self.rng.range(COMPREHENSIONS),
        };
        match kind {
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
                4 => self.projection(depth - 1, &mut text, &mut shapes),
                _ => self.branch(depth - 1, &mut text, &mut shapes),
            }
        }
        (text, shapes)
    }

    /// Nested pieces: the planned ones under a plan, else random.
    fn body(&mut self, depth: usize) -> (String, Vec<Shape>) {
        if self.plan.is_some() {
            self.planned_pieces(depth)
        } else {
            self.pieces(depth)
        }
    }

    fn projection(&mut self, depth: usize, text: &mut String, shapes: &mut Vec<Shape>) {
        let comp = self.comprehension();
        let sep = match self.plan {
            Some(p) => p.sep,
            None => self.rng.coin(40),
        };
        let (body, body_shapes) = self.body(depth);
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

    fn branch(&mut self, depth: usize, text: &mut String, shapes: &mut Vec<Shape>) {
        let (then, then_shapes) = self.body(depth);
        let has_else = match self.plan {
            Some(p) => p.has_else,
            None => self.rng.coin(50),
        };
        let (otherwise, else_shapes) = if has_else {
            self.body(depth)
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

    /// The planned pieces `depth` levels above the innermost: a static
    /// and a hole, and while `depth` allows, the planned directive over
    /// the next level, in a rotation the seed picks. Every directive
    /// body is planned in turn, so the tile nests exactly `depth` deep.
    fn planned_pieces(&mut self, depth: usize) -> (String, Vec<Shape>) {
        let directive = self.plan.map_or(0, |p| p.directive);
        let slots = if depth > 0 { 3 } else { 2 };
        let first = self.rng.range(slots);
        let mut text = String::new();
        let mut shapes = Vec::new();
        for k in 0..slots {
            match (first + k) % slots {
                0 => {
                    text.push_str(&self.static_text());
                    shapes.push(Shape::Static);
                }
                1 => {
                    let (h, s) = self.hole();
                    text.push_str(&h);
                    shapes.push(s);
                }
                _ if directive == 0 => self.projection(depth - 1, &mut text, &mut shapes),
                _ => self.branch(depth - 1, &mut text, &mut shapes),
            }
        }
        (text, shapes)
    }

    /// A whole program with one tile and a few plain statements.
    fn program(&mut self) -> (String, Vec<Shape>) {
        let (body, shapes) = match self.plan {
            Some(p) => self.planned_pieces(p.depth),
            None => self.pieces(MAX_DEPTH),
        };
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
        let form = match self.plan {
            Some(p) => p.body_form,
            None => self.rng.range(BODY_FORMS.len()),
        };
        let body_form = match form {
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
    // The engine sweep is sampled — it compiles the program once more
    // per tier, and a tile body is compiled in its own right on top of
    // that. `FUZZ_ENGINE_SWEEP` is the stride; 0 turns it off.
    let stride = env_u64("FUZZ_ENGINE_SWEEP", 8) as usize;
    let mut rng = Rng::new(seed);
    for i in 0..iterations {
        if failures.len() >= MAX {
            failures.push(format!("[seed {seed:#x}] … stopping after {MAX} failures"));
            break;
        }
        let mut g = Gen::new(rng.next_u64());
        let (source, expected) = g.program();
        let tag = format!("[seed {seed:#x}] iteration {}: ", i);
        let repro = format!(
            "reproduce: FUZZ_SEED={seed} FUZZ_ITERATIONS={} cargo test -p polydat --test suite fuzz_tile_syntax::wellformed",
            i + 1
        );
        let sweep = stride != 0 && i % stride == 0;
        failures.extend(check_wellformed(&source, &expected, sweep, &tag, &repro));
    }
    failures
}

/// The well-formed invariants over one tile program and the shapes the
/// generator emitted for it. `sweep` runs the engine sweep. Each
/// failure starts with `tag` and ends with `repro`.
fn check_wellformed(
    source: &str,
    expected: &[Shape],
    sweep: bool,
    tag: &str,
    repro: &str,
) -> Vec<String> {
    let mut failures = Vec::new();
    let parsed = match std::panic::catch_unwind(|| parse(source)) {
        Ok(Ok(f)) => f,
        Ok(Err(e)) => {
            failures.push(format!(
                "{tag}parse rejected a well-formed tile:\n  {e}\n  source:\n{source}\n  {}",
                repro
            ));
            return failures;
        }
        Err(p) => {
            failures.push(format!(
                "{tag}front end panicked: {}\n  source:\n{source}\n  {}",
                panic_text(&p),
                repro
            ));
            return failures;
        }
    };
    let Some(Statement::Tile(t)) = parsed
        .statements
        .iter()
        .find(|s| matches!(s, Statement::Tile(_)))
    else {
        failures.push(format!(
            "{tag}no tile statement parsed.\n  source:\n{source}\n  {}",
            repro
        ));
        return failures;
    };
    let got = strip_statics(&shapes_of(&t.pieces));
    let want = strip_statics(&normalize(expected));
    if got != want {
        failures.push(format!("{tag}pieces differ.\n  expected: {want:?}\n  got:      {got:?}\n  source:\n{source}\n  {}", repro));
        return failures;
    }
    // Printer fixed point.
    let printed = pp_file(&parsed);
    match parse(&printed) {
        Ok(again) => {
            if pp_file(&again) != printed {
                failures.push(format!(
                    "{tag}pretty-printer is not a fixed point.\n  printed:\n{printed}\n  {}",
                    repro
                ));
                return failures;
            }
        }
        Err(e) => {
            failures.push(format!(
                "{tag}printed form does not parse: {e}\n  printed:\n{printed}\n  {}",
                repro
            ));
            return failures;
        }
    }
    // Template renderer inverts the template parser.
    let span = polydat::dsl::lexer::Span { line: 1, col: 1 };
    let rendered = render_template(&t.pieces, &t.options);
    match parse_template(&rendered, &t.options, span) {
        Ok(pieces) => {
            if render_template(&pieces, &t.options) != rendered {
                failures.push(format!("{tag}render/parse of template is not a fixed point.\n  rendered:\n{rendered}\n  {}", repro));
                return failures;
            }
        }
        Err(e) => {
            failures.push(format!(
                "{tag}rendered template does not parse: {e}\n  rendered:\n{rendered}\n  {}",
                repro
            ));
            return failures;
        }
    }
    // The compiler either lowers the tile to a wire that renders, or
    // errors in a sentence naming the tile. It never panics. Holes
    // reference names the generator does not define, so most
    // programs take the error path; the ones that compile must
    // render without panicking.
    match std::panic::catch_unwind(|| {
        let mut k = polydat::dsl::compile_polydat(source).map_err(|e| e.to_string())?;
        k.set_inputs(&[1]);
        Ok::<String, String>(k.pull("t").to_display_string())
    }) {
        Ok(Ok(_)) => {
            // And the same text on every engine, when the caller
            // asks for the sweep.
            if sweep {
                for detail in engine_render_failures(source) {
                    failures.push(format!("{tag}{detail}\n  source:\n{source}\n  {repro}"));
                }
            }
        }
        Ok(Err(e)) => {
            if cryptic(&e)
                || !e.contains("tile 't'")
                    && !e.contains("unknown wire")
                    && !e.contains("unknown function")
            {
                failures.push(format!("{tag}compiler error is cryptic or does not name the tile: {e}\n  source:\n{source}\n  {}", repro));
            }
        }
        Err(p) => failures.push(format!(
            "{tag}compiler or renderer panicked: {}\n  source:\n{source}\n  {}",
            panic_text(&p),
            repro
        )),
    }
    failures
}

/// A tile renders the same on every engine.
///
/// The block above compiles once and throws the rendered string away —
/// it was asking whether the tile lowers and renders at all, on
/// whichever engine `compile_polydat` happens to build. A tile is a
/// projection body compiled in its own right, so which engine built it
/// is exactly the thing worth varying, and the rendered text is the
/// whole observable: one string per program, already the display form.
///
/// The interpreter with no cones is the oracle. A tier that declines
/// the program is skipped, and so is pure native's unset-extern trap,
/// which is the `None` rule holding rather than a disagreement
/// (engines.md §3.3 and §8's runtime exception).
fn engine_render_failures(source: &str) -> Vec<String> {
    use polydat::{Engine, JitMode, KernelError, Provenance};
    let mut out = Vec::new();

    let deterministic = match polydat::dsl::compile_polydat_interpreter(source) {
        Ok(k) => k.program().is_deterministic(),
        Err(_) => return out,
    };
    if !deterministic {
        return out;
    }

    let render_on = |engine: Engine| -> Result<Result<String, String>, String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut k = match polydat::dsl::compile_polydat_with(source, engine) {
                Ok(k) => k,
                Err(KernelError::Refused { .. }) => return Err("refused".to_string()),
                Err(e) => return Err(format!("compile: {e}")),
            };
            k.set_inputs(&[1]);
            Ok(k.pull("t").to_display_string())
        }))
        .map_err(|p| panic_text(&p))
    };

    let want = match render_on(Engine::Interpreter(JitMode::Off)) {
        Ok(Ok(s)) => s,
        _ => return out,
    };

    let mut engines = vec![
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Raw),
        Engine::Closures(Provenance::Auto),
    ];
    if cfg!(feature = "jit") {
        engines.push(Engine::Native(Provenance::Raw));
        engines.push(Engine::PureNative(Provenance::Auto));
    }

    for engine in engines {
        match render_on(engine) {
            Ok(Ok(got)) if got == want => {}
            Ok(Err(_)) => {}
            Ok(Ok(got)) => out.push(format!(
                "{engine} renders the tile differently from the interpreter:\n  \
                 interpreter: {want:?}\n  {engine}: {got:?}"
            )),
            // The `None` tripwire, not a disagreement (engines.md §3.3).
            Err(p) if p.contains("cannot carry a `None`") => {}
            Err(p) => out.push(format!(
                "{engine} panicked rendering a tile the interpreter rendered: {p}"
            )),
        }
    }
    out
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
    let mut rng = Rng::new(seed ^ 0x71E5_71E5);
    for i in 0..iterations {
        if failures.len() >= MAX {
            failures.push(format!("[seed {seed:#x}] … stopping after {MAX} failures"));
            break;
        }
        let mut g = Gen::new(rng.next_u64());
        let (clean, _) = g.program();
        let source = mutate(&mut rng, &clean);
        let tag = format!("[seed {seed:#x}] iteration {}: ", i);
        let repro = format!(
            "reproduce: FUZZ_SEED={seed} FUZZ_ITERATIONS={} cargo test -p polydat --test suite fuzz_tile_syntax::mutated",
            i + 1
        );
        failures.extend(check_mutant(&source, &tag, &repro));
    }
    failures
}

/// The mutant invariants over one damaged program. Each failure starts
/// with `tag` and ends with `repro`.
fn check_mutant(source: &str, tag: &str, repro: &str) -> Vec<String> {
    let mut failures = Vec::new();
    let outcome = std::panic::catch_unwind(|| {
        let parsed = parse(source)?;
        let printed = pp_file(&parsed);
        parse(&printed).map_err(|e| {
            format!("printed form of a parsed mutant does not parse: {e}\n  printed:\n{printed}")
        })?;
        polydat::dsl::compile_polydat(source).map_err(|e| e.to_string())
    });
    match outcome {
        Err(p) => failures.push(format!(
            "{tag}panic on mutant: {}\n  source:\n{source}\n  {}",
            panic_text(&p),
            repro
        )),
        Ok(Err(e)) => {
            if cryptic(&e) || e.to_string().contains("printed form of a parsed mutant") {
                failures.push(format!("{tag}{e}\n  source:\n{source}\n  {}", repro));
            }
        }
        Ok(Ok(_)) => {}
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

/// The dimensions of [`TilePlan`], in the order [`plan_of`] reads them.
fn tile_dimensions() -> Vec<(&'static str, usize)> {
    vec![
        ("delims", DELIMS.len()),
        ("sigil", SIGILS.len()),
        ("strict", 2),
        ("instring", 2),
        ("encoding", ENCODINGS.len()),
        ("body_form", BODY_FORMS.len()),
        ("depth", MAX_DEPTH + 1),
        ("directive", DIRECTIVES.len()),
        ("comprehension", COMPREHENSIONS),
        ("sep", 2),
        ("else", 2),
        ("expr", EXPRS.len()),
        ("type", TYPES.len() + 1),
        ("format", FORMATS.len() + 1),
        ("raw", 2),
        ("word", STATIC_WORDS.len()),
        ("escape", 2),
    ]
}

fn plan_of(v: &[usize]) -> TilePlan {
    TilePlan {
        delims: v[0],
        sigil: v[1],
        strict: v[2] == 1,
        in_string: v[3] == 1,
        encoding: v[4],
        body_form: v[5],
        depth: v[6],
        directive: v[7],
        comprehension: v[8],
        sep: v[9] == 1,
        has_else: v[10] == 1,
        expr: v[11],
        ty: v[12],
        format: v[13],
        raw: v[14] == 1,
        word: v[15],
        escape: v[16] == 1,
    }
}

#[test]
fn every_planned_form_parses_and_matches() {
    // One program per value of each dimension, through the parse and
    // shape invariants, so a plan the generator cannot honour fails
    // here rather than only in the manual superfuzz.
    let dims = tile_dimensions();
    let widest = dims.iter().map(|d| d.1).max().unwrap_or(1);
    let mut failures = Vec::new();
    for i in 0..widest {
        let v: Vec<usize> = dims.iter().map(|d| i % d.1).collect();
        let plan = plan_of(&v);
        let (source, expected) = Gen::planned(i as u64, plan).program();
        let tag = format!("{plan:?}: ");
        failures.extend(check_wellformed(&source, &expected, false, &tag, ""));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n---\n"));
}

/// MANUAL — every dimension of the generator under the shared budget:
///
/// ```text
/// cargo nextest run -p polydat --run-ignored only -E 'test(/superfuzz_tile_syntax/)' --no-capture
/// ```
#[test]
#[ignore = "manual superfuzz — up to a minute by default; run with `--run-ignored only`"]
fn superfuzz_tile_syntax() {
    use super::common::superfuzz;
    let base = env_u64("FUZZ_SEED", 0x7113_5EED);
    let seeds = superfuzz::seeds_per_combination();
    // Every combination sweeps the engines unless `FUZZ_ENGINE_SWEEP`
    // sets a stride over combinations; 0 turns the sweep off.
    let stride = env_u64("FUZZ_ENGINE_SWEEP", 1);
    superfuzz::run("superfuzz_tile_syntax", &tile_dimensions(), |v, index| {
        let plan = plan_of(v);
        let sweep = stride != 0 && index % stride == 0;
        let mut failures = Vec::new();
        for k in 0..seeds {
            let seed = superfuzz::seed_for(base, index, k);
            let (source, expected) = Gen::planned(seed, plan).program();
            let tag = format!("seed {seed:#x}: ");
            failures.extend(check_wellformed(&source, &expected, sweep, &tag, ""));
            let mutant = mutate(&mut Rng::new(seed ^ 0x71E5_71E5), &source);
            let tag = format!("seed {seed:#x} mutant: ");
            failures.extend(check_mutant(&mutant, &tag, ""));
        }
        failures
    });
}
