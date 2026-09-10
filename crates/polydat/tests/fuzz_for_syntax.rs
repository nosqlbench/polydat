// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Fuzz coverage for the `for` construct (SRD 113 step 1).
//!
//! Two generators feed the front end:
//!
//! - **Well-formed programs.** A grammar-directed generator emits
//!   random `for` forms — producer bindings, inline traversals,
//!   traversals over producers, nesting, every body statement kind,
//!   `where` predicates with `{name}` placeholders, `order` specs with
//!   every strategy, zip groups, string and generator sources, comments
//!   and odd whitespace. Invariants: lex and parse succeed, the
//!   element names match what was emitted, the pretty-printer round
//!   trips to a fixed point, and the compiler lowers every form to a
//!   child program or producer entry, or errors in a sentence, and
//!   never panics.
//! - **Mutated programs.** Random byte-level edits of well-formed
//!   programs. Invariants: no stage panics, and every error is a
//!   sentence rather than a leaked backtrace.
//!
//! Seeds and iteration counts follow the other fuzzers:
//! `FUZZ_SEED` and `FUZZ_ITERATIONS`, with an `#[ignore]`d superfuzz
//! sweep over `SUPERFUZZ_SEEDS` seeds.

use polydat::dsl::ast::{Expr, ForSourceKind, PolydatFile, Statement};
use polydat::dsl::pprint::pp_file;

// ---------------------------------------------------------------------------
// PRNG (splitmix64, same as fuzz_type_adapters)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Well-formed program generator
// ---------------------------------------------------------------------------

/// A generated comprehension: its canonical text and the element
/// names it dispenses, in declaration order.
struct Comp {
    text: String,
    names: Vec<String>,
}

struct Gen {
    rng: Rng,
    counter: usize,
    /// Producer wires bound so far at the current scope depth, with
    /// their element names, usable by `for <producer> { ... }` and by
    /// derivations.
    producers: Vec<(String, Vec<String>)>,
}

const STRATEGIES: &[&str] = &[
    "lex",
    "reverse_lex",
    "diagonal",
    "antidiagonal",
    "extrema",
    "shells",
    "halton",
    "sobol",
    "lhs",
];
const CMP: &[&str] = &["==", "!=", "<", ">", "<=", ">="];
const WORDS: &[&str] = &["load", "verify", "read", "warm", "cold"];

impl Gen {
    fn new(seed: u64) -> Self {
        Gen {
            rng: Rng::new(seed),
            counter: 0,
            producers: Vec::new(),
        }
    }

    fn name(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{prefix}{}", self.counter)
    }

    /// One clause source in the text parser's surface.
    fn source(&mut self) -> String {
        match self.rng.range(7) {
            0 => {
                let lo = self.rng.range(5);
                format!("{lo}..{}", lo + 1 + self.rng.range(6))
            }
            1 => {
                let n = 2 + self.rng.range(4);
                (0..n)
                    .map(|i| (i * 10 + self.rng.range(10)).to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            }
            2 => {
                let n = 2 + self.rng.range(3);
                (0..n)
                    .map(|_| self.rng.pick(WORDS).to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            }
            3 => format!(
                "partitions(\"*/{}\", {})",
                2 + self.rng.range(6),
                100 * (1 + self.rng.range(1000))
            ),
            4 => format!(
                "partitions('{}%,{}%,*')",
                10 + self.rng.range(30),
                10 + self.rng.range(30)
            ),
            5 => format!("{}..{}", 1 + self.rng.range(3), 5 + self.rng.range(20)),
            _ => {
                let n = 2 + self.rng.range(3);
                (0..n)
                    .map(|i| format!("{}.{}", i, self.rng.range(10)))
                    .collect::<Vec<_>>()
                    .join(",")
            }
        }
    }

    fn comprehension(&mut self) -> Comp {
        let mut clauses = Vec::new();
        let mut names = Vec::new();
        let n = 1 + self.rng.range(3);
        for _ in 0..n {
            if self.rng.coin(15) {
                // Parallel-iter (zip) group.
                let k = 2 + self.rng.range(2);
                let vars: Vec<String> = (0..k).map(|_| self.name("z")).collect();
                let size = 2 + self.rng.range(4);
                let srcs: Vec<String> = (0..k).map(|_| format!("0..{size}")).collect();
                let rhs = match self.rng.range(3) {
                    0 => format!("({})", srcs.join(", ")),
                    1 => format!("zip_truncate({})", srcs.join(", ")),
                    _ => format!("zip_cycle({})", srcs.join(", ")),
                };
                clauses.push(format!("({}) in {rhs}", vars.join(", ")));
                names.extend(vars);
            } else {
                let v = self.name("k");
                let src = self.source();
                clauses.push(format!("{v} in {src}"));
                names.push(v);
            }
        }
        let mut text = clauses.join(if self.rng.coin(20) { " , " } else { ", " });
        if self.rng.coin(40) {
            let a = names[self.rng.range(names.len())].clone();
            let cmp = self.rng.pick(CMP);
            let lit = self.rng.range(50);
            let mut pred = format!("{{{a}}} {cmp} {lit}");
            if self.rng.coin(40) {
                let b = names[self.rng.range(names.len())].clone();
                let join = if self.rng.coin(50) { "&&" } else { "||" };
                pred = format!(
                    "{pred} {join} {{{b}}} {} {}",
                    self.rng.pick(CMP),
                    self.rng.range(50)
                );
            }
            text.push_str(&format!(" where {pred}"));
        }
        if self.rng.coin(40) {
            let strat = self.rng.pick(STRATEGIES);
            if self.rng.coin(50) {
                text.push_str(&format!(" order {strat}/{}", 1 + self.rng.range(9)));
            } else {
                text.push_str(&format!(" order {strat}"));
            }
        }
        Comp { text, names }
    }

    /// A random body statement that references the scope's element
    /// names. `depth` bounds nesting of further `for` blocks.
    fn body_statement(
        &mut self,
        names: &[String],
        depth: usize,
        out: &mut Vec<Expected>,
    ) -> String {
        let pick = |g: &mut Gen| -> String {
            if names.is_empty() || g.rng.coin(30) {
                "cycle".into()
            } else {
                names[g.rng.range(names.len())].clone()
            }
        };
        match self.rng.range(9) {
            0 | 1 => {
                let n = self.name("v");
                let a = pick(self);
                format!("{n} := hash({a})")
            }
            2 => {
                let n = self.name("s");
                let a = pick(self);
                let b = pick(self);
                format!("{n} := \"{{{a}}}-{{{b}}}\"")
            }
            3 => format!("extern e{} : u64 = {}", self.counter, self.rng.range(1000)),
            4 => format!("const c{} := {}", self.counter, self.rng.range(1000)),
            5 => {
                let a = pick(self);
                format!(
                    "cursor rows{} = range(0, {}) over {a}",
                    self.counter,
                    100 + self.rng.range(1000)
                )
            }
            6 => {
                // Producer binding inside a body.
                let c = self.comprehension();
                let n = self.name("prod");
                out.push(Expected::Producer {
                    names: c.names.clone(),
                });
                self.producers.push((n.clone(), c.names.clone()));
                format!("{n} := for {}", c.text)
            }
            7 if depth > 0 => self.traversal(depth - 1, out),
            _ => {
                let n = self.name("w");
                let a = pick(self);
                format!("{n} := mod({a}, {})", 1 + self.rng.range(100))
            }
        }
    }

    /// A `for` traversal statement with a random body.
    fn traversal(&mut self, depth: usize, out: &mut Vec<Expected>) -> String {
        let use_producer = !self.producers.is_empty() && self.rng.coin(30);
        let (head, names) = if use_producer {
            let (p, names) = self.producers[self.rng.range(self.producers.len())].clone();
            out.push(Expected::TraversalOverProducer {
                producer: p.clone(),
            });
            if self.rng.coin(40) {
                // A derived head: `for base where ... order ... {`.
                (self.derivation_text(&p, &names), names)
            } else {
                (p, names)
            }
        } else {
            let c = self.comprehension();
            out.push(Expected::Traversal {
                names: c.names.clone(),
            });
            (c.text, c.names)
        };
        let n = 1 + self.rng.range(4);
        let saved = self.producers.len();
        let mut body = Vec::new();
        for _ in 0..n {
            body.push(self.body_statement(&names, depth, out));
        }
        self.producers.truncate(saved);
        let comment = if self.rng.coin(25) {
            " // traverse"
        } else {
            ""
        };
        let brace = if self.rng.coin(20) { "{" } else { " {" };
        format!(
            "for {head}{brace}{comment}\n{}\n}}",
            body.iter()
                .map(|s| format!("    {s}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    /// `base where <pred>`, `base order <spec>`, or both, over a
    /// producer with the given element names.
    fn derivation_text(&mut self, base: &str, names: &[String]) -> String {
        let mut text = base.to_string();
        let which = self.rng.range(3);
        if which != 1 && !names.is_empty() {
            let a = names[self.rng.range(names.len())].clone();
            text.push_str(&format!(
                " where {{{a}}} {} {}",
                self.rng.pick(CMP),
                self.rng.range(50)
            ));
        }
        if which != 0 || names.is_empty() {
            let strat = self.rng.pick(STRATEGIES);
            if self.rng.coin(50) {
                text.push_str(&format!(" order {strat}/{}", 1 + self.rng.range(9)));
            } else {
                text.push_str(&format!(" order {strat}"));
            }
        }
        text
    }

    /// A whole program: an input, a few top-level statements, one to
    /// three `for` forms.
    fn program(&mut self) -> (String, Vec<Expected>) {
        let mut out = Vec::new();
        let mut lines = vec!["input cycle: u64".to_string()];
        if self.rng.coin(50) {
            lines.push(format!("extern seed: u64 = {}", self.rng.range(100)));
        }
        let n = 1 + self.rng.range(3);
        for _ in 0..n {
            match self.rng.range(3) {
                0 => {
                    let c = self.comprehension();
                    let name = self.name("sweep");
                    out.push(Expected::Producer {
                        names: c.names.clone(),
                    });
                    self.producers.push((name.clone(), c.names.clone()));
                    let comment = if self.rng.coin(25) { "   # sweep" } else { "" };
                    lines.push(format!("{name} := for {}{comment}", c.text));
                    if self.rng.coin(50) {
                        // A derived producer over the one just bound. Its
                        // element names resolve at compile time, so the
                        // parsed form reports none.
                        let derived = self.name("derived");
                        let text = self.derivation_text(&name, &c.names);
                        out.push(Expected::Producer { names: Vec::new() });
                        self.producers.push((derived.clone(), c.names.clone()));
                        lines.push(format!("{derived} := for {text}"));
                    }
                }
                _ => {
                    let depth = self.rng.range(3);
                    let stmt = self.traversal(depth, &mut out);
                    lines.push(stmt);
                }
            }
            if self.rng.coin(30) {
                let plain = self.name("plain");
                lines.push(format!("{plain} := hash(cycle)"));
            }
        }
        (lines.join("\n") + "\n", out)
    }
}

/// What the generator emitted, in document order, for checking the
/// parsed AST against.
#[derive(Debug)]
enum Expected {
    Producer { names: Vec<String> },
    Traversal { names: Vec<String> },
    TraversalOverProducer { producer: String },
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

fn parse(src: &str) -> Result<PolydatFile, String> {
    let tokens = polydat::dsl::lexer::lex(src)?;
    polydat::dsl::parser::parse(tokens)
}

/// Collect every `for` form in document order.
fn collect_for_forms(stmts: &[Statement], out: &mut Vec<Expected>) {
    for s in stmts {
        match s {
            Statement::Binding(b) => {
                if let Expr::For(src) = &b.value {
                    out.push(Expected::Producer {
                        names: src.element_names(),
                    });
                }
            }
            Statement::For(f) => {
                match &f.source.kind {
                    ForSourceKind::Producer(p) => out.push(Expected::TraversalOverProducer {
                        producer: p.clone(),
                    }),
                    ForSourceKind::Derived { base, .. } => {
                        out.push(Expected::TraversalOverProducer {
                            producer: base.clone(),
                        })
                    }
                    ForSourceKind::Comprehension(_) => out.push(Expected::Traversal {
                        names: f.source.element_names(),
                    }),
                }
                collect_for_forms(&f.body, out);
            }
            _ => {}
        }
    }
}

fn same(a: &Expected, b: &Expected) -> bool {
    match (a, b) {
        (Expected::Producer { names: x }, Expected::Producer { names: y }) => x == y,
        (Expected::Traversal { names: x }, Expected::Traversal { names: y }) => x == y,
        (
            Expected::TraversalOverProducer { producer: x },
            Expected::TraversalOverProducer { producer: y },
        ) => x == y,
        _ => false,
    }
}

fn panic_text(p: &Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "<non-string panic>".into())
}

/// Count traversals and producers reachable from a program, recursing
/// into child programs.
fn count_lowered(p: &polydat::kernel::PolydatProgram, n: &mut usize) {
    *n += p.producers().len();
    for t in p.traversals() {
        *n += 1;
        count_lowered(&t.program, n);
    }
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
            "reproduce: FUZZ_SEED={seed} FUZZ_ITERATIONS={} cargo test -p polydat --test fuzz_for_syntax wellformed",
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

        // Invariant 1: the front end accepts what the grammar generator
        // produced, without panicking.
        let parsed = match std::panic::catch_unwind(|| parse(&source)) {
            Ok(Ok(f)) => f,
            Ok(Err(e)) => {
                failures.push(format!("[seed {seed:#x}] iteration {i}: parse rejected a well-formed program:\n  {e}\n  source:\n{source}\n  {}", repro(i)));
                continue;
            }
            Err(p) => {
                failures.push(format!("[seed {seed:#x}] iteration {i}: front end panicked: {}\n  source:\n{source}\n  {}", panic_text(&p), repro(i)));
                continue;
            }
        };

        // Invariant 2: every for form is present, in order, with the
        // element names the comprehension declared.
        let mut got = Vec::new();
        collect_for_forms(&parsed.statements, &mut got);
        if got.len() != expected.len() || !got.iter().zip(&expected).all(|(a, b)| same(a, b)) {
            failures.push(format!("[seed {seed:#x}] iteration {i}: for forms differ.\n  expected: {expected:?}\n  got:      {got:?}\n  source:\n{source}\n  {}", repro(i)));
            continue;
        }

        // Invariant 3: the pretty-printer reaches a fixed point in one
        // step, and its output parses to the same for forms.
        let printed = pp_file(&parsed);
        match parse(&printed) {
            Ok(again) => {
                let reprinted = pp_file(&again);
                if reprinted != printed {
                    failures.push(format!("[seed {seed:#x}] iteration {i}: pretty-printer is not a fixed point.\n  first:\n{printed}\n  second:\n{reprinted}\n  {}", repro(i)));
                    continue;
                }
                let mut got2 = Vec::new();
                collect_for_forms(&again.statements, &mut got2);
                if got2.len() != got.len() || !got2.iter().zip(&got).all(|(a, b)| same(a, b)) {
                    failures.push(format!("[seed {seed:#x}] iteration {i}: reparse of printed form changed the for forms.\n  printed:\n{printed}\n  {}", repro(i)));
                    continue;
                }
            }
            Err(e) => {
                failures.push(format!("[seed {seed:#x}] iteration {i}: printed form does not parse: {e}\n  printed:\n{printed}\n  {}", repro(i)));
                continue;
            }
        }

        // Invariant 4: the compiler never panics. It either lowers every
        // for form (one child program per traversal, one producer entry
        // per binding) or returns a sentence naming the offending form.
        match std::panic::catch_unwind(|| polydat::dsl::compile_polydat(&source)) {
            Ok(Ok(k)) => {
                let mut lowered = 0;
                count_lowered(k.program(), &mut lowered);
                if lowered != expected.len() {
                    failures.push(format!("[seed {seed:#x}] iteration {i}: compiled program lowered {lowered} for forms but the source has {}.\n  source:\n{source}\n  {}", expected.len(), repro(i)));
                }
            }
            Ok(Err(e)) => {
                if cryptic(&e) || !e.contains("for ") {
                    failures.push(format!("[seed {seed:#x}] iteration {i}: compiler error is cryptic or does not name the for form: {e}\n  source:\n{source}\n  {}", repro(i)));
                }
            }
            Err(p) => failures.push(format!(
                "[seed {seed:#x}] iteration {i}: compiler panicked: {}\n  source:\n{source}\n  {}",
                panic_text(&p),
                repro(i)
            )),
        }

        // Invariant 5: a compiled program's traversals activate and cycle
        // without panicking, and two hosts of the same source produce the
        // same trace (T1). Bounded so a wide comprehension stays cheap.
        let run = || -> Result<Vec<String>, String> {
            let mut k = polydat::dsl::compile_polydat(&source)?;
            k.set_inputs(&[3]);
            let mut trace = Vec::new();
            run_traversals_bounded(&mut k, 0, &mut trace)?;
            Ok(trace)
        };
        let first = std::panic::catch_unwind(run);
        match first {
            Err(p) => failures.push(format!("[seed {seed:#x}] iteration {i}: traversal runtime panicked: {}\n  source:\n{source}\n  {}", panic_text(&p), repro(i))),
            Ok(Err(e)) => {
                if cryptic(&e) {
                    failures.push(format!("[seed {seed:#x}] iteration {i}: cryptic runtime error: {e}\n  source:\n{source}\n  {}", repro(i)));
                }
            }
            Ok(Ok(trace_a)) => {
                if let Ok(Ok(trace_b)) = std::panic::catch_unwind(run)
                    && trace_a != trace_b
                {
                    failures.push(format!("[seed {seed:#x}] iteration {i}: T1 violated, two hosts diverged.\n  source:\n{source}\n  {}", repro(i)));
                }
            }
        }
    }
    failures
}

/// Activate every traversal of `k` (recursing into activations' own
/// traversals), running at most a few activations and cycles of each,
/// and append every pulled output's display form to `trace`.
fn run_traversals_bounded(
    k: &mut polydat::kernel::PolydatKernel,
    depth: usize,
    trace: &mut Vec<String>,
) -> Result<(), String> {
    const MAX_ACTIVATIONS: usize = 6;
    const MAX_CYCLES: u64 = 3;
    if depth > 4 {
        return Ok(());
    }
    let n = k.program().traversals().len();
    for t in 0..n {
        let stream = k.traverse(t)?;
        let outputs: Vec<String> = stream
            .traversal()
            .program
            .own_output_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        for a in 0..stream.len().min(MAX_ACTIVATIONS) {
            let mut act = stream.activation(a)?;
            for c in 0..act.cycle_count().min(MAX_CYCLES) {
                let kernel = act.cycle(c);
                for name in &outputs {
                    trace.push(format!(
                        "{t}/{a}/{c} {name}={}",
                        kernel.pull(name).to_display_string()
                    ));
                }
            }
            act.cycle(0);
            run_traversals_bounded(&mut act.kernel, depth + 1, trace)?;
        }
    }
    Ok(())
}

/// Byte-level mutations of well-formed programs.
fn mutate(rng: &mut Rng, src: &str) -> String {
    let mut chars: Vec<char> = src.chars().collect();
    let edits = 1 + rng.range(4);
    const INSERT: &[char] = &[
        '{', '}', '(', ')', '[', ']', ',', '"', '\'', '.', 'f', 'o', 'r', ' ', '\n', '=', ':', '/',
        '#', '<', '&', '|', '_', '0',
    ];
    for _ in 0..edits {
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
                // Duplicate a slice: exercises repeated `for` and unbalanced braces.
                let end = (at + 1 + rng.range(12)).min(chars.len());
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
            "reproduce: FUZZ_SEED={seed} FUZZ_ITERATIONS={} cargo test -p polydat --test fuzz_for_syntax mutated",
            i + 1
        )
    };
    let mut rng = Rng::new(seed ^ 0x5EED_F0F0);
    for i in 0..iterations {
        if failures.len() >= MAX {
            failures.push(format!("[seed {seed:#x}] … stopping after {MAX} failures"));
            break;
        }
        let mut g = Gen::new(rng.next_u64());
        let (clean, _) = g.program();
        let source = mutate(&mut rng, &clean);

        // Invariant: no stage panics, and every error is a sentence.
        let outcome = std::panic::catch_unwind(|| {
            let parsed = parse(&source)?;
            let printed = pp_file(&parsed);
            // A parse that succeeds must print to something that parses.
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
            Ok(Ok(k)) => {
                // A mutant that compiles must have lowered exactly the
                // for forms it still contains.
                if let Ok(f) = parse(&source) {
                    let mut forms = Vec::new();
                    collect_for_forms(&f.statements, &mut forms);
                    let mut lowered = 0;
                    count_lowered(k.program(), &mut lowered);
                    if lowered != forms.len() {
                        failures.push(format!("[seed {seed:#x}] iteration {i}: mutant has {} for forms but {lowered} were lowered.\n  source:\n{source}\n  {}", forms.len(), repro(i)));
                    }
                }
            }
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
fn wellformed_for_programs_parse_print_and_are_declined() {
    let seed = env_u64("FUZZ_SEED", 0xF0A_5EED);
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
fn mutated_for_programs_never_panic() {
    let seed = env_u64("FUZZ_SEED", 0xF0A_5EED);
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
    // The generator itself must produce every form across a modest
    // sample, or the fuzz above is weaker than it looks.
    let mut rng = Rng::new(7);
    let (
        mut producers,
        mut traversals,
        mut over_producer,
        mut nested,
        mut wheres,
        mut orders,
        mut zips,
    ) = (0, 0, 0, 0, 0, 0, 0);
    let mut derived = 0;
    for _ in 0..200 {
        let mut g = Gen::new(rng.next_u64());
        let (src, expected) = g.program();
        for e in &expected {
            match e {
                Expected::Producer { .. } => producers += 1,
                Expected::Traversal { .. } => traversals += 1,
                Expected::TraversalOverProducer { .. } => over_producer += 1,
            }
        }
        if src.matches("for ").count() > 1 && src.contains("    for ") {
            nested += 1;
        }
        if src.contains(" where ") {
            wheres += 1;
        }
        if src.contains(" order ") {
            orders += 1;
        }
        if src.contains("zip_") || src.contains(") in (") {
            zips += 1;
        }
        if src.contains(":= for sweep")
            || src.contains("for sweep") && (src.contains(" where ") || src.contains(" order "))
        {
            derived += 1;
        }
    }
    assert!(
        producers > 20
            && traversals > 20
            && over_producer > 5
            && nested > 5
            && wheres > 20
            && orders > 20
            && zips > 5
            && derived > 5,
        "generator coverage too thin: producers={producers} traversals={traversals} over_producer={over_producer} nested={nested} wheres={wheres} orders={orders} zips={zips} derived={derived}"
    );
}

/// Manual deep sweep. Run with `-- --ignored`.
#[test]
#[ignore = "manual superfuzz — minutes of runtime; run with `-- --ignored`"]
fn superfuzz_for_syntax() {
    let base = env_u64("FUZZ_SEED", 0xF0A_5EED);
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
