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
//! `FUZZ_SEED` and `FUZZ_ITERATIONS`.
//!
//! The `#[ignore]`d `superfuzz_for_syntax` enumerates the generator's
//! dimensions instead of drawing them: the top-level form (producer,
//! producer with a derived producer, traversal over a comprehension,
//! over a producer, over a derived head), the nesting depth 0 to 2,
//! the seven clause source kinds, the clause shape (single or one of
//! three zip forms), the `where` shape (none, one comparison, `&&`,
//! `||`), the comparison operator, the `order` spec (none, or each of
//! nine strategies bare or with `/n`), the derived head's clauses, the
//! seven body statement kinds, and plain or commented layout. A
//! combination fixes each of those choices everywhere the generator
//! makes it, and the traversal nests to exactly the planned depth; the
//! remaining choices (names, literals, extra statements) come from the
//! combination's seed. Each combination's program goes through the
//! well-formed invariants, the engine sweep, and one mutation.
//!
//! The combinations run in the stratified order of `common::superfuzz`,
//! which reaches every value of every dimension, depths included,
//! within the first 19 combinations. The run stops at 1000
//! combinations or 60 seconds, whichever comes first;
//! `SUPERFUZZ_MAX_COMBINATIONS` and `SUPERFUZZ_MAX_SECONDS` change the
//! limits (0 keeps the default, a large number lifts the limit).
//! `SUPERFUZZ_SEEDS` is the number of seeds run per combination
//! (default 1), `FUZZ_SEED` is the base seed, and `FUZZ_ENGINE_SWEEP`
//! is the stride over combinations of the engine sweep (default 1,
//! every combination; 0 turns it off).

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
    /// The superfuzz's fixed choices, or `None` for the random
    /// generator.
    plan: Option<ForPlan>,
}

/// One combination of the superfuzz's dimensions. Each field fixes a
/// choice the random generator otherwise draws, everywhere it draws
/// it, so the value is present in the program whenever the construct
/// is.
#[derive(Clone, Copy, Debug)]
struct ForPlan {
    /// Index into [`FORMS`].
    form: usize,
    /// Nesting of `for` blocks under the planned traversal,
    /// `0..=MAX_DEPTH`.
    depth: usize,
    /// Clause source kind, `0..SOURCE_KINDS`.
    source: usize,
    /// 0 a single variable, 1 a parenthesized zip group, 2
    /// `zip_truncate`, 3 `zip_cycle`.
    clause: usize,
    /// 0 no `where`, 1 one comparison, 2 two joined by `&&`, 3 by `||`.
    predicate: usize,
    /// Index into [`CMP`].
    cmp: usize,
    /// 0 no `order`, `1..=9` a bare strategy, `10..=18` a strategy with
    /// a `/n` parameter.
    order: usize,
    /// A derived head's clauses: 0 `where`, 1 `order`, 2 both.
    derive: usize,
    /// Index into [`BODY_KINDS`].
    body: usize,
    /// 0 plain layout, 1 comments, a tight brace, and a spaced comma.
    trivia: usize,
}

/// The planned top-level form: a producer binding, a producer and a
/// derived producer (each followed by a traversal over a
/// comprehension), a traversal over a comprehension, a traversal over
/// a producer, and a traversal over a derived head.
const FORMS: &[&str] = &[
    "producer",
    "producer+derived",
    "traversal",
    "traversal over producer",
    "traversal over derived head",
];
/// Clause source kinds [`Gen::source`] draws from.
const SOURCE_KINDS: usize = 7;
/// Clause shapes: a single variable and the three zip forms.
const CLAUSES: usize = 4;
/// `where` shapes: none, one comparison, `&&`, `||`.
const PREDICATES: usize = 4;
/// `order` shapes: none, each strategy bare, each with `/n`.
const ORDERS: usize = 1 + 2 * STRATEGIES.len();
/// Derived-head shapes: `where`, `order`, both.
const DERIVES: usize = 3;
/// The deepest `for` nesting the generator emits.
const MAX_DEPTH: usize = 2;
/// Body statement kinds, as arms of [`Gen::body_statement_kind`]: hash,
/// string, extern, const, cursor, producer binding, `mod`. A nested
/// traversal is the depth dimension.
const BODY_KINDS: &[usize] = &[0, 2, 3, 4, 5, 6, 8];

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
            plan: None,
        }
    }

    fn planned(seed: u64, plan: ForPlan) -> Self {
        Gen {
            plan: Some(plan),
            ..Gen::new(seed)
        }
    }

    fn name(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{prefix}{}", self.counter)
    }

    fn cmp(&mut self) -> &'static str {
        match self.plan {
            Some(p) => CMP[p.cmp],
            None => self.rng.pick(CMP),
        }
    }

    /// A trivia choice: the plan's, or a coin at `pct`.
    fn trivia(&mut self, pct: usize) -> bool {
        match self.plan {
            Some(p) => p.trivia == 1,
            None => self.rng.coin(pct),
        }
    }

    /// An `order` suffix. `always` asks for one even where the plan
    /// says none, as a derived head with an order clause does.
    fn order_suffix(&mut self, always: bool) -> Option<String> {
        let (strat, param) = match self.plan {
            Some(p) => match p.order {
                0 if always => (STRATEGIES[0], false),
                0 => return None,
                o if o <= STRATEGIES.len() => (STRATEGIES[o - 1], false),
                o => (STRATEGIES[o - 1 - STRATEGIES.len()], true),
            },
            None => {
                if !always && !self.rng.coin(40) {
                    return None;
                }
                let strat = self.rng.pick(STRATEGIES);
                (strat, self.rng.coin(50))
            }
        };
        Some(if param {
            format!(" order {strat}/{}", 1 + self.rng.range(9))
        } else {
            format!(" order {strat}")
        })
    }

    /// One clause source in the text parser's surface.
    fn source(&mut self) -> String {
        let kind = match self.plan {
            Some(p) => p.source,
            None => self.rng.range(SOURCE_KINDS),
        };
        match kind {
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
            let zip = match self.plan {
                Some(p) => p.clause > 0,
                None => self.rng.coin(15),
            };
            if zip {
                // Parallel-iter (zip) group.
                let k = 2 + self.rng.range(2);
                let vars: Vec<String> = (0..k).map(|_| self.name("z")).collect();
                let size = 2 + self.rng.range(4);
                let srcs: Vec<String> = (0..k).map(|_| format!("0..{size}")).collect();
                let form = match self.plan {
                    Some(p) => p.clause - 1,
                    None => self.rng.range(3),
                };
                let rhs = match form {
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
        let spaced = self.trivia(20);
        let mut text = clauses.join(if spaced { " , " } else { ", " });
        let has_where = match self.plan {
            Some(p) => p.predicate > 0,
            None => self.rng.coin(40),
        };
        if has_where {
            let a = names[self.rng.range(names.len())].clone();
            let cmp = self.cmp();
            let lit = self.rng.range(50);
            let mut pred = format!("{{{a}}} {cmp} {lit}");
            let compound = match self.plan {
                Some(p) => p.predicate >= 2,
                None => self.rng.coin(40),
            };
            if compound {
                let b = names[self.rng.range(names.len())].clone();
                let and = match self.plan {
                    Some(p) => p.predicate == 2,
                    None => self.rng.coin(50),
                };
                let join = if and { "&&" } else { "||" };
                let cmp = self.cmp();
                pred = format!("{pred} {join} {{{b}}} {cmp} {}", self.rng.range(50));
            }
            text.push_str(&format!(" where {pred}"));
        }
        if let Some(order) = self.order_suffix(false) {
            text.push_str(&order);
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
        let kind = self.rng.range(9);
        self.body_statement_kind(kind, names, depth, out)
    }

    /// The body statement of arm `kind`: 0 and 1 hash, 2 string, 3
    /// extern, 4 const, 5 cursor, 6 producer binding, 7 a nested
    /// traversal while `depth` allows, and `mod` otherwise.
    fn body_statement_kind(
        &mut self,
        kind: usize,
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
        match kind {
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
        self.traversal_body(head, &names, depth, out)
    }

    /// A traversal over the comprehension, producer, or derived head
    /// `head` with element names `names`. A planned body holds the
    /// plan's body statement, then a nested traversal while `depth`
    /// allows, so the planned depth is always reached, then up to two
    /// random statements.
    fn traversal_body(
        &mut self,
        head: String,
        names: &[String],
        depth: usize,
        out: &mut Vec<Expected>,
    ) -> String {
        let saved = self.producers.len();
        let mut body = Vec::new();
        match self.plan {
            Some(p) => {
                body.push(self.body_statement_kind(BODY_KINDS[p.body], names, depth, out));
                if depth > 0 {
                    body.push(self.traversal(depth - 1, out));
                }
                for _ in 0..self.rng.range(3) {
                    body.push(self.body_statement(names, depth, out));
                }
            }
            None => {
                let n = 1 + self.rng.range(4);
                for _ in 0..n {
                    body.push(self.body_statement(names, depth, out));
                }
            }
        }
        self.producers.truncate(saved);
        let comment = if self.trivia(25) { " // traverse" } else { "" };
        let brace = if self.trivia(20) { "{" } else { " {" };
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
        let which = match self.plan {
            Some(p) => p.derive,
            None => self.rng.range(DERIVES),
        };
        if which != 1 && !names.is_empty() {
            let a = names[self.rng.range(names.len())].clone();
            let cmp = self.cmp();
            text.push_str(&format!(" where {{{a}}} {cmp} {}", self.rng.range(50)));
        }
        if (which != 0 || names.is_empty())
            && let Some(order) = self.order_suffix(true)
        {
            text.push_str(&order);
        }
        text
    }

    /// A top-level producer binding, and a derived producer over it
    /// when `derived` says so (the plan's choice, or a coin).
    fn top_producer(
        &mut self,
        lines: &mut Vec<String>,
        out: &mut Vec<Expected>,
        derived: Option<bool>,
    ) -> (String, Vec<String>) {
        let c = self.comprehension();
        let name = self.name("sweep");
        out.push(Expected::Producer {
            names: c.names.clone(),
        });
        self.producers.push((name.clone(), c.names.clone()));
        let comment = if self.trivia(25) { "   # sweep" } else { "" };
        lines.push(format!("{name} := for {}{comment}", c.text));
        if derived.unwrap_or_else(|| self.rng.coin(50)) {
            // A derived producer over the one just bound. Its
            // element names resolve at compile time, so the
            // parsed form reports none.
            let derived = self.name("derived");
            let text = self.derivation_text(&name, &c.names);
            out.push(Expected::Producer { names: Vec::new() });
            self.producers.push((derived.clone(), c.names.clone()));
            lines.push(format!("{derived} := for {text}"));
        }
        (name, c.names)
    }

    /// The program of a plan: the planned form, whose traversal nests
    /// `depth` deep, then an optional plain binding.
    fn planned_program(&mut self, p: ForPlan) -> (String, Vec<Expected>) {
        let mut out = Vec::new();
        let mut lines = vec!["input cycle: u64".to_string()];
        if self.rng.coin(50) {
            lines.push(format!("extern seed: u64 = {}", self.rng.range(100)));
        }
        let comprehension_traversal = |g: &mut Gen, out: &mut Vec<Expected>| {
            let c = g.comprehension();
            out.push(Expected::Traversal {
                names: c.names.clone(),
            });
            g.traversal_body(c.text, &c.names, p.depth, out)
        };
        match p.form {
            0 | 1 => {
                self.top_producer(&mut lines, &mut out, Some(p.form == 1));
                let stmt = comprehension_traversal(self, &mut out);
                lines.push(stmt);
            }
            2 => {
                let stmt = comprehension_traversal(self, &mut out);
                lines.push(stmt);
            }
            _ => {
                let (name, names) = self.top_producer(&mut lines, &mut out, Some(false));
                out.push(Expected::TraversalOverProducer {
                    producer: name.clone(),
                });
                let head = if p.form == 4 {
                    self.derivation_text(&name, &names)
                } else {
                    name
                };
                let stmt = self.traversal_body(head, &names, p.depth, &mut out);
                lines.push(stmt);
            }
        }
        if self.rng.coin(30) {
            let plain = self.name("plain");
            lines.push(format!("{plain} := hash(cycle)"));
        }
        (lines.join("\n") + "\n", out)
    }

    /// A whole program: an input, a few top-level statements, one to
    /// three `for` forms. A planned generator makes the plan's program.
    fn program(&mut self) -> (String, Vec<Expected>) {
        if let Some(p) = self.plan {
            return self.planned_program(p);
        }
        let mut out = Vec::new();
        let mut lines = vec!["input cycle: u64".to_string()];
        if self.rng.coin(50) {
            lines.push(format!("extern seed: u64 = {}", self.rng.range(100)));
        }
        let n = 1 + self.rng.range(3);
        for _ in 0..n {
            match self.rng.range(3) {
                0 => {
                    self.top_producer(&mut lines, &mut out, None);
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
    // Invariant 6 is sampled — a sweep compiles the program once more
    // per tier, two of them through Cranelift, and a `for` body is a
    // child program compiled in its own right on top of that.
    // `FUZZ_ENGINE_SWEEP` is the stride; 0 turns it off, 1 sweeps every
    // program.
    let stride = env_u64("FUZZ_ENGINE_SWEEP", 8) as usize;
    let mut rng = Rng::new(seed);
    for i in 0..iterations {
        if failures.len() >= MAX {
            failures.push(format!("[seed {seed:#x}] … stopping after {MAX} failures"));
            break;
        }
        let mut g = Gen::new(rng.next_u64());
        let (source, expected) = g.program();
        let tag = format!("[seed {seed:#x}] iteration {i}: ");
        let repro = format!(
            "reproduce: FUZZ_SEED={seed} FUZZ_ITERATIONS={} cargo test -p polydat --test suite fuzz_for_syntax::wellformed",
            i + 1
        );
        let sweep = stride != 0 && i % stride == 0;
        failures.extend(check_wellformed(&source, &expected, sweep, &tag, &repro));
    }
    failures
}

/// Invariants 1 to 6 over one well-formed program and the `for` forms
/// the generator emitted for it. `sweep` runs invariant 6, the
/// engine sweep. Each failure starts with `tag` and ends with `repro`.
fn check_wellformed(
    source: &str,
    expected: &[Expected],
    sweep: bool,
    tag: &str,
    repro: &str,
) -> Vec<String> {
    let mut failures = Vec::new();
    // Invariant 1: the front end accepts what the grammar generator
    // produced, without panicking.
    let parsed = match std::panic::catch_unwind(|| parse(source)) {
        Ok(Ok(f)) => f,
        Ok(Err(e)) => {
            failures.push(format!(
                "{tag}parse rejected a well-formed program:\n  {e}\n  source:\n{source}\n  {}",
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

    // Invariant 2: every for form is present, in order, with the
    // element names the comprehension declared.
    let mut got = Vec::new();
    collect_for_forms(&parsed.statements, &mut got);
    if got.len() != expected.len() || !got.iter().zip(expected).all(|(a, b)| same(a, b)) {
        failures.push(format!("{tag}for forms differ.\n  expected: {expected:?}\n  got:      {got:?}\n  source:\n{source}\n  {}", repro));
        return failures;
    }

    // Invariant 3: the pretty-printer reaches a fixed point in one
    // step, and its output parses to the same for forms.
    let printed = pp_file(&parsed);
    match parse(&printed) {
        Ok(again) => {
            let reprinted = pp_file(&again);
            if reprinted != printed {
                failures.push(format!("{tag}pretty-printer is not a fixed point.\n  first:\n{printed}\n  second:\n{reprinted}\n  {}", repro));
                return failures;
            }
            let mut got2 = Vec::new();
            collect_for_forms(&again.statements, &mut got2);
            if got2.len() != got.len() || !got2.iter().zip(&got).all(|(a, b)| same(a, b)) {
                failures.push(format!("{tag}reparse of printed form changed the for forms.\n  printed:\n{printed}\n  {}", repro));
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

    // Invariant 4: the compiler never panics. It either lowers every
    // for form (one child program per traversal, one producer entry
    // per binding) or returns a sentence naming the offending form.
    match std::panic::catch_unwind(|| polydat::dsl::compile_polydat_interpreter(source)) {
        Ok(Ok(k)) => {
            let mut lowered = 0;
            count_lowered(k.program(), &mut lowered);
            if lowered != expected.len() {
                failures.push(format!("{tag}compiled program lowered {lowered} for forms but the source has {}.\n  source:\n{source}\n  {}", expected.len(), repro));
            }
        }
        Ok(Err(e)) => {
            if cryptic(&e.to_string()) || !e.to_string().contains("for ") {
                failures.push(format!("{tag}compiler error is cryptic or does not name the for form: {e}\n  source:\n{source}\n  {}", repro));
            }
        }
        Err(p) => failures.push(format!(
            "{tag}compiler panicked: {}\n  source:\n{source}\n  {}",
            panic_text(&p),
            repro
        )),
    }

    // Invariant 5: a compiled program's traversals activate and cycle
    // without panicking, and two hosts of the same source produce the
    // same trace (T1). Bounded so a wide comprehension stays cheap.
    let run = || -> Result<Vec<String>, String> {
        let mut k = polydat::dsl::compile_polydat_interpreter(source).map_err(|e| e.to_string())?;
        k.set_inputs(&[3]);
        let mut trace = Vec::new();
        run_traversals_bounded(&mut k, 0, &mut trace)?;
        Ok(trace)
    };
    let first = std::panic::catch_unwind(run);
    match first {
        Err(p) => failures.push(format!(
            "{tag}traversal runtime panicked: {}\n  source:\n{source}\n  {}",
            panic_text(&p),
            repro
        )),
        Ok(Err(e)) => {
            if cryptic(&e) {
                failures.push(format!(
                    "{tag}cryptic runtime error: {e}\n  source:\n{source}\n  {}",
                    repro
                ));
            }
        }
        Ok(Ok(trace_a)) => {
            if let Ok(Ok(trace_b)) = std::panic::catch_unwind(run)
                && trace_a != trace_b
            {
                failures.push(format!(
                    "{tag}T1 violated, two hosts diverged.\n  source:\n{source}\n  {}",
                    repro
                ));
            }
            // Invariant 6: and the same on every engine, when the
            // caller asks for the sweep.
            if sweep {
                for detail in engine_trace_failures(source) {
                    failures.push(format!("{tag}{detail}\n  source:\n{source}\n  {repro}"));
                }
            }
        }
    }
    failures
}

/// Invariant 6 — a composed program traverses the same on every engine.
///
/// Invariant 5 above runs one source twice and compares the traces,
/// which is T1: two hosts of the same program agree. Both runs were the
/// interpreter's, so nothing here ever asked a compiled tier what a
/// `for` does — and a traversal is where the engines have the most to
/// disagree about, since a body is a child program compiled in its own
/// right and an activation binds its coordinates through the same
/// kernel surface the parent uses.
///
/// The interpreter with no cones is the oracle; every other engine
/// walks the same bounded traversals and must produce the same trace.
/// A tier that declines the program says so (`KernelError::Refused`)
/// and is skipped; a tier that accepts it and answers differently is
/// the finding.
fn engine_trace_failures(source: &str) -> Vec<String> {
    use polydat::{Engine, JitMode, KernelError, Provenance};
    let mut out = Vec::new();

    // A program whose value is not a function of its inputs is not
    // comparable by value — `random`, a clock, a counter.
    let deterministic = match polydat::dsl::compile_polydat_interpreter(source) {
        Ok(k) => k.program().is_deterministic(),
        Err(_) => return out,
    };
    if !deterministic {
        return out;
    }

    let trace_on = |engine: Engine| -> Result<Result<Vec<String>, String>, String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut k = match polydat::dsl::compile_polydat_with(source, engine) {
                Ok(k) => k,
                Err(KernelError::Refused { .. }) => return Err("refused".to_string()),
                Err(e) => return Err(format!("compile: {e}")),
            };
            k.set_inputs(&[3]);
            let mut trace = Vec::new();
            run_traversals_bounded(k.as_mut(), 0, &mut trace)?;
            Ok(trace)
        }))
        .map_err(|p| panic_text(&p))
    };

    let want = match trace_on(Engine::Interpreter(JitMode::Off)) {
        Ok(Ok(t)) => t,
        // The oracle itself declined or failed; invariants 4 and 5
        // judged that, and there is nothing to compare against.
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
        match trace_on(engine) {
            Ok(Ok(got)) if got == want => {}
            // Declined, or failed the way the oracle would have.
            Ok(Err(_)) => {}
            Ok(Ok(got)) => out.push(format!(
                "I6: {engine} traverses differently from the interpreter:\n  \
                 interpreter: {want:?}\n  {engine}: {got:?}"
            )),
            // Pure native cannot carry a `None`, and one reaching it is
            // "a panic naming the extern, the tripwire of the rule,
            // never a wrong value" (engines.md §3.3). Whether an extern
            // is ever set is the host's, not the program's, so this
            // cannot be decided at build and refused there — the
            // tripwire *is* the decline, and the message says to run
            // the program on `native` instead. Not a finding: it is
            // the rule holding.
            Err(p) if p.contains("cannot carry a `None`") => {}
            Err(p) => out.push(format!(
                "I6: {engine} panicked traversing a program the interpreter walked: {p}"
            )),
        }
    }
    out
}

/// Activate every traversal of `k` (recursing into activations' own
/// traversals), running at most a few activations and cycles of each,
/// and append every pulled output's display form to `trace`.
fn run_traversals_bounded(
    k: &mut dyn polydat::Kernel,
    depth: usize,
    trace: &mut Vec<String>,
) -> Result<(), String> {
    const MAX_ACTIVATIONS: usize = 6;
    const MAX_CYCLES: u64 = 3;
    if depth > 4 {
        return Ok(());
    }
    let n = k.traversals().len();
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
            run_traversals_bounded(act.kernel.as_mut(), depth + 1, trace)?;
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
    let mut rng = Rng::new(seed ^ 0x5EED_F0F0);
    for i in 0..iterations {
        if failures.len() >= MAX {
            failures.push(format!("[seed {seed:#x}] … stopping after {MAX} failures"));
            break;
        }
        let mut g = Gen::new(rng.next_u64());
        let (clean, _) = g.program();
        let source = mutate(&mut rng, &clean);
        let tag = format!("[seed {seed:#x}] iteration {i}: ");
        let repro = format!(
            "reproduce: FUZZ_SEED={seed} FUZZ_ITERATIONS={} cargo test -p polydat --test suite fuzz_for_syntax::mutated",
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
    // Invariant: no stage panics, and every error is a sentence.
    let outcome = std::panic::catch_unwind(|| {
        let parsed = parse(source)?;
        let printed = pp_file(&parsed);
        // A parse that succeeds must print to something that parses.
        parse(&printed).map_err(|e| {
            format!("printed form of a parsed mutant does not parse: {e}\n  printed:\n{printed}")
        })?;
        polydat::dsl::compile_polydat_interpreter(source).map_err(|e| e.to_string())
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
        Ok(Ok(k)) => {
            // A mutant that compiles must have lowered exactly the
            // for forms it still contains.
            if let Ok(f) = parse(source) {
                let mut forms = Vec::new();
                collect_for_forms(&f.statements, &mut forms);
                let mut lowered = 0;
                count_lowered(k.program(), &mut lowered);
                if lowered != forms.len() {
                    failures.push(format!("{tag}mutant has {} for forms but {lowered} were lowered.\n  source:\n{source}\n  {}", forms.len(), repro));
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

#[test]
fn every_planned_form_parses_and_matches() {
    // One program per value of each dimension, through the parse and
    // shape invariants, so a plan the generator cannot honour fails
    // here rather than only in the manual superfuzz.
    let mut failures = Vec::new();
    for i in 0..ORDERS {
        let plan = ForPlan {
            form: i % FORMS.len(),
            depth: i % (MAX_DEPTH + 1),
            source: i % SOURCE_KINDS,
            clause: i % CLAUSES,
            predicate: i % PREDICATES,
            cmp: i % CMP.len(),
            order: i,
            derive: i % DERIVES,
            body: i % BODY_KINDS.len(),
            trivia: i % 2,
        };
        let (source, expected) = Gen::planned(i as u64, plan).program();
        let tag = format!("{plan:?}: ");
        failures.extend(check_wellformed(&source, &expected, false, &tag, ""));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n---\n"));
}

/// MANUAL — every dimension of the generator under the shared budget:
///
/// ```text
/// cargo nextest run -p polydat --run-ignored only -E 'test(/superfuzz_for_syntax/)' --no-capture
/// ```
#[test]
#[ignore = "manual superfuzz — up to a minute by default; run with `--run-ignored only`"]
fn superfuzz_for_syntax() {
    use super::common::superfuzz;
    let base = env_u64("FUZZ_SEED", 0xF0A_5EED);
    let seeds = superfuzz::seeds_per_combination();
    // Every combination sweeps the engines unless `FUZZ_ENGINE_SWEEP`
    // sets a stride over combinations; 0 turns the sweep off.
    let stride = env_u64("FUZZ_ENGINE_SWEEP", 1);
    superfuzz::run(
        "superfuzz_for_syntax",
        &[
            ("form", FORMS.len()),
            ("depth", MAX_DEPTH + 1),
            ("source", SOURCE_KINDS),
            ("clause", CLAUSES),
            ("where", PREDICATES),
            ("cmp", CMP.len()),
            ("order", ORDERS),
            ("derive", DERIVES),
            ("body", BODY_KINDS.len()),
            ("trivia", 2),
        ],
        |v, index| {
            let plan = ForPlan {
                form: v[0],
                depth: v[1],
                source: v[2],
                clause: v[3],
                predicate: v[4],
                cmp: v[5],
                order: v[6],
                derive: v[7],
                body: v[8],
                trivia: v[9],
            };
            let sweep = stride != 0 && index % stride == 0;
            let mut failures = Vec::new();
            for k in 0..seeds {
                let seed = superfuzz::seed_for(base, index, k);
                let (source, expected) = Gen::planned(seed, plan).program();
                let tag = format!("seed {seed:#x}: ");
                failures.extend(check_wellformed(&source, &expected, sweep, &tag, ""));
                let mutant = mutate(&mut Rng::new(seed ^ 0x5EED_F0F0), &source);
                let tag = format!("seed {seed:#x} mutant: ");
                failures.extend(check_mutant(&mutant, &tag, ""));
            }
            failures
        },
    );
}
