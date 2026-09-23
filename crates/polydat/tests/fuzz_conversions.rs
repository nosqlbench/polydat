// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Fuzzing the conversion table on every engine.
//!
//! The adapter nodes (`__u64_to_f32`, `__i128_to_u8`, …) are what the
//! assembler inserts between a wire and a port of another type, so
//! nearly every real program runs a few of them. Every other fuzzer
//! skipped them: the node pool and the coverage programs both filter
//! the `__` names out. Native code meanwhile lowered them from a table
//! of its own, a second copy of each node's rule, and it had drifted.
//! It put the f64 bits of a number in an `f32` slot, where every other
//! engine puts the f32 bits.
//!
//! This fuzzer reads the conversion table from the assembler itself
//! ([`boundary_adapter`], the superset of [`auto_adapter`]), so it
//! keeps no list of adapters to fall out of date. A value of a type is
//! produced from `cycle` by the table's own path from `u64`. Each
//! program runs on every engine at edge inputs and at seeded random
//! ones, and every read must agree with the interpreter running the
//! node bodies (cones off): the same value of the same type, or a
//! failure on both.
//!
//! - `every_conversion_agrees_*` takes each pair the table admits
//!   once, exhaustively, dealt over six tests.
//! - `random_conversion_chains_agree_on_every_engine` walks seeded
//!   random chains through the table (`FUZZ_SEED`, `FUZZ_ITERATIONS`),
//!   so a value crosses several representations before it is read.
//! - `conversion_superfuzz` is the manual deep run of the chains.

#![cfg(feature = "jit")]

use polydat::ast::{PortType, Value};
use polydat::compile::assembly::{auto_adapter, boundary_adapter};
use polydat::dsl::compile::compile_polydat_with;
use polydat::{Engine, JitMode, KernelError, Provenance};
use std::collections::{HashMap, VecDeque};

/// One edge of the conversion graph: the adapter node's name, and
/// whether the assembler inserts it on its own (class A) or only at a
/// boundary.
#[derive(Clone, Debug)]
struct Conversion {
    from: PortType,
    to: PortType,
    node: String,
    auto: bool,
}

/// Every conversion the table admits, read from the assembler.
fn table() -> Vec<Conversion> {
    let mut out = Vec::new();
    for &from in PortType::ALL {
        for &to in PortType::ALL {
            if from == to {
                continue;
            }
            if let Some(n) = boundary_adapter(from, to) {
                out.push(Conversion {
                    from,
                    to,
                    node: n.meta().name.clone(),
                    auto: auto_adapter(from, to).is_some(),
                });
            }
        }
    }
    out
}

/// A step of a chain: the node to call. A chain starts with at most one
/// step that is not a conversion, a seed.
#[derive(Clone, Debug)]
struct Step {
    node: String,
}

impl From<&Conversion> for Step {
    fn from(c: &Conversion) -> Self {
        Step {
            node: c.node.clone(),
        }
    }
}

/// A node that makes a value of a type no conversion from `u64`
/// reaches. The register types are the case: the table retags one
/// register view as another but makes none from a scalar, since that
/// is a producer's work (`reg_splat_f64`, …). The seed is found by
/// evaluation, not named: the first node, in name order, that takes
/// one wire and no constants, compiles over `cycle`, is deterministic,
/// and whose output has the type.
fn seed_for(ty: PortType) -> Option<Step> {
    use polydat::ast::SlotType;
    use polydat::dsl::compile::compile_polydat_interpreter;
    let mut sigs = polydat::dsl::registry::registry();
    sigs.sort_by_key(|s| s.name);
    sigs.into_iter()
        .filter(|s| s.outputs == 1 && !s.is_variadic())
        .filter(|s| s.params.len() == 1 && s.params[0].slot_type == SlotType::Wire)
        .find_map(|s| {
            let src = format!("input cycle: u64\nv := {}(cycle)\n", s.name);
            let k = compile_polydat_interpreter(&src).ok()?;
            let made = polydat::Kernel::output_type(&k, "v")?;
            (made == ty && k.program().is_deterministic()).then(|| Step {
                node: s.name.to_string(),
            })
        })
}

/// The shortest chain from `u64` to each type the table reaches, which
/// is how a program makes a value of that type out of `cycle`. Breadth
/// first, in table order, so the chain is stable. A type no conversion
/// reaches is started from a seed ([`seed_for`]) and the search goes on
/// from there, so the conversions out of it are fuzzed too.
fn paths_from_u64(table: &[Conversion]) -> HashMap<PortType, Vec<Step>> {
    let mut paths: HashMap<PortType, Vec<Step>> = HashMap::new();
    paths.insert(PortType::U64, Vec::new());
    let mut queue = VecDeque::from([PortType::U64]);
    grow(table, &mut paths, &mut queue);
    for &ty in PortType::ALL {
        if !paths.contains_key(&ty)
            && let Some(seed) = seed_for(ty)
        {
            paths.insert(ty, vec![seed]);
            queue.push_back(ty);
            grow(table, &mut paths, &mut queue);
        }
    }
    paths
}

fn grow(
    table: &[Conversion],
    paths: &mut HashMap<PortType, Vec<Step>>,
    queue: &mut VecDeque<PortType>,
) {
    while let Some(t) = queue.pop_front() {
        for c in table.iter().filter(|c| c.from == t) {
            if !paths.contains_key(&c.to) {
                let mut p = paths[&t].clone();
                p.push(c.into());
                paths.insert(c.to, p);
                queue.push_back(c.to);
            }
        }
    }
}

/// A program that applies `chain` to `cycle` in order and reads the
/// last value, and every value on the way so a wrong intermediate is
/// seen where it happens.
fn program(chain: &[Step]) -> String {
    let mut src = String::from("input cycle: u64\n");
    let mut prev = "cycle".to_string();
    for (i, c) in chain.iter().enumerate() {
        src.push_str(&format!("v{i} := {}({prev})\n", c.node));
        prev = format!("v{i}");
    }
    src
}

/// Inputs where conversions break: zero, one, the narrow maxima and
/// their successors, the edges of the signed types, the f32 and f64
/// exact-integer windows, and the top of the range.
const EDGES: &[u64] = &[
    0,
    1,
    127,
    128,
    255,
    256,
    32_767,
    32_768,
    65_535,
    65_536,
    2_147_483_647,
    2_147_483_648,
    4_294_967_295,
    4_294_967_296,
    (1 << 24) + 1,
    (1 << 53) + 1,
    i64::MAX as u64,
    1 << 63,
    u64::MAX,
];

fn engines() -> [Engine; 6] {
    [
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Raw),
        Engine::Closures(Provenance::Auto),
        Engine::Native(Provenance::Raw),
        Engine::Native(Provenance::Auto),
        Engine::PureNative(Provenance::Raw),
    ]
}

type Read = Result<Vec<Value>, String>;

fn payload_text(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn read(k: &mut dyn polydat::Kernel, names: &[String], c: u64) -> Read {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        k.set_inputs(&[c]);
        names.iter().map(|n| k.pull(n)).collect::<Vec<_>>()
    }))
    .map_err(payload_text)
}

/// `NaN != NaN`: equally undefined on both sides is agreement.
fn agree(a: &[Value], b: &[Value]) -> bool {
    a == b || format!("{a:?}") == format!("{b:?}")
}

/// Every disagreement between an engine and the node bodies over
/// `src` at `inputs`, one line each. An `Err` from the oracle's compile
/// is returned as the finding it is: every table entry must be
/// callable by the name the table gives it, or it cannot be fuzzed.
fn disagreements(src: &str, inputs: &[u64]) -> Vec<String> {
    // A trap in native code ends the process, not the test, and names
    // nothing; `FUZZ_TRACE` names each program before it runs.
    if std::env::var_os("FUZZ_TRACE").is_some() {
        eprintln!("fuzz_conversions:\n{src}");
    }
    let build = |e: Engine| compile_polydat_with(src, e);
    let mut oracle = match build(Engine::Interpreter(JitMode::Off)) {
        Ok(k) => k,
        Err(e) => return vec![format!("the interpreter does not compile it: {e}")],
    };
    let names: Vec<String> = oracle
        .output_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut want: Vec<Read> = Vec::new();
    for &c in inputs {
        let r = read(oracle.as_mut(), &names, c);
        if r.is_err() {
            // A kernel that panicked is not trusted for the next input.
            oracle = build(Engine::Interpreter(JitMode::Off)).unwrap();
        }
        want.push(r);
    }

    let mut out = Vec::new();
    for engine in engines() {
        let mut k = match build(engine) {
            Ok(k) => k,
            Err(KernelError::Refused { .. }) => continue,
            Err(e) => {
                out.push(format!("{engine}: does not build: {e}"));
                continue;
            }
        };
        for (i, &c) in inputs.iter().enumerate() {
            let got = read(k.as_mut(), &names, c);
            match (&want[i], &got) {
                (Ok(a), Ok(b)) if agree(a, b) => {}
                (Err(_), Err(_)) => {}
                // Pure native's unset-extern trap, engines.md §8.
                (_, Err(e)) if e.contains("cannot carry a `None`") => {}
                (a, b) => out.push(format!(
                    "at cycle={c} on {engine}:\n      interpreter: {a:?}\n      {engine}: {b:?}"
                )),
            }
            if got.is_err() {
                k = match build(engine) {
                    Ok(k) => k,
                    Err(_) => break,
                };
            }
        }
    }
    out
}

fn report(findings: Vec<String>, what: &str) {
    const SHOWN: usize = 40;
    assert!(
        findings.is_empty(),
        "{} {what} disagree across engines (first {SHOWN} shown):\n\n{}",
        findings.len(),
        findings
            .into_iter()
            .take(SHOWN)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// How many tests the exhaustive pass is dealt into. Each program
/// compiles on seven engines, two of them through Cranelift, so in one
/// piece the pass would be the suite's long pole; in shards the runner
/// spreads it over its threads.
const SHARDS: usize = 6;

/// Each conversion the table admits, once, from a value of its source
/// type made by the table's own path from `u64`, at every edge input,
/// on every engine. `shard` of [`SHARDS`] takes every `SHARDS`th entry.
fn every_conversion_agrees_on_every_engine(shard: usize) {
    let table = table();
    assert!(
        table.len() > 300,
        "the conversion table has only {} entries; is the assembler's table reachable?",
        table.len()
    );
    let paths = paths_from_u64(&table);
    let mut findings = Vec::new();
    let mut unreached = Vec::new();
    for c in table.iter().skip(shard).step_by(SHARDS) {
        let Some(prefix) = paths.get(&c.from) else {
            unreached.push(format!("{:?}", c.from));
            continue;
        };
        let mut chain = prefix.clone();
        chain.push(c.into());
        let src = program(&chain);
        for d in disagreements(&src, EDGES) {
            findings.push(format!(
                "{} ({:?} -> {:?}, {}): {d}\n    source:\n{src}",
                c.node,
                c.from,
                c.to,
                if c.auto { "auto" } else { "boundary" }
            ));
        }
    }
    unreached.sort();
    unreached.dedup();
    // Only a type no conversion from `u64` reaches is skipped, and the
    // table says which: `Ext` and `Handle` are host types.
    for t in &unreached {
        assert!(
            t == "Ext" || t == "Handle",
            "no chain of conversions reaches {t} from u64, so its conversions go unfuzzed"
        );
    }
    report(findings, "conversions");
}

macro_rules! conversion_shards {
    ($($name:ident = $shard:expr),* $(,)?) => {$(
        #[test]
        fn $name() {
            every_conversion_agrees_on_every_engine($shard);
        }
    )*};
}

conversion_shards!(
    every_conversion_agrees_0 = 0,
    every_conversion_agrees_1 = 1,
    every_conversion_agrees_2 = 2,
    every_conversion_agrees_3 = 3,
    every_conversion_agrees_4 = 4,
    every_conversion_agrees_5 = 5,
);

/// A splitmix64 stream: the seed decorrelates by the golden-ratio
/// multiply, so adjacent seeds give independent runs.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    /// A value of random magnitude: uniform draws are almost all huge,
    /// and the interesting edges are spread over every bit width.
    fn input(&mut self) -> u64 {
        let v = self.next();
        v >> self.below(64)
    }
}

/// `iterations` seeded chains at the edges and eight random inputs.
/// Each starts from `u64` or, one time in two, from the path to a
/// random reachable type (so the walks begin in the register views as
/// often as anywhere), then takes two to six random steps.
fn chain_pass(seed: u64, iterations: usize) -> Vec<String> {
    let table = table();
    let paths = paths_from_u64(&table);
    let mut starts: Vec<PortType> = paths.keys().copied().collect();
    // `HashMap` order is not stable across runs; the walk must be.
    starts.sort_by_key(|t| format!("{t:?}"));
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut findings = Vec::new();
    for i in 0..iterations {
        let steps = 2 + rng.below(5);
        let mut at = if rng.below(2) == 0 {
            PortType::U64
        } else {
            starts[rng.below(starts.len())]
        };
        let mut chain: Vec<Step> = paths[&at].clone();
        for _ in 0..steps {
            let next: Vec<&Conversion> = table.iter().filter(|c| c.from == at).collect();
            if next.is_empty() {
                break;
            }
            let c = next[rng.below(next.len())];
            at = c.to;
            chain.push(c.into());
        }
        let mut inputs = EDGES.to_vec();
        inputs.extend((0..8).map(|_| rng.input()));
        let src = program(&chain);
        for d in disagreements(&src, &inputs) {
            findings.push(format!(
                "[seed {seed:#x}] chain {i}: {d}\n    source:\n{src}    reproduce: \
                 FUZZ_SEED={seed} FUZZ_ITERATIONS={} cargo test --test suite \
                 fuzz_conversions::random_conversion_chains",
                i + 1
            ));
        }
    }
    findings
}

#[test]
fn random_conversion_chains_agree_on_every_engine() {
    let seed = std::env::var("FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xC0DE_CAFEu64);
    let iterations = std::env::var("FUZZ_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    report(chain_pass(seed, iterations), "conversion chains");
}

/// MANUAL — the deep run of the chains, `SUPERFUZZ_SEEDS` seeds of
/// `FUZZ_ITERATIONS` chains each:
///
/// ```text
/// cargo test -p polydat --all-features --test suite fuzz_conversions:: -- --ignored
/// ```
#[test]
#[ignore = "manual superfuzz — minutes of runtime; run with `-- --ignored`"]
fn conversion_superfuzz() {
    let base: u64 = std::env::var("FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xC0DE_CAFE);
    let seeds: u64 = std::env::var("SUPERFUZZ_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let iterations = std::env::var("FUZZ_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let mut all = Vec::new();
    for k in 0..seeds {
        all.extend(chain_pass(base.wrapping_add(k), iterations));
    }
    report(all, "conversion chains");
}
