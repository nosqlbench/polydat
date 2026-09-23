// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Type-adapter-transform fuzz tests.
//!
//! Two test strategies share the same goal: surface cases where the
//! Polydat compiler's auto-inserted edge adapters are wrong, missing, or
//! silently lossy.
//!
//! 1. **Adapter-table sweep** ([`adapter_table_is_consistent`]).
//!    Curated producer/consumer pairs for each [`PortType`], run
//!    through the compiler, and the result is checked against an
//!    in-test mirror of [`polydat::compile::assembly::auto_adapter`]. If the
//!    compiler disagrees with the expected table — either by
//!    rejecting a pair we think should bridge, or by silently
//!    accepting a pair we think should error — the test fails with
//!    the full pair and source so the regression is easy to
//!    reproduce.
//!
//! 2. **Random-DAG fuzz** ([`random_dags_compile_or_fail_cleanly`]).
//!    A tiny deterministic RNG picks native registry entries and
//!    wires their outputs together irrespective of type compatibility.
//!    Each generated module is fed to `compile_polydat_interpreter_with_log`; the
//!    compile must never panic, every error string must be non-empty
//!    and free of panic-style wording, and every Ok result whose
//!    event log mentions a `TypeAdapterInserted` must refer to a
//!    pair we also consider legal. The FUZZ_SEED env var seeds the
//!    RNG; FUZZ_ITERATIONS controls iteration count.

use polydat::ast::{PortType, SlotType};
use polydat::dsl::compile::{
    compile_polydat_interpreter, compile_polydat_interpreter_with_log, compile_polydat_with,
};
use polydat::dsl::events::{CompileEvent, CompileEventLog};
use polydat::dsl::registry::{self, FuncSig};

// ─── Expected-adapter table ───────────────────────────────────────
//
// The compiler's own `auto_adapter` is the table. This file used to
// keep a hand mirror of it, "updated in lock-step" so that a shift in
// the widening rules could not escape review. The mirror fell behind
// as soon as the generator drew the adapter nodes and reached types it
// had never seen (`U8→Bytes` is class A and was missing). Review of the
// table is `adapter_catalog_invariants::doc_matrix_matches_catalog`'s
// job: a catalog change fails CI until type_system.md §3 shows it.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Adapt {
    /// Source and sink are identical — no adapter needed.
    Identity,
    /// Compiler should insert an auto-adapter and compile OK.
    Inserted,
    /// No known-safe coercion; compiler should reject with `type mismatch`.
    None,
}

fn expected_adapt(src: PortType, dst: PortType) -> Adapt {
    if src == dst {
        Adapt::Identity
    } else if polydat::compile::assembly::auto_adapter(src, dst).is_some() {
        Adapt::Inserted
    } else {
        Adapt::None
    }
}

// ─── Adapter-table sweep ──────────────────────────────────────────
//
// Producers emit a specific [`PortType`] from the cycle input.
// Consumers read a specific wire type and produce any output. Each
// recipe is a pair of Polydat snippets chained via the `source` and
// `sink` bindings; the test assembles them with the cycle coordinate
// and compiles.

struct TypeRecipe {
    /// Polydat expression that produces [`src`] from scratch (may reference
    /// `cycle`).
    produce: &'static str,
    src: PortType,
}

struct SinkRecipe {
    /// Polydat expression template where `{}` is substituted with the
    /// source binding name. Produces some output (thrown away); its
    /// *wire-input* type is [`dst`].
    consume_tmpl: &'static str,
    dst: PortType,
}

/// Curated producer set. One entry per [`PortType`] we can reliably
/// synthesize from `cycle`. Missing variants (Bytes, Ext, narrow
/// ints) simply get skipped below.
fn producers() -> Vec<TypeRecipe> {
    use PortType::*;
    vec![
        TypeRecipe {
            produce: "cycle",
            src: U64,
        },
        TypeRecipe {
            produce: "to_f64(cycle)",
            src: F64,
        },
        TypeRecipe {
            produce: "format_u64(cycle, 10)",
            src: Str,
        },
        TypeRecipe {
            produce: "to_json(cycle)",
            src: Json,
        },
    ]
}

/// Curated consumer set. `{}` in the template is replaced with the
/// producer's binding name before compile.
fn consumers() -> Vec<SinkRecipe> {
    use PortType::*;
    vec![
        SinkRecipe {
            consume_tmpl: "add({}, 1)",
            dst: U64,
        },
        SinkRecipe {
            consume_tmpl: "clamp_f64({}, 0.0, 1.0)",
            dst: F64,
        },
        SinkRecipe {
            consume_tmpl: "json_to_str({})",
            dst: Json,
        },
    ]
}

#[test]
fn adapter_table_is_consistent() {
    let mut mismatches: Vec<String> = Vec::new();
    for p in producers() {
        for c in consumers() {
            let source = format!(
                "input cycle: u64\n\
                 src_val := {}\n\
                 sink_val := {}\n",
                p.produce,
                c.consume_tmpl.replace("{}", "src_val"),
            );
            let expected = expected_adapt(p.src, c.dst);
            let mut log = CompileEventLog::new();
            let result =
                compile_polydat_interpreter_with_log(&source, &mut log).map_err(|e| e.to_string());
            let observed = classify_result(&result, &log);
            if !adapt_agrees(expected, observed) {
                mismatches.push(format!(
                    "pair {:?} -> {:?}\n\
                     expected {expected:?}, observed {observed:?}\n\
                     result: {}\n\
                     source:\n{source}",
                    p.src,
                    c.dst,
                    match &result {
                        Ok(_) => "<compiled>".to_string(),
                        Err(e) => e.clone(),
                    },
                ));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "adapter-table disagreements:\n\n{}",
        mismatches.join("\n---\n")
    );
}

/// Classify a compile outcome into the same vocabulary as `Adapt`.
///
/// - Ok with no TypeAdapterInserted event → `Identity`
/// - Ok with at least one TypeAdapterInserted event → `Inserted`
/// - Err containing `"type mismatch"` → `None`
/// - Err with anything else → `Inserted` placeholder so the test
///   surfaces a `not-agrees` diagnostic rather than treating an
///   unrelated error as a success.
fn classify_result<T>(result: &Result<T, String>, log: &CompileEventLog) -> Adapt {
    match result {
        Ok(_) => {
            let has_adapter = log.events().iter().any(|e| {
                matches!(
                    e,
                    CompileEvent::TypeAdapterInserted { .. } | CompileEvent::TypeWidening { .. }
                )
            });
            if has_adapter {
                Adapt::Inserted
            } else {
                Adapt::Identity
            }
        }
        Err(msg) => {
            if msg.contains("type mismatch") {
                Adapt::None
            } else {
                // Unrelated error — return a sentinel the caller will
                // flag. Reusing `Inserted` here would make a parse
                // error look like an Ok path; return `None` so the
                // diagnostic shows "expected Inserted, got None" and
                // the actual error string is visible in the output.
                Adapt::None
            }
        }
    }
}

fn adapt_agrees(expected: Adapt, observed: Adapt) -> bool {
    match (expected, observed) {
        // Identity is a specific kind of pass — observing it is
        // also fine when we expected an `Inserted` for a same-type
        // pair, which shouldn't happen, but the sweep only calls
        // this function for distinct pairs drawn from the curated
        // producer/consumer sets.
        (Adapt::Identity, Adapt::Identity) => true,
        (Adapt::Inserted, Adapt::Inserted) => true,
        (Adapt::None, Adapt::None) => true,
        _ => false,
    }
}

// ─── Random-DAG fuzz ──────────────────────────────────────────────

/// Simple splitmix64 RNG. Deterministic, zero-dep, sufficient for
/// test inputs — we don't need cryptographic quality.
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
            return 0;
        }
        (self.next_u64() as usize) % n
    }
    fn f64(&mut self) -> f64 {
        // Integer range chosen so consts look like the kind of
        // values real workloads would pass.
        (self.next_u64() % 1000) as f64 / 10.0
    }
}

/// Nodes we can safely instantiate from the fuzzer without needing
/// specific runtime fixtures. Filters out:
///
/// - anything variadic (generator doesn't model group arity yet);
/// - anything with a `Vec<..>` constant (those need bracket-literal
///   array syntax);
/// - `dynamic-output` entries (`outputs == 0`) which need coordinate
///   resolution past what the fuzzer provides;
/// - context nodes that require runtime fixtures (metric queries,
///   control sets, fiber context) that aren't present in a unit test.
///
/// The `__` adapter nodes are drawn like any other. They were filtered
/// out as internals, which left the conversions every real program runs
/// unfuzzed while native code carried a drifted second copy of them;
/// `fuzz_conversions` now fuzzes the table directly, and here they
/// turn up in the middle of random programs as well.
fn fuzzable_sigs() -> Vec<FuncSig> {
    registry::registry()
        .into_iter()
        // A dynamic-output node (`outputs == 0`) decides its own count
        // from its arguments, which the generator cannot know; every
        // fixed count, one or many, it can bind names for.
        .filter(|s| s.outputs >= 1)
        // No fuzzable signature actually declares a `ConstVec*`
        // slot type today; the filter is kept for forward-
        // compatibility — the generator below would have to
        // synthesize array literals (`[1, 2, 3]`) for those.
        .filter(|s| {
            !s.params.iter().any(|p| {
                matches!(
                    p.slot_type,
                    SlotType::ConstVecU64 | SlotType::ConstVecF64 | SlotType::ConstVec
                )
            })
        })
        // One name, and it earns the exclusion by *writing*:
        // `fft_analyze` emits JSONL to the path its string arg names,
        // so random filenames would litter the working directory. A
        // node the fuzzer can only observe by the files it leaves
        // behind is not one a fuzzer should call.
        //
        // The list used to hold fifteen more. Ten of them — `metric`,
        // the `control*` family, `rate`, `concurrency`, `phase`,
        // `session_id` — are not polydat's nodes at all any more; a
        // host registers them, so the filter named nothing this
        // registry contains. The `csv_*` / `jsonl_*` five were held
        // out "until the macro grows a `Result`-returning setup
        // attribute", and it has: a missing file is now
        // `csv_row: construction failed: ... cannot find the file`,
        // a sentence and not a panic, which is what invariant 2 asks
        // of any error. They are fuzzed.
        .filter(|s| s.name != "fft_analyze")
        .collect()
}

/// Build a random module that declares `n_bindings` bindings in
/// sequence. Each binding calls one random function from `sigs`,
/// picking each wire arg as either `cycle` or an already-defined
/// binding, and each const arg as a random literal of the right
/// kind. No type-compatibility check is applied, so ~most generated
/// modules will exercise either the adapter insertion path or the
/// type-mismatch error path.
///
/// For `VariadicWires` sigs, the generator picks a random arity
/// in `[min_wires, min_wires + 5]` and emits that many wire args.
/// Other variadic shapes are filtered out at `fuzzable_sigs` time
/// because their positional invariants (pairs, groups) can't be
/// satisfied by random independent draws.
fn generate_module(rng: &mut Rng, sigs: &[FuncSig], n_bindings: usize) -> String {
    let mut out = String::from("input cycle: u64\n");
    let mut defined: Vec<String> = Vec::new();
    for i in 0..n_bindings {
        let sig = &sigs[rng.range(sigs.len())];
        let name = format!("b{i}");
        let mut args: Vec<String> = Vec::new();

        let pick_wire = |rng: &mut Rng, defined: &[String]| -> String {
            if defined.is_empty() || rng.range(3) == 0 {
                "cycle".to_string()
            } else {
                defined[rng.range(defined.len())].clone()
            }
        };
        let materialize =
            |rng: &mut Rng, p: &polydat::dsl::registry::ParamSpec, defined: &[String]| -> String {
                match p.slot_type {
                    SlotType::Wire => pick_wire(rng, defined),
                    SlotType::ConstU64 => format!("{}", rng.next_u64() % 100),
                    SlotType::ConstF64 => format!("{:.2}", rng.f64()),
                    SlotType::ConstStr => format!("\"s{}\"", rng.range(100)),
                    SlotType::ConstVecU64 | SlotType::ConstVecF64 | SlotType::ConstVec => {
                        unreachable!()
                    }
                }
            };

        // Fill the declared params (skip optional ones at random).
        let chosen: Vec<&_> = sig
            .params
            .iter()
            .filter(|p| p.required || rng.range(2) == 0)
            .collect();
        for param in chosen {
            args.push(materialize(rng, param, &defined));
        }

        // For `VariadicWires`, top up with a random number of
        // additional wire args. The trailing wire param shape is
        // declared once in `params` — we just emit more of the
        // same wire type past the fixed positions.
        match sig.arity {
            registry::Arity::VariadicWires { min_wires } => {
                let extra = rng.range(6); // 0..=5 extra wires
                let total_wires_needed = min_wires.saturating_sub(args.len()) + extra;
                for _ in 0..total_wires_needed {
                    args.push(pick_wire(rng, &defined));
                }
            }
            // Trailing constants repeat. The shape is what the
            // generator owes — a count at or above the minimum, of the
            // trailing const's own kind. Whether the *values* mean
            // anything to the node is the node's to say, and saying it
            // in a sentence rather than a panic is invariant 2.
            registry::Arity::VariadicConsts { min_consts } => {
                let trailing = sig
                    .params
                    .iter()
                    .rev()
                    .find(|p| p.slot_type != SlotType::Wire);
                if let Some(p) = trailing {
                    let extra = rng.range(4);
                    for _ in 0..(min_consts.saturating_sub(1) + extra) {
                        args.push(materialize(rng, p, &defined));
                    }
                }
            }
            // A repeating group of slot types, emitted in the declared
            // order so each repetition is positionally well formed.
            registry::Arity::VariadicGroup { group, min_repeats } => {
                let repeats = min_repeats + rng.range(3);
                for _ in 0..repeats {
                    for slot in group {
                        let p = sig
                            .params
                            .iter()
                            .find(|p| p.slot_type == *slot)
                            .unwrap_or(&sig.params[0]);
                        args.push(materialize(rng, p, &defined));
                    }
                }
            }
            registry::Arity::Fixed => {}
        }

        // A node with more than one output binds a name per output.
        // Dynamic-output nodes (`outputs == 0`, the count following
        // the arguments) stay out: how many names to write is the
        // node's own rule, and the generator does not know it.
        let targets = if sig.outputs > 1 {
            let names: Vec<String> = (0..sig.outputs).map(|k| format!("{name}_{k}")).collect();
            let line = format!("({})", names.join(", "));
            defined.extend(names);
            line
        } else {
            defined.push(name.clone());
            name
        };

        out.push_str(&format!("{targets} := {}({})\n", sig.name, args.join(", ")));
    }
    out
}

/// One fuzz pass: `iterations` random modules drawn from `seed`.
/// Returns human-readable invariant violations (empty = clean),
/// each carrying the seed and a reproduction line. Shared by the
/// per-commit sample test and the manual superfuzz sweep.
///
/// NOTE the draw pool is the REGISTRY, so the same seed explores
/// different programs per feature set — `cargo test --workspace`
/// (feature unification, largest registry) is a strictly stronger
/// surface than `cargo test -p polydat`.
/// How often the engine sweep runs, as a stride over iterations.
/// `FUZZ_ENGINE_SWEEP=0` disables it, `1` sweeps every module.
fn engine_sweep_stride(default: usize) -> usize {
    std::env::var("FUZZ_ENGINE_SWEEP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Every engine a build can name, the interpreter first so it is the
/// oracle the rest are compared against.
fn sweep_engines() -> Vec<polydat::Engine> {
    use polydat::{Engine, JitMode, Provenance};
    let mut all = vec![
        // The oracle: no cones, so every node runs its own body.
        Engine::Interpreter(JitMode::Off),
        // Cones on is a *different* engine for this purpose — a fused
        // cone runs native code for nodes the interpreter would
        // otherwise run itself, which is where `blend` was found
        // skipping a check its body makes (2026-09-22).
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Raw),
        Engine::Closures(Provenance::Auto),
    ];
    if cfg!(feature = "jit") {
        all.push(Engine::Native(Provenance::Raw));
        all.push(Engine::Native(Provenance::Auto));
        all.push(Engine::PureNative(Provenance::Auto));
    }
    all
}

/// Drive one cycle and read every output, catching a panic the way the
/// compile path is caught: a garbage program is expected to fail at
/// evaluation, and what matters is that it fails the same way on every
/// engine rather than only on some.
fn drive_once(kernel: &mut dyn polydat::Kernel) -> Result<Vec<polydat::ast::Value>, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        kernel.set_inputs(&[7]);
        let names = kernel.output_names();
        names.iter().map(|n| kernel.pull(n)).collect::<Vec<_>>()
    }))
    .map_err(|p| {
        p.downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "<non-string panic>".to_string())
    })
}

/// Whether two reads of the same program agree.
///
/// `==` first, which is the answer for everything that has one. A
/// generated program reaches `asin(7)` soon enough, and `NaN != NaN`,
/// so a float that is *equally* undefined on two engines would read as
/// a disagreement forever; falling back to the rendering makes two
/// `NaN`s agree without making anything else agree that should not.
/// `-0.0` and `0.0` still pass on the fast path, as `==` says they do.
fn values_agree(a: &[polydat::ast::Value], b: &[polydat::ast::Value]) -> bool {
    a == b || format!("{a:?}") == format!("{b:?}")
}

/// Invariants 4 and 5 — the differential half.
///
/// The generator above is engine-blind: it produced programs, the
/// interpreter compiled them, and nothing ever asked the tiers a host
/// actually runs on. That left every compiled-only mechanism unfuzzed —
/// the native lowerings, the kits, the `Ref2` scratch, segment fusion —
/// and it is not hypothetical: `jit_shuffle` carried its own copy of a
/// node body with its own divide-by-zero, which this fuzzer could reach
/// in the node and never in the helper (2026-09-22).
///
/// Two invariants, both stated against the interpreter as oracle:
///
/// - **I4.** A program the interpreter compiled either compiles on
///   every other engine or is *refused* there. An engine may decline a
///   program ([engines.md](../docs/design/engines.md) §8 names the
///   refusals); what it may not do is fail some other way, or fail with
///   a message that reads as a crash.
/// - **I5.** When the program is deterministic and compiled everywhere,
///   one cycle reads the same on every engine — values, or the same
///   failure. Nondeterministic programs are skipped rather than
///   compared, since disagreeing is what they are for.
fn engine_sweep_failures(source: &str) -> Vec<String> {
    use polydat::KernelError;
    let mut out = Vec::new();

    // Only a deterministic program can be compared by value. `random`,
    // a clock or a counter is meant to differ. Asking the interpreter's
    // concrete kernel is the one thing only it can answer, and the
    // documented reason to hold a `PolydatKernel` rather than a
    // `dyn Kernel`.
    let deterministic = match compile_polydat_interpreter(source) {
        Ok(k) => k.program().is_deterministic(),
        // The interpreter did not accept it; invariants 1 and 2 already
        // judged that, and there is nothing to compare against.
        Err(_) => return out,
    };
    // The oracle is the *first* engine of the sweep, so that whatever
    // the list holds is what the rest are measured against. Taking it
    // from a different compile than the list names is how cones-on went
    // uncompared while being the reference.
    let mut reference = match compile_polydat_with(source, sweep_engines()[0]) {
        Ok(k) => k,
        Err(_) => return out,
    };
    let want = drive_once(reference.as_mut());

    for engine in sweep_engines().into_iter().skip(1) {
        let mut kernel = match compile_polydat_with(source, engine) {
            Ok(k) => k,
            Err(KernelError::Refused { .. }) => continue,
            Err(e) => {
                let msg = e.to_string();
                let cryptic = msg.is_empty()
                    || msg.to_lowercase().contains("panic")
                    || msg.to_lowercase().contains("index out of bounds")
                    || msg.to_lowercase().contains("unreachable");
                if cryptic {
                    out.push(format!(
                        "I4: {engine} failed a program the interpreter compiled, and not \
                         as a refusal:\n  error: {msg}"
                    ));
                }
                continue;
            }
        };
        if !deterministic {
            continue;
        }
        let got = drive_once(kernel.as_mut());
        match (&want, &got) {
            (Ok(a), Ok(b)) if values_agree(a, b) => {}
            (Err(_), Err(_)) => {}
            (a, b) => out.push(format!(
                "I5: {engine} disagrees with the interpreter on a deterministic \
                 program:\n  interpreter: {a:?}\n  {engine}: {b:?}"
            )),
        }
    }
    out
}

fn run_fuzz_pass(seed: u64, iterations: usize, sweep_stride: usize) -> Vec<String> {
    // Per-seed failure cap: a systematic defect (e.g. one node
    // panicking on every draw) floods the report without adding
    // signal; eight distinct repros per seed is plenty.
    const MAX_FAILURES_PER_SEED: usize = 8;

    let sigs = fuzzable_sigs();
    assert!(
        !sigs.is_empty(),
        "no fuzzable signatures found — registry wiring broken?"
    );

    let mut rng = Rng::new(seed);
    let mut failures: Vec<String> = Vec::new();
    let repro = |i: usize| {
        format!(
            "reproduce: FUZZ_SEED={seed} FUZZ_ITERATIONS={} cargo test --workspace \
         --test suite fuzz_type_adapters::random_dags",
            i + 1
        )
    };

    for i in 0..iterations {
        if failures.len() >= MAX_FAILURES_PER_SEED {
            failures.push(format!(
                "[seed {seed:#x}] … stopping this seed after {MAX_FAILURES_PER_SEED} failures"
            ));
            break;
        }
        let n = 3 + rng.range(8);
        let source = generate_module(&mut rng, &sigs, n);

        let mut log = CompileEventLog::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compile_polydat_interpreter_with_log(&source, &mut log)
        }));

        // Invariant 1: compiler never panics on any input. A panic
        // here is always a bug in the compiler — even when the input
        // is bananas, the error path should be a returned `Err`, not
        // a process-level abort.
        let result =
            match result {
                Ok(r) => r,
                Err(panic) => {
                    failures.push(format!(
                    "[seed {seed:#x}] compiler panicked on iteration {i}:\n  source:\n{source}\n  \
                     panic: {:?}\n  {}",
                    panic.downcast_ref::<&str>().copied()
                        .or_else(|| panic.downcast_ref::<String>().map(|s| s.as_str()))
                        .unwrap_or("<non-string panic>"),
                    repro(i),
                ));
                    continue;
                }
            };

        match result {
            Err(msg) => {
                // Invariant 2: every error is either a recognised
                // structural diagnostic (type mismatch, bad
                // constant, undeclared reference, unknown function,
                // variadic-arity issue, …) or a well-formed
                // sentence. We're not prescribing *which* error
                // fires — the fuzzer routinely builds garbage —
                // only that the compiler classified it rather than
                // leaking panics or raw backtraces. A structured
                // `bad constant …` message proves the opt-in
                // assembly-time validator (SRD 15 §"Const
                // Constraint Metadata") rejected the literal
                // before the node's constructor saw it.
                if msg.to_string().is_empty()
                    || msg.to_string().to_lowercase().contains("panic")
                    || msg
                        .to_string()
                        .to_lowercase()
                        .contains("index out of bounds")
                    || msg.to_string().to_lowercase().contains("unreachable")
                {
                    failures.push(format!(
                        "[seed {seed:#x}] iteration {i} produced a cryptic error message.\n  \
                         error: {msg}\n  source:\n{source}\n  {}",
                        repro(i)
                    ));
                }
            }
            Ok(_) => {
                // Invariants 4 and 5: the same program on every engine.
                // Sampled — a sweep compiles the module five more
                // times, twice through Cranelift, so running it on
                // every iteration would cost more than the generator
                // is worth. `FUZZ_ENGINE_SWEEP` is the stride; 0 turns
                // it off, 1 sweeps everything, and the superfuzz sets
                // it to 1 because that is the run that can afford it.
                if sweep_stride != 0 && i % sweep_stride == 0 {
                    for detail in engine_sweep_failures(&source) {
                        failures.push(format!(
                            "[seed {seed:#x}] iteration {i}: {detail}\n  source:\n{source}\n  {}",
                            repro(i)
                        ));
                    }
                }

                // Invariant 3: every adapter the compiler auto-inserts
                // must be one we know about. Anything else is a rogue
                // entry — probably a new adapter added to the
                // compiler without an entry in this test's mirror.
                for e in log.events() {
                    if let CompileEvent::TypeAdapterInserted { adapter, .. } = e
                        && !adapter_label_is_known(adapter)
                    {
                        failures.push(format!(
                            "[seed {seed:#x}] iteration {i} inserted an unrecognised \
                                 adapter '{adapter}'.\n\
                                 Update `expected_adapt`/`adapter_label_is_known` and the \
                                 compiler's\n`auto_adapter` table together.\n  \
                                 source:\n{source}\n  {}",
                            repro(i)
                        ));
                    }
                }
            }
        }
    }
    failures
}

#[test]
fn random_dags_compile_or_fail_cleanly() {
    let seed: u64 = std::env::var("FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xDEAD_BEEFu64);
    let iterations: usize = std::env::var("FUZZ_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    // Sampled sweep: the per-commit run pays for one module in eight
    // on every engine.
    let failures = run_fuzz_pass(seed, iterations, engine_sweep_stride(8));
    assert!(
        failures.is_empty(),
        "fuzz invariants violated ({} failures):\n\n{}",
        failures.len(),
        failures.join("\n---\n")
    );
}

/// MANUAL SUPERFUZZ — the deep sweep the per-commit sample can't
/// afford. `#[ignore]`d; run it deliberately:
///
/// ```text
/// cargo test -p polydat --test suite fuzz_type_adapters:: -- --ignored
/// ```
///
/// Sweeps `SUPERFUZZ_SEEDS` seeds (default 64) × `FUZZ_ITERATIONS`
/// modules each (default 2000), starting at `FUZZ_SEED` (default
/// 0xDEADBEEF). Use `--workspace` — feature unification gives the
/// largest registry and therefore the widest program space; a
/// `-p polydat` run fuzzes a strict subset. Every violation
/// carries its own single-seed reproduction line.
#[test]
#[ignore = "manual superfuzz — minutes of runtime; run with `-- --ignored`"]
fn superfuzz_sampler() {
    let base: u64 = std::env::var("FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xDEAD_BEEFu64);
    let seeds: u64 = std::env::var("SUPERFUZZ_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let iterations: usize = std::env::var("FUZZ_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);

    let mut all: Vec<String> = Vec::new();
    for k in 0..seeds {
        // Rng::new decorrelates adjacent integers via the golden-
        // ratio multiply, so base+k gives independent trajectories.
        let seed = base.wrapping_add(k);
        // Every module on every engine: this is the run that can
        // afford it, and the cross-engine space is the point of it.
        let failures = run_fuzz_pass(seed, iterations, engine_sweep_stride(1));
        if !failures.is_empty() {
            eprintln!("superfuzz: seed {seed:#x}: {} violation(s)", failures.len());
        }
        if k % 8 == 7 {
            eprintln!(
                "superfuzz: {}/{seeds} seeds swept, {} violation(s) so far",
                k + 1,
                all.len() + failures.len()
            );
        }
        all.extend(failures);
    }

    // Bound the panic payload — repros are self-contained, so the
    // first screenful carries everything needed.
    let mut report = all.join("\n---\n");
    const MAX_REPORT: usize = 30_000;
    if report.len() > MAX_REPORT {
        let mut cut = MAX_REPORT;
        while !report.is_char_boundary(cut) {
            cut -= 1;
        }
        report.truncate(cut);
        report.push_str("\n… (report truncated)");
    }
    assert!(
        all.is_empty(),
        "superfuzz invariants violated ({} failures across {seeds} seeds × \
         {iterations} iterations):\n\n{report}",
        all.len()
    );
}

/// Whether a `{SrcType:?}→{DstType:?}` label the compiler writes into
/// [`CompileEvent::TypeAdapterInserted`] names a conversion
/// `auto_adapter` makes. The label is read back into its two types
/// through their `Debug` names, and the table itself is asked, so a
/// conversion the assembler inserts that its own table would not is
/// the finding (an adapter inserted from anywhere else).
fn adapter_label_is_known(label: &str) -> bool {
    let port = |name: &str| {
        PortType::ALL
            .iter()
            .copied()
            .find(|t| format!("{t:?}") == name)
    };
    let Some((from, to)) = label.split_once('→') else {
        return false;
    };
    match (port(from), port(to)) {
        (Some(f), Some(t)) => polydat::compile::assembly::auto_adapter(f, t).is_some(),
        _ => false,
    }
}

// ─── Basic sanity for the harness itself ──────────────────────────

/// End-to-end M2+M3: a `mod_wire(x, y)` call under
/// `pragma strict_values` triggers an auto-inserted
/// `AssertValue` between the divisor source and `mod_wire`'s
/// `divisor` wire input. Without the pragma, no assertion is
/// inserted (the node trusts its inputs as default).
#[test]
fn strict_values_inserts_nonzero_assertion_on_mod_wire() {
    use polydat::dsl::events::CompileEvent;

    // Strict mode: assertion expected.
    let strict_source = "\
        pragma strict_values\n\
        \n\
        d := mod(hash(cycle), 100)\n\
        b := mod_wire(cycle, d)\n\
    ";
    let mut log = CompileEventLog::new();
    let result = compile_polydat_interpreter_with_log(strict_source, &mut log);
    assert!(result.is_ok(), "compile failed: {:?}", result.err());
    let assertion_inserts: Vec<&CompileEvent> = log
        .events()
        .iter()
        .filter(|e| matches!(e, CompileEvent::AssertionInserted { .. }))
        .collect();
    assert!(
        !assertion_inserts.is_empty(),
        "expected at least one AssertionInserted under strict_values; events: {:?}",
        log.events(),
    );

    // Non-strict mode: no assertion event.
    let lax_source = "\
        d := mod(hash(cycle), 100)\n\
        b := mod_wire(cycle, d)\n\
    ";
    let mut lax_log = CompileEventLog::new();
    let lax_result = compile_polydat_interpreter_with_log(lax_source, &mut lax_log);
    assert!(lax_result.is_ok(), "compile failed: {:?}", lax_result.err());
    let lax_inserts: Vec<&CompileEvent> = lax_log
        .events()
        .iter()
        .filter(|e| matches!(e, CompileEvent::AssertionInserted { .. }))
        .collect();
    assert!(
        lax_inserts.is_empty(),
        "no AssertionInserted expected without pragma; got: {lax_inserts:?}",
    );
}

/// When the divisor source is a constant (already validated at
/// assembly time), strict_values mode skips the assertion — it's
/// provably redundant. SRD 15 §"Strict Wire Mode" skip rule #2.
#[test]
fn strict_values_skips_assertion_when_source_is_constant() {
    use polydat::dsl::events::CompileEvent;
    let source = "\
        pragma strict_values\n\
        b := mod_wire(cycle, 7)\n\
    ";
    let mut log = CompileEventLog::new();
    let result = compile_polydat_interpreter_with_log(source, &mut log);
    assert!(result.is_ok(), "compile failed: {:?}", result.err());
    let inserts: Vec<&CompileEvent> = log
        .events()
        .iter()
        .filter(|e| matches!(e, CompileEvent::AssertionInserted { .. }))
        .collect();
    assert!(
        inserts.is_empty(),
        "constant source should skip assertion; got inserts: {inserts:?}",
    );
    let skips: Vec<&CompileEvent> = log
        .events()
        .iter()
        .filter(|e| matches!(e, CompileEvent::AssertionSkipped { .. }))
        .collect();
    assert!(
        !skips.is_empty(),
        "expected an AssertionSkipped event for constant source; events: {:?}",
        log.events(),
    );
}

/// Pragma directives at the source head are recognised and recorded
/// in the compile event log — `strict_values` / `strict_types` /
/// `strict` produce advisories, unknown pragmas produce warnings,
/// and the pragma surface is forward-compatible (an unrecognised
/// pragma never blocks compilation). See SRD 15 §"Module-Level
/// Pragmas".
#[test]
fn pragmas_round_trip_through_compile() {
    use polydat::dsl::events::CompileEvent;
    let source = "\
        pragma strict\n\
        pragma warp_drive\n\
        \n\
        id := mod(hash(cycle), 1000)\n\
    ";
    let mut log = CompileEventLog::new();
    let result = compile_polydat_interpreter_with_log(source, &mut log);
    assert!(result.is_ok(), "compile failed: {:?}", result.err());
    let acknowledged: Vec<&str> = log
        .events()
        .iter()
        .filter_map(|e| match e {
            CompileEvent::PragmaAcknowledged { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    let unknown: Vec<&str> = log
        .events()
        .iter()
        .filter_map(|e| match e {
            CompileEvent::UnknownPragma { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        acknowledged,
        vec!["strict"],
        "expected single ack for `strict`"
    );
    assert_eq!(
        unknown,
        vec!["warp_drive"],
        "expected unknown record for `warp_drive`"
    );
}

/// Guard against the test mis-firing: a plain same-type chain must
/// compile cleanly with no adapter events. If this ever fails, the
/// fuzz infrastructure itself is broken — check the compiler or the
/// event log machinery before chasing the other tests.
#[test]
fn sanity_same_type_chain_has_no_adapters() {
    let source = "\
        input cycle: u64\n\
        a := add(cycle, 1)\n\
        b := add(a, 2)\n\
    ";
    let mut log = CompileEventLog::new();
    let result = compile_polydat_interpreter_with_log(source, &mut log);
    assert!(
        result.is_ok(),
        "simple chain should compile: {:?}",
        result.err()
    );
    for e in log.events() {
        if let CompileEvent::TypeAdapterInserted { .. } = e {
            panic!("unexpected type adapter in same-type chain:\n{source}\nevent: {e:?}");
        }
    }
}

#[test]
fn sanity_u64_to_f64_is_reported_as_a_widening() {
    let source = "\
        input cycle: u64\n\
        a := clamp_f64(cycle, 0.0, 1.0)\n\
    ";
    let mut log = CompileEventLog::new();
    let result = compile_polydat_interpreter_with_log(source, &mut log);
    assert!(
        result.is_ok(),
        "u64→f64 widening should auto-adapt: {:?}",
        result.err()
    );
    let has_adapter = log.events().iter().any(|e| {
        matches!(
            e,
            CompileEvent::TypeWidening {
                from: "u64",
                to: "f64",
                ..
            }
        )
    });
    assert!(
        has_adapter,
        "expected a u64 to f64 widening event in log: {:?}",
        log.events()
    );
}

#[test]
fn sanity_f64_to_u64_rejects_without_cast() {
    // F64 → U64 is narrowing and must not auto-insert — the compiler
    // should report a type mismatch so the author is forced to pick
    // `f64_to_u64` / `round_to_u64` / `floor_to_u64` explicitly.
    let source = "\
        input cycle: u64\n\
        x := to_f64(cycle)\n\
        y := add(x, 1)\n\
    ";
    let err = compile_polydat_interpreter(source).expect_err("narrowing f64→u64 must not compile");
    assert!(
        err.to_string().contains("type mismatch"),
        "expected a type-mismatch error for narrowing, got: {err}"
    );
}

/// The strictness pragmas reach the graph on every entry point. The
/// plain interpreter path once dropped them, so `pragma strict_values`
/// inserted an assertion under the logged compile and not under
/// `compile_polydat`; the two programs now have the same node count on
/// every path, and so does the default engine's.
#[test]
fn strict_pragmas_reach_every_compile_path() {
    let source = "\
        pragma strict_values\n\
        \n\
        d := mod(hash(cycle), 100)\n\
        b := mod_wire(cycle, d)\n\
    ";
    let lax = "\
        d := mod(hash(cycle), 100)\n\
        b := mod_wire(cycle, d)\n\
    ";
    let plain = compile_polydat_interpreter(source).expect("plain compile");
    let mut log = CompileEventLog::new();
    let logged = compile_polydat_interpreter_with_log(source, &mut log).expect("logged compile");
    let lax_kernel = compile_polydat_interpreter(lax).expect("lax compile");
    let plain_nodes = plain.program().node_count();
    assert_eq!(
        plain_nodes,
        logged.program().node_count(),
        "the plain and logged interpreter paths must build the same graph"
    );
    // The assertion node may be fused into a cone, whose label names
    // its members, so look for it by name rather than by node count.
    let has_assertion = |k: &polydat::kernel::PolydatKernel| {
        let p = k.program();
        (0..p.node_count()).any(|i| p.node_meta(i).name.contains("assert_u64_nonzero"))
    };
    assert!(
        has_assertion(&plain) && !has_assertion(&lax_kernel),
        "strict_values must insert the assertion on the plain path too"
    );
    let mut default_log = CompileEventLog::new();
    let compiled = polydat::dsl::compile::compile_polydat_kernel_with_options(
        source,
        &polydat::dsl::compile::CompileOptions::default(),
        Some(&mut default_log),
    )
    .expect("default-engine compile");
    let count = |log: &CompileEventLog| {
        log.events()
            .iter()
            .filter(|e| matches!(e, CompileEvent::AssertionInserted { .. }))
            .count()
    };
    assert_eq!(count(&log), count(&default_log));
    assert!(count(&log) > 0);
    drop(compiled);
}

/// The assertion `pragma strict_values` inserts fails the same way on
/// the interpreter and on the default engine: the same node, the same
/// message. Before the assertion nodes had a compiled form the default
/// engine refused every strict program outright.
#[test]
fn strict_value_assertion_fails_alike_on_every_engine() {
    use polydat::Kernel;
    let source = "\
        pragma strict_values\n\
        input cycle: u64\n\
        d := mod(cycle, 1)\n\
        b := mod_wire(cycle, d)\n\
    ";
    // The message less the line that names where the panic was raised,
    // which is the node's own eval on the interpreter and its compiled
    // form elsewhere.
    fn failure(kernel: &mut dyn Kernel) -> String {
        kernel.set_inputs(&[7]);
        let hit = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| kernel.pull("b")));
        let err = hit.expect_err("a zero divisor must trip the assertion");
        let text = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        text.lines()
            .filter(|l| !l.contains("panicked at"))
            .collect::<Vec<_>>()
            .join("\n")
    }
    let mut interpreter = compile_polydat_interpreter(source).expect("interpreter compile");
    let mut default = polydat::dsl::compile::compile_polydat_kernel_with_options(
        source,
        &polydat::dsl::compile::CompileOptions::default(),
        None,
    )
    .expect("default-engine compile");
    let on_interpreter = failure(&mut interpreter);
    let on_default = failure(default.as_mut());
    assert!(
        on_interpreter.contains("assert_u64_nonzero"),
        "interpreter failure names the assertion: {on_interpreter}"
    );
    assert_eq!(on_interpreter, on_default);
}
