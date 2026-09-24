// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `polydat perf`: measure programs on every engine, from a suite.
//!
//! A suite is a set of groups, each a program and the engines to run it
//! on, read from TOML or taken from the built-in suite (the engine
//! ladder graph and a graph of conversions). Every rung, one group on
//! one engine, runs through the surface a host uses: `set_inputs` and
//! `pull_at` on a `Box<dyn Kernel>`. A rung is calibrated into batches
//! of about `batch_ms`, warmed for `warmup_ms`, and measured for
//! `measure_ms`; the rung's value for a round is the median batch, in
//! nanoseconds per cycle. Rounds interleave every rung, and the order
//! rotates each round, so a machine that drifts during the run drifts
//! under every rung alike.
//!
//! `--save` writes the results as JSON and `--compare` reads them back
//! against a new run. `--against <exe>` is the comparison the
//! performance guide asks for: both binaries, this one and the other,
//! run one round each in turn, their order alternating round by round,
//! and each rung is reported as the mean of its per-round deltas with a
//! 95% interval. Pairing is what makes a small difference measurable:
//! drift between rounds falls out of a delta taken within one.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Args;
use polydat::{Engine, JitMode, Kernel, KernelError, Provenance};
use serde::{Deserialize, Serialize};

/// `polydat perf` arguments.
#[derive(Args)]
pub struct PerfArgs {
    /// A TOML suite. Without one, the built-in suite runs; see
    /// `--print-config` for its form.
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,
    /// Rounds, overriding the suite's `settings.rounds`.
    #[arg(long)]
    rounds: Option<u32>,
    /// Warmup per rung per round, in milliseconds.
    #[arg(long, value_name = "MS")]
    warmup_ms: Option<u64>,
    /// Measurement per rung per round, in milliseconds.
    #[arg(long, value_name = "MS")]
    measure_ms: Option<u64>,
    /// Provenance mode of the compiled engines, overriding the suite's.
    #[arg(long, value_name = "MODE")]
    provenance: Option<String>,
    /// Run only this group. Repeatable.
    #[arg(long = "group", value_name = "NAME")]
    groups: Vec<String>,
    /// Run only this engine. Repeatable.
    #[arg(long = "engine", value_name = "ENGINE")]
    engines: Vec<String>,
    /// Write the results to this file as JSON.
    #[arg(long, value_name = "FILE")]
    save: Option<PathBuf>,
    /// Compare the results with a file written by `--save`.
    #[arg(long, value_name = "FILE")]
    compare: Option<PathBuf>,
    /// Pair this binary with another `polydat` binary, round by round.
    #[arg(long, value_name = "EXE")]
    against: Option<PathBuf>,
    /// Print the suite as TOML and exit: the built-in one, or
    /// `--config` after its overrides.
    #[arg(long)]
    print_config: bool,
    /// Run one round and print its results as JSON. The paired mode
    /// runs each binary this way.
    #[arg(long, hide = true)]
    round_json: bool,
}

/// A suite: its settings and its groups.
#[derive(Serialize, Deserialize, Clone)]
struct Suite {
    #[serde(default)]
    settings: Settings,
    #[serde(rename = "group", default)]
    groups: Vec<Group>,
}

/// How long and how often each rung is measured.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Settings {
    /// Rounds; every rung is measured once per round.
    #[serde(default = "default_rounds")]
    rounds: u32,
    /// Warmup per rung per round.
    #[serde(default = "default_warmup_ms")]
    warmup_ms: u64,
    /// Measurement per rung per round.
    #[serde(default = "default_measure_ms")]
    measure_ms: u64,
    /// The length of one timed batch; the median batch is the round's
    /// value, so this sets how many samples a round has.
    #[serde(default = "default_batch_ms")]
    batch_ms: u64,
    /// Provenance mode of the compiled engines: `auto`, `raw`, `push`,
    /// `pull`, or `push-pull`. `auto` is what a host gets by default.
    /// Under `raw` every pull recomputes its output's cone, so a
    /// program read at several outputs pays for each, and on pure native
    /// code, one function for the whole program, it pays for all of it
    /// every time.
    #[serde(default = "default_provenance")]
    provenance: String,
}

fn default_rounds() -> u32 {
    10
}
fn default_warmup_ms() -> u64 {
    300
}
fn default_measure_ms() -> u64 {
    1000
}
fn default_batch_ms() -> u64 {
    10
}
fn default_provenance() -> String {
    "auto".into()
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            rounds: default_rounds(),
            warmup_ms: default_warmup_ms(),
            measure_ms: default_measure_ms(),
            batch_ms: default_batch_ms(),
            provenance: default_provenance(),
        }
    }
}

/// One program, measured on each of its engines.
#[derive(Serialize, Deserialize, Clone)]
struct Group {
    name: String,
    /// A `.polydat` file, relative to the suite file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    program: Option<PathBuf>,
    /// The program's text, when it is not a file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    /// A graph to build rather than read: see [`Generate`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    generate: Option<Generate>,
    /// `interpreter`, `interpreter-cones`, `closures`, `native`, or
    /// `pure-native`, measured in this order.
    #[serde(default = "default_engines")]
    engines: Vec<String>,
    /// The outputs pulled each cycle; empty is every output.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    outputs: Vec<String>,
    /// Fixed values for inputs other than the cycle input.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    inputs: BTreeMap<String, u64>,
    /// The input advanced by one each cycle.
    #[serde(default = "default_cycle")]
    cycle: String,
}

/// A generated graph: a shape and its dimensions. The graphs are built
/// to vary what a pull's cone covers, so that a suite can measure how
/// each engine's treatment of cones scales with the graph. Every graph
/// reads `cycle`, gives each path its own seed so no two are the same
/// expression, and names its outputs `o0`, `o1`, … — the ends a host
/// would pull.
///
/// - `chains`: `width` independent chains of `depth` hash steps. Each
///   output's cone is its own chain, `1 / width` of the graph.
/// - `trunk`: one shared chain of `depth` steps, then `width` branches
///   of `branch` steps each. The cones overlap in the trunk.
/// - `lattice`: `depth` layers of `width` nodes, each mixing two
///   neighbours of the layer below. A cone widens by one node a layer
///   until it covers the layer.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Generate {
    shape: String,
    #[serde(default = "one")]
    width: u32,
    #[serde(default = "one")]
    depth: u32,
    /// The steps in each branch of a `trunk`.
    #[serde(default = "one")]
    branch: u32,
}

fn one() -> u32 {
    1
}

impl Generate {
    /// The program's text and its outputs.
    fn build(&self) -> Result<(String, Vec<String>), String> {
        let (w, d, b) = (self.width.max(1), self.depth.max(1), self.branch.max(1));
        let mut src = String::from("input cycle: u64\n");
        let mut line = |s: String| {
            src.push_str(&s);
            src.push('\n');
        };
        let outputs: Vec<String> = (0..w).map(|j| format!("o{j}")).collect();
        match self.shape.as_str() {
            "chains" => {
                for j in 0..w {
                    line(format!("c{j}_0 := u64_add(cycle, {})", j + 1));
                    for k in 1..d {
                        line(format!("c{j}_{k} := hash(c{j}_{})", k - 1));
                    }
                    line(format!("o{j} := hash(c{j}_{})", d - 1));
                }
            }
            "trunk" => {
                line("t0 := hash(cycle)".into());
                for k in 1..d {
                    line(format!("t{k} := hash(t{})", k - 1));
                }
                for j in 0..w {
                    line(format!("b{j}_0 := u64_add(t{}, {})", d - 1, j + 1));
                    for k in 1..b {
                        line(format!("b{j}_{k} := hash(b{j}_{})", k - 1));
                    }
                    line(format!("o{j} := hash(b{j}_{})", b - 1));
                }
            }
            "lattice" => {
                for j in 0..w {
                    line(format!("l0_{j} := u64_add(cycle, {})", j + 1));
                }
                for i in 1..d {
                    for j in 0..w {
                        line(format!(
                            "x{i}_{j} := u64_xor(l{}_{j}, l{}_{})",
                            i - 1,
                            i - 1,
                            (j + 1) % w
                        ));
                        line(format!("l{i}_{j} := hash(x{i}_{j})"));
                    }
                }
                for j in 0..w {
                    line(format!("o{j} := hash(l{}_{j})", d - 1));
                }
            }
            other => {
                return Err(format!(
                    "unknown shape `{other}`; use chains, trunk, or lattice"
                ));
            }
        }
        Ok((src, outputs))
    }

    fn describe(&self) -> String {
        match self.shape.as_str() {
            "trunk" => format!(
                "trunk, depth {}, width {}, branch {}",
                self.depth, self.width, self.branch
            ),
            s => format!("{s}, width {}, depth {}", self.width, self.depth),
        }
    }
}

fn default_engines() -> Vec<String> {
    ["interpreter", "closures", "native", "pure-native"]
        .map(String::from)
        .to_vec()
}
fn default_cycle() -> String {
    "cycle".into()
}

/// A graph of the conversion adapters: checked narrowings, `u64` to and
/// from `f32`, and widenings, over a value every narrowing accepts.
/// The ladder graph is all `u64` and inserts no adapter, so it cannot
/// see how conversions are lowered.
const CONVERSIONS: &str = "input cycle: u64
m := mod(cycle, 1000)
a := __u64_to_u32(m)
b := __u32_to_i32(a)
c := __i32_to_i64(b)
f := __u64_to_f32(m)
g := __f32_to_f64(f)
h := __u64_to_i64(m)
k := __i64_to_u64(h)
w := __u32_to_u64(a)
";

/// The suite that runs without `--config`.
fn builtin_suite() -> Suite {
    Suite {
        settings: Settings::default(),
        groups: vec![
            Group {
                name: "ladder".into(),
                program: None,
                generate: None,
                source: Some(include_str!("../examples/engine_ladder.polydat").into()),
                engines: default_engines(),
                outputs: ["account_id", "shard", "payload_class", "event_token"]
                    .map(String::from)
                    .to_vec(),
                inputs: BTreeMap::from([
                    ("tenant_seed".into(), 0x5445_4e41_4e54),
                    ("operation_seed".into(), 0x4f50_4552_4154_494f),
                ]),
                cycle: default_cycle(),
            },
            Group {
                name: "conversions".into(),
                program: None,
                generate: None,
                source: Some(CONVERSIONS.into()),
                engines: default_engines(),
                outputs: ["c", "g", "k", "w", "f"].map(String::from).to_vec(),
                inputs: BTreeMap::new(),
                cycle: default_cycle(),
            },
        ],
    }
}

/// The results of a run: what the rungs measured, with the settings
/// that measured them.
#[derive(Serialize, Deserialize)]
struct Report {
    /// The version of the binary that measured.
    polydat: String,
    settings: Settings,
    results: Vec<RungResult>,
}

#[derive(Serialize, Deserialize, Clone)]
struct RungResult {
    group: String,
    engine: String,
    /// Nanoseconds per cycle, one value per round.
    rounds: Vec<f64>,
}

impl RungResult {
    fn median(&self) -> f64 {
        median(&self.rounds)
    }
}

/// One group on one engine, compiled and ready to cycle.
struct Rung {
    group: String,
    engine: String,
    kernel: Box<dyn Kernel>,
    coords: Vec<u64>,
    cycle_at: usize,
    outputs: Vec<usize>,
}

impl Rung {
    /// One cycle: advance the cycle input and pull every measured
    /// output, as a host does.
    #[inline]
    fn step(&mut self) {
        self.coords[self.cycle_at] = self.coords[self.cycle_at].wrapping_add(1);
        self.kernel.set_inputs(&self.coords);
        for &o in &self.outputs {
            black_box(self.kernel.pull_at(o));
        }
    }

    /// Nanoseconds per cycle: the median of batches of about `batch`,
    /// after `warmup`, over `measure`.
    fn measure(&mut self, warmup: Duration, measure: Duration, batch: Duration) -> f64 {
        let time = |rung: &mut Rung, n: u64| {
            let t = Instant::now();
            for _ in 0..n {
                rung.step();
            }
            t.elapsed()
        };
        // Grow the batch until it takes about `batch`.
        let mut n = 1u64;
        while time(self, n) < batch && n < (1 << 40) {
            n *= 2;
        }
        let warm_until = Instant::now() + warmup;
        while Instant::now() < warm_until {
            time(self, n);
        }
        let mut samples = Vec::new();
        let until = Instant::now() + measure;
        while Instant::now() < until || samples.is_empty() {
            samples.push(time(self, n).as_nanos() as f64 / n as f64);
        }
        median(&samples)
    }
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    match v.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => v[n / 2],
        n => (v[n / 2 - 1] + v[n / 2]) / 2.0,
    }
}

fn mean_and_sd(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = if values.len() > 1 {
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    (mean, var.sqrt())
}

fn parse_engine(name: &str, provenance: Provenance) -> Result<Engine, String> {
    Ok(match name {
        "interpreter" => Engine::Interpreter(JitMode::Off),
        "interpreter-cones" => Engine::Interpreter(JitMode::Auto),
        "closures" => Engine::Closures(provenance),
        "native" => Engine::Native(provenance),
        "pure-native" => Engine::PureNative(provenance),
        other => {
            return Err(format!(
                "unknown engine `{other}`; use interpreter, interpreter-cones, closures, \
                 native, or pure-native"
            ));
        }
    })
}

fn parse_provenance(name: &str) -> Result<Provenance, String> {
    Ok(match name {
        "raw" => Provenance::Raw,
        "push" => Provenance::Push,
        "pull" => Provenance::Pull,
        "push-pull" => Provenance::PushPull,
        "auto" => Provenance::Auto,
        other => {
            return Err(format!(
                "unknown provenance `{other}`; use raw, push, pull, push-pull, or auto"
            ));
        }
    })
}

/// The suite to run: the file or the built-in one, with the command
/// line's overrides and filters applied. Program paths are resolved
/// against the suite file's directory.
fn load_suite(args: &PerfArgs) -> Result<Suite, String> {
    let mut suite = match &args.config {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            let mut suite: Suite =
                toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
            let dir = path.parent().unwrap_or(Path::new("."));
            for g in &mut suite.groups {
                if let Some(p) = &g.program
                    && p.is_relative()
                {
                    g.program = Some(dir.join(p));
                }
            }
            suite
        }
        None => builtin_suite(),
    };
    if let Some(r) = args.rounds {
        suite.settings.rounds = r;
    }
    if let Some(w) = args.warmup_ms {
        suite.settings.warmup_ms = w;
    }
    if let Some(m) = args.measure_ms {
        suite.settings.measure_ms = m;
    }
    if let Some(p) = &args.provenance {
        parse_provenance(p)?;
        suite.settings.provenance = p.clone();
    }
    if !args.groups.is_empty() {
        suite.groups.retain(|g| args.groups.contains(&g.name));
    }
    if !args.engines.is_empty() {
        for g in &mut suite.groups {
            g.engines.retain(|e| args.engines.contains(e));
        }
    }
    suite.groups.retain(|g| !g.engines.is_empty());
    if suite.groups.is_empty() {
        return Err("no group and engine left to measure".into());
    }
    Ok(suite)
}

/// A group's program text, and a generated graph's outputs. Exactly
/// one of `program`, `source`, and `generate` names the program.
fn group_source(g: &Group) -> Result<(String, Option<Vec<String>>), String> {
    match (&g.source, &g.program, &g.generate) {
        (Some(s), None, None) => Ok((s.clone(), None)),
        (None, Some(p), None) => std::fs::read_to_string(p)
            .map(|s| (s, None))
            .map_err(|e| format!("group `{}`: cannot read {}: {e}", g.name, p.display())),
        (None, None, Some(generator)) => generator
            .build()
            .map(|(s, outs)| (s, Some(outs)))
            .map_err(|e| format!("group `{}`: {e}", g.name)),
        (None, None, None) => Err(format!(
            "group `{}` has no program: give `program`, `source`, or `generate`",
            g.name
        )),
        _ => Err(format!(
            "group `{}` names its program more than once: give one of `program`, `source`, \
             and `generate`",
            g.name
        )),
    }
}

/// Compile every rung. An engine that declines a program is noted and
/// skipped; any other failure is the suite's error.
fn build_rungs(suite: &Suite) -> Result<(Vec<Rung>, Vec<String>), String> {
    let provenance = parse_provenance(&suite.settings.provenance)?;
    let mut rungs = Vec::new();
    let mut notes = Vec::new();
    for g in &suite.groups {
        let (source, generated_outputs) = group_source(g)?;
        for e in &g.engines {
            let engine = parse_engine(e, provenance)?;
            let kernel = match polydat::dsl::compile::compile_polydat_with(&source, engine) {
                Ok(k) => k,
                Err(KernelError::Refused { .. }) => {
                    notes.push(format!(
                        "{} · {e}: this engine declines the program",
                        g.name
                    ));
                    continue;
                }
                Err(err) => return Err(format!("group `{}` on {e}: {err}", g.name)),
            };
            let names = kernel.input_names();
            let cycle_at = names.iter().position(|n| *n == g.cycle).ok_or_else(|| {
                format!("group `{}`: no input named `{}` to cycle", g.name, g.cycle)
            })?;
            for name in g.inputs.keys() {
                if !names.contains(name) {
                    return Err(format!("group `{}`: no input named `{name}`", g.name));
                }
            }
            let coords = names
                .iter()
                .map(|n| g.inputs.get(n).copied().unwrap_or(0))
                .collect();
            // A generated graph's own outputs by default: every binding
            // is an output, and its intermediates are not what a host
            // pulls.
            let wanted = match (g.outputs.is_empty(), &generated_outputs) {
                (false, _) => g.outputs.clone(),
                (true, Some(outs)) => outs.clone(),
                (true, None) => kernel.output_names(),
            };
            let outputs = wanted
                .iter()
                .map(|n| {
                    kernel
                        .output_index(n)
                        .ok_or_else(|| format!("group `{}`: no output named `{n}`", g.name))
                })
                .collect::<Result<_, _>>()?;
            rungs.push(Rung {
                group: g.name.clone(),
                engine: e.clone(),
                kernel,
                coords,
                cycle_at,
                outputs,
            });
        }
    }
    if rungs.is_empty() {
        return Err("no rung compiled".into());
    }
    Ok((rungs, notes))
}

/// Measure every rung for the suite's rounds, rotating the order each
/// round, with progress on stderr unless `quiet`.
fn measure_rounds(suite: &Suite, rungs: &mut [Rung], rounds: u32, quiet: bool) -> Vec<RungResult> {
    let s = &suite.settings;
    let (warmup, measure, batch) = (
        Duration::from_millis(s.warmup_ms),
        Duration::from_millis(s.measure_ms),
        Duration::from_millis(s.batch_ms.max(1)),
    );
    let mut values = vec![Vec::new(); rungs.len()];
    let k = rungs.len();
    for round in 0..rounds as usize {
        for j in 0..k {
            let i = (j + round) % k;
            if !quiet {
                eprint!(
                    "\r[round {}/{}] {:>2}/{k}  {} · {:<24}",
                    round + 1,
                    rounds,
                    j + 1,
                    rungs[i].group,
                    rungs[i].engine
                );
                let _ = std::io::stderr().flush();
            }
            values[i].push(rungs[i].measure(warmup, measure, batch));
        }
    }
    if !quiet {
        eprintln!("\r{:<72}", "");
    }
    rungs
        .iter()
        .zip(values)
        .map(|(r, v)| RungResult {
            group: r.group.clone(),
            engine: r.engine.clone(),
            rounds: v,
        })
        .collect()
}

/// The groups in suite order, each with its rungs in engine order.
fn by_group<'a, T>(
    suite: &Suite,
    items: &'a [T],
    key: impl Fn(&T) -> (&str, &str),
) -> Vec<(String, Vec<&'a T>)> {
    suite
        .groups
        .iter()
        .map(|g| {
            let mut rows: Vec<&T> = items.iter().filter(|t| key(t).0 == g.name).collect();
            rows.sort_by_key(|t| g.engines.iter().position(|e| e == key(t).1));
            (g.name.clone(), rows)
        })
        .filter(|(_, rows)| !rows.is_empty())
        .collect()
}

fn print_header(suite: &Suite, rounds: u32, what: &str) {
    let s = &suite.settings;
    println!(
        "polydat perf {what}: {rounds} round(s); per rung per round {} ms warmup, {} ms measured \
         in {} ms batches; provenance {}",
        s.warmup_ms, s.measure_ms, s.batch_ms, s.provenance
    );
}

/// A group's heading: where its program comes from, how many steps it
/// has, and how many of its outputs a cycle pulls, which together say
/// how much of the graph a pull's cone is.
fn group_line(suite: &Suite, name: &str) -> String {
    let Some(g) = suite.groups.iter().find(|g| g.name == name) else {
        return format!("\n{name}");
    };
    let from = match (&g.program, &g.generate) {
        (Some(p), _) => p.display().to_string(),
        (None, Some(generator)) => generator.describe(),
        (None, None) => "inline source".into(),
    };
    let size = group_source(g).ok().map(|(src, outs)| {
        let steps = src
            .lines()
            .filter(|l| l.contains(":=") && !l.trim_start().starts_with("//"))
            .count();
        let pulled = if g.outputs.is_empty() {
            outs.map_or("every output".into(), |o| format!("{} outputs", o.len()))
        } else {
            format!("{} output(s)", g.outputs.len())
        };
        format!("; {steps} bindings, pulling {pulled}")
    });
    format!("\n{name}  ({from}{})", size.unwrap_or_default())
}

/// The engines of the ladder in order, when a group has them: each must
/// be at least as fast as the one before (the performance guide).
fn ladder_check(rows: &[&RungResult]) -> Option<String> {
    let order = ["interpreter", "closures", "native"];
    let present: Vec<(&str, f64)> = order
        .iter()
        .filter_map(|e| {
            rows.iter()
                .find(|r| r.engine == *e)
                .map(|r| (*e, r.median()))
        })
        .collect();
    if present.len() < 2 {
        return None;
    }
    let broken: Vec<String> = present
        .windows(2)
        .filter(|w| w[1].1 > w[0].1)
        .map(|w| format!("{} is slower than {}", w[1].0, w[0].0))
        .collect();
    Some(if broken.is_empty() {
        format!(
            "  ladder ok: {}",
            present.iter().map(|p| p.0).collect::<Vec<_>>().join(" > ")
        )
    } else {
        format!("  ladder BROKEN: {}", broken.join("; "))
    })
}

fn print_report(suite: &Suite, results: &[RungResult], rounds: u32) {
    print_header(suite, rounds, "run");
    for (group, rows) in by_group(suite, results, |r| (&r.group, &r.engine)) {
        println!("{}", group_line(suite, &group));
        println!(
            "  {:<18} {:>11} {:>9} {:>11} {:>9}",
            "engine", "ns/cycle", "spread", "min", "speedup"
        );
        let slowest = rows.iter().map(|r| r.median()).fold(0.0, f64::max);
        for r in &rows {
            let (mean, sd) = mean_and_sd(&r.rounds);
            let min = r.rounds.iter().copied().fold(f64::INFINITY, f64::min);
            println!(
                "  {:<18} {:>11.1} {:>8.1}% {:>11.1} {:>8.2}x",
                r.engine,
                r.median(),
                sd / mean * 100.0,
                min,
                slowest / r.median()
            );
        }
        if let Some(line) = ladder_check(&rows) {
            println!("{line}");
        }
    }
    println!(
        "\nns/cycle is the median of the rounds; spread is their coefficient of variation; \
         speedup is against the slowest engine in the group."
    );
}

fn print_comparison(suite: &Suite, results: &[RungResult], saved: &Report) {
    println!(
        "\nagainst the saved results (polydat {}). The two runs were not interleaved, so drift \
         between them is part of every delta; `--against` pairs them instead.",
        saved.polydat
    );
    for (group, rows) in by_group(suite, results, |r| (&r.group, &r.engine)) {
        println!("{}", group_line(suite, &group));
        println!(
            "  {:<18} {:>11} {:>11} {:>9}",
            "engine", "saved", "now", "delta"
        );
        for r in rows {
            match saved
                .results
                .iter()
                .find(|s| s.group == r.group && s.engine == r.engine)
            {
                Some(s) => println!(
                    "  {:<18} {:>11.1} {:>11.1} {:>+8.1}%",
                    r.engine,
                    s.median(),
                    r.median(),
                    (r.median() - s.median()) / s.median() * 100.0
                ),
                None => println!("  {:<18} {:>11} {:>11.1}", r.engine, "-", r.median()),
            }
        }
    }
}

/// A rung by name: its group and its engine.
type RungKey = (String, String);

/// Run one round of `exe` and read its results.
fn run_leg(exe: &Path, args: &[String]) -> Result<Report, String> {
    let out = std::process::Command::new(exe)
        .arg("perf")
        .arg("--round-json")
        .args(args)
        .output()
        .map_err(|e| format!("cannot run {}: {e}", exe.display()))?;
    if !out.status.success() {
        return Err(format!(
            "{} failed: {}",
            exe.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    serde_json::from_slice(&out.stdout)
        .map_err(|e| format!("{} did not report a round: {e}", exe.display()))
}

/// Pair this binary with `other`: each round runs both, one round each,
/// in an order that alternates, and every rung is reported as the mean
/// of its per-round deltas with a 95% interval.
fn paired(args: &PerfArgs, suite: &Suite, other: &Path) -> Result<(), String> {
    let me = std::env::current_exe().map_err(|e| format!("cannot find this binary: {e}"))?;
    let mut leg_args = Vec::new();
    if let Some(c) = &args.config {
        let c = std::fs::canonicalize(c).unwrap_or_else(|_| c.clone());
        leg_args.push("--config".into());
        leg_args.push(c.display().to_string());
    }
    leg_args.extend(["--warmup-ms".into(), suite.settings.warmup_ms.to_string()]);
    leg_args.extend(["--measure-ms".into(), suite.settings.measure_ms.to_string()]);
    leg_args.extend(["--provenance".into(), suite.settings.provenance.clone()]);
    for g in &args.groups {
        leg_args.extend(["--group".into(), g.clone()]);
    }
    for e in &args.engines {
        leg_args.extend(["--engine".into(), e.clone()]);
    }
    let legs = [("this", me.as_path()), ("other", other)];
    let rounds = suite.settings.rounds;
    // (group, engine) -> per round, this and other.
    let mut pairs: BTreeMap<RungKey, Vec<(f64, f64)>> = BTreeMap::new();
    let mut versions = ("?".to_string(), "?".to_string());
    for round in 0..rounds {
        let order = if round % 2 == 0 { [0, 1] } else { [1, 0] };
        let mut got: [Option<Report>; 2] = [None, None];
        for leg in order {
            eprint!(
                "\r[round {}/{}] {} ({})          ",
                round + 1,
                rounds,
                legs[leg].0,
                legs[leg].1.display()
            );
            let _ = std::io::stderr().flush();
            got[leg] = Some(run_leg(legs[leg].1, &leg_args)?);
        }
        let [Some(this), Some(that)] = got else {
            unreachable!("both legs ran")
        };
        versions = (this.polydat.clone(), that.polydat.clone());
        for r in &this.results {
            if let Some(o) = that
                .results
                .iter()
                .find(|o| o.group == r.group && o.engine == r.engine)
            {
                pairs
                    .entry((r.group.clone(), r.engine.clone()))
                    .or_default()
                    .push((r.median(), o.median()));
            }
        }
    }
    eprintln!("\r{:<72}", "");
    print_header(suite, rounds, "paired");
    println!(
        "this:  {} (polydat {})\nother: {} (polydat {})",
        me.display(),
        versions.0,
        other.display(),
        versions.1
    );
    let rows: Vec<(RungKey, Vec<(f64, f64)>)> = pairs.into_iter().collect();
    for (group, rows) in by_group(suite, &rows, |((g, e), _)| (g, e)) {
        println!("{}", group_line(suite, &group));
        println!(
            "  {:<18} {:>11} {:>11} {:>9} {:>9}",
            "engine", "this", "other", "delta", "±95%"
        );
        for ((_, engine), v) in rows {
            let deltas: Vec<f64> = v.iter().map(|(a, b)| (a - b) / b * 100.0).collect();
            let (mean, sd) = mean_and_sd(&deltas);
            let ci = 1.96 * sd / (deltas.len() as f64).sqrt();
            let this: Vec<f64> = v.iter().map(|p| p.0).collect();
            let other: Vec<f64> = v.iter().map(|p| p.1).collect();
            println!(
                "  {:<18} {:>11.1} {:>11.1} {:>+8.1}% {:>8.1}%",
                engine,
                median(&this),
                median(&other),
                mean,
                ci
            );
        }
    }
    println!(
        "\nthis and other are the medians of each binary's rounds, in ns/cycle. delta is the mean \
         of the per-round deltas of this against other, and ±95% its interval: a delta inside \
         it is not a difference. A rung the change cannot reach is a canary; if it moves, the \
         run measured the machine."
    );
    Ok(())
}

/// The compiler's warnings about the suite's programs, once each. A
/// program is compiled once per engine, so printing them as they come
/// would repeat each one for every rung.
static WARNINGS: std::sync::Mutex<std::collections::BTreeSet<String>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

/// `polydat perf`.
pub fn perf(args: PerfArgs) -> Result<(), String> {
    use polydat::library::support::audit::{self, LogLevel};
    audit::set_log_fn(|level, msg| match level {
        LogLevel::Error => eprintln!("Error: {msg}"),
        LogLevel::Warn => {
            WARNINGS.lock().unwrap().insert(msg.to_string());
        }
        _ => {}
    });
    let suite = load_suite(&args)?;
    if args.print_config {
        print!(
            "{}",
            toml::to_string_pretty(&suite).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    if let Some(other) = &args.against {
        if args.save.is_some() || args.compare.is_some() {
            return Err(
                "`--against` pairs two binaries; `--save` and `--compare` are for a \
                        single run"
                    .into(),
            );
        }
        return paired(&args, &suite, other);
    }
    let (mut rungs, notes) = build_rungs(&suite)?;
    let rounds = if args.round_json {
        1
    } else {
        suite.settings.rounds
    };
    let results = measure_rounds(&suite, &mut rungs, rounds, args.round_json);
    let report = Report {
        polydat: env!("CARGO_PKG_VERSION").into(),
        settings: suite.settings.clone(),
        results,
    };
    if args.round_json {
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    print_report(&suite, &report.results, rounds);
    for note in notes {
        println!("note: {note}");
    }
    for w in WARNINGS.lock().unwrap().iter() {
        println!("note: the compiler warned: {w}");
    }
    if let Some(path) = &args.compare {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let saved: Report =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        print_comparison(&suite, &report.results, &saved);
    }
    if let Some(path) = &args.save {
        let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        println!("\nsaved to {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The built-in suite written by `--print-config` reads back as the
    /// same suite.
    #[test]
    fn the_builtin_suite_round_trips_through_toml() {
        let text = toml::to_string_pretty(&builtin_suite()).unwrap();
        let back: Suite = toml::from_str(&text).unwrap();
        assert_eq!(back.groups.len(), 2);
        assert_eq!(back.groups[0].name, "ladder");
        assert_eq!(back.groups[0].inputs.len(), 2);
        assert_eq!(back.settings.rounds, default_rounds());
    }

    /// A suite file may leave everything but the groups' names and
    /// programs to the defaults.
    #[test]
    fn a_minimal_suite_takes_the_defaults() {
        let s: Suite = toml::from_str(
            "[[group]]\nname = \"g\"\nsource = \"input cycle: u64\\nout := hash(cycle)\"\n",
        )
        .unwrap();
        assert_eq!(s.groups[0].engines, default_engines());
        assert_eq!(s.groups[0].cycle, "cycle");
        assert_eq!(s.settings.measure_ms, default_measure_ms());
    }

    /// Every shape builds a program that compiles on every engine, with
    /// the outputs it names, and a graph of the size its dimensions say.
    #[test]
    fn every_generated_shape_compiles_on_every_engine() {
        for (shape, w, d, b, bindings) in [
            ("chains", 3, 4, 1, 3 * (4 + 1)),
            ("trunk", 3, 4, 2, 4 + 3 * (2 + 1)),
            ("lattice", 3, 4, 1, 3 + 3 * 3 * 2 + 3),
        ] {
            let g = Generate {
                shape: shape.into(),
                width: w,
                depth: d,
                branch: b,
            };
            let (src, outs) = g.build().unwrap();
            assert_eq!(outs, ["o0", "o1", "o2"], "{shape}");
            let count = src.lines().filter(|l| l.contains(":=")).count();
            assert_eq!(count, bindings as usize, "{shape}:\n{src}");
            for e in default_engines() {
                let engine = parse_engine(&e, Provenance::Auto).unwrap();
                let mut k = polydat::dsl::compile::compile_polydat_with(&src, engine)
                    .unwrap_or_else(|err| panic!("{shape} on {e}: {err}\n{src}"));
                k.set_inputs(&[1]);
                for o in &outs {
                    let _ = k.pull(o);
                }
            }
        }
        let unknown = Generate {
            shape: "spiral".into(),
            width: 1,
            depth: 1,
            branch: 1,
        };
        assert!(unknown.build().unwrap_err().contains("spiral"));
    }

    #[test]
    fn an_unknown_engine_is_named() {
        let err = parse_engine("turbo", Provenance::Raw).unwrap_err();
        assert!(
            err.contains("turbo") && err.contains("pure-native"),
            "{err}"
        );
    }
}
