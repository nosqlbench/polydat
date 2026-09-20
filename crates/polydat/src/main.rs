// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The `polydat` binary: a harness that compiles, explains, and runs
//! Polydat programs.
//!
//! Optional behaviors are graph transforms, not runtime decorators.
//! `--emit` appends an `emit_row` binding to the program so row
//! emission is an ordinary side-channel node with access to the local
//! scope. Timing and statistics wrap the run from outside, since they
//! measure the kernel rather than participate in it.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};
use polydat::ast::Slot;
use polydat::dsl::ast::PolydatFile;
use polydat::dsl::ast::TileOptions;
use polydat::dsl::ast::{Statement, WireModifier};
use polydat::dsl::events::{CompileEvent, CompileEventLog};
use polydat::dsl::transform::{assign_values, parse_assignment};
use polydat::dsl::{CompileOptions, compile_ast_with_engine};
use polydat::iteration::cursor_partition::{Partition, cursor_over_partitions_on};
use polydat::kernel::activation::Activation;
use polydat::kernel::{KernelProgram, PolydatProgram, WireSource, extract_manifest};
use polydat::library::emit::{self, EmitFormat};
use polydat::library::support::audit::{self, LogLevel};
use polydat::{Engine as KernelEngine, EnginePlan, JitMode, Kernel, Provenance};

#[derive(Parser)]
#[command(
    name = "polydat",
    version,
    about = "Compile, explain, and run Polydat programs"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile a program and run cycles through it.
    Run(RunArgs),
    /// Compile a program and report diagnostics without running it.
    Check(CheckArgs),
    /// Narrate how the compiler realizes a program, phase by phase.
    Explain(ExplainArgs),
    /// Render the program graph as DOT, Mermaid, or SVG.
    Viz(VizArgs),
}

#[derive(Args, Clone)]
struct CompileArgs {
    /// Path to the `.polydat` source file.
    file: PathBuf,
    /// Directory searched for `.polydat` modules. Repeatable.
    #[arg(long = "lib", value_name = "DIR")]
    libs: Vec<PathBuf>,
    /// Reject implicit type coercions and require explicit inputs.
    #[arg(long)]
    strict: bool,
    /// Execution engine: `auto` is the build's default, native code
    /// where the build has it and closures otherwise; `interpreter`,
    /// `closures`, `native`, and `pure-native` name one. `pure-native`
    /// refuses the program when any node has no native lowering, where
    /// `native` runs that node's closure.
    #[arg(long, value_enum, default_value_t = Engine::Auto)]
    engine: Engine,
    /// Provenance mode of a compiled engine: how much work a cycle whose
    /// inputs repeat skips. `auto` lets the selector choose.
    #[arg(long, value_enum, default_value_t = ProvenanceArg::Auto)]
    provenance: ProvenanceArg,
    /// Native cones in the interpreter's program: the one `--engine
    /// interpreter` runs and the one `explain` and `--stats` describe.
    #[arg(long, value_enum, default_value_t = Cones::Auto)]
    cones: Cones,
    /// Keep only these outputs and what they depend on. Repeatable.
    #[arg(long = "output", value_name = "NAME")]
    required: Vec<String>,
    /// Default hole delimiters for tiles that declare none of their own:
    /// `--tile-delims '<%' '%>'`. A tile body is raw text until it is
    /// read, and these say how to read it, so they are given to the
    /// parse rather than applied to the program afterwards.
    #[arg(long = "tile-delims", num_args = 2, value_names = ["OPEN", "CLOSE"])]
    tile_delims: Vec<String>,
    /// Default directive sigil for tiles that declare none of their own.
    #[arg(long = "tile-sigil", value_name = "SIGIL")]
    tile_sigil: Option<String>,
}

impl CompileArgs {
    /// How much of the interpreter's program is fused into cones: what
    /// `--cones` says, and nothing else. The engine and the cone mode
    /// are separate choices, which is the whole point of splitting
    /// them out of the one flag that used to carry both.
    fn cones(&self) -> JitMode {
        match self.cones {
            Cones::Off => JitMode::Off,
            Cones::Force => JitMode::Force,
            Cones::Auto => JitMode::Auto,
        }
    }

    /// The provenance mode a compiled engine is built with.
    fn provenance(&self) -> Provenance {
        match self.provenance {
            ProvenanceArg::Auto => Provenance::Auto,
            ProvenanceArg::Raw => Provenance::Raw,
            ProvenanceArg::Push => Provenance::Push,
            ProvenanceArg::Pull => Provenance::Pull,
            ProvenanceArg::PushPull => Provenance::PushPull,
        }
    }

    /// The engine the run drives, from `--engine`, `--provenance`, and
    /// `--cones`.
    fn run_engine(&self) -> KernelEngine {
        let provenance = self.provenance();
        match self.engine {
            Engine::Auto => match KernelEngine::default() {
                KernelEngine::Native(_) => KernelEngine::Native(provenance),
                KernelEngine::Closures(_) => KernelEngine::Closures(provenance),
                other => other,
            },
            Engine::Interpreter => KernelEngine::Interpreter(self.cones()),
            Engine::Closures => KernelEngine::Closures(provenance),
            Engine::Native => KernelEngine::Native(provenance),
            Engine::PureNative => KernelEngine::PureNative(provenance),
        }
    }

    /// The host's tile defaults, when any were given.
    fn tile_defaults(&self) -> Option<TileOptions> {
        if self.tile_delims.is_empty() && self.tile_sigil.is_none() {
            return None;
        }
        let mut opts = TileOptions::default();
        if let [open, close] = self.tile_delims.as_slice() {
            opts.open = open.clone();
            opts.close = close.clone();
        }
        if let Some(s) = &self.tile_sigil {
            opts.sigil = s.clone();
        }
        Some(opts)
    }
}

/// What `--emit` asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
enum EmitSpec {
    Format(EmitFormat),
    Tile(String),
}

fn parse_emit_spec(s: &str) -> Result<EmitSpec, String> {
    if let Some(name) = s.strip_prefix("tile:") {
        if name.is_empty() {
            return Err("`--emit tile:<name>` needs a tile name".to_string());
        }
        return Ok(EmitSpec::Tile(name.to_string()));
    }
    EmitFormat::parse(s)
        .filter(|f| *f != EmitFormat::Text)
        .map(EmitSpec::Format)
        .ok_or_else(|| format!("unknown emit format '{s}'; use map, csv, jsonl, or tile:<name>"))
}

#[derive(Args)]
struct RunArgs {
    #[command(flatten)]
    compile: CompileArgs,
    /// Extern or input assignments as bare `name=value` arguments.
    #[arg(value_name = "NAME=VALUE")]
    assignments: Vec<String>,
    /// Number of cycles to run.
    #[arg(long, default_value_t = 10)]
    cycles: u64,
    /// First cycle value.
    #[arg(long, default_value_t = 0)]
    start: u64,
    /// Concurrent fibers, each with its own state over the shared program.
    #[arg(long, default_value_t = 1)]
    fibers: usize,
    /// Cycles per work unit claimed by a fiber.
    #[arg(long, default_value_t = 1024)]
    chunk: u64,
    /// Emit rows as they complete rather than in cycle order.
    #[arg(long)]
    unordered: bool,
    /// Emit the selected outputs each cycle: `map`, `csv`, `jsonl`, or
    /// `tile:<name>` to emit one tile's rendered text per cycle.
    #[arg(long, value_name = "FORMAT", value_parser = parse_emit_spec)]
    emit: Option<EmitSpec>,
    /// Comma-separated wires to emit. Defaults to every declared output.
    #[arg(long, value_name = "NAMES")]
    outputs: Option<String>,
    /// Write emitted rows here instead of stdout.
    #[arg(long, value_name = "PATH")]
    out: Option<PathBuf>,
    /// Set an extern or input slot: `--set name=value`. Same as a bare
    /// `name=value` argument. Repeatable.
    #[arg(long = "set", value_name = "NAME=VALUE")]
    sets: Vec<String>,
    /// When a cursor's `over` spec yields several partitions, run this one.
    /// Without it, a run with as many fibers as partitions gives each
    /// fiber its own partition and every fiber walks the full cycle range.
    #[arg(long, value_name = "INDEX")]
    partition: Option<usize>,
    /// Cycles to run before timing starts.
    #[arg(long, default_value_t = 0)]
    warmup: u64,
    /// Report timing after the run, as text or JSON.
    #[arg(long, value_enum, value_name = "FORMAT", num_args = 0..=1, default_missing_value = "text")]
    timing: Option<Report>,
    /// Print program statistics after compiling.
    #[arg(long)]
    stats: bool,
    /// Print the compiler event log after compiling.
    #[arg(long)]
    events: bool,
    /// Suppress compiler and runtime audit messages.
    #[arg(long, short)]
    quiet: bool,
}

#[derive(Args)]
struct CheckArgs {
    #[command(flatten)]
    compile: CompileArgs,
    /// Print program statistics.
    #[arg(long)]
    stats: bool,
    /// Print the compiler event log.
    #[arg(long)]
    events: bool,
    /// Print the output manifest.
    #[arg(long)]
    manifest: bool,
    /// Report as text or JSON.
    #[arg(long, value_enum, default_value_t = Report::Text)]
    format: Report,
}

#[derive(Args)]
struct ExplainArgs {
    #[command(flatten)]
    compile: CompileArgs,
    /// Phases to narrate. Omit for every phase in order.
    #[arg(value_enum)]
    phases: Vec<Phase>,
}

#[derive(Args)]
struct VizArgs {
    /// Path to the `.polydat` source file.
    file: PathBuf,
    #[arg(long, value_enum, default_value_t = VizFormat::Dot)]
    format: VizFormat,
}

/// The engine a run drives.
#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum Engine {
    /// The build's default engine: native code where the build has it,
    /// closures otherwise.
    Auto,
    /// The interpreter, with native cones per `--cones`.
    Interpreter,
    /// The closure tier: every node runs its generated closure.
    Closures,
    /// Native code where a node has a lowering, its closure elsewhere.
    Native,
    /// Native code and nothing else: refuses the program when any node
    /// has no native lowering, which is how you find out whether a
    /// program is fully native.
    PureNative,
}

/// How much of a compiled engine's work is skipped when inputs repeat.
#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum ProvenanceArg {
    /// The selector's choice from the graph's shape.
    Auto,
    /// Every evaluation runs every step.
    Raw,
    /// A changed input reruns only the steps downstream of it.
    Push,
    /// An output whose cone no changed input reaches is not recomputed.
    Pull,
    /// Both: per-step skipping and the cone guard.
    PushPull,
}

/// How much of the interpreter's graph is fused into native cones.
#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum Cones {
    /// Cones where the cost model says they pay.
    Auto,
    /// No native code: the differential baseline.
    Off,
    /// Every eligible node joins a cone.
    Force,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum Report {
    Text,
    Json,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum VizFormat {
    Dot,
    Mermaid,
    Svg,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Debug)]
enum Phase {
    Lex,
    Parse,
    Inputs,
    Modules,
    Wires,
    Types,
    Lifecycle,
    Constants,
    Fusion,
    Engines,
    Provenance,
    Outputs,
    Traversals,
    Tiles,
}

const ALL_PHASES: [Phase; 14] = [
    Phase::Lex,
    Phase::Parse,
    Phase::Inputs,
    Phase::Modules,
    Phase::Wires,
    Phase::Types,
    Phase::Lifecycle,
    Phase::Constants,
    Phase::Fusion,
    Phase::Engines,
    Phase::Provenance,
    Phase::Outputs,
    Phase::Traversals,
    Phase::Tiles,
];

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Run(args) => run(args),
        Command::Check(args) => check(args),
        Command::Explain(args) => explain(args),
        Command::Viz(args) => viz(args),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Compilation shared by every command
// ---------------------------------------------------------------------------

/// Everything a compile produced: the program on the run engine, its
/// event log, and the audit lines the compiler wrote while it ran.
struct Compiled {
    /// The interpreter's program: what `explain` and `--stats` describe
    /// node by node. The run's own when the run engine is the
    /// interpreter; otherwise compiled by [`describe`] on demand, since
    /// a run needs nothing from it.
    program: Option<Arc<PolydatProgram>>,
    /// The program on the run engine: every fiber, warmup, cursor probe,
    /// and traversal root is a kernel created from it.
    root: Arc<dyn KernelProgram>,
    /// The engine `--engine` named, which the activations open on too.
    engine: KernelEngine,
    /// What that engine decided for the root: its native segments,
    /// closure steps, and interpreted nodes.
    run_plan: EnginePlan,
    events: CompileEventLog,
    audit: Vec<(LogLevel, String)>,
    elapsed: Duration,
}

static AUDIT: Mutex<Vec<(LogLevel, String)>> = Mutex::new(Vec::new());
static AUDIT_ECHO: Mutex<bool> = Mutex::new(true);

fn install_audit(echo: bool) {
    *AUDIT_ECHO.lock().unwrap() = echo;
    audit::set_log_fn(|level, msg| {
        AUDIT.lock().unwrap().push((level, msg.to_string()));
        if *AUDIT_ECHO.lock().unwrap() && matches!(level, LogLevel::Warn | LogLevel::Error) {
            eprintln!("{level:?}: {msg}");
        }
    });
}

fn take_audit() -> Vec<(LogLevel, String)> {
    std::mem::take(&mut *AUDIT.lock().unwrap())
}

fn read_source(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))
}

/// Advance the coordinates to `coords`, one value per coordinate input.
/// Only coordinate inputs move; an extern, including an input a
/// `name=value` argument fixed for this run, keeps the value it was
/// given. A program with no coordinate left still runs a cycle: nothing
/// moved, so nothing would re-evaluate, and the kernel is told to run
/// everything again so per-cycle nodes such as the emit binding fire.
/// The output indices of `names` on `kernel`, resolved once so the
/// run pulls by index (SRD 117 step 3).
fn resolve_pulls(kernel: &dyn Kernel, names: &[String]) -> Vec<usize> {
    names
        .iter()
        .map(|n| {
            kernel
                .output_index(n)
                .unwrap_or_else(|| panic!("the program has no output named `{n}`"))
        })
        .collect()
}

fn drive_cycle(kernel: &mut dyn Kernel, coords: &[u64]) {
    if coords.is_empty() {
        kernel.invalidate_all();
    } else {
        kernel.set_inputs(coords);
    }
}

fn parse_source(source: &str) -> Result<PolydatFile, String> {
    let tokens = polydat::dsl::lexer::lex(source)?;
    polydat::dsl::parser::parse(tokens)
}

/// Parse a program, reading tiles that name no delimiters of their own
/// under the host's, when `--tile-delims` supplied any.
fn parse_program(source: &str, args: &CompileArgs) -> Result<PolydatFile, String> {
    let tokens = polydat::dsl::lexer::lex(source)?;
    match args.tile_defaults() {
        Some(defaults) => polydat::dsl::parser::parse_with_tile_defaults(tokens, &defaults),
        None => polydat::dsl::parser::parse(tokens),
    }
}

/// The options every command compiles under.
fn compile_options(args: &CompileArgs) -> CompileOptions {
    // A bare file name has an empty parent; modules beside it live in
    // the current directory.
    let source_dir = args.file.parent().map(|p| {
        if p.as_os_str().is_empty() {
            PathBuf::from(".")
        } else {
            p.to_path_buf()
        }
    });
    CompileOptions {
        source_dir,
        lib_paths: args.libs.clone(),
        required_outputs: args.required.clone(),
        strict: args.strict,
        context: args.file.display().to_string(),
        cursor_limit: None,
        ledger: None,
        // `--engine` and `--provenance` are the command line's spelling
        // of the options' engine preference; without them `run_engine`
        // resolves to `Engine::default()`, the most native form the
        // build has.
        engine: args.run_engine(),
    }
}

/// Compile the program once, on the run engine.
fn compile_ast(ast: &PolydatFile, source: &str, args: &CompileArgs) -> Result<Compiled, String> {
    let options = compile_options(args);
    let engine = options.engine;
    let mut events = CompileEventLog::new();
    take_audit();
    let start = Instant::now();
    let root = compile_ast_with_engine(ast, source, &options, Some(&mut events), engine)
        .map_err(|e| e.to_string())?;
    let elapsed = start.elapsed();
    let run_plan = root.plan();
    let root = root.into_program();
    let program = root.clone().as_interpreter();
    Ok(Compiled {
        program,
        root,
        engine,
        run_plan,
        events,
        audit: take_audit(),
        elapsed,
    })
}

/// The interpreter's program, for the commands that describe a program
/// node by node: the run's own when the run engine is the interpreter,
/// else compiled now under `--cones`, its events and audit lines
/// replacing the run engine's in `compiled`.
fn describe(
    ast: &PolydatFile,
    source: &str,
    args: &CompileArgs,
    compiled: &mut Compiled,
) -> Result<Arc<PolydatProgram>, String> {
    if let Some(program) = &compiled.program {
        return Ok(program.clone());
    }
    let options = compile_options(args);
    let mut events = CompileEventLog::new();
    take_audit();
    let kernel = compile_ast_with_engine(
        ast,
        source,
        &options,
        Some(&mut events),
        KernelEngine::Interpreter(args.cones()),
    )
    .map_err(|e| e.to_string())?;
    let program = kernel
        .into_program()
        .as_interpreter()
        .expect("the interpreter's program");
    compiled.events = events;
    compiled.audit = take_audit();
    compiled.program = Some(program.clone());
    Ok(program)
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn run(args: RunArgs) -> Result<(), String> {
    install_audit(!args.quiet);
    let source = read_source(&args.compile.file)?;

    // Assignments are a program transform: each `name=value` rewrites the
    // extern or input declaration, and the program's typing fuses the
    // text to the declared type. Nothing is set on states at runtime.
    let assignments: Vec<(String, String)> = args
        .assignments
        .iter()
        .chain(args.sets.iter())
        .map(|a| parse_assignment(a))
        .collect::<Result<_, _>>()?;
    let mut ast = parse_program(&source, &args.compile)?;
    assign_values(&mut ast, &assignments)?;

    // Probe compile: discovers the declared outputs the emit transform
    // names, read from a kernel of the run engine's program.
    let probe = compile_ast(&ast, &source, &args.compile)?;
    let shape = probe.root.clone().create_kernel();
    let selected: Vec<String> = match &args.outputs {
        Some(list) => list
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        None => {
            let inputs = shape.input_names();
            shape
                .output_names()
                .into_iter()
                .filter(|n| !n.starts_with("__") && !inputs.iter().any(|i| i == n))
                .collect()
        }
    };
    // Selected outputs must exist where they will be pulled: in every
    // traversal body when the program traverses, else at the root.
    if shape.traversals().is_empty() {
        for name in &selected {
            if shape.output_index(name).is_none() {
                return Err(format!(
                    "no output named '{name}'; declared outputs: {}",
                    shape.output_names().join(", ")
                ));
            }
        }
    } else if args.outputs.is_some() {
        for t in shape.traversals() {
            for name in &selected {
                if t.program.output_index(name).is_none() {
                    return Err(format!(
                        "no output named '{name}' in the body of `for {}`; its outputs: {}",
                        t.source_text,
                        body_wire_names(&t.program).join(", ")
                    ));
                }
            }
        }
    }

    // The emit transform: one appended binding that names the selected
    // wires. Everything else about emission is the node's business.
    // A program with top-level traversals runs in traversal mode: the
    // emit transform goes inside each for body, where it sees the
    // body's scope, and the run activates the traversals.
    let traversal_mode = !shape.traversals().is_empty();

    // `--emit tile:<name>` selects the tile and the text format: the
    // tile's rendered text is the row. The tile lives where the emit
    // binding will: in every traversal body, else at the root.
    let (emit_format, selected) = match &args.emit {
        Some(EmitSpec::Tile(name)) => {
            let present = if traversal_mode {
                shape
                    .traversals()
                    .iter()
                    .all(|t| t.program.output_index(name).is_some())
            } else {
                shape.output_index(name).is_some()
            };
            if !present {
                let known: Vec<String> = if traversal_mode {
                    shape
                        .traversals()
                        .iter()
                        .flat_map(|t| body_wire_names(&t.program))
                        .collect()
                } else {
                    shape.output_names()
                };
                return Err(format!(
                    "no tile or output named '{name}'; declared outputs: {}",
                    known.join(", ")
                ));
            }
            (Some(EmitFormat::Text), vec![name.clone()])
        }
        Some(EmitSpec::Format(f)) => (Some(*f), selected),
        None => (None, selected),
    };

    let emit_binding = |names: &[String]| -> Result<Statement, String> {
        let fmt_name = match emit_format {
            Some(EmitFormat::Map) => "map",
            Some(EmitFormat::Csv) => "csv",
            Some(EmitFormat::Jsonl) => "jsonl",
            Some(EmitFormat::Text) => "text",
            None => unreachable!(),
        };
        let binding = format!(
            "__emit := emit_row(\"{fmt_name}\", \"{}\", {})\n",
            names.join(","),
            names.join(", ")
        );
        let mut emit_ast = parse_source(&binding)?;
        Ok(emit_ast.statements.remove(0))
    };

    let compiled = if emit_format.is_some() {
        if traversal_mode {
            for stmt in ast.statements.iter_mut() {
                if let Statement::For(f) = stmt {
                    let explicit =
                        args.outputs.is_some() || matches!(args.emit, Some(EmitSpec::Tile(_)));
                    let names: Vec<String> = if explicit {
                        selected.clone()
                    } else {
                        body_output_names(&f.body)
                    };
                    f.body.push(emit_binding(&names)?);
                }
            }
        } else {
            ast.statements.push(emit_binding(&selected)?);
        }
        compile_ast(&ast, &source, &args.compile)?
    } else {
        probe
    };
    let mut compiled = compiled;

    if args.events {
        print!("{}", compiled.events.format());
    }
    if args.stats {
        let program = describe(&ast, &source, &args.compile, &mut compiled)?;
        print_stats(&program, &compiled, Report::Text);
    }

    if traversal_mode {
        return run_traversals(&args, &compiled, emit_format, &selected);
    }

    // Cursor narrowing. Each cursor declared `over <spec>` resolves to a
    // list of partitions; this run either represents one of them or, when
    // fibers and partitions match, spreads them one per fiber.
    let fibers = args.fibers.max(1);
    let mut plan = CursorPlan {
        per_cursor: Vec::new(),
        per_fiber: false,
    };
    {
        let mut probe = compiled.root.clone().create_kernel();
        for schema in probe.cursor_schemas().to_vec() {
            let parts = cursor_over_partitions_on(probe.as_mut(), &schema)?;
            if parts.is_empty() {
                continue;
            }
            let chosen: Vec<Partition> = match (parts.len(), args.partition) {
                (1, _) => vec![parts[0]],
                (_, Some(i)) => vec![*parts.get(i).ok_or_else(|| {
                    format!(
                        "cursor '{}' resolves to {} partitions; --partition {i} is out of range",
                        schema.name,
                        parts.len()
                    )
                })?],
                (n, None) if n == fibers => {
                    plan.per_fiber = true;
                    parts.clone()
                }
                (n, None) => {
                    return Err(format!(
                        "cursor '{}' resolves to {n} partitions; pass --partition INDEX to run one, or --fibers {n} to run one per fiber",
                        schema.name
                    ));
                }
            };
            plan.per_cursor.push((schema.name.clone(), chosen));
        }
        emit::take_rows();
    }
    if !args.quiet {
        for (name, parts) in &plan.per_cursor {
            for p in parts {
                eprintln!(
                    "cursor {name}: partition {}/{} [{}, {})",
                    p.idx + 1,
                    p.count.max(1),
                    p.start_ord,
                    p.end_ord
                );
            }
        }
    }

    // Which outputs each cycle pulls. With emission, pulling `__emit`
    // pulls everything it names; without it, pull the selection.
    let pull_names: Vec<String> = if emit_format.is_some() {
        compiled
            .root
            .clone()
            .create_kernel()
            .output_index("__emit")
            .ok_or("emit transform did not produce __emit")?;
        vec!["__emit".to_string()]
    } else {
        selected.to_vec()
    };
    // The coordinates: every input that is not an extern.
    let coord_count = shape.input_names().len() - shape.externs().len();

    let chunk = args.chunk.max(1);
    let total = args.cycles;
    let start_cycle = args.start;

    // Output sink.
    let sink: Box<dyn Write + Send> = match &args.out {
        Some(path) => Box::new(std::io::BufWriter::new(
            std::fs::File::create(path)
                .map_err(|e| format!("cannot create {}: {e}", path.display()))?,
        )),
        None => Box::new(std::io::BufWriter::new(std::io::stdout())),
    };
    let sink = Arc::new(Mutex::new(sink));
    if let Some(fmt) = emit_format {
        let names: Vec<&str> = selected.iter().map(String::as_str).collect();
        if let Some(h) = emit::header(fmt, &names) {
            writeln!(sink.lock().unwrap(), "{h}").map_err(|e| e.to_string())?;
        }
    }

    // Warmup on one kernel, untimed.
    if args.warmup > 0 {
        let mut kernel = compiled.root.clone().create_kernel();
        plan.apply(kernel.as_mut(), 0)?;
        let pulls = resolve_pulls(kernel.as_ref(), &pull_names);
        let mut coords = vec![0u64; coord_count];
        for c in 0..args.warmup {
            coords.fill(start_cycle.wrapping_add(c));
            drive_cycle(kernel.as_mut(), &coords);
            for &i in &pulls {
                kernel.pull_at(i);
            }
        }
        emit::take_rows();
    }

    // The run. Fibers claim chunks from a shared counter; each chunk's
    // rows travel with a sequence number so the writer can restore
    // cycle order when asked to.
    let next_chunk = AtomicU64::new(0);
    let chunk_count = total.div_ceil(chunk);
    let per_fiber = plan.per_fiber;
    let (tx, rx) = mpsc::channel::<(u64, Vec<String>)>();
    let fiber_busy: Mutex<Vec<Duration>> = Mutex::new(vec![Duration::ZERO; fibers]);
    let run_start = Instant::now();

    std::thread::scope(|s| {
        for fiber in 0..fibers {
            let root = compiled.root.clone();
            let tx = tx.clone();
            let pull_names = &pull_names;
            let next_chunk = &next_chunk;
            let fiber_busy = &fiber_busy;
            let emitting = emit_format.is_some();
            let plan = &plan;
            s.spawn(move || {
                let mut kernel = root.create_kernel();
                plan.apply(kernel.as_mut(), fiber)
                    .expect("the plan names cursors the program declares");
                // The outputs resolved once per fiber: no lookup per pull.
                let pulls = resolve_pulls(kernel.as_ref(), pull_names);
                let mut coords = vec![0u64; coord_count];
                let mut busy = Duration::ZERO;
                // Shared mode: fibers claim chunks of one cycle range.
                // Per-fiber mode: every fiber walks the whole range over
                // its own partition, and its rows sort after the
                // previous fiber's.
                let mut local = 0u64;
                loop {
                    let local_seq = if per_fiber {
                        let s = local;
                        local += 1;
                        s
                    } else {
                        next_chunk.fetch_add(1, Ordering::Relaxed)
                    };
                    if local_seq >= chunk_count {
                        break;
                    }
                    let seq = if per_fiber {
                        fiber as u64 * chunk_count + local_seq
                    } else {
                        local_seq
                    };
                    let lo = start_cycle.wrapping_add(local_seq * chunk);
                    let n = chunk.min(total - local_seq * chunk);
                    let t = Instant::now();
                    for i in 0..n {
                        coords.fill(lo.wrapping_add(i));
                        drive_cycle(kernel.as_mut(), &coords);
                        for &idx in &pulls {
                            kernel.pull_at(idx);
                        }
                    }
                    busy += t.elapsed();
                    if emitting {
                        let rows = emit::take_rows();
                        let _ = tx.send((seq, rows));
                    }
                }
                fiber_busy.lock().unwrap()[fiber] = busy;
            });
        }
        drop(tx);

        // Writer on the main thread.
        let mut out = sink.lock().unwrap();
        if args.unordered {
            for (_, rows) in rx {
                for r in rows {
                    let _ = writeln!(out, "{r}");
                }
            }
        } else {
            let mut expected = 0u64;
            let mut pending: BTreeMap<u64, Vec<String>> = BTreeMap::new();
            for (seq, rows) in rx {
                pending.insert(seq, rows);
                while let Some(rows) = pending.remove(&expected) {
                    for r in rows {
                        let _ = writeln!(out, "{r}");
                    }
                    expected += 1;
                }
            }
        }
        let _ = out.flush();
    });
    let wall = run_start.elapsed();

    if let Some(report) = args.timing {
        let busy = fiber_busy.into_inner().unwrap();
        let ran = if per_fiber {
            total * fibers as u64
        } else {
            total
        };
        print_timing(report, &compiled, ran, fibers, wall, &busy);
    }
    Ok(())
}

/// Which partition each cursor takes, per fiber.
struct CursorPlan {
    /// Cursor name and its chosen partitions. One entry means every
    /// fiber shares it; `fibers` entries means fiber `i` takes entry `i`.
    per_cursor: Vec<(String, Vec<Partition>)>,
    per_fiber: bool,
}

impl CursorPlan {
    fn apply(&self, kernel: &mut dyn Kernel, fiber: usize) -> Result<(), String> {
        for (name, parts) in &self.per_cursor {
            let p = if parts.len() == 1 {
                &parts[0]
            } else {
                &parts[fiber.min(parts.len() - 1)]
            };
            kernel.set_cursor(name, p).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// Wires a compiled body computes itself: its outputs that are not its
/// inputs (elements, cascade, cycle) and not compiler-internal.
fn body_wire_names(program: &PolydatProgram) -> Vec<String> {
    program
        .own_output_names()
        .into_iter()
        .filter(|n| !n.starts_with("__") && program.find_input(n).is_none())
        .map(str::to_string)
        .collect()
}

/// Binding targets a for body declares, excluding compiler-internal names.
fn body_output_names(body: &[Statement]) -> Vec<String> {
    let mut out = Vec::new();
    for stmt in body {
        if let Statement::Binding(b) = stmt {
            for t in &b.targets {
                if !t.starts_with("__") && !out.contains(t) {
                    out.push(t.clone());
                }
            }
        }
    }
    out
}

/// Run every top-level traversal of the program (SRD 113 §3.6). Fibers
/// take activations by index, stride `fibers`, so no coordination is
/// needed. Each activation runs its cycles under the §3.4 rule, capped
/// by `--cycles`. Rows are ordered by traversal and activation index
/// unless `--unordered` is given.
fn run_traversals(
    args: &RunArgs,
    compiled: &Compiled,
    emit_format: Option<EmitFormat>,
    selected: &[String],
) -> Result<(), String> {
    let fibers = args.fibers.max(1);
    let cap = args.cycles.max(1);

    let sink: Box<dyn Write + Send> = match &args.out {
        Some(path) => Box::new(std::io::BufWriter::new(
            std::fs::File::create(path)
                .map_err(|e| format!("cannot create {}: {e}", path.display()))?,
        )),
        None => Box::new(std::io::BufWriter::new(std::io::stdout())),
    };
    let sink = Arc::new(Mutex::new(sink));

    // Open every traversal against a root on the run engine, positioned
    // at --start.
    let mut root = compiled.root.clone().create_kernel();
    root.set_inputs(&[args.start]);
    let streams = root.traverse_all()?;
    if !args.quiet {
        for (i, s) in streams.iter().enumerate() {
            eprintln!(
                "traversal {i}: for {}  ({} activations)",
                s.traversal().source_text,
                s.len()
            );
        }
    }
    // Header rows travel with each traversal's first activation, so a
    // format with a header (csv) labels each traversal's columns even
    // when the bodies differ.
    let headers: Vec<Option<String>> = streams
        .iter()
        .map(|s| {
            let fmt = emit_format?;
            let names: Vec<String> = if args.outputs.is_some() {
                selected.to_vec()
            } else {
                body_wire_names(&s.traversal().program)
            };
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            emit::header(fmt, &refs)
        })
        .collect();

    // Sequence numbers: activation index within a traversal, offset by
    // the activations of the traversals before it.
    let offsets: Vec<u64> = streams
        .iter()
        .scan(0u64, |acc, s| {
            let o = *acc;
            *acc += s.len() as u64;
            Some(o)
        })
        .collect();
    let total_activations: u64 = streams.iter().map(|s| s.len() as u64).sum();

    let (tx, rx) = mpsc::channel::<(u64, Vec<String>)>();
    let cycles_run = AtomicU64::new(0);
    let fiber_busy: Mutex<Vec<Duration>> = Mutex::new(vec![Duration::ZERO; fibers]);
    let run_start = Instant::now();

    std::thread::scope(|s| {
        for fiber in 0..fibers {
            let tx = tx.clone();
            let streams = &streams;
            let offsets = &offsets;
            let headers = &headers;
            let cycles_run = &cycles_run;
            let fiber_busy = &fiber_busy;
            let emitting = emit_format.is_some();
            s.spawn(move || {
                let mut busy = Duration::ZERO;
                for (t, stream) in streams.iter().enumerate() {
                    let pull_names: Vec<String> = if emitting {
                        vec!["__emit".to_string()]
                    } else if args.outputs.is_some() {
                        selected.to_vec()
                    } else {
                        body_wire_names(&stream.traversal().program)
                    };
                    let mut i = fiber;
                    while i < stream.len() {
                        let start = Instant::now();
                        // Every activation runs on the run engine.
                        let activation: Result<Box<dyn CycleKernel>, String> = stream
                            .activation_on(i, compiled.engine)
                            .map(|act| Box::new(act) as Box<dyn CycleKernel>);
                        let rows = match activation {
                            Ok(mut act) => {
                                let n = act.count().min(cap);
                                let mut rows = Vec::new();
                                if i == 0
                                    && let Some(h) = &headers[t]
                                {
                                    rows.push(h.clone());
                                }
                                for c in 0..n {
                                    let kernel = act.at(c);
                                    if emitting {
                                        kernel.pull("__emit");
                                    } else {
                                        let line: Vec<String> = pull_names
                                            .iter()
                                            .map(|name| {
                                                format!(
                                                    "{name}={}",
                                                    kernel.pull(name).to_display_string()
                                                )
                                            })
                                            .collect();
                                        rows.push(format!("{t}/{i}/{c} {}", line.join(" ")));
                                    }
                                }
                                cycles_run.fetch_add(n, Ordering::Relaxed);
                                if emitting {
                                    rows.extend(emit::take_rows());
                                }
                                rows
                            }
                            Err(e) => vec![format!("error: activation {i} of traversal {t}: {e}")],
                        };
                        busy += start.elapsed();
                        let _ = tx.send((offsets[t] + i as u64, rows));
                        i += fibers;
                    }
                }
                fiber_busy.lock().unwrap()[fiber] = busy;
            });
        }
        drop(tx);

        let mut out = sink.lock().unwrap();
        if args.unordered {
            for (_, rows) in rx {
                for r in rows {
                    let _ = writeln!(out, "{r}");
                }
            }
        } else {
            let mut expected = 0u64;
            let mut pending: BTreeMap<u64, Vec<String>> = BTreeMap::new();
            for (seq, rows) in rx {
                pending.insert(seq, rows);
                while let Some(rows) = pending.remove(&expected) {
                    for r in rows {
                        let _ = writeln!(out, "{r}");
                    }
                    expected += 1;
                }
            }
        }
        let _ = out.flush();
    });
    let wall = run_start.elapsed();

    if let Some(report) = args.timing {
        let busy = fiber_busy.into_inner().unwrap();
        let ran = cycles_run.load(Ordering::Relaxed);
        if !args.quiet {
            eprintln!(
                "{total_activations} activations across {} traversal(s)",
                streams.len()
            );
        }
        print_timing(report, compiled, ran, fibers, wall, &busy);
    }
    Ok(())
}

fn print_timing(
    report: Report,
    compiled: &Compiled,
    cycles: u64,
    fibers: usize,
    wall: Duration,
    busy: &[Duration],
) {
    let secs = wall.as_secs_f64();
    let per_cycle_ns = if cycles > 0 {
        wall.as_nanos() as f64 / cycles as f64
    } else {
        0.0
    };
    let rate = if secs > 0.0 {
        cycles as f64 / secs
    } else {
        0.0
    };
    let busy_total: Duration = busy.iter().sum();
    let fiber_ns = if cycles > 0 {
        busy_total.as_nanos() as f64 / cycles as f64
    } else {
        0.0
    };
    match report {
        Report::Text => {
            println!();
            println!("compile      {:?}", compiled.elapsed);
            println!(
                "engine       {}: {}",
                compiled.root.engine(),
                compiled.run_plan
            );
            println!("cycles       {cycles}");
            println!("fibers       {fibers}");
            println!("wall         {wall:?}");
            println!("throughput   {:.3} M cycles/s", rate / 1e6);
            println!("wall/cycle   {per_cycle_ns:.1} ns");
            println!("fiber/cycle  {fiber_ns:.1} ns  (busy time summed over fibers, per cycle)");
            for (i, b) in busy.iter().enumerate() {
                println!("  fiber {i:<3} busy {b:?}");
            }
        }
        Report::Json => {
            let obj = serde_json::json!({
                "compile_ns": compiled.elapsed.as_nanos() as u64,
                "engine": compiled.root.engine().to_string(),
                "plan": {
                    "native_segments": compiled.run_plan.native_segments,
                    "closure_steps": compiled.run_plan.closure_steps,
                    "interpreted_nodes": compiled.run_plan.interpreted_nodes,
                },
                "cycles": cycles,
                "fibers": fibers,
                "wall_ns": wall.as_nanos() as u64,
                "cycles_per_second": rate,
                "wall_ns_per_cycle": per_cycle_ns,
                "fiber_ns_per_cycle": fiber_ns,
                "fiber_busy_ns": busy.iter().map(|b| b.as_nanos() as u64).collect::<Vec<_>>(),
            });
            println!("{obj}");
        }
    }
}

// ---------------------------------------------------------------------------
// check
// ---------------------------------------------------------------------------

fn check(args: CheckArgs) -> Result<(), String> {
    install_audit(true);
    let source = read_source(&args.compile.file)?;
    let ast = parse_program(&source, &args.compile)?;
    let mut compiled = compile_ast(&ast, &source, &args.compile)?;
    let program = describe(&ast, &source, &args.compile, &mut compiled)?;
    let program = &program;
    match args.format {
        Report::Text => {
            println!(
                "ok: {} nodes, {} wires, {} inputs, {} outputs, compiled in {:?}",
                program.node_count(),
                program.wire_count(),
                program.input_names().len(),
                program.output_count(),
                compiled.elapsed
            );
            if args.events {
                print!("{}", compiled.events.format());
            }
            if args.stats {
                print_stats(program, &compiled, Report::Text);
            }
            if args.manifest {
                for e in extract_manifest(program) {
                    println!(
                        "{:<24} {:?}{}",
                        e.name,
                        e.port_type,
                        modifier_suffix(&e.modifier)
                    );
                }
            }
        }
        Report::Json => {
            let mut obj = serde_json::Map::new();
            obj.insert("ok".into(), true.into());
            obj.insert(
                "compile_ns".into(),
                (compiled.elapsed.as_nanos() as u64).into(),
            );
            obj.insert("stats".into(), stats_json(program, &compiled));
            if args.manifest {
                let m: Vec<serde_json::Value> = extract_manifest(program)
                    .into_iter()
                    .map(|e| serde_json::json!({"name": e.name, "type": format!("{:?}", e.port_type), "const": e.modifier.has(WireModifier::Const), "shared": e.modifier.has(WireModifier::Shared)}))
                    .collect();
                obj.insert("manifest".into(), m.into());
            }
            if args.events {
                let ev: Vec<String> = compiled
                    .events
                    .events()
                    .iter()
                    .map(|e| format!("{e:?}"))
                    .collect();
                obj.insert("events".into(), ev.into());
            }
            println!("{}", serde_json::Value::Object(obj));
        }
    }
    Ok(())
}

fn modifier_suffix(m: &polydat::dsl::ast::BindingModifier) -> String {
    let mut parts = Vec::new();
    if m.has(WireModifier::Const) {
        parts.push("const");
    }
    if m.has(WireModifier::Shared) {
        parts.push("shared");
    }
    if m.has(WireModifier::Volatile) {
        parts.push("volatile");
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("  [{}]", parts.join(" "))
    }
}

/// A fused native cone appears in the program as one node whose name
/// carries its member list. It hosts P3 code even though the node
/// itself is dispatched by the interpreter.
fn is_cone(name: &str) -> bool {
    name.starts_with("jit_cone[")
}

/// Node counts by engine: (interpreter nodes, closure nodes, native
/// cones).
fn level_counts(program: &PolydatProgram) -> (usize, usize, usize) {
    let mut c = (0, 0, 0);
    for i in 0..program.node_count() {
        if is_cone(&program.node_meta(i).name) {
            c.2 += 1;
            continue;
        }
        match program.node_compile_level(i) {
            polydat::ast::CompileLevel::Phase1 => c.0 += 1,
            polydat::ast::CompileLevel::Phase2 => c.1 += 1,
            polydat::ast::CompileLevel::Phase3 => c.2 += 1,
        }
    }
    c
}

fn engine_label(program: &PolydatProgram, idx: usize) -> String {
    let name = &program.node_meta(idx).name;
    if is_cone(name) {
        "P3 native cone".to_string()
    } else {
        match program.node_compile_level(idx) {
            polydat::ast::CompileLevel::Phase1 => "P1 interpreter".to_string(),
            polydat::ast::CompileLevel::Phase2 => "P2 closure".to_string(),
            polydat::ast::CompileLevel::Phase3 => "P3 native".to_string(),
        }
    }
}

fn print_stats(program: &PolydatProgram, compiled: &Compiled, _report: Report) {
    let (p1, p2, p3) = level_counts(program);
    println!();
    println!("nodes         {}", program.node_count());
    println!("wires         {}", program.wire_count());
    println!("avg degree    {:.2}", program.avg_degree());
    println!("inputs        {}", program.input_names().len());
    println!(
        "outputs       {}  (const {}, shared {}, side-effect {})",
        program.output_count(),
        program.const_outputs().len(),
        program.shared_outputs().len(),
        program.outputs_with_side_effects().len()
    );
    println!("engines       P1 nodes {p1}  P2 nodes {p2}  P3 cones {p3}");
    println!(
        "run engine    {}: {}",
        compiled.root.engine(),
        compiled.run_plan
    );
    println!("deterministic {}", program.is_deterministic());
    println!("cursors       {}", program.cursor_schemas().len());
    println!(
        "traversals    {} (producers {})",
        program.traversals().len(),
        program.producers().len()
    );
    println!(
        "programs      {} (root plus one per for body at every depth)",
        polydat::kernel::program_count(program)
    );
    println!(
        "events        {} recorded ({} warnings, {} advisories)",
        compiled.events.events().len(),
        compiled.events.warnings().len(),
        compiled.events.advisories().len()
    );
    println!("compile       {:?}", compiled.elapsed);
}

fn stats_json(program: &PolydatProgram, compiled: &Compiled) -> serde_json::Value {
    let (p1, p2, p3) = level_counts(program);
    serde_json::json!({
        "nodes": program.node_count(),
        "wires": program.wire_count(),
        "avg_degree": program.avg_degree(),
        "inputs": program.input_names().len(),
        "outputs": program.output_count(),
        "const_outputs": program.const_outputs().len(),
        "shared_outputs": program.shared_outputs().len(),
        "side_effect_outputs": program.outputs_with_side_effects().len(),
        "engine_nodes": {"p1": p1, "p2": p2, "p3": p3},
        "run_engine": compiled.root.engine().to_string(),
        "run_plan": {
            "native_segments": compiled.run_plan.native_segments,
            "closure_steps": compiled.run_plan.closure_steps,
            "interpreted_nodes": compiled.run_plan.interpreted_nodes,
        },
        "deterministic": program.is_deterministic(),
        "cursors": program.cursor_schemas().len(),
        "events": compiled.events.events().len(),
        "warnings": compiled.events.warnings().len(),
    })
}

// ---------------------------------------------------------------------------
// explain
// ---------------------------------------------------------------------------

fn explain(args: ExplainArgs) -> Result<(), String> {
    install_audit(false);
    let source = read_source(&args.compile.file)?;
    let phases: Vec<Phase> = if args.phases.is_empty() {
        ALL_PHASES.to_vec()
    } else {
        args.phases.clone()
    };

    // Lex and parse are narrated from their own results so a program
    // that fails later still explains its front end.
    let tokens = polydat::dsl::lexer::lex(&source)?;
    let ast = parse_program(&source, &args.compile)?;
    let mut compiled = compile_ast(&ast, &source, &args.compile)?;
    let program = describe(&ast, &source, &args.compile, &mut compiled)?;
    let program = &program;
    let events = compiled.events.events();

    let input_name = |idx: usize| program.input_name_by_idx(idx).unwrap_or("?").to_string();
    let wire_label = |w: &WireSource| match w {
        WireSource::Input(i) => format!("input {}", input_name(*i)),
        WireSource::NodeOutput(n, p) => {
            let meta = program.node_meta(*n);
            if meta.outs.len() > 1 {
                format!("{}.{}", meta.name, meta.outs[*p].name)
            } else {
                meta.name.clone()
            }
        }
    };

    for phase in phases {
        println!("== {} ==", phase_title(phase));
        match phase {
            Phase::Lex => {
                let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
                for t in &tokens {
                    let d = format!("{:?}", t.kind);
                    let k = d.split(['(', ' ']).next().unwrap_or(&d).to_string();
                    *kinds.entry(k).or_default() += 1;
                }
                println!(
                    "The lexer turned {} bytes of source into {} tokens.",
                    source.len(),
                    tokens.len()
                );
                for (k, n) in kinds {
                    println!("  {k:<14} {n}");
                }
            }
            Phase::Parse => {
                println!(
                    "The parser produced {} top-level statements.",
                    ast.statements.len()
                );
                for stmt in &ast.statements {
                    match stmt {
                        Statement::InputDecl(d) => println!(
                            "  input     {}{}",
                            d.name,
                            d.ty.as_ref().map(|t| format!(": {t}")).unwrap_or_default()
                        ),
                        Statement::ExternPort(p) => println!(
                            "  extern    {}: {}{}",
                            p.name,
                            p.typ,
                            if p.default.is_some() {
                                " (with default)"
                            } else {
                                ""
                            }
                        ),
                        Statement::Cursor(c) => println!(
                            "  cursor    {}{}",
                            c.name,
                            if c.over.is_some() { " over ..." } else { "" }
                        ),
                        Statement::ModuleDef(m) => println!(
                            "  module    {}({} params) -> ({} outputs), {} body statements",
                            m.name,
                            m.params.len(),
                            m.outputs.len(),
                            m.body.len()
                        ),
                        Statement::Pragma { name, .. } => println!("  pragma    {name}"),
                        Statement::For(f) => println!(
                            "  for       {} {{ {} statements }}",
                            f.source.to_text(),
                            f.body.len()
                        ),
                        Statement::Tile(t) => println!(
                            "  tile      {} : {} ({} pieces)",
                            t.name,
                            t.encoding.as_deref().unwrap_or("text"),
                            t.pieces.len()
                        ),
                        Statement::Binding(b) => {
                            let mods = modifier_suffix(&b.modifier);
                            let targets = if b.targets.len() > 1 {
                                format!("({})", b.targets.join(", "))
                            } else {
                                b.targets.join("")
                            };
                            println!("  binding   {targets} := {}{mods}", expr_summary(&b.value));
                        }
                    }
                }
                println!(
                    "Every function call in a binding becomes a node; every name becomes a wire."
                );
            }
            Phase::Inputs => {
                let names = program.input_names();
                println!(
                    "The program has {} input slots. Coordinates are advanced by set_inputs; externs are written by the host or a parent scope.",
                    names.len()
                );
                for (i, n) in names.iter().enumerate() {
                    let kind = program
                        .input_kind(i)
                        .map(|k| format!("{k:?}"))
                        .unwrap_or_default();
                    let ty = program
                        .input_port_type_by_idx(i)
                        .map(|t| format!("{t:?}"))
                        .unwrap_or_default();
                    let default = program
                        .input_default_by_idx(i)
                        .map(|v| format!(" default {}", v.to_display_string()))
                        .unwrap_or_default();
                    println!("  [{i}] {n:<24} {ty:<8} {kind}{default}");
                }
            }
            Phase::Modules => {
                let inlined: Vec<_> = events
                    .iter()
                    .filter(|e| matches!(e, CompileEvent::ModuleInlined { .. }))
                    .collect();
                if inlined.is_empty() {
                    println!("No modules were inlined. Every call resolved to a registered node.");
                }
                for e in inlined {
                    if let CompileEvent::ModuleInlined { name, nodes_added } = e {
                        println!(
                            "  module {name} was inlined, adding {nodes_added} nodes. The module boundary no longer exists in the graph."
                        );
                    }
                }
            }
            Phase::Wires => {
                println!(
                    "The assembler resolved every reference to a wire. {} nodes, {} wires, average in-degree {:.2}.",
                    program.node_count(),
                    program.wire_count(),
                    program.avg_degree()
                );
                for i in 0..program.node_count() {
                    let meta = program.node_meta(i);
                    let ins: Vec<String> = program.node_wiring(i).iter().map(wire_label).collect();
                    let outs: Vec<String> = meta
                        .outs
                        .iter()
                        .map(|o| format!("{}:{:?}", o.name, o.typ))
                        .collect();
                    println!(
                        "  [{i:>3}] {:<28} <- ({})  -> {}",
                        meta.name,
                        ins.join(", "),
                        outs.join(", ")
                    );
                }
                let resolved = events
                    .iter()
                    .filter(|e| matches!(e, CompileEvent::BindingResolved { .. }))
                    .count();
                if resolved > 0 {
                    println!("{resolved} bindings were resolved to node types by name.");
                }
            }
            Phase::Types => {
                let adapters: Vec<_> = events
                    .iter()
                    .filter(|e| {
                        matches!(
                            e,
                            CompileEvent::TypeAdapterInserted { .. }
                                | CompileEvent::TypeWidening { .. }
                        )
                    })
                    .collect();
                println!(
                    "Every port has a PortType and every wire was checked before the kernel could run."
                );
                if adapters.is_empty() {
                    println!("No adapters were needed: every wire already matched its port.");
                }
                for e in adapters {
                    match e {
                        CompileEvent::TypeAdapterInserted {
                            from_node,
                            to_node,
                            adapter,
                        } => {
                            println!("  {adapter} was inserted between {from_node} and {to_node}.")
                        }
                        CompileEvent::TypeWidening { from, to, context } => {
                            println!("  {from} widened to {to} at {context}.")
                        }
                        _ => {}
                    }
                }
                let mut by_type: BTreeMap<String, usize> = BTreeMap::new();
                for i in 0..program.node_count() {
                    for o in &program.node_meta(i).outs {
                        *by_type.entry(format!("{:?}", o.typ)).or_default() += 1;
                    }
                }
                println!("Output ports by type:");
                for (t, n) in by_type {
                    println!("  {t:<10} {n}");
                }
            }
            Phase::Lifecycle => {
                let consts = program.const_outputs();
                let shared = program.shared_outputs();
                println!(
                    "Values are classified by when they may change. Const wires are computed once at scope init; dynamic wires are computed per cycle; shared wires live in cells visible across fibers."
                );
                println!(
                    "  const   {}",
                    if consts.is_empty() {
                        "(none)".to_string()
                    } else {
                        consts.join(", ")
                    }
                );
                println!(
                    "  shared  {}",
                    if shared.is_empty() {
                        "(none)".to_string()
                    } else {
                        shared.join(", ")
                    }
                );
                let dynamic: Vec<&str> = program
                    .own_output_names()
                    .into_iter()
                    .filter(|n| !consts.contains(n) && !shared.contains(n))
                    .collect();
                println!(
                    "  dynamic {}",
                    if dynamic.is_empty() {
                        "(none)".to_string()
                    } else {
                        dynamic.join(", ")
                    }
                );
                let init_ports: usize = (0..program.node_count())
                    .map(|i| program.node_meta(i).ins.iter().filter(|s| matches!(s, Slot::Wire(p) if p.lifecycle == polydat::ast::Lifecycle::Init)).count())
                    .sum();
                println!(
                    "{init_ports} wire ports require init-time values; wiring a cycle-time value to one is an assembly error."
                );
            }
            Phase::Constants => {
                println!(
                    "Constants are baked into nodes at assembly. Init-time expressions were folded before the graph was frozen."
                );
                let mut baked = 0;
                for i in 0..program.node_count() {
                    let meta = program.node_meta(i);
                    for s in &meta.ins {
                        if let Slot::Const { name, value } = s {
                            baked += 1;
                            println!("  {}.{name} = {value:?}", meta.name);
                        }
                    }
                }
                if baked == 0 {
                    println!("  no baked constants");
                }
                for e in events {
                    if let CompileEvent::ConstantFolded { node, value } = e {
                        println!("  folded {node} to {value}");
                    }
                }
            }
            Phase::Fusion => {
                let fusions: Vec<_> = events
                    .iter()
                    .filter(|e| matches!(e, CompileEvent::FusionApplied { .. }))
                    .collect();
                println!(
                    "Fusion replaces chains of nodes with single nodes so dispatch happens once per chain."
                );
                if fusions.is_empty() {
                    println!("  no fusion patterns matched");
                }
                for e in fusions {
                    if let CompileEvent::FusionApplied {
                        pattern,
                        nodes_replaced,
                    } = e
                    {
                        println!("  {pattern} replaced {nodes_replaced} nodes");
                    }
                }
            }
            Phase::Engines => {
                let (p1, p2, p3) = level_counts(program);
                println!(
                    "Each node runs on one engine. P1 is the typed interpreter, P2 a compiled closure, P3 native code through Cranelift. A cone is a region of eligible nodes fused into one native function; the interpreter dispatches it as a single node."
                );
                println!(
                    "A run drives the {} engine: {}. The interpreter's program below is what this command and --stats describe.",
                    compiled.root.engine(),
                    compiled.run_plan
                );
                println!("  P1 nodes {p1}   P2 nodes {p2}   P3 cones {p3}");
                for i in 0..program.node_count() {
                    println!(
                        "  [{i:>3}] {:<28} {}",
                        program.node_meta(i).name,
                        engine_label(program, i)
                    );
                }
                let cone_lines: Vec<&String> = compiled
                    .audit
                    .iter()
                    .filter(|(_, m)| m.contains("cone"))
                    .map(|(_, m)| m)
                    .collect();
                if cone_lines.is_empty() {
                    println!(
                        "The cone extractor reported nothing; either the engine is off or no region qualified."
                    );
                } else {
                    println!("Cone extraction:");
                    for m in cone_lines {
                        println!("  {m}");
                    }
                }
                for e in events {
                    if let CompileEvent::CompileLevelSelected { node, level } = e {
                        println!("  {node} selected {level}");
                    }
                }
            }
            Phase::Provenance => {
                println!(
                    "Provenance records which inputs can invalidate each output. A pull only recomputes nodes reachable from a changed input."
                );
                for name in program.output_names() {
                    let Some((node, _)) = program.resolve_output(name) else {
                        continue;
                    };
                    let deps: Vec<String> = program
                        .input_provenance_for(node)
                        .map(|m| m.iter_ones().map(&input_name).collect())
                        .unwrap_or_default();
                    println!(
                        "  {name:<24} <- {}",
                        if deps.is_empty() {
                            "(constant)".to_string()
                        } else {
                            deps.join(", ")
                        }
                    );
                }
            }
            Phase::Outputs => {
                let manifest = extract_manifest(program);
                println!(
                    "The program exposes {} named outputs. A pull by name resolves to a node and port.",
                    manifest.len()
                );
                for e in manifest {
                    let (n, p) = program.resolve_output(&e.name).unwrap_or((0, 0));
                    println!(
                        "  {:<24} {:?}{}  (node {n} port {p})",
                        e.name,
                        e.port_type,
                        modifier_suffix(&e.modifier)
                    );
                }
                let se = program.outputs_with_side_effects();
                if !se.is_empty() {
                    println!("Outputs with side effects: {}", se.join(", "));
                }
                for s in program.cursor_schemas() {
                    println!(
                        "  cursor {} extent {:?} projections {}",
                        s.name,
                        s.extent,
                        s.projections.len()
                    );
                }
                println!("Deterministic: {}", program.is_deterministic());
            }
            Phase::Tiles => {
                let tiles: Vec<_> = events
                    .iter()
                    .filter(|e| matches!(e, CompileEvent::TileCompiled { .. }))
                    .collect();
                let holes: Vec<_> = events
                    .iter()
                    .filter(|e| matches!(e, CompileEvent::TileHoleTyped { .. }))
                    .collect();
                if tiles.is_empty() {
                    println!("No tiles in this program.");
                } else {
                    println!(
                        "Each tile compiled to a skeleton: static runs copied whole, encoded holes, branches, and projections whose bodies are programs of their own."
                    );
                    for e in &tiles {
                        if let CompileEvent::TileCompiled {
                            tile,
                            encoding,
                            statics,
                            static_bytes,
                            holes,
                            branches,
                            projections,
                            bodies,
                        } = e
                        {
                            println!(
                                "  {tile:<12} {encoding}: {statics} static run(s) totalling {static_bytes} bytes, {holes} hole(s), {branches} branch(es), {projections} projection(s)"
                            );
                            for (i, body) in bodies.iter().enumerate() {
                                println!("               projection body {i}:");
                                for line in body.lines() {
                                    println!("                 {line}");
                                }
                            }
                        }
                    }
                    println!(
                        "Every hole was typed before the tile compiled: a declared type wins, otherwise the wire's type; the hole's position says what the encoding expects there, and the two pick the encoder."
                    );
                    for e in holes {
                        if let CompileEvent::TileHoleTyped {
                            tile,
                            hole,
                            wire_type,
                            declared,
                            expectation,
                            encoder,
                            adapter,
                        } = e
                        {
                            let declared = declared
                                .as_ref()
                                .map(|d| format!(", declared {d}"))
                                .unwrap_or_default();
                            let adapter = adapter
                                .as_ref()
                                .map(|a| format!("  adapter {a}"))
                                .unwrap_or_default();
                            println!("  {tile:<12} ${{{hole}}}");
                            println!(
                                "               wire {wire_type}{declared}; expects {expectation}"
                            );
                            println!("               -> {encoder}{adapter}");
                        }
                    }
                }
            }
            Phase::Traversals => {
                let ts = program.traversals();
                let ps = program.producers();
                if ts.is_empty() && ps.is_empty() {
                    println!("No for forms. Every statement compiled into the one program above.");
                } else {
                    println!(
                        "Each for body is one program for the life of the parent, keyed by where it appears. Activation only allocates state over it."
                    );
                    for p in ps {
                        println!(
                            "  producer {} := for {}  (line {})",
                            p.name, p.source_text, p.span.line
                        );
                    }
                    explain_traversals(ts, 1);
                }
            }
        }
        println!();
    }

    let warnings = compiled.events.warnings();
    if !warnings.is_empty() {
        println!("== warnings ==");
        for w in warnings {
            println!("  {w:?}");
        }
    }
    Ok(())
}

fn explain_traversals(ts: &[polydat::dsl::traversal::Traversal], indent: usize) {
    let pad = "  ".repeat(indent);
    for t in ts {
        let elems: Vec<String> = t
            .elements
            .iter()
            .map(|(n, ty)| format!("{n}:{ty:?}"))
            .collect();
        let cascade: Vec<String> = t
            .cascade
            .iter()
            .map(|(n, ty)| format!("{n}:{ty:?}"))
            .collect();
        println!(
            "{pad}for {}  (line {}, col {})",
            t.source_text, t.span.line, t.span.col
        );
        println!(
            "{pad}  elements {}",
            if elems.is_empty() {
                "(none)".to_string()
            } else {
                elems.join(", ")
            }
        );
        println!(
            "{pad}  cascade  {}",
            if cascade.is_empty() {
                "(none)".to_string()
            } else {
                cascade.join(", ")
            }
        );
        println!(
            "{pad}  program  {} nodes, {} outputs, {} nested traversals",
            t.program.node_count(),
            t.program.output_count(),
            t.program.traversals().len()
        );
        explain_traversals(t.program.traversals(), indent + 2);
    }
}

fn phase_title(p: Phase) -> &'static str {
    match p {
        Phase::Lex => "lex: source to tokens",
        Phase::Parse => "parse: tokens to statements",
        Phase::Inputs => "inputs: coordinate and extern slots",
        Phase::Modules => "modules: inlining and translation",
        Phase::Wires => "wires: nodes and their connections",
        Phase::Types => "types: port checking and adapters",
        Phase::Lifecycle => "lifecycle: const, dynamic, and shared values",
        Phase::Constants => "constants: baked and folded values",
        Phase::Fusion => "fusion: chains collapsed to single nodes",
        Phase::Engines => "engines: P1, P2, and P3 selection",
        Phase::Provenance => "provenance: which inputs invalidate which outputs",
        Phase::Outputs => "outputs: the manifest",
        Phase::Traversals => "traversals: for bodies compiled once per lexical position",
        Phase::Tiles => "tiles: how each hole is typed and encoded",
    }
}

fn expr_summary(e: &polydat::dsl::ast::Expr) -> String {
    let s = polydat::dsl::pprint::pp_expr(e).replace('\n', " ");
    if s.chars().count() > 96 {
        format!("{}...", s.chars().take(96).collect::<String>())
    } else {
        s
    }
}

// ---------------------------------------------------------------------------
// viz
// ---------------------------------------------------------------------------

fn viz(args: VizArgs) -> Result<(), String> {
    install_audit(false);
    let source = read_source(&args.file)?;
    let out = match args.format {
        VizFormat::Dot => polydat::viz::polydat_to_dot(&source)?,
        VizFormat::Mermaid => polydat::viz::polydat_to_mermaid(&source)?,
        VizFormat::Svg => polydat::viz::polydat_to_svg(&source)?,
    };
    print!("{out}");
    Ok(())
}

/// An activation driven cycle by cycle through the `Kernel` trait,
/// whichever engine it runs on.
trait CycleKernel {
    fn count(&self) -> u64;
    fn at(&mut self, i: u64) -> &mut dyn Kernel;
}

impl CycleKernel for Activation<Box<dyn Kernel>> {
    fn count(&self) -> u64 {
        self.cycle_count()
    }
    fn at(&mut self, i: u64) -> &mut dyn Kernel {
        self.cycle(i)
    }
}
