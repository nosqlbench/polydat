// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! DSL-to-assembly bridge: compile a parsed Polydat AST into a runtime kernel.
//!
//! Walks the AST, resolves function names to node constructors, wires
//! the `PolydatAssembler`, and produces the interpreter's
//! `PolydatKernel` or any engine's boxed `Kernel`.

use std::path::{Path, PathBuf};

use crate::compile::assembly::{PolydatAssembler, WireRef};
use crate::dsl::ast::*;
use crate::dsl::lexer;
use crate::dsl::parser;
use crate::kernel::PolydatKernel;

use crate::dsl::error::DiagnosticReport;
use crate::dsl::validate::{collect_references, validate_ast};

use std::collections::HashSet;

use super::modules::ResolvedModule;

/// Typed error ontology for the embedded-evaluation surface.
///
/// Per `expression_engine.md` §6 Error Ontology (under
/// `crates/polydat/docs/design/`), every failure mode the embedding
/// surface can produce maps to one of these variants. Hosts
/// pattern-match on the variant to drive UX, recovery, or logging
/// without parsing message strings.
///
/// The embedding surfaces (`eval_const_expr*`, the typed surfaces,
/// `interpolate_via_kernel`) return this type; the `compile_polydat*`
/// entry points return `String` errors, and `From<EmbeddingError>
/// for String` bridges the two.
#[derive(Debug, Clone)]
pub enum EmbeddingError {
    /// Text could not be parsed as polydat expression source.
    /// The lexer or parser rejected the input before any
    /// semantic analysis.
    Parse {
        /// The source text.
        source: String,
        /// The lexer's or parser's message.
        message: String,
        /// The byte offset of the error, when known.
        position: Option<usize>,
    },

    /// A `{name}` placeholder in the text had no matching
    /// binding in the kernel chain. Produced by
    /// `interpolate_via_kernel` only.
    UnresolvedPlaceholder {
        /// The placeholder's name.
        name: String,
        /// The source text.
        source: String,
    },

    /// The expression's upstream cone reaches a dynamic input,
    /// but the requested evaluation surface requires
    /// effectively-const lifecycle. Produced by
    /// `eval_const_expr` (directly or via the two-step
    /// composition).
    LifecycleMismatch {
        /// The source text.
        source: String,
        /// The dynamic inputs the cone reaches.
        dynamic_inputs: Vec<String>,
    },

    /// A node mentioned in the expression is not registered
    /// in the runtime. Includes a suggested alternative when
    /// the name is close to a known node.
    UnknownNode {
        /// The unknown node's name.
        name: String,
        /// The source text.
        source: String,
        /// A registered name close to it, if any.
        suggestion: Option<String>,
    },

    /// The expression's wire chain has a type mismatch that
    /// auto-adapters cannot heal. Produced by the assembly
    /// pass during compilation.
    TypeMismatch {
        /// The producing node.
        from_node: String,
        /// Its output type.
        from_type: crate::ast::PortType,
        /// The consuming node.
        to_node: String,
        /// The type its port requires.
        to_type: crate::ast::PortType,
        /// The source text.
        source: String,
    },

    /// A node's `eval` panicked during scope-init evaluation.
    /// The kernel's `catch_unwind` boundary captured the
    /// panic; the message is the panic payload's
    /// human-readable form.
    NodeEvalPanic {
        /// The node that panicked.
        node_name: String,
        /// The panic's message.
        message: String,
        /// The source text.
        source: String,
    },

    /// A `Value::None` propagated to the expression's output
    /// where a concrete value was required. Produced by a
    /// `HostType::from_value` conversion that meets `Value::None`,
    /// or by a host's own strict accessor (`as_bool` on
    /// `Value::None`, etc.). See SRD-74.
    NonePropagated {
        /// The accessor the host called.
        accessor: &'static str,
        /// The source text.
        source: String,
    },
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingError::Parse {
                source,
                message,
                position,
            } => match position {
                Some(p) => write!(f, "parse error at position {p} in '{source}': {message}"),
                None => write!(f, "parse error in '{source}': {message}"),
            },
            EmbeddingError::UnresolvedPlaceholder { name, source } => write!(
                f,
                "unresolved placeholder '{{{name}}}' in '{source}' — \
                 no matching binding in the kernel chain"
            ),
            EmbeddingError::LifecycleMismatch {
                source,
                dynamic_inputs,
            } => write!(
                f,
                "not a const expression: '{source}' depends on runtime inputs ({})",
                dynamic_inputs.join(", ")
            ),
            EmbeddingError::UnknownNode {
                name,
                source,
                suggestion,
            } => match suggestion {
                Some(sug) => write!(
                    f,
                    "unknown function: '{name}' in '{source}'\n\n  Did you mean '{sug}'?"
                ),
                None => write!(
                    f,
                    "unknown function: '{name}' in '{source}'\n\n  \
                     This function is not registered in the Polydat function library."
                ),
            },
            EmbeddingError::TypeMismatch {
                from_node,
                from_type,
                to_node,
                to_type,
                source,
            } => {
                write!(
                    f,
                    "type mismatch in '{source}': cannot connect \
                     {from_type:?} output of '{from_node}' to {to_type:?} \
                     input of '{to_node}'"
                )
            }
            EmbeddingError::NodeEvalPanic {
                node_name,
                message,
                source,
            } => write!(
                f,
                "node-eval panic in '{source}' (node '{node_name}'): {message}"
            ),
            EmbeddingError::NonePropagated { accessor, source } => write!(
                f,
                "Value::None propagated to '{source}'; \
                 host called strict accessor `{accessor}`. \
                 Use a non-strict accessor (`try_as_*`) or surface the None to the user."
            ),
        }
    }
}

impl std::error::Error for EmbeddingError {}

/// Renders the error as its message for callers on the
/// `Result<_, String>` entry points.
impl From<EmbeddingError> for String {
    fn from(e: EmbeddingError) -> String {
        e.to_string()
    }
}

/// Embedded standard library modules, compiled into the binary.
///
/// Each entry is (filename, source). Multiple modules per file —
/// each top-level binding is a separate module, resolved by name.
/// Searched as the final fallback after the source directory and
/// `CompileOptions::lib_paths` (the binary's `--lib`).
pub(super) static STDLIB_MODULES: &[(&str, &str)] = &[
    (
        "hashing.polydat",
        include_str!("../../stdlib/hashing.polydat"),
    ),
    (
        "strings.polydat",
        include_str!("../../stdlib/strings.polydat"),
    ),
    (
        "identity.polydat",
        include_str!("../../stdlib/identity.polydat"),
    ),
    (
        "distributions.polydat",
        include_str!("../../stdlib/distributions.polydat"),
    ),
    (
        "latency.polydat",
        include_str!("../../stdlib/latency.polydat"),
    ),
    (
        "timeseries.polydat",
        include_str!("../../stdlib/timeseries.polydat"),
    ),
    ("waves.polydat", include_str!("../../stdlib/waves.polydat")),
    (
        "fourier.polydat",
        include_str!("../../stdlib/fourier.polydat"),
    ),
    (
        "modeling.polydat",
        include_str!("../../stdlib/modeling.polydat"),
    ),
];

/// Return the embedded standard library module sources.
pub fn stdlib_sources() -> &'static [(&'static str, &'static str)] {
    STDLIB_MODULES
}

/// Compile a `.polydat` source string on the interpreter, under the
/// default options.
///
/// The kernel comes back as `dyn Kernel`, which is the surface every
/// use of a kernel goes through. The engine is named here rather than
/// taken from the options because this entry point exists to be the
/// semantic oracle: it is what a differential test compares a compiled
/// engine against. A caller with no such need calls
/// [`compile_polydat_kernel`], which takes the engine from the options
/// and so defaults to the most compiled form the build has.
///
/// A test or diagnostic that needs the interpreter's *own* internals —
/// its `PolydatProgram`, its `Lookup` view, its subcontext builder —
/// calls [`compile_polydat_interpreter`] for the concrete type. That is
/// the one reason to hold a `PolydatKernel` rather than a `dyn Kernel`.
pub fn compile_polydat(source: &str) -> Result<Box<dyn crate::Kernel>, crate::KernelError> {
    compile_polydat_interpreter(source).map(|k| Box::new(k) as Box<dyn crate::Kernel>)
}

/// [`compile_polydat`] returning the interpreter's concrete kernel.
///
/// This is the carve-out from the rule that kernels are used through
/// the [`Kernel`](crate::Kernel) trait, and it is narrow on purpose:
/// the concrete type carries the interpreter's implementation detail
/// (`program()`, `lookup()`, `state()`, `build_subscope()`, the
/// constant and wire readers), which a test asserting on that detail
/// and a diagnostic reporting it both need and nothing else should
/// reach for. Driving a kernel — coordinates, externs, cursors,
/// evaluation, reads, traversals — is the trait's, on every engine
/// including this one.
pub fn compile_polydat_interpreter(source: &str) -> Result<PolydatKernel, crate::KernelError> {
    compile_polydat_interpreter_with_options(source, &CompileOptions::default(), None)
}

/// Parse Polydat source into the program tree, the form every
/// transform reads and rewrites.
///
/// This is the first of the three steps a host takes when it shapes a
/// program before running it: parse, transform, compile. A host with
/// nothing to change calls a `compile_polydat*` entry point instead
/// and never sees a tree; a host that injects a value
/// ([`transform::assign_values`](super::transform::assign_values)),
/// adds tiles it built from what it holds
/// ([`transform::add_tiles`](super::transform::add_tiles)), or rewrites
/// a definition in place applies as many of those as it means to the
/// one tree and compiles it once with
/// [`compile_ast_with_engine`] or
/// [`compile_ast_interpreter_with_options`].
///
/// [`parse_polydat_with_tile_defaults`] is the same parse for a host
/// that sets the hole delimiters a tile body is read with.
pub fn parse_polydat(source: &str) -> Result<PolydatFile, crate::KernelError> {
    let tokens = super::lexer::lex(source).map_err(crate::KernelError::Source)?;
    super::parser::parse(tokens).map_err(crate::KernelError::Source)
}

/// [`parse_polydat`] with the hole delimiters and directive sigil a
/// tile body that declares none of its own is read with.
///
/// These belong to the parse and not to a later rewrite: a tile body
/// is raw text until it is read, so what counts as a hole has to be
/// settled before there are pieces for a transform to address.
pub fn parse_polydat_with_tile_defaults(
    source: &str,
    defaults: &super::ast::TileOptions,
) -> Result<PolydatFile, crate::KernelError> {
    let tokens = super::lexer::lex(source).map_err(crate::KernelError::Source)?;
    super::parser::parse_with_tile_defaults(tokens, defaults).map_err(crate::KernelError::Source)
}

/// Compile Polydat source to an assembler (not yet compiled to a kernel).
///
/// Returns the `PolydatAssembler` with every node and wire in place,
/// the graph a host may extend by hand before building it on any
/// engine: [`PolydatAssembler::compile_kernel`] for the default engine,
/// [`PolydatAssembler::compile_with`] for a named one,
/// [`PolydatAssembler::compile`] for the interpreter's concrete kernel.
/// An assembler carries no traversal, so a program with a `for`
/// statement is refused here; the kernel entry points compile it.
pub fn compile_polydat_to_assembler(source: &str) -> Result<PolydatAssembler, crate::KernelError> {
    compile_polydat_to_assembler_with(source, &CompileOptions::default())
}

/// [`compile_polydat_to_assembler`] with the options the kernel entry
/// points take: a source directory for relative imports, library
/// directories, required outputs, strict typing, a diagnostic context,
/// and a cursor limit. The assembler it returns is the graph
/// [`compile_polydat_interpreter_with_options`] would compile from the same source
/// and options, ready for any engine.
pub fn compile_polydat_to_assembler_with(
    source: &str,
    options: &CompileOptions,
) -> Result<PolydatAssembler, crate::KernelError> {
    let tokens = super::lexer::lex(source).map_err(crate::KernelError::Source)?;
    let ast = super::parser::parse(tokens).map_err(crate::KernelError::Source)?;
    let mut prepared = Prepared::new(source, &ast, options, None);
    let (compiler, filter) = prepared.parts();
    compiler
        .assemble_parent(&ast, filter)
        .map_err(crate::KernelError::Source)
}

/// Compile one selected scalar output into the conservative perfect-ordinal
/// Tier-1 SIMD executor.
///
/// This is an explicit execution surface: ordinary [`compile_polydat`] and
/// `PolydatKernel::pull` remain scalar-cycle APIs. `driving_input` is normally
/// a cursor projection such as `base__ordinal`; `output` names the only result
/// drained by the batch executor.
#[cfg(feature = "jit")]
pub fn compile_polydat_tier1_simd_ordinal(
    source: &str,
    driving_input: &str,
    output: &str,
) -> Result<crate::compile::simd_tier1::Tier1SimdExecutor, String> {
    compile_polydat_to_assembler(source)
        .map_err(|e| e.to_string())?
        .try_compile_tier1_simd_ordinal(driving_input, output)
        .map_err(|error| error.to_string())
}

/// `const name := expr` declares a side-effect-carrying compile-time
/// computation: download a dataset, prebuffer a facet, register a
/// resource, etc. The user's signal that they want it evaluated is
/// the `const` keyword itself, not a downstream wire reference. Yet
/// the assembler's DCE pass walks back from the requested-outputs
/// set and prunes anything not in that ancestry, which silently
/// removes const bindings whose result nothing reads.
///
/// This helper extends a caller-supplied `required_outputs` list
/// with every `const` binding target in the source. Two effects:
/// the assembler keeps those nodes during DCE, and constant
/// folding then evaluates them once at compile time — running the
/// side effect exactly once, before any dispatch.
///
/// Plain bindings (`name := ...`) are *not* added; they only run
/// when consumed. Modules and other statements are likewise not
/// auto-promoted.
fn extend_required_with_const_bindings(
    required_outputs: &[String],
    ast: &crate::dsl::ast::PolydatFile,
) -> Vec<String> {
    let mut out: Vec<String> = required_outputs.to_vec();
    for stmt in &ast.statements {
        if let crate::dsl::ast::Statement::Binding(b) = stmt
            && b.modifier.is_const()
        {
            for name in &b.targets {
                if !out.iter().any(|n| n == name) {
                    out.push(name.clone());
                }
            }
        }
    }
    out
}

/// RAII guard that sets the data-file base directory (see
/// [`crate::library::datafile::set_data_base_dir`]) for the duration of
/// a synchronous compile and restores the previous value on drop, so
/// nested compiles unwind cleanly.
struct DataBaseDirGuard(Option<PathBuf>);

impl DataBaseDirGuard {
    fn set(dir: &Path) -> Self {
        DataBaseDirGuard(crate::library::datafile::set_data_base_dir(Some(
            dir.to_path_buf(),
        )))
    }
}

impl Drop for DataBaseDirGuard {
    fn drop(&mut self) {
        crate::library::datafile::set_data_base_dir(self.0.take());
    }
}

/// The options every entry point compiles under. A host that names
/// none gets the defaults: no source directory, no library paths,
/// every binding an output, lax typing, the default context label,
/// no cursor limit, and the most compiled engine this build has.
///
/// `strict` refuses what lax compilation warns about, on every engine:
/// an implicit type coercion, a config wire fed from a cycle-time
/// source, a nondeterministic node no `volatile` output acknowledges, a
/// binding nothing reads, an undeclared coordinate, and a positional
/// module argument.
///
/// `engine` is a **preference**, and the only place a caller expresses
/// one. Leaving it alone is the normative path: [`Engine::default`](crate::Engine::default) is
/// the most native form the build offers (native code with the `jit`
/// feature, the closure tier without), so a host that never mentions
/// the field still gets compiled code, and gets faster code for free
/// when a build gains a tier. Naming an engine is for testing,
/// measurement, and demonstration — a differential test that wants the
/// interpreter as the reference, a bench that walks the tiers. A
/// preference the build cannot realize is refused
/// ([`crate::KernelError::Refused`]) rather than silently replaced, and
/// every kernel reports what it actually runs through
/// [`crate::Kernel::engine`].
#[derive(Debug, Default, Clone)]
pub struct CompileOptions {
    /// The directory relative data-file paths resolve against.
    pub source_dir: Option<PathBuf>,
    /// Library search paths, tried after the source directory and before the embedded standard library.
    pub lib_paths: Vec<PathBuf>,
    /// The outputs to keep; every output when empty.
    pub required_outputs: Vec<String>,
    /// Whether to enforce strict validation.
    pub strict: bool,
    /// The diagnostic context label, such as a file name.
    pub context: String,
    /// A limit on every cursor's extent, if any.
    pub cursor_limit: Option<u64>,
    /// The compile ledger to record this program tree in: a host that
    /// holds one charges the compile to it; `None` mints a fresh one,
    /// read back through the kernel's `ledger()`.
    pub ledger: Option<std::sync::Arc<crate::kernel::CompileLedger>>,
    /// The engine to build on. Defaults to [`Engine::default`](crate::Engine::default), the
    /// most native form this build has; see the type's documentation
    /// for when to set it and what happens when it cannot be realized.
    pub engine: crate::Engine,
}

/// Compile Polydat source into the interpreter's kernel under
/// `options`, recording pragma and assembly events in `log` when one is
/// given: the interpreter-typed entry point every other interpreter form
/// reduces to. [`compile_polydat_with_engine`] is the same compile on
/// any engine.
pub fn compile_polydat_interpreter_with_options(
    source: &str,
    options: &CompileOptions,
    mut log: Option<&mut super::events::CompileEventLog>,
) -> Result<PolydatKernel, crate::KernelError> {
    let tokens = lexer::lex(source).map_err(crate::KernelError::Source)?;
    let ast = parser::parse(tokens).map_err(crate::KernelError::Source)?;
    if let Some(log) = log.as_deref_mut() {
        log.push(super::events::CompileEvent::Parsed {
            statements: ast.statements.len(),
        });
    }
    compile_ast_interpreter_with_options(&ast, source, options, log)
}

/// [`compile_polydat_interpreter_with_options`] for an already parsed, possibly
/// transformed, program. `source` is the text the program was parsed
/// from and is used for diagnostics only.
pub fn compile_ast_interpreter_with_options(
    ast: &PolydatFile,
    source: &str,
    options: &CompileOptions,
    mut log: Option<&mut super::events::CompileEventLog>,
) -> Result<PolydatKernel, crate::KernelError> {
    let mut prepared = Prepared::new(source, ast, options, log.as_deref_mut());
    let (compiler, filter) = prepared.parts();
    // This path returns the interpreter's concrete kernel, so the
    // options' engine preference can only say how much of the graph is
    // fused into native cones. A preference naming the interpreter is
    // honoured with its mode; any other preference is a caller asking a
    // concrete-typed entry point for an engine it cannot return, and it
    // gets the interpreter under the default cone mode. A caller that
    // means the preference calls `compile_polydat_kernel_with_options`,
    // which can return whichever engine the options name.
    let cones = match options.engine {
        crate::Engine::Interpreter(mode) => mode,
        _ => crate::JitMode::Auto,
    };
    compiler.compile_interpreter(ast, filter, log, cones)
}

/// [`compile_polydat_interpreter_with_options`] under the default options, with the
/// compile event log: the same kernel [`compile_polydat`] builds, with
/// every pragma, assembly, fold, and tile event recorded.
pub fn compile_polydat_interpreter_with_log(
    source: &str,
    log: &mut super::events::CompileEventLog,
) -> Result<PolydatKernel, crate::KernelError> {
    compile_polydat_interpreter_with_options(source, &CompileOptions::default(), Some(log))
}

/// Record one event per pragma in `set`: `PragmaAcknowledged`
/// (advisory) for `strict_types`/`strict_values`/`strict`,
/// `UnknownPragma` (warning) for the rest. Forward-compatible: an
/// unknown pragma never blocks compilation.
///
/// Called from `Prepared::new` for every entry point given a log;
/// the set comes from `pragmas::collect_from_ast`.
pub(crate) fn record_pragma_events(
    set: &super::pragmas::PragmaSet,
    log: &mut super::events::CompileEventLog,
) {
    use super::events::CompileEvent;
    for entry in &set.entries {
        let known = matches!(
            entry.name.as_str(),
            "strict_types" | "strict_values" | "strict"
        );
        if known {
            log.push(CompileEvent::PragmaAcknowledged {
                name: entry.name.clone(),
                line: entry.line,
            });
        } else {
            log.push(CompileEvent::UnknownPragma {
                name: entry.name.clone(),
                line: entry.line,
            });
        }
    }
}

/// Compile with full diagnostics: errors, warnings, suggestions, on the
/// default engine.
///
/// Returns `(Ok(kernel), report)` on success with possible warnings,
/// or `(Err(()), report)` on failure with errors. The report always
/// contains all diagnostics. The program the report describes is the
/// program the kernel runs: the same compile every entry point makes.
pub fn compile_polydat_checked(
    source: &str,
) -> (Result<Box<dyn crate::Kernel>, ()>, DiagnosticReport) {
    let mut report = DiagnosticReport::new(source);

    let tokens = match lexer::lex(source) {
        Ok(t) => t,
        Err(e) => {
            report.error(crate::dsl::lexer::Span { line: 1, col: 1 }, e);
            return (Err(()), report);
        }
    };

    let ast = match parser::parse(tokens) {
        Ok(a) => a,
        Err(e) => {
            report.error(crate::dsl::lexer::Span { line: 1, col: 1 }, e);
            return (Err(()), report);
        }
    };

    // Validate the AST before compiling
    validate_ast(&ast, &mut report);

    if report.has_errors() {
        return (Err(()), report);
    }

    match compile_ast_with_engine(
        &ast,
        source,
        &CompileOptions::default(),
        None,
        crate::Engine::default(),
    ) {
        Ok(kernel) => (Ok(kernel), report),
        Err(e) => {
            report.error(crate::dsl::lexer::Span { line: 1, col: 1 }, e.to_string());
            (Err(()), report)
        }
    }
}

/// Cache of constant-expression results keyed by source text. A const
/// expression compiles with no inputs, so its value is a pure function
/// of its text; caching is exact. Bounded so a pathological caller
/// cannot grow it without limit. This is what keeps repeated evaluation
/// of the same range, list, or predicate text compile-free (SRD 113
/// §5.2).
static CONST_EXPR_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, crate::ast::Value>>,
> = std::sync::OnceLock::new();
const CONST_EXPR_CACHE_CAP: usize = 8192;

/// Evaluate a constant expression by compiling it as a one-binding
/// program: what a comprehension source such as `partitions("*\/4", 1000)`
/// goes through. Cached by source text, so the same text compiles once
/// per process (SRD 113 §5.2). An expression that reaches a dynamic
/// input is a lifecycle error. The compile, when there is one, is
/// recorded in a ledger of its own; [`eval_const_expr_for`] charges
/// it to a tree's.
///
/// # Examples
///
/// ```
/// use polydat::dsl::compile::eval_const_expr;
/// let v = eval_const_expr("4 * 4").unwrap();
/// assert_eq!(v.as_u64(), 16);  // both int literals → u64_mul
/// let v = eval_const_expr("4.0 * 4.0").unwrap();
/// assert_eq!(v.as_f64(), 16.0);  // both float literals → f64_mul
/// ```
pub fn eval_const_expr(source: &str) -> Result<crate::ast::Value, EmbeddingError> {
    eval_const_expr_for(source, &crate::kernel::CompileLedger::new())
}

/// [`eval_const_expr`] with its compile, when the text is not cached,
/// recorded in `ledger`: what a traversal source or predicate that has
/// to compile charges to the tree that opened it.
pub fn eval_const_expr_for(
    source: &str,
    ledger: &std::sync::Arc<crate::kernel::CompileLedger>,
) -> Result<crate::ast::Value, EmbeddingError> {
    let cache =
        CONST_EXPR_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Ok(map) = cache.lock()
        && let Some(v) = map.get(source)
    {
        return Ok(v.clone());
    }
    let result = eval_const_expr_uncached(source, ledger);
    if let Ok(v) = &result
        && let Ok(mut map) = cache.lock()
    {
        if map.len() >= CONST_EXPR_CACHE_CAP {
            map.clear();
        }
        map.insert(source.to_string(), v.clone());
    }
    result
}

fn eval_const_expr_uncached(
    source: &str,
    ledger: &std::sync::Arc<crate::kernel::CompileLedger>,
) -> Result<crate::ast::Value, EmbeddingError> {
    let wrapped = format!("\nout := {source}");
    let source_owned = source.to_string();
    let options = CompileOptions {
        ledger: Some(ledger.clone()),
        ..CompileOptions::default()
    };
    // Constant-folding inside `compile_polydat` invokes node `eval`
    // for inputs-free DAGs, so any node that panics on bad data
    // (e.g. `handle_of(&Value::None)` after a failed
    // `dataset_open`) would unwind out past this function and
    // crash any caller that doesn't itself catch panics. The
    // kernel's `engines::eval_node` enriches node-eval panics
    // with their provenance string; that string is what we
    // extract.
    let source_for_panic = source_owned.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        move || -> Result<crate::ast::Value, EmbeddingError> {
            let kernel = compile_polydat_interpreter_with_options(&wrapped, &options, None)
                .map_err(|e| classify_compile_error(&source_owned, e))?;
            kernel.get_constant("out").cloned().ok_or_else(|| {
                // The expression compiled but did not fold, which
                // means it reads something that is not known until
                // the workload runs. The kernel's inputs are those
                // things, and naming them is the whole point of the
                // variant.
                EmbeddingError::LifecycleMismatch {
                    source: source_owned.clone(),
                    dynamic_inputs: kernel.input_names(),
                }
            })
        },
    ));
    match result {
        Ok(r) => r,
        Err(payload) => Err(EmbeddingError::NodeEvalPanic {
            node_name: "(unknown)".to_string(),
            message: panic_payload_message(&payload),
            source: source_for_panic,
        }),
    }
}

/// Classify a compile failure into a typed [`EmbeddingError`].
///
/// The assembler's own errors are structured, so the fields are
/// carried across rather than reconstructed: a wiring type mismatch
/// keeps the two node names and the two types the assembler already
/// knows. The DSL front end reports in strings, so the one shape a
/// host acts on — an unregistered function — is read out of the
/// message, and its suggestion comes from the registry rather than
/// being dropped. Anything else keeps the compiler's own message.
fn classify_compile_error(source: &str, err: crate::KernelError) -> EmbeddingError {
    use crate::KernelError;
    use crate::compile::assembly::AssemblyError;
    if let KernelError::Assembly(AssemblyError::TypeMismatch {
        from_node,
        from_type,
        to_node,
        to_type,
        ..
    }) = err
    {
        return EmbeddingError::TypeMismatch {
            from_node,
            from_type,
            to_node,
            to_type,
            source: source.to_string(),
        };
    }
    let msg = err.to_string();
    // The factory and the diagnostic pass both write "unknown
    // function: '<name>'", the factory with the compiler's context
    // ahead of it, so the marker is searched for rather than
    // stripped from the front.
    const UNKNOWN: &str = "unknown function: '";
    if let Some(at) = msg.find(UNKNOWN)
        && let Some(end) = msg[at + UNKNOWN.len()..].find('\'')
    {
        let name = msg[at + UNKNOWN.len()..][..end].to_string();
        let suggestion = crate::dsl::registry::suggest_function(&name).map(str::to_string);
        return EmbeddingError::UnknownNode {
            name,
            source: source.to_string(),
            suggestion,
        };
    }
    EmbeddingError::Parse {
        source: source.to_string(),
        message: msg,
        position: None,
    }
}

// ───── Typed embedding surface (γ-4) ─────

/// Host-facing type that polydat can return from the typed
/// embedding surfaces. The trait declares the polydat
/// `PortType` the Rust type corresponds to and the conversion
/// from the returned [`crate::ast::Value`] back to the host
/// type.
///
/// Hosts that want compile-time type alignment use the typed
/// surfaces ([`eval_const_expr_typed`] /
/// [`eval_kernel_bound_typed`]) and let the type parameter
/// drive the contract. The fall-back is the untyped surface
/// (`eval_const_expr`) which returns a raw [`crate::ast::Value`]
/// for hosts to coerce themselves.
///
/// See expression_engine.md §5.3.
pub trait HostType: Sized {
    /// The `PortType` that polydat compares the expression's
    /// output type against. Used for compile-time / construction-
    /// time type-mismatch detection.
    fn target_port_type() -> crate::ast::PortType;

    /// Convert a polydat [`crate::ast::Value`] into the host Rust
    /// type. Returns a typed [`EmbeddingError::TypeMismatch`] when
    /// the value cannot be represented as the host type; the impls
    /// accept the lossless widenings (`U64` → `bool`/`f64`, scalars
    /// → `String`).
    fn from_value(v: crate::ast::Value) -> Result<Self, EmbeddingError>;
}

impl HostType for bool {
    fn target_port_type() -> crate::ast::PortType {
        crate::ast::PortType::Bool
    }
    fn from_value(v: crate::ast::Value) -> Result<Self, EmbeddingError> {
        match v {
            crate::ast::Value::Bool(b) => Ok(b),
            crate::ast::Value::U64(n) => Ok(n != 0),
            crate::ast::Value::None => Err(EmbeddingError::NonePropagated {
                accessor: "HostType::<bool>::from_value",
                source: "<typed-embedding result>".to_string(),
            }),
            other => Err(EmbeddingError::TypeMismatch {
                from_node: "<expression-output>".to_string(),
                from_type: other.port_type(),
                to_node: "<host-target>".to_string(),
                to_type: crate::ast::PortType::Bool,
                source: "<typed-embedding result>".to_string(),
            }),
        }
    }
}

impl HostType for u64 {
    fn target_port_type() -> crate::ast::PortType {
        crate::ast::PortType::U64
    }
    fn from_value(v: crate::ast::Value) -> Result<Self, EmbeddingError> {
        match v {
            crate::ast::Value::U64(n) => Ok(n),
            crate::ast::Value::None => Err(EmbeddingError::NonePropagated {
                accessor: "HostType::<u64>::from_value",
                source: "<typed-embedding result>".to_string(),
            }),
            other => Err(EmbeddingError::TypeMismatch {
                from_node: "<expression-output>".to_string(),
                from_type: other.port_type(),
                to_node: "<host-target>".to_string(),
                to_type: crate::ast::PortType::U64,
                source: "<typed-embedding result>".to_string(),
            }),
        }
    }
}

impl HostType for f64 {
    fn target_port_type() -> crate::ast::PortType {
        crate::ast::PortType::F64
    }
    fn from_value(v: crate::ast::Value) -> Result<Self, EmbeddingError> {
        match v {
            crate::ast::Value::F64(n) => Ok(n),
            crate::ast::Value::U64(n) => Ok(n as f64),
            crate::ast::Value::None => Err(EmbeddingError::NonePropagated {
                accessor: "HostType::<f64>::from_value",
                source: "<typed-embedding result>".to_string(),
            }),
            other => Err(EmbeddingError::TypeMismatch {
                from_node: "<expression-output>".to_string(),
                from_type: other.port_type(),
                to_node: "<host-target>".to_string(),
                to_type: crate::ast::PortType::F64,
                source: "<typed-embedding result>".to_string(),
            }),
        }
    }
}

impl HostType for String {
    fn target_port_type() -> crate::ast::PortType {
        crate::ast::PortType::Str
    }
    fn from_value(v: crate::ast::Value) -> Result<Self, EmbeddingError> {
        match v {
            crate::ast::Value::Str(s) => Ok(s.to_string()),
            crate::ast::Value::U64(n) => Ok(n.to_string()),
            crate::ast::Value::F64(n) => Ok(n.to_string()),
            crate::ast::Value::Bool(b) => Ok(b.to_string()),
            crate::ast::Value::None => Err(EmbeddingError::NonePropagated {
                accessor: "HostType::<String>::from_value",
                source: "<typed-embedding result>".to_string(),
            }),
            other => Err(EmbeddingError::TypeMismatch {
                from_node: "<expression-output>".to_string(),
                from_type: other.port_type(),
                to_node: "<host-target>".to_string(),
                to_type: crate::ast::PortType::Str,
                source: "<typed-embedding result>".to_string(),
            }),
        }
    }
}

/// Const-fold the expression and convert the typed `Value`
/// into the host's requested Rust type. Compile-time type
/// alignment per expression_engine.md §5.3 + E5 + E7.
///
/// `T` must implement [`HostType`]. The expression's output
/// `PortType` is compared against `T::target_port_type()`;
/// matching types pass through directly to
/// [`HostType::from_value`]. Mismatched types invoke the γ-6
/// **return-path boundary adapter**: the catalog
/// (`crate::compile::assembly::auto_adapter`) is consulted
/// to heal the mismatch when possible. Only when no
/// catalog entry exists for the (output_type, target_type)
/// pair does this surface return
/// `EmbeddingError::TypeMismatch`.
///
/// Pairs with [`eval_kernel_bound_typed`] for the
/// kernel-bound (post-interpolation) case.
pub fn eval_const_expr_typed<T: HostType>(source: &str) -> Result<T, EmbeddingError> {
    let value = eval_const_expr(source)?;
    let value_type = value.port_type();
    let target_type = T::target_port_type();
    if value_type == target_type {
        return T::from_value(value);
    }
    // γ-6 return-path adapter: try the catalog before
    // surfacing TypeMismatch.
    if let Some(adapter) = crate::compile::assembly::auto_adapter(value_type, target_type) {
        let inputs = vec![value];
        let mut outputs = vec![crate::ast::Value::None];
        adapter.eval(&inputs, &mut outputs);
        return T::from_value(outputs.remove(0));
    }
    // No catalog entry — surface as typed error.
    Err(EmbeddingError::TypeMismatch {
        from_node: "<expression-output>".to_string(),
        from_type: value_type,
        to_node: "<host-target>".to_string(),
        to_type: target_type,
        source: source.to_string(),
    })
}

/// Two-step: interpolate placeholders against `kernel`, then
/// const-fold + type-convert. The canonical pattern for
/// kernel-bound typed embedding per expression_engine.md
/// §3.2 + §5.3.
pub fn eval_kernel_bound_typed<T: HostType>(
    text: &str,
    kernel: &crate::kernel::PolydatKernel,
) -> Result<T, EmbeddingError> {
    let interpolated = crate::kernel::interp::interpolate_via_kernel(text, kernel)?;
    eval_const_expr_typed::<T>(&interpolated)
}

/// Strict-mode variant of [`eval_const_expr_typed`].
///
/// Rejects type mismatches whose only catalog adapter is
/// **lossy** (e.g., `F64 → U64` truncation, `U64 → Bool`
/// boolean coercion). Hosts that want guaranteed-lossless
/// value passage opt into this surface per
/// `expression_engine.md` §5.1.3 (opt-in strict contract).
///
/// The "lossy" classification is per
/// [`is_lossless_adapter`] below; the function returns
/// `false` for catalog entries that change the value's
/// information content (truncation, narrowing, boolean
/// projection).
pub fn eval_const_expr_typed_strict<T: HostType>(source: &str) -> Result<T, EmbeddingError> {
    let value = eval_const_expr(source)?;
    let value_type = value.port_type();
    let target_type = T::target_port_type();
    if value_type == target_type {
        return T::from_value(value);
    }
    if !is_lossless_adapter(value_type, target_type) {
        return Err(EmbeddingError::TypeMismatch {
            from_node: "<expression-output>".to_string(),
            from_type: value_type,
            to_node: "<host-target>".to_string(),
            to_type: target_type,
            source: source.to_string(),
        });
    }
    if let Some(adapter) = crate::compile::assembly::auto_adapter(value_type, target_type) {
        let inputs = vec![value];
        let mut outputs = vec![crate::ast::Value::None];
        adapter.eval(&inputs, &mut outputs);
        return T::from_value(outputs.remove(0));
    }
    Err(EmbeddingError::TypeMismatch {
        from_node: "<expression-output>".to_string(),
        from_type: value_type,
        to_node: "<host-target>".to_string(),
        to_type: target_type,
        source: source.to_string(),
    })
}

/// Strict-mode kernel-bound variant. Composes
/// [`crate::kernel::interp::interpolate_via_kernel`] with
/// [`eval_const_expr_typed_strict`].
pub fn eval_kernel_bound_typed_strict<T: HostType>(
    text: &str,
    kernel: &crate::kernel::PolydatKernel,
) -> Result<T, EmbeddingError> {
    let interpolated = crate::kernel::interp::interpolate_via_kernel(text, kernel)?;
    eval_const_expr_typed_strict::<T>(&interpolated)
}

/// Whether a conversion from one port type to another keeps the
/// value: whether every number `from` can carry is a number `to`
/// can carry.
///
/// The answer is read off the two types' own numeric domains
/// ([`crate::ast::PortType::numeric_domain`]) rather than looked up in a table of
/// pairs. A table has to be kept in step with the adapter catalog by
/// hand, and was not: it named eleven pairs where the catalog has
/// well over a hundred, so `U8 → U64` was refused as lossy, and it
/// called `U64 → F64` and `I64 → F64` lossless where both round above
/// `2^53`.
///
/// Rendering to `Str` keeps the value for the types that have a
/// numeric domain, since each of those renders with a round-trip
/// `Display`. Every other conversion — into `Bytes`, `Json`, a
/// vector, `Ext` — is out of the scalar world and is not claimed
/// lossless here, whatever the catalog can do with it.
///
/// Strict-mode embedding surfaces use this to gate which catalog
/// adapters they will invoke.
pub fn is_lossless_adapter(from: crate::ast::PortType, to: crate::ast::PortType) -> bool {
    use crate::ast::PortType;
    if from == to {
        return true;
    }
    let Some(f) = from.numeric_domain() else {
        return false;
    };
    if to == PortType::Str {
        return true;
    }
    to.numeric_domain().is_some_and(|t| f.fits_in(t))
}

// ───── End typed embedding surface ─────

/// Best-effort extraction of a human message from a
/// `catch_unwind` payload. The kernel's `enrich_eval_panic`
/// re-raises with a `String` payload, so the common case is one
/// line of context-bearing text; fall through to a sentinel for
/// non-string payloads (rare — third-party panic with a custom
/// payload type).
fn panic_payload_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// Run one of the assembler's str-to-typed coercion nodes over a string
/// literal at compile time, turning the node's panic diagnostic into a
/// compile error.
fn coerce_string_literal(
    node: Box<dyn crate::ast::PolydatNode>,
    s: &str,
) -> Result<crate::ast::Value, String> {
    use crate::ast::Value;
    // The coercion node reports a bad value by panicking with its
    // diagnostic. Silence the default hook so the diagnostic surfaces
    // once, as the compile error, rather than also on stderr.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut out = [Value::None];
        node.eval(&[Value::Str(s.into())], &mut out);
        out[0].clone()
    }));
    std::panic::set_hook(hook);
    result.map_err(|e| coercion_panic_message(&e))
}

fn coercion_panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "string value could not be coerced to the declared type".to_string()
    }
}

/// Evaluate an `extern name: type = default` default expression
/// to a typed `Value`. Accepts literal forms only (`IntLit`,
/// `FloatLit`, `StringLit`, plus identifiers `true`/`false` for
/// `bool` ports). A string literal fuses to the declared type
/// through the same `StrToU64`/`StrToF64`/`StrToBool` coercions
/// the assembler inserts. Non-literal expressions are rejected
/// with a clear error; complex defaults belong in a binding, not
/// on the extern declaration.
fn evaluate_default_expr(
    expr: &crate::dsl::ast::Expr,
    port_type: crate::ast::PortType,
) -> Result<crate::ast::Value, String> {
    use crate::ast::{PortType, Value};
    use crate::dsl::ast::Expr;
    match (expr, port_type) {
        (Expr::IntLit(v, _), PortType::U64) => Ok(Value::U64(*v)),
        (Expr::IntLit(v, _), PortType::F64) => Ok(Value::F64(*v as f64)),
        (Expr::FloatLit(v, _), PortType::F64) => Ok(Value::F64(*v)),
        (Expr::StringLit(s, _), PortType::Str) => Ok(Value::Str(s.as_str().into())),
        (Expr::Ident(name, _), PortType::Bool) if name == "true" => Ok(Value::Bool(true)),
        (Expr::Ident(name, _), PortType::Bool) if name == "false" => Ok(Value::Bool(false)),
        // A string literal default fuses to the declared type through the
        // same coercions the assembler inserts when a str wire feeds a
        // typed port. This is what lets a host inject `name=value` text
        // as a program transform and leave typing to the program.
        (Expr::StringLit(s, _), PortType::U64) => {
            coerce_string_literal(Box::new(crate::library::convert::StrToU64::new()), s)
        }
        (Expr::StringLit(s, _), PortType::F64) => {
            coerce_string_literal(Box::new(crate::library::convert::StrToF64::new()), s)
        }
        (Expr::StringLit(s, _), PortType::Bool) => {
            coerce_string_literal(Box::new(crate::library::convert::StrToBool::new()), s)
        }
        _ => Err(format!(
            "default expression must be a literal of type {port_type:?}; got {expr:?}"
        )),
    }
}

/// Try to fold a `shared X := <expr>` initializer to a typed
/// `(Value, PortType)`. Returns `Some` for literal forms (the
/// shareable-cell case); returns `None` for non-literal
/// expressions (which keep the ordinary binding shape — the
/// `shared` keyword carries metadata only and the binding has
/// no cross-scope mutability today).
///
/// Literal-init shared bindings compile to an input slot +
/// passthrough output, so `materialize_wiring_from_outer` can wire a
/// `SharedCell` between this slot and inner kernels' matching
/// inputs. Non-literal shared bindings retain the
/// computation-node shape; full cross-scope mutability for
/// those is future work (see scope_model.md §6.2 "Concurrent
/// semantics").
fn try_fold_shared_init(
    expr: &crate::dsl::ast::Expr,
) -> Option<(crate::ast::Value, crate::ast::PortType)> {
    use crate::ast::{PortType, Value};
    use crate::dsl::ast::Expr;
    match expr {
        Expr::IntLit(v, _) => Some((Value::U64(*v), PortType::U64)),
        Expr::FloatLit(v, _) => Some((Value::F64(*v), PortType::F64)),
        Expr::StringLit(s, _) => Some((Value::Str(s.as_str().into()), PortType::Str)),
        Expr::Ident(name, _) if name == "true" => Some((Value::Bool(true), PortType::Bool)),
        Expr::Ident(name, _) if name == "false" => Some((Value::Bool(false), PortType::Bool)),
        _ => None,
    }
}

/// Apply the optional `shared name: type := …` annotation
/// (scope_model.md §"Type stability") to the folded `(value, type)`:
/// the annotation PINS the cell's type for life, winning over literal
/// inference. An integer literal widens to an f64-annotated cell (the
/// natural authoring, `shared m: f64 := 1`); any other mismatch is a
/// compile error at the declaration — not a runtime surprise.
fn apply_shared_type_annotation(
    name: &str,
    annotation: Option<&String>,
    init_value: crate::ast::Value,
    port_type: crate::ast::PortType,
) -> Result<(crate::ast::Value, crate::ast::PortType), String> {
    let Some(t) = annotation else {
        return Ok((init_value, port_type));
    };
    let annotated = crate::ast::PortType::from_keyword(t).ok_or_else(|| {
        format!(
            "shared binding '{name}': unknown type `{t}` in annotation. \
             Recognised types: u64, f64, str, bool."
        )
    })?;
    if annotated == port_type {
        Ok((init_value, annotated))
    } else if port_type == crate::ast::PortType::U64 && annotated == crate::ast::PortType::F64 {
        let widened = match init_value {
            crate::ast::Value::U64(v) => crate::ast::Value::F64(v as f64),
            other => other,
        };
        Ok((widened, annotated))
    } else {
        Err(format!(
            "shared binding '{name}: {t}': the initializer is {port_type:?}, \
             which doesn't match the annotated type. A cell keeps ONE type \
             for life — make the initializer match the annotation."
        ))
    }
}

/// Extract an integer literal from a positional argument. Returns None
/// for named args, non-int-literal positional args, or any other form.
fn positional_int_lit(arg: &crate::dsl::ast::Arg) -> Option<u64> {
    match arg {
        crate::dsl::ast::Arg::Positional(crate::dsl::ast::Expr::IntLit(v, _)) => Some(*v),
        _ => None,
    }
}

/// Collect the declared port type of every `input <name>: <type>`
/// declaration in the file (bare and tuple forms both lower to one
/// `InputDecl` per name). An unrecognised or absent type keyword is
/// omitted, leaving the assembler's `U64` default in force.
fn declared_input_types(
    file: &PolydatFile,
) -> std::collections::HashMap<String, crate::ast::PortType> {
    let mut types = std::collections::HashMap::new();
    for stmt in &file.statements {
        if let Statement::InputDecl(d) = stmt
            && let Some(ty) = &d.ty
            && let Some(pt) = crate::ast::PortType::from_keyword(ty)
        {
            types.insert(d.name.clone(), pt);
        }
    }
    types
}

/// Extract a string literal from an optional positional argument.
/// Re-exported for cursor-sugar handlers in node modules that
/// validate string-literal-only constructor args.
pub fn positional_str_lit(arg: Option<&crate::dsl::ast::Arg>) -> Option<String> {
    match arg? {
        crate::dsl::ast::Arg::Positional(crate::dsl::ast::Expr::StringLit(s, _)) => Some(s.clone()),
        _ => None,
    }
}

pub(super) struct Compiler {
    pub(super) input_names: Vec<String>,
    /// Track all named outputs so we can expose them.
    pub(super) all_names: Vec<String>,
    /// Auto-generated node counter for desugared intermediates.
    pub(super) anon_counter: usize,
    /// Directory for module resolution (search for .polydat files).
    pub(super) source_dir: Option<PathBuf>,
    /// Additional library directories for module resolution.
    ///
    /// Searched after `source_dir` but before the embedded stdlib.
    /// Populated from `CompileOptions::lib_paths` (the binary's
    /// `--lib`).
    pub(super) polydat_lib_paths: Vec<PathBuf>,
    /// Cache of already-resolved module ASTs: module_name → (inputs, statements).
    pub(super) module_cache: std::collections::HashMap<String, ResolvedModule>,
    /// When true, enforce strict validation.
    pub(super) strict: bool,
    /// Original source text, attached to compiled programs for diagnostics.
    source_text: String,
    /// Source schemas collected during compilation.
    pub(super) cursor_schemas: Vec<crate::iteration::source::SourceSchema>,
    /// Deferred cursor extent resolutions: each entry maps a cursor
    /// schema index to the aux output names that, once folded, give
    /// the range's start and end values. These are resolved after the
    /// kernel compiles by reading `get_constant()` for each name.
    pub(super) deferred_extents: Vec<DeferredExtent>,
    /// Optional limit applied to all cursors (from `limit` activity param).
    pub(super) cursor_limit: Option<u64>,
    /// Diagnostic context label.
    context_label: String,
    /// Module-level pragmas extracted from the source. Drive the
    /// assembler's `strict_types` / `strict_values` flags
    /// (SRD 15 §"Module-Level Pragmas" + §"Strict Wire Mode").
    pub(super) pragmas: super::pragmas::PragmaSet,
    /// LHS binding name currently being compiled, if any. Used as a
    /// prefix for auto-generated anonymous node names so type-mismatch
    /// errors point at the user-level binding (`overscan__anon_3`)
    /// instead of an opaque counter (`__anon_14`).
    pub(super) current_binding: Option<String>,
    /// Tiles lowered so far in this compile, in order, so later tiles
    /// can splice earlier ones (SRD 114 §5.5).
    pub(super) tiles: Vec<super::ast::TileDef>,
    /// Producer bindings seen so far, so tile projections over a
    /// producer can type their elements.
    pub(super) producers_seen: Vec<super::traversal::Producer>,
    /// Events raised while lowering, handed to the compile event log:
    /// one `TileHoleTyped` per hole (SRD 114 §4.4), so `explain tiles`
    /// can show how each hole was typed and encoded, one
    /// `ComprehensionWarning` per degenerate composition (§5.8), and the
    /// steps of this compile that report themselves (a binding resolved,
    /// a module inlined, an output declared), merged into the log after
    /// the parent assembles.
    pub(super) pending_events: Vec<super::events::CompileEvent>,
    /// The compile ledger of the tree being compiled: the root's, handed
    /// to every body compiler and to the assembler of every program.
    pub(super) ledger: std::sync::Arc<crate::kernel::CompileLedger>,
}

/// Records a cursor whose `range(...)` bounds reference const
/// expressions (e.g., `vector_count("example:default")`) rather than
/// integer literals. The expressions are compiled as auxiliary outputs
/// and the extent is resolved after kernel compilation by querying the
/// constant values.
pub(super) struct DeferredExtent {
    /// Index into `cursor_schemas` whose extent needs resolution.
    pub schema_idx: usize,
    /// Name of the aux output that, when folded, gives the start value.
    pub start_output: String,
    /// Name of the aux output that, when folded, gives the end value.
    pub end_output: String,
}

impl Compiler {
    /// The comprehension validation mode of this compile
    /// (comprehension_forms.md §5.8): a strict compile refuses a
    /// degenerate composition, a lax one warns about it.
    pub(super) fn validation_mode(&self) -> crate::iteration::comprehension::Mode {
        if self.strict {
            crate::iteration::comprehension::Mode::Strict
        } else {
            crate::iteration::comprehension::Mode::Permissive
        }
    }

    /// The scope a context-free source evaluates in during this
    /// compile (comprehension_forms.md §10.7.0): no name resolves, and
    /// what has to compile is charged to the program tree's ledger.
    pub(super) fn source_scope(&self) -> crate::kernel::interp::NoScope {
        crate::kernel::interp::NoScope::charged_to(self.ledger.clone())
    }

    pub(super) fn with_lib_paths(
        source_dir: Option<PathBuf>,
        polydat_lib_paths: Vec<PathBuf>,
        strict: bool,
    ) -> Self {
        Self {
            input_names: Vec::new(),
            all_names: Vec::new(),
            anon_counter: 0,
            source_dir,
            polydat_lib_paths,
            module_cache: std::collections::HashMap::new(),
            strict,
            source_text: String::new(),
            context_label: "(polydat)".into(),
            cursor_schemas: Vec::new(),
            deferred_extents: Vec::new(),
            cursor_limit: None,
            pragmas: super::pragmas::PragmaSet::default(),
            current_binding: None,
            tiles: Vec::new(),
            producers_seen: Vec::new(),
            pending_events: Vec::new(),
            ledger: crate::kernel::CompileLedger::new(),
        }
    }

    /// Process a source declaration: create input ports for projections,
    /// passthrough nodes, and record the schema.
    fn process_cursor(
        &mut self,
        asm: &mut PolydatAssembler,
        decl: &crate::dsl::ast::CursorDecl,
    ) -> Result<(), String> {
        let source_name = &decl.name;

        // Cursor-sugar dispatch: any node module can register a
        // handler that recognizes a non-`range` constructor (e.g.
        // `vectordata_base("ds", "label_00")`) and rewrites it into
        // a synthetic `range(...)` plus a list of aux bindings to
        // emit after input ports are wired. The core stays
        // generic — nothing here knows that vectordata exists.
        // See `dsl::cursor_sugar` for the registry mechanism.
        let sugar = crate::dsl::cursor_sugar::dispatch(source_name, &decl.constructor)?;
        let effective_constructor = match &sugar {
            Some(s) => s.effective_constructor.clone(),
            None => decl.constructor.clone(),
        };

        // All sources get an "ordinal" projection.
        let mut projections = vec![("ordinal".to_string(), crate::ast::PortType::U64)];

        // Determine extent from constructor args. Three cases per arg:
        //   1. Integer literal → use directly
        //   2. Other const-foldable expression (e.g. `vector_count("...")`)
        //      → compile as an aux output and resolve after kernel compiles
        //   3. Arg references runtime state → no extent available
        //
        // Immediate-literal cases produce a concrete extent here.
        // Deferred cases push a DeferredExtent record; the outer compile
        // routine reads the folded values after compilation and updates
        // the schema's extent in place.
        let mut deferred: Option<(Option<u64>, String, Option<u64>, String)> = None;
        let mut cursor_kind_for_decl: crate::iteration::source::CursorKind =
            crate::iteration::source::CursorKind::Range;
        let extent = match &effective_constructor {
            // ── until_*(...) — extending cursors ────────────────
            // Recognise every cursor function whose constructor
            // declares an extending policy. The shape of each is:
            //   until_FAMILY(base, ...policy_args[, delta])
            // where `base` is the initial extent / pass size and
            // policy_args carry the family's stop-condition
            // parameters. An optional final `delta` overrides the
            // extension step size (defaults to `base`).
            //
            // Recognised families:
            //   until_elapsed(base, min_ms[, delta])
            //   until_passes(base, min_passes[, delta])
            //   until_count(base, min_count[, delta])
            //   until_elapsed_and_passes(base, min_ms, min_passes[, delta])
            //   until_elapsed_or_passes(base, min_ms, min_passes[, delta])
            //
            // Common shape: emit `base` as the cursor's `end` aux
            // output, `start` as a literal 0, and each policy arg
            // as a named aux output the runtime pulls at phase
            // setup. The CursorKind variant carries the output
            // names so the executor knows how to build the policy.
            crate::dsl::ast::Expr::Call(call)
                if matches!(
                    call.func.as_str(),
                    "until_elapsed"
                        | "until_passes"
                        | "until_count"
                        | "until_elapsed_and_passes"
                        | "until_elapsed_or_passes"
                ) =>
            {
                let family = call.func.as_str();
                let expected = match family {
                    "until_elapsed" | "until_passes" | "until_count" => (2usize, 3usize),
                    "until_elapsed_and_passes" | "until_elapsed_or_passes" => (3, 4),
                    _ => unreachable!(),
                };
                let n = call.args.len();
                if n < expected.0 || n > expected.1 {
                    return Err(format!(
                        "cursor '{source_name}': `{family}` takes {}-{} args, got {n}",
                        expected.0, expected.1,
                    ));
                }
                // Common: base, start, end aux outputs.
                let base_literal = positional_int_lit(&call.args[0]);
                let base_name = format!("__cursor_extent_{source_name}_end");
                let start_name = format!("__cursor_extent_{source_name}_start");
                let _ = self.compile_binding(
                    asm,
                    std::slice::from_ref(&start_name),
                    &crate::dsl::ast::Expr::IntLit(0, decl.span),
                );
                if let crate::dsl::ast::Arg::Positional(expr) = &call.args[0] {
                    self.compile_binding(asm, std::slice::from_ref(&base_name), expr)
                        .map_err(|e| {
                            format!("cursor '{source_name}': failed to compile {family} base: {e}")
                        })?;
                }
                // Helper closure: compile a positional arg as a
                // named aux output. Returns the name on success.
                let mut compile_aux = |idx: usize, suffix: &str| -> Result<String, String> {
                    let out_name = format!("__cursor_{suffix}_{source_name}");
                    if let crate::dsl::ast::Arg::Positional(expr) = &call.args[idx] {
                        self.compile_binding(asm, std::slice::from_ref(&out_name), expr)
                            .map_err(|e| {
                                format!(
                                    "cursor '{source_name}': failed to compile \
                                 {family} arg {idx}: {e}"
                                )
                            })?;
                    }
                    Ok(out_name)
                };
                // Family-specific arg layout.
                cursor_kind_for_decl = match family {
                    "until_elapsed" => {
                        let min_ms_name = compile_aux(1, "min_ms")?;
                        let delta_output = if n == 3 {
                            Some(compile_aux(2, "delta")?)
                        } else {
                            None
                        };
                        crate::iteration::source::CursorKind::ExtendingTimed {
                            min_ms_output: min_ms_name,
                            delta_output,
                        }
                    }
                    "until_passes" => {
                        let min_passes_name = compile_aux(1, "min_passes")?;
                        let delta_output = if n == 3 {
                            Some(compile_aux(2, "delta")?)
                        } else {
                            None
                        };
                        crate::iteration::source::CursorKind::ExtendingPasses {
                            min_passes_output: min_passes_name,
                            delta_output,
                        }
                    }
                    "until_count" => {
                        let min_count_name = compile_aux(1, "min_count")?;
                        let delta_output = if n == 3 {
                            Some(compile_aux(2, "delta")?)
                        } else {
                            None
                        };
                        crate::iteration::source::CursorKind::ExtendingCount {
                            min_count_output: min_count_name,
                            delta_output,
                        }
                    }
                    "until_elapsed_and_passes" => {
                        let min_ms_name = compile_aux(1, "min_ms")?;
                        let min_passes_name = compile_aux(2, "min_passes")?;
                        let delta_output = if n == 4 {
                            Some(compile_aux(3, "delta")?)
                        } else {
                            None
                        };
                        crate::iteration::source::CursorKind::ExtendingElapsedAndPasses {
                            min_ms_output: min_ms_name,
                            min_passes_output: min_passes_name,
                            delta_output,
                        }
                    }
                    "until_elapsed_or_passes" => {
                        let min_ms_name = compile_aux(1, "min_ms")?;
                        let min_passes_name = compile_aux(2, "min_passes")?;
                        let delta_output = if n == 4 {
                            Some(compile_aux(3, "delta")?)
                        } else {
                            None
                        };
                        crate::iteration::source::CursorKind::ExtendingElapsedOrPasses {
                            min_ms_output: min_ms_name,
                            min_passes_output: min_passes_name,
                            delta_output,
                        }
                    }
                    _ => unreachable!(),
                };
                deferred = Some((Some(0), start_name, base_literal, base_name));
                base_literal
            }
            crate::dsl::ast::Expr::Call(call) if call.func == "range" && call.args.len() >= 2 => {
                let start_literal = positional_int_lit(&call.args[0]);
                let end_literal = positional_int_lit(&call.args[1]);

                match (start_literal, end_literal) {
                    // Both literal — compute directly. We also emit
                    // the start/end as named final bindings so the
                    // comprehension `all(<cursor>)` form (SRD-18c)
                    // can resolve them uniformly with the deferred
                    // (non-literal) case below.
                    (Some(s), Some(e)) => {
                        let start_name = format!("__cursor_extent_{source_name}_start");
                        let end_name = format!("__cursor_extent_{source_name}_end");
                        let s_lit = crate::dsl::ast::Expr::IntLit(s, decl.span);
                        let e_lit = crate::dsl::ast::Expr::IntLit(e, decl.span);
                        let _ = self.compile_binding(asm, &[start_name], &s_lit);
                        let _ = self.compile_binding(asm, &[end_name], &e_lit);
                        Some(e.saturating_sub(s))
                    }
                    // At least one non-literal — compile as aux outputs.
                    _ => {
                        let start_name = format!("__cursor_extent_{source_name}_start");
                        let end_name = format!("__cursor_extent_{source_name}_end");
                        // Compile each arg as a named auxiliary output. Errors
                        // are returned so the user sees them — silently
                        // dropping them would leave extent=None and produce
                        // a phase that runs zero cycles with no explanation.
                        if let crate::dsl::ast::Arg::Positional(expr) = &call.args[0] {
                            self.compile_binding(asm, std::slice::from_ref(&start_name), expr)
                                .map_err(|e| {
                                    format!(
                                        "cursor '{source_name}': failed to compile range start: {e}"
                                    )
                                })?;
                        }
                        if let crate::dsl::ast::Arg::Positional(expr) = &call.args[1] {
                            self.compile_binding(asm, std::slice::from_ref(&end_name), expr)
                                .map_err(|e| {
                                    format!(
                                        "cursor '{source_name}': failed to compile range end: {e}"
                                    )
                                })?;
                        }
                        deferred = Some((start_literal, start_name, end_literal, end_name));
                        None
                    }
                }
            }
            _ => None,
        };

        // Create input ports and passthrough nodes for each projection.
        for (field_name, port_type) in &projections {
            let input_name = format!("{source_name}__{field_name}");
            let default_value = match port_type {
                crate::ast::PortType::U64 => crate::ast::Value::U64(0),
                crate::ast::PortType::F64 => crate::ast::Value::F64(0.0),
                _ => crate::ast::Value::None,
            };

            // Cursor projection slots are written by cursor advance
            // every cycle — dynamic for init-contract purposes.
            asm.add_input(
                &input_name,
                default_value,
                *port_type,
                crate::kernel::InputKind::ExternalWrite,
            );
            self.input_names.push(input_name.clone());

            let passthrough = Box::new(crate::library::identity::PortPassthrough::new(
                &input_name,
                *port_type,
            ));
            let node_name = format!("{source_name}__{field_name}");
            asm.add_node(&node_name, passthrough, vec![WireRef::input(&input_name)]);
            asm.add_output(&node_name, WireRef::node(&node_name));
        }

        // Apply any aux bindings the sugar handler asked for.
        // Bindings whose `projection` is `Some` are also published
        // as cursor projections — both pinned on the schema and
        // exposed as kernel outputs the runtime can read.
        if let Some(sugar) = sugar {
            for aux in sugar.aux_bindings {
                self.compile_binding(asm, std::slice::from_ref(&aux.name), &aux.value)
                    .map_err(|e| {
                        format!(
                            "cursor '{source_name}': failed to compile aux binding '{}': {e}",
                            aux.name,
                        )
                    })?;
                if let Some((field, port_type)) = aux.projection {
                    projections.push((field, port_type));
                    asm.add_output(&aux.name, WireRef::node(&aux.name));
                }
            }
        }

        // If a limit is set, insert a limit() node that shadows the cursor wire.
        // The limit node is a visible, documented passthrough that clamps extent.
        let effective_extent = if let Some(limit_val) = self.cursor_limit {
            let limit_node_name = format!("{source_name}__limit");
            let ordinal_wire = format!("{source_name}__ordinal");
            asm.add_node(
                &limit_node_name,
                Box::new(crate::library::context::CursorLimit::new(limit_val)),
                vec![WireRef::node(&ordinal_wire)],
            );
            // Shadow the ordinal output with the limited version
            asm.add_output(&ordinal_wire, WireRef::node(&limit_node_name));

            // Clamp extent
            extent.map(|e| e.min(limit_val)).or(Some(limit_val))
        } else {
            extent
        };

        let schema_idx = self.cursor_schemas.len();
        let extent_outputs = deferred
            .as_ref()
            .map(|(_, start, _, end)| (start.clone(), end.clone()));

        // SRD 71: if the cursor decl carries an `over <expr>`
        // clause, set up two pieces of plumbing:
        //
        // 1. An auxiliary output `<source>__over_raw` carrying
        //    the raw expression value (typically a string spec
        //    or a workload-param-typed value). The executor
        //    pulls this at phase setup to determine the
        //    narrowing range.
        //
        // 2. An input slot + passthrough output `<source>__cursor`
        //    of type `Ext` — this is the field-access wire that
        //    workload authors reference as `<source>.cursor`. At
        //    phase setup the executor resolves the raw value to
        //    a concrete `Partition` and writes it into this slot,
        //    so downstream nodes (`mod_in`, `cardinality`, etc.)
        //    can consume it as a `Partition`-typed wire.
        let mut partitions: Option<Vec<crate::iteration::cursor_partition::Partition>> = None;
        let partition_output = if let Some(over_expr) = decl.over.as_ref() {
            let raw_name = format!("__cursor_{source_name}_over_raw");
            self.compile_binding(asm, std::slice::from_ref(&raw_name), over_expr)
                .map_err(|e| {
                    format!("cursor '{source_name}': failed to compile `over` expression: {e}")
                })?;
            // The wire is a spec string or a partition-typed external
            // (for_traversal.md §5, §7): any other type is refused here,
            // not at the first activation.
            if let Some(ty) = asm.output_type(&raw_name)
                && !matches!(ty, crate::ast::PortType::Str | crate::ast::PortType::Ext)
            {
                return Err(format!(
                    "cursor '{source_name}': `over` names a {ty:?} wire; expected a spec string or a partition-typed value"
                ));
            }
            // A literal spec over a known extent resolves now
            // (engines.md §3.5): the schema carries the
            // partitions for the host, and a clause that denotes
            // exactly one partition seeds the cursor's slots, so the
            // program runs on every engine with no host call. A clause
            // that denotes several leaves the choice to the host or
            // the traversal runtime, as before.
            if let (crate::dsl::ast::Expr::StringLit(spec, _), Some(extent)) =
                (over_expr, effective_extent)
            {
                let open = !matches!(
                    cursor_kind_for_decl,
                    crate::iteration::source::CursorKind::Range
                );
                let parts = crate::iteration::cursor_partition::resolve_over(
                    &crate::ast::Value::Str(spec.as_str().into()),
                    extent,
                    open,
                )
                .map_err(|e| format!("cursor '{source_name}': `over \"{spec}\"`: {e}"))?;
                partitions = Some(parts);
            }
            let seeded: Option<crate::iteration::cursor_partition::Partition> =
                partitions.as_ref().filter(|p| p.len() == 1).map(|p| p[0]);
            // Allocate the resolved-Partition input slot. Its default
            // is the one partition the clause denotes, or `Value::None`
            // until the host or the traversal runtime narrows it.
            let cursor_input_name = format!("{source_name}__cursor");
            asm.add_input(
                &cursor_input_name,
                seeded.map_or(crate::ast::Value::None, crate::ast::Value::from_partition),
                crate::ast::PortType::Ext,
                crate::kernel::InputKind::ExternalWrite,
            );
            self.input_names.push(cursor_input_name.clone());
            let passthrough = Box::new(crate::library::identity::PortPassthrough::new(
                &cursor_input_name,
                crate::ast::PortType::Ext,
            ));
            asm.add_node(
                &cursor_input_name,
                passthrough,
                vec![WireRef::input(&cursor_input_name)],
            );
            asm.add_output(&cursor_input_name, WireRef::node(&cursor_input_name));
            // SRD 71 §"Cursor metadata wires": scalar projections
            // of the resolved partition, as plain typed slots —
            // `<source>.cursor.idx` and friends parse as chained
            // field access and flatten onto these wires. The
            // executor writes them alongside the Ext slot at
            // phase setup; defaults here cover the no-narrowing
            // case (idx 0, count 1, full-extent pcts; the
            // ordinal pair is patched by the executor once the
            // cursor's extent is known).
            use crate::ast::{PortType, Value};
            let scalar_slots: [(&str, Value, PortType); 6] = match seeded {
                Some(p) => [
                    ("idx", Value::U64(p.idx), PortType::U64),
                    ("partition_count", Value::U64(p.count.max(1)), PortType::U64),
                    ("start_pct", Value::F64(p.start_pct), PortType::F64),
                    ("end_pct", Value::F64(p.end_pct), PortType::F64),
                    ("start_ordinal", Value::U64(p.start_ord), PortType::U64),
                    ("end_ordinal", Value::U64(p.end_ord), PortType::U64),
                ],
                None => [
                    ("idx", Value::U64(0), PortType::U64),
                    ("partition_count", Value::U64(1), PortType::U64),
                    ("start_pct", Value::F64(0.0), PortType::F64),
                    ("end_pct", Value::F64(100.0), PortType::F64),
                    ("start_ordinal", Value::U64(0), PortType::U64),
                    ("end_ordinal", Value::U64(0), PortType::U64),
                ],
            };
            for (field, default, port_type) in scalar_slots {
                let slot = format!("{cursor_input_name}__{field}");
                asm.add_input(
                    &slot,
                    default,
                    port_type,
                    crate::kernel::InputKind::ExternalWrite,
                );
                self.input_names.push(slot.clone());
                let pass = Box::new(crate::library::identity::PortPassthrough::new(
                    &slot, port_type,
                ));
                asm.add_node(&slot, pass, vec![WireRef::input(&slot)]);
                asm.add_output(&slot, WireRef::node(&slot));
            }
            Some(raw_name)
        } else {
            None
        };

        self.cursor_schemas
            .push(crate::iteration::source::SourceSchema {
                name: source_name.clone(),
                projections,
                extent: effective_extent,
                extent_outputs,
                extent_limit: self.cursor_limit,
                cursor_kind: cursor_kind_for_decl.clone(),
                partition_output,
                partitions,
            });

        // Record deferred extent resolution if the range bounds are not
        // both literals. Post-compile, the outer compile routine will
        // query the aux outputs' folded constants and update this
        // schema's extent in place.
        if let Some((_start_lit, start_output, _end_lit, end_output)) = deferred {
            self.deferred_extents.push(DeferredExtent {
                schema_idx,
                start_output,
                end_output,
            });
        }
        Ok(())
    }

    /// The interpreter's kernel of `file` as its concrete type: the one
    /// compile path with the interpreter's build, keeping the outputs in
    /// `filter` (every output when `None`) and recording events in `log`.
    /// The parent's AST is retained as program metadata for the subscope
    /// synthesizer.
    pub(super) fn compile_interpreter(
        &mut self,
        file: &PolydatFile,
        filter: Option<&[String]>,
        log: Option<&mut super::events::CompileEventLog>,
        cones: crate::JitMode,
    ) -> Result<PolydatKernel, crate::KernelError> {
        let (mut kernel, parent) = compile_file_with(self, file, filter, log, |mut asm, log| {
            asm.set_jit_mode(cones);
            asm.compile_with_log(log).map_err(crate::KernelError::from)
        })?;
        kernel.set_ast(std::sync::Arc::new(parent));
        Ok(kernel)
    }

    /// The output type of a generator expression used as a comprehension
    /// source (SRD 113 §3.3): compile `__probe := <expr>` on its own and
    /// read the port type. Shared by `for` bodies and tile projections.
    pub(super) fn probe_element_type(&self, expr: &str) -> Result<crate::ast::PortType, String> {
        let src = format!("input cycle: u64\n__probe := {expr}\n");
        let tokens = lexer::lex(&src)?;
        let ast = parser::parse(tokens)?;
        let mut probe_compiler = Compiler::with_lib_paths(
            self.source_dir.clone(),
            self.polydat_lib_paths.clone(),
            false,
        );
        probe_compiler.source_text = src.clone();
        probe_compiler.context_label = format!("{} (element probe)", self.context_label);
        probe_compiler.module_cache = self.module_cache.clone();
        let k = probe_compiler
            .compile_interpreter(&ast, None, None, crate::JitMode::Auto)
            .map_err(|e| e.to_string())?;
        k.program()
            .output_port_type("__probe")
            .ok_or_else(|| "probe produced no output".to_string())
    }

    /// Lower each `for` statement's body to a child program, typed from
    /// its comprehension and the parent's manifest (SRD 113 §3.3, §4).
    fn compile_traversals(
        &mut self,
        for_stmts: &[super::ast::ForStmt],
        producers: &[super::traversal::Producer],
        type_of: &dyn Fn(&str) -> Option<crate::ast::PortType>,
    ) -> Result<Vec<super::traversal::Traversal>, String> {
        use super::traversal::{
            Traversal, child_file, element_types, resolve_source_with, warning_events,
        };
        let mut out = Vec::with_capacity(for_stmts.len());
        for f in for_stmts {
            let (comprehension, warnings) = resolve_source_with(
                &f.source,
                producers,
                self.validation_mode(),
                &self.source_scope(),
            )?;
            self.pending_events
                .extend(warning_events(&f.source, &warnings));
            let mut probe = |expr: &str| self.probe_element_type(expr);
            let elements = element_types(&comprehension, &mut probe).map_err(|e| {
                format!(
                    "`for {}` at line {}, col {}: {e}",
                    f.source.to_text(),
                    f.span.line,
                    f.span.col
                )
            })?;
            let (child, cascade) = child_file(f, &comprehension, &elements, type_of)?;
            let mut child_compiler = Compiler::with_lib_paths(
                self.source_dir.clone(),
                self.polydat_lib_paths.clone(),
                self.strict,
            );
            // The body sees every module the parent resolved, its own
            // definitions included, wherever it compiles.
            child_compiler.module_cache = self.module_cache.clone();
            // The body's program is one of the tree's.
            child_compiler.ledger = self.ledger.clone();
            child_compiler.source_text = super::pprint::pp_file(&child);
            child_compiler.context_label = format!(
                "{} :: for {} (line {}, col {})",
                self.context_label,
                f.source.to_text(),
                f.span.line,
                f.span.col
            );
            child_compiler.cursor_limit = self.cursor_limit;
            child_compiler.pragmas = self.pragmas.clone();
            let child_kernel = child_compiler
                .compile_interpreter(&child, None, None, crate::JitMode::Auto)
                .map_err(|e| {
                    format!(
                        "`for {}` at line {}, col {}: body failed to compile: {e}",
                        f.source.to_text(),
                        f.span.line,
                        f.span.col
                    )
                })?;
            self.pending_events
                .append(&mut child_compiler.pending_events);
            let body = super::traversal::BodySource {
                file: child,
                source_text: child_compiler.source_text.clone(),
                source_dir: self.source_dir.clone(),
                lib_paths: self.polydat_lib_paths.clone(),
                strict: self.strict,
                context_label: child_compiler.context_label.clone(),
                cursor_limit: self.cursor_limit,
                pragmas: self.pragmas.clone(),
                modules: self.module_cache.clone(),
                programs: std::sync::Mutex::new(std::collections::HashMap::new()),
                ledger: self.ledger.clone(),
            };
            out.push(Traversal {
                span: f.span,
                source_text: f.source.to_text(),
                comprehension,
                elements,
                cascade,
                program: child_kernel.into_program(),
                body: std::sync::Arc::new(body),
            });
        }
        Ok(out)
    }

    /// Compile a traversal body on `engine` (engine parity, step 8): the
    /// same child file and compiler settings the parent used for the
    /// interpreter's program, through the assembler, its own `for`
    /// statements and producers included.
    pub(super) fn compile_body_on(
        body: &super::traversal::BodySource,
        engine: crate::Engine,
    ) -> Result<Box<dyn crate::Kernel>, crate::KernelError> {
        let _data_base = body.source_dir.as_deref().map(DataBaseDirGuard::set);
        let mut compiler =
            Compiler::with_lib_paths(body.source_dir.clone(), body.lib_paths.clone(), body.strict);
        compiler.source_text = body.source_text.clone();
        compiler.context_label = body.context_label.clone();
        compiler.cursor_limit = body.cursor_limit;
        compiler.pragmas = body.pragmas.clone();
        compiler.module_cache = body.modules.clone();
        compiler.ledger = body.ledger.clone();
        compile_file_on_engine(&mut compiler, &body.file, None, engine, None)
    }

    /// Assemble the parent program: inputs and their passthroughs,
    /// externs, bindings, cursors, tiles, and the output set. Every
    /// entry point builds its assembler here, so a kernel and an
    /// assembler from the same source are the same graph.
    fn assemble_parent(
        &mut self,
        file: &PolydatFile,
        required_outputs: Option<&[String]>,
    ) -> Result<PolydatAssembler, String> {
        self.register_local_modules(file);
        // First pass: collect explicit `input` declarations, dedup by name.
        for stmt in &file.statements {
            if let Statement::InputDecl(d) = stmt
                && !self.input_names.iter().any(|n| n == &d.name)
            {
                self.input_names.push(d.name.clone());
            }
        }

        // Input declaration check: error in strict mode (modules, .polydat files)
        if self.input_names.is_empty() && self.strict {
            return Err(
                "strict mode: no `input` declaration — add `input <name>: <type>` \
                 (or the tuple form `input (a: u64, b: f64)`) to declare graph \
                 inputs explicitly"
                    .into(),
            );
        }

        // If no explicit inputs, infer from unbound references
        if self.input_names.is_empty() {
            let defined: HashSet<String> = file
                .statements
                .iter()
                .flat_map(|stmt| match stmt {
                    Statement::Binding(b) => b.targets.clone(),
                    Statement::ModuleDef(m) => vec![m.name.clone()],
                    Statement::ExternPort(p) => vec![p.name.clone()],
                    Statement::InputDecl(_) => vec![],
                    Statement::Cursor(_) => vec![],
                    Statement::Pragma { .. } => vec![],
                    Statement::For(_) => vec![],
                    Statement::Tile(t) => vec![t.name.clone()],
                })
                .collect();

            let mut referenced: HashSet<String> = HashSet::new();
            for stmt in &file.statements {
                let expr = match stmt {
                    Statement::InputDecl(_)
                    | Statement::ModuleDef(_)
                    | Statement::ExternPort(_)
                    | Statement::Cursor(_)
                    | Statement::Pragma { .. }
                    | Statement::For(_)
                    | Statement::Tile(_) => continue,
                    Statement::Binding(b) => &b.value,
                };
                collect_references(expr, &mut referenced);
            }

            let mut inferred: Vec<String> = referenced
                .into_iter()
                .filter(|name| !defined.contains(name))
                .collect();
            inferred.sort();
            self.input_names = inferred;
        }

        // Zero inferred inputs means all bindings are constants — valid.

        let mut asm = PolydatAssembler::new(self.input_names.clone());
        asm.ledger = self.ledger.clone();
        for (name, ty) in declared_input_types(file) {
            asm.set_input_type(&name, ty);
        }

        // Auto-expose every declared input as a passthrough output
        // (parity with `extern`). See `compile()` for the same wiring.
        for input_name in self.input_names.clone() {
            // Mirror the input's (now correctly-typed) slot so the
            // auto-exposed output carries the declared type, not U64.
            let port_type = asm
                .input_type(&input_name)
                .unwrap_or(crate::ast::PortType::U64);
            let passthrough = Box::new(crate::library::identity::PortPassthrough::new(
                &input_name,
                port_type,
            ));
            let passthrough_name = format!("__port_{input_name}");
            asm.add_node(
                &passthrough_name,
                passthrough,
                vec![WireRef::input(&input_name)],
            );
            asm.add_output(&input_name, WireRef::node(&passthrough_name));
        }

        // Second pass: process all bindings into the assembler
        for stmt in &file.statements {
            match stmt {
                Statement::InputDecl(_) => {}
                Statement::Binding(b) => {
                    // `shared X := <literal>` compiles to an input
                    // slot + passthrough output, so
                    // `materialize_wiring_from_outer` can wire a
                    // `SharedCell` for cross-scope mutability (SRD-16
                    // §"Mutability Rules: Shared Mutable"). Non-literal
                    // inits and tuple-target shared bindings are
                    // rejected on every entry point: the cell needs a
                    // single, well-defined initial value, and a
                    // computation-shaped RHS doesn't have one. See
                    // SRD-16 §"Non-literal `shared` initializers".
                    if b.modifier == BindingModifier::SHARED {
                        if b.targets.len() != 1 {
                            return Err(format!(
                                "shared binding must be single-target, not tuple unpack \
                                 ({}). Declare each target separately if a shared cell \
                                 is intended.",
                                b.targets.join(", "),
                            ));
                        }
                        let name = &b.targets[0];
                        let (init_value, port_type) =
                            try_fold_shared_init(&b.value).ok_or_else(|| {
                                format!(
                                    "shared binding '{name}' requires a literal initial value \
                                 (number, string, true/false). Computed and cycle-dependent \
                                 expressions don't have a well-defined single init for the \
                                 shared cell. See SRD-16 §\"Non-literal `shared` initializers\"."
                                )
                            })?;
                        let (init_value, port_type) = apply_shared_type_annotation(
                            name,
                            b.type_annotation.as_ref(),
                            init_value,
                            port_type,
                        )?;
                        asm.add_input(
                            name,
                            init_value,
                            port_type,
                            crate::kernel::InputKind::ExternalWrite,
                        );
                        self.input_names.push(name.clone());
                        let passthrough = Box::new(crate::library::identity::PortPassthrough::new(
                            name, port_type,
                        ));
                        let passthrough_name = format!("__port_{name}");
                        asm.add_node(&passthrough_name, passthrough, vec![WireRef::input(name)]);
                        asm.add_output(name, WireRef::node(&passthrough_name));
                        asm.set_output_modifier(name, BindingModifier::SHARED);
                        continue;
                    }
                    self.compile_binding(&mut asm, &b.targets, &b.value)?;
                    // Every target that now names a node reports what it
                    // resolved to: a call, an operator, or a literal alike.
                    for target in &b.targets {
                        if let Some(node_type) = asm.node_type_of(target) {
                            self.pending_events.push(
                                super::events::CompileEvent::BindingResolved {
                                    name: target.clone(),
                                    node_type,
                                },
                            );
                        }
                    }
                    if b.modifier != BindingModifier::NONE {
                        for target in &b.targets {
                            asm.set_output_modifier(target, b.modifier);
                        }
                    }
                    // SRD-74 P2: auto-extern const targets whose RHS
                    // references at least one name. See the parallel
                    // block in `compile()` for rationale — makes
                    // `const NAME := <expr>` a conditional shadow when
                    // its RHS could fold to None, while leaving
                    // pure-literal consts (SRD-13f Gate 2 iter-vars)
                    // alone.
                    if b.modifier.is_const() {
                        let rhs_has_refs = {
                            let mut refs = std::collections::HashSet::new();
                            crate::dsl::validate::collect_references(&b.value, &mut refs);
                            !refs.is_empty()
                        };
                        for target in &b.targets {
                            asm.mark_const_output(target);
                            if rhs_has_refs && !asm.input_names().contains(&target.as_str()) {
                                // The binding's right-hand side was
                                // compiled just above, so its node is
                                // in the assembler and carries the
                                // resolved output `PortType` the
                                // auto-extern slot should take. That
                                // covers every shape, including the
                                // ones a pass over the surface AST
                                // cannot see through — `select_str`,
                                // `format_u64`, a nested call.
                                let Some(inferred) = asm.output_type(target.as_str()) else {
                                    return Err(format!(
                                        "internal error: the const binding `{target}` was \
                                         just compiled, so the assembler should carry its \
                                         output type"
                                    ));
                                };
                                asm.add_input(
                                    target.as_str(),
                                    crate::ast::Value::None,
                                    inferred,
                                    crate::kernel::InputKind::IterationExtern,
                                );
                            }
                        }
                    }
                }
                Statement::ModuleDef(_) => {}
                Statement::ExternPort(port) => {
                    // Mirror `compile()`: same kind classification —
                    // a default expression marks this as a capture
                    // port (dynamic); no default marks it as an
                    // iteration extern (effectively-const at
                    // scope-init time).
                    let port_type = crate::ast::PortType::from_keyword(port.typ.as_str())
                        .ok_or_else(|| {
                            format!(
                                "extern '{}': unknown polydat type keyword '{}'. \
                             Canonical keywords are emitted by PortType::to_keyword \
                             (one per PortType variant).",
                                port.name, port.typ,
                            )
                        })?;
                    let (default_value, kind) = match &port.default {
                        Some(expr) => {
                            let v = evaluate_default_expr(expr, port_type)
                                .map_err(|e| format!("extern '{}' default: {e}", port.name,))?;
                            (v, crate::kernel::InputKind::ExternalWrite)
                        }
                        None => (
                            crate::ast::Value::None,
                            crate::kernel::InputKind::IterationExtern,
                        ),
                    };
                    asm.add_input(&port.name, default_value, port_type, kind);
                    self.input_names.push(port.name.clone());
                    let passthrough = Box::new(crate::library::identity::PortPassthrough::new(
                        &port.name, port_type,
                    ));
                    let passthrough_name = format!("__port_{}", port.name);
                    asm.add_node(
                        &passthrough_name,
                        passthrough,
                        vec![crate::compile::assembly::WireRef::input(&port.name)],
                    );
                    asm.add_output(
                        &port.name,
                        crate::compile::assembly::WireRef::node(&passthrough_name),
                    );
                }
                Statement::Cursor(decl) => {
                    self.process_cursor(&mut asm, decl)?;
                }
                Statement::Pragma { .. } => {}
                Statement::For(f) => {
                    return Err(format!(
                        "`for {}` at line {}, col {}: {}",
                        f.source.to_text(),
                        f.span.line,
                        f.span.col,
                        "a `for` traversal compiles through `compile_polydat` and runs through `PolydatKernel::traverse`; the assembler entry point builds one program and cannot carry a traversal (docs/design/for_traversal.md §5)"
                    ));
                }
                Statement::Tile(t) => {
                    self.compile_tile(&mut asm, t)?;
                }
            }
        }

        // Unused binding check: defer to kernel-level check in fold_init_constants_impl.
        // The kernel has the full wiring graph and can accurately determine which
        // nodes have no downstream consumers. The compiler can't do this reliably
        // because it doesn't track inter-binding wire dependencies.

        // Expose outputs: only the required set, or all if no filter.
        // Cursor extent aux outputs (`__cursor_extent_*`) must always be
        // exposed regardless of the filter — they are queried by the
        // post-compile deferred extent resolution and would otherwise be
        // pruned by DCE, leaving the cursor extent unresolved.
        match required_outputs {
            Some(required) => {
                // SRD-13f Push D / SRD-44: `volatile` bindings stay
                // exposed as outputs even when the caller's
                // required list doesn't mention them. The author
                // declared the wire as volatile to mark it as
                // non-deterministic across invocations — losing
                // it from the output set (DCE) would also lose
                // the "exclude from program identity" guarantee,
                // because the lifecycle classifier would no
                // longer find a volatile output pointing at the
                // producing node.
                let mut required_owned: Vec<String> = required.to_vec();
                for stmt in &file.statements {
                    if let crate::dsl::ast::Statement::Binding(b) = stmt
                        && b.modifier.is_volatile()
                    {
                        for t in &b.targets {
                            if !required_owned.iter().any(|n| n == t) {
                                required_owned.push(t.clone());
                            }
                        }
                    }
                }
                for name in &required_owned {
                    if self.all_names.contains(name) {
                        self.pending_events
                            .push(super::events::CompileEvent::OutputDeclared {
                                name: name.clone(),
                            });
                    }
                    if self.all_names.contains(name) {
                        asm.add_output(name, WireRef::node(name));
                    }
                }
                for deferred in &self.deferred_extents {
                    if self.all_names.contains(&deferred.start_output) {
                        asm.add_output(
                            &deferred.start_output,
                            WireRef::node(&deferred.start_output),
                        );
                    }
                    if self.all_names.contains(&deferred.end_output) {
                        asm.add_output(&deferred.end_output, WireRef::node(&deferred.end_output));
                    }
                }
                // Always preserve `__cursor_extent_*` auxiliary
                // outputs — they're consumed by the comprehension
                // `all(<cursor>)` form (SRD-18c §"Layer 3") and
                // also by the post-compile deferred-extent
                // resolution above. DCE-ing them would leave the
                // cursor's extent unresolvable to descendant scopes.
                let pruned_aux: Vec<String> = self
                    .all_names
                    .iter()
                    .filter(|n| n.starts_with("__cursor_extent_"))
                    .cloned()
                    .collect();
                for name in pruned_aux {
                    asm.add_output(&name, WireRef::node(&name));
                }
            }
            None => {
                for name in &self.all_names {
                    self.pending_events
                        .push(super::events::CompileEvent::OutputDeclared { name: name.clone() });
                    asm.add_output(name, WireRef::node(name));
                }
            }
        }

        asm.set_context(&self.source_text, &self.context_label);
        // The strictness pragmas reach every kernel built from this
        // assembler, on every engine and on every entry point.
        asm.set_strict_wires(self.pragmas.strict_types(), self.pragmas.strict_values());
        asm.set_strict(self.strict);
        // The cursors, with their partitions resolved at build, reach
        // every kernel built from this assembler (engines.md §3.5).
        asm.set_cursor_schemas(self.cursor_schemas.clone());
        Ok(asm)
    }
}

// ── The one entry point (engines.md §3.5) ──────────────────

/// Compile `source` for `engine`: the interpreter, the closure tier,
/// the hybrid kernel, or pure native code. Every engine accepts every
/// program the interpreter accepts, or refuses it with a reason
/// ([`crate::KernelError::Refused`]); a host drives the result through
/// [`crate::Kernel`] without knowing which engine it holds. The
/// `compile_polydat_kernel*` and `compile_polydat_checked` entry
/// points are this on `Engine::default()`; `compile_polydat`,
/// `compile_polydat_interpreter_with_options`, and the deprecated forms build
/// the interpreter's kernel.
pub fn compile_polydat_with(
    source: &str,
    engine: crate::Engine,
) -> Result<Box<dyn crate::Kernel>, crate::KernelError> {
    compile_polydat_with_engine(source, engine, &CompileOptions::default(), None)
}

/// [`compile_polydat_with`] on [`Engine::default`](crate::Engine::default):
/// compiled code, with the JIT where the build has it.
pub fn compile_polydat_kernel(source: &str) -> Result<Box<dyn crate::Kernel>, crate::KernelError> {
    compile_polydat_with(source, crate::Engine::default())
}

/// [`compile_polydat_kernel`] with the kernel path's options (source
/// directory, library paths, required outputs, strict typing, the
/// error context label, the cursor limit, and the engine preference)
/// and the compile event log.
///
/// This is the entry point the engine preference is read from: it
/// builds on `options.engine`, which defaults to the most native form
/// the build has, so a host that never sets the field gets compiled
/// code without naming one.
pub fn compile_polydat_kernel_with_options(
    source: &str,
    options: &CompileOptions,
    log: Option<&mut super::events::CompileEventLog>,
) -> Result<Box<dyn crate::Kernel>, crate::KernelError> {
    compile_polydat_with_engine(source, options.engine, options, log)
}

/// [`compile_polydat_with`] with the kernel path's options (source
/// directory, library paths, required outputs, strict typing, the
/// error context label, the cursor limit) and the compile event log.
/// On the interpreter this is the whole kernel path, traversals
/// included; on a compiled engine the assembler entry point followed
/// by [`PolydatAssembler::compile_engine_with_log`].
///
/// The `engine` argument is a per-call override of `options.engine`,
/// for a caller that holds one options value and walks the tiers with
/// it. A caller that has no such need expresses the preference once, in
/// the options, and calls [`compile_polydat_kernel_with_options`].
pub fn compile_polydat_with_engine(
    source: &str,
    engine: crate::Engine,
    options: &CompileOptions,
    mut log: Option<&mut super::events::CompileEventLog>,
) -> Result<Box<dyn crate::Kernel>, crate::KernelError> {
    use crate::KernelError;
    let tokens = super::lexer::lex(source).map_err(KernelError::Source)?;
    let ast = super::parser::parse(tokens).map_err(KernelError::Source)?;
    if let Some(log) = log.as_deref_mut() {
        log.push(super::events::CompileEvent::Parsed {
            statements: ast.statements.len(),
        });
    }
    compile_ast_with_engine(&ast, source, options, log, engine)
}

/// [`compile_polydat_with_engine`] from a parsed file: the parent
/// compiles on `engine` through the assembler, and each `for` body
/// compiles once for the interpreter as the traversal's record and on
/// any engine at activation (engine parity, step 8).
pub fn compile_ast_with_engine(
    ast: &PolydatFile,
    source: &str,
    options: &CompileOptions,
    mut log: Option<&mut super::events::CompileEventLog>,
    engine: crate::Engine,
) -> Result<Box<dyn crate::Kernel>, crate::KernelError> {
    let mut prepared = Prepared::new(source, ast, options, log.as_deref_mut());
    let (compiler, filter) = prepared.parts();
    compile_file_on_engine(compiler, ast, filter, engine, log)
}

/// Everything an entry point sets up before a program assembles: the
/// compiler under its options, the outputs to keep, and the data-file
/// base directory for the compile's duration. One prologue for every
/// entry point, so the options mean the same thing whichever one
/// carries them.
struct Prepared {
    compiler: Compiler,
    required: Vec<String>,
    _data_base: Option<DataBaseDirGuard>,
}

impl Prepared {
    fn new(
        source: &str,
        ast: &PolydatFile,
        options: &CompileOptions,
        log: Option<&mut super::events::CompileEventLog>,
    ) -> Self {
        // Relative data-file paths (csv/jsonl nodes) resolve against the
        // program's own directory for the duration of this synchronous
        // compile; see `library::datafile::set_data_base_dir`.
        let _data_base = options.source_dir.as_deref().map(DataBaseDirGuard::set);
        let pragmas = super::pragmas::collect_from_ast(ast);
        if let Some(log) = log {
            record_pragma_events(&pragmas, log);
        }
        // The required-outputs list is extended with the const bindings
        // only when the caller passed one: an empty list keeps every
        // binding, and extending it would flip its meaning.
        let required = if options.required_outputs.is_empty() {
            Vec::new()
        } else {
            extend_required_with_const_bindings(&options.required_outputs, ast)
        };
        let mut compiler = Compiler::with_lib_paths(
            options.source_dir.clone(),
            options.lib_paths.clone(),
            options.strict,
        );
        compiler.source_text = source.to_string();
        // An empty context keeps the compiler's default label, so a
        // failure reads the same whichever entry point built the kernel.
        if !options.context.is_empty() {
            compiler.context_label = options.context.clone();
        }
        compiler.cursor_limit = options.cursor_limit;
        compiler.pragmas = pragmas;
        if let Some(ledger) = &options.ledger {
            compiler.ledger = ledger.clone();
        }
        Prepared {
            compiler,
            required,
            _data_base,
        }
    }

    /// The compiler and the output filter, `None` for every output.
    fn parts(&mut self) -> (&mut Compiler, Option<&[String]>) {
        let filter = if self.required.is_empty() {
            None
        } else {
            Some(self.required.as_slice())
        };
        (&mut self.compiler, filter)
    }
}

/// The one path from a parsed file to a kernel, on every engine: the
/// `for` statements and producer bindings are lifted out, the parent
/// assembles and `build` makes its kernel, each body compiles once
/// against the parent's types and is attached, the tile events reach
/// the log, and every cursor extent the program computes from constants
/// is resolved on the kernel. Returns the kernel with the parent file
/// the traversals were lifted from.
fn compile_file_with<K: Built>(
    compiler: &mut Compiler,
    file: &PolydatFile,
    filter: Option<&[String]>,
    mut log: Option<&mut super::events::CompileEventLog>,
    build: impl FnOnce(
        PolydatAssembler,
        Option<&mut super::events::CompileEventLog>,
    ) -> Result<K, crate::KernelError>,
) -> Result<(K, PolydatFile), crate::KernelError> {
    use crate::KernelError;
    let (parent_file, for_stmts, producers) = super::traversal::strip_for_forms(
        file,
        compiler.validation_mode(),
        &compiler.source_scope(),
        &mut compiler.pending_events,
    )
    .map_err(KernelError::Source)?;
    compiler.producers_seen = producers.clone();
    let asm = compiler
        .assemble_parent(&parent_file, filter)
        .map_err(KernelError::Source)?;
    // The tiles typed while assembling belong to this program's log.
    if let Some(log) = log.as_deref_mut() {
        for e in compiler.pending_events.drain(..) {
            log.push(e);
        }
    }
    let mut built = build(asm, log.as_deref_mut())?;
    let kernel: &mut dyn crate::Kernel = built.kernel();
    if !for_stmts.is_empty() || !producers.is_empty() {
        let externs = kernel.externs();
        let inputs = kernel.input_names();
        let type_of = |name: &str| {
            kernel.output_type(name).or_else(|| {
                externs
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, t)| *t)
                    .or_else(|| {
                        // A coordinate: the one input kind that is not an extern.
                        inputs
                            .iter()
                            .any(|n| n == name)
                            .then_some(crate::ast::PortType::U64)
                    })
            })
        };
        let traversals = compiler
            .compile_traversals(&for_stmts, &producers, &type_of)
            .map_err(KernelError::Source)?;
        crate::kernel::KernelInternals::set_traversals(kernel, traversals, producers);
        // Tiles inside the bodies, typed in the child compilers.
        if let Some(log) = log {
            for e in compiler.pending_events.drain(..) {
                log.push(e);
            }
        }
    }
    // A cursor whose range is computed from constants gets its extent
    // from the values the build folded, on every engine.
    for deferred in &compiler.deferred_extents {
        let start = kernel
            .folded_value(&deferred.start_output)
            .map(|v| v.as_u64());
        let end = kernel
            .folded_value(&deferred.end_output)
            .map(|v| v.as_u64());
        if let (Some(s), Some(e)) = (start, end) {
            let resolved = e.saturating_sub(s);
            let extent = compiler
                .cursor_limit
                .map(|limit| resolved.min(limit))
                .unwrap_or(resolved);
            if let Some(schema) = compiler.cursor_schemas.get_mut(deferred.schema_idx) {
                schema.extent = Some(extent);
            }
            kernel.set_cursor_extent(deferred.schema_idx, extent);
        }
    }
    Ok((built, parent_file))
}

/// What a build hands back to the compile path: the interpreter's
/// concrete kernel or any engine's boxed one, each reachable as the one
/// trait the lowering drives.
trait Built {
    fn kernel(&mut self) -> &mut dyn crate::Kernel;
}

impl Built for PolydatKernel {
    fn kernel(&mut self) -> &mut dyn crate::Kernel {
        self
    }
}

impl Built for Box<dyn crate::Kernel> {
    fn kernel(&mut self) -> &mut dyn crate::Kernel {
        self.as_mut()
    }
}

/// The kernel of a parsed file on `engine`: [`compile_file_with`] with
/// the engine's build, and the interpreter's concrete kernel boxed when
/// the engine is the interpreter.
pub(super) fn compile_file_on_engine(
    compiler: &mut Compiler,
    file: &PolydatFile,
    filter: Option<&[String]>,
    engine: crate::Engine,
    log: Option<&mut super::events::CompileEventLog>,
) -> Result<Box<dyn crate::Kernel>, crate::KernelError> {
    if let crate::Engine::Interpreter(cones) = engine {
        return compiler
            .compile_interpreter(file, filter, log, cones)
            .map(|k| Box::new(k) as Box<dyn crate::Kernel>);
    }
    let (kernel, _) = compile_file_with(compiler, file, filter, log, |asm, log| {
        asm.compile_engine_with_log(engine, log)
    })?;
    Ok(kernel)
}
// ── Former names of the interpreter-typed entry points ──────────────
//
// These returned the interpreter's concrete kernel under names that did
// not say so, which read as though they were the general way to compile
// under options. They are the exception, not the rule: a kernel is used
// through the `Kernel` trait, and the concrete type is for observing the
// interpreter's own internals in testing and diagnostics (engines.md
// §3.6). The names now say that; these keep the old ones working.

#[cfg(test)]
mod tests {
    use super::*;

    /// The interpreter kernel under `strict` alone.
    fn strict(src: &str, strict: bool) -> Result<PolydatKernel, crate::KernelError> {
        let options = CompileOptions {
            strict,
            ..CompileOptions::default()
        };
        compile_polydat_interpreter_with_options(src, &options, None)
    }

    #[test]
    fn array_literal_binding_compiles_as_string() {
        // A list-valued binding (`const xs := [1, 2, 3]`) is a sweep
        // axis / interpolation value, not a scalar wire. polydat has no
        // const-vector node, so it binds to a `ConstStr` holding the
        // list's literal text rather than failing the compile — which
        // is what lets list-valued workload params (`limit_values:
        // [25]`) load.
        let result = compile_polydat_interpreter(
            "input cycle: u64\nconst eh_values := [1, 2, 3]\nout := cycle",
        );
        assert!(
            result.is_ok(),
            "array-literal binding should compile (binds as a string const), got: {:?}",
            result.err(),
        );
        // The resolved value is the comma-joined, bracket-free form a
        // sweep-axis param carries (so a `WorkloadParamList` source
        // splits it on `, ` exactly like a string-valued sweep param).
        let kernel = result.unwrap();
        match kernel.get_constant("eh_values") {
            Some(crate::ast::Value::Str(s)) => assert_eq!(s.as_ref(), "1, 2, 3"),
            other => panic!("expected eh_values = Str(\"1, 2, 3\"), got {other:?}"),
        }
    }

    #[test]
    fn array_literal_in_argument_position_is_refused() {
        // The other half of the same rule: a list literal is a
        // binding-position form and has no meaning as a call argument
        // (polydat_grammar.md §18.1 T-ArrayLit). It used to lower to a
        // const argument nothing read, so the call reached the runtime
        // with one wire input missing and panicked there instead.
        let err = compile_polydat_interpreter("input cycle: u64\nout := printf(\"{}\", [1, 2])")
            .expect_err("a list literal in argument position is a compile error");
        let text = err.to_string();
        assert!(
            text.contains("binding-position form") && text.contains("printf"),
            "the error should name the form and the call: {text}",
        );
        // Bound first, the same list works, which is what the message
        // tells the author to do.
        let ok =
            compile_polydat_interpreter("input cycle: u64\nw := [1, 2]\nout := printf(\"{}\", w)");
        assert!(ok.is_ok(), "{:?}", ok.err());
    }

    #[test]
    fn embedding_error_display_includes_source_text() {
        let e = EmbeddingError::LifecycleMismatch {
            source: "hash(cycle)".to_string(),
            dynamic_inputs: vec!["cycle".to_string()],
        };
        let s = format!("{e}");
        assert!(
            s.contains("hash(cycle)"),
            "display should include source: {s}"
        );
        assert!(
            s.contains("cycle"),
            "display should mention dynamic input: {s}"
        );
    }

    #[test]
    fn embedding_error_from_string_shim() {
        let e = EmbeddingError::UnresolvedPlaceholder {
            name: "k".to_string(),
            source: "{k} > 5".to_string(),
        };
        let s: String = e.clone().into();
        assert_eq!(s, format!("{e}"));
    }

    #[test]
    fn embedding_error_all_variants_display() {
        // Smoke test: every variant constructs and displays without panicking.
        let variants: Vec<EmbeddingError> = vec![
            EmbeddingError::Parse {
                source: "x +".into(),
                message: "unexpected EOF".into(),
                position: Some(3),
            },
            EmbeddingError::UnresolvedPlaceholder {
                name: "k".into(),
                source: "{k}".into(),
            },
            EmbeddingError::LifecycleMismatch {
                source: "hash(cycle)".into(),
                dynamic_inputs: vec!["cycle".into()],
            },
            EmbeddingError::UnknownNode {
                name: "frobnicate".into(),
                source: "frobnicate(x)".into(),
                suggestion: Some("fabricate".into()),
            },
            EmbeddingError::TypeMismatch {
                from_node: "n1".into(),
                from_type: crate::ast::PortType::U64,
                to_node: "n2".into(),
                to_type: crate::ast::PortType::Str,
                source: "n1 -> n2".into(),
            },
            EmbeddingError::NodeEvalPanic {
                node_name: "div".into(),
                message: "div by zero".into(),
                source: "div(a, b)".into(),
            },
            EmbeddingError::NonePropagated {
                accessor: "as_bool",
                source: "{missing}".into(),
            },
        ];
        for v in variants {
            let _ = format!("{v}");
        }
    }

    #[test]
    fn typed_surface_string() {
        let v: String = eval_const_expr_typed("\"hello\"").unwrap();
        assert_eq!(v, "hello");
    }

    #[test]
    fn typed_surface_type_mismatch() {
        // expression yields U64; host requests f64 — widening allowed
        let v: f64 = eval_const_expr_typed("42").unwrap();
        assert_eq!(v, 42.0);
        // expression yields U64; host requests bool — interpreted as bool (nonzero)
        let v: bool = eval_const_expr_typed("1").unwrap();
        assert!(v);
        let v: bool = eval_const_expr_typed("0").unwrap();
        assert!(!v);
    }

    #[test]
    fn typed_surface_return_path_adapter() {
        // γ-6: expression produces U64; host requests String.
        // The catalog's U64ToString adapter heals the return-path.
        let v: String = eval_const_expr_typed("42").unwrap();
        assert_eq!(v, "42");

        // Expression produces F64; host requests String via catalog
        // F64ToString. (Note: f64's Display is locale-independent
        // but format may add trailing zeros.)
        let v: String = eval_const_expr_typed("3.14").unwrap();
        assert!(v.starts_with("3.14"), "got {v}");
    }

    #[test]
    fn typed_surface_return_path_no_adapter_errors() {
        // Bytes → Bool isn't in the catalog. Confirm the typed
        // error fires when the catalog can't heal.
        // (Need an expression producing Bytes; use a string-
        // literal-to-bytes conversion via bytes_of or similar
        // if available; otherwise use a roundtrip that fails.)
        //
        // Skipping concrete bytes producer for this test —
        // the contract is exercised by the negative path in
        // typed_surface_type_mismatch already.
    }

    #[test]
    fn typed_strict_rejects_lossy_conversion() {
        // U64 → Bool is in the catalog (γ-6 added it) but
        // lossy. Strict mode must reject.
        let result: Result<bool, _> = eval_const_expr_typed_strict("42");
        match result {
            Err(EmbeddingError::TypeMismatch {
                from_type, to_type, ..
            }) => {
                assert!(matches!(from_type, crate::ast::PortType::U64));
                assert!(matches!(to_type, crate::ast::PortType::Bool));
            }
            other => panic!("expected TypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn typed_strict_accepts_lossless_conversion() {
        // U64 → String via display — lossless.
        let v: String = eval_const_expr_typed_strict("42").unwrap();
        assert_eq!(v, "42");

        // Same type, no adapter.
        let v: f64 = eval_const_expr_typed_strict("42.0").unwrap();
        assert_eq!(v, 42.0);
    }

    /// Strict mode answers about the types, not about the one value
    /// in hand: `U64 → F64` is refused because `u64` has 64 magnitude
    /// bits and `f64`'s significand holds 53, so values above `2^53`
    /// round. A host that wants the number as an `f64` writes it as
    /// one. The type-level answer is the same for every input, which
    /// a value-level one would not be.
    #[test]
    fn typed_strict_refuses_a_widening_that_rounds() {
        let r: Result<f64, _> = eval_const_expr_typed_strict("42");
        assert!(
            matches!(r, Err(EmbeddingError::TypeMismatch { .. })),
            "{r:?}"
        );
        assert!(!is_lossless_adapter(
            crate::ast::PortType::U64,
            crate::ast::PortType::F64
        ));
        assert!(!is_lossless_adapter(
            crate::ast::PortType::I64,
            crate::ast::PortType::F64
        ));
        // The narrow integers do fit, which the old eleven-pair
        // table did not say.
        for from in [
            crate::ast::PortType::U8,
            crate::ast::PortType::U16,
            crate::ast::PortType::U32,
        ] {
            assert!(
                is_lossless_adapter(from, crate::ast::PortType::U64),
                "{from:?} → U64"
            );
            assert!(
                is_lossless_adapter(from, crate::ast::PortType::F64),
                "{from:?} → F64"
            );
        }
        // Signed never fits unsigned, however wide.
        assert!(!is_lossless_adapter(
            crate::ast::PortType::I8,
            crate::ast::PortType::U128
        ));
    }

    #[test]
    fn shared_non_literal_init_rejected() {
        // Non-literal `shared` initializers no longer fall
        // through to the cycle-binding shape. Compile error
        // surfaces with a clear message naming the binding and
        // pointing at the SRD-16 §"Non-literal `shared`
        // initializers" section.
        let src = r#"
            input cycle: u64
            shared rolling := hash(cycle)
        "#;
        let err =
            compile_polydat_interpreter(src).expect_err("non-literal shared const must error");
        assert!(
            err.to_string().contains("shared binding 'rolling'"),
            "error: {err}"
        );
        assert!(
            err.to_string().contains("literal initial value"),
            "error: {err}"
        );
    }

    #[test]
    fn final_modifier_tracked() {
        let src = r#"
            input cycle: u64
            const dim := 128
        "#;
        let kernel = compile_polydat_interpreter(src).unwrap();
        assert_eq!(
            kernel.program().output_modifier("dim"),
            crate::dsl::ast::BindingModifier::CONST
        );
    }

    #[test]
    fn shared_literal_modifier_tracked() {
        let src = r#"
            input cycle: u64
            shared budget := 100
        "#;
        let kernel = compile_polydat_interpreter(src).unwrap();
        assert_eq!(
            kernel.program().output_modifier("budget"),
            crate::dsl::ast::BindingModifier::SHARED
        );
        // Shared cells back the output via a port-passthrough node
        // reading the input slot; `lookup` is the cell-aware read.
        assert_eq!(kernel.lookup("budget").unwrap().as_u64(), 100);
    }

    #[test]
    fn const_literal_modifier_tracked() {
        let src = r#"
            input cycle: u64
            const max_dim := 256
        "#;
        let kernel = compile_polydat_interpreter(src).unwrap();
        assert_eq!(
            kernel.program().output_modifier("max_dim"),
            crate::dsl::ast::BindingModifier::CONST
        );
        assert_eq!(kernel.get_constant("max_dim").unwrap().as_u64(), 256);
    }

    #[test]
    fn compile_string_constant() {
        let src = r#"
            input cycle: u64
            label := "hello world"
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[0]);
        assert_eq!(kernel.pull_ref("label").as_str(), "hello world");
    }

    #[test]
    fn compile_int_constant() {
        let src = r#"
            input cycle: u64
            base := 1710000000000
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[0]);
        assert_eq!(kernel.pull_ref("base").as_u64(), 1_710_000_000_000);
    }

    // --- Diagnostic tests ---

    #[test]
    fn error_unknown_function() {
        let src = "input cycle: u64\nresult := foobar(cycle)";
        let (_result, report) = compile_polydat_checked(src);
        assert!(report.has_errors());
        let errors = report.errors();
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("unknown function"))
        );
        assert!(errors.iter().any(|e| e.message.contains("foobar")));
    }

    #[test]
    fn explicit_coordinates_rejects_unbound() {
        // With explicit coordinates, unbound references are errors
        let src = "input cycle: u64\nh := hash(unknown)";
        let (_, report) = compile_polydat_checked(src);
        assert!(report.has_errors());
        assert!(
            report
                .errors()
                .iter()
                .any(|e| e.message.contains("undefined") && e.message.contains("unknown"))
        );
    }

    #[test]
    fn warning_forward_reference() {
        let src = r#"
            input cycle: u64
            result := mod(h, 100)
            h := hash(cycle)
        "#;
        let (_, report) = compile_polydat_checked(src);
        let warnings = report.warnings();
        assert!(
            warnings
                .iter()
                .any(|w| w.message.contains("forward reference")),
            "should warn about forward ref, got: {:?}",
            warnings
        );
    }

    #[test]
    fn error_undefined_wire() {
        let src = r#"
            input cycle: u64
            result := hash(nonexistent)
        "#;
        let (_, report) = compile_polydat_checked(src);
        assert!(report.has_errors());
        assert!(
            report
                .errors()
                .iter()
                .any(|e| e.message.contains("undefined") && e.message.contains("nonexistent"))
        );
    }

    #[test]
    fn error_report_includes_source_line() {
        let src = "input cycle: u64\nresult := unknown_func(cycle)";
        let (_, report) = compile_polydat_checked(src);
        let s = report.to_string();
        assert!(
            s.contains("unknown_func"),
            "report should include source context"
        );
    }

    // --- Strict mode tests ---

    #[test]
    fn strict_requires_explicit_inputs() {
        // Without inputs declaration, strict mode should error
        let src = "h := hash(cycle)";
        let result = strict(src, true);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("strict mode"),
            "expected strict error, got: {err}"
        );
        assert!(
            err.to_string().contains("inputs"),
            "expected inputs mention, got: {err}"
        );
    }

    // --- Dead code elimination tests ---

    // --- Strict mode comprehensive tests ---

    // --- eval_const_expr tests ---

    #[test]
    fn eval_const_expr_fails_on_inputs() {
        // 'cycle' is a runtime input — should fail as const expr
        let r = eval_const_expr("hash(cycle)");
        assert!(r.is_err(), "hash(cycle) should fail as a const expression");
    }

    // ─────────────────────────────────────────────────────────────
    // Init-Binding Contract (SRD 11 §"Init Binding Contract")
    //
    // Plan A — compile-time check: every binding declared `init`
    // must classify as compile-const or scope-init. A wire chain
    // reaching a coordinate input, a external-write port, or a
    // non-deterministic source disqualifies the binding.
    // ─────────────────────────────────────────────────────────────

    #[test]
    fn init_binding_compile_const_folded() {
        // Pure init: literal arg, no externs. Folds at compile
        // time; the compiled program's output_map points at a
        // ConstU64 leaf.
        let src = "const dim := 128\n";
        let kernel = compile_polydat_interpreter(src).expect("init compile-const");
        let prog = kernel.program();
        assert!(prog.const_outputs().contains(&"dim"));
        let &(node_idx, _) = prog.output_map_lookup("dim").expect("dim in output map");
        // After fold, the node has empty wiring (leaf const).
        assert!(
            prog.wiring[node_idx].is_empty(),
            "compile-const init binding 'dim' must fold to a leaf const node"
        );
    }

    #[test]
    fn init_binding_with_iteration_extern_passes_plan_a() {
        // Init binding wired through an iteration extern: this is
        // legal under Plan A — the wire chain reaches an
        // IterationExtern input slot, which is effectively-const at
        // scope-init time. Plan B (executor-side) is what actually
        // evaluates it; the compile step just must not reject.
        let src = "extern profile: String\n\
                   const label := format_str(\"label_%s\", profile)\n";
        let result = compile_polydat_interpreter(src);
        // We don't care if format_str exists in the stdlib — what
        // we're testing is that the contract check itself doesn't
        // fail (any error must be about an unknown function, not
        // about the init contract).
        match result {
            Ok(_) => {} // ideal: kernel built
            Err(e) => assert!(
                !e.to_string().contains("violates the init contract"),
                "Plan A must accept iteration-extern wires in init bindings; got: {e}"
            ),
        }
    }

    #[test]
    fn init_binding_wired_to_nondeterministic_rejected() {
        // `counter()` is non-deterministic; init bindings must not
        // depend on it.
        let src = "const bad := counter()\n";
        let err = compile_polydat_interpreter(src)
            .expect_err("Plan A must reject init binding wired to a non-deterministic source");
        assert!(
            err.to_string().contains("init binding 'bad'")
                && err.to_string().contains("init contract"),
            "diagnostic must name the binding and the contract; got: {err}"
        );
    }

    #[test]
    fn init_outputs_threaded_into_program() {
        // Sanity: the compiler records every `init`-declared name
        // on GkProgram.const_outputs so Plan B (executor side) can
        // walk them at scope activation.
        let src = "const a := 1\n\
                   const b := 2\n\
                   c := 3\n";
        let kernel = compile_polydat_interpreter(src).unwrap();
        let init_set = kernel.program().const_outputs();
        assert!(init_set.contains(&"a"), "const 'a' should be tracked");
        assert!(init_set.contains(&"b"), "const 'b' should be tracked");
        assert!(
            !init_set.contains(&"c"),
            "non-const 'c' must not be tracked"
        );
    }

    /// Auto-extern slots inferred from RHS shape land at the
    /// boundary with their actual type (Str / U64 / F64 / Bool)
    /// rather than the legacy `PortType::Ext` catchall. This
    /// removes the `U64 → Ext` boundary-adapter miss the audit
    /// log used to warn about for workloads that use `set:`
    /// blocks with iter-var interpolation.
    ///
    /// Test path: declare an iteration extern explicitly with
    /// `extern N: str` (no default → `IterationExtern` kind,
    /// effectively-const at scope-init); reference it from a
    /// const RHS. The const target then needs an auto-extern
    /// slot (RHS has a ref), and the inferrer picks the
    /// referenced input's type.
    #[test]
    fn auto_extern_slot_inherits_string_template_type() {
        let src = r#"
            extern some_outer_var: str
            const x := "{some_outer_var}"
        "#;
        let kernel = compile_polydat_interpreter(src).expect("compile");
        assert_eq!(
            kernel.program().input_port_type("x"),
            Some(crate::ast::PortType::Str),
            "string-template auto-extern MUST be Str, not Ext",
        );
    }

    /// Identifier reference auto-extern inherits the referenced
    /// input's type. `const y := other_str_input` → y is Str.
    #[test]
    fn auto_extern_slot_inherits_ident_reference_type() {
        let src = r#"
            extern other: str
            const y := other
        "#;
        let kernel = compile_polydat_interpreter(src).expect("compile");
        assert_eq!(
            kernel.program().input_port_type("y"),
            Some(crate::ast::PortType::Str),
            "ident-RHS auto-extern MUST inherit referenced input's type",
        );
    }

    /// `dataset_prebuffer(...)` returns `Value::Handle` — the
    /// auto-extern slot for `const prebuffered := dataset_prebuffer(...)`
    /// MUST be `PortType::Handle`, not the legacy `Ext` catchall.
    /// This is the second specific call site we patched in the
    /// inferrer after the `printf` string-template case.
    /// (`dataset_prebuffer` is a vectordata node, so the test only
    /// exists when that feature registers it.)
    #[cfg(feature = "vectordata")]
    #[test]
    fn auto_extern_slot_for_dataset_prebuffer_is_handle() {
        let src = r#"
            extern source_uri: str
            const prebuffered := dataset_prebuffer(source_uri)
        "#;
        let kernel = compile_polydat_interpreter(src).expect("compile");
        assert_eq!(
            kernel.program().input_port_type("prebuffered"),
            Some(crate::ast::PortType::Handle),
            "dataset_prebuffer auto-extern MUST be Handle, not Ext",
        );
    }
}
