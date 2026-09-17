// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Compile-time support for the `for` construct (SRD 113 steps 2 and
//! 3): element typing, body-to-child-program lowering, and the
//! metadata a parent program carries for each traversal and producer.
//!
//! A `for` body compiles exactly once, at parent compile time, into a
//! child [`PolydatProgram`] keyed by the statement's lexical position.
//! Its element names become `IterationExtern` inputs typed from the
//! comprehension's sources; outer wires it references become cascade
//! externs typed from the parent's manifest. Activation (step 4) only
//! ever allocates state over that program.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use crate::ast::PortType;
use crate::iteration::comprehension::source::{LiteralValue, Source};
use crate::iteration::comprehension::{
    Comprehension, Mode as ValidationMode, StreamerValue, ValidationWarning,
};
use crate::kernel::PolydatProgram;

use super::ast::{
    Arg, Binding, BindingModifier, CallExpr, Expr, ExternPort, ForSource, ForSourceKind, ForStmt,
    InputDecl, PolydatFile, Statement,
};
use super::lexer::Span;

/// A compiled traversal: one `for` statement and the child program its
/// body lowered to.
#[derive(Debug, Clone)]
pub struct Traversal {
    /// Lexical position of the `for` statement in its parent.
    pub span: Span,
    /// The text after `for`, as written.
    pub source_text: String,
    /// The comprehension traversed, with a producer reference already
    /// resolved to the producer's comprehension.
    pub comprehension: Comprehension,
    /// Element names and their compile-time types, in tuple order.
    pub elements: Vec<(String, PortType)>,
    /// Outer wires the body references, with the parent's types. Bound
    /// from the parent at activation.
    pub cascade: Vec<(String, PortType)>,
    /// The body's program on the interpreter. Compiled once; every
    /// interpreter activation shares it.
    pub program: Arc<PolydatProgram>,
    /// The body as the parent compiled it, for activations on the other
    /// engines (engine parity, step 8): compiled once per engine, on the
    /// first activation that asks.
    pub body: Arc<BodySource>,
}

/// A traversal body as its parent compiled it: the child file and the
/// compiler settings the parent used, so the same body compiles on any
/// engine, once, keyed by the engine as the interpreter's program is
/// keyed by the body's position (SRD 113 §5.1).
pub struct BodySource {
    pub(crate) file: PolydatFile,
    pub(crate) source_text: String,
    pub(crate) source_dir: Option<std::path::PathBuf>,
    pub(crate) lib_paths: Vec<std::path::PathBuf>,
    pub(crate) strict: bool,
    pub(crate) context_label: String,
    pub(crate) cursor_limit: Option<u64>,
    pub(crate) pragmas: super::pragmas::PragmaSet,
    /// The modules the parent program had resolved when the body was
    /// lowered, its own definitions included, so the body sees them
    /// wherever it compiles, as the parent did.
    pub(super) modules: HashMap<String, super::modules::ResolvedModule>,
    /// The body's program per engine, built on first use.
    pub(crate) programs: Mutex<HashMap<crate::Engine, Arc<dyn crate::kernel::KernelProgram>>>,
    /// The compile ledger of the tree, for the body's programs on every
    /// engine.
    pub(crate) ledger: Arc<crate::kernel::CompileLedger>,
}

impl BodySource {
    /// The body's source as the parent lowered it: the implicit `cycle`
    /// input, one extern per element and per cascaded wire, then the
    /// body's statements.
    pub fn source_text(&self) -> &str {
        &self.source_text
    }
}

impl std::fmt::Debug for BodySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BodySource")
            .field("context", &self.context_label)
            .field("statements", &self.file.statements.len())
            .finish()
    }
}

impl Traversal {
    /// The body's program on `engine`, compiled on the first call for
    /// that engine and shared by every activation after it, as the
    /// interpreter's program is (SRD 113 §5.1, §5.2), its own `for`
    /// statements included: an activation on any engine opens them.
    pub fn program_on(
        &self,
        engine: crate::Engine,
    ) -> Result<Arc<dyn crate::kernel::KernelProgram>, crate::KernelError> {
        if matches!(engine, crate::Engine::Interpreter(_)) {
            return Ok(self.program.clone());
        }
        let mut programs = self
            .body
            .programs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(program) = programs.get(&engine) {
            return Ok(program.clone());
        }
        let program = super::compile::Compiler::compile_body_on(&self.body, engine)?.into_program();
        programs.insert(engine, program.clone());
        Ok(program)
    }
}

/// A producer binding, `name := for ...`, recorded on the program that
/// declares it so traversals over `name` resolve at compile time.
#[derive(Debug, Clone)]
pub struct Producer {
    /// The wire the producer binds.
    pub name: String,
    /// Where the binding appears.
    pub span: Span,
    /// The text after `for`, as written.
    pub source_text: String,
    /// The comprehension, with derivations resolved.
    pub comprehension: Comprehension,
}

/// Split a parsed file into the statements the parent compiles directly,
/// the `for` statements to lower into child programs, and the producer
/// bindings to record. Order within each group is preserved.
///
/// Producer bindings stay in the parent as `const name := streamer(...)`
/// calls carrying the resolved comprehension, so the wire exists with a
/// `Streamer` value on it (SRD 113 §3.1). Derivations resolve against
/// producers bound earlier in the file, in document order.
pub fn strip_for_forms(
    file: &PolydatFile,
    mode: ValidationMode,
    events: &mut Vec<super::events::CompileEvent>,
) -> Result<(PolydatFile, Vec<ForStmt>, Vec<Producer>), String> {
    let mut parent = Vec::with_capacity(file.statements.len());
    let mut fors = Vec::new();
    let mut producers: Vec<Producer> = Vec::new();
    for stmt in &file.statements {
        match stmt {
            Statement::For(f) => fors.push(f.clone()),
            Statement::Binding(b) if matches!(b.value, Expr::For(_)) => {
                let Expr::For(source) = &b.value else {
                    unreachable!()
                };
                let (comprehension, warnings) = resolve_source_with(source, &producers, mode)?;
                events.extend(warning_events(source, &warnings));
                let name = b.targets.join(",");
                let value = StreamerValue::new(source.text.clone(), comprehension.clone());
                parent.push(Statement::Binding(Binding {
                    targets: b.targets.clone(),
                    value: Expr::Call(CallExpr {
                        func: "streamer".into(),
                        args: vec![Arg::Positional(Expr::StringLit(value.to_json(), b.span))],
                        span: b.span,
                    }),
                    modifier: BindingModifier::CONST,
                    type_annotation: None,
                    span: b.span,
                }));
                producers.push(Producer {
                    name,
                    span: b.span,
                    source_text: source.text.clone(),
                    comprehension,
                });
            }
            other => parent.push(other.clone()),
        }
    }
    Ok((PolydatFile { statements: parent }, fors, producers))
}

/// Resolve a traversal's source to a comprehension: inline text as is, a
/// producer reference to the producer bound in the same scope, and a
/// derivation to the base producer with its filter and order applied.
pub fn resolve_source(source: &ForSource, producers: &[Producer]) -> Result<Comprehension, String> {
    resolve_source_with(source, producers, ValidationMode::Permissive).map(|(c, _)| c)
}

/// The compile events for the validator's warnings on `source`, each
/// placed at the statement that names the comprehension.
pub(super) fn warning_events(
    source: &ForSource,
    warnings: &[ValidationWarning],
) -> Vec<super::events::CompileEvent> {
    warnings
        .iter()
        .map(|w| super::events::CompileEvent::ComprehensionWarning {
            source: source.text.clone(),
            line: source.span.line,
            col: source.span.col,
            warning: w.to_string(),
        })
        .collect()
}

/// [`resolve_source`] under a validation mode (comprehension_forms.md
/// §5.8): a strict compile refuses a degenerate composition, a
/// permissive one returns it as a warning for the compile event log.
pub fn resolve_source_with(
    source: &ForSource,
    producers: &[Producer],
    mode: ValidationMode,
) -> Result<(Comprehension, Vec<ValidationWarning>), String> {
    let find = |name: &str| -> Result<Comprehension, String> {
        producers
            .iter()
            .rev()
            .find(|p| p.name == name)
            .map(|p| p.comprehension.clone())
            .ok_or_else(|| {
                let known: Vec<&str> = producers.iter().map(|p| p.name.as_str()).collect();
                format!(
                    "`for {}` at line {}, col {}: no producer named '{name}' is bound in this scope{}",
                    source.text,
                    source.span.line,
                    source.span.col,
                    if known.is_empty() { String::new() } else { format!("; producers here: {}", known.join(", ")) }
                )
            })
    };
    let at = |e: &dyn std::fmt::Display| {
        format!(
            "`for {}` at line {}, col {}: {e}",
            source.text, source.span.line, source.span.col
        )
    };
    let comprehension = match &source.kind {
        ForSourceKind::Comprehension(c) => c.clone(),
        ForSourceKind::Producer(name) => find(name)?,
        ForSourceKind::Derived {
            base,
            filter,
            order,
        } => {
            let mut c = find(base)?;
            if let Some(pred) = filter {
                c = Comprehension::filter(c, pred.clone());
            }
            if let Some(spec) = order {
                let (strategy, truncation) = parse_order(spec).map_err(|e| at(&e))?;
                c = Comprehension::order(c, strategy, truncation);
            }
            c
        }
    };
    // Validation is a stage of the compile (comprehension_forms.md §5):
    // a comprehension that violates a V-axiom is refused at the
    // statement that names it, whether written inline or derived.
    let report =
        crate::iteration::comprehension::validate(&comprehension, mode).map_err(|e| at(&e))?;
    Ok((comprehension, report.warnings))
}

/// Parse an `order` spec such as `halton/5` into the algebra's strategy
/// and truncation by running it through the comprehension parser on a
/// one-clause carrier.
fn parse_order(
    spec: &str,
) -> Result<(crate::iteration::comprehension::StrategyName, Option<u64>), String> {
    let carrier = format!("__o in 0..1 order {spec}");
    let legacy = crate::iteration::comprehension::parse::parse_comprehension_text(&carrier)?;
    let algebra = crate::iteration::comprehension::spec::legacy_to_algebra(&legacy)
        .map_err(|e| e.to_string())?;
    match algebra {
        Comprehension::Order {
            strategy,
            truncation,
            ..
        } => Ok((strategy, truncation)),
        other => Err(format!(
            "order spec `{spec}` did not produce an ordering (got {other:?})"
        )),
    }
}

/// Type each element name of a comprehension from its source, per SRD
/// 113 §3.3. `probe` types a generator call expression the way the
/// enclosing compiler would.
pub fn element_types(
    comprehension: &Comprehension,
    probe: &mut dyn FnMut(&str) -> Result<PortType, String>,
) -> Result<Vec<(String, PortType)>, String> {
    let mut out = Vec::new();
    collect_element_types(comprehension, probe, &mut out)?;
    Ok(out)
}

fn collect_element_types(
    c: &Comprehension,
    probe: &mut dyn FnMut(&str) -> Result<PortType, String>,
    out: &mut Vec<(String, PortType)>,
) -> Result<(), String> {
    match c {
        Comprehension::Clause { name, source } => {
            if out.iter().any(|(n, _)| n == name) {
                return Ok(());
            }
            let ty = source_type(name, source, probe)?;
            out.push((name.clone(), ty));
            Ok(())
        }
        Comprehension::Cartesian { children } | Comprehension::Zip { children, .. } => {
            for child in children {
                collect_element_types(child, probe, out)?;
            }
            Ok(())
        }
        Comprehension::Union { children } => {
            // Union children share one tuple shape; the first child's
            // types stand for all of them.
            if let Some(first) = children.first() {
                collect_element_types(first, probe, out)?;
            }
            Ok(())
        }
        Comprehension::Filter { child, .. } | Comprehension::Order { child, .. } => {
            collect_element_types(child, probe, out)
        }
    }
}

fn source_type(
    name: &str,
    source: &Source,
    probe: &mut dyn FnMut(&str) -> Result<PortType, String>,
) -> Result<PortType, String> {
    match source {
        Source::Literal { values } => {
            let mut ty: Option<PortType> = None;
            for v in values {
                let t = match v {
                    LiteralValue::Int(_) => PortType::U64,
                    LiteralValue::Float(_) => PortType::F64,
                    LiteralValue::String(_) => PortType::Str,
                    LiteralValue::Bool(_) => PortType::Bool,
                    LiteralValue::Json(_) => PortType::Json,
                };
                match ty {
                    None => ty = Some(t),
                    // An int among floats widens; anything else is mixed.
                    Some(PortType::F64) if t == PortType::U64 => {}
                    Some(PortType::U64) if t == PortType::F64 => ty = Some(PortType::F64),
                    Some(prev) if prev != t => {
                        return Err(format!(
                            "element '{name}': literal list mixes {prev:?} and {t:?} values; a comprehension element has one type"
                        ));
                    }
                    Some(_) => {}
                }
            }
            ty.ok_or_else(|| format!("element '{name}': literal list is empty"))
        }
        Source::IntRange { .. } => Ok(PortType::U64),
        Source::ContinuousInterval { .. } | Source::Distribution { .. } => Ok(PortType::F64),
        // Workload parameter lists are text until a host types them.
        Source::WorkloadParamList { .. } => Ok(PortType::Str),
        Source::Generator { expr, .. } => {
            let head = expr.trim();
            if head.starts_with("partitions(")
                || head.starts_with("subdivide(")
                || head.ends_with(".partitions")
            {
                return Ok(PortType::Ext);
            }
            probe(head)
                .map_err(|e| format!("element '{name}': cannot type generator `{head}`: {e}"))
        }
    }
}

/// Names a body declares itself, so they are not cascaded from the
/// parent: binding targets, element names, declared inputs and externs,
/// module names, and cursors with their projection sources.
fn body_declared(body: &[Statement], elements: &[(String, PortType)]) -> BTreeSet<String> {
    let mut names: BTreeSet<String> = elements.iter().map(|(n, _)| n.clone()).collect();
    names.insert("cycle".to_string());
    for stmt in body {
        match stmt {
            Statement::Binding(b) => names.extend(b.targets.iter().cloned()),
            Statement::InputDecl(d) => {
                names.insert(d.name.clone());
            }
            Statement::ExternPort(p) => {
                names.insert(p.name.clone());
            }
            Statement::ModuleDef(m) => {
                names.insert(m.name.clone());
            }
            Statement::Cursor(c) => {
                names.insert(c.name.clone());
            }
            Statement::Pragma { .. } => {}
            Statement::For(f) => {
                // A nested traversal's own elements are its business,
                // but a producer it binds is visible after it.
                let _ = f;
            }
            Statement::Tile(t) => {
                names.insert(t.name.clone());
            }
        }
    }
    names
}

/// Wires a tile's holes and branch conditions reference.
fn tile_references(pieces: &[super::ast::TilePiece], out: &mut BTreeSet<String>) {
    use super::ast::TilePiece;
    use super::refs::collect_expr_refs;
    for piece in pieces {
        match piece {
            TilePiece::Static(_) => {}
            TilePiece::Hole(h) => collect_expr_refs(&h.expr, out),
            TilePiece::Projection { body, .. } => tile_references(body, out),
            TilePiece::Branch {
                cond,
                then,
                otherwise,
                ..
            } => {
                collect_expr_refs(cond, out);
                tile_references(then, out);
                if let Some(o) = otherwise {
                    tile_references(o, out);
                }
            }
        }
    }
}

/// Names a body references, gathered from binding expressions, cursor
/// constructors and `over` clauses, extern defaults, and nested `for`
/// bodies. Nested bodies contribute their outer references so the
/// cascade reaches through every level.
fn body_references(body: &[Statement], out: &mut BTreeSet<String>) {
    use super::refs::collect_expr_refs;
    for stmt in body {
        match stmt {
            Statement::Binding(b) => collect_expr_refs(&b.value, out),
            Statement::ExternPort(p) => {
                if let Some(d) = &p.default {
                    collect_expr_refs(d, out);
                }
            }
            Statement::Cursor(c) => {
                collect_expr_refs(&c.constructor, out);
                if let Some(over) = &c.over {
                    collect_expr_refs(over, out);
                }
            }
            Statement::For(f) => {
                let mut inner = BTreeSet::new();
                body_references(&f.body, &mut inner);
                let own = body_declared(&f.body, &[]);
                let elems: BTreeSet<String> = f.source.element_names().into_iter().collect();
                for n in inner {
                    if !own.contains(&n) && !elems.contains(&n) {
                        out.insert(n);
                    }
                }
            }
            Statement::Tile(t) => tile_references(&t.pieces, out),
            Statement::InputDecl(_) | Statement::ModuleDef(_) | Statement::Pragma { .. } => {}
        }
    }
}

/// Build the child file for a traversal body: an implicit `cycle`
/// input, one extern per element, one cascade extern per outer wire
/// the body references and the parent exposes, then the body itself.
/// Returns the file and the cascade list.
pub fn child_file(
    f: &ForStmt,
    comprehension: &Comprehension,
    elements: &[(String, PortType)],
    type_of: &dyn Fn(&str) -> Option<PortType>,
) -> Result<(PolydatFile, Vec<(String, PortType)>), String> {
    // §7: a body may not declare a coordinate other than `cycle`.
    for stmt in &f.body {
        if let Statement::InputDecl(d) = stmt
            && d.name != "cycle"
        {
            return Err(format!(
                "`for {}` at line {}, col {}: a traversal body cannot declare input '{}'; only `cycle` is a coordinate inside a body, and the comprehension supplies the rest",
                f.source.text, f.span.line, f.span.col, d.name
            ));
        }
    }
    let declared = body_declared(&f.body, elements);
    let mut referenced = BTreeSet::new();
    body_references(&f.body, &mut referenced);
    // Outer wires the comprehension's own sources reference through
    // `{name}` also cascade, so the tuple evaluation sees them.
    referenced.extend(comprehension.referenced_source_names());

    let mut cascade = Vec::new();
    for name in referenced {
        if declared.contains(&name) {
            continue;
        }
        let ty = type_of(&name);
        if let Some(ty) = ty {
            cascade.push((name, ty));
        }
        // Names the parent does not expose stay unresolved; the child
        // compile reports them as unknown wires with the body's spans.
    }

    let span = f.span;
    let mut statements = Vec::with_capacity(f.body.len() + elements.len() + cascade.len() + 1);
    if !f.body.iter().any(|s| matches!(s, Statement::InputDecl(_))) {
        statements.push(Statement::InputDecl(InputDecl {
            name: "cycle".into(),
            ty: Some("u64".into()),
            span,
        }));
    }
    for (name, ty) in elements {
        statements.push(Statement::ExternPort(ExternPort {
            name: name.clone(),
            typ: ty.to_keyword().to_string(),
            default: None,
            span,
        }));
    }
    for (name, ty) in &cascade {
        statements.push(Statement::ExternPort(ExternPort {
            name: name.clone(),
            typ: ty.to_keyword().to_string(),
            default: None,
            span,
        }));
    }
    statements.extend(f.body.iter().cloned());
    Ok((PolydatFile { statements }, cascade))
}
