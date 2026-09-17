// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Module resolution subsystem for the Polydat DSL compiler.
//!
//! Handles locating, parsing, and caching `.polydat` module files so the
//! compiler can inline them at call sites.  Resolution order:
//!
//! 1. In-process cache (already-resolved modules)
//! 2. `<name>.polydat` in the workload-local `source_dir`
//! 3. Any `.polydat` file in `source_dir` that exports a binding named `<name>`
//! 4. The same two searches repeated for each library path
//!    (`CompileOptions::lib_paths`, the binary's `--lib`)
//! 5. The embedded standard library

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::compile::assembly::{PolydatAssembler, WireRef};
use crate::dsl::ast::*;
use crate::dsl::lexer;
use crate::dsl::parser;
use crate::dsl::validate::collect_references;

use super::compile::{Compiler, STDLIB_MODULES};

/// A resolved Polydat module ready for inlining.
#[derive(Clone)]
pub(super) struct ResolvedModule {
    /// Input parameter names (from formal signature or inferred).
    pub(super) inputs: Vec<String>,
    /// Input parameter types (from formal signature; empty if inferred).
    pub(super) input_types: Vec<Option<String>>,
    /// Output binding names (from formal signature or last binding).
    pub(super) outputs: Vec<String>,
    /// Output types (from formal signature; empty if inferred).
    /// Reserved for future strict-mode type checking of downstream consumers.
    #[allow(dead_code)]
    pub(super) output_types: Vec<Option<String>>,
    /// Whether this module has a formal typed signature.
    pub(super) is_formal: bool,
    /// The module's AST statements.
    pub(super) statements: Vec<Statement>,
}

impl Compiler {
    /// Generate a fresh anonymous node name for desugared intermediates.
    ///
    /// When a user-level binding is in scope (set by `compile_binding`),
    /// the name is prefixed with that binding's LHS so type-mismatch
    /// errors point at a recognisable source name
    /// (`overscan__anon_3`) instead of an opaque counter
    /// (`__anon_14`).
    pub(super) fn anon_name(&mut self) -> String {
        let name = match &self.current_binding {
            Some(b) => format!("{b}__anon_{}", self.anon_counter),
            None => format!("__anon_{}", self.anon_counter),
        };
        self.anon_counter += 1;
        name
    }

    /// Splice a resolver between any `Str`-producing wire and any
    /// `Handle`-typed input port on a freshly built node, per the
    /// function's `default_resolver` hint
    /// (SRD 53 §"Source-string call-site sugar").
    ///
    /// For each `Handle` input port:
    ///   - if the wire already produces `Handle`, leave it alone;
    ///   - if the wire produces `Str` and the function declares a
    ///     `DefaultResolver`, build the resolver
    ///     (`dataset_open(<wire>, <facet_lit>)` or
    ///     `dataset_group_open(<wire>)`), add it as an anon node, and
    ///     redirect the wire to the resolver's output;
    ///   - if the wire produces something else, leave it — the
    ///     standard type-adapter pass will report the mismatch with a
    ///     standard error.
    ///
    /// No-op if the function isn't in the registry, has no
    /// `default_resolver`, or the node has no Handle inputs.
    pub(super) fn auto_promote_handle_inputs(
        &mut self,
        asm: &mut crate::compile::assembly::PolydatAssembler,
        func_name: &str,
        node: &dyn crate::ast::PolydatNode,
        wire_refs: &mut [crate::compile::assembly::WireRef],
    ) -> Result<(), String> {
        use crate::ast::PortType;
        #[cfg(feature = "vectordata")]
        use crate::compile::assembly::WireRef;
        #[cfg(feature = "vectordata")]
        use crate::dsl::registry::DefaultResolver;
        use crate::dsl::registry::registry;
        #[cfg(feature = "vectordata")]
        use crate::library::identity::ConstStr;

        // Find the FuncSig — only some functions opt into auto-promotion.
        let resolver = registry()
            .into_iter()
            .find(|sig| sig.name == func_name)
            .and_then(|sig| sig.default_resolver);
        let resolver = match resolver {
            Some(r) => r,
            None => return Ok(()),
        };

        let meta = node.meta();
        // Walk this node's input ports. The slot/wire alignment is the
        // same as wire_refs (positional), so we iterate the wire-typed
        // ports in order to match wire_refs[i].
        let mut wire_idx = 0;
        for slot in &meta.ins {
            let port = match slot {
                crate::ast::Slot::Wire(p) => p,
                crate::ast::Slot::Const { .. } => continue,
            };
            if wire_idx >= wire_refs.len() {
                break;
            }
            // Only promote Handle inputs.
            if port.typ == PortType::Handle {
                let src = &wire_refs[wire_idx];
                let src_type = asm.wire_type(src);
                if src_type == Some(PortType::Str) {
                    // Build the resolver call as anonymous nodes.
                    let resolver_name = self.anon_name();
                    // The SRD-53 source-string resolvers are dataset
                    // openers from `library::vectors`, which exists
                    // only with the `vectordata` feature. Without it,
                    // auto-promotion must fail loudly — silently
                    // leaving the Str wire in place would surface as
                    // a confusing type mismatch downstream.
                    #[cfg(not(feature = "vectordata"))]
                    {
                        let _ = &resolver_name;
                        return Err(format!(
                            "input '{}' needs the {resolver:?} source-string                              resolver, but polydat was built without the                              'vectordata' feature",
                            port.name
                        ));
                    }
                    #[cfg(feature = "vectordata")]
                    match resolver {
                        DefaultResolver::Facet(facet) => {
                            // Anonymous facet const wire.
                            let facet_const = self.anon_name();
                            asm.add_node(
                                &facet_const,
                                Box::new(ConstStr::new(facet.to_string())),
                                vec![],
                            );
                            let resolver_node: Box<dyn crate::ast::PolydatNode> =
                                Box::new(crate::library::vectors::DatasetOpen::new());
                            asm.add_node(
                                &resolver_name,
                                resolver_node,
                                vec![src.clone(), WireRef::node(facet_const)],
                            );
                        }
                        DefaultResolver::Group => {
                            let resolver_node: Box<dyn crate::ast::PolydatNode> =
                                Box::new(crate::library::vectors::DatasetGroupOpen::new());
                            asm.add_node(&resolver_name, resolver_node, vec![src.clone()]);
                        }
                    }
                    #[cfg(feature = "vectordata")]
                    {
                        wire_refs[wire_idx] = WireRef::node(resolver_name);
                    }
                }
            }
            wire_idx += 1;
        }
        Ok(())
    }

    /// Try to resolve a function call as a Polydat module and inline it.
    ///
    /// Returns `Ok(true)` if the module was found and inlined, `Ok(false)`
    /// if no module was found, or `Err` on resolution/inlining failure.
    pub(super) fn try_inline_module(
        &mut self,
        asm: &mut PolydatAssembler,
        func_name: &str,
        caller_args: &[Arg],
        targets: &[String],
    ) -> Result<bool, String> {
        use crate::dsl::validate::{literal_type, types_compatible};

        // Resolve the module (load + parse + cache)
        let module = match self.resolve_module(func_name)? {
            Some(m) => m,
            None => return Ok(false),
        };

        let module_inputs = module.inputs.clone();
        let module_input_types = module.input_types.clone();
        let module_outputs = module.outputs.clone();
        let module_is_formal = module.is_formal;
        let module_stmts = module.statements.clone();

        // Module inlining is the *flatten* combinator (SRD 13b
        // §"Inline"): the module's `Statement`s splice into this
        // host's DAG and the boundary disappears. Pragmas declared
        // inside the module body therefore become *additive*
        // contributions to the *same* `PragmaSet` — they're not a
        // separate scope. The outer-wins / conflict-detection
        // semantics from SRD 15 §"Pragma Scope" fire at *scope
        // composition* boundaries (workload → phase → for_each),
        // which live in `nbrs-runtime`, not here.
        for stmt in &module_stmts {
            if let crate::dsl::ast::Statement::Pragma { name, span } = stmt {
                self.pragmas.entries.push(super::pragmas::Pragma {
                    name: name.clone(),
                    args: Vec::new(),
                    line: span.line,
                });
            }
        }

        // Strict mode: require all module arguments to be named
        if self.strict {
            for arg in caller_args {
                if matches!(arg, Arg::Positional(_)) {
                    return Err(format!(
                        "strict mode: module '{}' called with positional args — use named args (e.g., param_name: value)",
                        func_name
                    ));
                }
            }
        }

        // Build argument mapping: module input name → caller's wire/const
        // Named args map by name, positional args map by order of module_inputs
        let mut arg_map: std::collections::HashMap<String, Arg> = std::collections::HashMap::new();
        let mut positional_idx = 0;

        for arg in caller_args {
            match arg {
                Arg::Named(name, _) => {
                    arg_map.insert(name.clone(), arg.clone());
                }
                Arg::Positional(_) => {
                    if positional_idx < module_inputs.len() {
                        arg_map.insert(module_inputs[positional_idx].clone(), arg.clone());
                        positional_idx += 1;
                    }
                }
            }
        }

        // Strict mode: require all module inputs to be provided by the caller
        if self.strict {
            for input_name in &module_inputs {
                if !arg_map.contains_key(input_name) {
                    return Err(format!(
                        "strict mode: module '{}' input '{}' not provided — add '{}: <value>'",
                        func_name, input_name, input_name
                    ));
                }
            }
        }

        // --- Type validation for formal modules ---
        if module_is_formal {
            // Arity check
            let provided = arg_map.len();
            let expected = module_inputs.len();
            if provided != expected {
                return Err(format!(
                    "module '{}' expects {} arguments, got {}",
                    func_name, expected, provided
                ));
            }

            // Named argument validation: all names must match declared params
            for arg_name in arg_map.keys() {
                if !module_inputs.contains(arg_name) {
                    return Err(format!(
                        "module '{}' has no parameter named '{}' — available: {}",
                        func_name,
                        arg_name,
                        module_inputs.join(", ")
                    ));
                }
            }

            // Missing parameter check
            for input_name in &module_inputs {
                if !arg_map.contains_key(input_name) {
                    return Err(format!(
                        "module '{}' parameter '{}' not provided",
                        func_name, input_name
                    ));
                }
            }

            // Literal type checking against declared param types
            for (i, input_name) in module_inputs.iter().enumerate() {
                if let Some(Some(declared_type)) = module_input_types.get(i)
                    && let Some(arg) = arg_map.get(input_name)
                {
                    let expr = match arg {
                        Arg::Positional(e) | Arg::Named(_, e) => e,
                    };
                    if let Some(lit_type) = literal_type(expr)
                        && !types_compatible(&lit_type, declared_type)
                    {
                        return Err(format!(
                            "module '{}' parameter '{}' expects {}, got {} literal",
                            func_name, input_name, declared_type, lit_type
                        ));
                    }
                    // Wire arguments: type checked when the assembler validates wiring
                }
            }

            // Output arity check
            if targets.len() > module_outputs.len() {
                return Err(format!(
                    "module '{}' produces {} outputs: outputs, but {} targets requested",
                    func_name,
                    module_outputs.len(),
                    targets.len()
                ));
            }
        }

        // Generate a unique prefix for this module inlining
        let prefix = format!("__{func_name}_{}_", self.anon_counter);
        self.anon_counter += 1;

        // Inline each statement from the module, rewriting names
        for stmt in &module_stmts {
            match stmt {
                Statement::InputDecl(_) => {} // skip — kernel inputs handled by caller
                Statement::Binding(b) if matches!(b.value, Expr::For(_)) => {
                    // A producer inside the module (SRD 113 §3.1): bound
                    // under the module prefix as a `streamer` constant and
                    // recorded so this module's tiles can project over it.
                    let Expr::For(source) = &b.value else {
                        unreachable!()
                    };
                    let rewritten = self
                        .rewrite_for_source(
                            source,
                            &prefix,
                            &module_inputs,
                            &arg_map,
                            &module_stmts,
                        )
                        .map_err(|e| {
                            format!(
                                "producer '{}' inside module '{}': {e}",
                                b.targets.join(","),
                                func_name
                            )
                        })?;
                    let (comprehension, warnings) = super::traversal::resolve_source_with(
                        &rewritten,
                        &self.producers_seen,
                        self.validation_mode(),
                        &self.source_scope(),
                    )
                    .map_err(|e| {
                        format!(
                            "producer '{}' inside module '{}': {e}",
                            b.targets.join(","),
                            func_name
                        )
                    })?;
                    self.tile_events
                        .extend(super::traversal::warning_events(&rewritten, &warnings));
                    let name = format!("{prefix}{}", b.targets.join(","));
                    let value = crate::iteration::comprehension::StreamerValue::new(
                        rewritten.text.clone(),
                        comprehension.clone(),
                    );
                    let call = Expr::Call(CallExpr {
                        func: "streamer".into(),
                        args: vec![Arg::Positional(Expr::StringLit(value.to_json(), b.span))],
                        span: b.span,
                    });
                    self.compile_binding(asm, std::slice::from_ref(&name), &call)?;
                    asm.mark_const_output(&name);
                    asm.set_output_modifier(&name, crate::dsl::ast::BindingModifier::CONST);
                    self.producers_seen.push(super::traversal::Producer {
                        name,
                        span: b.span,
                        source_text: rewritten.text.clone(),
                        comprehension,
                    });
                }
                Statement::Binding(b) => {
                    let prefixed_targets: Vec<String> =
                        b.targets.iter().map(|t| format!("{prefix}{t}")).collect();
                    let rewritten =
                        self.rewrite_module_expr(&b.value, &prefix, &module_inputs, &arg_map);
                    self.compile_binding(asm, &prefixed_targets, &rewritten)?;
                }
                Statement::ModuleDef(_) | Statement::ExternPort(_) => {} // nested module defs not inlined
                Statement::Cursor(_) => {}
                Statement::Pragma { .. } => {} // pragmas don't inline; they're module-scoped
                Statement::For(f) => {
                    return Err(format!(
                        "`for {}` inside module '{}': {}",
                        f.source.text,
                        func_name,
                        "the `for` construct is parsed but not compiled yet (SRD 113 step 2); see docs/design/for_traversal.md"
                    ));
                }
                Statement::Tile(t) => {
                    // A tile inlines like a binding: its name takes the
                    // module prefix and every expression in its pieces
                    // is rewritten against the caller's arguments.
                    let mut tile = t.clone();
                    tile.name = format!("{prefix}{}", t.name);
                    tile.pieces = self
                        .rewrite_module_pieces(
                            &t.pieces,
                            &prefix,
                            &module_inputs,
                            &arg_map,
                            &module_stmts,
                        )
                        .map_err(|e| {
                            format!("tile '{}' inside module '{}': {e}", t.name, func_name)
                        })?;
                    self.compile_tile(asm, &tile)?;
                }
            }
        }

        // Wire module outputs to caller's targets via identity nodes.
        // This makes the target name available as both a wire source
        // (for downstream nodes) and an output.
        for (i, target) in targets.iter().enumerate() {
            let output_name = module_outputs
                .get(i)
                .cloned()
                .unwrap_or_else(|| func_name.to_string());
            let prefixed = format!("{prefix}{output_name}");
            // Create a passthrough node named after the target, wired
            // from the module's prefixed output. This makes `target`
            // available as a node name for downstream wiring.
            // Use PortPassthrough which accepts any port type (f64, u64, Str).
            let out_type = self.output_type_of(asm, &prefixed).ok_or_else(|| {
                format!(
                    "module '{func_name}': declared output '{output_name}' \
                     resolves to internal node '{prefixed}' but that node \
                     is not present in the assembler — module body did not \
                     produce the declared output."
                )
            })?;
            asm.add_node(
                target,
                Box::new(crate::library::identity::PortPassthrough::new(
                    target, out_type,
                )),
                vec![WireRef::node(&prefixed)],
            );
            self.all_names.push(target.clone());
        }

        Ok(true)
    }
}

/// Parsed `.polydat` files by path, with the modification time they
/// were read at. Shared across compilers in the process: a for-body or
/// probe compiler resolving the same library module reuses the parse,
/// and a file edited between compiles is re-read because its time moved.
type ParsedFiles = std::collections::HashMap<PathBuf, (std::time::SystemTime, Arc<PolydatFile>)>;
static PARSED_FILES: std::sync::OnceLock<std::sync::Mutex<ParsedFiles>> =
    std::sync::OnceLock::new();

/// The parsed form of a `.polydat` file, from the cache when its
/// modification time is unchanged. `None` when the file cannot be read;
/// `Some(Err)` when it does not parse.
fn parsed_file(path: &Path) -> Option<Result<Arc<PolydatFile>, String>> {
    let modified = std::fs::metadata(path).ok()?.modified().ok()?;
    let cache =
        PARSED_FILES.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Ok(map) = cache.lock()
        && let Some((seen, ast)) = map.get(path)
        && *seen == modified
    {
        return Some(Ok(ast.clone()));
    }
    let source = std::fs::read_to_string(path).ok()?;
    let parsed = lexer::lex(&source).and_then(parser::parse);
    match parsed {
        Ok(ast) => {
            let ast = Arc::new(ast);
            if let Ok(mut map) = cache.lock() {
                map.insert(path.to_path_buf(), (modified, ast.clone()));
            }
            Some(Ok(ast))
        }
        Err(e) => Some(Err(format!("module file '{}': {e}", path.display()))),
    }
}

/// The `.polydat` files directly in `dir`, in name order so resolution
/// is deterministic across platforms.
fn polydat_files_in(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("polydat"))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

/// `{name}` placeholders in comprehension text.
fn placeholder_names(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '{' {
            let start = i + 1;
            let mut j = start;
            while j < chars.len() && (chars[j].is_ascii_alphanumeric() || chars[j] == '_') {
                j += 1;
            }
            if j > start && chars.get(j) == Some(&'}') {
                out.push(chars[start..j].iter().collect());
                i = j;
            }
        }
        i += 1;
    }
    out
}

impl Compiler {
    /// Query the output type of a named node in the assembler.
    /// Returns `None` when the named node is absent or has no
    /// output ports; the caller surfaces the absence as a loud
    /// diagnostic.
    fn output_type_of(&self, asm: &PolydatAssembler, name: &str) -> Option<crate::ast::PortType> {
        asm.node_output_type(name)
    }

    /// Rewrite a tile's pieces from a module (SRD 114 §5.6): hole and
    /// branch expressions go through [`Self::rewrite_module_expr`];
    /// `{name}` placeholders in projection sources are renamed the same
    /// way when they name a module input bound to a caller's wire or a
    /// module-internal binding, and left alone otherwise, since they
    /// then name the comprehension's own elements.
    fn rewrite_module_pieces(
        &self,
        pieces: &[crate::dsl::ast::TilePiece],
        prefix: &str,
        module_inputs: &[String],
        arg_map: &std::collections::HashMap<String, Arg>,
        module_stmts: &[Statement],
    ) -> Result<Vec<crate::dsl::ast::TilePiece>, String> {
        use crate::dsl::ast::{ForSourceKind, TilePiece};
        let mut out = Vec::with_capacity(pieces.len());
        for piece in pieces {
            out.push(match piece {
                TilePiece::Static(s) => TilePiece::Static(s.clone()),
                TilePiece::Hole(h) => {
                    let mut h = h.clone();
                    h.expr = self.rewrite_module_expr(&h.expr, prefix, module_inputs, arg_map);
                    TilePiece::Hole(h)
                }
                TilePiece::Branch {
                    cond,
                    then,
                    otherwise,
                    span,
                } => TilePiece::Branch {
                    cond: self.rewrite_module_expr(cond, prefix, module_inputs, arg_map),
                    then: self.rewrite_module_pieces(
                        then,
                        prefix,
                        module_inputs,
                        arg_map,
                        module_stmts,
                    )?,
                    otherwise: match otherwise {
                        Some(o) => Some(self.rewrite_module_pieces(
                            o,
                            prefix,
                            module_inputs,
                            arg_map,
                            module_stmts,
                        )?),
                        None => None,
                    },
                    span: *span,
                },
                TilePiece::Projection {
                    source,
                    sep,
                    body,
                    span,
                } => {
                    let rewritten = self.rewrite_for_source(
                        source,
                        prefix,
                        module_inputs,
                        arg_map,
                        module_stmts,
                    )?;
                    // Elements shadow module names inside the body.
                    let elements = match &rewritten.kind {
                        ForSourceKind::Comprehension(c) => c.coordinate_names(),
                        _ => super::traversal::resolve_source(&rewritten, &self.producers_seen)
                            .map(|c| c.coordinate_names())
                            .unwrap_or_default(),
                    };
                    // Inside the body the elements shadow module names:
                    // they map to themselves so the rewriter leaves them.
                    let mut body_inputs = module_inputs.to_vec();
                    let mut body_args = arg_map.clone();
                    for e in &elements {
                        body_inputs.push(e.clone());
                        body_args.insert(e.clone(), Arg::Positional(Expr::Ident(e.clone(), *span)));
                    }
                    TilePiece::Projection {
                        source: rewritten,
                        sep: sep.clone(),
                        body: self.rewrite_module_pieces(
                            body,
                            prefix,
                            &body_inputs,
                            &body_args,
                            module_stmts,
                        )?,
                        span: *span,
                    }
                }
            });
        }
        Ok(out)
    }

    /// Rewrite a `for` source from a module against the caller: `{name}`
    /// placeholders that name a module input bound to a caller's wire or
    /// a module-internal binding are renamed (others name the
    /// comprehension's own elements and stay); generator expressions go
    /// through [`Self::rewrite_module_expr`]; a producer the module bound
    /// itself takes the module prefix.
    fn rewrite_for_source(
        &self,
        source: &crate::dsl::ast::ForSource,
        prefix: &str,
        module_inputs: &[String],
        arg_map: &std::collections::HashMap<String, Arg>,
        module_stmts: &[Statement],
    ) -> Result<crate::dsl::ast::ForSource, String> {
        use crate::dsl::ast::{ForSource, ForSourceKind};
        let internal: Vec<String> = module_stmts
            .iter()
            .flat_map(|s| match s {
                Statement::Binding(b) => b.targets.clone(),
                Statement::Tile(t) => vec![t.name.clone()],
                _ => Vec::new(),
            })
            .collect();
        let mut text = source.text.clone();
        for name in placeholder_names(&text) {
            let replacement = if module_inputs.contains(&name) {
                match arg_map.get(&name) {
                    Some(Arg::Positional(Expr::Ident(w, _)) | Arg::Named(_, Expr::Ident(w, _))) => {
                        Some(w.clone())
                    }
                    Some(_) => {
                        return Err(format!(
                            "`for {}` reads module input `{name}` through a placeholder, so the caller must pass a wire for it",
                            source.text
                        ));
                    }
                    None => None,
                }
            } else if internal.contains(&name) {
                Some(format!("{prefix}{name}"))
            } else {
                None
            };
            if let Some(r) = replacement {
                text = text.replace(&format!("{{{name}}}"), &format!("{{{r}}}"));
            }
        }
        // A producer the module bound takes the prefix, in the kind and
        // at the head of the text.
        let renamed = |n: &str| {
            if internal.contains(&n.to_string()) {
                format!("{prefix}{n}")
            } else {
                n.to_string()
            }
        };
        match &source.kind {
            ForSourceKind::Producer(n) => {
                let p = renamed(n);
                Ok(ForSource {
                    text: p.clone(),
                    kind: ForSourceKind::Producer(p),
                    span: source.span,
                })
            }
            ForSourceKind::Derived {
                base,
                filter,
                order,
            } => {
                let b = renamed(base);
                let text = match text.strip_prefix(base.as_str()) {
                    Some(rest) => format!("{b}{rest}"),
                    None => text,
                };
                Ok(ForSource {
                    text,
                    kind: ForSourceKind::Derived {
                        base: b,
                        filter: filter.clone(),
                        order: order.clone(),
                    },
                    span: source.span,
                })
            }
            ForSourceKind::Comprehension(_) => {
                let mut rewritten = super::parser::for_source_from_text(&text, source.span, true)?;
                if let ForSourceKind::Comprehension(c) = &rewritten.kind {
                    let c = self.rewrite_generators(c, prefix, module_inputs, arg_map)?;
                    rewritten.kind = ForSourceKind::Comprehension(c);
                }
                Ok(rewritten)
            }
        }
    }

    /// Rewrite the generator-call expressions of a comprehension against
    /// the caller's arguments, as [`Self::rewrite_module_expr`] does for
    /// a binding.
    fn rewrite_generators(
        &self,
        c: &crate::iteration::comprehension::Comprehension,
        prefix: &str,
        module_inputs: &[String],
        arg_map: &std::collections::HashMap<String, Arg>,
    ) -> Result<crate::iteration::comprehension::Comprehension, String> {
        use crate::iteration::comprehension::Comprehension as K;
        use crate::iteration::comprehension::source::Source;
        Ok(match c {
            K::Clause {
                name,
                source:
                    Source::Generator {
                        expr,
                        cardinality_hint,
                    },
            } => {
                let parsed = super::tile::parse_hole_expr(expr).map_err(|e| {
                    format!("generator `{expr}` in projection element '{name}': {e}")
                })?;
                let rewritten = self.rewrite_module_expr(&parsed, prefix, module_inputs, arg_map);
                K::Clause {
                    name: name.clone(),
                    source: Source::Generator {
                        expr: super::pprint::pp_expr(&rewritten),
                        cardinality_hint: *cardinality_hint,
                    },
                }
            }
            K::Clause { .. } => c.clone(),
            K::Cartesian { children } => K::Cartesian {
                children: children
                    .iter()
                    .map(|ch| self.rewrite_generators(ch, prefix, module_inputs, arg_map))
                    .collect::<Result<_, _>>()?,
            },
            K::Zip { children, mode } => K::Zip {
                children: children
                    .iter()
                    .map(|ch| self.rewrite_generators(ch, prefix, module_inputs, arg_map))
                    .collect::<Result<_, _>>()?,
                mode: *mode,
            },
            K::Union { children } => K::Union {
                children: children
                    .iter()
                    .map(|ch| self.rewrite_generators(ch, prefix, module_inputs, arg_map))
                    .collect::<Result<_, _>>()?,
            },
            K::Filter { child, predicate } => K::Filter {
                child: Box::new(self.rewrite_generators(child, prefix, module_inputs, arg_map)?),
                predicate: predicate.clone(),
            },
            K::Order {
                child,
                strategy,
                truncation,
                seed,
            } => K::Order {
                child: Box::new(self.rewrite_generators(child, prefix, module_inputs, arg_map)?),
                strategy: *strategy,
                truncation: *truncation,
                seed: *seed,
            },
        })
    }

    /// Rewrite an expression from a module, substituting input references
    /// with the caller's arguments and prefixing internal names.
    pub(super) fn rewrite_module_expr(
        &self,
        expr: &Expr,
        prefix: &str,
        module_inputs: &[String],
        arg_map: &std::collections::HashMap<String, Arg>,
    ) -> Expr {
        match expr {
            Expr::Ident(name, span) => {
                if module_inputs.contains(name) {
                    // Replace with caller's argument
                    if let Some(arg) = arg_map.get(name) {
                        match arg {
                            Arg::Positional(e) | Arg::Named(_, e) => e.clone(),
                        }
                    } else {
                        // Unresolved module input — keep as-is (becomes
                        // a coordinate reference in the caller)
                        Expr::Ident(name.clone(), *span)
                    }
                } else {
                    // Internal name — prefix it
                    Expr::Ident(format!("{prefix}{name}"), *span)
                }
            }
            Expr::Call(call) => {
                let rewritten_args: Vec<Arg> = call
                    .args
                    .iter()
                    .map(|arg| match arg {
                        Arg::Positional(e) => Arg::Positional(self.rewrite_module_expr(
                            e,
                            prefix,
                            module_inputs,
                            arg_map,
                        )),
                        Arg::Named(n, e) => Arg::Named(
                            n.clone(),
                            self.rewrite_module_expr(e, prefix, module_inputs, arg_map),
                        ),
                    })
                    .collect();
                Expr::Call(CallExpr {
                    func: call.func.clone(),
                    args: rewritten_args,
                    span: call.span,
                })
            }
            Expr::ArrayLit(elems, span) => Expr::ArrayLit(
                elems
                    .iter()
                    .map(|e| self.rewrite_module_expr(e, prefix, module_inputs, arg_map))
                    .collect(),
                *span,
            ),
            Expr::BinOp(lhs, op, rhs) => Expr::BinOp(
                Box::new(self.rewrite_module_expr(lhs, prefix, module_inputs, arg_map)),
                *op,
                Box::new(self.rewrite_module_expr(rhs, prefix, module_inputs, arg_map)),
            ),
            Expr::UnaryNeg(inner, span) => Expr::UnaryNeg(
                Box::new(self.rewrite_module_expr(inner, prefix, module_inputs, arg_map)),
                *span,
            ),
            Expr::UnaryBitNot(inner, span) => Expr::UnaryBitNot(
                Box::new(self.rewrite_module_expr(inner, prefix, module_inputs, arg_map)),
                *span,
            ),
            other => other.clone(),
        }
    }

    /// Resolve a module by name.
    ///
    /// Resolution order:
    /// 1. Cache (already resolved)
    /// 2. `<name>.polydat` in `source_dir` (workload-local)
    /// 3. Any `.polydat` in `source_dir` containing a matching binding
    /// 4. Same two searches for each library path
    ///    (`CompileOptions::lib_paths`, the binary's `--lib`)
    /// 5. Embedded stdlib
    pub(super) fn resolve_module(&mut self, name: &str) -> Result<Option<&ResolvedModule>, String> {
        if self.module_cache.contains_key(name) {
            return Ok(self.module_cache.get(name));
        }

        // Strategies 1-3: the program's directory, then each library
        // directory. Files are read and parsed once per process and
        // reused while unmodified, so a cold compile of a large
        // directory pays for each file once, not once per unknown name.
        let mut dirs: Vec<PathBuf> = Vec::new();
        if let Some(source_dir) = &self.source_dir {
            dirs.push(source_dir.clone());
        }
        dirs.extend(self.polydat_lib_paths.iter().cloned());
        for dir in &dirs {
            // <name>.polydat first: the file named for the module.
            let named = dir.join(format!("{name}.polydat"));
            if named.exists() {
                let ast = parsed_file(&named)
                    .ok_or_else(|| format!("failed to read module '{}'", named.display()))??;
                let resolved = Self::module_from_ast(&ast, name)?;
                self.module_cache.insert(name.to_string(), resolved);
                return Ok(self.module_cache.get(name));
            }
            // Then any .polydat in the directory that exports the name.
            for path in polydat_files_in(dir) {
                let Some(Ok(ast)) = parsed_file(&path) else {
                    continue;
                };
                if let Ok(resolved) = Self::module_from_ast(&ast, name) {
                    self.module_cache.insert(name.to_string(), resolved);
                    return Ok(self.module_cache.get(name));
                }
            }
        }

        // Strategy 4: embedded stdlib
        if let Some(resolved) = self.resolve_stdlib(name)? {
            self.module_cache.insert(name.to_string(), resolved);
            return Ok(self.module_cache.get(name));
        }

        Ok(None)
    }

    /// Search the embedded stdlib for a module.
    fn resolve_stdlib(&self, name: &str) -> Result<Option<ResolvedModule>, String> {
        for (_filename, source) in STDLIB_MODULES {
            if let Ok(resolved) = Self::parse_module(source, name) {
                return Ok(Some(resolved));
            }
        }
        Ok(None)
    }

    /// Insert the file's formal module definitions into the cache.
    ///
    /// Make every formal module defined in `file` resolvable by name in
    /// this compile, ahead of the filesystem and the embedded library.
    /// A definition in the program shadows a library node of the same
    /// name, so an author's `pick(...)` is the author's.
    pub(super) fn register_local_modules(&mut self, file: &PolydatFile) {
        for stmt in &file.statements {
            if let Statement::ModuleDef(mdef) = stmt {
                self.module_cache
                    .insert(mdef.name.clone(), Self::resolved_from_def(mdef));
            }
        }
    }

    fn resolved_from_def(mdef: &crate::dsl::ast::ModuleDef) -> ResolvedModule {
        ResolvedModule {
            inputs: mdef.params.iter().map(|p| p.name.clone()).collect(),
            input_types: mdef.params.iter().map(|p| Some(p.typ.clone())).collect(),
            outputs: mdef.outputs.iter().map(|o| o.name.clone()).collect(),
            output_types: mdef.outputs.iter().map(|o| Some(o.typ.clone())).collect(),
            is_formal: true,
            statements: mdef.body.clone(),
        }
    }

    /// Whether `name` is a module this compile already knows: defined in
    /// the program or resolved earlier. Cheap, so callers may consult it
    /// before the function registry.
    pub(super) fn has_known_module(&self, name: &str) -> bool {
        self.module_cache.contains_key(name)
    }

    pub(super) fn parse_module(source: &str, target_name: &str) -> Result<ResolvedModule, String> {
        let tokens = lexer::lex(source)?;
        let ast = parser::parse(tokens)?;
        Self::module_from_ast(&ast, target_name)
    }

    /// Resolve `target_name` from an already-parsed file: a formal
    /// module definition of that name, else the subgraph that produces
    /// a binding of that name.
    fn module_from_ast(ast: &PolydatFile, target_name: &str) -> Result<ResolvedModule, String> {
        // Strategy 1: look for a formal ModuleDef with matching name
        for stmt in &ast.statements {
            if let Statement::ModuleDef(mdef) = stmt
                && mdef.name == target_name
            {
                let inputs: Vec<String> = mdef.params.iter().map(|p| p.name.clone()).collect();
                let input_types: Vec<Option<String>> =
                    mdef.params.iter().map(|p| Some(p.typ.clone())).collect();
                let outputs: Vec<String> = mdef.outputs.iter().map(|o| o.name.clone()).collect();
                let output_types: Vec<Option<String>> =
                    mdef.outputs.iter().map(|o| Some(o.typ.clone())).collect();
                return Ok(ResolvedModule {
                    inputs,
                    input_types,
                    outputs,
                    output_types,
                    is_formal: true,
                    statements: mdef.body.clone(),
                });
            }
        }

        // Strategy 2: subgraph extraction by binding name
        // Build a map: binding name → (statement index, references)
        let mut name_to_idx: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut stmt_refs: Vec<HashSet<String>> = Vec::new();

        for (i, stmt) in ast.statements.iter().enumerate() {
            let (names, expr) = match stmt {
                Statement::InputDecl(_)
                | Statement::ModuleDef(_)
                | Statement::ExternPort(_)
                | Statement::Cursor(_)
                | Statement::Pragma { .. }
                | Statement::For(_)
                | Statement::Tile(_) => {
                    stmt_refs.push(HashSet::new());
                    continue;
                }
                Statement::Binding(b) => (b.targets.clone(), &b.value),
            };
            for name in &names {
                name_to_idx.insert(name.clone(), i);
            }
            let mut refs = HashSet::new();
            collect_references(expr, &mut refs);
            stmt_refs.push(refs);
        }

        // Check that the target binding exists in this file
        let target_idx = match name_to_idx.get(target_name) {
            Some(&idx) => idx,
            None => return Err(format!("no binding named '{target_name}' in module")),
        };

        // Trace backward from target to find all needed statements
        let mut needed: HashSet<usize> = HashSet::new();
        let mut worklist = vec![target_idx];
        while let Some(idx) = worklist.pop() {
            if !needed.insert(idx) {
                continue;
            }
            if idx < stmt_refs.len() {
                for ref_name in &stmt_refs[idx] {
                    if let Some(&dep_idx) = name_to_idx.get(ref_name) {
                        worklist.push(dep_idx);
                    }
                }
            }
        }

        // Extract only the needed statements, preserving order
        let extracted: Vec<Statement> = ast
            .statements
            .iter()
            .enumerate()
            .filter(|(i, _)| needed.contains(i))
            .map(|(_, s)| s.clone())
            .collect();

        if extracted.is_empty() {
            return Err(format!("empty subgraph for '{target_name}'"));
        }

        // Infer inputs: referenced names not defined within the subgraph
        let mut defined: HashSet<String> = HashSet::new();
        let mut referenced: HashSet<String> = HashSet::new();
        for stmt in &extracted {
            match stmt {
                Statement::InputDecl(d) => {
                    defined.insert(d.name.clone());
                }
                Statement::Binding(b) => {
                    for t in &b.targets {
                        defined.insert(t.clone());
                    }
                }
                Statement::ModuleDef(_) | Statement::ExternPort(_) => {}
                Statement::Cursor(_) => {}
                Statement::Pragma { .. } => {}
                Statement::For(_) => {}
                Statement::Tile(_) => {}
            }
        }
        for stmt in &extracted {
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

        let mut inputs: Vec<String> = referenced
            .into_iter()
            .filter(|name| !defined.contains(name))
            .collect();
        inputs.sort();

        Ok(ResolvedModule {
            input_types: inputs.iter().map(|_| None).collect(),
            inputs,
            outputs: vec![target_name.to_string()],
            output_types: vec![None],
            is_formal: false,
            statements: extracted,
        })
    }
}
