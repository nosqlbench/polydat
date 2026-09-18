// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Abstract syntax tree for the Polydat DSL.

use crate::lexer::Span;

/// A complete `.polydat` file.
#[derive(Debug, Clone)]
pub struct PolydatFile {
    /// The statements, in document order.
    pub statements: Vec<Statement>,
}

/// A top-level statement.
#[derive(Debug, Clone)]
pub enum Statement {
    /// `input name[: type]` — declares one per-cycle kernel input slot.
    /// The name becomes both an input slot (settable via `set_input`)
    /// and a passthrough output (readable via `get_constant`/`pull`).
    ///
    /// Surface forms (parser desugars the tuple into N `InputDecl`s,
    /// mirroring the module-signature param-list shape from
    /// a host-provided cycle module):
    /// ```text
    /// input cycle: u64
    /// input (cycle: u64, q: f64)
    /// ```
    InputDecl(InputDecl),
    /// A name-to-expression binding. The modifier on the
    /// binding determines its lifecycle:
    ///
    /// - **no modifier** — per-cycle: re-evaluated every cycle.
    /// - **`const`** — effectively-const for the scope's
    ///   lifetime: materialized at the earliest opportunity
    ///   (compile-time fold if the RHS is fold-eligible,
    ///   otherwise scope-init pull after materialize-wiring
    ///   has populated extern slots). Authors don't need to
    ///   know which path the runtime takes — the contract is
    ///   "fixed once, then immutable."
    /// - **`shared`** — cell-backed, mutable across kernel
    ///   instances in the same lineage. See
    ///   `crates/polydat/docs/design/scope_model.md` §6
    ///   "Shared mutable bindings".
    /// - **`volatile`** — per-cycle, excluded from
    ///   `hash_const`.
    ///
    /// Surface forms:
    /// ```text
    /// x := mul(cycle, 2)                  // per-cycle
    /// const pi := 3.14                    // const, folds at compile
    /// const ann_opts := str_concat(...)   // const, materializes at scope-init
    /// shared budget := 100                // shared cell
    /// (a, b) := split_pair(...)           // tuple destructuring
    /// ```
    Binding(Binding),
    /// `name(param: type, ...) -> (output: type, ...) := { body }`
    ModuleDef(ModuleDef),
    /// `extern name: type = default`
    ExternPort(ExternPort),
    /// `cursor name = Cursor()` or `cursor name = constructor_expr`
    Cursor(CursorDecl),
    /// `pragma <name>` — a module-level directive opting into a
    /// compile-time graph transform (SRD 15 §"Module-Level
    /// Pragmas"). First-class grammar, distinct from line
    /// comments. Recognised pragmas trigger
    /// `CompileEvent::PragmaAcknowledged`; unknown names trigger
    /// `CompileEvent::UnknownPragma` and are otherwise ignored
    /// (forward-compatible).
    Pragma {
        /// The pragma's name, after the `pragma` keyword.
        name: String,
        /// Where the pragma appears.
        span: Span,
    },
    /// `for <source> { body }` — a traversal scope (SRD 113 §3.2).
    /// One child scope activates per tuple of the source; the
    /// comprehension's element names are wires inside the body.
    For(ForStmt),
    /// `tile name : encoding (options) := body` — a compiled variate
    /// template (SRD 114 §2).
    Tile(TileDef),
}

/// Per-tile template options (SRD 114 §2.3): the hole delimiters, the
/// directive sigil, and strictness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileOptions {
    /// The text that opens a hole, `${` by default.
    pub open: String,
    /// The text that closes a hole, `}` by default.
    pub close: String,
    /// The directive sigil, `@` by default.
    pub sigil: String,
    /// Whether an unknown hole or directive is an error rather than text.
    pub strict: bool,
    /// The body begins inside a JSON string literal (`instring`): holes
    /// encode as escaped text from the first byte. The compiler sets
    /// this on the tile it makes for a projection nested in a string
    /// position; authors rarely need it.
    pub in_string: bool,
}

impl Default for TileOptions {
    fn default() -> Self {
        Self {
            open: "${".into(),
            close: "}".into(),
            sigil: "@".into(),
            strict: false,
            in_string: false,
        }
    }
}

/// How a tile body was written, so the printer can reproduce it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileBodyKind {
    /// A brace- or bracket-balanced block, as `json` templates are written.
    Block,
    /// Text between `<<<` and `>>>`.
    Heredoc,
    /// An ordinary string literal.
    Literal,
}

/// A tile definition: its header, its raw body, and the parsed template.
#[derive(Debug, Clone)]
pub struct TileDef {
    /// The tile's name: the wire it binds.
    pub name: String,
    /// The declared encoding, such as `json` or `csv`, if any.
    pub encoding: Option<String>,
    /// The delimiter and strictness options.
    pub options: TileOptions,
    /// How the body was written: heredoc, block, or string literal.
    /// Presentation, not content: it decides how the body is delimited
    /// when the tile is printed.
    pub body_kind: TileBodyKind,
    /// The template: static runs, holes, projections, and branches.
    /// This is the tile's body — the only representation of it. Text
    /// admitted from source is parsed into pieces under the options in
    /// force at that moment and is not kept beside them.
    pub pieces: Vec<TilePiece>,
    /// Where the definition appears.
    pub span: Span,
}

impl TileDef {
    /// A tile whose body is admitted as **text**, parsed into pieces
    /// under the tile's own options.
    ///
    /// The text is consumed by this call: what the tile holds
    /// afterwards is the template it parsed to. Reading the body back
    /// renders from those pieces ([`Self::body_text`]), so the body a
    /// tile prints is always the body it renders (polytile.md §3).
    pub fn from_body(
        name: impl Into<String>,
        encoding: Option<String>,
        options: TileOptions,
        body_kind: TileBodyKind,
        body: impl Into<String>,
        span: Span,
    ) -> Result<Self, String> {
        let pieces = crate::tile::parse_template(&body.into(), &options, span)?;
        Ok(Self::from_pieces(
            name, encoding, options, body_kind, pieces, span,
        ))
    }

    /// A tile whose body is admitted as **pieces**, built directly.
    ///
    /// The counterpart of [`Self::from_body`], and the same tile: a
    /// template composed programmatically and one parsed from text that
    /// renders the same pieces are equal as tiles, because the pieces
    /// are what a tile is. Pieces from either source compose by
    /// concatenation, so a template may be assembled from parsed
    /// fragments, hand-built pieces, or any mixture, in any order.
    pub fn from_pieces(
        name: impl Into<String>,
        encoding: Option<String>,
        options: TileOptions,
        body_kind: TileBodyKind,
        pieces: Vec<TilePiece>,
        span: Span,
    ) -> Self {
        TileDef {
            name: name.into(),
            encoding,
            options,
            body_kind,
            pieces,
            span,
        }
    }

    /// The body as template text, rendered from the pieces under this
    /// tile's own delimiters and sigil.
    ///
    /// Canonical rather than verbatim: a tile parsed from source does
    /// not keep the author's spacing, the same way the projector does
    /// not keep the spacing around a binary operator. Re-parsing this
    /// text under the same options yields the same pieces.
    pub fn body_text(&self) -> String {
        crate::tile::render_template(&self.pieces, &self.options)
    }
}

/// One element of a parsed template.
#[derive(Debug, Clone)]
pub enum TilePiece {
    /// Bytes copied as written.
    Static(String),
    /// `${expr : type | format !}`.
    Hole(TileHole),
    /// `@for <source> [sep "..."] { body }`.
    Projection {
        /// What the projection iterates.
        source: ForSource,
        /// The separator emitted between tuples, if any.
        sep: Option<String>,
        /// The template rendered once per tuple.
        body: Vec<TilePiece>,
        /// Where the directive appears.
        span: Span,
    },
    /// `@if cond { body } [@else { body }]`.
    Branch {
        /// The condition, a boolean expression.
        cond: Expr,
        /// The template rendered when the condition holds.
        then: Vec<TilePiece>,
        /// The template rendered otherwise, if an `@else` was written.
        otherwise: Option<Vec<TilePiece>>,
        /// Where the directive appears.
        span: Span,
    },
}

/// A hole: an expression with an optional declared type, format spec,
/// and raw flag.
#[derive(Debug, Clone)]
pub struct TileHole {
    /// The expression the hole evaluates.
    pub expr: Expr,
    /// The declared type after the colon, if any.
    pub decl_type: Option<String>,
    /// The format after the bar, if any.
    pub format: Option<String>,
    /// Whether the value is emitted without the encoding's escaping.
    pub raw: bool,
    /// Where the hole appears.
    pub span: Span,
}

impl TileHole {
    /// The hole's body as template text, between the delimiters: the
    /// expression, then the declared type, the format, and the raw
    /// marker where each is present.
    ///
    /// This is the hole's only textual form. A hole parsed from source
    /// does not keep what was written, because the parts are what the
    /// renderer and the compiler read, and a second copy of the same
    /// thing is a second thing to disagree. Spacing the author used
    /// inside the delimiters is not reproduced, exactly as the
    /// projector does not reproduce the spacing around a binary
    /// operator.
    pub fn to_text(&self) -> String {
        let mut out = crate::pprint::pp_expr(&self.expr);
        if let Some(ty) = &self.decl_type {
            out.push_str(": ");
            out.push_str(ty);
        }
        if let Some(fmt) = &self.format {
            out.push_str(" | ");
            out.push_str(fmt);
        }
        if self.raw {
            out.push('!');
        }
        out
    }
}

/// What a `for` iterates: inline comprehension text, or the name of a
/// bound producer wire.
#[derive(Debug, Clone)]
pub struct ForSource {
    /// The raw text after `for`, preserved for diagnostics and for
    /// pretty-printing round trips.
    pub text: String,
    /// What the text denotes once parsed.
    pub kind: ForSourceKind,
    /// Where the source appears.
    pub span: Span,
}

#[derive(Debug, Clone)]
/// The parsed form of a `for` source.
pub enum ForSourceKind {
    /// A bare identifier naming a `Streamer` wire bound by a `for`
    /// expression elsewhere in scope.
    Producer(String),
    /// Comprehension text, parsed to the algebra AST.
    Comprehension(crate::comprehension::Comprehension),
    /// A derivation of a bound producer: `base where <pred>`,
    /// `base order <spec>`, or both (SRD 113 §3.1). Resolved against
    /// the producer at compile time.
    Derived {
        /// The producer wire the derivation starts from.
        base: String,
        /// The `where` predicate text, if any.
        filter: Option<String>,
        /// The `order` specification text, if any.
        order: Option<String>,
    },
}

impl ForSource {
    /// A traversal source over a comprehension, its text the tree's
    /// canonical text
    /// ([`Comprehension::to_text`](crate::comprehension::Comprehension::to_text)).
    /// `None` when the
    /// tree is outside the text grammar, so a source never carries a
    /// text that reads back as a different comprehension.
    pub fn comprehension(tree: crate::comprehension::Comprehension, span: Span) -> Option<Self> {
        let text = tree.to_text()?;
        Some(ForSource {
            text,
            kind: ForSourceKind::Comprehension(tree),
            span,
        })
    }

    /// A traversal source naming a producer bound in the same scope.
    pub fn producer(name: impl Into<String>, span: Span) -> Self {
        let name = name.into();
        ForSource {
            text: name.clone(),
            kind: ForSourceKind::Producer(name),
            span,
        }
    }

    /// A traversal source deriving from a bound producer: the base
    /// with an optional filter and order, written as the text writes
    /// them.
    pub fn derived(
        base: impl Into<String>,
        filter: Option<String>,
        order: Option<String>,
        span: Span,
    ) -> Self {
        let base = base.into();
        let mut text = base.clone();
        if let Some(predicate) = &filter {
            text.push_str(&format!(" where {predicate}"));
        }
        if let Some(spec) = &order {
            text.push_str(&format!(" order {spec}"));
        }
        ForSource {
            text,
            kind: ForSourceKind::Derived {
                base,
                filter,
                order,
            },
            span,
        }
    }

    /// The element names the source dispenses, when known statically.
    /// A producer reference or derivation resolves its names at compile
    /// time.
    pub fn element_names(&self) -> Vec<String> {
        match &self.kind {
            ForSourceKind::Producer(_) | ForSourceKind::Derived { .. } => Vec::new(),
            ForSourceKind::Comprehension(c) => c.coordinate_names(),
        }
    }
}

/// A traversal statement: `for <source> { statements }`.
#[derive(Debug, Clone)]
pub struct ForStmt {
    /// What the traversal iterates.
    pub source: ForSource,
    /// The body: one child scope per tuple.
    pub body: Vec<Statement>,
    /// Where the statement appears.
    pub span: Span,
}

/// An external input port declaration.
///
/// Ports persist across `set_inputs()` calls within a stanza.
/// Written by capture extraction, read by Polydat nodes.
///
/// ```text
/// extern balance: f64 = 0.0
/// extern session_id: u64 = 0
/// ```
#[derive(Debug, Clone)]
pub struct ExternPort {
    /// The port's name.
    pub name: String,
    /// The declared type keyword.
    pub typ: String,
    /// The default value, if one was written; without one the port is `None` until set.
    pub default: Option<Expr>,
    /// Where the declaration appears.
    pub span: Span,
}

/// One per-cycle kernel input slot.
///
/// Declared by `input <name>[: <type>]` (single) or
/// `input (<name>[: <type>], ...)` (tuple, sugar for N decls).
/// The name participates in the kernel's input-port wiring just
/// like `extern` participates in its port set, but inputs are
/// driven by the runtime cycle pump (cursors, captures, etc.)
/// rather than by external port writes.
///
/// `ty` is `None` when the author omitted the annotation; typed
/// downstream by inference. Authors are encouraged to declare
/// the type for clarity and editor support.
#[derive(Debug, Clone)]
pub struct InputDecl {
    /// The input's name.
    pub name: String,
    /// The declared type keyword, if one was written.
    pub ty: Option<String>,
    /// Where the declaration appears.
    pub span: Span,
}

/// One wire-coloring keyword. The single enum that names every
/// modifier the grammar recognises before a binding name.
/// Future modifiers are new variants here.
///
/// Each variant maps to a token the lexer emits and a parser
/// branch in `parse_modified_binding`. A binding can carry zero
/// or more of these, stored as a [`BindingModifier`] set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WireModifier {
    /// `const` — effectively-const for the scope's lifetime.
    /// Materialized at the earliest opportunity: compile-time
    /// const-fold when the RHS is fold-eligible, otherwise the
    /// scope-init pull pass after materialize-wiring has
    /// populated extern slots. The runtime contract is "fixed
    /// once per scope activation, then immutable for the rest of
    /// the scope's lifetime." Replaces the former `final` /
    /// `init` distinction — the two were redundant axes of the
    /// same lifecycle, and the surface now collapses to one
    /// keyword whose materialization timing is an internal
    /// optimization.
    Const,
    /// `shared` — mutable cell visible across kernel instances.
    /// The runtime propagates iteration N's end state into
    /// iteration N+1's start state.
    Shared,
    /// `volatile` — wire's value is excluded from `hash_const`
    /// (the const-folded identity hash). Authors mark wires
    /// whose value should NOT contribute to resume-identity
    /// even when the source's structural detection would
    /// otherwise allow folding.
    Volatile,
}

/// Set of wire modifiers carried by one binding declaration.
/// Stored as a bitset under the hood; consumers use
/// [`Self::has`] to test for individual modifiers and
/// [`Self::insert`] / `Self::from_iter` to build instances.
///
/// **Validity:** the combination `const` + `volatile` is
/// rejected at parse time as contradictory (`Self::from_iter`
/// is the validating builder). All other combinations are
/// representable.
///
/// Lives on every [`Statement::Binding`] — the modifier set
/// determines the binding's lifecycle. Other statement kinds
/// (`ExternPort`, `InputDecl`, etc.) don't carry modifiers
/// because their semantics are fixed by their statement form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BindingModifier {
    bits: u8,
}

impl BindingModifier {
    /// All-modifiers-off; the default state of an unannotated
    /// binding (per-cycle).
    pub const NONE: Self = Self { bits: 0 };

    /// Single-modifier convenience constants. Tests reach for
    /// these to express their intent compactly.
    pub const CONST: Self = Self {
        bits: Self::bit(WireModifier::Const),
    };
    /// The `shared` modifier alone.
    pub const SHARED: Self = Self {
        bits: Self::bit(WireModifier::Shared),
    };
    /// The `volatile` modifier alone.
    pub const VOLATILE: Self = Self {
        bits: Self::bit(WireModifier::Volatile),
    };

    /// `true` iff `m` is set.
    pub const fn has(&self, m: WireModifier) -> bool {
        self.bits & Self::bit(m) != 0
    }

    /// `true` iff at least one modifier is set.
    pub const fn has_any(&self) -> bool {
        self.bits != 0
    }

    /// Add `m` to the set.
    pub fn insert(&mut self, m: WireModifier) {
        self.bits |= Self::bit(m);
    }

    /// Build a modifier set from an iterator of variants. The
    /// parser uses this after collecting tokens. Rejects the
    /// contradictory `const` + `volatile` combo with a clear
    /// error.
    pub fn try_from_iter<I: IntoIterator<Item = WireModifier>>(
        items: I,
    ) -> Result<Self, &'static str> {
        let mut out = Self::NONE;
        for m in items {
            out.insert(m);
        }
        if out.has(WireModifier::Const) && out.has(WireModifier::Volatile) {
            return Err(
                "modifier conflict: `const` and `volatile` are contradictory \
                 — `const` materializes the value once and freezes it; \
                 `volatile` excludes the wire from const-fold and signals \
                 per-cycle variability. Drop one.",
            );
        }
        Ok(out)
    }

    /// Iterate the modifiers in the set, in fixed declaration
    /// order (`Const`, `Shared`, `Volatile`). Used for
    /// re-emission and stable hash output.
    pub fn iter(&self) -> impl Iterator<Item = WireModifier> + '_ {
        const ORDER: &[WireModifier] = &[
            WireModifier::Const,
            WireModifier::Shared,
            WireModifier::Volatile,
        ];
        ORDER.iter().copied().filter(move |m| self.has(*m))
    }

    /// Direct field-style accessors retained for sites that
    /// pattern-match on individual flags. Mechanically derive
    /// from `has(...)` so adding a new modifier is one variant
    /// + one bit assignment + (optionally) one accessor.
    #[inline]
    pub const fn is_const(&self) -> bool {
        self.has(WireModifier::Const)
    }
    #[inline]
    /// `true` iff `shared` is set.
    pub const fn is_shared(&self) -> bool {
        self.has(WireModifier::Shared)
    }
    #[inline]
    /// `true` iff `volatile` is set.
    pub const fn is_volatile(&self) -> bool {
        self.has(WireModifier::Volatile)
    }

    /// Compile-time bit index for a modifier.
    const fn bit(m: WireModifier) -> u8 {
        match m {
            WireModifier::Const => 1 << 0,
            WireModifier::Shared => 1 << 1,
            WireModifier::Volatile => 1 << 2,
        }
    }
}

/// A name-to-expression binding (per-cycle by default;
/// `const`/`shared`/`volatile` modifier changes the
/// lifecycle). Replaces the former `CycleBinding` and
/// `InitBinding` AST variants — the surface unified to one
/// shape `name := expr` (or `(a, b, c) := expr` for tuple
/// destructuring), with the modifier driving runtime
/// lifecycle.
#[derive(Debug, Clone)]
pub struct Binding {
    /// The bound names: one, or several for tuple destructuring.
    pub targets: Vec<String>,
    /// The right-hand side.
    pub value: Expr,
    /// The wire-coloring keywords written before the names.
    pub modifier: BindingModifier,
    /// Optional explicit type annotation — `shared name: f64 := 1`.
    /// Only meaningful on `shared` bindings (scope_model.md §"Type
    /// stability"): it pins the CELL's PortType for life, winning over
    /// literal inference (so `1` vs `1.0` stops being load-bearing).
    /// The parser rejects annotations on non-shared bindings.
    pub type_annotation: Option<String>,
    /// Where the binding appears.
    pub span: Span,
}

/// A cursor declaration: `cursor name = Cursor() [over partition_source]`
///
/// Declares a named positional cursor. The cursor's extent is
/// discovered at init time by interrogating its downstream consumers
/// for cardinality. The runtime advances the cursor to drive
/// phase iteration.
///
/// The optional `over` clause (SRD 71) names a partition source
/// — an in-scope wire that resolves to a `Partition` or a
/// `PartitionList`. When bound, the cursor's effective extent
/// narrows to the named partition's `[start_ord, end_ord)`
/// range; without it, the cursor uses its full declared extent.
#[derive(Debug, Clone)]
pub struct CursorDecl {
    /// The cursor's name.
    pub name: String,
    /// The constructor call, such as `range(0, 100)`.
    pub constructor: Expr,
    /// SRD 71 `over <expr>` clause. The expression is parsed
    /// the same way as any other Polydat expression so authors can
    /// name a workload parameter's `.partitions` projection
    /// (e.g. `cursor.partitions`), an iter-var bound by an
    /// enclosing `for:`, or a sibling cursor's `.cursor`
    /// projection (`q1.cursor`). `None` means no narrowing —
    /// the cursor uses its full declared extent.
    pub over: Option<Expr>,
    /// Where the declaration appears.
    pub span: Span,
}

/// An expression (right-hand side of a binding).
#[derive(Debug, Clone)]
pub enum Expr {
    /// A bare identifier referencing a wire or init binding: `cycle`, `lut`
    Ident(String, Span),
    /// An integer literal: `1000`
    IntLit(u64, Span),
    /// A float literal: `72.0`
    FloatLit(f64, Span),
    /// A string literal (may contain `{name}` interpolation): `"hello {name}"`
    StringLit(String, Span),
    /// An array literal: `[60.0, 20.0, 15.0]`
    ArrayLit(Vec<Expr>, Span),
    /// A function call: `hash(cycle)`, `dist_normal(mean: 72.0, stddev: 5.0)`
    Call(CallExpr),
    /// A binary arithmetic operation: `a + b`, `x * 0.25`.
    /// Desugared by the compiler into the equivalent function call.
    BinOp(Box<Expr>, BinOpKind, Box<Expr>),
    /// Unary negation: `-x`.
    /// Desugared to `f64_sub(0.0, x)`.
    UnaryNeg(Box<Expr>, Span),
    /// Unary bitwise NOT: `!x`.
    /// Desugared to `u64_not(x)`.
    UnaryBitNot(Box<Expr>, Span),
    /// Source field projection: `base.ordinal`, `base.vector`.
    /// Resolved by the compiler to a node that reads from the source item.
    FieldAccess {
        /// The wire the field is read from.
        source: String,
        /// The field's name.
        field: String,
        /// Where the projection appears.
        span: Span,
    },
    /// `<expr> as <type>` — SRD-84 Part 1b type-coercion cast. An
    /// *optional, alignment-only* type-fusion infill: a no-op when the
    /// inner expression's type already matches the target, otherwise
    /// the compiler inserts the SRD-79 fusion adapter (or errors if no
    /// valid fusion exists). The cast's type is its target.
    Cast(Box<Expr>, crate::PortType, Span),
    /// `for <comprehension>` in expression position — a comprehension
    /// producer (SRD 113 §3.1). Binds a `Streamer` wire.
    For(Box<ForSource>),
}

/// Binary arithmetic operator kind.
#[derive(Debug, Clone, Copy)]
pub enum BinOpKind {
    /// `+` — desugars to `u64_add` or `f64_add` based on operand types
    Add,
    /// `-` — desugars to `u64_sub` or `f64_sub` based on operand types
    Sub,
    /// `*` — desugars to `u64_mul` or `f64_mul` based on operand types
    Mul,
    /// `/` — desugars to `u64_div` or `f64_div` based on operand types
    Div,
    /// `%` — desugars to `u64_mod` or `f64_mod` based on operand types
    Mod,
    /// `**` — desugars to `pow(a, b)` (always f64)
    Pow,
    /// `&` — desugars to `u64_and(a, b)`
    BitAnd,
    /// `|` — desugars to `u64_or(a, b)`
    BitOr,
    /// `^` — desugars to `u64_xor(a, b)`
    BitXor,
    /// `<<` — desugars to `u64_shl(a, b)`
    Shl,
    /// `>>` — desugars to `u64_shr(a, b)`
    Shr,
    /// `==` — desugars to `u64_eq` / `f64_eq`. Output type is `u64`
    /// (0 = false, 1 = true).
    Eq,
    /// `!=` — desugars to `u64_ne` / `f64_ne`. Output type is `u64`.
    Ne,
    /// `<` — desugars to `u64_lt` / `f64_lt`. Output type is `u64`.
    Lt,
    /// `>` — desugars to `u64_gt` / `f64_gt`. Output type is `u64`.
    Gt,
    /// `<=` — desugars to `u64_le` / `f64_le`. Output type is `u64`.
    Le,
    /// `>=` — desugars to `u64_ge` / `f64_ge`. Output type is `u64`.
    Ge,
    /// `&&` — eager logical-and (SRD-84 Part 1). Desugars to
    /// `u64_and(a != 0, b != 0)`: both operands evaluate, each is
    /// normalised to truthiness (`0`/`1`), and the bitwise-and of two
    /// truthiness values is logical-and. Output type is `u64` (`0`/`1`).
    /// Lowest precedence, below comparison. Short-circuit is a deferred
    /// optimisation (SRD-84 §"eager").
    And,
    /// `||` — eager logical-or (SRD-84 Part 1). Desugars to
    /// `u64_or(a != 0, b != 0)`. Output type is `u64` (`0`/`1`). Binds
    /// looser than `&&`.
    Or,
}

/// A typed parameter in a module signature.
#[derive(Debug, Clone)]
pub struct TypedParam {
    /// The parameter's name.
    pub name: String,
    /// The declared type keyword.
    pub typ: String, // "u64", "f64", "String", "bytes", etc.
}

/// A formal module definition with typed interface.
///
/// ```text
/// hash_range(input: u64, max: u64) -> (value: u64) := {
///     h := hash(input)
///     value := mod(h, max)
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ModuleDef {
    /// The module's name.
    pub name: String,
    /// The typed inputs, in signature order.
    pub params: Vec<TypedParam>,
    /// The typed outputs, in signature order.
    pub outputs: Vec<TypedParam>,
    /// The module body.
    pub body: Vec<Statement>,
    /// Where the definition appears.
    pub span: Span,
}

/// A function call expression.
#[derive(Debug, Clone)]
pub struct CallExpr {
    /// The function's name.
    pub func: String,
    /// The arguments, in call order.
    pub args: Vec<Arg>,
    /// Where the call appears.
    pub span: Span,
}

/// A function argument: positional or named.
#[derive(Debug, Clone)]
pub enum Arg {
    /// Positional: just an expression
    Positional(Expr),
    /// Named: `name: expr`
    Named(String, Expr),
}

#[cfg(test)]
mod modifier_tests {
    use super::*;

    #[test]
    fn empty_set_has_no_modifiers() {
        let m = BindingModifier::NONE;
        assert!(!m.has_any());
        assert!(!m.is_const() && !m.is_shared() && !m.is_volatile());
    }

    #[test]
    fn single_modifier_consts_match_expected_flags() {
        assert!(BindingModifier::CONST.is_const());
        assert!(!BindingModifier::CONST.is_shared());
        assert!(!BindingModifier::CONST.is_volatile());

        assert!(BindingModifier::SHARED.is_shared());
        assert!(!BindingModifier::SHARED.is_const());

        assert!(BindingModifier::VOLATILE.is_volatile());
        assert!(!BindingModifier::VOLATILE.is_const());
    }

    #[test]
    fn from_iter_collects_combinations() {
        let m = BindingModifier::try_from_iter([WireModifier::Const, WireModifier::Shared])
            .expect("const+shared is valid");
        assert!(m.is_const() && m.is_shared());
        assert!(!m.is_volatile());

        let m = BindingModifier::try_from_iter([WireModifier::Shared, WireModifier::Volatile])
            .expect("shared+volatile is valid");
        assert!(m.is_shared() && m.is_volatile());
    }

    #[test]
    fn from_iter_rejects_const_plus_volatile() {
        let err = BindingModifier::try_from_iter([WireModifier::Const, WireModifier::Volatile])
            .expect_err("const+volatile must be rejected");
        assert!(
            err.contains("const") && err.contains("volatile"),
            "error should name both keywords: {err}"
        );
    }

    #[test]
    fn from_iter_rejects_const_shared_volatile() {
        // Triple combination subsumes the contradiction.
        let err = BindingModifier::try_from_iter([
            WireModifier::Const,
            WireModifier::Shared,
            WireModifier::Volatile,
        ])
        .expect_err("triple combo includes the contradictory pair");
        assert!(err.contains("const") && err.contains("volatile"));
    }

    #[test]
    fn iter_yields_modifiers_in_stable_order() {
        let m =
            BindingModifier::try_from_iter([WireModifier::Volatile, WireModifier::Shared]).unwrap();
        // Insertion order was Volatile, Shared — but iter yields
        // in fixed declaration order: Const, Shared, Volatile.
        let collected: Vec<_> = m.iter().collect();
        assert_eq!(
            collected,
            vec![WireModifier::Shared, WireModifier::Volatile]
        );
    }

    #[test]
    fn equality_distinguishes_combinations() {
        let const_only = BindingModifier::CONST;
        let const_shared =
            BindingModifier::try_from_iter([WireModifier::Const, WireModifier::Shared]).unwrap();
        assert_ne!(
            const_only, const_shared,
            "const-only must not equal const+shared"
        );
    }
}
