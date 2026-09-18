# None Semantics

The polydat-internal contract for how `Value::None` flows
through the kernel and the language surface: the None rule at
the language level. It gives mechanism detail to three
axiom-level statements — T1 (typed return) in
[composition_substrate.md](composition_substrate.md), where
None is the canonical "absent" sentinel of the `Value` enum;
D1 in [runtime_model.md](runtime_model.md), whose typed return
None never silently becomes a string; and G1 (auto-extern) in
[polydat_grammar.md §18](polydat_grammar.md#sec-gaxioms), through the conditional-shadow
`const` compilation rule below.

This document is normative for the implemented language and
runtime surfaces.

## Motivation

Polydat's `Value` enum carries `Value::None` as the canonical "absent"
sentinel. The kernel's name-resolution chokepoints
(`get_constant`, `lookup` in `kernel/state.rs`) treat
`Value::None` outputs as "not present in this scope" and fall
through accordingly. The same discipline has to hold at the
language surface: string-literal interpolation in source code
and bind-point substitution at op-template render time must
not coerce `None` into either `Str("")` or the literal text
`"None"` (Debug formatting). Doing so conflates **absent** with
**present-but-empty** — a type-system error whose symptom is
corrupted bytes at a wire-protocol boundary, where an empty
field is a real value a remote system stores and echoes back.

The contract: ordinary nodes propagate `Value::None` unless
their metadata explicitly declares `accepts_none_inputs()`.
Rendering boundaries never coerce absence silently.

The principle is **None-propagation**, the same pattern SQL uses
for NULL (`NULL || 'foo'` → NULL) and Rust uses for `Option`'s
`?` operator. Absence is sticky; mixing it into a string yields
no string at all.

## Three orthogonal rules

The full semantic discipline factors into three rules, applied
at three different sites. Each is independently necessary; the
combination produces the workload-author surface the `set:`
desugar (and any other "shadow if present" pattern) relies on.

### Rule 1 — String interpolation propagates None

A source-level string literal containing one or more `{X_i}`
placeholders desugars (via `parse_interpolated_string` in
`polydat-grammar/src/parser.rs`) to a `printf(fmt, X_1, ..., X_n)` call. The
`Printf` node's `eval` applies:

> If any input slot referenced by a `{}` placeholder holds
> `Value::None`, the output is `Value::None`. Otherwise the
> output is the formatted `Value::Str(...)`.

Inputs not referenced by any placeholder (unusual but
permitted by the variadic signature) do not trigger
propagation.

Cascading consequence through `const`:

```
const X := "{Y}"            // Y is unbound (read as Value::None)
   ↓
   printf("{}", Y) → Value::None     // Rule 1
   ↓
   const-fold writes Value::None to X's output buffer
   ↓
   get_constant("X") filters Value::None → returns None
   ↓
   lookup("X") falls through to the input-slot tier
```

`printf` never renders `Value::None` through its debug
representation. Doing so would create the real string `"None"`,
which would be returned by `get_constant` and incorrectly shadow
an outer binding for `X`.

Rule 1 is enforced on every engine, and for every node, not
only `printf`: the interpreter's `EngineCore::eval_node`
guards each node before invoking it; the closure tier and the
hybrid kernel keep a None mask over the slot buffer, and a step
whose node does not accept None marks every output None without
running when any input is None. Native code never accepts None:
a step downstream of an unset extern is a closure step, so a
None reaches a native segment only if a host clears an extern
after the build, and that is a panic; the interpreter's native-
cone extraction admits a None-tolerant node into a cone only
when every input is an intra-cone wire, where no None can
arrive. `Kernel::pull` reads a slot that holds None as `None`
on every engine.

### Rule 2 — Defaults are explicit expressions

`{X}` is the only interpolation bind-point form. If `X` is
absent, Rule 1 propagates `None`. The grammar does not define
`{X?}` or `{X ?? Y}`.

Authors who require a fallback use an explicit None-aware node
such as `default_or(X, fallback)` in the expression graph, then
interpolate that binding. Empty-string substitution is therefore
an authored value, never an implicit rendering rule.

### Rule 3 — Op-template render refuses silent None

`Value::to_display_strict` lives in `polydat-core/src/ast.rs` and returns
`Option<String>`: `None` for `Value::None` and `Some(text)` otherwise.
Any host wire renderer must use this strict primitive and report an
explicit error naming the bind point when it returns `None`. Calling
`to_display_string`, which renders the literal text `"None"`, does not
satisfy the op-template rendering contract.

The render path is the wire-protocol boundary: bytes leaving by it go
to a remote system, where an empty field is a real value the system
stores and echoes back, so absence must not render as one.
`to_display_string` is the lossy form, for a log or a diagnostic where
empty is acceptable; a render site uses the strict one.

Structural omission of an op-template segment is not a
bind-point operation. The host must construct or select the
complete segment explicitly before rendering.

## Interaction with `set:` and the grammar invariant

The workload-parser sugar `set: { X: "{Y}" }` desugars to:

```
const X := "{Y}"
```

This is canonical Polydat grammar — the desugar produces no
special-case AST. The semantic correctness lives in **how Polydat
compiles and evaluates `const NAME := <expr>`**, not in the
sugar layer.

With Rule 1, `const X := "{Y}"` where `Y` is unbound
yields `Value::None` as X's output buffer. `get_constant`
filters it. `lookup` falls through to the input-slot tier —
which finds nothing unless the compiler also gave `X` an input
slot.

The conditional-shadow compiler rule completes the behavior.

## Conditional-shadow semantics for `const`

`assemble_parent` in `dsl/compile.rs` — the one lowering every
entry point goes through, on every engine — collects the names
each binding's RHS references (`dsl::validate::collect_references`)
and gives every referenced name not defined locally an input
slot, so a const whose RHS references at least one name gets an
implicit input slot in addition to its const output. The
two-tier read in `lookup` then provides the fall-through
automatically:

1. `get_constant(NAME)` — own scope's folded output
   (`kernel/state.rs`). Real value → return it. `Value::None`
   → **fall through**.
2. the input slot `NAME` — populated at
   `materialize_wiring_from_outer` time from the outer
   scope's `NAME` binding (if any). If wired, return that
   value.

Net behavior: `const NAME := <expr>` becomes a *conditional
shadow* when its RHS could fold to None. Real value → shadows
outer. None → outer's binding shows through. The `set:` sugar
is unchanged canonical Polydat; the semantics emerge from the
compiler's handling of every const, not from special-casing the
sugar.

**Pure-literal exception:** consts whose RHS has zero name
references (e.g. `const x := 1`, the form a comprehension's
per-iteration binding takes) are NOT auto-externed. They always
fold to a real value, never reach the fall-through path, and
must not appear as input slots. The reference-presence check is
the precise discriminator.

**Wiring composition (`materialize_wiring_from_outer`):** the
read invariant of [wire_materialization.md](wire_materialization.md)
requires that reading `X` on the inner scope returns what
`outer.lookup(X)` returns. For const outputs, where the buffer
may hold `Value::None` (Rule 1) but `outer.lookup` falls through
to the outer's wired-from-grandparent slot, step 2 of the
materializer uses value-copy via `outer.lookup` instead of
cell-attach. Cell-attach would broadcast the raw `None` buffer
and defeat the chain walk; value-copy of the `lookup` result
keeps the inner scope aligned with the invariant. For non-const
computed outputs (per-cycle dynamic values) cell-attach is
still the right primitive — those truly need live broadcast.

The four cases the rule fixes:

- a const whose interpolation resolves shadows the outer
  binding of the same name;
- a const whose interpolation is unbound falls through to the
  outer binding;
- a None in a middle scope is transparent across descendants:
  the fall-through is transitive through any number of scope
  layers;
- a pure-literal const is not auto-externed.

Note: when both outer and inner declare `const NAME := <lit>`
with non-None values, inner's shadow wins via the normal
`get_constant` path — the fall-through only fires when the
inner const evaluates to None.

## Implementation correspondence

- `EngineCore::eval_node` enforces Rule 1 on the interpreter
  for `Printf` and every node that does not opt into
  `accepts_none_inputs`; the closure tier's step guard and the
  hybrid kernel's `run_hybrid_step` enforce it through the None
  mask on the compiled engines, and native code never sees a
  None.
- `default_or` is the explicit None-aware fallback primitive
  for Rule 2.
- The compiler's conditional-shadow auto-extern rule
  (`assemble_parent`) and `materialize_wiring_from_outer`
  implement scope fall-through.
- `Value::to_display_strict` is the Polydat primitive required by
  Rule 3; the host renderer owns bind-point error construction.

## Why this is safe

- **The desugar invariant is preserved.** `set:` writes
  canonical Polydat; the semantics emerge from how the compiler
  handles `const`, not from special-casing the sugar.
- **Resolved bind points retain ordinary rendering semantics.**
  Rule 1's hot path is one boolean check per placeholder slot —
  negligible.
- **Absent bind points fail visibly.** `Value::None` propagates
  and then reaches either the fall-through path (correct),
  Rule 3's render-time error (loud), or — if the author wrote a
  `default_or` — the authored fallback (explicit).
- **`Value::None` semantics are unified.** Every operation that
  consumes a value either propagates None upward or refuses
  to silently coerce it, on every engine.

## See also

- [Scope Model](scope_model.md) — scope chain, visibility
  rules, two-tier lookup. The machinery this document
  leverages.
- [Wire Materialization](wire_materialization.md) —
  cross-scope wire materialization invariants. Conditional-
  shadow `const` extends the read invariant: a None const
  output is structurally identical to "no const declared" for
  consumers.
