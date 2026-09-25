---
type: specification
title: None Semantics
timestamp: 2026-09-25
description: How Value::None flows through the kernel and the language, including the conditional-shadow const rule.
tags: [runtime, language, types]
---

# None Semantics

This document specifies how `Value::None` flows through the kernel
and the language surface: the None rule at the language level, the
three rules that apply it at interpolation, default, and render sites,
and the conditional-shadow compilation rule for `const`. It gives the
mechanism behind three axiom-level statements: T1 (typed return) in
[composition_substrate.md](composition_substrate.md), where None is the
canonical "absent" sentinel of the `Value` enum; D1 in
[runtime_model.md](runtime_model.md), under which a typed return of
None never silently becomes a string; and G1 (auto-extern) in
[polydat_grammar.md §18](polydat_grammar.md#sec-gaxioms), through the
conditional-shadow `const` compilation rule below. This document is
normative for the implemented language and runtime surfaces.

**Related specifications.** [Scope Model](scope_model.md) (the scope
chain, visibility rules, and the two-tier lookup this document relies
on); [Wire Materialization](wire_materialization.md) (the cross-scope
read invariant, which conditional-shadow `const` extends).

## Motivation

`Value::None` is the canonical "absent" sentinel of the `Value` enum.
The kernel's name-resolution chokepoints (`get_constant`, `lookup` in
`kernel/state.rs`) treat a `Value::None` output as "not present in
this scope" and fall through to the next tier. The language surface
follows the same discipline: string-literal interpolation in source
code and bind-point substitution at op-template render time must not
coerce `None` into `Str("")` or into the literal text `"None"` (Debug
formatting). Either coercion conflates **absent** with
**present-but-empty**, a type error whose symptom is corrupted bytes at
a wire-protocol boundary, where an empty field is a real value that a
remote system stores and returns.

The contract is that ordinary nodes propagate `Value::None` unless
their metadata declares `accepts_none_inputs()`, and that rendering
boundaries never coerce absence silently. This is
**None-propagation**, the pattern SQL uses for NULL
(`NULL || 'foo'` → NULL) and Rust uses for `Option`'s `?` operator:
absence propagates, and interpolating it into a string yields no
string at all.

## Three orthogonal rules

The discipline consists of three rules, each applied at a different
site. Each rule is independently necessary, and together they give
the `set:` desugar (and any other "shadow if present" pattern) its
behavior.

### Rule 1 — String interpolation propagates None

A source-level string literal containing one or more `{X_i}`
placeholders desugars (via `parse_interpolated_string` in
`polydat-grammar/src/parser.rs`) to a `printf(fmt, X_1, ..., X_n)` call. The
`Printf` node's `eval` applies:

> If any input slot referenced by a `{}` placeholder holds
> `Value::None`, the output is `Value::None`. Otherwise the
> output is the formatted `Value::Str(...)`.

Inputs not referenced by any placeholder (unusual but permitted by the
variadic signature) do not trigger propagation.

The consequence through `const` is:

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
representation. Doing so would create the real string `"None"`, which
`get_constant` would return and which would incorrectly shadow an
outer binding for `X`.

Rule 1 applies to every node, not only `printf`, on the interpreter
(P1), the closure tier (P2), and native (P3). The interpreter's
`EngineCore::eval_node` checks each node before invoking it. The
closure tier and the hybrid kernel of P3 keep a None mask over the
slot buffer, and a step whose node does not accept None marks every
output None without running when any input is None. Native code never
accepts None. On P3 a step downstream of an unset extern is a closure
step, so a None arrives at a native segment only if a host clears an
extern after the build, and that is a panic. The interpreter's
native-cone extraction admits a None-tolerant node into a cone only
when every input is an intra-cone wire, where no None can arrive. Pure
native refuses to run while any extern is unset, so no None arises
there. `Kernel::pull` returns `None` for a slot that holds None on the
interpreter, the closure tier, and native.

### Rule 2 — Defaults are explicit expressions

`{X}` is the only interpolation bind-point form. If `X` is absent,
Rule 1 propagates `None`. The grammar does not define `{X?}` or
`{X ?? Y}`.

An author who requires a fallback uses an explicit None-aware node
such as `default_or(X, fallback)` in the expression graph and then
interpolates that binding. Empty-string substitution is therefore an
authored value, never an implicit rendering rule.

### Rule 3 — Op-template render refuses silent None

`Value::to_display_strict` lives in `polydat-core/src/ast.rs` and returns
`Option<String>`: `None` for `Value::None` and `Some(text)` otherwise.
Any host wire renderer must use this strict primitive and report an
explicit error naming the bind point when it returns `None`. Calling
`to_display_string`, which renders the literal text `"None"`, does not
satisfy the op-template rendering contract.

The render path is the wire-protocol boundary. Bytes rendered there
are sent to a remote system, where an empty field is a real value that
the system stores and returns, so absence must not render as an empty
field. `to_display_string` is the lossy form, for a log or a
diagnostic where empty is acceptable; a render site uses the strict
form.

Structural omission of an op-template segment is not a bind-point
operation. The host must construct or select the complete segment
explicitly before rendering.

## Interaction with `set:` and the grammar invariant

The workload-parser sugar `set: { X: "{Y}" }` desugars to:

```
const X := "{Y}"
```

This is canonical Polydat grammar, and the desugar produces no
special-case AST. The semantics are defined by **how Polydat compiles
and evaluates `const NAME := <expr>`**, not by the sugar layer.

Under Rule 1, `const X := "{Y}"` with `Y` unbound writes `Value::None`
to X's output buffer. `get_constant` filters it, and `lookup` falls
through to the input-slot tier, which finds nothing unless the
compiler also gave `X` an input slot. The conditional-shadow compiler
rule below gives `X` that slot.

## Conditional-shadow semantics for `const`

`assemble_parent` in `dsl/compile.rs` is the one lowering that every
entry point uses, for all four engines (the interpreter, the closure
tier, native, and pure native). It collects the names each binding's
RHS references (`dsl::validate::collect_references`) and gives every
referenced name not defined locally an input slot, so a const whose
RHS references at least one name gets an implicit input slot in
addition to its const output. The two-tier read in `lookup` then
provides the fall-through:

1. `get_constant(NAME)` — own scope's folded output
   (`kernel/state.rs`). Real value → return it. `Value::None`
   → **fall through**.
2. the input slot `NAME` — populated at
   `materialize_wiring_from_outer` time from the outer
   scope's `NAME` binding (if any). If wired, return that
   value.

As a result, `const NAME := <expr>` is a *conditional shadow* when its
RHS can fold to None. A real value shadows the outer binding; a None
leaves the outer binding visible. The `set:` sugar remains canonical
Polydat, and the semantics come from the compiler's handling of every
const, not from special-casing the sugar.

**Pure-literal exception:** a const whose RHS has zero name references
(e.g. `const x := 1`, the form a comprehension's per-iteration binding
takes) is NOT auto-externed. It always folds to a real value, never
takes the fall-through path, and must not appear as an input slot.
The presence of a name reference is the exact criterion.

**Wiring composition (`materialize_wiring_from_outer`):** the read
invariant of [wire_materialization.md](wire_materialization.md)
requires that reading `X` on the inner scope returns what
`outer.lookup(X)` returns. A const output's buffer may hold
`Value::None` (Rule 1) while `outer.lookup` falls through to the
outer's slot wired from the grandparent, so step 2 of the materializer
copies the value returned by `outer.lookup` instead of attaching the
cell. Attaching the cell would broadcast the raw `None` buffer and
defeat the chain walk; copying the `lookup` result keeps the inner
scope consistent with the invariant. For a non-const computed output,
whose value changes when its inputs change, cell attachment is the
correct primitive, because consumers need the value broadcast live.
Conditional-shadow `const` thereby extends the read invariant: to its
consumers, a const whose output is None is structurally identical to
no const declared.

The rule produces four behaviors:

- a const whose interpolation resolves shadows the outer binding of
  the same name;
- a const whose interpolation is unbound falls through to the outer
  binding;
- a None in a middle scope is transparent to its descendants: the
  fall-through is transitive through any number of scope layers;
- a pure-literal const is not auto-externed.

When both the outer and the inner scope declare `const NAME := <lit>`
with non-None values, the inner const shadows the outer through the
normal `get_constant` path; the fall-through applies only when the
inner const evaluates to None.

## Implementation correspondence

- `EngineCore::eval_node` enforces Rule 1 on the interpreter for
  `Printf` and every node that does not declare `accepts_none_inputs`.
  The closure tier's step guard and the hybrid kernel's
  `run_hybrid_step` enforce it through the None mask on the closure
  tier and native, and native code never receives a None; pure native
  refuses to run with an unset extern.
- `default_or` is the explicit None-aware fallback primitive for
  Rule 2.
- The compiler's conditional-shadow auto-extern rule
  (`assemble_parent`) and `materialize_wiring_from_outer` implement
  scope fall-through.
- `Value::to_display_strict` is the Polydat primitive required by
  Rule 3; the host renderer constructs the bind-point error.

## Why this is safe

- **The desugar invariant is preserved.** `set:` produces canonical
  Polydat; the semantics come from how the compiler handles `const`,
  not from special-casing the sugar.
- **Resolved bind points keep ordinary rendering semantics.** Rule 1
  adds one boolean check per placeholder slot on the hot path.
- **Absent bind points fail visibly.** A `Value::None` propagates and
  then arrives at one of three places: the fall-through path, Rule 3's
  render-time error, or, if the author wrote a `default_or`, the
  authored fallback.
- **`Value::None` semantics are unified.** Every operation that
  consumes a value either propagates None or refuses to coerce it
  silently. This holds on the interpreter, the closure tier, and
  native; pure native admits no None, because it refuses to run with
  an unset extern.

## See also

- [Scope Model](scope_model.md) — scope chain, visibility rules,
  two-tier lookup.
- [Wire Materialization](wire_materialization.md) — cross-scope wire
  materialization invariants.
