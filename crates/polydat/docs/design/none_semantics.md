# None Semantics

The polydat-internal contract for how `Value::None` flows
through the kernel and the language surface. This doc
extends three axiom-level statements with the mechanism
detail they reference:

- [composition_substrate.md T1 (typed return)](composition_substrate.md)
  — None as the canonical "absent" sentinel in the Value enum.
- [runtime_model.md D1 (typed return)](runtime_model.md)
  + clean-flag interaction at the runtime layer.
- [grammar.md G1 (auto-extern)](grammar.md)
  — interaction with the interpolation boundary and the
  conditional-shadow `const` compilation rule.

This document is normative for the implemented language and
runtime surfaces.

## Motivation

Polydat's `Value` enum carries `Value::None` as the canonical "absent"
sentinel. The kernel's name-resolution chokepoints
(`PolydatKernel::get_constant`, `PolydatKernel::lookup`) already treat
`Value::None` outputs as "not present in this scope" and fall
through accordingly. But the same discipline wasn't applied at
the language surface: string-literal interpolation in source
code and bind-point substitution at op-template render time
both silently coerced `None` into either `Str("")` or the literal
text `"None"` (Debug formatting). This conflated **absent** with
**present-but-empty** — a type-system error that corrupted
downstream wire-protocol bytes (an empty `'source_model': ''`
field reached a CQL cluster; cndb echoed it back as `"NONE"`).

This SRD pins down the semantics: ordinary nodes propagate
`Value::None` unless their metadata explicitly declares
`accepts_none_inputs()`. Rendering boundaries never coerce
absence silently.

The principle is **None-propagation**, the same pattern SQL uses
for NULL (`NULL || 'foo'` → NULL) and Rust uses for `Option`'s
`?` operator. Absence is sticky; mixing it into a string yields
no string at all.

## Three orthogonal rules

The full semantic discipline factors into three rules, applied
at three different sites. Each is independently necessary; the
combination produces the workload-author surface the SRD-73
`set:` desugar (and any other "shadow if present" pattern)
relies on.

### Rule 1 — String interpolation propagates None

A source-level string literal containing one or more `{X_i}`
placeholders desugars (via `parse_interpolated_string` in
`dsl/parser.rs`) to a `printf(fmt, X_1, ..., X_n)` call. The
`Printf` node's `eval` now applies:

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
   PolydatKernel::get_constant("X") filters Value::None → returns None
   ↓
   PolydatKernel::lookup("X") falls through to find_input tier
```

`printf` never renders `Value::None` through its debug
representation. Doing so would create the real string `"None"`,
which would be returned by `get_constant` and incorrectly shadow
an outer binding for `X`.

### Rule 2 — Defaults are explicit expressions

`{X}` is the only interpolation bind-point form. If `X` is
absent, Rule 1 propagates `None`. The grammar does not define
`{X?}` or `{X ?? Y}`.

Authors who require a fallback use an explicit None-aware node
such as `default_or(X, fallback)` in the expression graph, then
interpolate that binding. Empty-string substitution is therefore
an authored value, never an implicit rendering rule.

### Rule 3 — Op-template render refuses silent None

`Value::to_display_strict` lives in `src/ast.rs` and returns
`Option<String>`: `None` for `Value::None` and `Some(text)` otherwise.
Any host wire renderer must use this strict primitive and report an
explicit error naming the bind point when it returns `None`. Calling
`to_display_string`, which renders the literal text `"None"`, does not
satisfy the op-template rendering contract.

The render path is the wire-protocol boundary — bytes leaving via
this path go to a remote system, so the silent-empty default
that `to_display_string` used to provide had to be removed here.
`to_display_string` stays unchanged for log / diagnostic contexts
where empty is acceptable (it's renamed conceptually as the "lossy"
form; render sites must use the strict variant).

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
filters it. `lookup` falls through to `find_input` — which
finds nothing because `set:` only emitted `const X` and not
also `extern X`.

The conditional-shadow compiler rule completes the behavior.

## Conditional-shadow semantics for `const`

`polydat/src/dsl/compile.rs` (both
the main `compile()` and `compile_filtered()` paths) detects
const declarations whose RHS references at least one name
(via the existing `dsl::validate::collect_references` walker)
and emits an implicit `extern NAME: Ext` input slot in
addition to the const output. The two-tier read in `lookup`
then provides the fall-through automatically:

1. `get_constant(NAME)` — own scope's folded output. Real
   value → return it. `Value::None` → **fall through**
   (existing filter, polydatkernel.rs:458-462).
2. `find_input(NAME)` — own scope's input slot, populated at
   `materialize_wiring_from_outer` time from the outer
   scope's `NAME` binding (if any). If wired, return that
   value.

Net behavior: `const NAME := <expr>` becomes a *conditional
shadow* when its RHS could fold to None. Real value → shadows
outer. None → outer's binding shows through. The `set:` sugar
from SRD-73 is unchanged canonical Polydat; the new semantics
emerge from the compiler's handling of every const, not from
special-casing the sugar.

**Pure-literal exception (SRD-13f Gate 2):** consts whose RHS
has zero name references (e.g. `const x := 1` from polydat's
per-iteration comprehension synthesis path — see polydat spec
§9.5's `scope_once`) are NOT auto-externed. They always fold
to a real value, never reach the fall-through path, and must
not appear as input slots per the Gate 2 invariant. The
reference-presence check is the precise discriminator.

**Wiring composition (`materialize_wiring_from_outer`):** the
read invariant from SRD-13f §"The read invariant" requires
`inner.read_input(X) ≡ outer.lookup(X)`. For const outputs
where the buffer may hold `Value::None` (Rule 1) but
`outer.lookup` falls through to outer's wired-from-grandparent
slot, the wiring uses value-copy via `outer.lookup` instead
of cell-attach (polydatkernel.rs:711-727). Cell-attach would
broadcast the raw `None` buffer and defeat the chain walk;
value-copy of `lookup` result keeps inner aligned with the
invariant. For non-const computed outputs (per-cycle dynamic
values) cell-attach is still the right primitive — those
truly need live broadcast.

Note: when both outer and inner declare `const NAME := <lit>`
with non-None values, inner's shadow wins via the normal
get_constant path — the fall-through only fires when the
inner const evaluates to None.

## Test contract

The executable contract is covered at three layers:

1. `polydat/src/kernel/engines.rs` implements the generic
   `None` propagation guard before node invocation.
   `polydat/src/library/format.rs::tests` verifies the defined-input
   `printf` path, while the scope-composition tests exercise its
   `None` path through a compiled kernel.

2. `polydat/tests/scope_composition.rs` — scope-
   composition integration tests proving the const →
   get_constant → lookup chain:
   - `const_with_bound_interpolation_shadows_outer` — happy
     path (real value shadows).
   - `const_with_unbound_interpolation_falls_through_to_outer`
     — conditional-shadow fall-through (None → outer wins).
   - `three_scope_chain_transitive_fall_through` — None in
     a middle scope is transparent across descendants
     (covers Step 2's wiring fix).
   - `pure_literal_const_does_not_auto_extern` — Gate 2
     invariant preserved.

3. `polydat/src/library/probability.rs::tests` verifies
   `default_or` for present, absent, typed, and extern-backed input.

Host renderers are responsible for testing Rule 3 at their own wire
substitution boundary.

## Implementation correspondence

- `PolydatState::eval_node` enforces Rule 1 for `Printf` and every
  node that does not opt into `accepts_none_inputs`.
- `default_or` is the explicit None-aware fallback primitive
  for Rule 2.
- The compiler's conditional-shadow auto-extern rule and
  `materialize_wiring_from_outer` implement scope fall-through.
- `Value::to_display_strict` is the Polydat primitive required by
  Rule 3; the host renderer owns bind-point error construction.

## Why this is safe

- **The desugar invariant is preserved.** `set:` writes
  canonical Polydat; the new semantics emerge from how the compiler handles
  `const`, not from special-casing the sugar.
- **Resolved bind points retain ordinary rendering semantics.**
  Rule 1's hot path is
  one boolean check per placeholder slot — already negligible.
- **Absent bind points fail visibly.** `Value::None` propagates and
  then reaches either the
  fall-through path (correct), Rule 3's render-time error
  (loud), or — if author opted in — empty substitution
  (explicit).
- **`Value::None` semantics are unified.** Every operation that
  consumes a value either propagates None upward or refuses
  to silently coerce it.

## See also

- [Scope Model](scope_model.md) — scope chain, visibility
  rules, two-tier lookup. The existing machinery this SRD
  leverages.
- [Wire Materialization](wire_materialization.md) —
  cross-scope wire materialization invariants. Conditional-
  shadow `const` extends the read invariant: a None const
  output is structurally identical to "no const declared" for
  consumers.
