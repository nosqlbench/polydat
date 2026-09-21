# The Expression Engine — Polydat Design

**Subtitle:** Polydat as host-embeddable evaluation utility.

Formalises the host-facing evaluation surface that emerges
from polydat's grammar. Names the embedding contract,
catalogs the surfaces, and shows how the substrate and
graph compiler are re-used at expression scale to give host
crates a typed, deterministic, library-rich evaluation
engine for free.

## Companion documents

- [Composition Substrate](composition_substrate.md) — the
  S/T/L pillars and the slot contract. The expression
  engine's typed-result guarantee follows directly from T1
  + T2.
- [The Graph Compiler](graph_compiler.md) — the construction
  pipeline. Embedded expression evaluation is the *same
  compiler* operating on smaller input — no separate
  evaluator exists.
- [The Runtime Model](runtime_model.md) — the R-axioms
  (data flow, caching, invalidation) and D-axioms
  (determinism guarantees). E3's bounded-determinism claim
  is the realisation of D1/D2/D3 at expression scale.
- [The Polydat Grammar](polydat_grammar.md) — G-axioms. G3
  (scope-chain transparency) + G6 (single grammar for
  expressions and full programs) compose into E1
  (self-contained submission) + E4 (library inheritance) +
  the expression-as-kernel correspondence in §2.
- [The Evaluation Model](evaluation_model.md) — the
  two-lifecycle classification and the const-binding
  contract; `eval_const_expr` is that contract at
  single-expression scale.

The forcing question: **polydat was designed to compile
workloads — full programs over typed coordinate streams. So
why does it also turn out to be a near-zero-cost embedded
expression engine for every host crate in the project? What
contract does that emergence rest on, and what does the host
agree to in exchange?** This doc says: the contract emerges
from the substrate + compiler operating uniformly across all
input sizes; the host agrees to submit self-contained text
and receive typed values; the cost is the substrate's
ordinary slot-contract overhead, which is small when the
expression is small.

---

## 1. The claim

Polydat's grammar is its own expression engine. The same
machinery that compiles a 200-line workload kernel compiles
a four-character expression like `"k+1"`. The substrate's
slot contract holds at every scale; the compiler's passes
fire uniformly; the result is a typed value the host
consumes.

> **For any text the polydat grammar accepts, the host crate
> can ask polydat to evaluate it against an optional context
> and receive a typed `Value` (or a typed list of `Value`).
> The host inherits the full node library, the lifecycle
> classification, the typed slot contract, and the
> deterministic-evaluation guarantee — without writing a
> parser, an evaluator, or a type checker. The host's
> obligation is the text and (when needed) the context; the
> output is a typed `Value` answer.**

This is what was informally described as "polydat doubles as
an embedded expression engine." This doc names it as the
**Embedding Contract** — the host-facing utility surface
that emerges from the substrate + compiler.

The capability is not bolted on. It is the substrate
operating at small scale. No additional infrastructure
beyond what already exists for workload kernels is required;
the expression engine *is* the workload engine, just with
shorter input.

---

## 2. The expression-as-kernel correspondence

Every text input the host submits compiles through the same
pipeline, to the same kind of program, under the same slot
contract, as a full workload kernel. The size of the
program is proportional to the input's complexity; an
expression like `"k * 2 + 1"` compiles to a three-node
program (two ops + a const-fold output binding).

```text
host text input            polydat compile pipeline
──────────────────         ──────────────────────────────────────
"k * 2 + 1"                Parse → Bind → wire resolution with
                           adapters → Node Fusion → DCE →
                           topological sort → ResolvedDag →
                           kernel on the chosen Engine, its
                           constants folded at build (3 ops, 1 output)

                                            │
                                            ▼

host context (binding         materialize_subscope / set_input
for "k" as scope-input)       ─────────────────────────────────
                                            │
                                            ▼

                              Kernel::pull → Value::U64(<result>)

                                            │
                                            ▼

                              host receives Value
```

The correspondence is total: every expression has a kernel
representation, and every kernel can be reduced to a
single-expression case if its body is a single binding. The
distinction between "expression" and "workload" is
quantitative (lines of input, count of bindings), not
qualitative (no different machinery).

This correspondence is what makes the expression engine
*free* — there's no separate engine to maintain. Any
improvement to the substrate or graph compiler improves the
expression engine automatically.

---

## 3. The host-facing surfaces

Three public surfaces, one for each evaluation depth.

### 3.1 `eval_const_expr` — const-fold at compile time

Location: [`crate::dsl::compile::eval_const_expr`].

Signature:

```rust
pub fn eval_const_expr(source: &str) -> Result<Value, EmbeddingError>
```

Semantics: compile the text wrapped as a single output
binding (`out := <source>`); read the folded constant from
the compiled kernel; return it. The compilation succeeds iff
the expression is statically foldable — its upstream cone
reaches *no* dynamic inputs. The grammar's full surface is
available — node calls, literals, arithmetic, string ops —
but the expression's lifecycle must be Effectively-const
(per H1 / H2 / H3). The result is cached by source text, so
the same text compiles once per process.

Hosts that want the host's Rust type back directly use
`eval_const_expr_typed::<T>` (§5.3); the typed surface
returns `Result<T, EmbeddingError>` and removes the Value
enum from the boundary.

Use case: host-side `{...}` config expressions, where the
host has a small expression and a guarantee it should
resolve at activity-construction time (no `cycle` reference,
no external-write inputs). The host wraps the result in its
config-value contract — typically calling `.as_u64()` for
numeric fields or `to_display_string()` for string contexts;
the projection is host policy, not part of the embedding
contract.

**What works as a const expression:**

- Literals: `{42}`, `{3.14}`, `{"hello"}`
- Arithmetic: `{1000 * 1000}`, `{4 ** 0.5}`
- Function calls with constant args: `{hash(42)}`, `{mod(hash(42), 100)}`
- Catalog-registered metadata accessors (e.g.
  `{vector_count(h)}`, where `h` is a dataset handle wire)
- Nested: `{vector_count(h) / 10}` (after the host's
  param substitution pass has bound `h` — itself outside
  the embedding contract)

**What does NOT work:**

- References to cycle inputs: `{hash(cycle)}` → error
- References to undefined names: `{undefined_var}` → error
- Non-deterministic functions: `{counter()}` → error

Cost: one full compile + fold, ~ms scale. The compile
dominates; once compiled, the value is folded into the
program and reading it is free.

Failure modes (returned as typed `Err(EmbeddingError)` per
§6 Error Ontology):
- `Parse` — parse error in the text.
- `LifecycleMismatch` — the expression's upstream reaches a
  dynamic input (the surface promises const-only; the chain
  returns "depends on runtime inputs").
- `NodeEvalPanic` — a node `eval` panics during the fold
  (caught via `catch_unwind` and surfaced as a node-eval-panic
  error).

### 3.2 `interpolate_via_kernel` + evaluation — kernel-bound dynamic evaluation

#### 3.2.1 Why interpolation is text-level

The substrate holds scope state in typed slots accessible
to nodes inside a kernel's program. But host expression
text is *outside* any specific kernel program — it's text
the host is about to submit for compilation. To bridge
"text the host has" with "values the kernel holds,"
polydat exposes a **text-level** interpolation surface:
slot values are rendered to their display strings and
substituted into the text in place of `{name}` placeholders.
The output is text; the next step is ordinary
`eval_const_expr`.

This deliberately is **not** value-level injection
(where the host programmatically builds an expression tree
with bound values pre-substituted). Three reasons for the
text-level choice:

- **Grammar preservation.** Interpolation produces valid
  expression text; the eval step compiles ordinary text;
  the grammar is the contract at every stage. There's no
  separate "expression-with-bound-values" intermediate
  representation to maintain.
- **Decoupling of interpolation from eval.** A host that
  just wants placeholder substitution (e.g., to render a
  label string with kernel values) uses interpolation
  alone. A host with already-resolved text (no
  placeholders) uses eval alone. The two compose only when
  needed.
- **Lifecycle-gating naturally falls out at eval time.**
  Interpolation is type-erased (everything becomes a
  display string); the typed-and-lifecycle-checked
  evaluation happens at the eval step over the resulting
  text. If the post-interpolation text reaches a dynamic
  input that wasn't substituted away, `eval_const_expr`
  rejects it with a typed error.

The `{name}` surface is the *contract* for textual
embedding of slot values. Hosts can author expression text
freely with `{...}` placeholders; the kernel chain's slot
contract is the source of substitution; the eval step is
the typed result producer.

Note what this is not: braces are not a Polydat expression
form. In Polydat source, `{name}` has meaning only inside a
string literal, as interpolation
([polydat_grammar.md §9](polydat_grammar.md)). A host whose
config fields carry brace-delimited expressions — a
`dim: {vector_dim("glove-25-angular")}` in a workload file —
is using its own surface, and it is the host that decides
when to hand the inner text to `eval_const_expr`. Keeping
the two apart is why a Polydat program can be read without
knowing which host embedded it.

#### 3.2.2 The surfaces

Location: [`crate::kernel::interp::interpolate_via_kernel`].

Signature:

```rust
pub fn interpolate_via_kernel(
    text: &str, kernel: &dyn Lookup,
) -> Result<String, EmbeddingError>
```

Semantics: replace `{name}` placeholders in `text` with the
display string of `kernel.lookup(name)`. Returns the
substituted text. A `{name}` whose lookup yields nothing
(including a slot holding `Value::None`) is
`UnresolvedPlaceholder`.

`Lookup` (`kernel::interp::Lookup`) is the name resolution a
placeholder reads plus the compile ledger a source or
predicate that has to compile is charged to (`lookup` and
`ledger`): the interpreter kernel implements it, `KernelLookup`
wraps a kernel of any engine as one, and `Layered` puts a
tuple's bindings in front of any other lookup, forwarding
both. The typed kernel-bound surfaces,
`eval_kernel_bound_typed::<T>(text, &dyn Lookup)` and its
`_strict` variant (§5.3), compose interpolation with the
typed const fold over any of them, so a host holding a
`Box<dyn Kernel>` interpolates against the kernel it has:
`eval_kernel_bound_typed(text, &KernelLookup::new(kernel.as_ref()))`.

A name resolves to what the kernel holds for it now — an
input the host wrote, a coordinate it was positioned at —
and otherwise to what the build folded for it. The live
answer comes first because it is the later one: on some
engines a coordinate has a folded value, and that is the
value the program was built with, not the value the kernel
is at.

`KernelScope` is a wrapper rather than an
`impl Lookup for dyn Kernel` because one trait object cannot
become another: `&dyn Kernel` has no `dyn Lookup` vtable to
coerce into.

The canonical two-step composition:

```rust
let interpolated = interpolate_via_kernel(text, &kernel)?;
let value = eval_const_expr(&interpolated)?;
let truth = value.as_bool();
```

Step 1 (interpolate) brings the kernel's bound values into
the text. Step 2 (eval) compiles the now-bound text and
folds it. The lifecycle gating is preserved at step 2: if
the post-interpolation text still reaches a dynamic input,
the eval rejects it; if every name was substituted to a
static value, the fold succeeds.

Use case: a host's predicate evaluation — the host has a
text like `"{k} > 5"` (where `{k}` is an iter-var bound in
the calling kernel) and needs a boolean answer. The two-step
composition resolves the placeholder and folds the resulting
`5 > 5` (or `7 > 5`, etc.) to a boolean.

Cost: interpolation is O(name lookups + text length); the
follow-on `eval_const_expr` is a compile + fold (~ms scale)
the first time a given text is seen, and a cache hit after.

**One name-resolution contract.** Kernel-aware interpolation is
polydat's, through this surface, and nothing else resolves a
placeholder: a comprehension source or predicate that refers to a
scope binding reaches the same `Lookup`, and an unresolved name is
the typed error above rather than an empty substitution. A host does
not add a second interpolation dialect; a text it wants resolved goes
through this surface, and nested placeholders follow the bounded
fixed-point rules of §3.2.3.

#### 3.2.3 Interpolation alone — text rendering without evaluation

Interpolation is useful as a standalone operation when the
host needs the *rendered text* but not an evaluated value.
The output is host-domain text (a filesystem path, an SQL
fragment, a log line, a keyspace name) — text whose
consumer is not polydat. The host calls
`interpolate_via_kernel` and uses the returned string
directly; no `eval_const_expr` follows.

**Worked example: rendering a per-iteration data path.**

Suppose the host has a path template that depends on the
current scope's iter-vars:

```text
"data/{dataset}/k{k}_limit{limit}.bin"
```

and the current kernel's bindings are
`dataset = "sift1m"`, `k = 10`, `limit = 100` (typical
post-Context-Fusion state in a `for k in …, limit in … { … }`
traversal body running over a configured dataset). The
host's code:

```rust
use polydat::kernel::interp::interpolate_via_kernel;

let template = "data/{dataset}/k{k}_limit{limit}.bin";
let path = interpolate_via_kernel(template, &kernel)?;
// path = "data/sift1m/k10_limit100.bin"

let bytes = std::fs::read(&path)?;
// host proceeds with the resolved path; no polydat
// evaluation needed.
```

Three things to notice:

- **The result is text, not a `Value`.** The host gets a
  `String` and uses it for a host-domain purpose
  (filesystem read). Polydat is the *renderer*, not the
  *consumer*.
- **No expression grammar required.** The template is not
  a polydat expression — it's a string with `{placeholder}`
  syntax. Polydat doesn't try to parse `"data/.../"` as a
  Polydat expression; the `{...}` form is the only syntactic
  surface interpolation cares about.
- **Lifecycle gating doesn't apply.** Since no eval
  follows, there's no `LifecycleMismatch` to fire. If a
  placeholder is unresolved, `interpolate_via_kernel`
  returns `UnresolvedPlaceholder` (per §6); the host
  surfaces it as a missing-binding diagnostic.

The composability principle (E6) says interpolation and
evaluation compose only when both are needed. Standalone
interpolation is the half of that composition that uses
just the substitution.

### 3.3 `evaluate_spec` — list-yielding evaluation against a scope

Location:
[`crate::iteration::comprehension::eval::evaluate_spec`].

Signature:

```rust
pub fn evaluate_spec(
    spec_text: &str, kernel: &dyn Lookup,
) -> Result<Vec<Value>, EmbeddingError>
```

Semantics: a layered evaluator that recognises the clause-
source forms — `all(cursor)`, a bare wire / param / const
reference (resolved through the same `lookup` the `{name}`
path uses), bracket lists with spreads, partition calls, and
otherwise `eval_const_expr` on the interpolated text, falling
back to typed literal-list parsing (`1` → `U64`, `1.5` →
`F64`, `true` → `Bool`, anything else → `Str`). Returns a
vector of values per the recognised form's expansion.

Use case: comprehension clause-source expansion (the source
of every `k in <text>` clause of a `for` comprehension). The host text can
declare a *list* of values, not just a single value, and
`evaluate_spec` does the expansion against the scope's
`Lookup` — which is why a traversal opens on every engine:
the sources are evaluated against a `Layered` view of the
opening kernel's values, not against an engine-specific
kernel.

Cost: dominated by the recognition cascade (~us per cheap
form) + an `eval_const_expr` fallback for the literal-list
case (~ms, once per text).

### 3.4 The underlying surface: compiling a kernel

Location: [`crate::dsl::compile::compile_polydat`].

Signatures:

```rust
pub fn compile_polydat_with(source: &str, engine: Engine)
    -> Result<Box<dyn Kernel>, KernelError>;
pub fn compile_polydat_kernel(source: &str)
    -> Result<Box<dyn Kernel>, KernelError>;   // on Engine::default()
pub fn compile_polydat(source: &str)
    -> Result<PolydatKernel, String>;          // the interpreter's kernel
```

The full compilation entry points: text → a kernel on the
named engine, driven through the `Kernel` trait; or the
interpreter's concrete `PolydatKernel`, which is what the
kernel-bound typed surfaces take. `Engine::default()` is
native code where the build has it and the closure tier
otherwise. The three higher-level surfaces above are built on
this; the host reaches for it directly when it wants a kernel
rather than just a value.

Use case: host crates that pre-compile expressions for
repeated evaluation. A kernel becomes a shareable program
with `Kernel::into_program` (`Arc<dyn KernelProgram>`), and
each thread creates its own kernel from the program with
`create_kernel`; the interpreter's `PolydatKernel::into_program`
yields the concrete `Arc<PolydatProgram>`.

Cost: one full compile (~ms scale for small expressions).
Subsequent kernels from the program are fast (the program is
shared, state is per thread).

---

## 4. The Embedding Contract — E-axioms

The host gets seven guarantees in exchange for submitting
self-contained text. Each is a substrate / compiler
property at expression scale.

### Axiom E1 — Self-contained submission

**A host submits self-contained text (and optionally a
`Lookup` context — a kernel or a layered view over one).
Polydat does not reach for ambient state, global
registries-not-named-in-the-call, or thread-local context.
The submission is the input; the return is the output; there
is no third channel.**

Enforcement: the public function signatures themselves —
each is a pure function of its declared arguments + the
process-level node library (linked at build, fixed
thereafter).

### Axiom E2 — Typed result

**The returned `Value` (or each element of a returned
`Vec<Value>`) carries a declared type per T1. The host
reads the type via `Value`'s typed accessors (`as_u64`,
`as_f64`, `as_str`, `as_bool`, etc.) or via pattern matching.
There is no untyped result.**

Enforcement: T1 (every slot typed) flows through the entire
compiler pipeline; the output binding's slot is typed; the
returned value's type is the slot's declared type. The
typed `Err(EmbeddingError)` for failure modes is symmetric —
even errors are typed (one variant of the `Result`).

### Axiom E3 — Bounded determinism via the Runtime Model

**For a fixed expression text, a fixed context, and a
fixed node registry, embedded evaluation produces a
deterministic typed return value (D1), with deterministic
side channels conditional on per-node metadata (D2), and
structurally bounded cost (D3). The full mechanism — data
flow, dependency tracking, node caching, invalidation, and
the state-layering contracts that compose them — is owned
by the [Runtime Model](runtime_model.md); §5 records how
those properties specialise to embedded eval.**

Enforcement: composition of the Runtime Model's R1–R3
(memoization, lazy pull-through, forward-only flow) with
the substrate's S/T/L axioms and the Graph Compiler's
H-axioms.

### Axiom E4 — Library inheritance

**Every node in the linked node registry — polydat's
built-in library plus every host-crate registration made
with `register_nodes!` or `#[polydat_node]` — is available
to embedded expressions. The host inherits the full node
catalog — hash, arithmetic, string, math, distributions,
datetime, noise, vector ops — without declaring
per-expression node availability.**

Enforcement: the compiler reads the link-time
`NodeRegistration` inventory at compile time (§5.5). A host
crate that links a registration makes the node available to
all embedded expression evaluation in the process;
`PolydatRuntime`'s object-local factories are a separate
channel the standard surfaces do not consult.

### Axiom E5 — Lifecycle transparency

**The host chooses the evaluation depth that matches its
need: const-fold via `eval_const_expr` (the expression must
be Effectively-const), kernel-bound dynamic via
`interpolate_via_kernel` + eval (the expression sees the
kernel's bound state), or full compile via `compile_polydat`
and its engine-taking forms (the host owns the resulting
kernel for repeated evaluation). Each surface preserves the
substrate's lifecycle classification — they differ in
*which* lifecycle window they evaluate against.**

Enforcement: the surfaces are distinct entry points with
distinct contracts. `eval_const_expr` rejects expressions
that reach dynamic inputs (typed error). The two-step
interpolate-then-eval composition handles dynamic-via-
kernel cases. The compile entry points expose the full
kernel for any remaining use case.

### Axiom E6 — Composability via interpolation

**The interpolation surface (`interpolate_via_kernel`) +
evaluation surface compose. The host can use them as a
pipeline: text → interpolation → resolved text → evaluation
→ value. The composition's invariants are: interpolation
preserves text grammar (substitutions are syntactically
sound); evaluation operates on the post-interpolation text
under the same E1–E5 guarantees.**

Enforcement: interpolation's contract is text-to-text
(no semantic transformation; just placeholder replacement
via `lookup` + `Value::to_display_string`).
Evaluation's contract is text-to-Value. The two compose
naturally; the pipeline is the canonical host pattern.

### Axiom E7 — Typed error ontology

**Every failure mode the embedding surface produces is
classified into a typed `EmbeddingError` variant per the
error ontology in §6. The host pattern-matches on the
error to drive UX, recovery, or logging. `From<EmbeddingError> for String`
is a display compatibility conversion, not a second
error ontology.**

Enforcement: §6 enumerates the variants and all
standard embedding surfaces return `EmbeddingError`.

---

## 5. The Embedding System Contract

This section is the canonical reference for the contract
between polydat and host crates that embed expression
evaluation. It establishes:

- what host and polydat each provide (§5.1)
- how types cross the boundary (§5.2)
- l-value type inference at the embedding surface (§5.3)
- type-matching adapters at the boundary (§5.4)
- virtual nodes — linked registry contributions (§5.5)
- virtual wires — context-fusion-conditioned host bindings
  (§5.6)
- how the [Runtime Model] applies to embedded expressions
  specifically (§5.7)

The contract is what makes the embedding capability load-
bearing: it's not "polydat happens to be usable as an
expression engine" but "polydat and the host share a typed,
mechanised contract whose terms are explicit."

### 5.1 The contract — what host and polydat each provide

The Embedding System Contract is bidirectional and has
**two engagement levels**: a baseline contract every host
must satisfy to use the surfaces at all, and an opt-in
strict contract a host can engage for stronger
compile-time type alignment.

#### 5.1.1 Polydat's obligations

Polydat's obligations to every host, regardless of
engagement level:

| Obligation | Discharged by |
|---|---|
| Typed result | T1+T2 (substrate) → E2 |
| Deterministic typed return | D1 (runtime model) → E3 |
| Library inheritance | E4 — every registered node is available |
| Lifecycle transparency | E5 — three surfaces for three depths |
| Typed error ontology | E7 + §6's `EmbeddingError` enum |
| Forward-only data flow | R3 (runtime model) — no surprise side channels |

These are unconditional. A host using only the baseline
contract gets all of these. The strict opt-in adds
guarantees on top; it does not remove any.

#### 5.1.2 Host's baseline obligations

The minimum a host must do to use the surfaces:

| Obligation | Required surface |
|---|---|
| Self-contained text | A `&str` submitted to one of §3's surfaces |
| Context (when needed) | A `Lookup` (the interpreter kernel, or a layered view) for kernel-bound evaluation |
| Registry contributions (when needed) | Node registrations linked before evaluation |

That's it. A baseline-only host calls a surface, receives
a `Value` (or `Result<Value, EmbeddingError>`), and
handles the value however it likes — typed accessor,
pattern match, or even string-display rendering. The host
takes responsibility for any type expectations it imposes
on the result (accessor panics, mismatch handling).

A host whose predicate evaluation reaches for `.as_bool()`
post-hoc, or whose parameter evaluation reaches for
`.as_u64()`, operates at this level; it works because the
host has out-of-band knowledge of the expected type.

#### 5.1.3 Host's opt-in strict contract

A host that wants polydat to enforce type alignment at
*kernel compile time* engages additional obligations in
exchange for additional guarantees. The opt-in surface
is the typed embedding API (§5.3):

| Opt-in obligation | What polydat guarantees in return |
|---|---|
| Declare the expected return type via `eval_const_expr_typed::<T>` | Compile-time check: expression's output type matches `T` or is healable via the catalog (§5.4) |
| Use the typed accessor on the unwrapped Rust value | No accessor panic risk — the result is a Rust `T`, not a `Value` |
| Treat `TypeMismatch` errors as compile-time signals | Error variant fires at embed-call rather than at downstream use |

The opt-in is a *contract upgrade*, not a separate
contract. A host can use the baseline surfaces alongside
the opt-in surfaces in the same crate — different sites
can engage different levels.

**Why opt-in, not mandatory.** Some hosts have legitimate
reasons to operate at the baseline:

- Hosts that compose expressions whose return type
  varies across calls (e.g., a generic configuration
  evaluator that may return `U64`, `Str`, or `Bool`
  depending on the configuration key).
- Hosts that already have their own type-coercion layer
  and just want polydat's value as input.
- Hosts wrapping polydat for an interpreted-language
  binding (e.g., a Python embedding) where Rust's static
  typing isn't the boundary.

The opt-in keeps these hosts welcome at the baseline
while letting Rust-native hosts that want stricter
compile-time guarantees opt into them.

#### 5.1.4 Shared vocabulary

Both engagement levels rest on a shared vocabulary:

| Shared element | Role |
|---|---|
| `Value` enum | The carrier type for all typed return values |
| `PortType` enum | The type vocabulary for slot declarations and value classifications |
| `{name}` placeholder syntax | The textual surface for interpolation |
| The grammar | The expression-text language both produce/consume |

This shared vocabulary is the contract's *currency*. A
host wishing to speak the substrate's type system — at
baseline or strict level — uses these exact types and
syntaxes. Polydat exports them; host crates depend on
the polydat crate and import them directly. There is no
opaque value, no host-side type that polydat treats as
a black box, no syntactic surface other than what the
grammar declares.

Hosts that want to speak Polydat's `Value` type system
deeply (e.g., constructing `Value`s programmatically,
pattern-matching exhaustively, contributing virtual
nodes per §5.5 that produce typed values) are explicitly
*allowed and supported*. The substrate's type vocabulary
is public; deep host integration is a first-class
pattern, not a workaround.

### 5.2 Types at the embedding boundary

Every value crossing the boundary is typed. There is no
untyped slot, no untyped return, no untyped error in the
contract. The type vocabulary is `PortType` (declarations)
and `Value` (runtime carrier); the two are isomorphic in
the sense that every `Value` has a `port_type()` method
returning the matching `PortType` and every `PortType` has
a non-empty set of `Value` variants that satisfy it.

**Boundary type checks:**

- **Inputs (host → polydat):** the host's submitted text
  must be parseable as expression text whose result wire
  has a `PortType`. The compiler infers this from the
  expression's structure (T1+T2). If the host supplies a
  kernel context with bindings whose types are wrong for
  the slots the expression declares (e.g., slot expects
  `U64`, binding is `Str`), the boundary adapter catalog
  heals it if an adapter exists; otherwise the typed write
  is rejected and compilation of a mismatched graph emits
  `EmbeddingError::TypeMismatch`.

- **Outputs (polydat → host):** the typed `Value`
  returned to the host carries its `PortType` via the enum
  variant. The host accesses it through typed accessors
  (`Value::as_u64`, `Value::as_f64`, etc.) or
  pattern-matching. The accessors (`as_u64`, `as_bool`, …)
  panic on a type mismatch; a host that must not panic
  pattern-matches on the `Value` variant or uses the typed
  surfaces (§5.3).

- **Errors (polydat → host):** the `EmbeddingError` enum
  (§6) is itself typed — every error class is a
  discriminable variant, not a stringly-typed message.

The boundary is type-strict in both directions. The
shared `Value` / `PortType` vocabulary makes the strictness
implementable without per-call negotiation.

### 5.3 L-value type inference

The embedding surface supports both result-typed and
l-value-typed evaluation. The result-typed surface returns a
`Value`, and the host applies a typed accessor post-hoc:

```rust
let result_value = eval_const_expr("k > 5")?;
let truth = result_value.as_bool();  // post-hoc accessor
```

On this surface the host's expected type (`bool` in the
example) is not visible to polydat at compile time.

The l-value-typed surface makes that type part of the call:

```rust
let truth: bool = eval_const_expr_typed::<bool>("k > 5")?;
```

Here the type parameter `bool` drives the conversion: polydat
knows the host expects a `Value::Bool`, checks that the
expression's output `PortType` matches (or is healable to)
`Bool`, and returns a Rust `bool` (not a `Value`). A
mismatch surfaces as `EmbeddingError::TypeMismatch` from the
call, not as a runtime panic from `.as_bool()`.

The mechanism for inference:

- The host's type parameter selects a target `PortType` via
  the `HostType` trait (one impl per Rust type that has a
  natural polydat correspondence).
- The expression is evaluated as ordinary; the resulting
  value's `PortType` is compared against the target.
- If they match: return the unwrapped Rust value.
- If they mismatch but a return-path adapter exists (per
  §5.4): apply the adapter, return.
- Otherwise: typed error.

The corresponding kernel-bound entry point is
`eval_kernel_bound_typed::<T>`. The `_strict` variants reject
catalog conversions `is_lossless_adapter` classifies as
lossy. The raw-`Value` surfaces remain supported for generic
hosts and exhaustive value handling.

### 5.4 Type-matching adapters at the boundary

The substrate's T2 axiom says type mismatches between
adjacent wires are healed by auto-inserted edge adapters.
The Graph Compiler's wire resolution
([graph_compiler.md](graph_compiler.md) §5.1) implements
this via the catalog of known conversions in
[`library::convert`] and its polyfill companions:
`__u64_to_string`, `__f64_to_string`, `__u64_to_f64`,
`json_to_str`, and so on. Each catalog entry is itself a
`PolydatNode` with declared input and output `PortType`s;
the assembler inserts the appropriate adapter node when a
wire's source type differs from its consumer's expectation
in a way the catalog can heal.

#### 5.4.1 Catalog application sites

The catalog operates at three typed boundaries. The first is
**intra-graph wire validation during assembly**. The
assembler (`compile::assembly::resolve_with_log`) walks each
wire, checks the source's output `PortType` against the
consumer's input `PortType`, and:

- If they match exactly → no adapter needed.
- If they mismatch but a catalog adapter exists → insert
  the adapter as an intermediate node, rewriting the wire
  to pass through it.
- If they mismatch and no catalog adapter exists → fail
  with `AssemblyError::TypeMismatch`.

The second is **input binding**: the scope-init
materializer (`materialize_subscope`, driving
`materialize_wiring_from_outer`) and the typed host writes
(`Dataflow::set_wire`) pass every value crossing into a
slot through `adapt_boundary_value`, which consults
`boundary_adapter` — `auto_adapter` plus the boundary-only
parsers ([type_system.md](type_system.md) §6.2). When the
host's context kernel has a binding `k: F64` and the
expression's extern slot declares `k: U64`, the catalog's
`F64 → U64` adapter runs at the synthesis site; an
unhealable residual mismatch is rejected by the typed write
surface.

The third is **typed return conversion**. When the host calls
`eval_const_expr_typed::<bool>` and the expression
produces `U64`, the surface applies the catalog's
`U64 → Bool` rule (`auto_adapter`). The host's contract
receives `bool` without an accessor panic risk.

#### 5.4.2 The contract's rules for boundary adapters

Across all three sites, the rules are uniform:

- **Only catalog adapters apply.** No silent generic
  coercion. `U64` → `Str` uses the catalog's
  `__u64_to_string`; a pair with no entry surfaces as
  `TypeMismatch`. The host knows what's healable by reading
  the catalog.
- **Lossy conversions are refusable.** The `_strict` typed
  surfaces refuse a conversion `is_lossless_adapter`
  classifies as lossy and apply only the lossless ones. That
  classification is computed from the two types' numeric
  domains (`PortType::numeric_domain`), not declared per
  catalog entry: a conversion keeps the value when every
  number the source type can carry is a number the target
  type can carry. Integers fit by their magnitude bits — an
  unsigned domain needs a signed one strictly wider, and a
  signed domain never fits an unsigned one. An integer fits
  a float when its magnitude bits fit the significand, so
  `U32 → F64` is lossless and `U64 → F64` and `I64 → F64`
  are not, since 64 magnitude bits do not fit 53 and values
  above `2^53` round. Rendering to `Str` keeps the value for
  every type with a numeric domain. Everything outside that
  world — `Bytes`, `Json`, the vectors, `Ext` — is not
  claimed lossless, whatever the catalog can do with it.
  The answer is a property of the types, so it is the same
  for every input; a host that wants a number widened writes
  it at the wider type.
- **Adapter insertion is observable.** The compile event
  log records every adapter the assembler inserts
  (`TypeAdapterInserted`), on every engine; the typed
  boundary APIs report boundary errors as typed values.
- **The catalog is the single source of truth.** New
  conversion needs are added to the catalog *once*. After
  registration, the conversion is available wherever that
  catalog is the applicable boundary. `auto_adapter` governs
  graph and return-path conversions; `boundary_adapter` is
  its scope/host-boundary superset.

### 5.5 Virtual nodes — linked registry contributions

Compiler-visible node extensions use the same link-time
`NodeRegistration` inventory as Polydat's built-in library.
Host crates contribute registrations with `register_nodes!` or
the `#[polydat_node]` attribute macro. Each registration supplies static
`FuncSig` metadata, a builder, and an optional constant validator.
The standard compiler's `registry()` and `build_node()` paths
consult that inventory directly.

Once linked, contributed nodes are indistinguishable from built-ins:
they declare typed ports, compile levels, purity, commutativity, and
optional compiled or SIMD hooks through the ordinary node contract.
The slot, lifecycle, and runtime axioms apply uniformly.

`PolydatRuntime::register_factory` is a separate object-local
factory catalog. Its `registry()` and `build_from_factory()` methods
support explicit host orchestration, but the standard
`eval_const_expr` and `compile_polydat` entry points do not accept a
`PolydatRuntime` and therefore do not consult object-local factories.
A node that must be visible to those standard embedding surfaces
must use the linked inventory channel.

This is the host's node-vocabulary integration point. Virtual wires
(§5.6) extend the synthesis vocabulary instead.

### 5.6 Virtual wires — context-fusion-conditioned bindings

Where virtual nodes (§5.5) extend the *node vocabulary*,
virtual wires extend the *synthesis vocabulary*. The host
interposes at Context Fusion's slot-filling step: when
polydat's auto-extern discovers a slot the outer scope
can't satisfy from its direct bindings, the host's
resolver fires and may provide the value.

The host registers a resolver via
[`crate::dsl::factories::register_extern_resolver`]:

```rust
register_extern_resolver(Box::new(|slot_name, slot_type| {
    // Host-mediated resolution; returns Option<Value>.
    // If Some, Context Fusion fills the slot with the
    // returned value; if None, falls through to ordinary
    // resolution (typed error if no binding exists).
}));
```

A resolver can bind a name such as `cluster_metadata.region`
from host configuration when the outer kernel chain has no
matching value. At scope-init, polydat sees the extern slot,
consults registered resolvers in registration order, and uses
the first returned value. If all resolvers return `None`, normal
unresolved-slot handling applies.

#### 5.6.1 Why this is a distinct integration tier

Virtual wires differ from virtual nodes in three ways
that matter for the substrate:

- **Timing.** Virtual nodes fire at *evaluation* time
  (their `eval` runs per cycle, like any node). Virtual
  wires fire at *synthesis* time (the resolver runs at
  scope-init, the result is frozen for the scope's
  lifetime per S3).
- **Surface.** Virtual nodes appear in the expression
  *text* (the workload author writes the registered function call
  somewhere). Virtual wires appear as *bindings* the
  expression text references via `{...}` — the resolution
  is invisible to the expression author.
- **Substrate role.** Virtual nodes are consumers of the
  slot contract (they read inputs, write outputs).
  Virtual wires are *contributors* to the slot contract
  (they fill slots that auto-extern declared).

This is why virtual wires are a deeper integration point
— the host becomes a *participant* in S2 (binding-time
materialisation), not just a consumer of S1 (auto-extern
discovery). S1's binding source is the outer kernel chain or a
registered host resolver.

#### 5.6.2 Virtual-wire resolver contract

The resolver's contract must preserve every substrate
axiom for the slot it fills:

- **T1, T2 (typed slots).** The resolver MUST return a
  typed `Value` matching the slot's declared `PortType`,
  or `None` to fall through. A returned value passes
  through the boundary adapter catalog; a value of a type
  the catalog cannot heal is a typed error.
- **S3 (coordinate advance).** The resolver fires at scope-init
  time only. It cannot re-condition the slot per coordinate.
  If the host needs per-coordinate resolution, a virtual node
  (§5.5) is the correct surface.
- **L1 (each layer owns its state).** The resolver does
  not see other layers' state outside the kernel context
  passed as its argument. The context is the synthesis
  envelope at this scope-init.
- **D1 (typed-return determinism).** The resolver MUST
  be deterministic in its inputs (slot name + type +
  context). Polydat's D1 holds conditional on this; a
  non-deterministic resolver breaks D1 for any expression
  that consumes its slot.

#### 5.6.3 Composition with virtual nodes

Virtual wires fill slots; virtual nodes consume them.
The composition is uniform: a virtual node's `eval`
sees its input slots filled per the ordinary contract —
whether the slot was filled by the outer chain, by a
virtual wire resolver, or by an ordinary binding, the
node consumes a typed `Value` from the slot. The slot
contract is the abstraction barrier. The resolver is the
substrate boundary for host configuration and
external-system bindings; workload parameters remain the
explicit alternative.

### 5.7 The runtime model applied to embedded expressions

The mechanism by which embedded expressions execute — data
flow along wires, dependency tracking, per-generation node
caching, lazy pull-through invalidation, and the
determinism guarantees the runtime delivers — is the
subject of the [Runtime Model](runtime_model.md). That doc
owns the R-axioms (R1 memoization, R2 lazy pull-through,
R3 forward-only flow) and the D-axioms (D1 typed-return
determinism, D2 side-channel determinism, D3 cost
determinism). E3's guarantee is the realisation of
D1/D2/D3 at expression scale.

#### 5.7.1 The embedded expression's kernel is its own scope tier

Per the Runtime Model's L1 realisation (per-thread kernel
state), an embedded expression's kernel is its own scope
tier owned by the host call. The kernel's state is not
shared with the host's other state; the kernel's program is
an `Arc`, sharable across threads if the host caches it.

The host context (the `Lookup` it passes) is the **outer
scope** for the embedded expression. Context Fusion (per
the Graph Compiler) populates the expression kernel's
extern slots from the context kernel's bindings at
scope-init — including virtual-wire resolutions per §5.6.

#### 5.7.2 Cone size is small — cost stays small

D3 (cost determinism) gives the host a structural cost
prediction: cone size × node-eval cost per generation. For
embedded expressions, cone size is typically small —
single-digit nodes for a one-line expression, low-double-
digits for a complex predicate. This is what makes the
embedding cost predictable in practice: a host pattern of
"compile-once, evaluate-many" pays a one-time compile cost
plus per-evaluation cost bounded by a small cone.

#### 5.7.3 External-write-aware embedding patterns

Per the Composition Substrate's S4 (external-write
synthesis as the open-granularity fill path), an embedded
expression that consumes externally-written wire values
must be evaluated *after* the host has written the value
into the slot. The host pattern:

```rust
// host writes the value through the typed dataflow boundary
kernel.set_wire("recall_at_k", recall_value)?;

// NOW the expression sees the written value
let ok: bool = eval_kernel_bound_typed::<bool>("{recall_at_k} >= 0.8", &kernel)?;
```

If the host calls eval **before** the slot is written, the
slot holds `Value::None` (or the slot's default); per the
None-propagation contract, `{recall_at_k}` is an
`UnresolvedPlaceholder` at interpolation, and a `None` that
reaches a typed surface's result is `NonePropagated` (per
§6's ontology).

Hosts that consume this pattern give their write events
host-specific names — flowing one op's result values into
externally-written slots between op executions is one such
pattern — but the polydat embedding contract sees only the
generic external-write mechanism.

#### 5.7.4 Cross-host determinism — what hosts share

Two host crates evaluating the same expression text against
the same kernel context get the same typed return value
(D1). This is the load-bearing property that lets
expression evaluation be a shared utility across a
workspace: two hosts calling `eval_const_expr` on
`"{k} * 2 + 1"` with the same kernel get identical
`Value::U64`s, every time, on every thread.

D2 (side-channel determinism) is more nuanced: if the
expression includes a diagnostic node (`log_info`,
`log_debug`), the resulting log output is deterministic
per the impure node's declared semantics. Hosts that
share a sink for diagnostic output observe deterministic
emission *per node*, with combined output ordering
governed by the diagnostic node's per-eval invocation
order — which is itself deterministic from R3 (forward-
only flow along the wire chain).

For the canonical formal statement of these properties,
see [Runtime Model §7 (D-axioms)](runtime_model.md).

---

## 6. The Error Ontology

The standard embedding entry points emit seven classes of
`EmbeddingError`, reachable from parsing, compilation,
lifecycle validation, evaluation, and typed result
conversion. Each carries the context shown below; the shape
is normative, and every variant is one a host can actually
receive — the enum used to declare a `ResultMissing` and a
`Timeout` that no entry point constructed.

Each variant's fields are the compiler's own. A wiring type
mismatch carries the two node names and the two port types
the assembler resolved; an unknown function carries the
registry's nearest name; a lifecycle mismatch carries the
kernel's inputs, which are what the expression is waiting
on. None of them is rebuilt by reading an error message.

```rust
pub enum EmbeddingError {
    /// Text could not be parsed as polydat expression source.
    /// The lexer or parser rejected the input before any
    /// semantic analysis. Also the class of an interpolation
    /// that does not stabilise (cyclic placeholders).
    Parse {
        source: String,
        message: String,
        position: Option<usize>,
    },

    /// A `{name}` placeholder in the text had no matching
    /// binding in the lookup. Produced by
    /// `interpolate_via_kernel` and `evaluate_spec`.
    UnresolvedPlaceholder {
        name: String,
        source: String,
    },

    /// The expression's upstream cone reaches a dynamic
    /// input, but the requested evaluation surface requires
    /// effectively-const lifecycle. Produced by
    /// `eval_const_expr` (directly or via the two-step
    /// composition).
    LifecycleMismatch {
        source: String,
        dynamic_inputs: Vec<String>,
    },

    /// A node mentioned in the expression is not registered.
    /// `suggestion` is the nearest registered name within
    /// three edits, when there is one.
    UnknownNode {
        name: String,
        source: String,
        suggestion: Option<String>,
    },

    /// The expression's wire chain has a type mismatch that
    /// auto-adapters cannot heal (from the assembly pass), or
    /// a typed surface's target type has no adapter from the
    /// expression's result type.
    TypeMismatch {
        from_node: String,
        from_type: PortType,
        to_node: String,
        to_type: PortType,
        source: String,
    },

    /// A node's `eval` panicked during the fold. The
    /// `catch_unwind` boundary captured the panic; the message
    /// is the panic payload's human-readable form.
    NodeEvalPanic {
        node_name: String,
        message: String,
        source: String,
    },

    /// A `Value::None` reached a typed surface's result: the
    /// `HostType` conversion refuses to read an absent value
    /// as a Rust value. See none_semantics.md.
    NonePropagated {
        accessor: &'static str,
        source: String,
    },
}
```

### 6.1 Variant guide

| Variant | When it fires | Host remediation |
|---|---|---|
| `Parse` | Lexer/parser rejects the input; or interpolation does not stabilise. | Surface the parse position to the user; the input is malformed expression text. |
| `UnresolvedPlaceholder` | `{name}` has no binding in the lookup (an unset extern reads as none). | Check the kernel's inputs and outputs; suggest declaring the name as a workload param or fixing the spelling. |
| `LifecycleMismatch` | `eval_const_expr` was called on text reaching a dynamic input. | Either: (a) use the two-step interpolate-then-eval pattern to resolve dynamic names, or (b) accept the expression must be evaluated per-cycle via a compiled kernel + cycle dispatch. |
| `UnknownNode` | A node call uses a name not in the registry. | Surface `suggestion` when it is set, and point the user at the node catalog otherwise. Host crates that register custom nodes must link the registration. |
| `TypeMismatch` | Wire types incompatible and no auto-adapter exists; or the typed surface's target has no adapter. | Surface the from/to node and types; suggest an explicit conversion (`x as str`, `format_u64(x)`, `to_f64(x)`) or a different node. |
| `NodeEvalPanic` | A node panicked during the fold. | Surface the panic message; this is typically a node-internal contract violation (invalid argument range, etc.). Forwarded to the user with provenance. |
| `NonePropagated` | A typed surface's result was `Value::None`. | Use the raw-`Value` surface and handle the None, or surface it to the user with context about which input was missing. See [none_semantics.md](none_semantics.md). |

### 6.2 Provenance

Every variant carries the source text that produced the
error. Host crates that wrap embedded evaluation should
*also* record (a) the file or YAML key the source text
came from, and (b) the calling host context (e.g.,
"workload `query` op, field `where`"). Together these
form the diagnostic chain: polydat owns the polydat-layer
error variant; the host owns the host-layer location and
naming.


## 7. The composition pattern

The canonical host pattern for kernel-bound evaluation:

```rust
use polydat::kernel::interp::{interpolate_via_kernel, Lookup};
use polydat::dsl::compile::{eval_const_expr, EmbeddingError};

fn evaluate_predicate(
    text: &str, kernel: &dyn Lookup,
) -> Result<bool, EmbeddingError> {
    let resolved = interpolate_via_kernel(text, kernel)?;
    let value = eval_const_expr(&resolved)?;
    Ok(value.as_bool())
}
```

The two-step composition has these properties:

- **Interpolation is text-preserving.** `{name}` becomes the
  display string of `kernel.lookup(name)`. The
  post-interpolation text remains grammatically valid as a
  polydat expression.
- **Evaluation is interpolation-agnostic.** `eval_const_expr`
  doesn't know the text was interpolated; it just compiles
  what it receives.
- **Lifecycle gating moves to the eval step.** If the
  post-interpolation text still references a dynamic input
  (e.g., interpolation didn't substitute everything, or the
  remaining names are dynamic-bound), eval returns a clean
  error.
- **The two steps are reusable independently.** A host that
  wants raw interpolation (text → text) calls just the
  first; a host with already-resolved text calls just the
  second.

The pattern is what gives the host the *full* expressive
range without compromising the substrate's deterministic-
evaluation guarantee. `eval_kernel_bound_typed::<bool>` is
this pattern with the typed return conversion of §5.3, on
the interpreter's kernel.

---

## 8. Surface boundaries

- Evaluation APIs compile eagerly. Callers that need reuse
  compile a kernel and cache the resulting program.
- Each embedding call accepts one expression. Hosts batch by
  compiling a graph with multiple named outputs or by managing
  a collection of cached programs.
- Source provenance is carried by the existing source and
  `EmbeddingError` fields. There is no separate `HostText`
  wrapper in the embedding contract.

---

[`crate::ast`]: ../../../polydat-core/src/ast.rs
[`crate::kernel`]: ../../../polydat-core/src/kernel/mod.rs
[`crate::dsl::compile::eval_const_expr`]: ../../../polydat-core/src/dsl/compile.rs
[`crate::dsl::compile::compile_polydat`]: ../../../polydat-core/src/dsl/compile.rs
[`crate::dsl::factories::register_extern_resolver`]: ../../../polydat-core/src/dsl/factories.rs
[`crate::kernel::interp::interpolate_via_kernel`]: ../../../polydat-core/src/kernel/interp.rs
[`crate::iteration::comprehension::eval::evaluate_spec`]: ../../../polydat-core/src/iteration/comprehension/eval.rs
[`library::convert`]: ../../../polydat-core/src/library/convert.rs
[Runtime Model]: runtime_model.md
