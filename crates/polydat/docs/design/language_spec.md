# Language Spec

The detailed specification of the Polydat DSL surface: syntax
productions, type system, node contract, wiring model, and
compilation pipeline. This doc is the mechanism-level
companion to [grammar.md](grammar.md), which states the
axioms (G1-G6); read grammar.md first for the formal
contract; come here for the operator catalog, precedence
tables, type-system enum, the `PolydatNode` trait surface, and
the pipeline-stage breakdown.

This doc extends axiom-level statements across multiple
substrate docs:
- [grammar.md §2 productions + §3 type rules + G2 const lifecycle](grammar.md)
- [composition_substrate.md §2 slot contract + T1 typed return + L2 effectively-const](composition_substrate.md)
- [graph_compiler.md §2 pipeline + §5 Node Fusion catalog + §6 ordered composition](graph_compiler.md)
- [runtime_model.md R2 hybrid push/pull invalidation](runtime_model.md)
- [expression_engine.md §3.1 const expression evaluation + §5 embedding contract](expression_engine.md)

The host-side framing (why a host uses Polydat as its unified
access surface, output selection, Polydat as unified state
holder, op-level bindings, cursor declarations) is documented
host-side.

---

## DSL Syntax

Polydat programs are written in `.polydat` files or inline in workload
`bindings:` blocks.

### Input Declaration

```
input cycle: u64
input (cycle: u64, partition: u64, cluster: u64)
```

Inputs are the external values that drive the DAG. A workload
declares any cursor names it wants as inputs; the compiler
treats them as unbound wires that the host must supply at
runtime.

**Inputs are inferred when the declaration is omitted.** A
binding block that references `cycle` (or any other unbound
name) implicitly declares those names as its input set —
the compiler's closure inference already identifies unbound
wires on both the input and output sides, so requiring an
explicit `input ...: u64` line in every block was redundant.
Strict checking still applies: the host closure feeding the
kernel must provide every inferred input, and the compiler
reports a mismatch if it doesn't.

**`cycle` is not a magic identifier.** It's a conventional
name for the primary cursor — common in examples because it
matches the default cursor the runner supplies — but inputs
can be named anything and any cursor shape (single, nested,
decomposed via `mixed_radix`) is fine. The engine treats
`cycle` identically to any other user-named input.

> **Note:** `input` is the only keyword; there is no `coordinates`
> form. The program still counts its leading coordinate inputs as
> `coord_count`; that is an implementation detail and doesn't surface
> in user-visible source or error messages.

### Coordinate Decomposition

Most workloads use a single `cycle` input. Multi-dimensional
iteration is modeled inside the Polydat via mixed_radix decomposition:

    input cycle: u64
    (row, col) := mixed_radix(cycle, 1000, 1000)

This keeps the activity executor simple (it only passes `[cycle]`)
while enabling N-dimensional access patterns within the DAG.
Decomposed coordinates are ordinary Polydat wires — they can feed into
hash, interleave, mod, or any other node. Any traversal strategy
(nested loop, strided, random) is expressed as Polydat nodes rather
than activity-layer configuration, keeping domain logic in one place.

### Bindings

```
// Cycle-time binding (evaluated per cycle)
user_id := mod(hash(cycle), 1000000)

// Init-time constant (evaluated once, folded into DAG)
dim := vector_dim("glove-25-angular")

// Function composition (output of one feeds input of next)
hashed := hash(cycle)
bucket := mod(hashed, 100)
name := weighted_strings(bucket, "alice:0.3;bob:0.3;carol:0.4")
```

### String Interpolation

```
email := "{format_u64(hash(cycle), 10)}@example.com"
query := "SELECT * FROM {keyspace}.{table} WHERE id = {user_id}"
sum   := "x + y = {x + y}"
slot  := "row {row.ordinal}"
```

The body inside each `{ … }` is parsed as a full Polydat expression
— bare identifiers, function calls, infix arithmetic, and field
access all work, exactly the same as on the right-hand side of
any binding. The compiler:

1. Splits the literal into segments: literal text + placeholder
   bodies. The scan is brace-aware (parens / brackets nest)
   and string-aware (a `}` inside a `"…"` doesn't terminate the
   placeholder).
2. Lexes and parses each placeholder body via the same
   expression parser the rest of the language uses.
3. Emits a `printf(fmt, expr1, expr2, ...)` call where `fmt`
   is the literal segments joined by `{}` placeholders, in the
   same positional order as `expr1, expr2, ...`.

`{{` and `}}` are printf's own escapes for emitting literal
braces — they keep their meaning and don't open a placeholder.
A printf format spec the user wrote by hand
(`"x={:05}"`, `"{0:.3}"`) isn't a valid Polydat expression, so the
literal stays unchanged and the user's format spec reaches
printf intact. An unbalanced `{` likewise leaves the literal
alone.

This is pure syntactic sugar — no special runtime support is
needed beyond the standard `printf` node. Iteration variables
that appear inside string literals
(`vector_dim("{dataset}:{profile}")`) flow through the same
wire mechanism as any other identifier reference: they're
declared as `extern` ports on the scope, the runner sets them
per iteration, and the dataset function reads its `source`
input wire at eval time.

### Comments

```
// Line comment
/// Doc comment (markdown, attached to next binding)
/* Block comment */
```

Line comments (`//`) for inline annotations. Triple-slash (`///`)
for documentation comments in markdown format, attached to the
following binding — these are extractable by tooling for
auto-generated documentation. Block comments (`/* ... */`) for
temporarily disabling sections.

### Infix Operators

Polydat supports arithmetic, bitwise, comparison, and power
operators with standard precedence. Operators desugar to
function calls in the DAG — `a + b` becomes `u64_add(a, b)` or
`f64_add(a, b)` by operand type, `a & b` becomes `u64_and(a, b)`,
`a < b` becomes `u64_lt(a, b)` or `f64_lt(a, b)`.

```
// Arithmetic (f64)
wave := sin(to_f64(cycle) * 0.1)
scaled := (x + 1.0) / 2.0

// Bitwise (u64)
low_byte := hash(cycle) & 0xFF
flags := (region << 48) | (tenant << 32) | sequence
masked := hash(cycle) ^ 0xDEADBEEF

// Power
decay := amplitude ** 0.5

// Comparisons (yield u64 truth: 0 or 1)
hot   := err_rate > 0.05
exact := flags == 0
```

**Precedence** (lowest to highest, follows Rust):

| Level | Operators | Associativity |
|-------|-----------|---------------|
| 1 | `==` `!=` (equality) | left |
| 2 | `<` `>` `<=` `>=` (relational) | left |
| 3 | `\|` (bitwise OR) | left |
| 4 | `^` (bitwise XOR) | left |
| 5 | `&` (bitwise AND) | left |
| 6 | `<<` `>>` (shifts) | left |
| 7 | `+` `-` (add/sub) | left |
| 8 | `*` `/` `%` (mul/div/mod) | left |
| 9 | `**` (power) | right |
| 10 | `-` `!` (unary neg/not) | prefix |

Parentheses override precedence: `(a + b) * c`. Comparison
binds looser than arithmetic and bitwise, so `a + b < c * d`
parses as `(a + b) < (c * d)`. Equality is below relational, so
`a < b == c` parses as `(a < b) == c`.

**Operator → Node mapping:**

| Operator | Node function |
|----------|--------------|
| `+` `-` `*` `/` | `u64_add`, `u64_sub`, `u64_mul`, `u64_div` when both operands are u64; otherwise `f64_add`, `f64_sub`, `f64_mul`, `f64_div` |
| `%` | `u64_mod` or `f64_mod` by operand type |
| `**` | `pow` |
| `&` `\|` `^` | `u64_and`, `u64_or`, `u64_xor` |
| `<<` `>>` | `u64_shl`, `u64_shr` |
| `==` `!=` | `u64_eq` / `u64_ne` (or `f64_*` if either operand is f64) |
| `<` `>` `<=` `>=` | `u64_lt` / `u64_gt` / `u64_le` / `u64_ge` (or `f64_*`) |
| `!` (prefix) | `u64_not` |
| `-` (prefix) | `f64_sub(0.0, x)` |

Comparison results are always `u64` truth values (0 = false,
1 = true) regardless of operand types — they compose cleanly
with bitwise operators (`a < b & c < d`) and with the `if(...)`
intrinsic below.

### Conditional Selection — `if(cond, a, b)` and `if cond { a } else { b }`

`if` is a compiler intrinsic, not a registered function: at
compile time it desugars to `select_u64(cond, a, b)`,
`select_f64(cond, a, b)`, or `select_str(cond, a, b)` based on
the inferred types of `a` and `b`. When one branch is u64 and the other f64, the u64
branch is auto-widened via `to_f64`. The condition is u64 —
any nonzero value selects `a`, zero selects `b`.

```
// Step throttle: above a 5%-error threshold the multiplier
// drops to 0.5; below it sits at 1.05.
factor := if(err_rate > 0.05, 0.5, 1.05)

// Mixed branches widen automatically (cycle is u64, 100.0 is f64).
default := if(cycle == 0, 100.0, to_f64(cycle))

// Pure u64 stays u64.
clamped := if(x > 1000, 1000, x)
```

Both branches are *always* evaluated — there is no short-
circuit. `if` is an expression-level select, not a control-flow
construct. Side-effecting nodes inside an unselected branch
still run (they're part of the DAG); design accordingly.

#### Block form

The same intrinsic also accepts a block spelling, which reads
better when the branches are themselves expressions:

```
// Identical to if(segments > 0, total / segments, 0)
mean := if segments > 0 { total / segments } else { 0 }

// `else if` chains; right-associative, so this is
// if(a, 1, if(b, 2, 3))
tier := if a { 1 } else if b { 2 } else { 3 }

// It is an expression, so it composes like one.
scaled := 1 + if hot { 10 } else { 0 }
```

This is **sugar only**: the parser rewrites it to
`if(cond, a, b)` before anything else sees it, in the same way
`a + b` becomes `u64_add(a, b)`. Branch-type dispatch, u64→f64
widening, and the `select_*` node it compiles to are therefore
identical between the two spellings — there is one construct
here, with two ways to write it. Pretty-printed output shows
the canonical call form.

Two consequences of that, both inherited rather than special to
the block form:

* **`else` is mandatory.** A Polydat expression always produces
  a value and there is no unit type, so a one-armed `if` would
  have no result on the false path.
* **The braces are not a guard.** Because both branches always
  evaluate, `if n > 0 { total / n } else { 0 }` does *not*
  protect the division — write `total / max(n, 1)` instead. The
  block form looks imperative; the semantics remain dataflow.

`if` stays a *soft* keyword: `if(` is still parsed as the call
form, so existing kernels are unaffected and a wire may still
be named `if` in positions where no block can follow.

### Literal Promotion

Literal values in wire positions are automatically promoted to
constant nodes. This means function calls with mixed wire and
literal arguments work naturally:

```
// All equivalent:
exp := 2.0
out := pow(x, exp)

out := pow(x, 2.0)   // 2.0 auto-promoted to ConstF64 node
```

The compiler inserts anonymous `ConstF64(2.0)` nodes for literals
in wire positions. These nodes are constant-folded at compile time,
so there is no runtime cost.

### Type Inference and Auto-Widening

Infix operators select the correct function variant based on
operand types:

```
cycle * 2          → u64_mul (both u64)
to_f64(cycle) * 0.5  → f64_mul (both f64)
cycle * 0.5        → f64_mul (cycle auto-widened to f64)
hash(cycle) & 0xFF → u64_and (bitwise always u64)
```

When operands have different types, the compiler auto-widens
the narrower operand (u64 → f64 via `to_f64`). This is a safe,
lossless conversion. The compiler emits an advisory event:

```
polydat[advisory]: widening u64 → f64 in operator *
```

### Auto-Conversion to String

Auto-adapters are one half of the compiler's type-and-value
wiring story. The other half is assertion nodes: runtime guards
the compiler can splice in when it can't *prove* a wire already
satisfies a downstream node's contract. Both are invisible to the
module author — adapters handle type coercion, assertions handle
value validity — and both are skipped whenever the static type
system already proves the wire is safe. The host's input-validity
model defines the full two-layer design (unsafe-by-default fast path,
opt-in strict wire guards, const constraint metadata, type and
value assertion families).

When a non-string value feeds a string wire input, the compiler
auto-inserts a conversion adapter:

| From | To | Adapter |
|------|----|---------|
| u64 | String | `__u64_to_string` (decimal) |
| f64 | String | `__f64_to_string` |
| bool | String | `__bool_to_str` ("true"/"false") |
| JSON | String | `json_to_str` (compact JSON) |

These are inserted transparently. The compiler emits an
advisory event for each insertion, queryable with
`polydat explain <file> types`.

### Compiler Diagnostics

The compiler emits tagged diagnostic events at three levels:

| Level | Tag | Meaning |
|-------|-----|---------|
| Info | `polydat[info]` | Normal compilation steps |
| Advisory | `polydat[advisory]` | Implicit conversions, type widenings — review for module design quality |
| Warning | `polydat[warning]` | Potential performance or correctness issues |

Query advisories with `polydat explain <file> types` to review all
implicit conversions in your module:

```bash
polydat explain mymodule.polydat types
# Shows: polydat[advisory]: type adapter U64→F64: cycle → sin
# Shows: polydat[advisory]: widening u64 → f64 in operator *
```

---

## Bitwise Operations

Polydat provides six u64 bitwise node functions. Applying bitwise
operators to f64 operands is a compile-time error.

| Node | Signature | Description |
|------|-----------|-------------|
| `u64_and` | `u64, u64 → u64` | bitwise AND |
| `u64_or` | `u64, u64 → u64` | bitwise OR |
| `u64_xor` | `u64, u64 → u64` | bitwise XOR |
| `u64_shl` | `u64, u64 → u64` | left shift |
| `u64_shr` | `u64, u64 → u64` | logical right shift |
| `u64_not` | `u64 → u64` | bitwise complement |

```
// Mask the low byte
low_byte := u64_and(hash(cycle), 0xFF)

// Pack fields into a single u64
packed := u64_or(u64_shl(region, 48), u64_shl(tenant, 32))

// Flip bits deterministically
flipped := u64_xor(hash(cycle), 0xDEADBEEF)

// Complement (infix: !x)
inv := u64_not(flags)
```

Infix operator `&` desugars to `u64_and`, `|` to `u64_or`,
`^` to `u64_xor`, `<<` to `u64_shl`, `>>` to `u64_shr`,
prefix `!` to `u64_not`.

---

## Const Expressions at the Host Boundary

Braces are not a Polydat expression form: in Polydat source, `{name}`
has meaning only inside a string literal, as interpolation. A host
whose config fields carry brace-delimited expressions (for example
`dim: {vector_dim("glove-25-angular")}` in a YAML workload) evaluates
the inner text itself with `eval_const_expr` /
`eval_const_expr_for`, per [Evaluation Model](evaluation_model.md)'s
compile-const lifecycle. The const-evaluation API and embedding
mechanics are formalised in
[expression_engine.md §3.1](../design/expression_engine.md);
the host-side resolution order and param-substitution
interaction are a host concern.

---

## Type Inference Details

The compiler selects operator variants according to this
dispatch table:

| Left operand | Right operand | Operator | Selected variant |
|-------------|--------------|----------|-----------------|
| u64 | u64 | `+` `-` `*` `/` `%` | u64 variant |
| f64 | f64 | `+` `-` `*` `/` `%` | f64 variant |
| u64 | f64 | `+` `-` `*` `/` `%` | u64 auto-widened → f64, f64 variant |
| f64 | u64 | `+` `-` `*` `/` `%` | u64 auto-widened → f64, f64 variant |
| any | any | `**` | always f64 (`pow`) |
| u64 | u64 | `&` `\|` `^` `<<` `>>` | always u64 |
| f64 | any | `&` `\|` `^` `<<` `>>` `!` | **compile error** |

Auto-widening inserts an implicit `to_f64` adapter and emits
a `polydat[advisory]` diagnostic. Narrowing (f64 → u64) is never
implicit — use an explicit cast function.

### The `as` cast

`expr as type` declares the type a wire should have and inserts
the adapter that gets it there. When the expression already has
the type, the cast is a no-op passthrough. Otherwise it is
exactly the adapter the assembler would insert between a wire of
the expression's type and a port of the target type, from the
whole adapter catalog: every lossless widening, every register
retag, the string-to-number parses (`"42" as u64` is a declared
reading, not a narrowing), and the JSON and byte-string
conversions the catalog defines. Two things `as` never does:
narrow a float to an integer, because the rounding is a semantic
choice the author must make by name (`f64_to_u64`,
`round_to_u64`, `floor_to_u64`, `ceil_to_u64`), and invent a
conversion the catalog lacks, which is a compile error naming
both types. A hole's declared type in a tile (`${x: u64}`) is the
same cast, so a tile can reach every catalog adapter a binding
can.

---

## Compilation Pipeline

```
Source text
  │
  ▼
Parse ─────────▶ AST (assignments, function calls, wiring)
  │
  ▼
Desugar ───────▶ Normalize sugar forms:
  │               - String interpolation → `printf` calls
  │               - Inline nesting → auto-named intermediates
  │               - Bare {name} → wire references
  │
  ▼
Wire Resolution ▶ Map names to node outputs, input indices,
  │               or external ports
  │
  ▼
Type Inference ─▶ Validate port types match wiring.
  │               Insert auto-adapters (u64→f64, etc.)
  │
  ▼
Topological Sort ▶ Determine evaluation order
  │
  ▼
Output Selection ▶ Mark which nodes are outputs (referenced by
  │               op fields, params, or extra bindings)
  │
  ▼
Constant Folding ▶ Evaluate compile-const nodes (no
  │               extern / cycle-input dependency), replace
  │               with leaf const nodes — see
  │               [Evaluation Model](evaluation_model.md)
  │
  ▼
PolydatProgram ─▶ Immutable compiled DAG (shared via Arc)
```

The Output Selection step's host-facing details — which op
fields, params, and extra bindings count as output consumers —
are a host concern, documented by the application that embeds polydat.

---

## Type System

Runtime values use the `Value` enum. Its carrier families are:

- `U64`, `I64`, and `F64` for scalar machine values and the
  bit-stuffed narrow scalar representations;
- two-limb `U128` and `I128` values;
- `Reg128(Bits128, RegLanes)` for raw and homogeneous 128-bit
  register views;
- `Bool`, shared `Str`, shared `Bytes`, and shared `Json` values;
- reflected `Ext` values and type-erased shared `Handle` resources;
- `VecF16`, `VecF32`, `VecF64`, `VecI8`, `VecI16`, `VecI32`, and
  `VecI64` typed slice carriers; and
- `None`, the absent or uninitialized sentinel.

The exact static-to-runtime representation is specified in
[type_system.md](type_system.md) and
[type_system_alignment.md](type_system_alignment.md).

Nodes declare their port types via `NodeMeta`. The compiler
inserts type adapter nodes where wiring crosses types (e.g.,
`u64 → f64` auto-conversion). Type mismatches that can't be
adapted are compile-time errors.

Type keywords are `u64`, `i64`, `f64`, `bool`, `str`, `bytes`,
`json`, `ext`, `handle`, and the `vec_*`/`reg_*` families;
`String` is accepted as an alias of `str`. The internal `Value`
enum mirrors these types directly, avoiding any mapping layer.

`Handle` is the typed-resource carrier (`PortType::Handle`):
an `Arc<dyn Any + Send + Sync>` produced by resolver nodes
(e.g., `dataset_open`) and consumed by reader nodes that
downcast to the concrete resource type. Cloning a `Value::Handle`
during input-gather is one `Arc::clone` (atomic increment,
zero allocations) — the design that lets resolved resources
flow on wires between scope-stable resolvers (compile-const or
scope-init) and per-cycle readers without re-doing the
resolution work. See
[Evaluation Model](evaluation_model.md) §"Two
Evaluation Lifecycles" for the lifecycle taxonomy; the host's
dataset-handle surface is the canonical use case.

`VecF32` / `VecI32` are typed-vector carriers
(`PortType::VecF32`, `PortType::VecI32`) — `Arc<[f32]>` and
`Arc<[i32]>` respectively. They flow on wires the same as any
other value. Cloning is one `Arc::clone`, zero allocations.
The `to_display_string()` fallback renders
them as JSON-array text (`"[0.1,0.2,...]"`), so workloads can
mix typed-vector and string-substitution paths without a
separate node family. (Adapter-side native-vector binding is a
host concern.)

---

## Node Contract

Every node implements `PolydatNode` (defined in `polydat-core/src/ast.rs`).
The trait's behavioral surface is:

```rust
pub trait PolydatNode: Send + Sync {
    fn meta(&self) -> &NodeMeta;
    fn eval(&self, inputs: &[Value], outputs: &mut [Value]);
    fn scratch_layout(&self) -> Vec<ScratchElem> { Vec::new() }
    fn eval_in(&self, scratch: &mut [ScratchBuf], inputs: &[Value], outputs: &mut [Value]) { self.eval(inputs, outputs) }
    fn commutativity(&self) -> Commutativity { Commutativity::Positional }
    fn accepts_none_inputs(&self) -> bool { false }
    fn compiled_u64(&self) -> Option<CompiledU64Op> { None }
    fn compiled_slot(&self, wire_types: &[PortType]) -> Option<CompiledSlotKit> { None }
    fn jit_constants(&self) -> Vec<u64> { Vec::new() }
    fn purity(&self) -> Purity { Purity::Pure }
    fn simd_variant(&self) -> Option<SimdVariant> { None }
    fn fusion_subgraph(&self) -> Option<FusionSubgraph<'_>> { None }
}
```

`NodeMeta` declares:
- `name: String` — function name for DSL and diagnostics
- `ins: Vec<Slot>` — input port names and types (Slot::Wire or Slot::Const)
- `outs: Vec<Port>` — output port names and types

Nodes default to `Purity::Pure`; nodes with observable
side channels (logging, file I/O) override `purity()` per
the substrate's D2 axiom. The slot contract is formalised
in
[polydat composition_substrate.md §2](../design/composition_substrate.md).

---

## Wiring Model

Conceptually, the immutable program stores the DAG as parallel
vectors plus typed input and ordered-output metadata:

```rust
pub struct PolydatProgram {
    nodes: Vec<Box<dyn PolydatNode>>, // node instances
    wiring: Vec<Vec<WireSource>>,     // per-node input sources
    input_defs: Vec<InputDef>,         // typed inputs and lifecycle classes
    coord_count: usize,                // leading coordinate-input prefix
    output_map: HashMap<String, (usize, usize)>,  // name → (node, port)
    output_list: Vec<(String, usize, usize)>,     // declaration order
}

pub enum WireSource {
    Input(usize),               // coordinate, iteration, or external input
    NodeOutput(usize, usize),   // input from (node_index, port_index)
}
```

Input lifecycle is carried separately by `InputDef::kind` as
`Coordinate`, `IterationExtern`, or `ExternalWrite`; it is not a
third `WireSource` variant.

Evaluation proceeds in topological order. Each node reads inputs
from upstream node output buffers or graph input values, and writes
to its own output buffer slots in `PolydatState`.

---

## Incremental Invalidation

Input mutation uses provenance-based invalidation. The compiled
program records the transitive dependent set for each input.
`set_input` and `set_inputs` dirty that set unconditionally,
including on same-value writes. Pull evaluation then recomputes
only dirty nodes in the requested output cone; clean shared
intermediates remain cached.

This hybrid push/pull rule applies equally to linear chains,
diamonds, and general acyclic graphs. A diamond join is dirtied
when the changed input is in its provenance and remains clean for
changes confined to an unrelated cone. The normative runtime
contract and its invariants are in
[The Runtime Model §3–§4](runtime_model.md).

---

## Polydat Scope Model

Polydat programs exist within a scope hierarchy (root program,
`for` traversal bodies, module bodies). Each scope is a
self-contained kernel that sees its outer scopes' values via
auto-generated `extern` input slots. The full model — scope
hierarchy, visibility and mutability rules, input lifecycle
classes, and the auto-extern composition mechanism — is
specified in
[scope_model.md](scope_model.md), with axiom-level
coverage in
[composition_substrate.md](composition_substrate.md).

The language-level surface that intersects scopes — op-level
bindings (which are syntactic sugar, not new scopes) and
cursor declarations — is documented host-side, by the
application that embeds polydat.
