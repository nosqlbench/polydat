# Input variance: converter nodes, not healing writes

Status: implemented 2026-09-24, except where §11 says otherwise.
Supersedes the runtime healing paths described in
[composition_substrate.md](composition_substrate.md) (the typed-write
paragraph and Axiom T2's second site) and [type_system.md](type_system.md)
§6.2; see §8 and §11.

## 1. The problem

An input slot has one declared type, and every engine refuses a write
that does not satisfy it (engines.md §3.5, `WriteError::TypeMismatch`).
That is the right rule for almost every program: a host that writes the
wrong type has a bug, and the refusal names it at the write.

Some embedders legitimately write values whose type varies over a
kernel's lifetime. A workload parameter arrives as text from a command
line in one run and as a number from a config file in the next; a scope
binder copies a parent's value into a child whose slot the child's own
program typed differently. Polydat meets these today with three runtime
paths that convert a value at the write, out of sight of the graph:

- the interpreter's public `Dataflow::set_wire` / `set_wire_idx`, which
  runs the boundary adapter catalog, including the entries that can
  panic, before the typed check, and bypasses the coordinate refusal;
- the scope binder's value copies (`adapt_boundary_value`), which heal
  what the catalog heals and drop the rest silently, since the copy's
  `set_input_at` result is discarded;
- the write-through commit, which heals a fixed list of widenings.

Each contradicts the rule every other write follows, none is visible in
the program, none is reported, and the second loses a value without a
word. The engines documentation says a write is "never healed", and the
code says otherwise.

## 2. The rule

Conversion of a varying input is a node in the program, placed at
assembly, reported, and never implicit.

- **A declared input is invariant.** `input x: T`, `extern x: T`, and
  `shared x: T := …` declare `x`'s type for the kernel's lifetime. A
  write must satisfy it, on every engine, and a mismatch is refused at
  the write. No path converts it.
- **An undeclared input is open.** An `input x` with no type, an
  auto-extern (SRD-74 P2), an extern the subcontext builder synthesizes,
  and an inferred coordinate have types the compiler chose, not the
  author. Their type is an inference, and a host may reasonably write
  something else.
- **An open input is converted only when the host asks.** The kernel's
  configuration names what the compiler does with open inputs (§4). By
  default it does what it does today: the inferred type is the slot's
  type and a mismatched write is refused.
- **A converter is a node.** When conversion is asked for, the compiler
  makes the slot accept any value and inserts a converter node between
  the slot and every consumer. The node converts to the type the
  consumers read, through the boundary adapter catalog, and a value the
  catalog cannot convert is that node's failure, attributed to it like
  any node's (engines.md §3.4). Nothing converts at the write.

What changes for a host is therefore visible in three places: the
configuration that asked for conversion, the compile log that lists
every converter it placed, and the program, where each converter is a
node a pull runs.

## 3. Declared and inferred types

The assembler records how each input's type was established:

```rust
pub enum TypeOrigin {
    /// The author wrote the type.
    Declared,
    /// The compiler inferred it: an untyped `input`, an auto-extern's
    /// right-hand side, a synthesized extern, an inferred coordinate.
    Inferred,
}
```

`InputDef` gains `type_origin: TypeOrigin`. The grammar already keeps
the distinction for `input` (`InputDecl::ty` is optional) and for
auto-externs (the slot type comes from the compiled right-hand side);
today it is dropped at assembly. A synthesized extern the subcontext
builder writes as an `ExternPort` statement is `Inferred` even though
it is spelled like a declaration, since its type is a collapse of the
parent's (`port_type_keyword`) and not the author's choice.

`Kernel::input_type_origin(name) -> Option<TypeOrigin>` exposes it, so a
host can see which of its inputs are open before it writes.

## 4. The configuration

`CompileOptions` gains one field:

```rust
pub enum InputVariance {
    /// Open inputs take their inferred type; a mismatched write is
    /// refused. Today's behavior, and the default.
    Fixed,
    /// Every open input is a construction error naming it: the program
    /// must declare every input's type.
    Error,
    /// Every open input gets a converter node; each placement is a
    /// `Warning` event in the compile log.
    Warn,
    /// Every open input gets a converter node; each placement is an
    /// `Info` event.
    Info,
}

pub input_variance: InputVariance, // default Fixed
```

The subcontext builder's `CompileOptions` carries the same field and
passes it through, so a scope tree has one setting for all its scopes.

The three non-default settings channel the one situation, an input whose
type the author did not declare, into the three severities a host can
act on:

- `Error` stops construction. `KernelError::OpenInputs` names every open
  input and where its type came from, so a host that wants every type
  pinned finds out at build, not at the first write.
- `Warn` places converters and reports each one as a warning. This is
  the setting for an embedder that knows its writes vary and wants each
  place it relies on that to stay visible.
- `Info` places converters quietly, for an embedder that has audited
  them.

`strict` is independent: it refuses implicit coercions between nodes
and does not decide what happens at inputs.

Each placement is one event:

```rust
CompileEvent::InputConverterInserted {
    input: String,
    /// The type the consumers read.
    to: PortType,
    /// Why the input is open.
    origin: OpenInputOrigin, // Untyped | AutoExtern | Synthesized | Coordinate
    node: String,            // the converter's node name
}
```

Its level is the configured one. `CompileEvent::level()` is fixed per
variant today; this variant takes its level from the options, which is
the first event to do so, and the log's `warnings()` and `advisories()`
read it the same way.

## 5. The converter

The slot of a converted input is typed `PortType::Dyn`: it holds a
`Value` as written, `None` included, and a `Dyn` slot is readable only
by a node whose port accepts any type. The converter, `__convert_<name>`,
is such a node: one `Dyn` input, one output of the consumers' type,
whose body is the boundary adapter catalog applied to the value (§6.1 of
type_system.md for the catalog, class B included).

- If the consumers read more than one type, there is one converter per
  type, each named for its target.
- A `None` passes through as `None`, so SRD-74 propagation is unchanged.
- A value already of the target type passes through; a value the catalog
  cannot convert fails the converter, and the failure names the input,
  the value's type, and the target.

A converter is a closure step on every compiled engine: native code
cannot carry a `Dyn` value. It runs once per write of its input, not per
pull, since its input is current until the next write (R1). Every node
downstream of it is typed and lowers as before. This is the cost of
variance, and the reason it is asked for rather than assumed: a declared
input costs nothing, and an open one under `Fixed` costs nothing.

## 6. The host side

A host has three honest ways to write a value that may not match:

1. **Declare the type and convert before writing.** When the host knows
   the target type, it converts the value itself and writes the result.
   The conversion is the catalog's, exposed as
   `polydat::convert::to_port(value, PortType) -> Result<Value, ConvertError>`,
   so host-side conversion agrees with the converter node. This is the
   trusted surface: a host that converts through it gets exactly the
   conversions a converter node would make, and an error it can handle
   at the write instead of a node failure at the pull.
2. **Insert a converter as a transform.** A host that wants one input
   converted without opening every undeclared input rewrites the program
   before compiling: `transform::convert_input(&mut file, "x")` declares
   `x` as `Dyn` and inserts its converter, exactly as §5 describes,
   reported as `InputConverterInserted` at `Info`.
3. **Open the inputs the author left untyped.** Set `input_variance` to
   `Warn` or `Info` and write what arrives; the compile log lists what
   was opened.

What a host must not do is rely on a write that converts. None does,
after §8.

## 7. Scope trees

A scope binder copies a parent's values into a child's inputs (the
subcontext wiring; subcontext_construction.md). Under this design the
copy is a write like any other:

- A child input the child's program declared refuses a parent value of
  another type, and the binder reports the refusal as a construction
  error naming the parent output, the child input, and both types,
  instead of discarding it.
- A child input that is open and converted (the child was compiled with
  `Warn` or `Info`) takes the parent's value as written, and the child's
  converter converts it.
- A child built from source under a parent (`ParentView`) can do better
  than either: the builder knows the parent's types, so an input it
  synthesizes for a parent value takes the parent's exact type and needs
  no converter at all. The collapse in `port_type_keyword` goes.

## 8. Changes to existing paths and specs

When this lands:

- `Dataflow::set_wire` / `set_wire_idx` are deprecated for one minor
  version, then removed. Their callers write through `set_input_at`,
  converting first with `convert::to_port` if they must.
- `adapt_boundary_value` stops being a write-time path. The binder's
  copies go through `set_input_at` and report a refusal (§7). The
  function becomes the body of the converter node and of
  `convert::to_port`.
- The write-through commit's widening list becomes a declared rule:
  a shared cell's type is its declaration's, and a write-through of a
  narrower numeric type is converted by the catalog's lossless
  widenings only, reported once per binding at build.
- composition_substrate.md's typed-write paragraph and Axiom T2 name
  converter nodes as the only runtime conversion; type_system.md §6.2 is
  rewritten to the converter node and `convert::to_port`.
- engines.md §3.5's "never healed" stays true and gains the pointer
  here.

## 9. Embedders

For nmbrs, which writes workload parameters of varying type through a
scope tree:

- compile every scope with `input_variance: Warn`, so every open input a
  scenario relies on is listed in the compile log;
- convert values whose target is known, such as a CLI argument bound to
  a declared parameter, through `convert::to_port` before writing;
- let scope-tree children built from source take their parents' exact
  types (§7), so most copies need no converter.

The embedding guide's §3 describes the pattern for hosts.

## 10. Tests

- Every engine refuses a mismatched write to a declared input, under
  every `input_variance`.
- Under `Error`, a program with an untyped `input`, an auto-extern, and
  a synthesized extern fails construction naming all three.
- Under `Warn` and `Info`, the same program builds, logs one
  `InputConverterInserted` per converter at the configured level, and
  converts a text write to a numeric consumer identically on every
  engine; a value the catalog cannot convert fails the converter with
  the same attribution on every engine.
- `convert::to_port` agrees with a converter node on every catalog entry
  (a property test over the catalog).
- A binder copy into a declared child input of another type is a
  construction error naming both sides.

## 11. What landed, and what differs from the proposal

Landed 2026-09-24, with `crates/polydat/tests/input_variance.rs` running
every case on every engine:

- `PortType::Dyn` (keyword `dyn`): a value-pair slot naming the stored
  `Value`, whatever its variant, on every compiled engine and at a native
  cone's boundary; a `Dyn` slot accepts any value.
- `InputDef::type_origin` and `InputDef::converts_to`;
  `Kernel::input_type_origin(name)`; `input_port_type` of a converted
  input reports the type its readers see, not `Dyn`, so a host that
  re-declares a parent's inputs as source text declares the reader's type.
- `CompileOptions::input_variance` (`Fixed` | `Error` | `Warn` | `Info`)
  on both option types, and `CompileEvent::InputConverterInserted`, whose
  level is the configured one. An opened input that nothing reads is still
  reported, once, so no opened input goes unseen.
- The converter `__convert_<input>_<type>`, one per input and target type,
  a slot kit on the compiled engines and a `SlotCall` inside native code.
- `polydat::convert::to_port` and `ConvertError`, the one conversion rule
  the converter applies.
- `transform::convert_input(&mut file, name)`, which declares an extern
  `dyn`; its converters are reported at `Info` under any setting.
- `Dataflow::set_wire` and `set_wire_idx` are deprecated; `ScopedExpr::set`
  converts through `convert::to_port` instead of the healing write.

Where the implementation differs from §3–§8:

- **Synthesized externs are named, not detected.** A synthesizer tells the
  compiler which externs it typed, through
  `CompileOptions::inferred_externs`; the subcontext builder passes its
  result and write-through externs this way. Source text carries no mark
  of who wrote a declaration, so a host that synthesizes source with exact
  types (nmbrs's cascade) leaves them declared, as it should.
- **Coordinates are never open.** They are positioned with `set_inputs`
  and are always `u64`; an untyped `input` is recorded `Inferred` for
  reporting, and no setting converts it.
- **`Error` is an `AssemblyError::OpenInputs`**, which a kernel builder
  returns as `KernelError::Assembly`, rather than a new `KernelError`.

Not yet done, and why:

- **§7, the binder's copies into declared inputs** still heal through the
  catalog, as before this design. Making them refuse would change what
  existing scope trees build, and nmbrs's cascade copies exact types, so
  the refusal is deferred to the release that removes `Dataflow::set_wire`.
- **§7, exact types for synthesized write-through externs**:
  `port_type_keyword`'s collapse stays until then, for the same reason.
- **§8, the write-through widening list** is unchanged.

Found while implementing, and fixed with it: `set_inputs` given more
values than a program has coordinates wrote the extra values into its
externs, on every engine (the interpreter into the first extern's value,
the compiled engines into its slots, where a by-reference extern keeps a
pointer). Each engine now writes the coordinates and nothing past them
(`set_inputs_never_reaches_past_the_coordinates`).
