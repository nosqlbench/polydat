# Compiled Non-Scalar Slots — Handles for `Str`, `Bytes`, `Json`, and `Ext`

**Status:** Proposed SRD 115, first revision. Step 1 of §11, the arena
lifetime, has landed: a `PolydatState` is root or nested, a root state's
cycle advance resets the thread's cycle arena and advances its
generation, and every kernel the runtime creates inside another's cycle
(traversal activations and their snapshot and canonical kernels,
materialized subscopes, projection body states, the renderer's
evaluator kernels) is constructed nested before scope-init seeds any
input, so none of them resets. This also fixed the arena's unbounded
growth. Step 2, the color, has landed: `SlotColor::Hdl1` is the color
of `Str`, `Bytes`, `Json`, `Ext`, and `Handle`; `PortType::handle_kind()`
fixes each to byte-string or table handles; raw readers refuse handle
slots as they refuse `Ref2` slots; pure-P3 layout rejects a
handle-colored output until step 4's marshalling; cone admission
requires the `Imm1` color by name; and the alignment, engines, and JIT
boundary documents carry the new color. Step 3, interning at build, has
landed: the static interner is indexed by content and resolves whole
handles; a `const_str` node classifies to `JitOp::StaticStr` with its
text interned, lowered as an immediate store of the handle; and the
Polytile renderer interns every static run and separator when it is
constructed and copies them from the interner, so tile constants never
enter the arena. Step 4, boundary marshalling, has landed: the cone
planner admits `Str` and `Bytes` ports, a boundary input of either is
copied into the cycle arena and a boundary output is copied out to an
owned value, and every wire inside a cone must be exactly its port's
type, so a variadic node fed untyped stays on P1. With this the string
lowerings that existed run at cone boundaries, bit-identical to P1;
getting there fixed three of them: `format_u64` with a non-decimal radix
was classified as decimal, `str_concat` with other than two inputs was
classified for the two-input helper, and the parse helpers returned zero
where P1 raises a diagnostic. Pure-P3 layout now refuses only
table-handle outputs. Tests in `tests/handle_boundaries.rs`. Steps 5
through 8 are not started.
This document fixes the slot representation that lets string, byte,
JSON, and extension values ride through the P2 and P3 engines, so that
the nodes which produce and consume them (string operations, JSON
construction, `printf`, and the Polytile renderers of SRD 114 step 7)
can leave P1. It amends [Engines](engines.md) §5, [Type-System
Alignment](type_system_alignment.md) §5–§7, and the S-axioms of [JIT
Boundary](jit_boundary.md), and it ratifies in Polydat the handle format
that nmbrs SRD 111 introduced.

**Ownership:** Polydat owns the slot color, the handle format, the
arena and value-table lifetimes, the boundary marshalling, the helper
ABI, and the equivalence tests. Hosts own nothing here; a host sees only
the same `Value`s it sees today, at the same P1 boundary.

**Companion documents:**
[Engines](engines.md) (the P1/P2/P3 lattice and the slot colors),
[JIT Boundary](jit_boundary.md) (the S-axioms, helpers, setjmp/longjmp),
[Type-System Alignment](type_system_alignment.md) (type planes),
[Polytile](polytile.md) §7 and §12 step 7 (the first consumer),
nmbrs SRD 111 (the handle format's origin).

## 1. The claim

The compiled tiers run over a flat buffer of `u64` slots. Every port has
a static slot color, `Imm1`, `Imm2`, or `Ref2`, and the classifier admits
a node to a cone only when its ports have a lowering. Today that stops
at scalars and typed slices. A `Str`, `Bytes`, `Json`, `Ext`, or
`Handle` port has no color of its own: `PortType::slot_color()` falls
through to `Imm1` for them, no boundary marshals them, and so no node
that produces or consumes one has a compiled form. The consequences are
visible from the graph: `printf`, every JSON node, every string node,
and every tile stay on P1, and the scalar cones the engines do fuse are
cut short at each of them.

Three pieces of the answer already exist and are not connected:

- **A handle format and a cycle arena.** `kernel/arena.rs` carries
  nmbrs SRD 111 into Polydat: a 64-bit handle with a two-bit tag naming
  a static interner entry, a cycle-arena byte range, or a resource; a
  thread-local bump arena with a one-instruction reset; a process-wide
  static interner.
- **String helpers and lowerings.** `compile/jit/codegen.rs` declares
  and registers `extern "C"` helpers for the string conversions and
  operations (`jit_u64_to_str`, `jit_str_concat`, `jit_str_upper`, and
  their siblings), and `JitOp` has a variant and a lowering for each.
- **A colored slot contract with ratified axioms.** S1–S10 give `Ref2`
  its ownership, publication, and single-dereference rules and the
  tripwires that enforce them.

What is missing is the contract between them. The string helpers can
only run inside a cone whose boundary is all scalar, because the cone
planner rejects any `Str` boundary as unmarshalable, so in practice
they never run. The arena that those helpers allocate into is never
reset by any caller: a thread that did reach a string lowering would
grow its arena for the life of the process. The slot color for a
handle-bearing type is a fall-through, so nothing in the build
distinguishes a handle from an integer, which S1 forbids in spirit.
And `Json` and `Ext` values, which are not byte strings, have no
representation at all.

This document supplies the contract: a fourth slot color for handles,
the lifetime rules that make a handle safe to hold in a slot, the
value table that carries non-byte values, the boundary marshalling in
both directions, the helper ABI, and the P2 closure form that serves
as the equivalence oracle for P3. It is designed so that every node
that today inspects `Value` variants can gain a compiled form by
writing one helper, and so that the Polytile renderer's lowering in
SRD 114 §12 step 7 becomes mechanical.

Three limits shape it, as they shape the kernel:

- **A slot is a complete description of its value, or it names one.**
  Immediate slots are values; handle slots name values in exactly one
  of three places whose lifetimes are proven. There is no third kind.
- **Rendering cost is the cost of the bytes.** Strings and byte
  strings produced during a cycle live in a bump arena that costs one
  pointer increment to allocate into and one instruction to reset.
- **P1 remains the oracle.** Every compiled form is bit-identical to
  typed eval; nothing coerces a value to make a tier available.

## 2. The slot color

### 2.1 `Hdl1`

`SlotColor` gains a fourth member:

| Color | Width | Members | Meaning |
|---|---:|---|---|
| `Imm1` | 1 | scalars | Immediate data; never an address. |
| `Imm2` | 2 | `U128/I128`, `Reg128` views | Two immediate limbs; never an address. |
| `Ref2` | 2 | all `Vec*` types | Engine-internal `(ptr, len)` with a proven owner. |
| `Hdl1` | 1 | `Str`, `Bytes`, `Json`, `Ext`, `Handle` | A handle naming a value in the static interner, the cycle arena, or the state's value table. Never dereferenced by generated code. |

`slot_color()` stays total and static. The fall-through arm that today
maps these five types to `Imm1` becomes an explicit `Hdl1` arm, and
`Imm1`'s definition regains its full meaning: a slot that never holds
an address or a name.

A `Hdl1` slot is opaque to generated code. Native code loads it, passes
it to a helper, and stores what a helper returns. It never masks the
tag, never adds to the offset, and never compares two handles for
equality of the values they name. Equality of names is not equality of
values, and the axioms below depend on generated code not assuming
otherwise.

### 2.2 Handle format

The format is SRD 111's, adopted as Polydat's own:

```text
bits 63..62   tag
tag 00  static      [tag:2][reserved:30][interner id:32]      process-wide, immutable
tag 01  arena       [tag:2][offset:31][length:31]             valid for the current cycle generation
tag 10  table       [tag:2][kind:6][reserved:24][entry:32]    state-owned, single-writer per entry
tag 11  reserved
```

- A **static** handle names an entry in the process-wide string
  interner. Interned bytes are immutable and live for the process. The
  compiler interns every string literal, every `Const<&str>` argument,
  and every static run of a tile skeleton at kernel build, so constants
  never touch the arena.
- An **arena** handle names a byte range in the thread's cycle arena.
  It is valid from the moment its bytes are written until the arena is
  next reset, which is the start of the next root cycle on that thread
  (§4). `Str` and `Bytes` values produced during a cycle live here.
- A **table** handle names an entry in the owning state's value table,
  a `Vec<Value>` sized at kernel build with one entry per `Hdl1` output
  port whose values are not byte strings (§3). The `kind` field records
  the `PortType` so a boundary can decode without consulting the port
  table.

Byte-string values (`Str`, `Bytes`) use static or arena handles. Other
non-scalar values (`Json`, `Ext`, `Handle`) use table handles. A helper
that produces a `Str` writes bytes into the arena and returns an arena
handle; a helper that produces a `Json` writes an `Arc<serde_json::Value>`
into its own table entry and returns that entry's handle. The kind of
handle a port carries is decided by its `PortType` at build time, so
generated code never branches on the tag.

## 3. The value table

Each `PolydatState` that backs a compiled tier owns a value table:
`Vec<Value>` with one entry per `Hdl1`-colored output port of `Json`,
`Ext`, or `Handle` type in the program, plus one entry per boundary
input of those types. Entries are owned exactly as `Ref2` scratch is
owned (S3): a single (step, output port) writes an entry, and every
execution of that step republishes it. Consumers read the entry through
the handle inside the producer's validity interval (S4). There is no
allocation per cycle: the `Arc` in the entry is replaced, and the
previous value drops when it was the last reference.

The table lives beside the buffer and the scratch arena in the
compiled state and is passed to native code as a context pointer, the
second argument every segment already threads through for the buffer.
A helper that reads a table handle receives the context pointer and the
handle; the engine, not the helper, owns the table.

`Ext` values stay opaque. A helper may forward an `Ext` handle, read the
value's reflected projection (`display`, `to_json_value`, the
`try_as_*` accessors), or pass it to a node that declares an `Ext`
input. Nothing downcasts through a handle. This is the same escape
status `Ext` has at P1.

## 4. Lifetimes and the cycle generation

The cycle arena is thread-local. Several kernels run on one thread: the
root kernel the host drives, the child programs of `for` bodies, the
body programs of tile projections, and probe kernels the compiler runs
at build. Only one of them may reset the arena, and every handle in
every slot on the thread must be valid until it does.

- **The root cycle owns the arena.** The arena resets when the root
  kernel on the thread begins a new cycle, and nowhere else. A kernel
  is the root when the host created it; a kernel created by activation,
  materialization, or tile rendering is nested and never resets. The
  reset is the one instruction SRD 111 promised.
- **Generation.** The thread keeps a cycle generation counter that the
  reset increments. In debug and test builds every arena handle written
  to a slot is stamped with the generation in a side vector parallel to
  the buffer, and every resolve asserts the stamp matches, so a handle
  held across a reset fails deterministically by slot number. Release
  builds carry no stamp.
- **Nothing outlives its cycle inside the compiled tier.** A slot value
  that must persist across cycles, an `init`-lifecycle output or a
  `shared` cell, is never an arena handle: constants are interned, and
  a `Str` or `Bytes` value that a lifecycle class carries across cycles
  is copied out to an `Arc` at the boundary and re-encoded when it
  enters a cone. The lifecycle classifier already knows which outputs
  those are.
- **Values leaving the compiled tier are copied.** A cone's `Hdl1`
  boundary output decodes to an owned `Value` (`Arc<str>`, `Arc<[u8]>`,
  or the table entry's `Arc`) before P1 sees it. P1 never holds an
  arena handle, so the whole of P1, every host API, and every nested
  kernel keep their present semantics.

The reset chokepoint is the root kernel's cycle advance, `set_inputs`
on a root `PolydatState`. Kernels constructed through the activation,
subscope, and tile paths record that they are nested and skip the
reset. A host that drives two root kernels on one thread interleaved by
cycle is supported: each root's cycle advance resets, and the other
root's arena handles were already copied out at its boundary.

## 5. Boundary marshalling

A cone boundary today admits `U64`, `F64`, and `Bool`. It gains `Hdl1`
ports in both directions:

| Direction | `Str` / `Bytes` | `Json` / `Ext` / `Handle` |
|---|---|---|
| P1 value → cone input | interned if the wire is a compile-time constant, else copied into the arena; the slot holds the handle | written into the boundary's table entry; the slot holds its handle |
| cone output → P1 value | the arena bytes are copied into a fresh `Arc<str>` / `Arc<[u8]>` | the table entry's `Arc` is cloned out |

The copy at the outbound boundary is the price of keeping P1's
semantics intact, and it is paid once per boundary port per cycle,
never per interior wire. The cone planner's marshalability check admits
`Hdl1` alongside the three scalar types, and the `encode_boundary` and
`decode_boundary` arms grow accordingly. Interior `Hdl1` wires between
two members of the same cone are handles end to end and cost nothing.

## 6. The helper ABI

A helper is an `extern "C"` function over `u64` arguments and a `u64`
return, registered with the JIT symbol table as the existing string
helpers are. Its conventions:

- Scalars ride in slot form (`f64` as bits, `bool` as 0/1).
- Byte-string arguments and returns are handles; the helper resolves
  through the thread arena or the interner and writes results into the
  arena.
- Table-handle arguments and returns are handles plus the context
  pointer; the helper resolves and writes through the engine's table.
- A helper never allocates on the Rust heap for a `Str` or `Bytes`
  result; it writes bytes into the arena directly. A helper that must
  build an intermediate (`serde_json::to_string`) writes the final bytes
  into the arena and drops the intermediate; a later revision may give
  `serde_json` an arena writer.
- Panics do not cross the boundary. A helper that can fail follows the
  predicate-violation pattern of jit_boundary.md: it reports through the
  `longjmp` helpers, never by unwinding.

Every node that inspects `Value` variants today becomes compiled by
supplying one helper per operation and one `JitOp` variant. The first
set, in order of leverage: the string operations that already have
helpers, `printf`, the JSON constructors and `json_text`, `to_json` and
`json_to_str`, and then `tile_encode` and `tile_render`.

## 7. The P2 closure form

P2 is the equivalence oracle for P3 and stays one. A `compiled_slot`
kit for a `Hdl1`-bearing node is a Rust closure over the same slot
buffer, arena, and table that P3 uses; it calls the same Rust function
the `extern "C"` helper wraps. The `#[polydat_node]` macro emits this
kit for any node whose ports are all `Imm1` or `Hdl1` when the body is
expressible over handles, which for the string and JSON nodes it is.
The P1↔P2↔P3 equivalence tests then pin all three tiers to the same
bytes, as `polydat_node_macro.rs` does for scalars today.

## 8. Axioms

The S-axioms stand for `Ref2`. `Hdl1` has its own, numbered H so that a
SAFETY comment can cite the one it depends on.

- **H1 — A handle is a name, not an address.** Generated code loads,
  stores, and passes handles; it never decodes one. Only helpers and
  the engine decode. *Chokepoint: no `Hdl1` slot is an operand of any
  arithmetic or memory instruction in emitted IR; the tripwire is the
  IR verifier pass in the JIT tests.*
- **H2 — Three places, decided at build.** A `Hdl1` port's handles are
  static or arena for byte strings and table for everything else, by
  `PortType`. No slot ever holds a handle of another kind. *Chokepoint:
  `PortType::handle_kind()`.*
- **H3 — Arena handles live one cycle.** An arena handle is valid from
  its write until the root cycle's next reset. Nothing in the compiled
  tier carries one across a reset; init and shared values are interned
  or copied out. *Tripwire: the debug generation stamp of §4.*
- **H4 — One writer per table entry.** Exactly one (step, output port)
  writes an entry and republishes it every execution, and every reader
  reads inside the writer's validity interval. This is S3 and S4 for
  the table. *Tripwire: the same post-pass assertion S9(a) makes for
  scratch, extended to table entries.*
- **H5 — Only the root resets.** The arena resets at a root kernel's
  cycle advance and nowhere else; nested kernels never reset. *Chokepoint:
  the `is_root` flag set by the constructors that create nested
  kernels.*
- **H6 — P1 never holds a handle.** Every `Hdl1` boundary output is
  copied out to an owned `Value` before P1 or a host sees it. The host
  API, the lifecycle classes, and the nested kernels keep their present
  semantics. *Tripwire: no `Value` variant carries a raw handle.*
- **H7 — Equivalence.** For every node with a compiled form, P1, P2,
  and P3 produce identical bytes for identical inputs across the fuzz
  corpus, and identical `Value`s at the boundary. *Tripwire: the
  differential suite of §11 step 7.*

S1's statement that immediate slots never contain an address is
unchanged; `Hdl1` is not an immediate color. S7's ban on arena handles
applies to `Ref2` data, where a handle would be a second dereference;
for `Hdl1` the handle is the value's representation, and the single
dereference S7 protects is the helper's. Both readings are stated in
jit_boundary.md when this SRD lands.

## 9. Boundaries

- **Not a general GC.** The arena is a bump allocator with a cycle
  lifetime. A value that must outlive the cycle leaves the compiled
  tier as an owned `Value`. There is no reference counting inside a
  slot buffer.
- **Not a change to P1.** Typed eval, `Value`, the host API, and every
  nested kernel are untouched. A program that never enters P2 or P3
  behaves exactly as before.
- **Not a new string type.** `Str` is still `Arc<str>` at P1; the handle
  is its compiled representation only.
- **`Ext` stays opaque.** No handle grants a downcast. An `Ext` value
  crosses a boundary as its `Arc`, is forwarded or projected inside a
  cone, and comes back as the same `Arc`.
- **No coercion for eligibility.** The classifier never re-types a port
  to admit a node; a `Hdl1` port is admitted as `Hdl1` or the node stays
  on P1.

## 10. Findings this design rests on

Recorded here so the implementation plan is honest about its starting
point.

- `PortType::slot_color()` maps `Str`, `Bytes`, `Json`, `Ext`, and
  `Handle` to `Imm1` through the fall-through arm.
- `JitOp` has variants and lowerings for thirteen string operations,
  each calling a registered helper that allocates in the thread arena,
  and the cone planner's `scalar_ok` admits only `U64`, `F64`, and
  `Bool` at boundaries, so those lowerings run only when both ends of
  every string wire sit inside one cone with scalar boundaries.
- No caller resets `CycleArena`; the only writers are the JIT string
  helpers. A thread that reaches one grows its arena for the life of
  the process. Step 1 below fixes this before anything else is built on
  the arena.
- `encode_boundary` and `decode_boundary` in `compile/cone.rs` marshal
  `U64`, `F64`, and `Bool`, and their comment asks to be widened when a
  node gains a lowering. This is that widening.

## 11. Implementation plan

1. **Arena lifetime.** The root/nested distinction on states, the reset
   and the generation increment at a root state's cycle advance, and
   tests that a nested state never resets and that the arena's cursor
   returns to zero each root cycle. Fixes the unbounded growth. No new
   node compiles yet. The per-slot generation stamp of §4 lands with
   step 4, when handles first occupy slots.
2. **`Hdl1`.** The fourth `SlotColor`, `PortType::handle_kind()`, and
   the `Hdl1` arm in `slot_color()`; the alignment doc's §6 table and
   the S-axiom notes updated; the IR-verifier tripwire for H1.
3. **Static interner at build.** String literals, `Const<&str>`
   arguments, and tile static runs interned at kernel build with their
   handles baked as constants, so constants never enter the arena.
4. **Boundary marshalling.** `Hdl1` admitted by the cone planner;
   `encode_boundary`/`decode_boundary` for `Str` and `Bytes` (interner
   or arena in, copy out). With this, the existing string lowerings run
   at cone boundaries. Equivalence tests for each string `JitOp`.
5. **The value table.** The per-state table, its context pointer in the
   segment ABI, table handles for `Json`, `Ext`, and `Handle`, the H4
   tripwire, and boundary marshalling for them.
6. **Helpers and lowerings.** `printf`, the JSON constructors,
   `json_text`, `to_json`, `json_to_str`; then `tile_encode` and
   `tile_render` per SRD 114 §12 step 7, with tile statics interned and
   projection bodies activated as `for` bodies are.
7. **P2 closures and differential tests.** `compiled_slot` kits for
   `Hdl1` nodes from the macro; the P1↔P2↔P3 suite over the fuzz corpus,
   including the tile fuzzer; the S9-style post-pass assertions for
   table entries.
8. **Docs.** engines.md §5 and §7, type_system_alignment.md §5–§7,
   jit_boundary.md's axiom section with H1–H7, and the Polytile SRD's
   step 7 marked done.

Each step lands with its tests and leaves the previous surfaces working.
Step 1 is independent of the rest and is a fix in its own right.
