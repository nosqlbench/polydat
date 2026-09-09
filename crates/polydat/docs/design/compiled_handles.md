# Compiled Non-Scalar Slots — Handles for `Str`, `Bytes`, `Json`, and `Ext`

**Status:** SRD 115, second revision. Implemented in full; §12 is the
landing record. Ratification is pending.

**Purpose.** The compiled tiers (P2 closures, P3 native code, and the
hybrid kernel) run over a flat buffer of `u64` slots. Before this
document, only scalars and typed slices had a slot representation, so
every node that produces or consumes a string, byte string, JSON value,
or extension value stayed on P1, and the cones the engines fuse were
cut short at each one. This document gives those values a slot
representation, a 64-bit handle that names the value in one of three
stores with proven lifetimes, so that string operations, JSON
construction, `printf`, and the Polytile renderers run compiled and
bit-identical to P1. It amends [Engines](engines.md) §5–§7,
[Type-System Alignment](type_system_alignment.md) §5–§7, and the
S-axioms of [JIT Boundary](jit_boundary.md), and it ratifies in Polydat
the handle format nmbrs SRD 111 introduced.

**Ownership.** Polydat owns the slot color, the handle format, the arena
and value-table lifetimes, the boundary marshalling, the helper ABI, the
closure ABI, and the equivalence tests. Hosts own nothing here; a host
sees the same `Value`s it sees today, at the same P1 boundary.

**Companion documents.**
[Engines](engines.md) (the P1/P2/P3 lattice and the slot colors),
[JIT Boundary](jit_boundary.md) (the S-axioms, helpers, setjmp/longjmp),
[Type-System Alignment](type_system_alignment.md) (type planes),
[Polytile](polytile.md) §7 and §12 step 7 (the first consumer),
nmbrs SRD 111 (the handle format's origin).

## 1. The problem

Every port has a static slot color. `Imm1` and `Imm2` slots hold
immediate values, `Ref2` slots hold a `(ptr, len)` pair with a proven
owner, and the classifier admits a node to a cone only when its ports
have a lowering. Before this document, `Str`, `Bytes`, `Json`, `Ext`,
and `Handle` had no color of their own: `PortType::slot_color()` fell
through to `Imm1` for them, no boundary marshalled them, and no node
that touched one had a compiled form.

Three pieces of the answer existed and were not connected: a handle
format and a thread-local bump arena carried in from SRD 111
(`kernel/arena.rs`); `extern "C"` helpers and `JitOp` lowerings for the
string operations, which never ran because the cone planner rejected
every string boundary; and the colored-slot contract S1–S10, which gave
`Ref2` its rules and tripwires. The arena was never reset by any caller,
so a thread that did reach a string lowering would have grown it for the
life of the process.

This document supplies the contract between those pieces: a fourth slot
color, the lifetime rules that make a handle safe to hold in a slot, a
value table for the values that are not byte strings, boundary
marshalling in both directions, the helper ABI, and the P2 closure form
that serves as the equivalence oracle for P3. Three limits shape it:

- **A slot is a complete description of its value, or it names one.**
  Immediate slots are values; handle slots name values in exactly one
  of three places whose lifetimes are proven. There is no third kind.
- **Rendering cost is the cost of the bytes.** Strings produced during a
  cycle live in a bump arena that costs one pointer increment to
  allocate into and one instruction to reset.
- **P1 remains the oracle.** Every compiled form is bit-identical to
  typed eval; nothing coerces a value to make a tier available.

## 2. The slot color

### 2.1 `Hdl1`

`SlotColor` has four members:

| Color | Width | Members | Meaning |
|---|---:|---|---|
| `Imm1` | 1 | scalars | Immediate data; never an address. |
| `Imm2` | 2 | `U128/I128`, `Reg128` views | Two immediate limbs; never an address. |
| `Ref2` | 2 | all `Vec*` types | Engine-internal `(ptr, len)` with a proven owner. |
| `Hdl1` | 1 | `Str`, `Bytes`, `Json`, `Ext`, `Handle` | A handle naming a value in the static interner, the cycle arena, or the engine's value table. Never decoded by generated code. |

`slot_color()` is total and static; the `Hdl1` arm is explicit, so
`Imm1` keeps its full meaning: a slot that never holds an address or a
name. `PortType::handle_kind()` refines `Hdl1` by port type: `Str` and
`Bytes` carry byte-string handles (interner or arena), `Json`, `Ext`,
and `Handle` carry table handles. The kind is decided at build, so
generated code never branches on a handle's tag.

A `Hdl1` slot is opaque to generated code. Native code loads it, passes
it to a helper, and stores what a helper returns. It never masks the
tag, never adds to the offset, and never compares two handles for
equality of the values they name; equality of names is not equality of
values. H1 in §8 is this rule and its tripwire.

### 2.2 Handle format

The format is SRD 111's, adopted as Polydat's own:

```text
bits 63..62   tag
tag 00  static      [tag:2][reserved:30][interner id:32]      process-wide, immutable
tag 01  arena       [tag:2][offset:31][length:31]             valid until the root cycle's next reset
tag 10  table       [tag:2][kind:6][generation:24][entry:32]  engine-owned, single-writer per entry
tag 11  reserved
```

- A **static** handle names an entry in the process-wide string
  interner. Interned bytes are immutable and live for the process. The
  compiler interns every string literal, every `Const<&str>` argument,
  and every static run and separator of a tile skeleton at kernel build,
  so constants never touch the arena. Parsed forms of such constants
  (a `printf` format, a hole encoding, a tile program) are interned by
  content the same way, so a lowering can bake their addresses.
- An **arena** handle names a byte range in the thread's cycle arena.
  It is valid from the moment its bytes are written until the arena is
  next reset, at the start of the next root cycle on that thread (§4).
  `Str` and `Bytes` values produced during a cycle live here. The arena
  is a list of fixed-size chunks rather than one growable buffer, so
  bytes already written never move within a cycle; the offset is
  `chunk << 16 | position`, and an allocation larger than a chunk takes
  a chunk of its own. Chunks are retained across resets.
- A **table** handle names an entry in the value table of the engine
  that ran the code (§3). The `kind` field records the value's variant,
  so a reader can refuse a handle of the wrong kind without touching the
  entry. The `generation` field is the low 24 bits of the cycle
  generation the entry was written in, and every read checks it against
  the table's current generation, in every build: a table handle that
  outlives its cycle fails by handle rather than naming whatever the
  entry holds now (H3). Where SRD 111 reserved these bits, Polydat
  spends them on the stamp, because the table is where a stale handle is
  cheapest to catch and most expensive to miss. Arena handles carry no
  stamp; their bound is the root cycle's reset.

## 3. The value table

A value table is owned by the engine that runs compiled code, never by
the thread. Its shape is fixed when the code is compiled: one entry per
table-kind slot the code can write, numbered in step and port order,
plus one per table-kind boundary input where the engine has boundary
inputs. Each entry has exactly one writer, the helper or closure for the
step that owns the slot, which is told its entry number at compile time
and replaces the entry in place every time it runs (H4). Nothing
appends; compiled code cannot grow the table, and an entry number
outside it is refused. A steady cycle allocates nothing beyond the `Arc`
that enters an entry, and the table's size is a property of the program,
not of how many values a cycle happened to make.

Compiled code reaches the table through an installation the engine
makes around each run and removes when it returns. A helper or closure
that runs with no table installed has no owner for what it writes and
panics. The installation is a thread-local pointer rather than a
parameter in the segment signature; it is set and cleared by the engine
that holds the `&mut` to the table, restored on unwind, and nests
correctly when one engine's run contains another's.

The engines use the table in two ways:

- **Embedded cones** in a P1 kernel borrow a table for one eval. The
  cone node sizes it at the top of `eval`, stamps it with the thread's
  cycle generation, writes its table-kind boundary inputs into their
  entries, runs, copies every output out, then clears the table and
  releases the cycle arena back to the mark it took before the call.
  Nothing a cone allocates outlives its eval, so a cycle's arena and
  table use is bounded by its largest cone rather than by the number of
  cone evals. The root cycle's reset (§4) remains the outer bound.
- **Whole kernels** (P2, pure P3, hybrid) own a table for their
  lifetime, sized from the `(slot, entry)` pairs the builder numbers.
  A kernel that a host drives directly begins a root cycle at each run;
  one wrapped by a state that owns the cycle takes the state's
  generation. After every run, in debug builds, the kernel checks that
  each table-kind slot holds a table handle of the current generation
  naming exactly its own entry and that the entry was written: the S9
  validator restated for handles (H4). A hybrid kernel owns one table
  across its JIT segments and closure steps, numbered together at build.

A helper or closure that produces a `Json` writes the
`Arc<serde_json::Value>` into its entry and returns the entry's handle;
one that reads a table handle resolves it to a borrow of the entry and
copies out what it needs. A read checks the tag, the generation, and
that the entry exists and was written.

`Ext` values stay opaque. A helper may forward an `Ext` handle, read the
value's reflected projection (`display`, `to_json_value`, the
`try_as_*` accessors), or pass it to a node that declares an `Ext`
input. Nothing downcasts through a handle. This is the escape status
`Ext` has at P1.

## 4. Lifetimes and the cycle generation

The cycle arena is thread-local, and several kernels run on one thread:
the root kernel the host drives, the child programs of `for` bodies, the
body programs of tile projections, and probe kernels the compiler runs
at build. Only one of them may reset the arena, and every handle in
every slot on the thread must be valid until it does.

- **The root cycle owns the arena.** The arena resets when the root
  kernel on the thread begins a new cycle, and nowhere else. A kernel is
  the root when the host created it; a kernel created by activation,
  materialization, or tile rendering is nested and never resets. The
  reset chokepoint is the root state's cycle advance (`set_inputs`), or
  the run of a whole kernel a host drives directly.
- **Generation.** The thread keeps a cycle generation counter that the
  reset increments. Table handles carry it (§2.2). Arena handles carry
  no stamp; instead, a step that writes a handle slot is never skipped
  as clean in any provenance kernel, so no slot holds an arena handle
  from before the last reset. The rule is mechanical in each tier: the
  P3 codegen emits no clean guard for such a step, and the P2 and hybrid
  kernels never mark it clean.
- **Nothing outlives its cycle inside the compiled tier.** A slot value
  that must persist across cycles, an `init`-lifecycle output or a
  `shared` cell, is never an arena handle: constants are interned, and a
  `Str` or `Bytes` value a lifecycle class carries across cycles is
  copied out to an `Arc` at the boundary and re-encoded when it enters
  a cone.
- **Values leaving the compiled tier are copied.** A cone's `Hdl1`
  boundary output decodes to an owned `Value` before P1 sees it, and a
  whole kernel's `get_value` copies out the same way. P1 never holds a
  handle, so the whole of P1, every host API, and every nested kernel
  keep their present semantics.

A host that drives two root kernels on one thread interleaved by cycle
is supported, with one rule: read a kernel's handle outputs, through
`get_value`, before running any other root kernel on the thread. What
the host holds after `get_value` is its own; a handle output read after
another root's cycle advance would read whatever that root wrote in the
same arena bytes. The tier differential reads each kernel immediately
after its own run for this reason.

## 5. Boundary marshalling

A cone boundary admits `U64`, `F64`, `Bool`, and every `Hdl1` port, in
both directions:

| Direction | `Str` / `Bytes` | `Json` / `Ext` / `Handle` |
|---|---|---|
| P1 value → cone input | interned if the wire is a compile-time constant, else copied into the arena; the slot holds the handle | written into the boundary's table entry; the slot holds its handle |
| cone output → P1 value | the arena bytes are copied into a fresh `Arc<str>` / `Arc<[u8]>` | the table entry's `Arc` is cloned out |

The copy at the outbound boundary is the price of keeping P1's
semantics intact, paid once per boundary port per cycle, never per
interior wire. Interior `Hdl1` wires between two members of the same
cone are handles end to end and cost nothing. The marshalling rule
lives in one place, `compile::marshal`, and the pure-P3, P2, and hybrid
kernels' `get_value` readers decode with it by each output's port type.

Inside a cone every wire is exactly its port's type, with one waiver:
the wire-typed lowerings of §6.1 fix a variadic node's wire types
themselves, so their advertised port types do not bind. The waiver is
by the op the classification produced, never by node name.

## 6. The helper ABI

A helper is an `extern "C"` function over `u64` arguments and a `u64`
return, registered with the JIT symbol table. Its conventions:

- Scalars ride in slot form (`f64` as bits, `bool` as 0/1).
- Byte-string arguments and returns are handles; the helper resolves
  through the arena or the interner and writes results into the arena.
- Table-handle arguments and returns are handles. A producer takes the
  entry it owns as its first argument, an immediate from the layout, and
  writes through the installed table (§3); a reader resolves through the
  same table.
- A helper writes a string result straight into the arena through an
  `ArenaWriter`, which extends one allocation at the cursor as the
  bytes are produced and yields the handle at the end: scalar
  formatting, case changes, concatenation, JSON serialization
  (`serde_json::to_writer` through the writer's `io::Write`), `printf`,
  `json_text`, the tile encoder, and the tile renderer all write this
  way, with no intermediate `String`. If something else allocates
  between two of the writer's pushes (a projection body's kernels
  running inside a tile render) or the chunk fills, the writer moves
  what it has to a fresh allocation and continues, so the result is
  always one range. This is sound because the arena is chunked: bytes
  already written never move within a cycle, so a resolved source
  string stays valid while its result is written (§4).
- Panics do not cross the boundary. A helper that can fail reports
  through the `longjmp` path of jit_boundary.md, never by unwinding. A
  helper whose body may panic as its P1 node panics (`printf` on a
  missing argument, a tile whose projection fails) catches the panic and
  re-raises it through the same path, so it surfaces in Rust land as
  the panic the P1 node raises.

Each node gains a compiled form by supplying one helper and one `JitOp`
variant, and the helper calls the same function the P1 node calls, so
H7 holds by construction. The helper table is in jit_boundary.md.

### 6.1 Wire-typed lowerings

A node whose P1 body dispatches on `Value` variants (`printf` and the
JSON constructors over `&[Value]`, `to_json`, `json_text`, and the tile
nodes over a `Value` port) has no fixed port types to lower by. Its
lowering is decided with the types of its wires, which the assembler
knows for every edge: `classify_node_typed(node, wire_types)` is the
classifier the P3 layout, the cone planner, and the hybrid builder use,
and it falls back to the untyped classifier for every other node.

- **One code per argument.** Each wire type maps to a one-byte code
  (`u i f b s y j e h` for `U64 I64 F64 Bool Str Bytes Json Ext
  Handle`); the codes of a node's wires are interned as one static
  string whose handle is baked into the call. A wire of any other type
  has no code, and the node stays on P1. `str_concat`, whose helper
  takes strings, is classified the same way and needs string wires.
- **Arguments in a stack array.** Generated code stores the argument
  slots into an array in its own frame and passes its address; the
  helper decodes each argument by its code, scalars from bits, strings
  by reference from the arena or interner, table kinds from the
  installed table.
- **Parsed constants are interned** (§2.2), and the helper receives the
  address.
- **Arguments are borrowed, not copied.** A helper sees each argument
  as a `ValueRef`, a borrowed view of a `Value`: a scalar by value, a
  string or byte string by reference into the arena or interner, a JSON
  value by reference into the installed table, anything else by
  reference to the `Value`. `printf` (through `FmtArg`), the JSON
  constructors and coercion, `json_text`, and the tile encoder all run
  over the view, and the P1 nodes build the same view from their
  `Value` inputs, so one body serves both tiers and a string argument
  is never copied to be inspected, formatted, or encoded. The one
  exception is `tile_render`, whose inputs are re-bound into projection
  states and so must be owned `Value`s.

### 6.2 Projections and re-entrancy

A tile skeleton with a projection re-runs its body program per tuple
through nested kernels, at P3 exactly as at P1 and P2. The nested
kernels never reset the arena, take their own cone scratch and tables,
and install and restore their own value tables, so the render runs
inside the calling engine's cycle and leaves only its text behind. For
this the cone eval is re-entrant: it takes its scratch buffer and table
out of their thread-local cells for the duration of the native call, so
a nested cone eval finds the cells free, and returns them on every exit
path.

## 7. The P2 closure form

P2 is the equivalence oracle for P3 and stays one. A P2 closure for a
`Hdl1`-bearing node runs over the same slot buffer, arena, and table
that P3 uses and calls the same body the P1 node and the native helper
call. The `#[polydat_node]` macro emits the `compiled_u64` kit for byte
strings (a `&str` argument resolves its handle, a `String` result enters
the arena); for the table kinds it emits a second kit,
`compiled_handle(entry_base, wire_types)`:

- **Eligibility.** The node has at least one shape the u64 kit cannot
  carry but the table can: a JSON port (`&serde_json::Value` or
  `Arc<serde_json::Value>`), a polymorphic `Value` port, an `Ext<T>`
  port, or a variadic of anything but `u64`; every other argument is a
  one-slot carrier, a const, or a setup derived from consts; the return
  is a one-slot carrier, a JSON value, or an `Ext<T>`. Session-static
  setup (`from = ()`), fallible bodies, tuple and dynamic returns, split
  variadics, and `Handle` downcasts stay on P1.
- **Entries and types from the kernel.** The kit takes the first
  value-table entry the node's table-kind outputs own and the type of
  each wire input. A JSON result is written to its entry through the
  installed table; a polymorphic or variadic argument decodes by its
  wire type, a JSON argument by handle. This is the P2 form of §6.1.
- **Setup recomputed.** A `#[poly_const]` value is a pure function of
  the node's consts, so the kit recomputes it from the captured consts
  at construction, once, and the closure borrows its own copy.
- **The kernel owns the table.** The P2 kernels carry a value table
  sized from the `(slot, entry)` pairs the assembler numbers, install it
  around every run, begin a root cycle when a host drives them and adopt
  the wrapping state's generation otherwise, never mark a handle-writing
  step clean, and run the H4 validator after every run in debug builds.
  Hybrid kernels take the same closures for the nodes they do not JIT,
  install the table around every step, and number entries with their
  segments'.

## 8. Axioms

The S-axioms stand for `Ref2`. `Hdl1` has its own, numbered H so that a
SAFETY comment can cite the one it depends on.

- **H1 — A handle is a name, not an address.** Generated code loads,
  stores, and passes handles; it never decodes one. Only helpers,
  closures, and the engine decode. *Tripwire: `verify_handle_discipline`
  runs over the emitted IR of every compiled function before it is
  defined. Every value loaded from a handle slot or stored into one is
  tainted; a tainted value may reach only a store as its data, a stack
  store, or a call as an argument, and a value stored into a handle slot
  must come from a load, a helper call, or an interned immediate. A
  violation fails the compile, naming the slot and the instruction.*
- **H2 — Three places, decided at build.** A `Hdl1` port's handles are
  static or arena for byte strings and table for everything else, by
  `PortType`. No slot ever holds a handle of another kind. *Chokepoint:
  `PortType::handle_kind()`.*
- **H3 — Handles live one cycle.** An arena handle is valid from its
  write until the root cycle's next reset, and a cone's handles only
  until its eval returns; a table handle is valid within the generation
  that wrote it. Nothing in the compiled tier carries one across.
  *Tripwire: the generation stamp in every table handle, checked on
  every read; handle-writing steps never skipped as clean (§4).*
- **H4 — One writer per table entry.** Exactly one (step, output port)
  writes an entry, is told its entry number at compile time, and
  republishes it every execution; every reader reads inside the writer's
  validity interval. This is S3 and S4 for the table. *Tripwire: the
  post-run validator of §3 in the P2, P3, and hybrid kernels.*
- **H5 — Only the root resets.** The arena resets at a root kernel's
  cycle advance and nowhere else; nested kernels never reset.
  *Chokepoint: the nested flag set by the constructors that create
  nested kernels.*
- **H6 — P1 never holds a handle.** Every `Hdl1` boundary output and
  every `get_value` read copies out to an owned `Value`. *Tripwire: no
  `Value` variant carries a raw handle.*
- **H7 — Equivalence.** For every node with a compiled form, P1, P2,
  P3, and the hybrid kernel produce identical bytes for identical
  inputs, and identical `Value`s at the boundary. *Tripwire: the
  differential suite of §10.*

S1's statement that immediate slots never contain an address is
unchanged; `Hdl1` is not an immediate color. S7's ban on arena handles
applies to `Ref2` data, where a handle would be a second dereference;
for `Hdl1` the handle is the value's representation, and the single
dereference S7 protects is the helper's. jit_boundary.md states both
readings.

## 9. Boundaries

- **Not a general GC.** The arena is a bump allocator with a cycle
  lifetime. A value that must outlive the cycle leaves the compiled tier
  as an owned `Value`. There is no reference counting inside a slot
  buffer.
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
  on P1. The wire-typed lowerings read the wire types the assembler
  already established and change nothing about them.
- **The SRD-74 None rule is refined, not relaxed.** A fused cone is one
  node to the kernel guard: a None on any boundary input makes every
  output None. A node that tolerates None inputs and would have produced
  a value from one (`tile_encode` writes `null`, `to_json` keeps going,
  `identity` passes it through) may therefore join a cone only when
  every one of its inputs is an intra-cone wire from another member,
  where no None can arrive because the cone's own boundary is guarded
  and no lowered op produces None. Eligibility is decided in topological
  order on that basis, and the planner rejects a cone whose component
  split left such a node on the boundary. Fed by a kernel input, the
  node stays on P1 and keeps its P1 semantics exactly.

## 10. Verification

Every axiom has a mechanical check, and every tier is pinned to P1:

| Check | Where |
|---|---|
| Arena lifetime: root resets, nested never does, cursor returns to zero | `tests/arena_lifetime.rs` |
| Slot color, handle kinds, raw readers refuse handle slots | `tests/slot_state_axioms.rs` |
| Static interning by content; `const_str` and tile statics baked | `tests/static_interner.rs` |
| Byte-string boundaries, string lowerings, eval-scoped release | `tests/handle_boundaries.rs` |
| Value table: fixed entries, in-place replacement, generation refusal, JSON through cones | `tests/value_table.rs` |
| Wire-typed lowerings, None-rule refinement, tiles with and without projections, hybrid arena bound | `tests/variadic_lowering.rs` |
| Tier differential: random string, JSON, and tile programs across the interpreter, forced cones, P2, hybrid, and pure P3; the hand-written corpus; provenance kernels on repeated coordinates | `tests/handle_tiers.rs` (`FUZZ_SEED`, `FUZZ_ITERATIONS`) |
| H1 taint pass on hand-built IR | `h1_tests` in `compile/jit/codegen.rs` |
| H4 validator | every P2, P3, and hybrid run in debug builds |
| Miri lane: arena bytes never move, sources inside the arena, the writer's relocation, mark and release, table installation nesting and unwind, P2 handle closures with and without projections | `tests/handle_miri.rs` under `cargo +nightly miri test -p polydat --no-default-features --test handle_miri` |

## 11. Implementation plan

All eight steps have landed; §12 records how.

1. **Arena lifetime.** Root and nested states, the reset and generation
   increment at a root state's cycle advance, and tests that a nested
   state never resets.
2. **`Hdl1`.** The fourth `SlotColor`, `PortType::handle_kind()`, raw
   readers refusing handle slots, the companion docs, and the H1
   tripwire.
3. **Static interner at build.** String literals, `Const<&str>`
   arguments, and tile static runs interned at kernel build with their
   handles baked as constants.
4. **Boundary marshalling.** `Hdl1` admitted by the cone planner;
   encode and decode for `Str` and `Bytes`; the existing string
   lowerings running at cone boundaries with equivalence tests.
5. **The value table.** The engine-owned table of §3, the installed
   context, table handles with the generation stamp, eval-scoped
   release in cones, the H4 validator in whole kernels.
6. **Helpers and lowerings.** `printf`, the JSON constructors,
   `json_text`, `to_json`, `json_to_str`, `tile_encode`, and
   `tile_render`, projections included, by the wire-typed
   classification of §6.1.
7. **P2 closures and differential tests.** The `compiled_handle` kit
   from the macro; the tier differential; the H4 validator in the P2
   kernels.
8. **Docs.** engines.md, type_system_alignment.md, jit_boundary.md, and
   the Polytile SRD's step 7 marked done.

## 12. Landing record

Kept short; the normative text above is what the code does. Dates are
2026-09-05 and 2026-09-06.

- **Step 1.** `PolydatState` is root or nested; every kernel the runtime
  creates inside another's cycle (activations and their snapshot and
  canonical kernels, materialized subscopes, projection bodies, the
  renderer's kernels) is constructed nested before scope-init seeds any
  input. This also fixed the arena's unbounded growth.
- **Step 2.** `SlotColor::Hdl1`, `PortType::handle_kind()`, raw readers
  refusing handle slots. The H1 tripwire was deferred and landed after
  step 8 as `verify_handle_discipline`.
- **Step 3.** Interner indexed by content; `const_str` lowered as an
  immediate static handle; tile statics and separators interned at
  render-node construction.
- **Step 4.** Cone planner admits `Str` and `Bytes`. Three latent
  defects in the string lowerings were fixed on the way: `format_u64`
  with a non-decimal radix classified as decimal, `str_concat` with
  other than two inputs classified for the two-input helper, and the
  parse helpers returning zero where P1 raises a diagnostic.
- **Step 5.** First landed as a thread-local append-only table, then
  redone as §3 specifies after review: engine-owned, fixed entries
  assigned at codegen, installed context, generation stamp stated in
  §2.2 (this is the second revision). `compile::marshal` became the one
  place for encode and decode.
- **Step 6.** The wire-typed classification, the interned parsed
  constants, and the SRD-74 refinement of §9. Projection tiles landed
  afterwards as a follow-up, with the re-entrant cone eval and the
  panic guard of §6. Fixed on the way: a host-driven hybrid kernel
  never began a root cycle, so string-producing segments grew the arena
  across runs.
- **Step 7.** The `compiled_handle` kit, the P2 kernels' table
  ownership and validator, and `tests/handle_tiers.rs`. The shared body
  signature now qualifies `Const<T>` so modules need not import it.
- **Step 8.** The companion documents.
- **Hybrid kernel.** Brought in fully after step 8: `get_value`, the H4
  validator, the wire-typed classifier, the table installed around every
  step, and handle-writing steps never marked clean. Bringing it into
  the differential exposed two defects: the hybrid builder had used the
  untyped classifier, so `str_concat` with a u64 wire reached the string
  helper, and the differential itself had read one root kernel's
  handles after another root's cycle advance, the host rule §4 now
  states.

- **Direct arena writing.** Landed after the record above: the arena
  became chunked so bytes never move within a cycle, and every
  string-producing helper writes through `ArenaWriter` instead of
  building a `String`. The tile renderer, encoder, `printf`, and
  `json_text` gained sink-generic forms so P1 and the helpers share one
  body over a `String` or the writer.

- **Borrowed arguments.** `ValueRef` (in `ast`) is the view every
  argument-inspecting helper runs over; `encode_ref`, `json_of_ref`,
  `json_array_of_refs`, `json_object_of_refs`, and `json_text_ref_into`
  are the borrowed bodies, and the owned forms call them. A table
  entry is borrowed for a helper's duration through
  `current_table_value`, on the same interval argument as arena bytes.

- **Miri lane.** `tests/handle_miri.rs` runs the arena, writer, table,
  and P2 closure paths under Miri without the JIT feature, the handle
  analogue of the S9(b) lane; the same tests run natively in the suite.
  Its first run found a real defect the native runs could not: the
  arena allocated by mutably indexing a whole chunk, which under the
  aliasing model invalidates every shared reference a helper still
  holds into that chunk, so "bytes never move" was necessary but not
  sufficient. The arena now builds every mutable view from a raw
  pointer over exactly the fresh range; `src/kernel/arena.rs` is on the
  S10 allowlist for that reason. The lane passes clean, leak check
  included.

- **Coverage review.** Extending the differential's generator to every
  landed shape found two nodes without a P2 form that the P3 tiers had
  masked: a string literal (`const_str`, now a `compiled_u64` override
  that stores the interned static handle, as its P3 lowering does) and
  `identity`, which is polymorphic and so has no kit of its own; the
  P2 and hybrid builders now synthesize a slot copy for it on every
  color but `Ref2`, whose pairs may not be forwarded (S3). A longer
  sweep then found the compiler's own port passthrough
  (`__port_<name>`, inserted for a hole's adapter chain) in the same
  state; it now carries the same slot copy.

- **Extension values on the closure tier.** Landed 2026-09-09. The
  `compiled_handle` kit accepts `Ext<T>` arguments and returns: an
  argument is read from the installed table through `current_table_value`
  and downcast by the same `Wire::extract` the interpreter uses, and a
  return is written with `write_table_entry` through `Wire::inject`.
  Nothing changed in the engines, which already treated `Ext` as a
  table kind; the gap was only the macro's plan. Every library node with
  an `Ext` signature (the partition family, `streamer`) and every host
  node like them now runs on the closure tier and in hybrid kernels;
  none has a native form. `tests/ext_tiers.rs` is the differential.

No refinements remain recorded.
