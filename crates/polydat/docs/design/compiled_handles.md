# Compiled Non-Scalar Slots — Handles for `Str`, `Bytes`, `Json`, and `Ext`

**Status:** Proposed SRD 115, second revision (§2.2 states the table
handle's generation stamp as a design choice; §3 is restated by tier
around an engine-owned, fixed-entry table). Step 1 of §11, the arena
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
table-handle outputs. Tests in `tests/handle_boundaries.rs`. Step 5,
the value table, has landed as §3 specifies: a table is owned by the
engine that runs the native code, sized at compile time to one entry
per table-kind slot, written in place by the one helper that owns each
entry, and reached by helpers through the installation the engine makes
around its native call. An embedded cone borrows a table for one eval
and releases its entries and its arena bytes when the outputs are
copied out; a whole compiled kernel owns its table for its lifetime and
checks H4 after every run. A `Json`, `Ext`, or `Handle` value enters a
cone as a table handle and leaves as a clone of its entry, and a handle
from another cycle generation is refused. The typed JSON adapters
(`__u64_to_json` and siblings, `__str_to_json`, `json_to_str`) lower to
helpers that write and read the table, so a JSON value can be built and
serialized natively in one cone. The polymorphic `to_json` and
`identity` stay on P1 by the SRD-74 None rule, which excludes
None-tolerant nodes from cones. A handle-producing step in a provenance
kernel is never skipped as clean, since its storage belongs to the
cycle that ran it. The marshalling rule lives in one place,
`compile::marshal`, and the pure-P3 kernels carry a slot mask and their
outputs' port types, so their raw readers refuse handle slots and
`get_value` decodes by type. Tests in `tests/value_table.rs`. Step 6,
helpers and lowerings, has landed for every node it names: `printf`,
`json_array`, `json_object`, `to_json`, `json_text`, `tile_encode`, and
`tile_render` lower by the wire-typed classification of §6, run the
same body P1 runs, and agree with it across the type set. Cones admit a
None-tolerant node behind a member (§9). The P3 kernels lower the same
set by port type. A tile with a projection renders natively too: the
helper re-runs the body program per tuple through nested kernels, the
cone eval is re-entrant so those kernels' own cones run inside it, and
a helper that can panic as its P1 node does re-raises through the
longjmp path (§6.1). Getting there fixed a hybrid-kernel gap: a host-driven hybrid
kernel never began a root cycle, so string-producing segments grew the
arena across runs. Tests in `tests/variadic_lowering.rs`. Step 7, the
P2 closure form and the differential suite, has landed: the
`#[polydat_node]` macro emits a `compiled_handle` kit (§7) for every
node with a JSON, polymorphic, or variadic port whose other shapes fit
the buffer, including nodes with const-derived setup, so `printf`, the
JSON constructors and conversions, `json_with`, `json_merge`,
`tile_encode`, and `tile_render` (projections included) run as P2
closures; the P2 kernels own a value table, install it around every
run, begin a root cycle when a host drives them, validate H4 after
every run in debug builds, and read handle outputs through
`get_value`; hybrid kernels take the same closures. The differential
suite in `tests/handle_tiers.rs` generates random programs over the
string, JSON, and tile nodes and checks every output across the
interpreter, forced cones, P2, and pure P3 (`FUZZ_SEED`,
`FUZZ_ITERATIONS`), beside the hand-written corpus. Step 8, the docs,
has landed: engines.md §5–§7, type_system_alignment.md §5 and §7, and
jit_boundary.md's helper table and axiom section carry the handle
color, the helper ABI, and H1–H7, and the Polytile SRD's step 7 is
marked done. Every step of §11 has landed, including the native form
of a projection tile.
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
tag 10  table       [tag:2][kind:6][generation:24][entry:32]  engine-owned, single-writer per entry
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
- A **table** handle names an entry in the value table of the engine
  that ran the code (§3), sized at compile time with one entry per
  `Hdl1` slot whose values are not byte strings. The `kind` field
  records the value's variant, so a reader can refuse a handle of the
  wrong kind without touching the entry. The `generation` field is the
  low 24 bits of the cycle generation the entry was written in, and a
  read checks it against the table's current generation: a table handle
  that outlives its cycle fails by handle rather than naming whatever
  the entry holds now (H3). Where SRD 111 reserved these bits, Polydat
  spends them on the stamp, in every build, because the table is where
  a stale handle is cheapest to catch and most expensive to miss. Arena
  handles carry no stamp; their bound is the root cycle's reset (§4).

Byte-string values (`Str`, `Bytes`) use static or arena handles. Other
non-scalar values (`Json`, `Ext`, `Handle`) use table handles. A helper
that produces a `Str` writes bytes into the arena and returns an arena
handle; a helper that produces a `Json` writes an `Arc<serde_json::Value>`
into its own table entry and returns that entry's handle. The kind of
handle a port carries is decided by its `PortType` at build time, so
generated code never branches on the tag.

## 3. The value table

A value table is owned by the engine that runs native code, never by
the thread. Its shape is fixed when the code is compiled: one entry per
table-kind slot the code can write, numbered in step order, plus one
per table-kind boundary input where the engine has boundary inputs.
Each entry has exactly one writer, the helper lowered for the step that
owns the slot, which is told its entry number as an immediate and
replaces the entry in place every time it runs (H4). Nothing appends;
native code cannot grow the table, and an entry number outside it is
refused. A steady cycle therefore allocates nothing beyond the `Arc`
that enters an entry, and the table's size is a property of the program
rather than of how many values a cycle happened to make.

Native code reaches the table through a context the engine installs
around each native call and removes when it returns. A helper that runs
with no table installed has no owner for what it writes and panics.
The context is a thread-local pointer rather than a parameter in the
segment signature; it is set and cleared by the engine that holds the
`&mut` to the table, is restored on unwind, and nests correctly when one
engine's call contains another's. This keeps the entry-point signatures
unchanged while giving the table a single owner.

The two tiers use the table differently:

- **Embedded cones** in a P1 kernel borrow a table for one eval. The
  cone node sizes it at the top of `eval`, stamps it with the thread's
  cycle generation, writes its table-kind boundary inputs into their
  entries, runs, copies every output out, then clears the table and
  releases the cycle arena back to the mark it took before the call.
  Nothing a cone allocates outlives its eval, so a cycle's arena and
  table use is bounded by its largest cone rather than by the number of
  cone evals, and pulling an output a thousand times in one cycle costs
  no storage. The root cycle's reset (§4) remains the outer bound.
- **Whole compiled kernels** (pure P3 now, P2 with step 7) own a table
  for their lifetime, sized from the `(slot, entry)` pairs codegen
  returns. A kernel that a host drives directly begins a root cycle at
  each run; one wrapped by a state that owns the cycle takes the
  state's generation. After every run, in debug builds, the kernel
  checks that each table-kind slot holds a table handle of the current
  generation naming exactly its own entry and that the entry was
  written: the S9 validator, restated for handles (H4). A hybrid kernel
  owns one table across all its JIT segments, with entries numbered
  across segments at build.

A helper that produces a `Json` writes the `Arc<serde_json::Value>` into
its entry and returns the entry's handle; a helper that reads one
resolves the handle to a borrow of the entry and copies out what it
needs. A read checks the handle's tag, its generation against the
table's, and that the entry exists and was written, so a handle held
across a cycle is a deterministic failure by handle (the H3 tripwire).

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
  reset increments. A table handle carries the generation in its own
  bits (§2.2) and every read checks it, in every build. Arena handles
  carry no stamp; a handle-producing step in a provenance kernel is
  never skipped as clean, so no slot holds an arena handle from before
  the last reset, and a cone releases its arena bytes at the end of its
  eval (§3).
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
- Table-handle arguments and returns are handles. A producer takes the
  entry it owns as its first argument, an immediate from the layout,
  and writes through the table the engine installed around the call
  (§3); a reader resolves through the same table.
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

### 6.1 Wire-typed lowerings

A node whose P1 body dispatches on `Value` variants (`printf` and the
JSON constructors over `&[Value]`, `to_json`, `json_text`, and the tile
nodes over a `Value` port) has no fixed port types to lower by. Its
lowering is decided with the types of its wires, which the assembler
knows for every edge: `classify_node_typed(node, wire_types)` is the
classifier the P3 layout and the cone planner use, and it falls back to
the untyped classifier for every other node. The rules:

- **One code per argument.** Each wire type maps to a one-byte code
  (`u i f b s y j e h` for `U64 I64 F64 Bool Str Bytes Json Ext
  Handle`); the codes of a node's wires are interned as one static
  string, and its handle is baked into the call. A wire of any other
  type has no code, and the node stays on P1. This is the only place
  the advertised port types of a variadic node are set aside: the cone
  planner's rule that a wire equals its port type is waived exactly for
  the ops this classification produced, never by node name, so
  `str_concat`, whose helper takes strings, still needs string wires.
- **Arguments in a stack array.** Generated code stores the argument
  slots into an array in its own frame and passes its address; the
  helper decodes each argument by its code, scalars from bits, strings
  by reference from the arena or interner, table kinds from the
  installed table, and runs the same function the P1 node runs
  (`ParsedFormat::render_with`, `json_array_of`, `json_object_of`,
  `value_to_json`, `json_text_of`, `encode`, `TileProgram::render`).
  H7 holds by construction: there is one body per node.
- **Parsed constants are interned.** The parsed format of a `printf`,
  the encoding of a `tile_encode` spec, and the program of a
  `tile_render` skeleton are interned by content for the process, as
  static strings are, and the helper receives the address. They are
  immutable once made and outlive every kernel compiled from them.
- **Strings are not copied to be formatted.** `printf` takes its
  arguments as borrowed views (`FmtArg`), so a string argument is
  formatted from the arena in place. The JSON and tile helpers build
  owned `Value`s for their arguments, which copies a string once; a
  borrowed form of `encode` and `value_to_json` is a refinement.
- **Projections render inside the helper.** A skeleton with a
  projection re-runs its body program per tuple through nested
  kernels, at P3 exactly as at P1 and P2: the nested kernels never
  reset the arena, take their own cone scratch and tables, and install
  and restore their own value tables, so the render runs inside the
  calling engine's cycle and leaves only its text. For this the cone
  eval is re-entrant: it takes its scratch buffer and table out of
  their thread-local cells for the duration of the native call, so a
  nested cone eval finds the cells free. A helper whose body may panic
  as its P1 node panics (`printf`, the tile nodes) catches the panic
  and re-raises it through the longjmp path, since a panic cannot
  cross an `extern "C"` frame.

## 7. The P2 closure form

P2 is the equivalence oracle for P3 and stays one. A P2 closure for a
`Hdl1`-bearing node runs over the same slot buffer, arena, and table
that P3 uses and calls the same body the P1 node and the native helper
call. The `#[polydat_node]` macro already emits the `compiled_u64` kit
for byte strings (a `&str` argument resolves its handle, a `String`
result enters the arena); for the table kinds it emits a second kit,
`compiled_handle(entry_base, wire_types)`:

- **Eligibility.** The node has at least one shape the u64 kit cannot
  carry but the table can: a JSON port (`&serde_json::Value` or
  `Arc<serde_json::Value>`), a polymorphic `Value` port, or a variadic
  of anything but `u64`; every other argument is a one-slot carrier, a
  const, or a setup derived from consts; the return is a one-slot
  carrier or a JSON value. Session-static setup (`from = ()`), fallible
  bodies, tuple and dynamic returns, and split variadics stay on P1.
- **Entries and types from the kernel.** The kit takes the first
  value-table entry the node's table-kind outputs own and the type of
  each wire input. A JSON result is written to its entry through the
  installed table; a polymorphic or variadic argument decodes by its
  wire type (`decode_arg`), a JSON argument by handle
  (`read_table_json`). This is the P2 form of §6.1's type codes.
- **Setup recomputed.** A `#[poly_const]` value is a pure function of
  the node's consts, so the kit recomputes it from the captured consts
  at construction, once, and the closure borrows it; nothing on the
  node is borrowed by the closure.
- **The kernel owns the table.** The P2 kernels carry a value table
  sized from the `(slot, entry)` pairs the assembler numbers in node
  and port order, install it around every run, begin a root cycle
  when a host drives them and adopt the wrapping state's generation
  otherwise, and run the H4 validator after every run in debug builds,
  as they run the S9 validator for Ref pairs. Named handle outputs are
  read through `get_value`. Hybrid kernels take the same closures for
  the nodes they do not JIT, numbering their entries with the
  segments'.

The P1↔P2↔P3 equivalence suite (`tests/handle_tiers.rs`) pins all
tiers to the same bytes over a random corpus of string, JSON, and tile
programs, including tiles with projections, which run at P2 through
the closure and at P3 through the render helper.

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
- **H3 — Handles live one cycle.** An arena handle is valid from its
  write until the root cycle's next reset, and a cone's handles only
  until its eval returns; a table handle is valid within the generation
  that wrote it. Nothing in the compiled tier carries one across; init
  and shared values are interned or copied out. *Tripwire: the
  generation stamp in every table handle (§2.2), checked on every read;
  handle-producing steps never skipped as clean (§4).*
- **H4 — One writer per table entry.** Exactly one (step, output port)
  writes an entry, is told its entry number at compile time, and
  republishes it every execution; every reader reads inside the
  writer's validity interval. This is S3 and S4 for the table.
  *Tripwire: the post-run validator of §3, the S9(a) assertion restated
  for handles: each table-kind slot's handle names its own entry, in
  the current generation, and the entry was written.*
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
  on P1. The wire-typed lowerings of §6.1 read the wire types the
  assembler already established; they change nothing about them.
- **The SRD-74 None rule is refined, not relaxed.** A fused cone is one
  node to the kernel guard: a None on any boundary input makes every
  output None. A node that tolerates None inputs and would have
  produced a value from one (`tile_encode` writes `null`, `to_json`
  keeps going, `identity` passes it through) may therefore join a cone
  only when every one of its inputs is an intra-cone wire from another
  member, where no None can arrive because the cone's own boundary is
  guarded and no lowered op produces None. Eligibility is decided in
  topological order on that basis, and the planner rejects a cone whose
  component split left such a node on the boundary. Fed by a kernel
  input, the node stays on P1 and keeps its P1 semantics exactly.

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
5. **The value table.** The engine-owned table of §3 with fixed entries
   assigned at codegen, the installed context helpers write through,
   table handles for `Json`, `Ext`, and `Handle` with the generation
   stamp, eval-scoped release in cones, the H4 validator in whole
   kernels, and boundary marshalling for them.
6. **Helpers and lowerings.** `printf`, the JSON constructors,
   `json_text`, `to_json`, `json_to_str`; then `tile_encode` and
   `tile_render` per SRD 114 §12 step 7, with tile statics interned and
   projection bodies activated as `for` bodies are. Landed by the
   wire-typed classification of §6.1, projections included: the helper
   activates the body program through nested kernels as P1 does, and
   the cone eval is re-entrant so that works inside a cone.
7. **P2 closures and differential tests.** `compiled_handle` kits for
   `Hdl1` nodes from the macro; the P1↔P2↔P3 suite over a random
   corpus of string, JSON, and tile programs; the S9-style post-run
   assertions for table entries in the P2 kernels. Landed.
8. **Docs.** engines.md §5 and §7, type_system_alignment.md §5–§7,
   jit_boundary.md's axiom section with H1–H7, and the Polytile SRD's
   step 7 marked done. Landed.

Each step lands with its tests and leaves the previous surfaces working.
Step 1 is independent of the rest and is a fix in its own right.
