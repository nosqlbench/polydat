# Compiled By-Reference Slots — `Str`, `Bytes`, `Json`, `Ext`, and `Handle` in Compiled Kernels

**Purpose.** The compiled tiers (the closure tier P2, P3, and the pure
native tier behind it) run over a flat buffer of `u64` slots. A string, byte
string, JSON value, extension value, or handle value does not fit a
slot, so it needs a slot representation that keeps the runtime model's
one rule ([Runtime Model](runtime_model.md), R1): an output stands
until an input in its provenance is written, and nothing else reclaims
it. This document gives those five types that representation, states
who owns the storage behind it, and says what crosses the boundary
between the interpreter and the compiled tiers. It amends
[Engines](engines.md) §6, [Type-System Alignment](type_system_alignment.md)
§5–§7, and the S-axioms of [JIT Boundary](jit_boundary.md).

**Ownership.** Polydat owns the slot color, the scratch ownership rule,
the boundary marshalling, the closure kit, and the equivalence tests.
Hosts own nothing here; a host sees the same `Value`s it sees today, at
the same P1 boundary, and every read is a copy.

**Companion documents.**
[Runtime Model](runtime_model.md) (R1, the one rule; L1, state per
fiber), [Composition Substrate](composition_substrate.md) (L1, each
layer owns its state), [Engines](engines.md) (the engine lattice and
the slot colors), [JIT Boundary](jit_boundary.md) (axioms S1–S10),
[Type-System Alignment](type_system_alignment.md) (type planes),
[Polytile](polytile.md) §7 (the first consumer).

## 1. The problem

Every port has a static slot color. `Imm1` and `Imm2` slots hold
immediate values, `Ref2` slots hold a `(ptr, len)` pair with a proven
owner, and the classifier admits a node to a compiled tier only when
its ports have a lowering. The five by-reference types need a slot
representation of their own, and three limits decide which one it can
be.

- **There is no cycle.** The runtime model has one rule, R1: a step is
  current until an input in its provenance changes, and invalidation is
  per input, never all at once. A representation whose storage is
  reclaimed at a periodic reset is an all-or-none invalidation the
  model does not have, and a step exempted from R1 to survive that
  reset is a hole in the model's caching, not a refinement of it.
- **No storage belongs to a thread.** L1
  ([Composition Substrate](composition_substrate.md)) puts every piece
  of state in the fiber's own `PolydatState`; the program is shared and
  read-only. Storage that belongs to no state has a lifetime that must
  be legislated by an axiom instead of following from ownership.
- **A reference is one dereference.** S7
  ([JIT Boundary](jit_boundary.md)) allows exactly one static
  dereference per reference access, and forbids index tables and arena
  handles for that reason: a table handle is looked up in a table and
  an arena handle is decoded against a chunk list, both the second hop
  S7 exists to prevent.

The typed vectors already meet all three. A `VecF32` output is a `Ref2`
pair into a scratch buffer the kernel's state owns for exactly that
(step, port), republished when the step runs and valid until it runs
again (S3, S4). That is the model's own answer, and it is the whole of
this document: the five by-reference types are `Ref2` too.

## 2. The slot color

`SlotColor` has three members, as S1 states, and it is total:

| Color | Width | Members | Meaning |
|---|---:|---|---|
| `Imm1` | 1 | scalars | Immediate data; never an address. |
| `Imm2` | 2 | `U128/I128`, `Reg128` views | Two immediate limbs; never an address. |
| `Ref2` | 2 | all `Vec*` types, `Str`, `Bytes`, `Json`, `Ext`, `Handle` | Engine-internal `(ptr, len)` with a proven owner. |

For a string or byte string the pair is the bytes: `ptr` to the first
byte, `len` the byte count, the same view a `&str` or `&[u8]` is. For
a JSON, extension, or handle value the pair is a one-element slice of
`Value`: `ptr` to the `Value`, `len` one. A `len` of zero on such a
slot is the empty string, the empty byte string, or `None`, which is
how an unset extern of these types reads where nothing keeps a `None`
mask.

`PortType::scratch_elem()` names the scratch entry a producer of each
`Ref2` type owns (§3): `Str`, `Bytes`, or `Value` beside the seven
vector element types. Generated native code treats a pair as it treats
a vector's: it loads and stores the two slots and passes them to a
helper; it never dereferences one (S7 belongs to the helper) and never
compares two pairs for the equality of what they name.

## 3. Ownership

Every `Ref2` pair in a slot buffer points into storage with exactly one
owner, decided at build, and the owner is always part of the state the
buffer belongs to or something that outlives it:

- **A step's own scratch (S3).** A step that produces a `Ref2` value
  owns one scratch entry per `Ref2` output port, in port order, held by
  the kernel's state beside its slot buffer. When the step runs it
  writes the value into its entry (a string's bytes in place, reusing
  the allocation; a value replaced) and republishes the pair. The pair
  is valid from that publish until the step's next run (S4), which is
  exactly the interval R1 gives the output: the step runs again only
  when an input in its provenance is written.
- **An extern's stored value.** An extern of a `Ref2` type keeps its
  `Value` in the state's extern table; its slots hold the pair into
  that value, written through at every `set_input` and at build. The
  pair is valid until the host writes the extern again, which is when
  R1 invalidates the extern's dependents anyway.
- **An interned constant.** A string literal folded at build is interned
  once for the process (`kernel::intern`), and its step publishes the
  pair into the interned bytes, which never move and are never freed.
  The step owns no scratch and copies nothing.
- **A boundary value, for the call.** When the interpreter evaluates a
  native cone, each `Ref2` boundary input is borrowed into its slot for
  the duration of the call: the pair points into the `Value` the
  interpreter holds, which outlives the call. Every `Ref2` boundary
  output is copied out to an owned `Value` before the call returns (§4).

A pair is never forwarded. A copy step (`identity`, the compiler's
`__port_` passthrough, a type assertion) of a `Ref2` value copies the
elements into its own scratch entry and publishes its own pair, as S3
requires for vectors; an upstream pair reached through a copy would tie
the copy's validity to a producer it does not depend on.

Storage never belongs to a thread. Where a node needs state of its own
to evaluate (a native cone's slot buffer; the kernels a tile render
keeps over its projection bodies; the memo of the last spec a
`dynamic_weighted_select` parsed, in a `ScratchElem::State` entry the
node types for itself and fills on first use), it declares that state
as scratch entries through `PolydatNode::scratch_layout`, and the
state that evaluates it hands them in through `PolydatNode::eval_in`.
The node itself is shared by every state of the program and holds
nothing that changes. A clone of a state is a new state: its scratch entries and
extern values are copies of its own, and every pair in its buffer is
republished into them as it is made, so no state's buffer points into
another's storage (S3); an entry that holds kernels starts empty.

The interpreter holds owned `Value`s in its buffers and needs none of
this; its by-reference outputs are cached under R1 like every other
output, and were throughout.

## 4. Boundary marshalling

`compile::marshal` is the one place that turns a `Value` into slots and
slots into a `Value`, and every compiled kernel's typed reader
(`get_value`, `pull`) and every cone boundary decodes with it by the
port's type:

| Direction | scalar | `Ref2` kind |
|---|---|---|
| `Value` → slots | the bits | the pair into the value (`borrow_pair`); valid while the value is |
| slots → `Value` | the bits | copied out (`decode_pair`): a fresh `Arc<str>`, `Arc<[u8]>`, `SliceArc`, or a clone of the `Value` |

The copy at the outbound boundary is the price of the host's contract:
a value read from any engine is the reader's own, valid for as long as
the reader holds it, whatever the kernel does next. The interpreter
never holds a pair, and no public API returns a borrow into a state's
buffers. Raw readers (`get`, `get_slot`) refuse a `Ref2` slot (S2) as
they always have for vectors.

Inside a cone every wire is exactly its port's type; there is no
lowering that fixes wire types itself.

## 5. The closure kit

The `#[polydat_node]` macro emits one of two closure kits for a node.
The `compiled_u64` kit serves a node whose every port is an immediate;
the `compiled_slot` kit, `compiled_slot(wire_types) -> CompiledSlotKit`,
serves every other node, over the same slot buffer, with the step's
scratch handed in:

- **Eligibility.** Every argument is an immediate of one or two slots,
  a typed vector slice, a `&str` or owned string, a `&[u8]` or owned
  byte string, a JSON port (`&serde_json::Value` or
  `Arc<serde_json::Value>`), an `Ext<T>` port, a polymorphic `Value`
  port, a variadic (a split variadic included), an `Option<T>` over an
  immediate, a `Config<T>` over an immediate or an owned string or byte
  string, a const, a const list, or a setup; and
  the return is an immediate, a vector, a string, a byte string, a JSON
  value, an `Ext<T>`, a polymorphic value, or a tuple of these. A
  fallible body (`-> Result<T, E>`, const arguments only) ran once at
  construction; its cached value is written by whichever kit its shape
  names. Dynamic returns and `Handle` downcasts stay on P1.
- **Scratch from the kit.** The kit declares one scratch entry per
  `Ref2` output, in port order (`ScratchElem::Str`, `Bytes`, `Value`,
  or a vector element), beside any entry the node keeps for itself
  (`Slots`, `Kernels`, `State`, which publish no pair), and the kernel
  allocates them in its state. A
  `Ref2` return is written into its entry and its pair republished; a
  polymorphic return encodes by the node's resolved output type
  (`derive_support::write_poly`), which is the type of the first value
  wire for a split variadic. A polymorphic return whose resolved color
  differs from the declared port's has no slot to land in, and the node
  stays interpreted.
- **Reads through the pair.** A `&str`, `&[u8]`, or `&[T]` argument is
  the pair's slice, borrowed for the closure's run (S4 keeps the
  producer's storage alive); an owned `String` or `Vec<u8>` argument is
  copied from it. A JSON or `Ext<T>` argument reads the `Value` the
  pair names (`derive_support::ref_value`) and downcasts by the same
  `Wire::extract` the interpreter uses. A polymorphic or variadic
  argument decodes by the wire types the kernel hands the kit
  (`derive_support::read_poly`), walking the slots by each wire's
  width.
- **Setup recomputed or captured.** A `#[poly_const]` value derived
  from consts is a pure function of them, so the kit recomputes it from
  the captured consts at construction, once. A session-static setup
  (`from = ()`) is cloned from the node.
- **`Option<T>` reads as `Some` in the kit.** The kit sees slot bits
  and no mask; a `None` on the closure tier and P3 is the kernel's
  per-slot mask (a step whose node does not accept `None` emits `None`
  without running), and pure
  native code refuses to run with an unset extern.

A node that supplies its own closure names it with
`compiled_slot = <path>` (`fn(&Node, &[PortType]) -> CompiledSlotKit`);
a node that keeps state per evaluating kernel names it with
`state = <path>` (a module with `layout(&Node) -> Vec<ScratchElem>` and
`eval(&Node, &mut [ScratchBuf], &[Value], &mut [Value])`). The tile
render node uses both: its closure renders straight into its own
string entry, and its body kernels live in a `ScratchElem::Kernels`
entry of the rendering state.

## 6. The native tier

Native code carries a `Ref2` pair as it carries any slot: it loads the
two words, passes them, and stores them, and never dereferences one.
The work on a by-reference value is done by the node's own kit (§5),
which native code calls in place through one helper, `jit_slot_call`:
the generated code gathers the step's input slots into its frame,
calls the helper with the kit's address, the frame, and the evaluating
state's scratch with the index of the step's first entry, and scatters
the outputs the kit wrote back into their slots. The kit writes the
value into the step's own entry and publishes the pair exactly as it
does when the step is a closure step, so §3 holds unchanged: the owner
of every pair native code produces is the entry the builder placed for
that step in the state that runs it. Every native function therefore
takes the state's scratch beside its slot buffer
(`fn(coords, buffer, scratch)`), and every site that runs native code
hands its own in: a hybrid kernel its scratch vector, a pure native
kernel its own, and a cone node the entries its `scratch_layout`
declares after its slot buffer.

The classifier lowers a pure node this way whenever it has no named
native lowering and a kit exists (`JitOp::SlotCall`), whatever the
colors of its ports, so the twelve `Ref2` types and every immediate
shape a kit accepts join segments and cones alike; a variadic or
polymorphic node's kit is built for the types of its wires, so inside a
cone its wires are read as the graph typed them, not as its ports
advertise. A nondeterministic node or a side channel lowers the same
way, and the kernel that runs the code keeps its currency its own: on
the hybrid kernel it is a segment by itself, so a segment of pure
nodes is never made never-current or observably rerun by it; on pure
native code a never-current step's clean flag is cleared at every
write and the cone guard yields to a write while one exists (R1.v),
and a side channel runs at every evaluation in which it is not
current, which on that tier is what a pull is. A node with no kit
stays interpreted, and only such a node keeps a program off pure
native code. The kits a function calls are
kept alive beside its code (`JitCode`), shared by every kernel
compiled from the program.

A slot call costs what the closure step it replaces cost, less the
step runner: one helper call, a gather and a scatter through the frame.
The string producers whose kits allocate an intermediate `String`
have a named lowering on the same ownership that writes into the
step's entry directly: `__u64_to_string`, `__i64_to_string`, and
`__f64_to_string` format their digits into the entry, `str_concat`
over string wires appends each input pair's bytes, and `json_to_str`
serializes the value the pair names into the entry
(`jit_u64_to_str` and siblings, jit_boundary.md). Each takes the
state's scratch and the step's entry index, the buffer and the output
slot, and its typed arguments, and publishes the pair itself; the
bytes are the ones the node's body produces on the interpreter, since
each uses the same formatter. Nothing in §3 changes for them, and a
shape the named lowering does not take (a concatenation over a mixed
wire) takes the slot call.

The vector and register groups have named lowerings on the same
ownership. A vector producer (`vec_add`, `vec_scale`, `vec_norm`,
`hash_vec`, `xxhash3_vec`, `reg_to_vec_f32`) owns one `F32` entry and
its helper writes the result there and publishes the pair; a vector
reduction (`vec_dot`, `vec_l2`, `vec_cosine`, `lid_mle`) returns its
`f64` bits; a register lane read (`reg_lane_f32`, `reg_lane_i16`,
`reg_lane_i64`) returns the lane as the word its port stores, and a
register producer whose body checks a bound or does what Cranelift has
no instruction for (`reg_with_lane_f32`, `reg_gather_f32`,
`vec_to_reg_f32`, `reg_mul_i8`) writes its word into the output slots.
Each helper takes the step's input words in order, and each runs the
function the node's body runs (`polydat_core::numeric::vector` and
`numeric::register` hold those functions; the node bodies in
`polydat-nodes` and the helpers both call them), so the bytes and the
failure messages are the body's,
including which SIMD kernel or scalar loop computed a dot product.
`reg_dot_f32` and `reg_shuffle_bytes` are inline instructions: the
former the body's fixed tree at f32 precision (`fmul.f32x4`, four
lane extracts, three adds, a promotion), the latter one `shuffle` with
the mask the node exposes as its constants. Each lowering is keyed on
the exact wire types its body is written for, and any adapted shape
takes the slot call.

## 7. Axioms

The slot-state axioms S1–S10 of [JIT Boundary](jit_boundary.md) are
the whole contract; this document adds no axioms of its own. The
citations a SAFETY comment or a test needs are:

- **S1** — `Str`, `Bytes`, `Json`, `Ext`, and `Handle` are `Ref2`, two
  slots, total and static. *Chokepoint: `PortType::slot_color()`;
  `PortType::scratch_elem()` names the entry.*
- **S2** — raw readers refuse the pair; the typed readers copy out.
- **S3** — one scratch entry per producing (step, port), republished
  every run; a copy step owns an entry of its own. *Tripwire: a kit
  with more publishing entries than `Ref2` outputs fails at
  construction (`assembly::scratch_pairs`).*
- **S4** — a pair is valid from its publish until its producer's next
  run, which R1 makes the output's own lifetime.
- **S7** — one dereference, in the closure or the boundary decode,
  never in generated code. *Tripwire: S10's source scan lists
  `compile::marshal`, the copy step, the string assertion, the
  slot-call helper's frame view in `jit/codegen.rs`, and the pair view
  the node macro emits in its slot kit as the only
  `from_raw_parts` sites outside the vector substrate.*
- **S8** — the interpreter is the oracle. *Tripwire: the random and
  corpus differentials over every tier, and the same for host-defined
  values.*
- **S9(a)** — after every run, every scratch-backed `Ref2` slot equals
  its entry's current `(ptr, len)`, in debug builds. A pair into
  interned bytes or a borrowed boundary value is not scratch-backed and
  is not checked.

## 8. Boundaries

- **Not a change to P1.** Typed eval, `Value`, and the host API are
  untouched. A program that never enters a compiled tier behaves as
  before, and every read from any tier is an owned `Value`.
- **Not a new string type.** `Str` is still `Arc<str>` at P1; the pair
  is its compiled representation only.
- **`Ext` and `Handle` stay opaque.** No slot form grants a downcast
  the interpreter would not grant. An `Ext` value crosses a boundary as
  its `Arc`, is forwarded or projected inside a closure, and comes back
  as the same `Arc`.
- **No coercion for eligibility.** The classifier never re-types a
  port to admit a node; a `Ref2` port is admitted as `Ref2` or the node
  stays on the tier that carries it.
- **The None rule is unchanged** ([None Semantics](none_semantics.md)).
  A `None` on a compiled kernel
  is the extern mask; a `Ref2` extern that is unset reads as an empty
  pair where nothing keeps the mask, and the consumers that tolerate
  `None` read it as `None` through `derive_support::ref_value`.

## 9. Verification

Each of these is checked by the suite, by contract:

- The slot color, width, and scratch element of every by-reference
  type; a raw reader refuses the pair on every compiled engine; pure
  native code carries a `Ref2` output through a slot call.
- Strings across the tiers: a read is an owned copy that outlives the
  next write.
- Value-port nodes, JSON, and tiles across the tiers, and a hybrid
  kernel's string output owned by its step.
- The tier differential: random string, JSON, and tile programs over
  the interpreter, forced cones, the closure tier, and the hybrid
  kernel, plus the corpus; repeated coordinates keep reference outputs
  current; a JSON output is replaced in place.
- Extension values and externs of every kind across the tiers.
- Every node at the corners of its inputs, the `u64` edges of a
  coordinate and the special values of an `f64` extern, with every
  output compared bit for bit against the interpreter's on the closure
  tier, the native kernel, and pure native code, and failures compared
  by message.
- The vector and register classifier picking the named lowering for
  each node's wire shape, and a program over both groups compiled as
  one pure native function that agrees with the interpreter.
- A kernel created from a shared program pointing its pairs into its
  own storage, read after the source state is dropped.
- A rendering state's body kernels created once and reused, and a
  clone of the set starting empty.
- S9(a)'s validator, on every closure-tier, hybrid, pure-native, and
  cone run in debug builds.
- S10 by source scan: no `thread_local!` holds a value, a pointer, or
  a state.

