---
type: specification
title: Type System
timestamp: 2026-09-25
description: The static PortType contract on wires, its runtime Value representation, and the adapter catalog between types.
tags: [types]
---

# Type System

This document specifies the static type of every wire (`PortType`),
the runtime representation of values (`Value`), and the adapter
catalog: the set of conversion nodes the compiler and the host
boundary use to convert a value of one type to another.

**Related specifications.**
[Type-System Alignment](type_system_alignment.md) (how these types map
to compiled slots, Cranelift, and JSON);
[None Semantics](none_semantics.md) (the `None` sentinel);
[Input Variance](input_variance.md) (converter nodes for inputs whose
written type varies); [composition_substrate.md](composition_substrate.md)
(the typed-slot axioms T1–T3).

**Implementation.** `polydat-core/src/ast.rs` (`PortType`, `Value`),
`polydat-core/src/library/convert.rs` (adapter nodes),
`polydat-core/src/compile/assembly.rs::auto_adapter` (catalog
dispatch), `polydat-core/src/kernel/state.rs::adapt_boundary_value`
(boundary application).

## Terms

- **Carrier.** The `Value` variant that stores a port type's values
  at runtime. Several port types can share one carrier; for example,
  every unsigned integer width is carried by `Value::U64`.
- **Bit-stuffing.** Storing a narrower type in a wider carrier's low
  bits, as `U8` in `Value::U64` or an `f32` bit pattern in
  `Value::U64`. The port type records how to read the bits.
- **Adapter.** A node that converts a value of one port type to
  another. The assembler inserts adapters between a producer and a
  consumer whose types differ (§6.1).
- **Class A and class B.** An adapter is class A when it produces a
  value for every input and never fails; class B when some input has
  no result (a string that does not parse, a float out of an integer's
  range), in which case the adapter panics. §3 defines both classes
  and lists their members.
- **Host boundary.** The point where a value from outside the graph
  (a host write, or a value copied from an outer scope) enters a
  typed input slot.

---

## 1. PortType — the static wire contract

`PortType` is the type of the value a wire passes from its producer to
its consumers. Every `Port` (input or output) on every node declares
one. The assembler checks that each producer and consumer agree, and
where they do not, it inserts an adapter node from the catalog if one
exists for the pair (§6.1).

| Variant | Width | Storage at runtime | Notes |
| --- | --- | --- | --- |
| `U64`     | 64-bit unsigned | `Value::U64(u64)` | Workhorse: hash outputs, counters, primary keys |
| `I64`     | 64-bit signed   | `Value::I64(i64)` | **Honest signed carrier** (alignment §2) — display/JSON render negatives correctly |
| `U32`/`U16`/`U8` | narrow unsigned | `Value::U64` (zero-extended) | Widen to `U64` automatically |
| `I32`/`I16`/`I8` | narrow signed | `Value::I64` (sign-extended) | Widen to `I64` automatically |
| `F64`     | 64-bit float    | `Value::F64(f64)` | IEEE 754 double |
| `F32`     | 32-bit float    | `Value::U64` (as `f32::to_bits() as u64`) | Bit-stuffed; widens to `F64` |
| `F16`     | 16-bit float    | `Value::U64` (as `f16::to_bits() as u64`) | binary16; widens exactly to F32/F64 |
| `U128`/`I128` | 128-bit int | `Value::U128`/`I128(Bits128)` | Two u64 limbs (keeps `Value` at align 8); no named native lowering — a closure on the closure tier, a slot call of the kit on the native engine |
| `Reg128`, `RegI8x16`, `RegI16x8`, `RegI32x4`, `RegI64x2`, `RegF16x8`, `RegF32x4`, `RegF64x2` | 128-bit SIMD word | `Value::Reg128(Bits128, RegLanes)` | One register word under 8 lane-views; reg→reg is a free bitcast retag (alignment §3) |
| `Bool`    | logical          | `Value::Bool(bool)` | Distinct runtime variant |
| `Str`     | UTF-8 string     | `Value::Str(Arc<str>)` | Cheap-clone via Arc |
| `Bytes`   | raw bytes        | `Value::Bytes(Arc<[u8]>)` | Cheap-clone via Arc |
| `Json`    | serde_json Value | `Value::Json(Arc<serde_json::Value>)` | Cheap-clone via Arc |
| `Ext`     | adapter-contributed | `Value::Ext(Box<dyn ReflectedValue>)` | UUIDs, timestamps, IPs |
| `Handle`  | type-erased Arc  | `Value::Handle(Arc<dyn Any + Send + Sync>)` | Datasets, prepared stmts |
| `VecF32`/`VecF64`/`VecF16` | float lanes | `Value::Vec*(SliceArc<T>)` | Typed slices; native CQL vector binding |
| `VecI8`/`VecI16`/`VecI32`/`VecI64` | int lanes | `Value::Vec*(SliceArc<T>)` | Complete cranelift lane family (alignment §4) |

The scalar-width set is the full cranelift scalar vocabulary
(every integer width in both signednesses, f16/f32/f64) minus
F128, which stable Rust cannot carry; the vector element set is
every cranelift lane type with a JSON Number projection. See
`type_system_alignment.md` for the derivation.

**Four storage classes** group the variants by how the runtime
stores them:

- **Single-word scalars** (`U8`–`U64`, `I8`–`I64`,
  `F16`/`F32`/`F64`, `Bool`) — one machine word per value, in three
  carriers (`Value::U64`, `Value::I64`, `Value::F64`) plus
  `Value::Bool`. The 64-bit types use their carrier directly, and the
  narrow widths are *bit-stuffed* into it. These types can be
  compiled to native code at P3.
- **Two-limb 128-bit** (`U128`, `I128`) — `Bits128([u64; 2])`, stored
  in two immediate slots (`Imm2`) in the compiled layout. There is no
  *named* native lowering (a translation written for the specific
  node): a 128-bit node runs its closure on the closure tier, and on
  the native engine the native code calls the node's closure kit (a
  *slot call*; [Compiled By-Reference Slots](compiled_handles.md) §6).
- **128-bit SIMD register plane** (`Reg128` + 7 lane-views) — a
  16-byte word with a `RegLanes` view tag; reg→reg retags are free
  bitcasts and the arithmetic ops JIT to native SIMD.
- **Arc-backed handles** (`Str`, `Bytes`, `Json`, the `Vec*` lane
  family, `Handle`, and the boxed `Ext`) — cloning one is a single
  atomic reference-count increment, and consumer fibers share one
  allocation.

The sections below specify each family. A slot's declared `PortType`
is always the type contract; the runtime [`Value`] variant (§2) is how
the value is stored.

### 1.1 Integer widths — `U8`/`U16`/`U32`/`U64`, `I8`/`I16`/`I32`/`I64`

The integer vocabulary is every cranelift scalar width in both
signednesses. At runtime **all unsigned widths share `Value::U64`**
(zero-extended into the low bits) and **all signed widths share
`Value::I64`** (sign-extended) — the *honest signed carrier*
(alignment §2): a negative `I32` is stored as a negative `i64`, so
its display and JSON projection render `-1`, not `4294967295`. The
`PortType` is the sole record of the declared width; the carrier
only knows "u-bits" or "s-bits".

- **Storage cost** — one `u64` slot regardless of width. There is
  no `Value::U8`; an `U8` wire is a `Value::U64` whose producer
  promises the high 56 bits are zero.
- **JIT (P3)** — because every width occupies one slot, narrow widths
  compile at no extra cost: widening is `uextend`/`sextend`, narrowing
  is a mask, all on the one `compiled_u64` register, with no boxing.
- **Conversions** — widening (`u8 → u64`, `u32 → i64`, …) is class A
  and auto-inserted; narrowing is class B (range-checked, boundary
  only). See §3.
- **Use** — `U64` is the workhorse (hashes, counters, keys); the
  narrow widths exist so a node can declare the *real* column width
  (CQL `tinyint`/`smallint`/`int`) and bind it natively without a
  widening detour.

### 1.2 128-bit integers — `U128` / `I128`

The cranelift `I128` lane under both signedness readings. Unlike
the ≤64-bit widths, a 128-bit value does not fit a single `u64`
slot, so it has its **own `Value` variants** (`U128`, `I128`),
each carrying a [`Bits128`] — two little-endian `u64` limbs
(`[lo, hi]`). Storing two limbs instead of a raw Rust `u128` keeps
`Value`'s alignment at 8 and its footprint inside the 40-byte
buffer-slot envelope, which a size probe in the suite guards.

- **No *named* native lowering** — 128-bit integer operations run
  as ordinary evals on the interpreter; on the closure tier the node
  runs its closure, and on the native engine it is a slot call of
  its kit from native code (`JitOp::SlotCall`), over two immediate
  slots. The result is the same on all four engines (the interpreter,
  the closure tier, native, and pure native). For the arithmetic, the
  two limbs are reassembled into a native `u128`/`i128` in two
  register moves and split again afterwards.
- **JSON** — projects as a **decimal string**, not a JSON Number
  (which tops out at the `u64`/`i64`/`f64` leaves); the extractor
  also accepts an in-range Number for convenience.
- **`Bytes`** — exactly 16 little-endian bytes.
- **Conversions** — every ≤64-bit integer and `Bool` widens into
  `U128`/`I128` (class A); `→ f64` is class A; every narrowing back
  out (`→ {u8…i64, f16, f32}`) is a range-checked class-B boundary
  adapter.

### 1.3 Floats — `F16` / `F32` / `F64`

IEEE 754 binary16 / binary32 / binary64. `F64` is the primary
float and the canonical widening target. `F32` and `F16` are
**bit-stuffed**: a node that outputs `F32` stores
`f32::to_bits() as u64` in `Value::U64` (and `F16` stores its
16-bit pattern the same way), so they cost one slot and JIT like
integers. A *host-written* float slot may instead arrive as a
materialised `Value::F64`; [`Value::satisfies_slot`] accepts both
forms for both: the bit-stuffed `Value::U64` a node produces
and the materialised `Value::F64` a host writes (§4).

- **`F128` is absent** — stable Rust has no `f128` carrier, so the
  scalar float set stops at `F64` (alignment §2).
- **Conversions** — `f16 → f32 → f64` widens exactly (every `f16`
  is exact in `f32`/`f64`). `int → f64` is always class A (total,
  rounds above 2⁵³); `int → f32`/`f16` is class A only when the
  integer fits the target's exact-int window (`u8`/`i8` → `f16`,
  `u8`/`u16`/`i8`/`i16` → `f32`), else class B. Float → int is
  always explicit/boundary (rounding + NaN/Inf are user choices).

### 1.4 Boolean — `Bool`

A distinct `Value::Bool(bool)` variant (not bit-stuffed) used by
conditional ops, selection nodes, and flag computation. `Bool`
widens to **every** numeric width as `1`/`0` (class A) and every
numeric reduces to `Bool` by a nonzero test (class A) — both
directions are total, so the matrix row/column for `Bool` is
almost entirely `A`.

### 1.5 SIMD register plane — `Reg128` + lane views

A `Reg128` is a **128-bit SIMD register word** carried as
`Value::Reg128(Bits128, RegLanes)`: the 16 bytes plus a
[`RegLanes`] tag recording the *current* interpretation. The seven
typed `PortType`s name a homogeneous lane reading of the same
word:

| PortType | Lanes | PortType | Lanes |
| --- | --- | --- | --- |
| `RegI8x16` | `[i8; 16]` | `RegI64x2` | `[i64; 2]` |
| `RegI16x8` | `[i16; 8]` | `RegF16x8` | `[f16; 8]` |
| `RegI32x4` | `[i32; 4]` | `RegF32x4` | `[f32; 4]` |
| `Reg128` (`Raw`) | algorithm-defined bytes | `RegF64x2` | `[f64; 2]` |

- **Views are free bitcasts** — every reg→reg wire is healed by
  the assembler with a `__reg_view_*` retag node (one per view,
  `__reg_view_raw` … `__reg_view_f64x2`) that changes the `RegLanes`
  tag and touches **no bits** (alignment §3). They are registered
  nodes like every other adapter, so a program and the conversion
  fuzzer can both call them by name.
  A word can be `[i64; 2]` for one op, raw bytes for a shuffle, and
  `[f32; 4]` for a dot product, at zero cost. This is the one
  family where every intra-plane conversion is class A and every
  cross-plane conversion (reg ↔ scalar/container/vec) is absent —
  a closed clique (§3).
- **Native SIMD JIT** — the element-wise ops (`add`/`sub`/`mul`
  across each lane family) and `reg_dot_f32` compile to native
  vector instructions. Integer lane arithmetic **wraps** (modular);
  range-checking belongs to the scalar adapter system, not the
  register ops.
- **Determinism** — `reg_dot_f32` uses a *fixed* reduction tree
  `((l0+l1)+(l2+l3))` so float dot products are bit-reproducible
  across machines (determinism rule D2).
- **Use** — SWAR/state-word algorithms, fixed-width SIMD kernels,
  and `reg_shuffle_bytes` (arbitrary byte permutation from a
  16-entry const mask). Nodes live in `polydat-nodes/src/register.rs`;
  the retag nodes in `polydat-core/src/library/register_view.rs`.

### 1.6 Strings, bytes, JSON — `Str` / `Bytes` / `Json`

The Arc-backed scalar containers. Each clones in one atomic
increment with no allocation:

- **`Str`** — `Arc<str>`, UTF-8. The universal sink: every numeric,
  `Bool`, and `Json` renders to `Str` via Display (class A).
  Template interpolation shares the string by pointer rather than
  allocating a copy on each evaluation.
- **`Bytes`** — `Arc<[u8]>`. Little-endian serialisation target for
  every numeric, `Bool`, and the `Vec*` lanes. `Bytes ↔ Str` is
  lowercase hex; `Bytes ↔ Json` is a hex `Json::String`. Unsigned
  byte buffers are `Bytes`, not a `Vec` lane (§3.1).
- **`Json`** — `Arc<serde_json::Value>`. Structured capture and
  result-body projection share the tree by Arc; consumers that need
  an owned tree deep-clone at the use site. Non-finite floats make
  `f64`/`f32`/`VecF32 → Json` class B (`serde_json` rejects them).

### 1.7 Typed vector lanes — `VecF32`/`VecF64`/`VecF16`, `VecI8`/`VecI16`/`VecI32`/`VecI64`

Typed slices (`Value::Vec*(SliceArc<T>)`) that pass vector data
from accessors to native-binding adapters with **no string-format or
byte-serialise step on any evaluation**. The element set is every
cranelift lane type that has a JSON Number projection — completing
the lane family alongside the register plane (alignment §4).

- **[`SliceArc<T>`]** — a typed slice handle with two storage
  modes: *owned* (`Arc<[T]>`, one allocation) or *zero-copy* (a raw
  `(ptr, len)` borrow into a long-lived owner such as an mmap'd
  dataset, kept alive by the owner's Arc). Clone is one atomic
  increment either way.
- **Native binding** — `VecF32`/`VecF64`/`VecF16` bind CQL
  `vector<float|double|half_float, N>`; `VecI8`/`VecI16`/`VecI32`/
  `VecI64` bind `vector<tinyint|smallint|int|bigint, N>`. The
  adapter handles dataset→column precision at bind time.
- **Catalog coverage** — all seven lanes inter-convert
  element-wise (widening lanes class A, narrowing / `float→int`
  class B, panic-on-bad-element like the scalar adapters) and each
  serialises to / parses from `Bytes` / `Json` / `Str` (§3). Only
  `Vec → scalar` is deliberately absent, because no reduction is
  canonical (§3.3).

### 1.8 Type-erased — `Ext` / `Handle`

Opaque handles the producer mints and the consumer downcasts;
never in the adapter catalog (§3).

- **`Ext`** — `Box<dyn ReflectedValue>`. Protocol-native values
  (UUIDs, timestamps, inet addresses) that flow through Polydat
  without boxing to strings. Consumers reach typed access via
  `ReflectedValue::try_as_str` / `try_as_u64` / `to_json_value` / …
  at the consume site.
- **`Handle`** — `Arc<dyn Any + Send + Sync>`. A resolved resource
  (dataset, prepared statement); the producer node (`dataset_open`,
  …) populates it and the consumer downcasts with
  `Value::as_handle::<T>()`. Each evaluation that reads it costs one
  `Arc::clone` and no allocation.

---

## 2. Value — the runtime representation

`Value` has fewer variants than `PortType` because
the bit-stuffed numeric forms share two storage
slots:

```rust
pub enum Value {
    U64(u64),                          // also U32/U16/U8 (zero-extended) and the F32/F16 bit patterns
    I64(i64),                          // honest signed carrier; also I32/I16/I8 (sign-extended)
    U128(Bits128), I128(Bits128),      // two u64 limbs (align-8 envelope)
    Reg128(Bits128, RegLanes),         // 128-bit SIMD word + its current lane-view tag
    F64(f64),                          // F64; a host-written F32/F16 slot may arrive materialised here
    Bool(bool),
    Str(Arc<str>),
    Bytes(Arc<[u8]>),
    Json(Arc<serde_json::Value>),
    Ext(Box<dyn ReflectedValue>),
    Handle(Arc<dyn Any + Send + Sync>),
    VecF32(SliceArc<f32>), VecF64(SliceArc<f64>), VecF16(SliceArc<half::f16>),
    VecI8(SliceArc<i8>), VecI16(SliceArc<i16>),
    VecI32(SliceArc<i32>), VecI64(SliceArc<i64>),
    None,                              // sentinel for absent / uninit
}
```

Note the floats: an `F32`-typed *node output* stores its bit
pattern in `Value::U64` (the `Wire for f32` impl in `derive_support.rs`), while
a host-written `F32` slot value may arrive as `Value::F64`;
`satisfies_slot` accepts both. `F16` follows the same dual
convention.

`Value::None` is the *absent* sentinel (§2.2).

`Value::port_type()` reports the *runtime variant's*
PortType, which collapses the narrow widths into their
carriers — U32/U16/U8 → U64, I32/I16/I8 → I64, F32/F16
→ their stuffed carrier — (or returns `PortType::U64`
for `None` as a placeholder). The static slot's declared `PortType`
is the canonical type contract; `Value::port_type()`
is only used at boundary check sites
(`adapt_boundary_value`) to detect mismatches.

### 2.1 Carriers

Three helper types back the multi-`PortType` variants:

- **`Bits128([u64; 2])`** — little-endian limbs (`[lo, hi]`)
  behind `U128`, `I128`, and `Reg128`. The `[u64; 2]` (vs a raw
  `u128`) keeps `Value` at alignment 8 and inside the 40-byte slot
  envelope; reassembly to a native `u128`/`i128` is two register
  moves. Provides per-lane views (`lanes_i8` → `[i8; 16]`, …,
  `lanes_f16`/`f32`/`f64`) that the register plane reads.
- **`RegLanes`** — the view tag inside `Reg128`: `Raw` (algorithm-
  defined heterogeneous bytes) or one of the seven homogeneous
  readings (`I8x16` … `F64x2`). Changing the tag is a free bitcast
  (§1.5); the bytes never move.
- **`SliceArc<T>`** — the typed-slice handle behind every `Vec*`
  lane. Holds `(owner: Arc<dyn Any>, ptr, len)` so it supports both
  an owned `Arc<[T]>` and a zero-copy borrow into an mmap'd owner;
  clone is one atomic increment (§1.7).

### 2.2 The `None` sentinel

`Value::None` is the *absent* marker. It appears in freshly allocated
buffer slots before the first evaluation and as the "no value yet"
state of optional ports. Per [none_semantics.md](none_semantics.md) it
propagates through node evaluation: a node whose inputs include `None`
emits `None` on every output unless it declares
`PolydatNode::accepts_none_inputs()`. This holds on the interpreter,
the closure tier, and native; pure native refuses to run while an
extern is unset, so no `None` arises there. `None` is not a
`PortType`: no wire declares `None`, and `port_type()` reports `U64`
as a placeholder.

---

## 3. Adapter catalogs

Two catalogs live in
`polydat-core/src/compile/assembly.rs`:

```rust
pub fn auto_adapter(from: PortType, to: PortType) -> Option<Box<dyn PolydatNode>>;
pub fn boundary_adapter(from: PortType, to: PortType) -> Option<Box<dyn PolydatNode>>;
```

Each function takes the source type `from` and the target type `to`
and returns the adapter node that converts a `from` value to `to`, or
`None` when the catalog has no adapter for the pair.

- **`auto_adapter`** is used for wires inside a graph. The assembler
  calls it when a producer node's output `PortType` differs from a
  consumer node's input `PortType`. It is strict: it never returns an
  adapter that can panic on its input, so any wire mismatch it cannot
  resolve is a compile-time error the author can see.
- **`boundary_adapter`** is used at the host boundary, through
  `adapt_boundary_value`. It is a strict superset of `auto_adapter`:
  it tries `auto_adapter` first and then adds every class-B adapter of
  §3 (the narrowings, and the `Str`/`Bytes`/`Json` parsers and
  extractors) for boundary values whose source is text or whose
  conversion can lose information.

When either function returns `Some(node)`, that adapter node is
inserted into the wire chain. When it returns `None`, the boundary
returns `WriteError::TypeMismatch`, or at compile time the assembler
returns an error, with `from` and `to` in the diagnostic.

The matrix below is **generated from the two catalog
functions** in `compile/assembly.rs` (the cell-level source of
truth) and covers every conversion type that has at least one
catalog entry. Cells: **A** (in `auto_adapter` — intra-graph +
boundary, never panics), **B** (in `boundary_adapter` only —
lossy/parseable/shape-checking, can panic on bad input), **·**
(no adapter — fails with `TypeMismatch`), **─** (identity).

The narrow widths (`u8`/`i8`/`u16`/`i16`/`f16`) live in
`library/polyfill_narrow.rs`, the 128-bit integers in
`library/polyfill_128.rs`, and the macro-generated
matrix-completion adapters (every remaining narrowing + the full
vector-lane family) in `library/polyfill_complete.rs`. The matrix
is **complete** — every meaningful pair has an adapter, with only
the deliberate scalar↔vector exclusion left as `·` (§3.3). The
catalog's own property test enforces both the
widening invariant and total coverage.

```
from╲to  u8  i8 u16 i16 f16 u32 i32 f32 u64 i64 f64 u12 i12  bl str  by  js  vF  vI  vD  vL  vH  vS  v8
u8        ─   B   A   A   A   A   A   A   A   A   A   A   A   A   A   A   A   ·   ·   ·   ·   ·   ·   ·
i8        B   ─   B   A   A   B   A   A   B   A   A   B   A   A   A   A   A   ·   ·   ·   ·   ·   ·   ·
u16       B   B   ─   B   B   A   A   A   A   A   A   A   A   A   A   A   A   ·   ·   ·   ·   ·   ·   ·
i16       B   B   B   ─   B   B   A   A   B   A   A   B   A   A   A   A   A   ·   ·   ·   ·   ·   ·   ·
f16       B   B   B   B   ─   B   B   A   B   B   A   B   B   A   A   A   B   ·   ·   ·   ·   ·   ·   ·
u32       B   B   B   B   B   ─   B   B   A   A   A   A   A   A   A   A   A   ·   ·   ·   ·   ·   ·   ·
i32       B   B   B   B   B   B   ─   A   B   A   A   B   A   A   A   A   A   ·   ·   ·   ·   ·   ·   ·
f32       B   B   B   B   B   B   B   ─   B   B   A   B   B   A   A   A   B   ·   ·   ·   ·   ·   ·   ·
u64       B   B   B   B   B   B   B   B   ─   B   A   A   A   A   A   A   A   ·   ·   ·   ·   ·   ·   ·
i64       B   B   B   B   B   B   B   B   B   ─   A   B   A   A   A   A   A   ·   ·   ·   ·   ·   ·   ·
f64       B   B   B   B   B   B   B   B   B   B   ─   B   B   A   A   A   B   ·   ·   ·   ·   ·   ·   ·
u12       B   B   B   B   B   B   B   B   B   B   A   ─   B   A   A   A   A   ·   ·   ·   ·   ·   ·   ·
i12       B   B   B   B   B   B   B   B   B   B   A   B   ─   A   A   A   A   ·   ·   ·   ·   ·   ·   ·
bl        A   A   A   A   A   A   A   A   A   A   A   A   A   ─   A   A   A   ·   ·   ·   ·   ·   ·   ·
str       B   B   B   B   B   B   B   B   B   B   B   B   B   B   ─   B   B   B   B   B   B   B   B   B
by        B   B   B   B   B   B   B   B   B   B   B   B   B   B   B   ─   B   B   B   B   B   B   B   B
js        B   B   B   B   B   B   B   B   B   B   B   B   B   B   A   B   ─   B   B   B   B   B   B   B
vF        ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   B   A   B   ─   B   A   B   B   B   B
vI        ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   A   A   A   A   ─   A   A   B   B   B
vD        ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   B   A   B   B   B   ─   B   B   B   B
vL        ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   A   A   A   B   B   A   ─   B   B   B
vH        ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   B   A   B   A   B   A   B   ─   B   B
vS        ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   A   A   A   A   A   A   A   B   ─   B
v8        ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   A   A   A   A   A   A   A   A   A   ─

  A = always-defined (auto_adapter: intra-graph + boundary)
  B = can panic on input (boundary_adapter only — keeps intra-graph strict)
      A is totality, not losslessness: a conversion that always
      produces a value is class A even where it rounds. `i32 → f32`
      and `i64 → f64` round past their mantissa, and `VecI32 → VecF32`
      rounds per lane; none can fail, so all three are A. What makes a
      pair B is that some input has no answer — a float that does not
      fit an integer, a string that does not parse.
  · = no adapter (only scalar↔vector, intentionally — see §3.3)
  ─ = identity (no adapter needed)
  abbrev: u12=U128 i12=I128 bl=Bool by=Bytes js=Json
          vF=VecF32 vI=VecI32 vD=VecF64 vL=VecI64 vH=VecF16 vS=VecI16 v8=VecI8
```

The grid is **complete**: every meaningful pair has an adapter, and
the only `·` cells are **scalar ↔ vector**, which are deliberately
undefined (§3.3). Two type groups have no row or column in the grid:

- **Register views** (`Reg128`, `RegI8x16` … `RegF64x2`) — any
  reg→reg pair is class A via a zero-cost `__reg_view_*` retag; reg ↔
  scalar/container/vec is never in the catalog (a closed clique).
- **`Ext` / `Handle`** — never adapted; typed access happens at
  the consume site (`ReflectedValue` / `as_handle::<T>()`).

**Class A — always-defined** (lossless or fully-
defined for every valid runtime input). The
catalog's test encodes this set as a predicate
and proves `auto_adapter` covers all of it.

Always-defined is weaker than lossless, and the two are not
interchangeable: `u64 → f64` and `u64 → bool` are both class A and
neither keeps the value. Whether a conversion keeps the value is
computed from the two types' numeric domains and is what the
`_strict` embedding surfaces gate on — see
[expression_engine.md](expression_engine.md) §5.4.2.

- **Numeric widening** — the rule, which has no exceptions:
  unsigned → any **strictly-wider** integer of either signedness
  (`u8 → i16`, `u32 → i64`, `u16 → u128`, …); signed → any wider
  signed (`i8 → i64`, `i32 → i128`); float → wider float
  (`f16 → f32 → f64`); and `int → f64` always (the canonical float,
  total even when it rounds above 2⁵³). `int → f32`/`f16` joins
  class A exactly when the integer fits the target's exact-int
  window (`u8`/`i8 → f16`; `u8`/`u16`/`i8`/`i16 → f32`); otherwise
  it is a class-B rounding cast.
- **X → Str** — every scalar numeric (narrow widths and
  `U128`/`I128` included) plus `Bool` and `Json` render as a
  string via Display.
- **Bool ↔ numeric** — 1/0 mapping out, nonzero test in, both
  total and never panicking. Covers **every** numeric width in both
  directions, including `U128`/`I128`.
- **X → Bytes** — little-endian serialize for every
  scalar numeric (narrow + 128-bit included), Bool,
  and every `Vec*` lane; always succeeds.
- **X → Json** — wraps as the corresponding
  `Json::Number` / `Json::Bool` / `Json::Array` (or
  a decimal string for `U128`/`I128`, which JSON
  numbers can't hold). Integer and Bool sources
  never panic; the integer lanes (`VecI8`…`VecI64`)
  are in this class because every integer element is
  representable. `F64/F32/F16→Json` and the float
  lanes `VecF32/VecF64/VecF16→Json` are class B
  because non-finite floats are not representable in
  standard JSON.
- **Widening lane casts** — every element-wise
  widening (`VecI8`→any, `VecI16`→…,
  `VecI32→VecF32/VecF64/VecI64`, `VecI64→VecF64`,
  `VecF16→VecF32/VecF64`), lossless per element.

**Class B — can panic on input** (lossy, parseable,
or shape-checking):

- **Numeric narrowings + non-widening casts** —
  `U64→{U32, I64, I32, F32}`, `I64→{U64, U32, I32,
  F32}`, `F64→{U64, U32, I64, I32, F32}`, etc.
  Range-checked; panic on out-of-range. Float→int
  also panics on NaN/Inf. Narrow targets are
  reachable from the wide carriers
  (`{U64,U32,U16,I64,F64}→U8`,
  `{U64,U32,I64,F64}→U16`, the `I*→I8/I16` set,
  `{F64,F32,U64}→F16`); 128-bit narrows to every
  ≤64-bit numeric (`U128/I128→{u8…i64, f16, f32}`),
  the cross-sign `U128↔I128`, and every
  `{f16,f32,f64}→{U128,I128}` cast.
- **Str → X parsers** — every scalar numeric
  (`Str→{Bool, U8…I128, F16…F64}`) plus
  `Str→{Bytes, Json}` and `Str→` every `Vec*` lane. Trim +
  parse; panic on unparseable. Workload-param flow
  (YAML interpolation, comma-split iter-values)
  lives here.
- **Bytes → X parsers** — length-checked, little-
  endian decode into every scalar numeric (narrow +
  128-bit), `Bool`, `Str`, `Json`, and every `Vec*`
  lane. Numeric targets require exactly
  `sizeof(N)` bytes; Vec targets require a multiple
  of `sizeof(element)`; panic on wrong length.
- **Json → X extractors** — shape-checked into every
  scalar numeric (narrow + 128-bit), `Bool`,
  `Bytes`, and every `Vec*` lane. `Json::Number`
  expected for numerics, `Json::Bool` for Bool,
  `Json::Array` for Vec, `Json::String` (hex) for
  Bytes. Panic on mismatch.
- **Every float scalar / lane → `Json` / `Str`**
  (`F64/F32/F16→Json`, `VecF32/VecF64/VecF16→Json`,
  `VecF32/VecF64/VecF16→Str`) — class B because a
  non-finite value can appear and
  `serde_json::Number::from_f64` rejects it; the
  adapter panics with a useful diagnostic.
- **Every float lane → integer lane** — round each
  element; panic on non-finite or out-of-range.

**Class · — not in either catalog.** The matrix is complete
(§3.3), so every `·` is an **intentional** exclusion:

- **Scalar ↔ vector** — `VecX→{numeric, Bool}` and
  `{numeric, Bool}→VecX` are both excluded because no
  single collection-to-scalar convention (or scalar-to-
  collection length) is natural (first? last? length?
  sum? mean?). When the
  boundary rejects this pair, `WriteError::TypeMismatch`
  appends a hint that the reduction is the author's
  choice (§3.3).
- **`Ext` / `Handle`** — never in the auto-adapter
  catalog. `Ext` exposes typed access via
  `ReflectedValue::try_as_str` etc. at consume
  sites; `Handle` is downcast by the consumer node
  to the concrete type. The matrix above doesn't
  list these rows / columns.

### 3.1 Bytes conventions

| Axis | Convention |
| --- | --- |
| Numeric ↔ Bytes | little-endian, exactly `sizeof(N)` bytes (panic on wrong length) |
| Bool ↔ Bytes | 1 byte (`0x01` / `0x00`) |
| Vec ↔ Bytes | little-endian element bytes; length must be a multiple of `sizeof(element)` |
| Bytes ↔ Str | lowercase hex (`data_encoding::HEXLOWER`) — roundtrip-lossless, unambiguous, JSON-safe |
| Bytes ↔ Json | `Json::String` holding the lowercase hex |

**Little-endian** matches native CPU layout
(x86_64, ARM64) and the binary protocols this
substrate adapts to (CQL `vector<float, N>`, Postgres
binary). Authors who need big-endian byte order write
their own node; polyfill provides none.

**Lowercase hex** for Bytes ↔ Str round-trip avoids
the URL-safety question entirely and reads
unambiguously in logs. Compact base64 encoding is
available through the explicit `to_base64` /
`from_base64` nodes (polydat-nodes `digest.rs`) when
size matters.

### 3.2 Str → Json convention: try-parse-or-error-wrap

`Str→Json` attempts `serde_json::from_str`. Well-
formed JSON parses through:

- `"{"a":1}"` → `Json::Object({"a": 1})`
- `"[1, 2, 3]"` → `Json::Array([1, 2, 3])`
- `"\"hello\""` → `Json::String("hello")`
- `"42"` → `Json::Number(42)`

Malformed JSON wraps in a structured error rather
than panicking, so the substrate never silently
loses the original content:

```json
{
  "error": "invalid JSON",
  "message": "expected value at line 1 column 1",
  "raw": "{bad json"
}
```

Workload authors who pull the resulting `Json`
downstream see the wrapped error and can branch on
its shape. This matches the substrate's general
posture of "fail loud with useful context, don't
panic when the upstream might be human-typed."

### 3.3 Completeness

The matrix is **complete**: every meaningful `(from, to)` pair has
an adapter. The catalog's own test enforces this together with the
widening invariant, and it re-derives the grid above so that the grid
cannot diverge from the catalog unnoticed. Three properties hold:

- **Widening totality** — every lossless widening, canonical
  `int → f64`, and the full `Bool ↔ numeric` / `int → bool`
  families are class A.
- **Scalar block total** — the 14×14 scalar block is fully A/B:
  every narrowing, cross-sign cast, `float → int`, and
  `int → narrow-float` is a class-B boundary adapter (range-checked,
  panics on overflow / non-finite).
- **Vector lanes total** — all seven lanes inter-convert
  element-wise (widening lanes class A, narrowing/`float→int`
  class B), and each serialises to / parses from `Bytes` / `Json` /
  `Str`.

The bulk of these (~130 trivial transforms) are macro-generated in
`library/polyfill_complete.rs`; the hand-written cores stay in
`convert.rs` / `polyfill*.rs`.

The **only** `·` cells are **scalar ↔ vector**, and that is a
deliberate exclusion, not a gap: a scalar has no canonical
vector length, and a vector no canonical scalar reduction. The
library provides no scalar reduction node; an author writes the
reduction node they mean (or uses `vec_dot` / `vec_norm` where
those are the reduction wanted), and `TypeMismatch` on a rejected
`vec → scalar` pair tells the author the choice is theirs.

---

## 4. The `auto_adapter` / `boundary_adapter` split

Wire validation inside a graph is **strict**. When an author writes
`s := "LATENCY"; overscan := s << 2`, the compiler rejects it at
compile time instead of inserting a parser that panics at runtime on
"LATENCY". The intra-graph catalog (`auto_adapter`) therefore holds
only the **class A** adapters of §3, whose `eval` is total over the
input domain.

The host boundary (`adapt_boundary_value`) is **permissive**.
Workload parameter values arrive as `Value::Str` (from YAML
interpolation or comma-split iteration values) and must be converted
into typed slots the workload author declared as `Bool`, `U64`,
`F64`, and so on. The boundary catalog (`boundary_adapter`) is
`auto_adapter` plus every **class B** adapter: narrowings, parsers,
shape-checking extractors, and the adapters that refuse non-finite
floats. A panic on bad input is acceptable at the boundary because
the string is data the host supplied, and parsing it or failing with
a useful diagnostic is what the host expects.

`Value::satisfies_slot(slot_type)` returns true when a value's stored
form is acceptable for a slot of type `slot_type`, taking
bit-stuffing into account. The boundary uses it after the adapter
runs, as the residual check: `Value::U64` storage is accepted
for the unsigned widths, `F16`, and compatibility producers
for the signed widths;
`Value::I64` for `I64`/`I32`/`I16`/`I8` slots;
`Value::F64` for `F64`/`F32`/`F16`; and any
register-view word for any reg-view slot (the
free-bitcast rule — a `Reg128` under one `RegLanes`
view satisfies a slot declaring any other, §1.5).
This lets narrowing adapters that output the bit-
stuffed runtime form (e.g. `__u64_to_u32` produces
`Value::U64` with low 32 bits) pass validation for
the narrower slot type without a separate Value
variant per port type. The pre-adapter strict
check `value.port_type() == slot_type` keeps an
unadapted `Value::U64` from silently truncating
into a U32 slot — the adapter MUST run.

---

## 5. Explicit conversion nodes (outside the catalog)

Nodes that perform deliberately lossy, formatted, or
parameterized conversions are NOT in the auto_adapter
catalog. Users place them deliberately by name.
Defined in `polydat-core/src/library/convert.rs`:

| Node | Signature | Semantics |
| --- | --- | --- |
| `f64_to_u64` | `(f64) → u64` | Truncating cast (`as u64`) |
| `round_to_u64` | `(f64) → u64` | Round half-up to nearest |
| `floor_to_u64` | `(f64) → u64` | Truncate down |
| `ceil_to_u64` | `(f64) → u64` | Round up |
| `discretize` | `(input, range, buckets) → u64` | Bin a continuous f64 into `buckets` equal-width buckets over `[0, range)`; clamps out-of-range |
| `format_u64` | `(u64, base) → str` | Hex / binary / decimal rendering with prefix |
| `format_f64` | `(f64, precision) → str` | Fixed-precision decimal |

The principle: **if the user has to choose a rounding
mode, a base, a precision, a base unit, or a
truncation policy, the node is explicit**. The
auto-adapter catalog only contains conversions where
there is no semantic choice (widening, to-string via
Display, Bool ↔ U64 as 1/0).

---

## 6. Boundary adapter application

Two call sites apply the catalog: the assembler, to wires inside a
graph (§6.1), and `adapt_boundary_value`, to values written across the
host boundary (§6.2). The converter nodes and `convert::to_port` of
[input_variance.md](input_variance.md) also apply `boundary_adapter`
(§6.2).

### 6.1 Assembler — intra-graph wire validation

`compile::assembly::resolve_with_log` walks every wire after
parse, compares the producer's output `PortType` to
the consumer's input `PortType`, and:

- Equal types → wire it through, no adapter.
- Mismatch with a catalog entry → insert the adapter
  node inline, rewriting the wire to route
  `producer → adapter → consumer`, and record
  `TypeAdapterInserted` in the compile log.
- Mismatch with no catalog entry → fail construction
  with `AssemblyError::TypeMismatch` carrying both
  port types and the offending wire path.

This pass runs once, before any engine is built, so the resolved graph
that each of the four engines (the interpreter, the closure tier,
native, and pure native) receives has no remaining mismatches, and
the inserted adapters are the same on all four. Strict mode's refusal
of an implicit coercion is one check at the same insertion point (the
assembler's `strict` flag), and its error names the explicit
conversion to write, so strict mode behaves identically on all four
engines.

### 6.2 Boundary — `adapt_boundary_value`

`polydat-core/src/kernel/state.rs::adapt_boundary_value(slot_name,
slot_type, value)` converts a value arriving at runtime into an input
slot whose type the assembler could not check against it. It is
called for outer-scope values copied into inner kernels by the `set:`
/ `bindings:` lowering (the scope binder's copies) and for host writes
through the deprecated `Dataflow::set_wire_idx`. It returns the value
to store:

1. It compares the incoming `Value`'s `port_type()` to the
   slot's declared `PortType`.
2. If they are equal, it returns the value unchanged.
3. If they differ, it looks up `boundary_adapter(from, to)`
   (which checks `auto_adapter` first, then the
   boundary-only Str→X parsers of §3), runs the
   adapter's `eval` on the value, and returns
   the result.
4. If the catalog has no entry, it returns the value unchanged
   and logs a one-line audit warning. `set_wire_idx` then detects the
   residual type mismatch and returns `WriteError::TypeMismatch`
   to the host.

`adapt_boundary_value` is not the conversion rule for inputs. Under
[input_variance.md](input_variance.md), a write is never converted:
an input whose type may vary is typed `Dyn`, and a converter node in
front of its readers applies the catalog when its input changes,
reported at the level `CompileOptions::input_variance` names. The
host-side form is `polydat::convert::to_port(value, to)`, which
applies the same catalog and returns a `ConvertError` where this
helper logs a warning and returns the value unconverted. Only the
deprecated `Dataflow::set_wire_idx` and the scope binder's copies use
this helper (input_variance.md §7, §8, §11).

---

## 7. Interpolation type boundary

The DSL parser
(`polydat-grammar/src/parser.rs::parse_interpolated_string`)
turns every interpolated string literal into a
`printf` call:

| Source | Desugared |
| --- | --- |
| `"hello"`           | `Expr::StringLit("hello")` |
| `"hello {name}"`    | `printf("hello {}", name)` |
| `"{a}-{b}"`         | `printf("{}-{}", a, b)` |
| `"{eh}"`            | `printf("{}", eh)`  ← **always Str** |

The string quotes are semantically decisive: `"{eh}"`
requests textual formatting and therefore produces `Str`.
A typed passthrough is written as the bare expression `eh`.
This uniform lowering keeps interpolation's result type
independent of template shape; a sole placeholder is not a
special type-preserving form.

When textual values cross into typed inputs, the boundary
adapter catalog supplies the supported Str→X parses. If no
adapter exists, compilation or the typed write rejects the
mismatch. `Value::None` follows
[none_semantics.md](none_semantics.md) and is never converted
to an empty string implicitly.

---

## 8. Cross-references

- [composition_substrate.md](composition_substrate.md) §T1,
  §T2, §T3 — the typed-slot axioms this catalog enforces.
- [none_semantics.md](none_semantics.md) — `Value::None`
  propagation, the absent-sentinel rule.
- [`polydat-core/src/ast.rs`](../../../polydat-core/src/ast.rs) — `PortType`, `Value`,
  `satisfies_slot`, the carriers.
- [`polydat-core/src/compile/assembly.rs`](../../../polydat-core/src/compile/assembly.rs) —
  `auto_adapter` / `boundary_adapter` (catalog dispatch) and the
  wire resolution that inserts adapters.
- [`polydat-core/src/kernel/state.rs`](../../../polydat-core/src/kernel/state.rs) —
  `adapt_boundary_value`, boundary-time application.
- [`polydat-core/src/library/convert.rs`](../../../polydat-core/src/library/convert.rs) — the
  core adapter nodes; the narrow-width, 128-bit, and
  matrix-completion adapters are its `polyfill_*` companions.
