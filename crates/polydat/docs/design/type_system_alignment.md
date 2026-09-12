# Type-System Alignment Specification

This document defines how Polydat aligns its static wire types,
runtime values, compiled buffer slots, Cranelift 0.116, and
serde JSON. The detailed adapter catalog lives in
[type_system.md](type_system.md); the JIT safety contract lives
in [jit_boundary.md](jit_boundary.md).

Polydat is pinned to the Cranelift 0.116 release family. The
compiler MUST use only types and lowerings supported by that
version. Host AVX2 or AVX-512 capability may improve lowering or
kernel selection, but it does not authorize inventing a wider
Polydat register type that the pinned backend cannot represent
and lower through the implemented path.

## 1. Governing type planes

Four related planes are intentionally distinct:

| Plane | Type | Purpose |
|---|---|---|
| Static graph | `PortType` | Exact producer/consumer contract and slot layout. |
| Runtime | `Value` | Owned or shared value carrier used by P1. |
| Compiled buffer | `SlotColor` + `slot_width()` | Immediate limbs or engine-internal reference pairs. |
| Interchange | `serde_json::Value` | JSON/YAML-facing projection and extraction. |

`PortType` is authoritative at a wire. `Value::port_type()`
reports the runtime carrier and therefore cannot recover a
narrow width that was bit-stuffed into a wider carrier.
Compiled layout must derive from `PortType::slot_color()`; no
engine may restate widths independently.

## 2. Scalar alignment

| Polydat family | Runtime carrier | Cranelift representation | JSON projection |
|---|---|---|---|
| `U8/U16/U32/U64` | `Value::U64`, zero-extended | integer lane/word operations, one u64 slot | JSON unsigned number |
| `I8/I16/I32/I64` | `Value::I64`, sign-extended | sign-agnostic integer type plus signed operations, one u64 slot | JSON signed number |
| `F16/F32` | low-bit IEEE pattern in `Value::U64` for node outputs; accepted host carrier as specified by `satisfies_slot` | F16/F32 operation where supported, one u64 slot | JSON number after widening |
| `F64` | `Value::F64` | F64, one u64 slot by bitcast | JSON number when finite |
| `Bool` | `Value::Bool` | 0/1 integer convention, one u64 slot | JSON bool |
| `U128/I128` | `Value::U128/I128(Bits128)` | no native lowering; two immediate slots, run as a closure step | decimal JSON string |

Cranelift integer types are sign-agnostic; signedness is chosen
by operations. Polydat keeps signedness in `PortType` and,
for 64-bit runtime/interchange honesty, in distinct
`Value::U64` and `Value::I64` carriers.

`F128` is excluded. Stable Rust and the pinned backend path do
not provide the complete carrier and lowering contract Polydat
requires. `U128/I128` are valid typed runtime values with no
native lowering: they occupy two immediate slots (`Imm2`, never
a pointer), and a node over them runs as a closure step on the
closure tier and the native engine, with the same result as on
the interpreter.

## 3. Register-value plane

The explicit SIMD value is one fixed 128-bit word:

- `Reg128` for raw bytes;
- `RegI8x16`, `RegI16x8`, `RegI32x4`, `RegI64x2`;
- `RegF16x8`, `RegF32x4`, `RegF64x2`.

All views use `Value::Reg128(Bits128, RegLanes)` and occupy two
immediate slots. A reg-to-reg conversion is a bit-preserving
`RegView` retag. Lane arithmetic is defined by the selected
view; integer arithmetic wraps, and float reductions use the
operation's documented fixed tree.

The fixed 128-bit type is a semantic and ABI choice:

1. it is representable by the pinned Cranelift backend paths;
2. it has a stable two-slot P1/P2/P3 representation;
3. its lane count and results do not vary by host ISA;
4. wider host execution can unroll or schedule multiple words
   without changing graph types.

Effective ISA selection therefore chooses an implementation,
not a different public value width. See
[simd_isa_autopromotion.md](simd_isa_autopromotion.md).

## 4. Heap-slice plane

The typed vector family is:

```text
VecI8 VecI16 VecI32 VecI64
VecF16 VecF32 VecF64
```

Each runtime value is a `SliceArc<T>`. In compiled slot
layouts it is `Ref2`: a `(ptr, len)` pair that refers to a
live input slice or kernel-owned output scratch.
`CompiledSlotOp` receives these pairs and the step's
`ScratchBuf` allocation. A producer publishes its pair only
after writing scratch; a consumer may read it only while the
producer's publication remains valid.

Heap vectors are arbitrary-length containers, not public SIMD
registers. Nodes may use 128-bit Cranelift kernels internally
for chunked bodies and a scalar tail. Reducing SIMD kernels may
reassociate floating-point addition and are verified with the
operation's tolerance contract rather than byte equality with a
scalar left fold.

`Bytes` is the unsigned-byte buffer type. There is no
`VecU8`; `VecI8` is numeric signed-lane data.

## 5. Heap and extension values

| PortType | Runtime carrier | Compiled status | JSON behavior |
|---|---|---|---|
| `Str` | `Arc<str>` | `Ref2` pair of its bytes; P2 closures | string |
| `Bytes` | `Arc<[u8]>` | `Ref2` pair of its bytes; P2 closures | documented hex convention |
| `Json` | `Arc<serde_json::Value>` | `Ref2` pair to the `Value`; P2 closures | identity |
| `Ext` | `Box<dyn ReflectedValue>` | `Ref2` pair to the `Value`; crosses compiled tiers, is forwarded or projected, never operated on natively | extension-defined reflected projection |
| `Handle` | `Arc<dyn Any + Send + Sync>` | `Ref2` pair to the `Value`; crosses compiled tiers, never downcast natively | no general JSON identity |

`Ext` and `Handle` are explicit escape types. They do not
gain implicit structural adapters or JIT layout by resemblance
to a built-in type. In the compiled tiers a pair names the
`Value` in its producing step's scratch, and every read copies
the same `Arc` out
([Compiled By-Reference Slots](compiled_handles.md) §3, §8).

The crate enables serde_json's `preserve_order` feature so
object iteration order does not depend on whether Polydat is
built alone or in a larger workspace.

## 6. Slot-color contract

`PortType::slot_color()` is total:

| Color | Width | Members | Meaning |
|---|---:|---|---|
| `Imm1` | 1 | scalars | Immediate data; never interpreted as an address or a name. |
| `Imm2` | 2 | `U128/I128`, all `Reg128` views | Two limbs of immediate data; never interpreted as an address. |
| `Ref2` | 2 | all `Vec*` types, `Str`, `Bytes`, `Json`, `Ext`, `Handle` | Engine-internal `(ptr, len)` pair with a proven owner ([Compiled By-Reference Slots](compiled_handles.md)). |

`PortType::scratch_elem()` names the scratch entry a `Ref2` port's
producer owns: an element type for a vector, `Str` or `Bytes` for a
byte string, `Value` for the rest. The entry is fixed by the port type,
so no code branches on what a pair names.

The following are invariants:

- all slots of one logical value share provenance;
- port offsets are computed from cumulative slot widths;
- `Imm2` and `Ref2` are never interchangeable even though
  both occupy two slots;
- only engine-owned code, a closure or the boundary decode,
  dereferences `Ref2`;
- every `Ref2` output's scratch has one logical writer;
- publication precedes any read;
- a skipped producer cannot leave a consumer observing an
  incoherent pointer/length pair.

The complete unsafe and validation contract is S1–S10 in
[jit_boundary.md](jit_boundary.md).

## 7. Execution eligibility

Type support and engine eligibility are separate:

- A valid `PortType` is executable on every engine — the
  interpreter, the closure tier (P2), and the native engine
  (P3) — when its node implementation exists. An engine that
  cannot run a program refuses it at construction with a
  reason naming the node or construct; it never runs it
  differently.
- P2 is total: every node has a closure form, derived from its
  signature by the `#[polydat_node]` macro (or supplied by a
  hand-written override), whatever the slot colors of its
  ports — one-slot scalars, two-slot immediates, and `Ref2`
  pairs over the step's own scratch, vectors and by-reference
  values alike ([Compiled By-Reference Slots](compiled_handles.md)
  §5).
- P3 is native where a node has a lowering and its closure
  elsewhere. The native form is the node's own: the classifier
  (`classify_node_typed`) selects it from the node and the
  types of its wires, fixed at classification; a wire of a type
  no lowering takes leaves the node on its closure. Native code
  carries no reference pairs yet, so a node with a `Ref2` port
  on either side keeps its closure beside the native segments
  ([Compiled By-Reference Slots](compiled_handles.md) §6).
  Register-plane operations with a lowering run as native SIMD;
  slice-bearing nodes' internal vector math may call compiled
  SIMD helpers from their closure.
- `U128/I128` operations and nodes that downcast an `Ext` or
  `Handle` have no native form and run as closure steps; the
  values themselves cross compiled tiers as two immediate slots
  and as reference pairs respectively.

The one typed read is `Kernel::pull`: on every engine it
returns the named output as the `Value` its port type names,
decoding whichever slot color the engine stored it in
(`marshal::decode_output` for the compiled engines — one-slot
carriers, reference pairs copied out, and the two-limb
reassembly of a 128-bit integer or register word). A pair is
never handed to the host, and a slot that holds `None` reads
as `None`.

What an engine decided is observed, not inferred:
`Kernel::plan` returns an `EnginePlan` — how many runs of nodes
run as native segments (on the interpreter, its native cones),
how many nodes run their closure, and how many the interpreter
dispatches itself — on every engine.

No compiler may coerce a value merely to make a higher engine
tier available. Engine selection follows the typed graph; it
does not weaken it.

## 8. JSON alignment

JSON is the interchange model, not the execution type system:

- `None` projects as null where a JSON projection is
  requested, but no wire has a `None` port type.
- Homogeneous `Vec*` values project as JSON arrays of
  numbers.
- `U128/I128` use decimal strings because JSON numbers cannot
  represent their complete integer domains.
- non-finite floats have no standard JSON number and therefore
  use the adapter's explicit error behavior;
- object order follows `preserve_order`.

JSON arrays do not erase a graph's vector lane type. Extraction
requires the destination `PortType`, validates every element,
and applies the adapter catalog's narrowing/error rules.

## 9. Change rule

Adding or changing a type requires one coherent change across:

1. `PortType` keyword and slot-color mapping;
2. `Value` carrier and `satisfies_slot`;
3. `Wire` injection/extraction;
4. display, strict display, bytes, and JSON projection;
5. adapter catalog membership;
6. P1 behavior and any claimed P2/P3 lowering;
7. provenance/layout tests and P1-as-oracle equivalence tests;
8. this specification and [type_system.md](type_system.md).

A backend type appearing in a newer Cranelift release is not,
by itself, sufficient. Polydat admits it only after the pinned
dependency, stable Rust carrier, slot ABI, lowering,
cross-engine equivalence, and target-ISA behavior all satisfy
this rule.
