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
| `U128/I128` | `Value::U128/I128(Bits128)` | no production JIT lowering | decimal JSON string |

Cranelift integer types are sign-agnostic; signedness is chosen
by operations. Polydat keeps signedness in `PortType` and,
for 64-bit runtime/interchange honesty, in distinct
`Value::U64` and `Value::I64` carriers.

`F128` is excluded. Stable Rust and the pinned backend path do
not provide the complete carrier and lowering contract Polydat
requires. `U128/I128` are valid typed runtime values but remain
P1; their two-slot shape is reserved as immediate data, never a
pointer.

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
| `Str` | `Arc<str>` | closure/hosted operation where supported | string |
| `Bytes` | `Arc<[u8]>` | closure/hosted operation where supported | documented hex convention |
| `Json` | `Arc<serde_json::Value>` | P1/hosted | identity |
| `Ext` | `Box<dyn ReflectedValue>` | P1 | extension-defined reflected projection |
| `Handle` | `Arc<dyn Any + Send + Sync>` | P1 | no general JSON identity |

`Ext` and `Handle` are explicit escape types. They do not
gain implicit structural adapters or JIT layout by resemblance
to a built-in type.

The crate enables serde_json's `preserve_order` feature so
object iteration order does not depend on whether Polydat is
built alone or in a larger workspace.

## 6. Slot-color contract

`PortType::slot_color()` is total:

| Color | Width | Members | Meaning |
|---|---:|---|---|
| `Imm1` | 1 | scalars | Immediate data; never interpreted as an address or a name. |
| `Imm2` | 2 | `U128/I128`, all `Reg128` views | Two limbs of immediate data; never interpreted as an address. |
| `Ref2` | 2 | all `Vec*` types | Engine-internal `(ptr, len)` pair with a proven owner. |
| `Hdl1` | 1 | `Str`, `Bytes`, `Json`, `Ext`, `Handle` | A handle naming a value in the static interner, the cycle arena, or the state's value table ([Compiled Non-Scalar Slots](compiled_handles.md)). Opaque to generated code. |

`PortType::handle_kind()` refines `Hdl1`: `Str` and `Bytes` handles are
byte-string handles (interner or arena); `Json`, `Ext`, and `Handle`
handles name value-table entries. The kind is fixed by the port type, so
no code branches on a handle's tag.

The following are invariants:

- all slots of one logical value share provenance;
- port offsets are computed from cumulative slot widths;
- `Imm2` and `Ref2` are never interchangeable even though
  both occupy two slots, and `Imm1` and `Hdl1` are never
  interchangeable even though both occupy one;
- only engine-owned code dereferences `Ref2`, and only helpers and
  boundary marshalling decode `Hdl1`;
- vector output scratch has one logical writer;
- publication precedes any read;
- a skipped producer cannot leave a consumer observing an
  incoherent pointer/length pair.

The complete unsafe and validation contract is S1–S10 in
[jit_boundary.md](jit_boundary.md).

## 7. Execution eligibility

Type support and engine eligibility are separate:

- A valid `PortType` is always executable through P1 when its
  node implementation exists.
- One-slot scalar nodes with a supported lowering may execute
  through P2 or P3.
- Register-plane nodes may execute through P2 and, when
  `classify_node` has a lowering for the operation, native P3
  SIMD. Unsupported register operations remain on a lower tier.
- Slice-bearing nodes may use `CompiledSlotOp` and
  kernel-owned scratch; their internal vector math may call
  compiled SIMD helpers.
- `U128/I128`, `Ext`, `Handle`, and other unclassified
  operations remain P1.

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
