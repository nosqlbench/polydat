# Polydat Execution Engines

This specification defines the P1/P2/P3 execution lattice, provenance modes,
automatic selection for whole-kernel compiled engines, and the production
mixed-kernel path. The Cranelift call and failure boundary is specified in
[jit_boundary.md](jit_boundary.md); graph qualification and fusion order are
specified in [graph_compiler.md](graph_compiler.md).

## 1. Execution lattice

| Level | Representation | Semantic coverage | Construction |
| --- | --- | --- | --- |
| P1 | `PolydatKernel` over `Box<dyn PolydatNode>` and typed `Value` buffers | Complete node, type, scope, and lifecycle model | Normal DSL and assembler compile paths |
| P2 | Direct closures over a flat `u64` slot buffer | Nodes that provide `compiled_u64` or `compiled_slot` kits | Explicit `try_compile*` and `auto_compile_p2` paths |
| P3 | Cranelift native code for every node with a lowering and the node's closure elsewhere, over one flat slot buffer | Every program the closure tier accepts | Explicit `try_compile_jit*` and `auto_compile_p3` paths; embedded cones in normal compilation |

P1 is the semantic host and fallback. P2 and P3 are constructive: a compiled
builder succeeds only when every required operation and slot shape is supported.
Failure leaves or returns the P1 kernel rather than weakening its semantics.
Pure native code, one function for the whole program, is the differential tier
behind P3 since engine parity step 7: it refuses a node without a lowering and
is reachable only through the hidden `try_compile_pure_jit*` builders.

The lattice is monotonic in optimization, not in feature coverage. A node can
be valid at P1 without a P2 closure or P3 lowering. P2 also remains an
equivalence oracle for the compiled slot ABI.

## 2. Production mixed kernel

Normal compilation produces an interpreter-hosted mixed kernel when the `jit`
feature is enabled. `JitMode` controls cone extraction:

- `Off` leaves the typed P1 graph unchanged;
- `Auto`, the process default, extracts eligible connected cones containing at
  least two nodes; and
- `Force` permits a one-node eligible cone.

An extracted cone is replaced by a synthetic fusion node whose `eval` invokes
the compiled native segment. Unsupported nodes, boundary types, lifecycle
classes, and purity shapes remain in P1. The graph's canonical identity walks
through fusion nodes to their scalar subgraph, so changing JIT mode does not
change program identity.

P2 closures are not inserted as an intermediate production cone tier. They are
available through explicit whole-kernel builders and tests.

## 3. Whole-kernel provenance modes

The explicit P2 and P3 builders expose monomorphic kernel variants. Each
variant has its provenance behavior fixed in its type, so the inner evaluation
loop does not branch on a selected strategy.

| Mode | Push-side invalidation | Pull-side guard | Explicit builder suffix |
| --- | --- | --- | --- |
| `Raw` | No | No | `_raw` |
| `Push` | Yes | No | `_push` |
| `Pull` | No | Yes | `_pull` |
| `PushPull` | Yes | Yes | default `try_compile` / `try_compile_jit` |

`Push` exists as an explicit measurement and equivalence surface. Automatic
selection returns only `Raw`, `Pull`, or `PushPull`.

Since engine parity step 4 the same variants are named by `Engine` and
`Provenance` and built by one constructor, `compile_with(Engine)`, behind
the `Kernel` trait; `Provenance::Auto` is the selector below. The explicit
builders remain as aliases ([engine parity](engine_parity.md), A8).

### 3.1 Push-side invalidation

Compiled push kernels store a dependent-step list for each graph input and a
clean flag for each compiled step. `set_inputs` compares the new coordinate
values with the previous ones and marks the dependent steps of changed inputs
dirty. Evaluation skips clean steps inside an otherwise-entered cone.

P1 uses the same dependency relation but treats the act of `set_input` as the
invalidation signal; it does not require value equality before dirtying
dependents. That distinction preserves side-channel and explicit-write
semantics in the complete runtime.

### 3.2 Pull-side guard

Compiled pull kernels store an exact `ProvMask` for every output slot and a
multi-word changed-input mask. Before entering evaluation for a slot, the
kernel tests whether the slot provenance intersects the changed-input mask. A
disjoint mask returns the cached slot without executing the compiled body.

The P2 and P3 kernels therefore support more than 64 coordinate inputs. The
older hybrid test engine retains a single-word internal mask and is not the
normative production provenance representation.

### 3.3 Composition

For `PushPull`, the pull guard decides whether to enter the requested output's
cone. If entered, the push clean flags suppress unaffected steps within that
cone. The two checks preserve the same result as `Raw`; they change only the
amount of work performed.

## 4. Automatic whole-kernel selector

`analyze_graph` records total node count, input count, output count, and exact
per-output upstream-cone sizes. `select_prov_mode` applies this fixed rule:

```text
if total_nodes < 15 and num_inputs <= 1:
    Raw
else if num_inputs >= 2:
    PushPull
else:
    Pull
```

`auto_compile_p2` and `auto_compile_p3` return both the selected engine and the
`GraphAnalysis` used to select it. Cone ratios remain diagnostic metadata; the
selector does not use a `stable_ratio` threshold.

This selector applies only to explicit whole-kernel compiled construction. The
production mixed path uses `JitMode` and cone qualification instead.

## 5. Slot representation

Compiled buffers are arrays of `u64` slots, not arrays of logical values. A
static `SlotColor` maps each port type to one of three layouts:

- `Imm1` — one immediate slot for ordinary scalar bit patterns;
- `Imm2` — two immediate slots for 128-bit integers and register words;
- `Ref2` — a `(ptr, len)` pair referencing kernel-owned typed-slice scratch; and
- `Hdl1` — one slot holding a handle that names a `Str`, `Bytes`, `Json`,
  `Ext`, or `Handle` value, per [Compiled Non-Scalar
  Slots](compiled_handles.md). Byte strings are static-interner or
  cycle-arena handles; the other kinds name entries of a value table the
  running engine owns: a whole P2 or P3 kernel for its lifetime, an
  embedded cone for one eval. Generated code never decodes a handle;
  helpers, closures, boundary marshalling, and the kernels' `get_value`
  readers do, and raw readers refuse the slot.

Narrow integers and `f16`/`f32` use defined bit-stuffing rules inside `Imm1`.
Signedness and exact width remain properties of `PortType`; the common physical
slot does not permit untyped wiring. Ref-bearing nodes remain subject to the
ownership, lifetime, and no-forwarding rules in [jit_boundary.md](jit_boundary.md)
and [type_system_alignment.md](type_system_alignment.md). Handle-bearing nodes
are subject to the cycle lifetime and single-writer rules of SRD 115 §4 and
§8: the root state's cycle advance resets the arena, nested kernels never do,
a cone releases what its eval took, and every table entry has one writer,
checked after every run in debug builds.

## 6. Engine equivalence

For any graph accepted by two engine forms and for the same ordered input and
pull sequence:

1. output `Value` semantics are identical;
2. `None` propagation is identical;
3. typed assertion failures identify the same violated contract;
4. side-channel and nondeterministic nodes are not moved into an engine form
   whose caching rules would suppress required observations; and
5. provenance modes may reuse cached slots only when their exact dependency
   masks prove the requested result unaffected.

The engine ladder, slot-state axioms, cone tests, and equivalence harnesses are
the regression contract for these properties. For handle-bearing nodes the
contract is `tests/handle_tiers.rs`: random programs over the string, JSON,
and tile nodes checked across the interpreter, forced cones, P2, P3, and pure
native code, each read through its typed `get_value`.

Property 2 has one refinement for fused cones (SRD 115 §9). A cone is a single
node to the kernel guard, so a `None` on any of its boundary inputs makes every
output `None`. A node that tolerates `None` inputs and would have produced a
value from one may therefore join a cone only when all of its inputs are wires
from other members, where no `None` can arrive; fed by a kernel input it stays
P1 with its exact semantics.

## 7. What every engine accepts, and what stays where

Every engine a host can choose accepts every program the interpreter
accepts, with the two refusals named below, and computes what the
interpreter computes ([engine parity](engine_parity.md); the node-by-engine
matrix in the [node reference](../reference/nodes.md) is generated by the
parity suite). The other distinctions are placements inside an engine, not
refusals:

- A node without a native lowering runs its closure, as a closure step of
  P3 and on the interpreter as itself; only pure native code, the
  differential tier, refuses it.
- A Ref-bearing value cannot cross a P3 cone boundary or be forwarded by an
  identity-style compiled step; such nodes are closure steps.
- A node whose variadic wires carry a type its helper cannot decode runs its
  closure; the classifier never re-types a wire to admit it.
- The two refusals: an extern wider than one slot (a vector, or a 128-bit
  integer) has no compiled form, because the compiled engines seed and
  publish externs one slot at a time; and a build without the `jit` feature
  has no P3. Each is refused by name with its reason
  (`KernelError::Refused`); a program with such an extern runs on the
  interpreter.
- SIMD scalar-flow promotion is not selected by ordinary engine choice; it has
  its own explicit qualification and execution contract in
  [simd_isa_autopromotion.md](simd_isa_autopromotion.md).
- Engine selection never changes a graph's public port types or named outputs.
