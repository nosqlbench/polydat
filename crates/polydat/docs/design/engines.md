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
| P3 | Cranelift-generated native code over a flat slot buffer | Nodes and complete signatures accepted by JIT classification and lowering | Explicit `try_compile_jit*` and `auto_compile_p3` paths; embedded cones in normal compilation |

P1 is the semantic host and fallback. P2 and P3 are constructive: a compiled
builder succeeds only when every required operation and slot shape is supported.
Failure leaves or returns the P1 kernel rather than weakening its semantics.

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
- `Imm2` — two immediate slots for 128-bit integers and register words; and
- `Ref2` — a `(ptr, len)` pair referencing kernel-owned typed-slice scratch.

Narrow integers and `f16`/`f32` use defined bit-stuffing rules inside `Imm1`.
Signedness and exact width remain properties of `PortType`; the common physical
slot does not permit untyped wiring. Ref-bearing nodes remain subject to the
ownership, lifetime, and no-forwarding rules in [jit_boundary.md](jit_boundary.md)
and [type_system_alignment.md](type_system_alignment.md).

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
the regression contract for these properties.

## 7. Unsupported combinations

- A node without the required compiled closure or lowering remains P1.
- A Ref-bearing value cannot cross a P3 cone boundary or be forwarded by an
  identity-style compiled step.
- SIMD scalar-flow promotion is not selected by ordinary engine choice; it has
  its own explicit qualification and execution contract in
  [simd_isa_autopromotion.md](simd_isa_autopromotion.md).
- Engine selection never changes a graph's public port types or named outputs.
