# Compilation Levels

Polydat compiles a function graph at one of three levels. The compiled
artifact is the same shape at each level — a kernel callers can poke
coordinates into and pull outputs out of — so picking a level is a
performance / build-size trade rather than a behaviour switch.

| Level   | Mechanism                 | Measured, per cycle of an eleven-node graph | Feature |
|---------|---------------------------|--------------------------------------------:|---------|
| Phase 1 | Pull-through interpreter  | 398.66 ns                                   | always  |
| Phase 2 | Compiled u64 closures     | 93.855 ns, 4.25× faster than Phase 1        | always  |
| Hybrid  | Per-node optimal          | between Phase 2 and Phase 3                 | `jit`   |
| Phase 3 | Cranelift JIT native code | 45.923 ns, 8.68× faster than Phase 1        | `jit`   |

The measured column is the reference run recorded in
[Engine-ladder performance](performance.md): one graph, one machine, one
date, with Criterion's confidence intervals. It is the only measurement
this documentation stands behind; per-node figures depend on the graph,
and the ratios depend on the machine, so treat them as one data point
rather than a constant.

Phase 1 and 2 are always available — no extra dependency, no codegen
backend. Phase 3 needs the Cranelift JIT and ships behind the `jit`
feature flag.

The Hybrid level isn't a fourth implementation; it's the runtime
picking, per node, between the Phase 2 closure path and the Phase 3
JIT path depending on which produced better code for that node. The
Hybrid result is always ≤ the throughput of pure Phase 3 and ≥ pure
Phase 2.

## Cargo Features

- **`jit`** (default) — Cranelift JIT for Phase 3 / Hybrid compilation.
  Disable with `default-features = false` for a lighter build (~50MB
  smaller).
- **`vectordata`** — vector dataset access nodes for ML/AI workloads.
