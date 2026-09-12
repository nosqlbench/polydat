# Compilation Levels

Polydat compiles a function graph at one of three levels. Every level
accepts every program and computes the same values (the one exception: an
extern wider than one slot, a vector or a 128-bit integer, runs on the
interpreter only), and every level is driven through one trait, `Kernel`:
set inputs, pull outputs, and read names and types the same way whatever
the engine ([engine parity](../design/engine_parity.md)). Picking a level
is a performance and build-size trade rather than a behaviour switch.

| Level   | Mechanism                 | Measured, per cycle of an eleven-node graph | Feature |
|---------|---------------------------|--------------------------------------------:|---------|
| Phase 1 | Pull-through interpreter  | 441.83 ns                                   | always  |
| Phase 2 | Compiled u64 closures     | 121.05 ns, 3.65× faster than Phase 1        | always  |
| Phase 3 | Native segments, closures elsewhere | 76.104 ns, 5.81× faster than Phase 1 | `jit`   |

The measured column is the reference run recorded in
[Engine-ladder performance](performance.md): one graph, one machine, one
date, with Criterion's confidence intervals. It is the only measurement
this documentation stands behind; per-node figures depend on the graph,
and the ratios depend on the machine, so treat them as one data point
rather than a constant.

Phase 1 and 2 are always available — no extra dependency, no codegen
backend. Phase 3 needs the Cranelift JIT and ships behind the `jit`
feature flag.

Phase 3 is not all-or-nothing: every node that has a native lowering
runs as native code and every other node runs its Phase 2 closure, over
the one slot buffer, so Phase 3 accepts every program Phase 2 accepts.
The pure native form, one function for the whole graph, is the
differential reference behind it and is not a host surface; the
performance guide measures it as a fourth rung to show what native
lowering alone does on a graph where every node has one.

## Cargo Features

- **`jit`** (default) — Cranelift JIT for Phase 3 compilation.
  Disable with `default-features = false` for a lighter build (~50MB
  smaller).
- **`vectordata`** — vector dataset access nodes for ML/AI workloads.
