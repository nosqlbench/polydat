---
type: guide
title: Compilation Levels
timestamp: 2026-09-25
description: "The interpreter, the closure tier, the hybrid kernel, and native code: features and trade-offs."
tags: [engines, performance]
---

# Compilation Levels

Polydat compiles a function graph at one of three levels. All three
levels accept every program and compute the same values, except that a
program with an extern of a 128-bit integer or register word type runs
only on the interpreter. All three are driven through one trait,
`Kernel`: a host sets inputs, pulls outputs, and reads names and types
with the same calls on every level ([Engines](../design/engines.md)
§3.5). Picking a level
changes performance and build size, not behaviour.

| Level   | Mechanism                 | Measured, per cycle of an eleven-node graph | Feature |
|---------|---------------------------|--------------------------------------------:|---------|
| Phase 1 | Pull-through interpreter  | 322.25 ns                                   | always  |
| Phase 2 | Compiled u64 closures     | 71.006 ns, 4.54× faster than Phase 1        | always  |
| Phase 3 | Native segments, closures elsewhere | 25.347 ns, 12.71× faster than Phase 1 | `jit`   |

The measured column is the reference run recorded in
[Engine-ladder performance](performance.md): one graph, one machine, one
date, with Criterion's confidence intervals. It is the only measurement
this documentation cites as a reference; per-node figures depend on the graph,
and the ratios depend on the machine, so treat them as one data point
rather than a constant.

Phase 1 and 2 are always available — no extra dependency, no codegen
backend. Phase 3 needs the Cranelift JIT and ships behind the `jit`
feature flag.

Phase 3 is not all-or-nothing: every node that has a native lowering
runs as native code and every other node runs its Phase 2 closure, over
the one slot buffer, so Phase 3 accepts every program Phase 2 accepts.
The pure native form, one function for the whole graph, is the
reference Phase 3 is differentially tested against and is not offered to hosts; the
performance guide measures it as a fourth rung to show what native
lowering alone does on a graph where every node has one.

## Cargo Features

- **`jit`** (default) — Cranelift JIT for Phase 3 compilation.
  Disable with `default-features = false` for a lighter build (~50MB
  smaller).
- **`vectordata`** — vector dataset access nodes for ML/AI workloads.
