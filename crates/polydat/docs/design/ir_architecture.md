---
type: specification
title: Comprehension IR Architecture
timestamp: 2026-09-25
description: "The comprehension IR as a stack machine over stream operands: interpretation, materialization barriers, stack effects, and adding an opcode."
tags: [iteration]
---

# Comprehension IR Architecture

This document specifies how polydat's comprehension IR
(`polydat::iteration::comprehension::ir`) is executed: the
stack-machine interpreter and its stream operands, which opcodes
are materialization barriers, the stack effect of each opcode,
where the IR compiler applies reductions R1 and R2, and the steps
for adding an opcode. The algebra the IR implements is specified
separately.

**Related specifications:** [comprehension_forms.md](comprehension_forms.md)
(the comprehension algebra; "spec §" references below are to it).

## The execution model: stack machine + stream operands

The IR is interpreted by a **stack machine** whose operands are
**streams**. The two properties are distinct:

- The **machine model** is a stack machine. The interpreter
  processes a linear sequence of opcodes in order over a stack of
  operands; each opcode pops `pop` operands off the top and
  pushes `push` operands onto the top.
- The **operands on the stack** are **streams**, not raw
  values. Each operand is a lazy producer of tuples — a
  `Box<dyn TupleStream>` whose `advance()` method yields one
  tuple at a time or `None` when exhausted.

The full name of the model, per spec §9.1, is
**"stack-machine interpreter that maintains a stream stack."**
Neither "stream machine" nor plain "stack machine" is a correct
shortening: the first omits the stack semantics, and the second
omits that the stack holds streams rather than values, which is
what makes evaluation lazy.

## How interpretation works

The [`interpret`](../../../polydat-core/src/iteration/comprehension/ir/interpreter.rs)
function walks the IR opcode sequence **once**, manipulating
the stream stack:

```
for op in program.ops() {
    match op {
        PushClause { name, source }      => stack.push(ClauseStream::new(...))
        Cartesian { n }                  => stack.push(CartesianStream::new(pop_n(n)))
        Zip { n, mode }                  => stack.push(ZipStream::new(pop_n(n), mode))
        Union { n }                      => stack.push(UnionStream::new(pop_n(n)))
        Filter { predicate }             => stack.push(FilterStream::new(pop(), pred))
        OrderStreaming { kind, trunc }   => stack.push(OrderStreamingStream::new(pop(), kind, trunc))
        OrderMaterialize { strategy, .. }=> stack.push(OrderMaterializeStream::new(pop(), ...))
        Dispense                         => /* no-op; top of stack is result */
    }
}
return stack.pop();  // the final stream
```

The walk happens **once at interpret-time** and returns a tree
of `TupleStream` trait objects. **Tuples are produced lazily**
when the consumer pulls from the returned stream: each
`advance()` on it calls `advance()` on its children, down to the
leaf clause streams, which are pulled only as needed.

### Concrete example

For the AST `cartesian(clause(k, [1, 2]), clause(b, [10, 20]))`:

1. Compiler emits IR:
   ```
   PushClause(k, [1, 2])
   PushClause(b, [10, 20])
   Cartesian(2)
   Dispense
   ```
2. Interpreter walk:
   ```
   step 1: stack = [ClauseStream(k)]
   step 2: stack = [ClauseStream(k), ClauseStream(b)]
   step 3: stack = [CartesianStream([ClauseStream(k), ClauseStream(b)])]
   step 4: dispense — return stack.pop()
   ```
3. Consumer calls `advance()` on the returned `CartesianStream`,
   which pulls from its child streams lazily and emits
   `[(k, 1), (b, 10)]`, then `[(k, 1), (b, 20)]`, then
   `[(k, 2), (b, 10)]`, then `[(k, 2), (b, 20)]`, then `None`.

## Rationale

1. **The IR sequence is small and analyzable.** It is linear,
   typed, and immutable, so tools such as the bounds checker
   inspect it in one walk, and it can be serialized. Any other
   executor of the same IR — for example a stream-fusion
   compiler that rewrites it into a single nested generator —
   must produce the identical dispense sequence under §9.2's
   correctness contract. The IR therefore fixes the semantics,
   and the executor may be chosen freely.
2. **Per-tuple cost stays bounded.** There is no opcode dispatch
   per tuple: the per-opcode work happens once, at
   interpret-time. Per-tuple cost is the cost of the stream
   types' `advance()` methods, which are direct virtual calls.

## Materialization barriers vs streaming

Per spec §6.2 + §6.3, exactly two opcodes are materialization
barriers:

- `OrderMaterialize` — non-Lex `order` strategy. The
  `OrderMaterializeStream` pulls its input fully on first
  `advance()`, applies the strategy, then emits permuted
  tuples one at a time.
- `Zip { mode: Cycle }` — the shorter children buffer their
  values for replay. `ZipStream`'s Cycle branch pulls all
  children to exhaustion on first `advance()`, then iterates
  with modular cursors.

Every other stream type is **streaming**: each `advance()` uses
O(operator-local state) above its arity. `CartesianStream` is
the least obvious case: it caches axes 1..N, which it iterates
repeatedly, but streams axis 0 lazily.

## The R1 + R2 boundary: AST vs IR

Two of the algebra's reductions, R1 ("order Lex → counter
wrapper") and R2 ("order non-Lex → indexed push-down"), are
**IR compilation decisions**, not AST rewrites. `optimize()`
never rewrites the AST for them; the IR compiler
(`ir::compile`) chooses the opcode from
`metadata.index_addressable`:

- `order(Lex, _)` → `Op::OrderStreaming` (R1).
- `order(non-Lex, Some(n))` over index-addressable input →
  `Op::OrderMaterialize { indexed: true }` (R2; the
  interpreter's `OrderMaterializeStream` uses the strategy's
  closed-form indexed lookup over the input's `IndexFn`).
- `order(non-Lex, Some(n))` over non-addressable input →
  `Op::OrderMaterialize { indexed: false }` (pull the full
  input, apply the strategy, emit).

The AST shape is the same for R1/R2; only the chosen IR opcode
differs. The reducibility catalog (spec §10.10.3) records R1 and
R2 as IR-compilation eligibilities so the optimizer's
introspection surface reports them.

## Trait object choice

`Box<dyn TupleStream>` is the required representation for the
stack. The rejected alternatives are:

- **Generic enums** — one enum variant per stream type. This
  avoids trait-object indirection, but the enum must hold every
  stream type's distinct state, and adding a new strategy means
  a new enum arm at every match.
- **Stream-fusion via closures** — a "compile to a single
  nested generator" representation. It is not part of this IR
  and cannot be selected by the compiler.

Trait-object overhead is one vtable dispatch per `advance()`.
This cost is part of the specified interpreter representation.

## Stack effect rules (spec §9.1)

The `Op::stack_effect()` method returns `(pop, push)` per
opcode:

| Opcode | Pop | Push |
|---|---|---|
| `PushClause` | 0 | 1 |
| `Cartesian { n }` | n | 1 |
| `Zip { n, .. }` | n | 1 |
| `Union { n }` | n | 1 |
| `Filter` | 1 | 1 |
| `OrderStreaming` | 1 | 1 |
| `OrderMaterialize` | 1 | 1 |
| `Dispense` | 1 | 0 |

A well-formed program has exactly one stream on the stack
immediately before `Dispense`, which consumes it. The
`Program::stack_depth()` method computes the maximum stack depth
reached during interpretation, which bounds spec §9.3's
`O(depth(C))` operator-stack term.

## What lives in this layer, what doesn't

**In scope for the IR layer:**

- Opcode definitions and their stack semantics.
- AST → IR compilation.
- Interpretation: streams, lazy evaluation, predicate
  evaluation (subset covering §10.9.5 catalog).
- Resource-bound checking.

**IR interpreter boundaries:**

- Predicate evaluation beyond the §10.9.5 catalog. The static
  IR interpreter treats unknown predicates as `true`
  (conservative pass-through); production iteration uses the
  runtime evaluator and the kernel's scope.
- Source evaluation for `Generator` / `WorkloadParamList` /
  continuous sources. The static interpreter exhausts these
  to `None` (no tuple); evaluated-source behavior is defined by
  the runtime evaluator.
- Stream-fusion compilation is not part of this IR. The
  specified executor is the stack-machine interpreter, and
  its dispense sequence is governed by §9.2.

**The runtime evaluator.** Production iteration evaluates a
comprehension through `iteration::comprehension::eval`
(`evaluate_spec`, one clause source to its list of values) and
`iteration::comprehension::runtime::evaluate_for_iteration`
(a whole comprehension to its tuples). Both evaluate against
`&dyn Lookup` (`kernel::interp::Lookup`), the name resolution
used to read a `{name}` placeholder or a bare identifier. The
interpreter kernel implements `Lookup`, and `Layered` resolves a
tuple's bindings before any other lookup. The evaluator is
therefore engine-neutral: opening a traversal on the closure
tier or the native engine evaluates the same comprehension
against the same scope, and does not require a kernel of the
engine that opens it.

## Adding a new opcode

A new IR opcode is added only when the behavior cannot be
expressed as a new strategy, that is, as a parameterization of
`OrderMaterialize`. The steps are:

1. Add the variant to `Op` in `op.rs`. Implement
   `stack_effect` and `is_barrier`.
2. Add a stream type in `interpreter.rs` implementing
   `TupleStream`.
3. Add the dispatch arm in `interpret()`.
4. Update `compile.rs` if any AST node should emit the new
   opcode.
5. Update `bounds.rs` if the new opcode is a barrier.
6. Add the opcode to the resource-bound formula and the
   `Op` enum's documentation in the comprehension spec
   §9.1.

The eight §9.1 opcodes constitute the IR, so adding one is a
coordinated change to the algebra spec and the IR. New
strategies are normally added as `StrategyName` variants, which
require no IR change, rather than as new opcodes.
