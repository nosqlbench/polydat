---
type: specification
title: Cross-Fiber Cell Invalidation
timestamp: 2026-09-25
description: "The SharedCell publish and consume protocol: revisions, intent bits, memory ordering, and happens-before across fibers on all four engines."
tags: [runtime, scopes]
---

# Cross-Fiber Cell Invalidation

This specification defines how a write to a
`SharedCell` is published and how every kernel that
reads the cell detects the write. A `SharedCell` is one
mutex-protected value register that several kernels
read and write for a `shared` binding. A *producer* is
a kernel that writes a cell; a *consumer* is a kernel
with an input slot bound to it; a *fiber* is the host
thread or task that owns a kernel. A producer's write
through any kernel is observed by every consumer at its
next read, with no host action, on all four engines
(the interpreter, the closure tier, native, and pure
native). This satisfies the reader contract of
[composition_substrate.md] §S5.

**Related specifications:**
[composition_substrate.md] (§S5, the reader contract;
S3, the coordinate-slot invariant),
[scope_model.md](scope_model.md) (where cells are
attached), [engines.md](engines.md) (the engines).

[composition_substrate.md]: composition_substrate.md

---

## 1. Contract and mechanism

The contract is the same on all four engines:

> **Publish** is one act with three parts: the value, a
> monotonic revision, and an intent bit on the word of
> the scope that created the cell. **A consumer re-reads
> a cell when a revision it has seen moves**, and every
> step whose provenance includes the cell's slot is then
> not current.

The *revision* is a per-cell counter incremented by
every publish. An *intent word* is a 64-bit mask owned
by the scope (kernel) that created a set of cells, with
one bit per cell; a publish sets the cell's bit, so a
consumer can test many cells with one load. A cell's
*interest mask*, for a consumer, is the OR of the bits
of the cells the consumer reads from one intent word.

Two consumer implementations satisfy the contract:

- **The interpreter** checks the cells in a node's cone
  at every memoized read. It loads the intent words the
  cone reads cells from, ANDs each with the cone's
  interest mask, and compares per-cell revisions only
  where a bit is set. On any moved revision it
  invalidates every node whose provenance covers the
  moved slot (§5.1).
- **A compiled kernel** (closure tier, native, or pure
  native) polls the revisions of all its cells at the
  first evaluation after a write and before each pull.
  It takes the value of each cell whose revision moved
  and marks the dependents of that slot not current
  (§5.2).

Both implementations rely on the same producer protocol
(§4) and the same memory ordering (§6).

Each cell has a 64-bit revision counter. Each scope
holds intent words, one `AtomicU64` per 64 cells it
creates. Each cell holds a clone of the `Arc` of its
own intent word plus its bit position within that word,
so a publish follows one `Arc` pointer and performs no
lookup.

On write, the producer mutates the cell value under the
cell's `Mutex`, bumps the cell's revision (one
`fetch_add(Release)`), and sets the cell's bit on the
scope's word (one `fetch_or(Release)`). Three Release
stores, O(1) total, no fan-out.

On the interpreter's read, the bulk check is one Acquire
load plus one AND per scope word the cone reads cells
from, typically at most a few per node evaluation
regardless of how many cells exist substrate-wide;
per-cell compares are reached only when the bulk mask
indicates change.

---

## 2. Data shapes

- **A cell** (`SharedCellInner`, shared as an `Arc`): the
  value under a `Mutex`, the revision (`AtomicU64`), the
  `Arc<AtomicU64>` of its scope's intent word, and its
  bit within that word. All four engines use the same type.
- **A scope's allocator.** The interpreter's `EngineCore`
  keeps a `Vec<Arc<AtomicU64>>` of intent words and a
  next-bit cursor; a compiled kernel's extern table
  keeps one word and a next-bit cursor (§9.1).
- **The interpreter's per-kernel consumer state**: the
  cell attached to each input slot (`shared_cells`), the
  broadcast cell for each output (`output_cells`), the
  last revision it observed per cell (`last_seen`, keyed
  by the cell's `Arc` pointer, compared by identity and
  never dereferenced), and a per-node cone cache
  (`cell_cones`) built lazily and cleared whenever a cell
  is attached.
- **A compiled kernel's per-slot consumer state**: for
  each `shared` extern slot, its cell and the revision
  the slot last took its value from (`seen`); and, keyed
  by output slot, the broadcast cells a descendant asked
  for (`output_cells`), empty until one does.

A cell cone (interpreter) is a list of groups, one per
intent word among the cells attached to slots in the
node's provenance. A group holds the word, an interest
mask (the OR of `1 << bit` of its cells), and the
cells' bits and input slots.

---

## 3. Lifecycle

### 3.1 Cell creation

Three sites create cells. Each draws the next bit from
its own scope's allocator, so every cell carries the word
of the scope that created it, wherever it is later
attached:

- **Interpreter, `shared` bindings.** At kernel
  construction, one cell per `shared` output that has a
  backing input slot, holding the slot's current value
  (`seed_shared_cells`, via `EngineCore::make_shared_cell`).
- **Interpreter, broadcast outputs.** At kernel
  construction, one cell per output, holding the output's
  current buffer value (`seed_output_cells`). A descendant
  whose input matches a computed parent output is
  attached to this cell by the parent-gated binder
  ([scope_model.md](scope_model.md) §4), and the parent's
  every pull of the output publishes through it.
- **Compiled kernels, `shared` bindings.** At build, one
  cell per `shared` slot (`compile::externs::Externs::new`,
  via `new_cell`), holding the slot's value; a kernel
  created from a shared program reseeds cells of its own
  (`reseed_cells`), on a fresh word.
- **Compiled kernels, broadcast outputs.** On the first ask
  rather than at construction: `Kernel::output_cell(name)`
  makes one, holding the output's current value, and every
  later pull of that output publishes through it. The
  interpreter can afford to seed one per output at
  construction because it already holds a `Value` per port;
  a compiled kernel holds slots, so it makes a cell only
  where a descendant binds to one, and a program with no
  descendant bound to it allocates nothing and pays one
  emptiness check per pull. This holds on the closure tier
  and native. Pure native makes no broadcast cells: it is the
  differential oracle and Tier-1's carrier, not a surface a
  host builds child scopes under
  ([engines.md](engines.md) §1, §8).

`Kernel::attach_shared_cell(name, cell)` replaces the cell
of the `shared` binding `name` with `cell`, a cell another
kernel holds, on all four engines. The replaced cell keeps
its word and bit, and the attaching kernel's
consumer state for the slot starts over: the interpreter
clears its cone cache, and a compiled kernel clears the
slot's `seen`.

### 3.2 Cone metadata (interpreter)

Built lazily on the first check per node:

1. Read the node's provenance, the exact multi-word mask
   of every input slot that flows into the node.
2. For each set bit, take the cell attached to that input
   slot, if any.
3. Group cells by identity of their intent word. Within
   each group, OR `1 << cell.bit` into the interest mask
   and record the cell's bit and input slot.

The cone is cached per node and rebuilt after any cell
attach, since which slots hold cells is what it encodes.

### 3.3 Per-kernel state

The interpreter's `last_seen` starts empty and is filled
on first observation in §5's protocol. A compiled slot's
`seen` starts at the revision of the cell the build gave
it and is cleared by an attach. Both are per kernel,
hence per fiber — no contention.

### 3.4 Scope teardown

Dropping a kernel releases its intent words and consumer
caches. Cells created in this scope are dropped when
their `Arc` refcount hits zero (descendants that hold
clones keep them alive). Consumer `last_seen` entries for
dropped cells stay in the map but the pointer is never
observed again — harmless.

---

## 4. Producer protocol

`SharedCellInner::publish(value)` does, in this order:

1. lock the value mutex, store the value, unlock;
2. `revision.fetch_add(1, Release)`;
3. `scope_intent_dirty.fetch_or(1 << bit, Release)`.

Every cell-write path calls `publish`, on all four
engines:

- the interpreter's `set_input` on a cell-bound slot,
  which also covers `commit_write_throughs` and the
  binder's writes;
- the interpreter's `pull` of an output whose broadcast
  cell is attached;
- a compiled kernel's (closure tier, native, or pure
  native) `set_input` on a `shared` slot, which then
  records the new revision as seen, so the kernel does
  not refresh from its own write;
- a compiled kernel's (closure tier or native) pull of an
  output a descendant asked a broadcast cell for (§3.1).

Cost: one mutex acquire/release + one `fetch_add` + one
`fetch_or`. O(1). No upward propagation, no fan-out, no
consumer enumeration.

---

## 5. Consumer protocol

### 5.1 The interpreter

`check_cell_clean(node)` runs ahead of the memoization
early-return in `eval_node`: when the node's clean flag
is set, the early return fires only if the check also
returns clean. It has two passes:

**Pass 1 — collect.** For each group of the node's cone,
load the intent word (Acquire) and AND it with the
group's interest mask; if zero, every cell in the group
is clean for this kernel, skip. Otherwise, for each cell
whose bit is set, load its revision (Acquire) and compare
with `last_seen` (absent counts as 0); a mismatch is
recorded with the cell's input slot.

**Pass 2 — consume and invalidate.** If nothing
mismatched, the node is clean. Otherwise update
`last_seen` for every mismatched cell and clear the clean
flag of **every node whose provenance intersects the set
of mismatched slots**, not only the checked node.

Rationale for invalidating by provenance: updating
`last_seen` consumes this kernel's only signal that the
cell moved, so the re-evaluation it triggers must cover
every memoized node between the moved slot and any
consumer. If only the checked node were invalidated, its
recursive upstream walk would re-check each parent's own
cone, which now reads the just-updated `last_seen` and
reports clean, and the checked node would recompute from
stale parents. A predicate downstream of a cell would
then stay memoized at its pre-write value indefinitely.
The read side therefore applies the write side's rule: a
detected write invalidates every node whose transitive
provenance covers the slot, exactly as `set_input` on
that slot would.

The re-evaluation re-reads the cells through `read_input`,
which takes the cell's value under its mutex.

### 5.2 The compiled kernels

This section applies to the closure tier, native, and
pure native. A compiled kernel keeps no cone cache and
reads no intent word; instead its extern table polls
every cell it holds:

1. `cells_dirty` loads every cell's revision (Acquire)
   and compares it with the slot's `seen`.
2. `refresh_cells` takes the value of each cell whose
   revision moved and writes it into the slot buffer at
   once (the carrier for a one-slot type, the pair for a
   `Ref2` type). It records the revision as seen and
   lists the slot as changed.
3. The kernel marks every step the slot's dependents
   list names as not current and not run since the last
   write.

The poll runs at the first evaluation after a write
(inside the externs' materialization) and before every
`pull` and `eval` between writes. A pull between writes
therefore sees the register's current value, as the
interpreter's revision check does.

---

## 6. Memory ordering

| Operation | Ordering | Pairs with | Guarantee |
| --- | --- | --- | --- |
| Producer cell value write | mutex acquire/release | consumer mutex acquire | atomic value visibility |
| Producer `revision.fetch_add` | Release | consumer `revision.load(Acquire)` | revision visible |
| Producer `intent_dirty.fetch_or` | Release | consumer `intent_dirty.load(Acquire)` | dirty bit visible |
| Consumer `last_seen` / `seen` access | non-atomic | n/a (per-kernel) | no contention |

The producer issues two atomic Release stores; the
interpreter consumer issues two atomic Acquire loads, the
compiled consumer one. Per the C++20 / Rust memory model,
any `load(Acquire)` that observes the value of a
`store(Release)` synchronizes-with that store, establishing
happens-before from every write the producer made before
the Release to every read the consumer makes after the
Acquire. The cell's `Mutex` provides the analogous
synchronizes-with edge for the value itself.

---

## 7. Correctness

**Claim.** For any producer write `W` on cell `C` that
completes its `revision.fetch_add(Release)` before
consumer `F` issues its `revision.load(Acquire)` on `C`:
either `F` re-evaluates every step whose provenance
covers `C`'s slot, or `F`'s last-seen revision for `C`
already reflects a revision ≥ `W`'s post-bump revision.

**Proof sketch (interpreter).** `W`'s `intent_dirty.fetch_or`
happens-after `W`'s `revision.fetch_add` in program order
on the producer. If `F`'s `intent_dirty.load(Acquire)`
observes the bit set by `W`, the load synchronizes-with
`W`'s `fetch_or` Release, so `F` also observes `W`'s
revision bump on its subsequent `revision.load(Acquire)`
— the compare returns a mismatch and §5.1's pass 2
invalidates every node over the slot. If `F`'s load does
not observe the bit set, either `F.last_seen` already
reflects a revision ≥ `W`'s (a prior check captured it;
the bit is sticky), or `F`'s next check will observe it
(memory propagation is bounded).

**Proof sketch (compiled).** The poll reads the revision
directly; a `revision.load(Acquire)` that observes `W`'s
bump synchronizes-with it, and the subsequent `snapshot`
takes the value under the mutex, which synchronizes-with
`W`'s unlock. The dependents of the slot are marked not
current before any step runs in the evaluation or pull.

**Multi-consumer independence.** No consumer's update to
its own last-seen state affects any other consumer. Each
kernel owns its own; only the producer-side atomic state
is shared.

**Sticky-bit semantics.** Intent bits are never cleared.
A scope word that has ever published a write reports its
bit as set for the remainder of the scope's lifetime. The
bulk-mask check therefore forces the interpreter consumer
into the per-cell drill-down, where the revision compare
correctly returns clean for cells whose `last_seen`
already matches. The cost is bounded by the number of
cells in the cone, not the number of writes.

---

## 8. Parent-child composition

A child scope's intent words are independent of its
parent's, and nothing propagates between them. A
consumer that reads cells from several scopes tests them
all through §5.1's loop over cone groups. The lazy cone
builder enumerates every scope (every distinct intent word)
whose cells the cone reads — parent, child,
sibling-of-ancestor, a compiled kernel's word, any depth —
and emits one group per word.

Equivalent to one combined bitmask check:

```text
(parent.intent_dirty & parent_interest)
    | (child.intent_dirty & child_interest)
    | (other.intent_dirty & other_interest)
    | ...   != 0
```

decomposed into a per-scope loop that early-outs per scope
and avoids contention on any single mask.

---

## 9. Bit allocation

### 9.1 Per-scope capacity

The interpreter scope holds a `Vec<Arc<AtomicU64>>` — one
word per 64 cells, appended on demand by the allocator.
Each cell carries a clone of the specific word for its
bit, so growing the list is non-disruptive: every cell
already allocated keeps its reference, and a new cell that
needs a new word appends one. There is no upper bound on
cells per interpreter scope.

A compiled kernel holds one word. Its allocator gives the
first 64 cells distinct bits and every later cell bit 63.
This is a bound, not a fault: no compiled consumer reads
the word, and an interpreter consumer that reads it finds
the shared bit set and drills down to per-cell revisions,
which stay exact. Past 64 `shared` bindings a compiled
kernel's word only makes the interpreter's bulk early-out
coarser.

Cone grouping (§5.1) groups cells by identity of the word:
cells in the same scope's same word share one group; cells
in different words (even in the same scope) form separate
groups, each with its own bulk-mask early-out.

### 9.2 Allocation policy

First-fit, monotonic. Bits are not reused within a scope's
lifetime. Bit positions are stable for the cell's lifetime.

### 9.3 Per-cell, not per-consumer

Bits are allocated per cell (producer-side), so the vector
bound is the scope's cell count — known at scope
construction. Per-consumer allocation would require
registration at sub-context construction, an unbounded
vector that grows with the consumer population, and a
registration-time fan-out. The per-cell scheme avoids all
three.

---

## 10. Happens-before for cell publication

The producer's three-store publish and the consumer's
loads form a synchronization pattern that delivers the
appearance of a single atomic `(value, revision,
intent_bit)` triple, even though it is several separate
atomic operations:

```
                Producer fiber FW                Consumer fiber FA
              ──────────────────────           ──────────────────────

  P1: mutex.lock() ─────╮
  P2: *value = V_new    │ value
  P3: mutex.unlock() ───╯ publication

  P4: revision.fetch_add(1, Release) ════╗
  P5: intent.fetch_or(bit, Release) ════╗║
                                        ║║
                                        ║║          C1: intent.load(Acquire)
                                        ║╠════════► ──synchronizes-with P5──
                                        ║              ↓ happens-before edge:
                                        ║              every store FW made
                                        ║              before P5 is visible
                                        ║              to FA after C1
                                        ║
                                        ║              (bulk-mask dirty;
                                        ║               drill down)
                                        ║
                                        ║           C2: revision.load(Acquire)
                                        ╚═════════► ──synchronizes-with P4──
                                                       ↓ happens-before edge:
                                                       every store FW made
                                                       before P4 is visible
                                                       to FA after C2

                                                    (revision mismatched;
                                                     mark dependents not
                                                     current; re-evaluate)

                                                    C3: read_input(slot)
                                                        → mutex.lock()
                                                        ──synchronizes-with P3──
                                                          mutex acquire-release
                                                          carries the new value
                                                          across the barrier
                                                        ← V_new
```

A compiled consumer begins at C2. If the consumer's
bulk-mask check observes the intent bit set (C1 sees P5's
store), the consumer's subsequent `revision.load(Acquire)`
(C2) is guaranteed to observe at least the revision P4
set, and the subsequent `value.lock()` (C3) is guaranteed
to observe at least the value P2 wrote. If the consumer's
bulk-mask check observes the intent bit clear, the
consumer's cached value and `last_seen` reflect the prior
revision consistently; the next check after P5 propagates
will detect the change.

There is no observable interleaving where the consumer
reads a torn `(value, revision, intent_bit)` triple —
i.e., sees a new intent bit but an old revision, or a new
revision but the old value.

---

## 11. Out of scope

- **Volatile-node handling.** Nodes declared
  `Purity::Nondeterministic` re-evaluate after every write
  unconditionally; that path is independent of cell
  validity tracking. Shared and volatile are orthogonal.
- **Cell value-read atomicity primitive.** The
  `Mutex<Value>` provides single-value atomicity. Validity
  tracking is layered on that primitive.
- **Intent-bit clearing.** Bits are sticky; the per-cell
  revision compare handles the consequence (§7
  "Sticky-bit semantics").
- **Cross-process / distributed cells.** polydat is
  single-process; the substrate makes no claim beyond that
  boundary.

---

## 12. Bounds and invariants

- `revision: u64` — wraparound is treated as a non-event
  (would require 2⁶⁴ writes per cell).
- An interpreter scope's word list grows monotonically
  with cell allocations; cells per scope are unbounded. A
  compiled kernel has one word and the 64-cell bound of
  §9.1.
- `last_seen` per interpreter kernel is unbounded by type;
  in practice bounded by the distinct cell handles the
  kernel has read.
- Cell handles (`Arc::as_ptr`) are stable for the cell's
  `Arc` lifetime. Attaching a cell is a construction-time
  operation on the spawn path and an explicit act through
  `attach_shared_cell` otherwise; both reset the attaching
  kernel's consumer state for the slot, since the identity
  token changed.
- Per-write cost is O(1). Interpreter clean-read cost is
  O(scope words in the node's cone); interpreter dirty-read
  cost is O(cells in the cone) for the compare plus O(nodes
  in the program) for the provenance-wide invalidation.
  Compiled cost is O(`shared` slots) per first evaluation after a write and
  per pull, whether or not anything moved.
- `attach_shared_cell` to a coordinate input slot is a
  precondition violation — coordinate slots have an
  independent write path (`set_inputs`) that bypasses
  cells, producing dual writers. The trait form refuses any
  slot that is not a `shared` binding; the invariant is
  stated at [composition_substrate.md] S3.
