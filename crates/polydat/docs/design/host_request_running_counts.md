---
type: record
title: "Host Request: Running Counts and Accumulation"
timestamp: 2026-09-25
description: A host request for state that survives across pulls, and what any answer must satisfy. A record, pending fold into the specifications.
tags: [record, host]
---

# Host Request: Running Counts and Accumulation

A requirements record, not a design. A host
converting the auxiliary binaries that check vector datasets and
results named the capabilities polydat lacks for that work. Three of
the four are node-library additions of the ordinary kind and are not
recorded here. The second asks for state that survives across pulls,
which polydat's evaluation model does not have, so the request is
written down as the host stated it, with what it is for, what polydat
offers in its place today, and what any future answer has to satisfy.

## What was asked

> A running count usable from bindings. Either shared-cell
> accumulation under the type-stable cell rules, or a phase-scoped
> `running_count(cond)` node. This is the only way to express "rank
> of this record among the matching ones", which the recall audit
> needs to pair a global query ordinal with its label-local
> ground-truth row, and "count of records matching", which the
> per-label cardinality checks compare against profile counts.

The other three items of the same request were pure functions of their
inputs and belonged in the node library. All three landed on
2026-09-19:

- The element and set operations, as `vec_len`, `vec_at`, `vec_max`,
  `vec_min`, `vec_contains`, `vec_position`, `vec_intersect_count` and
  `vec_set_eq`, each in an `_i32` and an `_i64` form, plus
  `vec_count_below`, which is the rank a sorted list gives through
  `vec_position` but without requiring the list to be sorted or the
  value to be in it. Getting a list onto a wire needed
  `str_to_vec_i32` and `str_to_vec_i64`, which a code comment already
  told authors to use and which did not exist; reaching a signed port
  needed `to_i64`, since an integer literal is a `u64` and the adapter
  catalog will not heal that pair silently.
- The name reading, as `str_to_u64` for a whole string and
  `str_digit_suffix` for the run of digits ending one, which is what
  "a profile name's numeric suffix" asked for and what the `as` cast
  cannot do.

- `metadata_count_of(handle, value)` and `predicate_count_of(handle,
  value)`, which answer the same two quantities over a facet rather
  than over a list, for the host that has the handle where it needs the
  count. Each is a linear read of the facet, which is what the question
  is: the reader holds the values and nothing indexes them by content.
  Both arguments are fixed for a scope in the intended use, so the
  count is computed once at scope init and reused, the way the facet
  itself is loaded once. The value is an `i64`, the facet's own scalar
  type; `metadata_value_at` formats that same scalar as text, so a host
  comparing labels as strings and one counting them here read one
  facet.

## What it is for

Two audits over a vector dataset with per-vector metadata and
per-query predicates:

- **Recall audit.** A query is addressed by a global ordinal. Its
  ground truth is a row in a label-local table, indexed by the
  query's rank among the queries whose predicate matches the same
  label. The audit needs, for each query, that rank.
- **Per-label cardinality check.** For each label, the number of
  records whose metadata matches it, compared against the count the
  profile declares.

The host's framing of both is a counter that advances as records are
visited: rank is "how many matching records have I seen before this
one", cardinality is "how many matching records did I see in all".

## Why polydat has no such primitive

- An output is owned by the provenance of its inputs. A value that
  depends on how many times a node has been pulled is not a function
  of any input, so it is nondeterministic by construction, it cannot
  be replayed from its coordinates, and it cannot agree across
  engines or fibers ([Evaluation Model](evaluation_model.md),
  [Engines](engines.md) §7, [Cross-Fiber
  Invalidation](cross_fiber_invalidation.md)).
- Shared cells are typed and first-writer-wins
  ([Scope Model](scope_model.md) §6.1; the `set_or_get` node). They
  publish a value computed once to later phases; they are not an
  accumulator, and making one accumulate would give every fiber and
  every re-run a different program.
- A `for` traversal captures its scope when it opens and materializes
  its tuples then ([The `for` Construct](for_traversal.md) §3.1); an
  activation has no "previous activation" to read a count from.

## What polydat offers today

Both quantities are pure functions of data the host already has, once
the integer-vector operations from the same request exist:

- **Rank of a record among the matching ones** is the number of
  matching ordinals smaller than the record's ordinal: with the
  matching-ordinal list for the predicate, `vec_position(list,
  ordinal)` on a sorted list, or a count-below over it. The result is
  owned by the record and the list, replays, and agrees everywhere.
- **Count of records matching** is the length of the same list, or
  `predicate_count_of` / `metadata_count_of` over the handle, cached
  per handle the way the facet loaders are.

Neither needs a visit order, which is the point: the counter framing
smuggles in an evaluation order that polydat deliberately does not
promise.

## What a future answer must satisfy

If a genuine accumulation need appears that no pure formulation
covers, a solution is admissible only when:

1. its value is a function of declared inputs and provenance, never
   of pull count or fiber schedule;
2. it replays: the same coordinates give the same value on every
   engine and after invalidation;
3. it does not add a special-cased wire or a per-cycle notion to the
   scope rules (the same rule that keeps `cycle` an ordinary
   coordinate);
4. it is a program transform or a node, not a runtime mode.

A fold over a materialized traversal, computed once at scope init from
the tuple set the traversal already holds, is the shape most likely to
meet all four; it is not designed here.

## Status

The accumulation question is **open**, recorded 2026-09-17.

Every pure item landed on 2026-09-19. Both quantities the host wanted a
counter for are expressible twice over: from a list, as
`vec_count_below_i32(matching, ordinal)` and `vec_len_i32(matching)`,
and from a facet, as `predicate_count_of(h, v)` and
`metadata_count_of(h, v)`. `tests/vector_set_ops.rs` checks the list
pair gives the same answer on the interpreter with and without cones,
on the closure tier, and on native code — which is the property a
counter could not have had.

The facet pair is tested for its ports, its registration, and its
fallback only. Counting real records needs a dataset download, which
the suite does not do, and which is why its neighbours
(`metadata_value_at`, `predicate_value_at`, `metadata_content_count`)
have no tests at all. Closing that gap means a fixture facet the tests
can open, and it would cover the whole accessor family rather than
these two.
