# Host Request: Running Counts and Accumulation

A requirements record, not a design. On 2026-09-17 an nmbrs agent
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

The other three items of the same request, for the record: element
and set operations over `vec_i32` and `vec_i64` (`vec_len`, `vec_at`,
`vec_max`, `vec_min`, `vec_contains`, `vec_position`,
`vec_intersect_count`, `vec_set_eq`); `metadata_count_of(handle,
value)` and `predicate_count_of(handle, value)` over a dataset handle;
and a `str_to_u64` that parses a profile name's numeric suffix. Those
are pure functions of their inputs and belong in the node library.

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
  [Engine Parity](engine_parity.md), [Cross-Fiber
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

Open. Recorded 2026-09-17. The three pure items of the request are the
node library's to add and are not blocked on this question.
