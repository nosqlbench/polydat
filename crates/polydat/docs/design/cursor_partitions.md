# Cursor Partitions (SRD 71)

**Status:** Implemented. This document is the authoritative specification for
Polydat cursor-partition values, the partition-spec language, resolution math,
ordering, cursor narrowing, and partition metadata wires.

**Implementation:**
[`iteration/cursor_partition.rs`](../../src/iteration/cursor_partition.rs),
[`library/partition.rs`](../../src/library/partition.rs), and the cursor
materialization path in
[`dsl/compile.rs`](../../src/dsl/compile.rs).

## 1. Purpose

A cursor describes an ordinal domain such as `[0, 1_000_000)`. A partition
describes a stable sub-domain of that cursor which one scope or fiber owns.
Partitioning is separate from traversal:

```text
base cursor domain       [0 ................................ 1_000_000)
partition resolution      [------ p0 ------)[-- p1 --)[----- p2 -----)
comprehension order                  chooses which p is visited next
cursor `over p`                      narrows iteration to p's interval
```

This separation lets the same domain be split by percentages, absolute sizes,
recipes, windows, or deterministic ordering without adding special operators to
the comprehension algebra. A comprehension sees a `PartitionList` as an
ordinary typed list source. The cursor materializer is responsible for applying
one selected `Partition` to a cursor.

## 2. Data model

### 2.1 `PartitionSpec`

`PartitionSpec` is the parsed but unresolved request. It contains:

- one `Chunking` shape;
- an optional `in start..end` window; and
- one `PartitionOrder`.

A spec has no concrete ordinals until it is resolved against a base half-open
range `[base_start, base_end)`.

### 2.2 `Partition`

A resolved `Partition` contains:

| Field | Meaning |
| --- | --- |
| `idx` | Zero-based generation position. Reordering never changes it. |
| `count` | Number of siblings produced by the same resolution. |
| `start_ord` | Inclusive absolute start ordinal. |
| `end_ord` | Exclusive absolute end ordinal. |
| `start_pct` | Start position in the full base frame. |
| `end_pct` | End position in the full base frame. |
| `base_extent` | Extent of the full base frame used for percentage labels. |

Its cardinality is exactly `end_ord - start_ord`.

### 2.3 `PartitionList`

`PartitionList` is the ordered result of resolving one spec. `Partition`,
`PartitionSpec`, and `PartitionList` travel through Polydat wires as typed
`Value::Ext` reflected values. A resolved partition is effectively constant for
one scope activation.

## 3. Spec language

The complete outer grammar is:

```text
partition-spec := chunking [ "in" window ] [ order ]
window         := sized-bound ".." sized-bound
order          := unchanged | smallest_first | largest_first | random
```

Whitespace outside tokens is ignored. Brackets and parentheses around ranges
are accepted as advisory notation, but every resolved interval is half-open:
`[start, end)`.

### 3.1 Sized bounds

Three spellings identify the unit without a separate suffix declaration:

| Spelling | Interpretation | Example |
| --- | --- | --- |
| Bare integer | Ordinal count or position | `1000` |
| Decimal in `[0.0, 1.0]` | Fraction of the active extent | `0.25` |
| Number followed by `%` | Percentage of the active extent | `25%` |

Percentage and fraction forms are equivalent at resolution. Ordinal bounds are
relative to the active domain's start and are clamped to its end.

### 3.2 Form 1: one range

A range creates exactly one partition:

```text
0..53%
[25%..75%)
100..1000
0.05..0.5
```

Both endpoints must be sized bounds. Tail tokens and gaps are invalid inside a
range. A range that is reversed or rounds to zero ordinals is an error rather
than a silent empty workload.

### 3.3 Form 2: a delta list

A comma-separated list walks left to right from the active domain's start. Each
sized entry advances by that amount and emits one partition:

```text
20%,30%,*
1000,5000,*
0.02,0.10,*
```

Against `[0, 100)`, `20%,30%,*` resolves to `[0,20)`, `[20,50)`, and
`[50,100)`.

The following list-only forms control the remainder or modify entries:

| Form | Meaning |
| --- | --- |
| `*` or `*%` | Emit the remaining extent as one partition. |
| `...` | Repeat the preceding sized delta to the end; truncate, but do not drop, the last chunk. |
| `*/N` | Split the remainder into `N` non-empty, near-equal partitions. |
| `*/recipe:args` | Split the remainder according to normalized recipe weights. |
| `<delta>xN` | Expand one sized delta into `N` consecutive copies. |
| `~<delta>` | Consume a sized gap without emitting a partition. |

At most one remainder-consuming tail is allowed. `...`, `*/N`, and
`*/recipe:args` must be last. `...` requires an immediately preceding sized,
non-gap delta. A list containing only gaps is an error.

If a list has no tail and sums to less than the domain, the trailing gap is
intentionally unassigned. If its sized entries exceed the domain, resolution is
an error.

### 3.4 Form 3: recipes

A recipe expands to normalized percentage deltas and otherwise follows the
same resolution rules as a delta list:

```text
linear:5
ratios:1,3,6
fib:7
geom:8,1.5
zipf:1.2,10
front_heavy:6
```

Implemented recipe families are:

- `linear:N` and `ratios:a,b,c,...`;
- `mul:R` and `mul:S,R`;
- `bin:N`, `fib:N`, and `ln:N`;
- `geom:N,R`, `zipf:s,N`, and `pareto:alpha,N`; and
- `front_heavy:N` and `back_heavy:N`.

Recipes may also shape only the unallocated remainder, for example
`90%,*/fib:5`. Equal-count remainder splitting uses the canonical `*/N` form;
`*/linear:N` is rejected with that spelling as guidance.

## 4. Windows

An `in` clause narrows the domain before chunking:

```text
linear:5 in 25%..75%
10%,20%,* in 100..900
```

Window endpoints resolve against the full base range. Chunking then resolves
against the window, so percentages inside the chunking are window-relative.
Resolved partition ordinals remain absolute.

Percentage metadata uses the full base frame, not a window-local frame. For
example, a partition beginning at the start of `25%..75%` reports
`start_pct = 25`, not zero. This is required when a partition is later applied
to a different cursor extent through `over`.

## 5. Ordering and identity

Ordering is applied only after all partitions have been resolved:

| Order | Behavior |
| --- | --- |
| `unchanged` | Generation order; the default. |
| `smallest_first` | Stable ascending sort by cardinality. |
| `largest_first` | Stable descending sort by cardinality. |
| `random` | Deterministic permutation seeded from the complete spec. |

The ambiguous words `ascending` and `descending` are rejected because they do
not say whether position or size is the sort key. Use `unchanged`,
`smallest_first`, or `largest_first`.

Reordering changes visitation order only. `idx` always identifies the original
generation position, and `count` remains the total sibling count.

## 6. Resolution and rounding

Resolution is a pure operation:

```text
resolve(spec, base_start, base_end) -> ordered PartitionList
```

The resolver follows these steps:

1. Validate the base half-open range.
2. Resolve the optional window against the full base.
3. Resolve chunking against that active domain.
4. Assign generation indices and the sibling count.
5. Apply the requested visitation order without changing indices.

Fractional boundaries use exact cumulative positions and round each boundary
once. Partition sizes are differences between adjacent rounded boundaries. This
prevents rounding error from accumulating across a list: an equal three-way
split of 1,000 ordinals covers all 1,000 rather than producing three 333-item
chunks and silently losing one ordinal.

`*/N` and `subdivide(partition, N)` share the same boundary calculation.
Their non-empty chunk sizes differ by at most one ordinal.

## 7. DSL and comprehension integration

### 7.1 Producing partition lists

The standard `partitions(spec[, extent])` node parses a string and resolves it
against `[0, extent)`; the default extent is 100. Its result is a
`PartitionList` and can be used as a normal comprehension source:

```text
p in partitions("20%,30%,*", 1000000)
```

An existing partition can be divided again with `subdivide(p, n)`, which is
also a valid comprehension source.

### 7.2 Narrowing a cursor

The cursor declaration grammar is:

```text
cursor <name> = <constructor> [over <partition-expression>]
```

For example:

```polydat
cursor q = range(0, 1000000) over p
partition_index := q.cursor.idx
partition_start := q.cursor.start_ordinal
```

At scope setup, the executor resolves the `over` value, narrows the cursor to
the resulting interval, and writes the partition plus its scalar projections
into the cursor's external slots. The partition remains fixed for that scope
activation.

Available cursor metadata wires are:

```text
<cursor>.cursor
<cursor>.cursor.idx
<cursor>.cursor.partition_count
<cursor>.cursor.start_pct
<cursor>.cursor.end_pct
<cursor>.cursor.start_ordinal
<cursor>.cursor.end_ordinal
```

Chained field syntax is flattened to internal wire names by the normal Polydat
field-access rules; it is not a separate runtime object lookup.

### 7.3 Partition value functions

Typed library nodes expose partition values without relying on field syntax:
`cardinality`, `start_of`, `end_of`, `idx_of`, `count_of`, `at`, `mod_in`,
`clamp_in`, `random_in`, and `subdivide`. Their signatures and error behavior
are cataloged in [Library Catalog](library_catalog.md#cursor-partitions-srd-71).

## 8. Error policy

The parser and resolver fail with an explanatory diagnostic for at least:

- an empty spec, unknown recipe, or unknown order;
- a missing or malformed `in` window;
- tail tokens in a range or more than one tail in a list;
- a misplaced fill, split, shaped remainder, repetition, or gap;
- a reversed or zero-width explicit range;
- sized deltas that exceed the active extent;
- a remainder split with no remainder or more chunks than ordinals; and
- shaped weights that would create an empty partition.

Errors are not converted into empty partition lists. A spec that would silently
schedule no useful work is rejected at its earliest authoritative boundary.

## 9. Normative invariants

**CP1 — Half-open ranges.** Every resolved partition denotes
`[start_ord, end_ord)`.

**CP2 — Exact ownership.** A partition's cardinality is
`end_ord - start_ord`; emitted siblings do not overlap. Gaps and an intentionally
unassigned trailing suffix are not owned by any emitted partition.

**CP3 — Cumulative rounding.** Fractional delta boundaries are rounded from the
cumulative exact position, never by summing independently rounded sizes.

**CP4 — No silent oversubscription.** Sized entries may not consume more than
the active domain.

**CP5 — Stable generation identity.** Ordering never rewrites `idx` or `count`.

**CP6 — Stable random order.** The same complete spec produces the same random
permutation.

**CP7 — Full-frame labels.** Windows affect sizing and placement; percentage
metadata continues to describe the full base range.

**CP8 — Scope stability.** A resolved cursor partition is immutable within one
scope activation.

**CP9 — Algebra separation.** Partition lists enter comprehensions through the
ordinary list-source contract. Cursor partitioning does not add a new
comprehension operator.

## 10. Related specifications

- [Comprehension Forms](comprehension_forms.md) owns list-source composition,
  traversal, filtering, ordering, and consumption surfaces.
- [Polydat Grammar](polydat_grammar.md#sec-fields-cursors) owns general cursor
  declaration and field-access syntax.
- [Library Catalog](library_catalog.md#cursor-partitions-srd-71) owns the typed
  node catalog that consumes `Partition` values.
- [Runtime Model](runtime_model.md) owns scope activation, state ownership,
  caching, and invalidation semantics.
