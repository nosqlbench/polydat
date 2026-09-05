# Polytile Tutorial

Polytile is the templating layer of Polydat. A **tile** is a template
whose holes are Polydat expressions. It compiles into the same graph as
everything else, so a rendered document is just another wire: it is
computed per cycle, it can feed other wires, and it costs what its
holes cost.

This tutorial builds up from a one-line text tile to a JSON document
with nested projections that is carried inside a database statement.
Every program here is run by
[`examples/polytile_tutorial.rs`](../examples/polytile_tutorial.rs);
the quoted outputs are what it prints. The complete grammar and
semantics are in [Polytile](design/polytile.md) (SRD 114).

## 1. A text tile

A tile is declared with the `tile` keyword, a name, an encoding, and a
body. Holes are written `${expr}`.

```polydat
input cycle: u64
user_id := mod(hash(cycle), 1000000)
tile greeting : text := "user ${user_id} on cycle ${cycle}"
```

```text
cycle 0 greeting: user 607535 on cycle 0
cycle 1 greeting: user 822465 on cycle 1
```

Three things to notice:

- **A hole is an expression, not a name.** `${cycle * 2}` or
  `${hashed_id(input: cycle, bound: 10)}` are holes too. Any expression
  you could bind with `:=` can sit in a hole.
- **The tile is a wire named `greeting`.** Pull it like any output,
  reference it in another binding, or emit it from the binary with
  `--outputs greeting`.
- **`:= "..."` is the string-literal body form.** It is convenient for
  one-liners and for tiles that must be carried inside another file
  format as a string. Block and heredoc bodies come next.

## 2. A JSON tile with a static arm

For a document, use the `json` encoding with a block body. The body is
the document itself; anything that is not a hole or a directive is
copied byte for byte.

```polydat
input cycle: u64
user_id := mod(hash(cycle), 1000000)
name    := "user-{user_id}"
score   := unit_interval(hash(cycle)) * 100.0
tile doc : json {
    "meta": { "schema": 3, "source": "polydat", "units": { "score": "pct" } },
    "id": ${user_id},
    "name": ${name},
    "label": "id-${user_id}",
    "score": ${score | .1}
}
```

```json
{
    "meta": { "schema": 3, "source": "polydat", "units": { "score": "pct" } },
    "id": 607535,
    "name": "user-607535",
    "label": "id-607535",
    "score": 88.3
}
```

The `json` encoding is **position aware**. Look at the three holes:

- `"id": ${user_id}` sits in value position and its wire is a `u64`, so
  it renders as a bare number.
- `"name": ${name}` also sits in value position, but the wire is a
  string, so the encoder adds the quotes and escapes the contents. You
  do not write `"${name}"` for string values; you may, and then the hole
  is in string position instead.
- `"label": "id-${user_id}"` is a hole **inside a string literal**. Here
  the encoder emits the text of the value with JSON escaping and no
  quotes of its own, because the quotes are already there.

The `"meta"` object has no holes. It is one static run in the compiled
skeleton and costs a single copy per render. A tile with no holes at all
is a constant.

`${score | .1}` shows a **format**. The part after `|` is a printf-style
spec applied before encoding: `.1` gives one decimal place. The
supported specs are width (`8`), zero padding (`04`), alignment (`<8`,
`>8`), precision (`.2`), and hex (`x`, `X`).

## 3. Declared types and formats

The encoder picks its rule from the value's type. When you want a
different rule, declare the type in the hole with `: type`.

```polydat
input cycle: u64
flag := u64_gt(mod(cycle, 2), 0)
tile typed : json {"n": ${cycle}, "as_text": ${cycle: str}, "hex": "${cycle | x}", "odd": ${flag: bool}, "padded": "${cycle | 04}"}
```

```text
cycle 10  typed: {"n": 10, "as_text": "10", "hex": "a", "odd": false, "padded": "0010"}
cycle 255 typed: {"n": 255, "as_text": "255", "hex": "ff", "odd": true, "padded": "0255"}
```

- `${cycle: str}` renders the number as a JSON string.
- `${flag: bool}` renders the `u64` comparison result as `true` or
  `false` rather than `1` or `0`.
- Formats and declared types combine: `${x: str | 04}` pads, then
  quotes.

Every hole is typed when the tile compiles, not when it renders. The
type comes from three places, in priority order:

1. **The declared type**, `${expr: type}`, wins. It must be reachable
   from the expression's type through the same adapter catalog the
   compiler uses between wires: lossless widening such as `u64` to
   `f64`, display text for `str`, and truth values for `bool` are
   inserted for you. A conversion the catalog does not have, such as
   `str` to `u64` or `f64` to `u64`, is a compile error naming the
   tile, the hole, and both types. Write it in the expression instead,
   as `${s as u64}` or `${floor_to_u64(f)}`.
2. **The wire type** otherwise: the same inference every binding gets,
   covering literals, inputs and externs, node return types, and
   `for` elements.
3. **The position** says what the encoding expects there. A `json`
   value position accepts any JSON value and lets the type pick the
   form; a position inside a JSON string, a CSV field, or any `text`
   position expects text and renders any type as text.

A tile declared `(strict)`, or a program compiled in strict mode,
rejects the implicit adapter a declaration would need, so
`${cycle: f64}` becomes an error and `${to_f64(cycle)}` is required.

`polydat explain <file> tiles` prints each hole with its wire type, its
declared type, the expectation of its position, the encoder chosen,
and any adapter inserted, so the typing of a document can be read
before it runs:

```text
== tiles: how each hole is typed and encoded ==
  typed        ${cycle: str}
               wire u64, declared str; expects any JSON value (str)
               -> json string, quoted and escaped  adapter u64 -> str
```

## 4. A CSV row and a raw hole

The `csv` encoding quotes a field only when it needs to. A trailing `!`
makes a hole **raw**: its text is copied with no encoding at all.

```polydat
input cycle: u64
note := "has, comma"
tile row : csv := "${cycle},${note},${note!}"
```

```text
7,"has, comma",has, comma
```

The second field is quoted because it contains a comma. The third is the
same wire, raw, and breaks the row. Raw holes exist for the case where
the value is already in the target encoding, which is how a document is
carried inside a statement in section 7.

## 5. Branches

`@if cond { body } @else { body }` renders one of two bodies. The
condition is any expression; nonzero is true.

```polydat
input cycle: u64
hot := u64_gt(mod(cycle, 3), 1)
tile status : json {"cycle": ${cycle}, "state": @if hot { "hot" } @else { "cold" }}
```

```text
cycle 0 status: {"cycle": 0, "state": "cold"}
cycle 1 status: {"cycle": 1, "state": "cold"}
cycle 2 status: {"cycle": 2, "state": "hot"}
```

The whitespace that pads the braces is not part of the body: `{ "hot" }`
renders `"hot"`. Put any spacing you want to keep in the static text
outside the block. Both bodies may contain holes, other directives, and
JSON structure of their own, as `"alert"` does in the demo file at the
end.

## 6. Projections

`@for` repeats its body once per tuple of a comprehension. The
comprehension's element names are wires inside the body, and so is
everything the enclosing program defines.

```polydat
input cycle: u64
base := cycle * 100
axes := for k in 1..3, side in left,right
tile samples : json {
    "base": ${base},
    "points": [ @for i in 0..3 { {"i": ${i}, "v": ${base + i}} } ],
    "grid": "@for axes sep \"; \" {${k}-${side}}"
}
```

```json
{
    "base": 200,
    "points": [ {"i": 0, "v": 200},{"i": 1, "v": 201},{"i": 2, "v": 202} ],
    "grid": "1-left; 1-right; 2-left; 2-right"
}
```

Two forms of source appear here:

- **Inline comprehension text**, `i in 0..3`. The body reads `i` and
  the outer wire `base`.
- **A bound producer**, `axes`, declared once with the `for` construct
  and reused. Its elements `k` and `side` are the body's wires. This is
  the same producer surface as [The `for`
  Construct](design/for_traversal.md); a tile projects over it instead
  of traversing it.

`sep "..."` sets the text between repetitions. The default is `,` for
`json` and `csv` and nothing for `text`. Inside a JSON block body the
quotes of the separator are written `\"` because the body is inside a
string; the `"grid"` line shows the spelling.

The body compiles to its own small program that is built once and
re-run per tuple with reused scratch state, so a projection allocates
nothing per element. A projection body can contain holes, branches,
static text, and further projections.

### Nested projections, derivations, and generators

Every source the `for` construct accepts projects. A body may contain
another `@for`, a producer may be filtered or ordered in place, and a
generator call is an expression over the program's wires:

```polydat
input cycle: u64
base := cycle * 10
ks := for k in 1..7
tile grid : json {
    "rows": [ @for r in 0..2 { {"r": ${r}, "cells": [ @for c in 0..3 { ${base + r * 10 + c} } ]} } ],
    "big": [ @for ks where {k} > 4 { ${k} } ],
    "sampled": [ @for k in 1..100 order halton/4 { ${k} } ],
    "pick": [ @for g in hash_range(cycle, 1000) { ${g} } ]
}
```

```json
{
    "rows": [ {"r": 0, "cells": [ 10,11,12 ]},{"r": 1, "cells": [ 20,21,22 ]} ],
    "big": [ 5,6 ],
    "sampled": [ 50,25,75,13 ],
    "pick": [ 465 ]
}
```

- **Nested `@for`.** The inner projection reads the outer element `r`,
  the program wire `base`, and `cycle` alike. It compiles as a tile of
  its own inside the outer body's program, so each level is one
  compiled program however many tuples flow through it.
- **Derivations.** `ks where {k} > 4` filters the producer in place;
  `order halton/4` samples four tuples of a hundred in Halton order.
  A predicate sees the comprehension's elements. A `{name}` for a wire
  outside the comprehension is a compile error, as it is for `for`.
- **Generators.** `hash_range(cycle, 1000)` is compiled as a wire of
  the program and its value is the element, one tuple for a scalar and
  one per item for a list. It is typed by the same inference as a hole,
  so it may read any wire in scope, including an outer element when
  nested.

A projection's source must have a finite tuple set. A continuous
interval such as `x in 0.0..1.0` is rejected with a message; project
over a discrete range or list instead.

## 7. Splicing one tile into another

A hole that names another tile of the same encoding is a **splice**: the
inner tile's skeleton is inlined at compile time, holes and all. Across
encodings the inner tile is an ordinary wire whose rendered text enters
through the hole.

```polydat
input cycle: u64
tile inner : json {"n": ${cycle}, "double": ${cycle * 2}}
tile outer : json {"first": ${inner}, "second": ${inner}, "wrapped": true}
tile stmt  : text := "INSERT INTO docs (id, body) VALUES (${cycle}, '${outer!}')"
```

```text
outer: {"first": {"n": 5, "double": 10}, "second": {"n": 5, "double": 10}, "wrapped": true}
stmt:  INSERT INTO docs (id, body) VALUES (5, '{"first": {"n": 5, "double": 10}, "second": {"n": 5, "double": 10}, "wrapped": true}')
```

`outer` splices `inner` twice; both copies of `${cycle}` are the same
wire, evaluated once. `stmt` is a `text` tile that carries the JSON
document raw with `${outer!}`. Without `!` a `text` tile would still
copy the text unchanged, but inside a `json` tile a non-raw hole
holding a `text` tile would be encoded as a JSON string, which is
usually what you want when a document embeds a message body.

## 8. Living inside another template

Tiles are often written inside YAML, a Jinja template, or another
system's own `${...}` syntax. Options on the tile change its delimiters
and directive sigil so it can coexist:

```polydat
input cycle: u64
tile page : text (delims "<%" "%>", sigil "#") := "Hello {{ user.name }}, cycle <%cycle%> #if cycle { is live } #else { is zero }"
```

```text
cycle 0 page: Hello {{ user.name }}, cycle 0 is zero
cycle 3 page: Hello {{ user.name }}, cycle 3 is live
```

The `{{ user.name }}` passes through untouched for whatever renders it
next. With default delimiters, a literal `${` is written `${${`.

## 9. Templates handed in as data

A workload runner rarely has a `.polydat` file with a `tile` statement
in it. It has a YAML document, a JSON value it already parsed, or a
string a user typed. Polytile treats that boundary as first class: a
template may arrive as **template text**, as **structural JSON text**,
or as an **already-parsed JSON value**, and all three become the same
`tile` statement inside the program.

### Structural JSON

In the structural form the template is a JSON document, and its strings
are the insertion points. A string that is exactly one hole is a
**value hole**: the node is replaced by the value, encoded by type. A
string with text around a hole is a **string hole**. Directives use the
carrier's own shapes: an array whose first element is `"@for ..."` or
`"@if ..."`, and an object key that is `"@for ..."`.

```polydat
input cycle: u64
tags := for t in a,b
verbose := u64_gt(cycle, 0)
doc := polytile_json(<<<
{
  "id": "${cycle}",
  "label": "row-${cycle}",
  "samples": [ "@for s in 0..2", { "n": "${s}", "twice": "${s * 2}" } ],
  "audit": [ "@if verbose", { "by": "ops" }, "@else", null ],
  "tags": { "@for tags": { "${t}": true } }
}
>>>)
line := polytile("text", "cycle ${cycle}: ${doc!}")
```

```text
cycle 0 doc: {"id": 0, "label": "row-0", "samples": [{"n": 0, "twice": 0},{"n": 1, "twice": 2}], "audit": [null], "tags": {"a": true,"b": true}}
cycle 1 doc: {"id": 1, "label": "row-1", "samples": [{"n": 0, "twice": 0},{"n": 1, "twice": 2}], "audit": [{"by": "ops"}], "tags": {"a": true,"b": true}}
cycle 1 line: cycle 1: {"id": 1, "label": "row-1", "samples": [{"n": 0, "twice": 0},{"n": 1, "twice": 2}], "audit": [{"by": "ops"}], "tags": {"a": true,"b": true}}
```

Read the strings against the rules:

| String in the template | Rendered as |
| --- | --- |
| `"${cycle}"` alone | a value hole, so `"id": 0` is a bare number |
| `"row-${cycle}"` | a string hole, so `"row-0"` stays a string |
| `"${cycle: str}"` | a declared `str` forces string position: `"0"` |
| `[ "@for s in 0..2", {...} ]` | the remaining elements repeat per tuple as array items |
| `[ "@if verbose", {...}, "@else", null ]` | one side or the other, as array items |
| `{ "@for tags": { "${t}": true } }` | the members repeat per tuple and merge into the object |

The structural template is valid JSON, so it can live as data in any
configuration file and be validated by ordinary JSON tooling before
Polydat ever sees it. A JSON value that contains no holes is just a
constant document.

### The `polytile` forms in source

`polytile_json(<json>)` takes structural JSON text; `polytile(<encoding>,
<template>)` takes template text in any encoding. The body may be a
string literal or a `<<< ... >>>` heredoc, and it is taken raw: the
template's holes belong to the tile, not to Polydat's string
interpolation. Delimiters and the sigil are options:

```polydat
page := polytile("text", "<%cycle%> {{ keep }} #if cycle { on } #else { off }", open: "<%", close: "%>", sigil: "#")
```

These are program transforms, not runtime calls. The parser turns a
`polytile` binding into a `tile` statement before compilation, so a
host that only holds strings can emit one line of source per template
and the result is indistinguishable from a tile written by hand. The
pretty-printer prints it as the `tile` statement it became.

### The Rust boundary

A host that has already parsed its configuration skips source text
entirely. `polydat::tile` builds a `TileDef` from a template string, a
JSON string, or a `serde_json::Value`, and
`compile_polydat_with_tiles` compiles it with a program:

```rust
use polydat::tile::{compile_polydat_with_tiles, tile_from_json_value, Span, TileOptions};

let template = serde_json::json!({
    "id": "${cycle}",
    "points": [ "@for s in 0..3", { "n": "${s}", "v": "${cycle + s}" } ]
});
let tile = tile_from_json_value("doc", &template, &TileOptions::default(), Span { line: 0, col: 0 })?;
let mut kernel = compile_polydat_with_tiles("input cycle: u64\n", vec![tile])?;
kernel.set_inputs(&[4]);
println!("{}", kernel.pull("doc").as_str());
```

```text
cycle 4 doc: {"id": 4, "points": [{"n": 0, "v": 4},{"n": 1, "v": 5},{"n": 2, "v": 6}]}
```

The three entry points are `tile_from_text`, `tile_from_json_text`, and
`tile_from_json_value`. Each returns the same `TileDef` the `tile`
keyword produces, and the tiles see every wire the program defines.

## 10. Running a tile program from the command line

[`examples/polytile_demo.polydat`](../examples/polytile_demo.polydat)
puts the pieces together: a reading document with a projection and a
branch, and the statement that loads it.

```polydat
input cycle: u64

const keyspace := "toy"
const table := "readings"

(tenant, device, reading) := mixed_radix(cycle, 20, 50, 0)
tenant_id := hashed_id(input: tenant, bound: 1000000)
device_id := hashed_uuid(interleave(tenant, device))
temp_c    := normal_sample(input: hash(cycle), mean: 21.5, stddev: 2.0)
status    := weighted_strings(hash(hash(cycle)), "ok:0.97;degraded:0.02;error:0.01")
alert     := u64_gt(mod(cycle, 5), 3)

tile doc : json {
    "meta": { "schema": 3, "source": "polydat", "units": { "temp": "C" } },
    "tenant": ${tenant_id},
    "device": ${device_id},
    "reading": { "n": ${reading}, "temp": ${temp_c | .2}, "status": ${status} },
    "history": [ @for i in 0..3 { {"n": ${reading + i}, "temp": ${temp_c + to_f64(i) | .1}} } ],
    "alert": @if alert { { "level": "high", "cycle": ${cycle} } } @else { null }
}

tile load : text <<<
INSERT INTO ${keyspace}.${table} (tenant_id, device_id, doc) VALUES (${tenant_id}, '${device_id}', '${doc!}')
>>>
```

The heredoc body `<<< ... >>>` is the third body form. It keeps the
text exactly, minus one newline on each side, and is the natural form
for statements.

Run it with the `polydat` binary (see the README for installing it):

```sh
polydat run examples/polytile_demo.polydat --cycles 5 --emit map --outputs load -q
```

The fifth row, where the `alert` branch is taken:

```text
load=INSERT INTO toy.readings (tenant_id, device_id, doc) VALUES (603978, '86a03fe5-bba5-4b06-b8ec-3cb9441ca1b6', '{
    "meta": { "schema": 3, "source": "polydat", "units": { "temp": "C" } },
    "tenant": 603978,
    "device": "86a03fe5-bba5-4b06-b8ec-3cb9441ca1b6",
    "reading": { "n": 0, "temp": 22.14, "status": "ok" },
    "history": [ {"n": 0, "temp": 22.1},{"n": 1, "temp": 23.1},{"n": 2, "temp": 24.1} ],
    "alert": { "level": "high", "cycle": 4 }
}')
```

`--emit jsonl --outputs cycle,doc` gives one JSON object per cycle with
the document as a string field, and `polydat check --stats` reports the
tile's nodes alongside everything else in the program. The tile is not
special to the harness; it is a string output.

To write the document itself, one per cycle with nothing around it,
name the tile in the emit format:

```sh
polydat run examples/polytile_demo.polydat --cycles 3 --emit tile:doc -q
```

This is the same transform as every other `--emit`: the binary appends
a binding that emits the selected wire in the `text` format, so the
output is the rendered tile byte for byte.

When a file's tiles are written in another system's delimiters and
declare none of their own, `--tile-delims '<%' '%>'` and
`--tile-sigil '#'` supply the defaults. They too are a transform: each
tile that left its options to the default is re-read under the host's
before the program compiles, and a tile that declared any option keeps
all of its own. `polydat explain <file> tiles` then shows the program
as it compiled, with each tile's skeleton summary above the hole
typing.

A tile may also be declared inside a module body. It inlines with the
call like any binding: the tile is named with the module's prefix, its
holes and branch conditions read the caller's arguments, and its
projections' generator expressions are rewritten the same way. Two
calls to the module give two tiles.

## What a tile compiles to

Knowing the lowering makes the rules above predictable:

1. Each hole becomes a `tile_encode` node in the enclosing scope. Its
   constant spec records the encoding, the hole's position (value or
   inside a string), the declared type, the format, and the raw flag.
   Its input is the hole's expression, compiled like any binding.
2. Each branch condition becomes a `tile_encode` node that yields `1`
   or `0`.
3. Each projection becomes a child program: an input carrying the
   tuple index, one `extern` per element, one `extern` per outer wire
   the body reads (`cycle` included), and one `tile_encode` binding per
   hole in the body. The comprehension is embedded in the skeleton.
4. The tile itself is one `tile_render` node whose constant is the
   skeleton and whose inputs are the encoded holes. It concatenates
   static runs and hole text, selects branch arms, and drives child
   programs over the comprehension stream.

Because the hole nodes are ordinary wires, `explain` and `viz` show
them, dead holes are pruned with the rest of the graph, and a hole
shared by two tiles is computed once.

## Limits at this level

- Inside a projection body, only the `u64` to `f64` widening and the
  `str` and `bool` readings are available as declared-type adapters;
  other widenings must be written in the expression.
- A nested `@for` cannot sit inside a string position of a JSON tile;
  project the inner text into a wire of its own and reference it.
- Continuous sources do not project; a predicate sees only the
  comprehension's elements; a generator's list value arrives as text,
  so JSON arrays and scalars expand, but stream-valued generators do
  not.
- In the structural form, a projection or branch that sits beside
  static members or elements contributes its own separators; if it
  renders zero tuples the document keeps a dangling comma. Put such
  projections in their own array or object until the cardinality check
  lands.
- Rendering runs at engine level P1. The P2 and P3 renderers, the
  structural JSON front end, and the `polytile` node for hosts that only
  hold strings are later steps of the SRD's plan.
