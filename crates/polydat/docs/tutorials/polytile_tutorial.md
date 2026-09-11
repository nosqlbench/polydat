# Polytile Tutorial

Polytile is the templating layer of Polydat. A **tile** is a template
whose holes are Polydat expressions. It compiles into the same graph as
everything else, so a rendered document is just another wire: it is
computed per cycle, it can feed other wires, it runs on every engine
level, and it costs what its holes cost.

This tutorial builds up from a one-line text tile to a JSON document
with nested projections carried inside a database statement, then works
through the corner cases that matter in practice: escapes, layout,
encoding rules, empty projections, modules, what the compiler refuses,
and what the engines guarantee. Every program here is run by
[`examples/polytile_tutorial.rs`](../../examples/polytile_tutorial.rs);
the quoted outputs are what it prints. The complete grammar and
semantics are in [Polytile](../design/polytile.md) (SRD 114); the compiled
representation is in [Compiled Non-Scalar
Slots](../design/compiled_handles.md) (SRD 115).

Part one, sections 1 to 10, is the basics. Part two, sections 11 to 16,
is the corner cases.

## 1. A text tile

A tile is declared like any other wire binding: the `tile` keyword, a
name, an optional encoding after a colon, and `:=` before the body. The
encoding defaults to `text`, so a one-line text tile needs none. Holes
are written `${expr}`.

```polydat
input cycle: u64
user_id := mod(hash(cycle), 1000000)
tile greeting := "user ${user_id} on cycle ${cycle}"
```

```text
cycle 0 greeting: user 607535 on cycle 0
cycle 1 greeting: user 822465 on cycle 1
```

Three things to notice:

- **A hole is an expression, not a name.** `${cycle * 2}` or
  `${hashed_id(input: cycle, bound: 10)}` are holes too. Any expression
  you could bind with `:=` can sit in a hole.
- **The tile is a wire named `greeting`.** That is what `:=` says, as
  it does in every binding: pull it like any output, reference it in
  another binding, or emit it from the binary with `--outputs greeting`.
- **`"..."` is the string-literal body form.** It is convenient for
  one-liners and for tiles that must be carried inside another file
  format as a string. Block and heredoc bodies come next; every form
  follows `:=`.

## 2. A JSON tile with a static arm

For a document, use the `json` encoding with a block body. The body is
the document itself; anything that is not a hole or a directive is
copied byte for byte.

```polydat
input cycle: u64
user_id := mod(hash(cycle), 1000000)
name    := "user-{user_id}"
score   := unit_interval(hash(cycle)) * 100.0
tile doc : json := {
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

A block body keeps the layout you wrote, minus the indentation of the
statement it sits in: the common leading whitespace of its lines is
removed, so a tile declared inside a `for` body renders the same bytes
as one at top level. Section 11 shows the rule in detail.

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
tile typed : json := {"n": ${cycle}, "as_text": ${cycle: str}, "hex": "${cycle | x}", "odd": ${flag: bool}, "padded": "${cycle | 04}"}
```

```text
cycle 10 typed: {"n": 10, "as_text": "10", "hex": "a", "odd": false, "padded": "0010"}
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
   from the expression's type through the lossless adapters the
   compiler inserts between wires: widening such as `u64` to `f64`,
   display text for `str`, and truth values for `bool`. A conversion
   outside that set, such as `str` to `u64` or `f64` to `u64`, is a
   compile error naming the tile, the hole, and both types. Write it in
   the expression instead: `${s as u64}` reaches the whole adapter
   catalog, including parses, and `${floor_to_u64(f)}` names the
   rounding.
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
Section 15 shows the messages.

`polydat explain <file> tiles` prints each tile's skeleton and then
each hole with its wire type, its declared type, the expectation of its
position, and the encoder chosen, so the typing of a document can be
read before it runs. For the demo file of section 10, abridged to a
few of its holes:

```text
$ polydat explain polytile_demo.polydat tiles
== tiles: how each hole is typed and encoded ==
Each tile compiled to a skeleton: static runs copied whole, encoded holes, branches, and projections whose bodies are programs of their own.
  doc          json: 14 static run(s) totalling 245 bytes, 8 hole(s), 1 branch(es), 1 projection(s)
               projection body 0:
                 input __tuple: u64
                 extern i: u64
                 extern reading: u64
                 extern temp_c: f64
                 __b0 := (reading + i)
                 __b1 := (temp_c + to_f64(i))
  load         text: 6 static run(s) totalling 59 bytes, 5 hole(s), 0 branch(es), 0 projection(s)
Every hole was typed before the tile compiled: a declared type wins, otherwise the wire's type; the hole's position says what the encoding expects there, and the two pick the encoder.
  doc          ${tenant_id}
               wire u64; expects any JSON value (u64)
               -> json number
  doc          ${device_id}
               wire str; expects any JSON value (str)
               -> json string, quoted and escaped
  doc          ${temp_c | .2}
               wire f64; expects any JSON value (f64)
               -> json number, format .2
  doc          ${if alert}
               wire u64; expects a truth value (u64)
               -> truth value as 1 or 0
  load         ${doc!}
               wire str; expects text (str)
               -> raw text, no escaping
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
cycle 7 row: 7,"has, comma",has, comma
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
tile status : json := {"cycle": ${cycle}, "state": @if hot { "hot" } @else { "cold" }}
```

```text
cycle 0 status: {"cycle": 0, "state": "cold"}
cycle 1 status: {"cycle": 1, "state": "cold"}
cycle 2 status: {"cycle": 2, "state": "hot"}
```

The whitespace that pads the braces is not part of the body: `{ "hot" }`
renders `"hot"`. Put any spacing you want to keep in the static text
outside the block. Both bodies may contain holes, other directives, and
JSON structure of their own, as `"alert"` does in the demo file of
section 10.

## 6. Projections

`@for` repeats its body once per tuple of a comprehension. The
comprehension's element names are wires inside the body, and so is
everything the enclosing program defines.

```polydat
input cycle: u64
base := cycle * 100
axes := for k in 1..3, side in left,right
tile samples : json := {
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
  Construct](../design/for_traversal.md); a tile projects over it instead
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
tile grid : json := {
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
  outside the comprehension is a compile error, as it is for `for`;
  section 15 shows the message.
- **Generators.** `hash_range(cycle, 1000)` is compiled as a wire of
  the program and its value is the element, one tuple for a scalar and
  one per item for a list such as a JSON array, a vector, or a stream.
  It is typed by the same inference as a hole, so it may read any wire
  in scope, including an outer element when nested.

A projection's source must have a finite tuple set. A continuous
interval such as `x in 0.0..1.0` has none of its own, so it is rejected
unless an order with a sampling strategy and a count gives it one:
`x in 2.0..4.0 order halton/4` renders four Halton points from the
interval, and `x in 0.0..1.0, y in 10.0..20.0 order sobol/3` samples
the two axes jointly. The sampling strategies are `halton`, `sobol`,
`lhs`, and `shuffle`; ordering a continuous source with `lex` is a
compile error.

## 7. Splicing one tile into another

A hole that names another tile of the same encoding is a **splice**: the
inner tile's skeleton is inlined at compile time, holes and all. Across
encodings the inner tile is an ordinary wire whose rendered text enters
through the hole.

```polydat
input cycle: u64
tile inner : json := {"n": ${cycle}, "double": ${cycle * 2}}
tile outer : json := {"first": ${inner}, "second": ${inner}, "wrapped": true}
tile stmt := "INSERT INTO docs (id, body) VALUES (${cycle}, '${outer!}')"
```

```text
cycle 5 outer: {"first": {"n": 5, "double": 10}, "second": {"n": 5, "double": 10}, "wrapped": true}
cycle 5 stmt: INSERT INTO docs (id, body) VALUES (5, '{"first": {"n": 5, "double": 10}, "second": {"n": 5, "double": 10}, "wrapped": true}')
```

`outer` splices `inner` twice; both copies of `${cycle}` are the same
wire, evaluated once. `stmt` is a `text` tile that carries the JSON
document raw with `${outer!}`. Without `!` a `text` tile would still
copy the text unchanged, but inside a `json` tile a non-raw hole
holding a `text` tile would be encoded as a JSON string, which is
usually what you want when a document embeds a message body.

A splice belongs where a value belongs. Writing `"s": "${inner}"`, a
same-encoding tile's name inside a string literal, would inline its
skeleton unescaped inside the string, and the compile-time check of
section 15 rejects the result. To carry the document's text inside a
string, bind it first (`s := inner`) and use `${s}`: a binding is an
ordinary string wire, and the encoder escapes it.

## 8. Living inside another template

Tiles are often written inside YAML, a Jinja template, or another
system's own `${...}` syntax. Options on the tile change its delimiters
and directive sigil so it can coexist:

```polydat
input cycle: u64
tile page (delims "<%" "%>", sigil "#") := "Hello {{ user.name }}, cycle <%cycle%> #if cycle { is live } #else { is zero }"
```

```text
cycle 0 page: Hello {{ user.name }}, cycle 0 is zero
cycle 3 page: Hello {{ user.name }}, cycle 3 is live
```

The `{{ user.name }}` passes through untouched for whatever renders it
next. With default delimiters, a literal `${` is written `${${`;
section 11 shows it.

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
cycle 0 line: cycle 0: {"id": 0, "label": "row-0", "samples": [{"n": 0, "twice": 0},{"n": 1, "twice": 2}], "audit": [null], "tags": {"a": true,"b": true}}
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
`compile_polydat_kernel_with_tiles` compiles it with a program:

```rust
use polydat::tile::{compile_polydat_kernel_with_tiles, tile_from_json_value, Span, TileOptions};

let template = serde_json::json!({
    "id": "${cycle}",
    "points": [ "@for s in 0..3", { "n": "${s}", "v": "${cycle + s}" } ]
});
let tile = tile_from_json_value("doc", &template, &TileOptions::default(), Span { line: 0, col: 0 })?;
let mut kernel = compile_polydat_kernel_with_tiles("input cycle: u64\n", vec![tile])?;
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

[`examples/polytile_demo.polydat`](../../examples/polytile_demo.polydat)
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

tile doc : json := {
    "meta": { "schema": 3, "source": "polydat", "units": { "temp": "C" } },
    "tenant": ${tenant_id},
    "device": ${device_id},
    "reading": { "n": ${reading}, "temp": ${temp_c | .2}, "status": ${status} },
    "history": [ @for i in 0..3 { {"n": ${reading + i}, "temp": ${temp_c + to_f64(i) | .1}} } ],
    "alert": @if alert { { "level": "high", "cycle": ${cycle} } } @else { null }
}

tile load := <<<
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
$ polydat run polytile_demo.polydat --cycles 5 --emit map --outputs load -q
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
typing, as in section 3.

## 11. Escapes, braces, and block layout

Two questions come up as soon as a template contains the characters
the grammar uses. A literal open delimiter is written by doubling it,
`${${`, and it renders as `${`. Braces in static text are ordinary
text; they only mean something after a directive keyword, and even
there they are balanced, so a JSON object can sit inside a branch or a
projection body without escaping.

```polydat
input cycle: u64
tile lit := "a literal ${${cycle} and braces {ok} around cycle ${cycle}"
tile block : json := {
    "a": ${cycle},
        "b": { "deep": ${cycle} }
}
```

```text
cycle 3 lit: a literal ${cycle} and braces {ok} around cycle 3
cycle 3 block: {
    "a": 3,
        "b": { "deep": 3 }
}
```

The block body shows the layout rule exactly. In the source the tile
statement is indented twelve spaces and its lines sixteen and twenty;
the rendered document has the statement's indentation removed and
keeps the four extra spaces of the `"b"` line. What you wrote relative
to the statement is what you get, wherever the statement sits. Heredoc
and string bodies are copied exactly.

## 12. Encoding corner cases

The encoders differ in what they do with a value that contains the
encoding's own special characters. One string with a quote, a comma,
and a newline in it, through each of them:

```polydat
input cycle: u64
line := printf("quote {} and comma, then newline{}end", "\"", "\n")
tile injson : json := {"s": ${line}, "in": "x-${line}-y", "n": ${cycle: str}}
tile row : csv := "${line},${cycle},${cycle | 04}"
tile txt := "[${line}]"
tile raw : json := {"raw": "${line!}"}
```

```text
cycle 1 injson: {"s": "quote \" and comma, then newline\nend", "in": "x-quote \" and comma, then newline\nend-y", "n": "1"}
cycle 1 row: "quote "" and comma, then newline
end",1,0001
cycle 1 txt: [quote " and comma, then newline
end]
cycle 1 raw: {"raw": "quote " and comma, then newline
end"}
```

- **JSON, value position** (`"s"`): quoted, with `\"` and `\n`
  escapes. **JSON, inside a string** (`"in"`): the same escaping with no
  quotes of its own, so the surrounding literal stays one string.
- **CSV**: a field containing a comma, a quote, or a newline is wrapped
  in quotes and its quotes are doubled, the RFC 4180 rule; the other
  fields are bare. Formats apply first, so `${cycle | 04}` pads.
- **Text**: nothing is escaped, so the newline is a real newline in the
  output.
- **Raw in JSON** (`"raw"`): `!` copies the text with no escaping, and
  the result is not valid JSON. Raw holes are for text that is already
  in the target encoding. The compile-time check of section 15 cannot
  see this, because the hole's value is only known at run time; the
  author is responsible for what a raw hole carries.

A `None` value in JSON value position renders as `null`; in text and
CSV it renders as empty text.

## 13. Empty projections and separators

A projection over zero tuples renders nothing, and the object or array
around it stays valid: the separator is written only between
repetitions, never before the first or after the last.

```polydat
input cycle: u64
tile doc : json := {"before": 1, "none": [@for k in 1..1 { ${k} }], "after": 2, "ys": [@for k in 0..3 sep " | " { ${k + cycle} }]}
tile line := "@for k in 0..3 {${k}}|@for k in 0..3 sep \", \" {${k}}"
tile cells : csv := "@for k in 0..3 {${k + cycle}}"
```

```text
cycle 1 doc: {"before": 1, "none": [], "after": 2, "ys": [1 | 2 | 3]}
cycle 1 line: 012|0, 1, 2
cycle 1 cells: 1,2,3
```

`sep` is any text. The defaults follow the encoding: `,` for `json` and
`csv`, nothing for `text`, so the first `@for` in `line` runs its
digits together and the second separates them. In the structural form
(section 9) a projection that is an object member carries its own
comma inside each repetition, so a member projection that renders zero
tuples also leaves the object valid.

## 14. A tile inside a module

A tile may be declared inside a module body. It inlines with the call
like any binding: the tile is named with the module's prefix, its
holes and branch conditions read the caller's arguments, and its
projections' generator expressions are rewritten the same way. Two
calls give two tiles.

```polydat
input cycle: u64
label(n: u64, tag: str) -> (out: str) := {
    tile t : json := {"n": ${n}, "twice": ${n * 2}, "tag": ${tag}}
    out := t
}
a := label(n: cycle, tag: "first")
b := label(n: cycle + 100, tag: "second")
```

```text
cycle 2 a: {"n": 2, "twice": 4, "tag": "first"}
cycle 2 b: {"n": 102, "twice": 204, "tag": "second"}
```

A module's parameter types are the port-type keywords, and `str` and
`String` name the same type. A tile inside a `for` traversal body works
the same way: it compiles inside the body's program, its holes see the
elements and the cascaded outer wires, and `--emit tile:<name>` can
select it. The toy test definition in
[`docs/tutorials/toy_test_definition.md`](toy_test_definition.md) renders a
document per reading that way.

## 15. What the compiler refuses

Every rule above has an error with the tile's name in it. These are
the messages for the mistakes that come up most:

```text
a declared type the catalog cannot reach:
  tile 't': hole `s: u64`: no conversion from the wire type str to the declared type u64; write the conversion explicitly in the expression
a predicate over a per-cycle outer wire:
  tile 't': projection `for k in 1..9 where {k} < {limit}`: predicate placeholder `{limit}` names a wire outside the comprehension; a projection's predicate sees only its elements
a continuous source without a sampling order:
  tile 't': projection `for x in 0.0..1.0` ranges over a continuous source, which has no finite tuple set; add `order <strategy>/<count>` (halton, sobol, lhs, or shuffle) to sample that many points
strict mode and an implicit adapter:
  tile 't': hole `cycle: f64`: strict mode rejects the implicit u64 -> f64 adapter the declaration needs; write the conversion explicitly in the expression
a json body that is not valid JSON once the holes are typed:
  tile 't': the json body is not valid JSON once every hole is a placeholder: expected `,` or `}` at line 1 column 9; with holes as `0`, one repetition per projection, and each branch's first arm, the skeleton reads: {"n": 0 "x": 1}
```

The programs that produced them, in order:

```polydat
s := __u64_to_string(cycle)
tile t : json := {"n": ${s: u64}}              -- write ${s as u64}

limit := cycle
tile t : text := "@for k in 1..9 where {k} < {limit} {${k}}"

tile t : text := "@for x in 0.0..1.0 {${x}}"   -- add order halton/8

tile t : json (strict) := {"f": ${cycle: f64}} -- write ${to_f64(cycle)}

tile t : json := {"n": ${cycle} "x": 1}        -- a missing comma
```

The last one is the compile-time check every `json` tile passes: the
skeleton with every hole replaced by `0`, each projection body written
once, and each branch showing its first arm must parse as JSON. `0` is
a value in value position and text inside a string or a key, so every
well-placed hole passes, and a missing comma, an unquoted key, or a
hole where no value can go is caught before the first render.

## 16. One tile, every engine

Polydat runs a program on one of three engine levels: the interpreter
(P1), closures over a flat slot buffer (P2), and native code (P3), the
default, either as native segments in a compiled kernel or as fused
cones inside the interpreter's graph. Strings, JSON values, and rendered documents ride through the
compiled levels as handles into a per-cycle arena and a value table,
and a tile renders there by the same code path it renders on the
interpreter. The result is bit-identical on every level:

```polydat
input cycle: u64
user_id := mod(hash(cycle), 1000000)
name    := "user-{user_id}"
tile doc : json := {"id": ${user_id}, "name": ${name}, "label": "id-${user_id}", "tags": [@for k in 0..2 { ${k + user_id} }]}
```

```text
cycle 0 P1:    {"id": 607535, "name": "user-607535", "label": "id-607535", "tags": [607535,607536]}
cycle 0 cones: identical
cycle 0 P2:    identical
cycle 0 P3:    identical
cycle 1 P1:    {"id": 822465, "name": "user-822465", "label": "id-822465", "tags": [822465,822466]}
cycle 1 cones: identical
cycle 1 P2:    identical
cycle 1 P3:    identical
```

The example compiles the program four ways: the interpreter, the
interpreter with cone extraction forced on, the P2 closure kernel, and
P3, the default `compile_polydat_kernel` builds, and reads the tile
from each. Each engine chooses its own mix of native and closure work;
nothing in a program selects an engine, and nothing about a tile
changes its meaning when the engine changes. The differential suite in `tests/handle_tiers.rs`
holds that line over random programs of string, JSON, and tile nodes,
including every corner case in this tutorial.

One rule belongs to hosts that drive a whole compiled kernel directly:
read a kernel's handle outputs, through `get_value`, before running
another root kernel on the same thread, because each root kernel's
cycle advance resets the thread's arena. The interpreter never hands
out a handle, so nothing about this applies to `pull`.

## What a tile compiles to

Knowing the lowering makes the rules above predictable:

1. Each hole's expression becomes a binding of the enclosing scope,
   compiled like any other; a hole that names a wire uses the wire
   itself. The skeleton records, per hole, the encoding, the hole's
   position (value or inside a string), the declared type, the format,
   and the raw flag.
2. Each branch condition is a hole whose value is read for its truth.
3. Each projection becomes a child program: an input carrying the
   tuple index, one `extern` per element, one `extern` per outer wire
   the body reads (`cycle` included), and one binding per hole in the
   body. The comprehension is embedded in the skeleton.
4. The tile itself is one `tile_render` node whose constant is the
   skeleton and whose inputs are the hole values. It copies static
   runs, encodes each value at its hole straight into the document,
   selects branch arms, and drives child programs over the
   comprehension stream.

Because the hole expressions are ordinary wires, `explain` and `viz`
show them, dead holes are pruned with the rest of the graph, and a
hole shared by two tiles is computed once. On the compiled levels the
render node has the same body behind a native helper or a closure,
reading its hole values from their slots, which is why section 16
holds.

## Limits at this level

- A predicate sees only the comprehension's elements; a `{name}` for a
  wire outside it is a compile error.
- A continuous source needs an order with a sampling strategy and a
  count, such as `order halton/16`; without one it has no finite tuple
  set and is rejected.
- A splice belongs in value position; inside a string literal, bind
  the tile first and use the binding.
- A raw hole is the author's responsibility; the compile-time JSON
  check cannot see through it.
- Binary encodings, parsing rendered output back into values,
  unbounded projections, and user-defined template functions are out
  of scope for this revision.
