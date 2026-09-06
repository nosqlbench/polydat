# Polytile — Compiled Variate Templates

**Status:** Proposed SRD 114, third revision. Every step of §12 is
implemented and tested: the grammar in every
body form with options, with `:=` binding the tile's wire as in every
other binding; the structural JSON form; tiles handed in by a host as
template text, JSON text, a parsed JSON value, or a `polytile` binding;
compile-time typing of every hole per §4 with adapter insertion, strict
mode, and `explain tiles`; rendering at P1 for the `text`, `json`, and
`csv` encodings with position-aware encoding, formats, raw holes,
branches, splicing, and projections over inline comprehensions, bound
producers, derivations, generator calls, sampled continuous sources,
and nested bodies; the binary's `--emit tile:<name>`, `--tile-delims`,
and `--tile-sigil`; tiles inside module bodies; and the toy test
definition rendering its readings as documents.

Step 7, the P2 and P3 renderers, is done through SRD 115. A tile
renders natively: `tile_encode` and `tile_render` lower by the types
of their wires, join the fused cones of the production kernel and the
pure-P3 kernels, and produce the same bytes P1 does
(`tests/variadic_lowering.rs`). A projection re-runs its body program
per tuple through nested kernels inside the render helper, as it does
at P1, and at P2 the whole renderer runs as a closure. The tier
differential in `tests/handle_tiers.rs` pins random tiles, projections
included, to the interpreter across the tiers.

The walk-through with real output is
[the Polytile tutorial](../polytile_tutorial.md).

**Revisions:** the first draft set the textual grammar, the skeleton,
and the engine plan; the second added containment inside other
templating grammars, the structural JSON form, and type awareness; the
third made `:=` mandatory in every tile form, added `instring`, nested
and continuous projections, typed transport into projection bodies, the
host surfaces, and recorded the P2/P3 prerequisite.

**Ownership:** Polydat owns the tile grammar in both its textual and
structural forms, the skeleton IR, the type model for holes, the
renderers at every engine level, and the encodings named here. Hosts own
what they do with a rendered tile and which containment form they hand
Polydat.

**Companion documents:**
[The `for` Construct](for_traversal.md) (producers, traversal, activation),
[Language Spec](language_spec.md) (string interpolation, expressions),
[Type System](type_system.md) and [Type-System Alignment](type_system_alignment.md)
(`Str`, `Bytes`, `Json`, adapters),
[Engines](engines.md) and [JIT Boundary](jit_boundary.md) (P1, P2, P3),
nmbrs SRD 111 (cycle arenas and handle-encoded non-scalar slots).

## 1. The claim

Polydat produces variates. Most of what consumes them wants an encoding:
a CQL statement, a JSON document, a CSV row, a protobuf message. Today
that encoding is assembled from pieces. Flat text goes through string
interpolation, which is `printf` over segments. JSON goes through
`json_object`, `json_array`, and `to_json`, which build a `serde_json`
value per cycle and serialize it. A document with a deep static skeleton
and a few dynamic leaves is rebuilt and re-serialized in full on every
cycle, and every nested object costs an allocation.

Polytile is a template language for that last step. A tile is a skeleton
of static bytes with typed holes bound to wires. The skeleton is fixed at
compile time, including every nested arm that contains no hole, so
rendering copies static byte ranges and encodes hole values, nothing
more. A tile is a wire like any other, so it composes with the rest of
the graph, participates in lifecycle classification, and lowers to native
code through the same handle-encoded arena machinery that carries strings
through P3. Projections over comprehensions are part of the grammar, so a
tile can repeat a sub-skeleton over a producer or an inline
comprehension.

A tile has to live where templates actually live: inside a YAML workload,
inside a JSON body in a config file, inside a statement string that
another tool also templates, or as a data structure rather than text at
all. So the grammar has two front ends, textual and structural, and every
form is chosen so it can be carried as an ordinary string or an ordinary
document by whatever contains it.

Three limits shape the design, the same three that shape the kernel:

- **Structure lives in the template, not in the data.** However deep a
  document is, its static arms are serialized once at compile time and
  copied by range at render time. Adding a static field costs bytes, not
  work.
- **Rendering cost is the cost of the output.** Bytes copied plus holes
  encoded plus projection tuples times body cost. No intermediate tree.
- **A tile is a pure function of its coordinate.** Same inputs, same
  bytes, on every engine and every host.

## 2. The textual form

### 2.1 Statement grammar

Extends [Grammar](grammar.md) §2.

```ebnf
statement   ::= ...existing...
             |  tile_def

tile_def    ::= "tile" ident (":" encoding)? options? ":=" tile_body
encoding    ::= "json" | "text" | "csv"                (* extensible; default text *)
options     ::= "(" option ("," option)* ")"
option      ::= "delims" string string                  (* hole delimiters *)
             |  "sigil" string                          (* directive prefix *)
             |  "strict"
             |  "instring"

tile_body   ::= json_block                              (* json: balanced { } or [ ] *)
             |  heredoc                                 (* <<< ... >>> *)
             |  string_literal                          (* one-line tiles *)
```

A tile statement has the shape of every other wire binding in the
grammar, `modifier name : type := value`: `tile` is the modifier, the
encoding is the type of the document that flows on the wire (its port
type is `Str`), and `:=` binds the wire, so `${doc!}` in another tile
and `f(doc)` in a binding read it as they read any wire. The encoding
defaults to `text`, so `tile greeting := "..."` is the whole statement
for a one-line text tile.

The lexer captures a tile body raw after `:=`, as it does the text
after `for`. A `json` body is a brace- or bracket-balanced block,
string-aware, so the template is written in JSON's own syntax. A heredoc
body is everything between `<<<` and `>>>` and suits any encoding. A
string-literal body is an ordinary Polydat string and suits short tiles.

### 2.2 Template grammar

Inside a body:

```ebnf
hole        ::= open expr (":" type)? ("|" format)? ("!")? close
projection  ::= sigil "for" for_source ("sep" string)? "{" body "}"
branch      ::= sigil "if" expr "{" body "}" (sigil "else" "{" body "}")?
splice      ::= open tile_name close
escape      ::= open open                               (* a literal open delimiter *)
```

`open` and `close` default to `${` and `}`; `sigil` defaults to `@`.
Both are overridable per tile (§2.3) and per host (§5).

- A **hole** is a Polydat expression in the enclosing scope. `:type` is a
  declared type (§4). `|format` is a printf format spec applied before
  encoding. `!` marks the hole raw: its text is copied without the
  encoding's escaping. Polydat's own `{name}` interpolation stays
  available inside string literals within a hole expression.
- A **projection** repeats its body once per tuple of a comprehension.
  `for_source` is the same surface as the `for` construct: inline
  comprehension text or a bound producer, including derivations. Element
  names are wires inside the body. `sep` overrides the encoding's
  default separator between repetitions.
- A **branch** renders one of two bodies by a `u64` condition. Both
  bodies must satisfy the encoding's structural rules.
- A **splice** is a hole whose expression is the bare name of another
  tile in scope. It is resolved at compile time by inlining that tile's
  skeleton.
- A doubled open delimiter is a literal open delimiter.
- The braces of a directive block delimit it; whitespace padding them
  is not body. `@if x { "hot" }` renders `"hot"`. Braces inside body
  text are balanced, so JSON objects sit in a block unescaped.
- The `instring` option declares that the body begins inside a JSON
  string literal, so holes encode as escaped text from the first byte.
  The compiler sets it on the tile it makes for a projection nested in
  a string position; authors rarely need it.
- A block body (`{ ... }` or `[ ... ]` after the header) keeps the
  author's layout but not the indentation of the statement around it:
  the common leading whitespace of the lines after the first is
  removed. A tile inside a `for` body therefore renders the same bytes
  as the same tile at top level. Heredoc and string bodies are exact.
- A hole cannot appear in a directive header. Polydat's `{name}`
  interpolation may, as in `where {k} > 0`; a free-standing `{word}`
  after the header is the block itself (`@if x {plain}`), and `{name}`
  is read as interpolation only when it is attached to header
  punctuation (`1..{n}`) or followed by an operator.

Examples:

```text
tile reading : json := {
    "meta": { "schema": 3, "source": "polydat", "units": { "temp": "C", "rh": "%" } },
    "tenant": ${tenant_id},
    "device": "${device_id}",
    "ts": ${ts: u64},
    "samples": [ @for s in 0..4 { { "n": ${s}, "temp": ${temp_c + s | .2} } } ],
    "status": "${status}"
}

tile load := <<<
INSERT INTO ${keyspace}.${table} (tenant_id, device_id, ts, doc)
VALUES (${tenant_id}, '${device_id}', ${ts}, '${reading!}')
>>>

tile row : csv := "${tenant_id},${device_id},${ts},${status}"
```

`tile` is a hard keyword. Directives and delimiters have no meaning
outside a tile body.

### 2.3 Containment: living inside another grammar

A tile is often carried by something that has its own template syntax or
its own idea of what braces, dollars, and at-signs mean: a YAML workload
whose values are strings, a JSON config, a shell here-doc, a CQL string
that another tool also expands, a Jinja or Handlebars page. Four rules
make a tile portable into those places.

1. **Delimiters are declarable.** `tile t (delims "<%" "%>")`
   uses `<%expr%>` for holes; `(sigil "#")` uses `#for` and `#if`. A
   host may also set defaults for every tile it passes in (§5). The
   canonical defaults are `${`, `}`, and `@`, chosen because they are
   inert in JSON, CQL, SQL, YAML double-quoted strings, and Markdown, and
   because `{name}` interpolation inside Polydat strings is untouched.
2. **Every body form is carryable as a string.** The heredoc and
   string-literal bodies contain no construct that a YAML or JSON string
   cannot hold. Where the carrier's own escaping interferes, the doubled
   open delimiter (`${${`) is the only escape a tile needs, and a host
   can choose delimiters the carrier never uses.
3. **The tile keyword is optional at the host boundary.** A host that
   holds only a string can hand it to Polydat as a tile without wrapping
   it in a statement: the `polytile(encoding, text)` node and the
   `compile_tile` API (§5) accept bare template text, so a YAML value
   `body: '{"tenant": ${tenant_id}}'` becomes a tile with no grammar the
   YAML author has to learn beyond the hole syntax.
4. **Nested carriers compose by encoding, not by text.** A tile that is
   itself the value of a hole in another tile is spliced (compile time)
   or rendered and copied (raw hole), never re-parsed. A tile carried by
   another template engine is opaque text to that engine as long as the
   two do not share delimiters, which rule 1 guarantees the author can
   arrange.

A tile body never needs a construct outside its own hole and directive
syntax, so no carrier ever has to understand Polydat to carry one.

## 3. The structural form

A template can be a document rather than text: a JSON value (or a YAML
value, which is the same thing once loaded) in which strings define
insertion points. This is the natural form for hosts whose configuration
is already structured, and it is how a workload file can carry a
document template as data.

### 3.1 Insertion points

In a structural template, a string is examined for holes:

| String value | Meaning |
| --- | --- |
| exactly one hole, `"${expr}"` | a **value hole**: the string node is replaced by the hole's value, encoded by type. `"${ts}"` renders as `1700000000000`, a bare number. |
| text with one or more holes, `"row-${row}"` | a **string hole**: renders as a string with the holes encoded as text inside it. |
| exactly one hole with a `str` declaration, `"${ts: str}"` | a string hole whose value is the number's text; the declaration forces string position. |
| a directive string, `"@for s in 0..4"` or `"@if cond"` | a **structural directive**; see §3.2. |
| no hole | static content, folded into the skeleton. |

The distinction between a value hole and a string hole is the one that
lets a structural template express `"ts": 1700000000000` and
`"device": "d9ac..."` from the same string syntax: the wire's type
decides, and a declaration overrides.

### 3.2 Directives in structure

Projections and branches are expressed with the carrier's own array and
object shapes:

```json
{
  "meta": { "schema": 3, "units": { "temp": "C" } },
  "tenant": "${tenant_id}",
  "samples": [ "@for s in 0..4", { "n": "${s}", "temp": "${temp_c + s | .2}" } ],
  "audit": [ "@if verbose", { "by": "${operator}" } ],
  "tags": { "@for t in tags": { "${t}": true } }
}
```

- An **array** whose first element is a `@for` string is a projection:
  the remaining elements form the body, rendered per tuple and separated
  as array items. With one body element the projection yields that
  element per tuple; with several, each tuple contributes all of them in
  order.
- An **array** whose first element is an `@if` string is a branch: the
  remaining elements render when the condition holds; a following
  `"@else"` string separates the alternative.
- An **object** with a single `@for` key is a member projection: the
  value is an object template rendered per tuple, and its members are
  merged into the enclosing object. A key that is itself a hole is
  rendered as the member name.

Arrays and objects without a leading directive are static structure with
holes inside.

### 3.3 One skeleton

The structural front end produces the same skeleton IR as the textual
front end (§6). Static arms of the document, however deep, fold into
single byte ranges exactly as in the textual form; the only difference
is that the structural form is validated by construction rather than by
parsing the template with placeholders.

A host may hand Polydat a `serde_json::Value` directly through the
`compile_tile_value` API, or embed the document in a Polydat file as a
`json` tile body, which is the textual form of the same thing. The two
forms are interconvertible: a structural template pretty-prints as a
valid textual `json` tile, and a textual `json` tile parses to the same
structure.

## 4. Type awareness

Every hole has a type, known at compile time, and the encoder for a hole
is chosen by that type. The type comes from three sources in priority
order.

### 4.1 Declared type

`${expr: u64}` declares the hole's type. The compiler inserts the same
adapter it would insert for a wire of the expression's type feeding a
port of the declared type: lossless widening is automatic, narrowing and
string-to-number conversions must be written explicitly, and an
impossible conversion is a compile error at the hole's position. The
declaration is the author's statement of what the encoding should see,
and it wins over everything else.

Type keywords are the port-type keywords of the type system: `u64`,
`i64`, `f64`, `str`, `bool`, `json`, `bytes`, and the rest.

### 4.2 Wire type

Without a declaration, the hole takes the compile-time type of its
expression, resolved by the same inference the compiler applies to any
binding: literals, declared inputs and externs, node return types, and
`for` elements all carry types. This covers most holes, and it is what
lets `"${ts}"` in a structural template become a number.

### 4.3 Contextual type

Where an encoding assigns a meaning to a position, that position carries
an expected type, and the hole is checked against it:

| Encoding | Position | Expected | Rule |
| --- | --- | --- | --- |
| `json` | inside a string literal | text | any type renders as text, escaped |
| `json` | value position | any JSON value | the wire's type picks the JSON form; `Str` is quoted, numbers bare, `Bool` bare, `Json` serialized, `None` is `null` |
| `json` | object key | text | any type renders as text, escaped |
| `csv` | field | text | any type renders as text, quoted when needed |
| `text` | anywhere | text | display form |

Contextual expectations never silently change a value's meaning. A `Str`
wire at a JSON value position stays a string; if the author wants a
number there, the declaration `: u64` says so and the conversion is
explicit. A `Json` wire inside a string literal is serialized and
escaped, not spliced.

### 4.4 Diagnostics and strict mode

A hole whose declared type cannot be reached from its wire type, or whose
wire type is unknown, is a compile error naming the tile, the hole's
position in the template, the expression, and both types. Under
`(strict)` or the compiler's strict mode, implicit adapters at holes are
rejected exactly as implicit adapters on wires are, so every conversion
is written down.

`explain tiles` prints each hole with its expression, its wire type, its
declared type if any, its contextual expectation, and the encoder that
was chosen, so the typing of a document is inspectable before it runs.

## 5. Semantics

### 5.1 A tile is a wire

A tile binds a wire named by its definition. Its port type is `Str` for
text encodings and `Bytes` for binary ones. Its value is the byte
sequence obtained by substituting each hole's encoded text into the
template, in order. Its lifecycle follows its holes: a tile whose holes
are all const is const, and any dynamic hole makes it dynamic. A tile
with no holes is a constant and folds like one.

A tile depends exactly on the wires its holes and projections reference,
so provenance and invalidation treat it like any other node. Pulling a
tile evaluates only the holes that changed.

### 5.2 Encodings

An encoding defines how a body is captured, how each hole is encoded by
type, what structural rules the skeleton must satisfy, and the default
separator for projections.

| Encoding | Capture | Structural rules | Separator |
| --- | --- | --- | --- |
| `json` | balanced block, heredoc, string, or structural document | the skeleton with holes at value positions is valid JSON; `@for` in an array repeats items; `@for` in an object repeats members | `,` |
| `text` | heredoc or string | none | none |
| `csv` | heredoc or string | one record per render; fields separated by the delimiter | the delimiter |

A `json` tile is checked at compile time: the template with every hole
replaced by a placeholder of its type must parse as JSON, holes must sit
at value positions, inside string literals, or in key position, and a
projection's body must be a complete value or member. A tile that fails
this is a compile error carrying the position inside the template.

Raw holes bypass the encoding's escaping. They exist for splicing
pre-encoded content, such as one tile's rendered bytes into another at
render time when compile-time splicing is not possible, and they are the
author's responsibility.

### 5.3 Holes

A hole's expression is compiled as a binding in the enclosing scope, so
it sees every wire the scope sees, including `for` elements and cascaded
outer wires when the tile is declared inside a traversal body. The
expression's type decides the encoder per §4. A format spec applies
first, then the encoder, then the raw flag decides whether escaping
applies.

### 5.4 Projections

A projection's comprehension must have bounded cardinality; an unbounded
source is a compile error. The body's holes may reference the
comprehension's element names and any wire of the enclosing scope. The
body renders once per tuple with the elements bound, separated by the
encoding's default separator or `sep`. Element names are typed from the
comprehension's sources by the same table as [The `for`
Construct](for_traversal.md) §3.3.

A body may contain further projections; each nests as one compiled
program per lexical position, exactly as the `for` construct nests
bodies, and reads outer elements and scope wires alike. A generator-call
source is an expression over the enclosing scope: it compiles as a wire
there, so it may read any wire in scope, and its value is the clause's
element list, one tuple for a scalar and one per item for a list. A
filter predicate sees the comprehension's elements; a `{name}` that
names a wire outside it is a compile error. A continuous source has no
finite tuple set of its own; it projects when its order names a
sampling strategy (`halton`, `sobol`, `lhs`, `shuffle`) with a count,
which samples that many points from its intervals.

A projection over a bound producer dispenses the producer's stream at
render time. Two renders of the same tile never share dispense state.

### 5.5 Splicing

`${name}` where `name` is a tile in scope splices that tile's skeleton
into this one at compile time. Its holes join this tile's holes; its
static bytes join this tile's static bytes; adjacent statics coalesce.
Splicing is transitive and must be acyclic. Only a tile of the same
encoding is spliced; across encodings the named tile is an ordinary
wire whose rendered text enters through the hole and is encoded by the
outer tile's rules, or inlined with `!`. A `json` document carried in a
`text` statement is `'${doc!}'`; a `text` message carried in a `json`
document is `"body": ${msg}` and arrives quoted and escaped.

### 5.6 Host APIs

Beyond the statement form, hosts build tiles from what they hold:

```text
tile_from_text(name, encoding, text, options)   textual body, bare
tile_from_json_text(name, json, options)        structural body as text
tile_from_json_value(name, value, options)      structural body, already parsed
compile_polydat_with_tiles(source, tiles)       compile them with a program

name := polytile(encoding, body, options...)    in source; body is a string or heredoc
name := polytile_json(body, options...)         in source; structural JSON
```

The Rust functions live in `polydat::tile` and each returns the
`TileDef` the `tile` keyword produces. `polytile` and `polytile_json`
are binding forms the parser rewrites into `tile` statements, so a host
that only has strings, such as a YAML workload runner, lowers
`body: '{"tenant": ${tenant_id}}'` to `doc := polytile("json", "...")`
as a program transform and never touches a runtime decorator. The body
is taken raw, never evaluated. Options `open`, `close`, `sigil`, and
`strict` are named arguments, and a host may set process defaults for
all of them.

## 6. Compilation

A tile compiles to a **skeleton**: a straight-line program over a small
instruction set.

```text
Copy    { static: handle, range }           copy bytes from the static interner
Hole    { wire, encoder, format }           encode a wire's value
Repeat  { stream, body: skeleton, sep }     render body per tuple
Branch  { cond, then: skeleton, else }      render one body
```

Compilation proceeds in six passes:

1. **Parse** the body into segments, holes, projections, and branches,
   with template positions for diagnostics. The textual front end
   tokenizes by delimiter; the structural front end walks the document
   and classifies strings per §3.1.
2. **Splice** referenced tiles, checking for cycles.
3. **Type** every hole per §4, inserting adapters where allowed and
   reporting mismatches.
4. **Fold statics.** Every maximal run of bytes containing no hole,
   including whole nested objects and arrays, becomes one `Copy` of an
   interned byte range. This is the pass that gives static arms their
   O(1) render cost regardless of depth.
5. **Validate** against the encoding's structural rules.
6. **Lower.** Each hole expression compiles to an anonymous binding. The
   tile itself compiles to a `tile_render` node whose wire inputs are
   the hole bindings in skeleton order and whose constant is the
   skeleton. A projection body compiles to a child program keyed by its
   lexical position, exactly as a `for` body does, with one
   `IterationExtern` per element; the `Repeat` instruction carries its
   identity. One program per position, however many tuples flow.

Tiles declared inside a `for` body compile inside that body's program.

## 7. Runtime

### 7.1 The render program

A tile renders by walking its skeleton once. `Copy` appends an interned
static run; `Hole` appends the encoded text the hole's `tile_encode`
binding produced; `Branch` selects a body on a `1`/`0` hole; `Repeat`
evaluates its comprehension with the evaluator the `for` runtime opens a
traversal with, binds each tuple into a per-thread scratch state over
the body program, and renders the body per tuple with the separator
between. The comprehension and the static runs are parsed and interned
once, when the render node is constructed; scratch states are reused
across renders; two renders never share dispense state. The same walk
runs on every tier.

### 7.2 Tiers

The tiers differ only in how the hole texts arrive and where the result
goes, per [Compiled Non-Scalar Slots](compiled_handles.md):

- **P1.** `tile_encode` runs per hole and `tile_render` walks the
  skeleton, both as ordinary nodes on `Value`s. The document is built in
  a `String` and surfaced as a `Str`; writing straight into the cycle
  arena is a refinement recorded in SRD 115.
- **P2.** Both nodes run as `compiled_handle` closures over handle
  slots: hole texts are arena handles decoded by the wire types the
  kernel supplies, and the rendered document enters the arena.
- **P3.** In a fused cone or a pure-P3 kernel, `tile_encode` lowers to a
  helper over the hole's wire type and the interned encoding, and
  `tile_render` to a helper over the interned tile program and the
  encoded hole texts, passed in a stack array with their type codes.
  Projections activate the body program through nested kernels inside
  the helper, exactly as at P1; the cone eval is re-entrant so a body's
  own cones run inside it. The rendered handle flows on as a slot, so a
  tile feeds an adapter or `emit_row` without leaving native code. One
  rule keeps semantics exact: `tile_encode` tolerates a `None` input (it
  writes `null`), so fed straight by a kernel input it stays on P1 and
  the render node takes its text as a boundary input (SRD 115 §9).

### 7.3 Cost

Rendering a tile costs the bytes it copies, the holes it encodes, and,
for each projection, the tuple count times its body's cost. Skeleton
depth does not appear in that sum. A one-hole document with a
thousand-byte static arm renders in one copy and one encode.

## 8. Axioms

- **L1, Purity.** A tile's bytes are a pure function of its hole wires.
  Same coordinate, same bytes, on every engine and every host. Follows
  from D1 in the runtime model applied to the tile node.
- **L2, Static invariance.** The skeleton and every interned static range
  are fixed at compile time. No render re-serializes structure.
- **L3, Cost.** Render cost is linear in output bytes plus hole count plus
  projection tuples, independent of skeleton depth.
- **L4, Encoding soundness.** A `json` tile whose holes carry values of
  their compile-time types renders valid JSON. A `csv` tile renders a
  valid record. Guaranteed by compile-time validation plus per-type
  encoders.
- **L5, Form equivalence.** A textual `json` tile and the structural
  template it parses to compile to the same skeleton and render the same
  bytes.
- **L6, Type determinism.** Every hole's encoder is fixed at compile
  time from its declared, wire, or contextual type. No render inspects a
  value's runtime variant to choose an encoding.

## 9. Boundaries

Not in this revision:

- binary encodings such as protobuf and Avro, which need length-prefix
  and back-patch instructions in the skeleton; the instruction set is
  designed to grow those without changing the grammar;
- parsing rendered output back into values;
- unbounded projections;
- recursion or user-defined template functions; splicing is the only
  composition, and it is static;
- YAML as an output encoding; YAML is supported as a carrier of
  structural templates, which is the same as JSON once loaded.

A tile is not a general templating language. Its expressions are
Polydat expressions, its loops are comprehensions, and its structure is
fixed.

## 10. The `polydat` binary

`--emit` gains a tile form: `--emit tile:<name>` emits the named tile
per cycle instead of a formatted row. Under the transform rule, this
appends an `emit_row("text", "<name>", <name>)` binding, so nothing new
happens at runtime. `--tile-delims OPEN CLOSE` and `--tile-sigil S` set
the defaults for tiles the binary compiles, again as a transform: every
tile that declares no options of its own is re-read under them before
the program compiles, so the program that runs is the program `explain`
narrates. `explain` gains a `tiles` phase that prints each tile's
skeleton: static runs with their byte total, holes with their
expressions, types, and encoders per §4.4, and projections with their
body programs.

## 11. Worked example

The toy test definition's load statement as a tile, with a JSON document
per reading written in the structural form as it would sit in a
workload file, and its textual twin. The textual form is now the
definition itself
([`examples/toy_test_definition.polydat`](../../examples/toy_test_definition.polydat)),
with `kind` and `flagged` members added:

```json
{
  "meta": { "schema": 3, "source": "polydat", "units": { "temp": "C", "rh": "%" } },
  "tenant": "${tenant_id}",
  "device": "${device_id}",
  "ts": "${ts}",
  "reading": { "temp": "${temp_c | .2}", "rh": "${humidity | .1}", "status": "${status}" },
  "samples": [ "@for s in 0..4", { "n": "${s}", "temp": "${temp_c + s | .2}" } ]
}
```

```text
tile doc : json := {
    "meta": { "schema": 3, "source": "polydat", "units": { "temp": "C", "rh": "%" } },
    "tenant": ${tenant_id},
    "device": "${device_id}",
    "ts": ${ts},
    "reading": { "temp": ${temp_c | .2}, "rh": ${humidity | .1}, "status": "${status}" },
    "samples": [ @for s in 0..4 { { "n": ${s}, "temp": ${temp_c + s | .2} } } ]
}

tile load := <<<
INSERT INTO ${keyspace}.${table} (tenant_id, device_id, ts, doc)
VALUES (${tenant_id}, '${device_id}', ${ts}, '${doc!}')
>>>
```

In the structural form, `"${tenant_id}"` and `"${ts}"` are value holes
and render bare because their wires are `u64`; `"${device_id}"` and
`"${status}"` render quoted because their wires are `Str`; the `meta`
arm is one static range. `doc`'s skeleton is eleven instructions with a
`Repeat` for `samples`. `load` splices nothing at compile time because
`doc` is dynamic; the raw hole copies `doc`'s rendered range into
`load`'s arena range. Rendering both per cycle is two memcpy sequences,
a handful of integer and float encodes, and a four-tuple loop.

## 12. Implementation plan

All eight steps have landed; §13 is the landing record.

1. **Grammar.** `TokenKind::Tile`, the raw body token after `:=`,
   `Statement::Tile`, the template parser in `dsl::tile` with its
   `render_template` inverse, and the printer. `tests/tile_syntax.rs`,
   `tests/fuzz_tile_syntax.rs`.
2. **Structural front end.** `dsl::tile_structural` classifies strings
   per §3.1 and textualizes directives per §3.2 into the template
   grammar, so both forms share one parser. `tests/tile_structural.rs`.
3. **Typing.** Every hole typed in `dsl::tile_lower` per §4, adapters
   inserted, the effective type written into the encode spec, and the
   `TileHoleTyped` events `explain tiles` prints. `tests/tile_typing.rs`.
4. **Skeleton and P1 renderer.** `tile_encode` per hole and one
   `tile_render` per tile; the skeleton IR, encoders, and formats in
   `library::tile_render`; splicing per §5.5. `tests/tile_render.rs`.
5. **Projections.** One body program per lexical position, sources
   resolved through the `for` construct's own resolver, the comprehension
   evaluated with the `for` runtime's evaluator, generator sources bound
   as literal lists, nested and continuous projections, `instring`.
   `tests/tile_projections.rs`.
6. **Host surfaces.** `polytile` and `polytile_json` bindings, the
   `polydat::tile` functions, `apply_tile_defaults`, the binary's
   `--emit tile:<name>`, `--tile-delims`, and `--tile-sigil`, tiles in
   module bodies, and the `TileCompiled` events.
   `tests/tile_host_surfaces.rs`.
7. **P2 and P3.** Through SRD 115: handle slots, the wire-typed
   lowerings of both tile nodes, the `compiled_handle` closures, and
   projections rendering inside the helper. `tests/variadic_lowering.rs`,
   `tests/handle_tiers.rs`.
8. **Docs.** [The Polytile tutorial](../polytile_tutorial.md) with real
   output, the illustrations page, and the toy test definition rendering
   its readings as documents.

## 13. Landing record

Decisions and defects worth knowing that the sections above state only
as rules. Dates are 2026-09-04 to 2026-09-06.

- **`:=` in every form.** The first drafts allowed `tile doc : json {`
  without `:=`. The third revision made `:=` mandatory because a tile is
  a wire binding like every other: `:=` assigns a wire that can be named
  symbolically and wired wherever a wire is accepted, which neither `:`
  nor `=` means.
- **Grammar defects found by the fuzzer.** Braces in static text ended a
  directive block early; hole delimiters beginning with `{` hid a block
  brace; a free-standing `{word}` after a header was read as
  interpolation; a hole in a directive header was reported as a range
  error; projection scratch states were keyed by a program address a
  later program could reuse. All fixed; the rules in §2.2 record the
  outcomes.
- **Structural commas.** A directive member beside static members carries
  its own separating comma inside each repetition, leading or trailing
  as its position requires, so a projection that renders zero tuples or a
  branch that renders nothing leaves the object valid without a
  cardinality check.
- **Projections.** Sources resolve through `resolve_source` and evaluate
  through `evaluate_for_iteration`, so order strategies, truncation, and
  predicates behave exactly as in `for`. Generator-call clauses compile
  to wires of the enclosing program and bind into the comprehension as
  literal lists, so a projection never needs a kernel of its own. The
  body's own input is the tuple index; the program's `cycle` cascades
  like any outer wire. Render-node inputs cross as typed values, not
  display text. Continuous sources project only under a sampling order
  (`halton`, `sobol`, `lhs`, `shuffle`) with a count; the comprehension
  runtime gained that sampling for tiles and `for` alike. A predicate
  placeholder naming a wire outside the comprehension is a compile
  error, as in `for`. Declared hole types reach every catalog adapter
  because `as` was widened to the whole catalog.
- **Modules.** A tile inside a module body inlines with the call under
  the module prefix; producers bound in a module are bound under the
  prefix as `streamer` constants; modules defined in the program itself
  register before compilation and shadow library nodes of the same name
  (Module System §2).
- **Block bodies** are dedented by the indentation of the statement
  around them, so a tile inside a `for` body renders the same bytes as
  at top level. Tile events from traversal bodies reach the parent's
  compile log, and `--emit tile:<name>` accepts a tile in a traversal
  body.
- **P2 and P3.** The typed transport into bodies, the skeleton counts,
  and one program per position were shaped so that the SRD 115 lowering
  undid nothing; it did not. The P3 renderer is the P1 walk behind a
  helper, not the straight-line skeleton code the first drafts sketched;
  SRD 115 §12 records what remains a refinement.
