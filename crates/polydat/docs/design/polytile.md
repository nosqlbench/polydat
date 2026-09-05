# Polytile — Compiled Variate Templates

**Status:** Proposed SRD 114, second revision. Steps 1 through 6 and 8
of §12 are implemented: the grammar in
every body form with options; the structural JSON form; tiles handed
in by a host as template text, JSON text, a parsed JSON value, or a
`polytile` binding; and tiles that compile to wires and render at P1
for the `text`, `json`, and `csv` encodings with type- and
position-aware encoding, formats, raw holes, branches, projections
over inline comprehensions and bound producers, and splicing; and
compile-time typing of every hole per §4 with adapter insertion, strict
mode, and `explain tiles`. P2/P3 renderers are not yet implemented;
§12 records what each step still owes. The walk-through with real output is
[the Polytile tutorial](../polytile_tutorial.md).

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

tile_def    ::= "tile" ident (":" encoding)? options? (":=")? tile_body
                                                       (* `:=` is required before a string body *)
encoding    ::= "json" | "text" | "csv"                (* extensible *)
options     ::= "(" option ("," option)* ")"
option      ::= "delims" string string                  (* hole delimiters *)
             |  "sigil" string                          (* directive prefix *)
             |  "strict"

tile_body   ::= json_block                              (* json: balanced { } or [ ] *)
             |  heredoc                                 (* <<< ... >>> *)
             |  string_literal                          (* one-line tiles *)
```

The lexer captures a tile body raw, as it does the text after `for`. A
`json` body is a brace- or bracket-balanced block, string-aware, so the
template is written in JSON's own syntax. A heredoc body is everything
between `<<<` and `>>>` and suits any encoding. A string-literal body is
an ordinary Polydat string and suits short tiles.

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
tile reading : json {
    "meta": { "schema": 3, "source": "polydat", "units": { "temp": "C", "rh": "%" } },
    "tenant": ${tenant_id},
    "device": "${device_id}",
    "ts": ${ts: u64},
    "samples": [ @for s in 0..4 { { "n": ${s}, "temp": ${temp_c + s | .2} } } ],
    "status": "${status}"
}

tile load : text <<<
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

1. **Delimiters are declarable.** `tile t : text (delims "<%" "%>")`
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

### 7.1 P1

The interpreter renders into the thread-local cycle arena from nmbrs SRD
111: `Copy` is a memcpy from the static interner, `Hole` runs the
encoder for the hole's type straight into the arena with no intermediate
`String`, `Repeat` activates the body program over a scratch state that
is reset rather than reallocated per tuple, and `Branch` selects. The
result is the arena range, surfaced as a `Str` or `Bytes` value. Nothing
is allocated on the heap per render once the arena is warm.

### 7.2 P2

The `tile_render` node's closure form is monomorphic over the skeleton:
a loop over instructions with the encoders specialized by type. Hole
values arrive as typed slots rather than `Value`.

### 7.3 P3

The skeleton lowers to Cranelift IR as straight-line code. `Copy` becomes
a call to the arena copy helper with a static handle. `Hole` becomes a
call to the typed encoder helper, `put_u64`, `put_f64`, `put_escaped`,
and so on, each taking the arena pointer and the slot. `Repeat` with a
bounded stream becomes a counted loop that binds element slots and calls
the body cone. `Branch` is a conditional jump. Hole expressions are
ordinary cones and inline where eligible. The rendered handle flows on as
a 64-bit slot, so a tile feeds an adapter or an `emit_row` without
leaving native code.

The helper ABI is the one SRD 111 defines for string nodes; Polytile adds
no new calling convention.

### 7.4 Cost

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
tile doc : json {
    "meta": { "schema": 3, "source": "polydat", "units": { "temp": "C", "rh": "%" } },
    "tenant": ${tenant_id},
    "device": "${device_id}",
    "ts": ${ts},
    "reading": { "temp": ${temp_c | .2}, "rh": ${humidity | .1}, "status": "${status}" },
    "samples": [ @for s in 0..4 { { "n": ${s}, "temp": ${temp_c + s | .2} } } ]
}

tile load : text <<<
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

1. **Grammar.** Done. `TokenKind::Tile` and a raw `TileBody` token
   captured by the lexer for block and heredoc bodies, with `:=`
   optional before them; `Statement::Tile(TileDef)` carrying the
   encoding, options, raw body, and parsed pieces; `dsl::tile` parsing
   holes with declared type, format, and raw flag, projections with
   separators, branches, splices as bare-name holes, the doubled-open
   escape, and balanced braces in static text, with a `render_template`
   inverse; the printer reproduces every body form. Tests in
   `tests/tile_syntax.rs` cover each form and each error; the fuzzer in
   `tests/fuzz_tile_syntax.rs` generates tiles over every encoding, six
   delimiter pairs, four sigils, and nested directives, checks parsed
   pieces against the emitted shape, printer and renderer fixed points,
   and the compiler's interim rejection, and mutates programs to prove
   no stage panics. The first sweep found two grammar defects, braces in
   static text ending a directive block early and hole delimiters that
   begin with `{` hiding a block brace, both fixed.
   A second sweep after step 4 found and fixed three more: a
   free-standing `{word}` block read as interpolation, a hole in a
   directive header reported as a range error instead of a missing
   block, and projection scratch states keyed by a program address that
   a later program could reuse.
2. **Structural front end.** Done. `dsl::tile_structural` classifies
   strings per §3.1 (value hole, string hole, `str`-declared string
   hole, static), lowers directive arrays and `@for`/`@if`/`@else`
   object keys per §3.2 by textualizing the value into the template
   grammar, and rejects directive strings in value position. The
   textual parser then produces the pieces, so the two forms are
   equivalent by construction; `tests/tile_structural.rs` renders the
   §3.2 example both ways and compares the documents. A directive
   member beside static members carries its own separating comma inside
   each repetition, leading when a member precedes it and trailing when
   members follow, so a projection that renders zero tuples or a branch
   that renders nothing leaves the object valid; the cardinality check
   §3.2 called for is unnecessary by construction.
3. **Typing.** Done. Every hole is typed in `dsl::tile_lower` before
   its encode binding is emitted: the wire type by the compiler's own
   expression inference (projection elements and cascaded outer wires
   from the body context), the declared type checked for reachability
   through `assembly::auto_adapter`, the adapter node inserted between
   the expression and `tile_encode` when the declaration needs one (in
   a projection body, as the `as` fusion), and the effective type
   written into the encode spec so the encoder never consults the
   runtime variant. Unreachable declarations, unknown type keywords,
   and wires with no text form (`Ext`, `Handle`, registers) are compile
   errors naming the tile, the hole, and both types; `(strict)` or
   compiler strict mode rejects the implicit adapter. Each hole records
   a `TileHoleTyped` compile event, which `polydat explain <file>
   tiles` prints. Tests in `tests/tile_typing.rs`, one per row of the
   §4.3 table and per error case.
4. **Skeleton and P1 renderer.** Done. `dsl::tile_lower` lowers each
   hole to a `tile_encode` node whose constant spec carries encoding,
   position, declared type, format, and raw flag; branch conditions to
   `1`/`0` holes; static runs to single copies; the tile to one
   `tile_render` node over the encoded holes, constant when it has no
   holes. `library::tile_render` holds the skeleton IR (`TileSpec`,
   `TileOp`), the encoders for `text`, `json` (value and in-string
   positions), and `csv`, printf formats, and the P1 renderer. Splicing
   inlines same-encoding tiles and treats others as wires (§5.5).
   Tests in `tests/tile_render.rs`; `tests/function_coverage.rs`
   exercises both nodes. The `Bytes` form and arena-backed output are
   still owed.
5. **Projections.** Done. A child program per body with element and
   cascaded outer externs, the comprehension embedded in the skeleton
   as a `StreamerValue`, per-thread scratch-state reuse, and separators
   with per-encoding defaults. Sources resolve through the `for`
   construct's own `resolve_source`, so inline text, bound producers,
   and derivations (`base where ... order ...`) all project. The
   renderer evaluates the comprehension with the same
   `evaluate_for_iteration` the `for` runtime opens a traversal with,
   over an empty parent kernel and the body program as canonical, so
   order strategies with truncation and element predicates behave
   identically. Generator-call clauses compile to wires of the
   enclosing program, typed by hole inference, and the render node
   binds their values into the comprehension as literal lists, so a
   projection never needs a kernel of its own. Nested projections lower
   to a `tile` statement inside the body program (one program per
   lexical position at every depth), with producers re-bound from
   their source text and outer wires cascaded through. Member
   projections come through the structural form. The bounded
   cardinality check rejects continuous sources, which the runtime does
   not sample into tuples, and accepts generator sources whose count is
   unknown. A predicate placeholder for a wire outside the
   comprehension is a compile error, as it is for `for`. The body's
   own input is the tuple index and the program's `cycle` is cascaded
   like any outer wire. Values cross into the render node as they are:
   its variadic inputs are exempt from wire typing, as `printf`'s are,
   so cascaded wires, generator scalars, and generator lists (streams,
   vectors, JSON arrays) arrive typed rather than as display text.
   A nested projection inside a JSON string position compiles its tile
   with the `instring` option, so its holes escape as text from the
   first byte. A continuous source projects when its order names a
   sampling strategy with a count: the comprehension runtime now
   samples `halton`, `sobol`, `lhs`, and `shuffle` points from the
   intervals, for tiles and for the `for` construct alike, and the tile
   compiler rejects a non-sampling strategy over a continuous source
   ahead of time. Inside a body every catalog adapter is available to a
   declared type, since `as` now reaches the whole catalog. Tests in
   `tests/tile_projections.rs`.
6. **Host surfaces.** Done. In source, `name := polytile(enc, body,
   options...)` and `name := polytile_json(body, options...)` are
   parsed into `tile` statements before compilation; the body is a
   string literal or a `<<< >>>` heredoc, which the lexer accepts as a
   string literal anywhere. In Rust, `polydat::tile` exposes
   `tile_from_text`, `tile_from_json_text`, `tile_from_json_value` (a
   `serde_json::Value`), and `compile_polydat_with_tiles`. Host
   defaults for delimiters and sigil are the transform
   `transform::apply_tile_defaults`: tiles that declare no options of
   their own are re-read under the host's, tiles that declare any keep
   all of theirs; the binary applies it from `--tile-delims OPEN CLOSE`
   and `--tile-sigil S` before compiling, so `explain` narrates the
   program as it compiled. `--emit tile:<name>` selects the tile and
   the `text` emit format, so the appended `emit_row` binding writes
   the rendered document per cycle and nothing new happens at runtime.
   Tiles inside module bodies inline with the call: the tile takes the
   module prefix, hole and branch expressions and generator expressions
   are rewritten against the caller's arguments, `{name}` placeholders
   in projection sources are renamed when they name a module input
   bound to a caller's wire or a module-internal binding, and
   projection elements shadow module names inside their bodies. A
   producer bound inside a module (`axes := for ...`) is bound under
   the module prefix as a `streamer` constant and its tiles project over
   it. Modules defined in the program itself are registered by name
   before compilation, so `compile_polydat` on a string resolves them
   without a source directory and an author's definition shadows a
   library node of the same name. Each
   tile records a `TileCompiled` event with its static runs and bytes,
   holes, branches, projections, and body programs, which `explain
   tiles` prints ahead of the hole typing. Tests in
   `tests/tile_host_surfaces.rs`, including the binary end to end.
7. **P2 and P3.** Not started, and blocked on a prerequisite outside
   this SRD. The P2 closure tier and the P3 cone tier both run over a
   flat scalar slot buffer: no `Str`-, `Json`-, or `Ext`-valued node
   has a compiled form today, cone classification admits scalar ports
   only, and the SRD 111 handle helpers (`jit_str_to_u64` and its
   siblings) read string handles into scalars rather than producing
   strings. A tile renderer at P2/P3 therefore needs first a non-scalar
   slot representation for the compiled tiers (arena handles in the
   slot buffer, a `Copy` over interned static ranges, encoders that
   write into the arena), which is the SRD 111 arena ABI landing in
   Polydat's compiled tiers rather than a tile-specific piece of work.
   Once that exists, the lowering is mechanical: `tile_encode` becomes
   a helper per encoder over a scalar or handle, `tile_render` a helper
   over a handle vector, and the skeleton's `Repeat` an activation of
   the body program as `for` bodies already are. Differential tests
   against P1 across the fuzz corpus remain the acceptance criterion.
   Everything that precedes this step (typed transport into bodies,
   skeleton counts, one program per position) was shaped so that this
   lowering does not have to undo anything.
8. **Docs.** Done. [The Polytile tutorial](../polytile_tutorial.md)
   walks every implemented form with output from
   `examples/polytile_tutorial.rs` and `examples/polytile_demo.polydat`;
   the illustrations page has a tile section; and the toy test
   definition renders each reading as a JSON document from a `tile`
   inside its traversal body and carries it in the load statement, as
   §11 sketched. Getting there fixed two things: block bodies are now
   dedented by the indentation of the statement around them, and tile
   events from traversal bodies reach the parent's compile log so
   `explain tiles` sees them. `--emit tile:<name>` accepts a tile that
   lives in a traversal body, and a bare file name resolves its
   same-directory modules. Tested end to end in
   `tests/tile_host_surfaces.rs`.

Each step lands with its tests and leaves the previous surfaces working.
