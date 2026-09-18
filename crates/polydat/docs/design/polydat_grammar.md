# The Polydat Grammar — Definitive Specification & Guide

**Subtitle:** The complete, verified reference for the Polydat surface
language.

> **Operator contract:** `&&` and `||` are eager truthiness combinators;
> they do not short-circuit. They occupy the lowest precedence band.
> The uniform `<expr> as <type>` cast occupies the tightest postfix
> position.

<a id="sec-authority"></a>
## 0. Authority

This document is the normative reference for the Polydat surface
language, its lexical grammar, statement and expression productions,
type-naming vocabulary, desugaring and projection behaviour, and
rejection rules, and every example in it is checked by the suite;
[`grammar.md`](grammar.md) is its formal appendix (type rules and the
G-axioms in full), [`graph_compiler.md`](graph_compiler.md) the
compilation companion (what the passes do to what this document
parses), [`comprehension_forms.md`](comprehension_forms.md)
the algebra behind the `for` construct (§[16](#sec-for)),
[`polytile.md`](polytile.md) the semantics behind tiles
(§[17](#sec-tiles)), and [`type_system.md`](type_system.md) the adapter
machinery behind the type keywords.

The language is its own crate, `polydat-grammar`: the lexer, the
parser, the AST and its projector, the free-name collector, pragmas,
the diagnostic types, the tile template parsers, the comprehension
sub-language with its algebra, and the port type vocabulary, with no
runtime behind them. `polydat` depends on it and re-exports every
module at the path it always had (`polydat::dsl::ast`,
`polydat::dsl::parser`, `polydat::iteration::comprehension::parse`,
`polydat::ast::PortType`), so nothing that compiles against `polydat`
changes; a tool that only reads, checks, or prints Polydat source links
the grammar crate alone. What a type means to a compiled buffer (its
slot color and width) is the runtime's reading of it, an extension
trait in `polydat::ast`, and what a comprehension evaluates to, what a
tile renders, and what a cursor sugar rewrites into are likewise the
runtime's.

<a id="sec-roundtrip"></a>
### 0.1 How the examples are verified — the round-trip contract

Polydat ships a projector,
[`polydat::dsl::pprint::pp_file`](../../../polydat-grammar/src/pprint.rs), that turns a
parsed AST back into canonical `.polydat` source. Throughout this spec,
**“the syntax the runtime gives back”** means the output of `pp_file`.

The projector's guarantee is **idempotence**, not textual fidelity:

```text
let p1 = pp_file(parse(src));
let p2 = pp_file(parse(p1));
assert_eq!(p1, p2);            // second-pass print == first-pass print
```

Projection is *canonicalizing*. It deliberately differs from the input
text in three ways, each of which you will see in the examples below:

1. **Every `BinOp` is fully parenthesized.** `y := x + 1` projects as
   `y := (x + 1)`. This is uniformly safe and makes precedence explicit.
2. **String interpolation is desugared at parse time.** `name := "{a}-{b}"`
   parses to a `printf` call and projects as
   `name := printf("{}-{}", a, b)` (see §[10](#sec-interpolation)).
3. **Integer-valued finite floats gain a trailing `.0`.** `60.0` stays
   `60.0`; `6.283185307179586` is preserved exactly.

Code blocks tagged <code>```polydat</code> in this document are parsed
and checked for idempotent round-trip. Blocks additionally tagged
<code>```polydat compile</code> are also compiled (and must succeed).
Blocks tagged <code>```text</code> are illustrative only — EBNF,
token tables, rejected forms, and programs that need wires or modules
this file does not define — and are **not** parsed.

A second document,
[`polydat_grammar_programmatic.md`](polydat_grammar_programmatic.md),
builds a selection of these same kernels **programmatically** via the
public AST types, and the test proves the two construction paths project
to **identical** canonical syntax
(`pp_file(builder_ast) == pp_file(parse(grammar_src))`). Those paired
examples are flagged **[↔ programmatic]** below.

---

<a id="sec-lexical"></a>
## 1. Lexical basics

Polydat source is a sequence of **statements**. It is
**newline-insensitive and has no statement terminators** — statement
boundaries are determined entirely by the parser, not by line breaks or
semicolons. Identifiers are ASCII `[A-Za-z_][A-Za-z0-9_]*`.

<a id="sec-comments"></a>
### 1.1 Comments

Three comment forms exist; all are stripped by the lexer:

- `#` to end of line
- `//` (and `///`) to end of line
- `/* … */` block comments — **non-nesting**

```text
# a hash comment, to end of line
x := 1   // a line comment after a statement
y := 2   /* a block comment */
/* block comments do /* not */ nest — this trailing text is live code
```

Note that `pragma` (§[14](#sec-pragmas)) is a real first-class statement,
**not** a comment, despite looking directive-like.

<a id="sec-keywords"></a>
### 1.2 Keywords

There are exactly **ten hard keyword tokens**:

```text
const   input   extern   shared   volatile   cursor   over   pragma   for   tile
```

Everything else — including `module`, `as`, `if`, `else`, `range`, and
every type name (`u64`, `f64`, …) — is an ordinary identifier. Three of
the ten are **soft** in expression position and may be used as names:
`input`, `cursor`, `over`. The other seven (`const`, `extern`, `shared`,
`volatile`, `pragma`, `for`, `tile`) cannot be used as names. `for` and
`tile` are also the two keywords after which the lexer captures raw
text: the comprehension text after `for` (§[16](#sec-for)) and the
template body after a tile's `:=` (§[17](#sec-tiles)). `as` is a *soft
postfix keyword*: it is a cast only in `<expr> as <type>` position and is
otherwise an ordinary identifier (it is also the conventional module
parameter name). `if` is a soft keyword: `if(` is the call form of the
intrinsic and `if <cond> {` its block form (§[7.1](#sec-if-block)), and
a wire may still be named `if` where no block can follow.

<a id="sec-operators"></a>
### 1.3 Operator and punctuation tokens

```text
+  -  *  /  %  **          arithmetic ( ** = power )
&  |  ^  <<  >>            bitwise / shift
&&  ||  !                  logical and/or (eager), bitwise NOT
==  !=  <  >  <=  >=       comparison
:=                         binding
=                          only after `extern …: T` and `cursor …`
:  ->  .  ,                annotation, module arrow, field access, separator
( ) [ ] { }               grouping, arrays, module bodies
```

Two tokens are easy to confuse:

- `:=` (`ColonEq`) is the **binding** operator, used by *every* binding
  form. `=` (`Eq`) is used **only** by `extern …: T = default` and
  `cursor … = constructor`. `x = 1` is **not** a binding.
- `-` is always the minus token; there is no negative-literal token.
  `-5` is unary negation applied to `5` (§[6.3](#sec-unary)).

There is **no `..` token**: `..` lexes as two `.` tokens, and `range(a, b)`
is an ordinary function call, not special syntax.

---

<a id="sec-literals"></a>
## 2. Literals

<a id="sec-int-literals"></a>
### 2.1 Integer literals

Decimal and `0x`/`0X` hexadecimal only (no binary or octal). Integer
literals have type `u64`.

```polydat
decimal := 1000
hexa := 0xFF
big := 4294967296
```

> **Gotcha — no underscore separators.** `1_000` does **not** lex as
> `1000`; it lexes as `1` followed by the identifier `_000`. Write
> `1000`.

<a id="sec-float-literals"></a>
### 2.2 Float literals

A digit run becomes a float only when a `.` is **followed by a digit**.
This is what disambiguates a float from a field access: `1.0` is the
float one; `1.foo` is `1` `.` `foo`; `.5` is `.` then `5`; `1.` is `1`
then `.`. Scientific notation (`1e10`, `2.5e3`, `3.14e-2`) is supported,
with or without a decimal point. Float literals have type `f64`.

```polydat
pi := 3.14
scaled := 1e10
small := 3.14e-2
tau := 6.283185307179586
```

<a id="sec-si-suffixes"></a>
### 2.3 SI suffixes on numbers

A numeric literal may carry an SI suffix, applied only when the
character after the suffix is not identifier-continuation:

| Class | Suffixes | Multiplier |
|---|---|---|
| Decimal | `K` `M` `G` `T` `P` | 10³ … 10¹⁵ |
| Binary | `Ki` `Mi` `Gi` `Ti` `Pi` | 2¹⁰ … 2⁵⁰ |
| Sub-unit | `m` `u` `n` | 10⁻³, 10⁻⁶, 10⁻⁹ |

The two-character binary forms are checked first (so `K` does not eat the
`K` of `Ki`). An SI value that is exactly integral collapses to a `u64`;
a sub-unit suffix promotes to `f64`.

```polydat
thousand := 1K
kibi := 1Ki
half_milli := 5m
integral := 1.5K
not_a_suffix := 1
distance_label := one_Kilometer
```

Here `1K` → `1000` (`u64`), `1Ki` → `1024`, `5m` → `0.005` (`f64`),
`1.5K` → `1500` (`u64`, integral). `1Kilometers` would lex as `1`
followed by the identifier `Kilometers` — the suffix only applies when
nothing identifier-like follows it.

<a id="sec-string-literals"></a>
### 2.4 String literals

Delimited by `"…"` or `'…'`; both kinds are ordinary string literals
(single quotes are **not** character literals — they are commonly used
for charset specs like `'0-9A-Za-z'`). Recognized escapes are `\n`,
`\t`, `\\`, and the active quote character; any other `\x` keeps the
backslash literally.

```polydat
greeting := "hello world"
charset := '0-9A-Za-z'
escaped := "line one\nline two"
quote := "she said \"hi\""
```

`{…}` placeholders inside a string are **interpolation**, desugared to a
`printf` call — see §[9](#sec-interpolation).

<a id="sec-array-literals"></a>
### 2.5 Array literals

`[expr, …]`, possibly empty.

```polydat
weights := [60.0, 20.0, 15.0, 5.0]
empty := []
```

<a id="sec-bool"></a>
### 2.6 `true` and `false`

`true` and `false` are **plain identifiers**, not a distinct boolean
literal token. Comparison and logical operators yield a `u64` `0`/`1`,
not a separate bool value at the wire level (§[7](#sec-comparison)).

---

<a id="sec-bindings"></a>
## 3. Bindings  **[↔ programmatic]**

The fundamental statement is a **binding**: a name `:=` an expression.
A bare binding is **per-cycle** (re-evaluated every cycle).

```polydat compile
input cycle: u64
hashed := hash(cycle)
user_id := mod(hashed, 1000000)
```

This is the minimal idiom: a cycle coordinate in, a bounded id out.
Projected, it is byte-for-byte the same (no `BinOp`, no interpolation, no
float to canonicalize). It is verified end-to-end — built both from this
source and from a hand-constructed AST — in
[the programmatic guide](polydat_grammar_programmatic.md#p-minimal).

<a id="sec-destructuring"></a>
### 3.1 Tuple-destructuring bindings  **[↔ programmatic]**

A multi-output node call binds to a parenthesized target list. The
binding operator is still `:=`.

```polydat compile
input cycle: u64
(region, store, tx) := mixed_radix(cycle, 50, 200, 0)
region_id := mod(hash(region), 10000)
store_id := mod(hash(interleave(region, store)), 100000)
```

`mixed_radix` decomposes a flat cycle into positional digits; a trailing
radix of `0` means unbounded. Paired AST builder:
[programmatic guide](polydat_grammar_programmatic.md#p-destructure).

---

<a id="sec-inputs"></a>
## 4. Inputs

`input name: type` declares a per-cycle coordinate slot driven by
`set_inputs`. The type annotation is **optional and advisory** — every
coordinate input rides `u64` at runtime regardless of the annotation;
the annotation aids inference and documents intent.

```polydat
input cycle: u64
input thread
```

The declaration may be **omitted entirely**. A name a program
references but never binds locally becomes a slot the host must supply,
which the bind pass discovers by collecting free names
([graph_compiler.md §2](graph_compiler.md)) — so `y := hash(cycle)` is
a complete program with one coordinate input. Writing the `input` line
declares the same thing explicitly, and is worth doing where the intent
or the type matters to a reader. Either way the check is the same at
the boundary: a host that fails to supply an inferred slot gets a
reported mismatch, not a default.

Nothing distinguishes `cycle` from any other name. It is the
conventional spelling of the primary coordinate, and it appears
throughout these examples because it is what hosts usually call the
one they advance, but the compiler has no rule about it. A program may
name its coordinates anything, take several, or take one and decompose
it (§[3.1](#sec-destructuring)).

<a id="sec-input-tuple"></a>
### 4.1 Tuple input form

`input (a: T, b: T, …)` is sugar that **desugars at parse time** into N
separate `InputDecl`s. The empty form `input ()` is rejected.

```polydat compile
input (cycle: u64, thread: u64)
combined := interleave(cycle, thread)
row_key := mod(hash(combined), 1000000)
```

> **Projection note.** Because the tuple form desugars to N decls, this
> program projects back as two lines —
> `input cycle: u64` / `input thread: u64` — not as the tuple. That is
> the canonical form, and the round-trip is still idempotent. The
> [programmatic builder](polydat_grammar_programmatic.md#p-tuple-input)
> constructs the two `InputDecl`s directly, matching the projection.

---

<a id="sec-modifiers"></a>
## 5. Binding modifiers — `const`, `shared`, `volatile`

A binding may be prefixed by one or more **wire modifiers**, in any
order. They declare *lifecycle* at the syntactic surface (this is
G-axiom **G2**, §[18](#sec-gaxioms)) — the compiler verifies the
declaration against the wire chain, it does not infer it.

| Modifier | Meaning |
|---|---|
| `const` | Effectively-const: materialised once per scope activation, then frozen. Cannot be shadowed by an inner scope. |
| `shared` | A mutable cell that propagates upward to the enclosing scope after a `for` traversal body. Last-write-wins by default. |
| `volatile` | Per-cycle, but forced Dynamic: excluded from compile-time folding and from program-identity hashing. |

```polydat compile
input cycle: u64
const base := 42
const seed := hash(base)
shared error_budget := 100
volatile attempt := mod(hash(cycle), 8)
user_id := mod(hash(cycle), 1000000)
```

`base` and `seed` have no cycle dependency and const-fold to literals;
`user_id` varies per cycle.

<a id="sec-modifier-combos"></a>
### 5.1 Valid and rejected combinations

`shared const` and `shared volatile` are valid combinations. The
combination **`const volatile` is rejected at parse time** (the two are
contradictory — one freezes, the other excludes from folding), as is a
**duplicate modifier**.

```text
const volatile x := 1    # REJECTED: contradictory modifiers
const const y := 2       # REJECTED: duplicate modifier
shared const z := 100    # OK: a shared cell whose initial value folds
```

> The retired `init` / `final` keyword pair is **gone** — `const`
> subsumes both. Do not reintroduce them.

<a id="sec-shared-typed"></a>
### 5.2 Typed shared cells — `shared x: T := …`

A `shared` binding may carry a type annotation between the name and
`:=`. The annotation pins the cell's type for the cell's lifetime
([Scope Model](scope_model.md) §6.1: a cell keeps one `PortType` and a
write never changes it), so literal inference (`1` versus `1.0`) stops
being load-bearing for the cell's type. An integer literal initializer
widens to an `f64`-annotated cell; any other mismatch between the
initializer's type and the annotation is a compile error at the
declaration. The annotation is accepted **only** on `shared` bindings,
because only a `shared` binding has a cell whose type an annotation can
pin; on any other binding it is a parse error pointing at
`extern name: T = …` as the typed-slot form. The annotation is part of
the canonical projection.

```text
shared total: f64 := 0          # a float cell, initialised from an integer literal
shared status: str := "init"
const scale: f64 := 1           # REJECTED: annotation only on `shared`
shared count: u64 := 1.5        # REJECTED: initializer does not match the annotation
```

---

<a id="sec-expressions"></a>
## 6. Expressions and operators

There is one expression grammar, used identically for a four-character
expression and a two-hundred-line kernel (G-axiom **G6**). The non-sugar
constructors are: identifier, integer literal, float literal, string
literal, array literal, call, field access, the `as` cast
(§[10](#sec-casts)), and the `for` expression (§[16](#sec-for)). Three
**sugar** forms — binary operator, unary negation (`-`), and bitwise NOT
(`!`) — desugar to calls, as does the block form of `if`
(§[7.1](#sec-if-block)).

<a id="sec-precedence"></a>
### 6.1 Precedence and associativity

Loosest to tightest; all left-associative except `**`:

```text
||                     (lowest)
&&
== !=
< > <= >=
|
^
&
<< >>
+ -
* / %
**                     (RIGHT-associative)
unary - !              (- = arithmetic negate; ! = bitwise NOT)
postfix  as <type>     (tightest; binds to the atom; chains left-to-right)
```

So `a > b && c > d` parses as `(a > b) && (c > d)`; `a + b < c * d` as
`(a + b) < (c * d)`; `a + b as u64` as `a + (b as u64)`; and `2 ** 3 ** 2`
as `2 ** (3 ** 2)` (right-associative).

<a id="sec-arithmetic"></a>
### 6.2 Arithmetic

`+ - * / %` choose `u64_*` when **both** operands are `u64`, otherwise
the `f64_*` form with the `u64` side widened by an inserted `to_f64`
adapter. `**` is always `pow` (`f64`). Bitwise/shift operators
(`& | ^ << >>`) and prefix `!` are `u64` unconditionally: they desugar
to the `u64_*` node whatever the operand type, so an `f64` operand
reaches a `u64` port and the wire fails to resolve — a compile error
naming both types, never a silent reinterpretation of the float's bits.

Widening is the only implicit numeric conversion, and it announces
itself. The inserted `to_f64` is logged at advisory level (`widening
u64 → f64 in operator *`), so an author reviewing a module can find
every place the compiler chose the float variant rather than the
integer one. The reverse direction is never implicit: narrowing an
`f64` to a `u64` takes a named cast function, because the rounding is
the author's choice to make (§[10](#sec-casts)).

```polydat compile
input cycle: u64
a := hash(cycle)
sum := a + 7
masked := a & 0xFF
shifted := a >> 3
power := 2 ** 10
```

Projected, the binops are fully parenthesized: `sum := (a + 7)`,
`masked := (a & 255)`, `power := (2 ** 10)`. (Note `0xFF` projects as its
decimal value `255` — the projector emits a canonical integer.)

<a id="sec-unary"></a>
### 6.3 Unary `-` and `!`

`-x` desugars to `f64_sub(0.0, x)`; `!x` desugars to bitwise NOT
(`u64_not`), **not** logical negation.

```polydat
delta := -100.0
bits := !0
```

`delta` projects as `(-100.0)` and `bits` as `(!0)`.

---

<a id="sec-comparison"></a>
## 7. Comparisons and logical operators

Comparison operators (`== != < > <= >=`) select `str_*`/`f64_*`/`u64_*`
by operand family (string beats float beats u64) and produce a `u64`
`0`/`1`. Ordered comparisons (`< > <= >=`) on **strings are a compile
error**; only `==`/`!=` are defined for strings.

The logical operators `&&` and `||` are **eager** (no short-circuit —
both sides always evaluate): each operand is reduced to truthiness
(`x != 0`) and combined with `u64_and`/`u64_or`.

```polydat compile
input cycle: u64
x := hash(cycle)
big := x > 1000000
in_band := (x > 100) && (x < 1000000)
flag := (x == 0) || big
```

`big` projects as `(x > 1000000)`; `in_band` as
`((x > 100) && (x < 1000000))`; `flag` as `((x == 0) || big)`.

The built-in `if(cond, a, b)` is a compiler intrinsic (a closed
parse-time form, G-axiom **G6.i**) lowering to `select_u64/f64/str`;
like the logical operators it evaluates **both** branches. The branch
type is chosen by priority `Str` > `f64` > `u64`: when one branch is
`u64` and the other `f64`, the `u64` branch is widened through `to_f64`.
The condition is `u64`; any nonzero value selects `a`, zero selects `b`.

```polydat compile
input cycle: u64
optimize_for := "LATENCY"
latency_factor := 1.5
recall_factor := 9.5
overscan := if(optimize_for == "LATENCY", latency_factor, recall_factor)
```

<a id="sec-if-block"></a>
### 7.1 The block form — `if cond { a } else { b }`

The same intrinsic has a block spelling. The parser rewrites
`if <cond> { <then> } else { <else> }` to `if(cond, then, else)` before
anything else sees it, in the same way `a + b` becomes `u64_add(a, b)`,
so branch-type dispatch, the `u64`→`f64` widening, and the `select_*`
node are identical between the two spellings: one construct, two ways
to write it. `else if` chains by recursion, so
`if a { 1 } else if b { 2 } else { 3 }` is `if(a, 1, if(b, 2, 3))`. The
block form is an expression and composes like one
(`scaled := 1 + if hot { 10 } else { 0 }`).

```polydat compile
input cycle: u64
n := 12
out := if n > 0 { 100 } else { 7 }
```

```polydat compile
input cycle: u64
n := 5
out := if n > 10 { 1 } else if n > 3 { 2 } else { 3 }
```

```polydat compile
input cycle: u64
s := "LATENCY"
out := if s == "LATENCY" { "fast" } else { "thorough" }
```

Two consequences follow from Polydat being a dataflow language, and the
block form does not pretend otherwise:

- **`else` is mandatory.** A Polydat expression always produces a value
  and there is no unit type, so a one-armed `if` would have no result on
  the false path. A missing `else` is a parse error.
- **The braces are not a guard.** Both branches always evaluate, so
  `if n > 0 { total / n } else { 0 }` does *not* protect the division;
  write `total / max(n, 1)` instead. The block form looks imperative;
  the semantics remain dataflow.

> **Projection note.** The block form projects as the canonical call
> form: `out := if((n > 0), 100, 7)`. The round-trip is idempotent from
> there.

---

<a id="sec-calls"></a>
## 8. Function calls

`name(arg, …)`, where each argument is either **positional** (`expr`) or
**named** (`ident: expr`). Positional and named may be mixed (positional
first by convention); the empty call `name()` is allowed. The soft
keyword `input` is accepted as an argument name.

```polydat compile
input cycle: u64
h := hash(cycle)
u := unit_interval(h)
raw := icd_normal(u, 100.0, 15.0)
clamped := clamp_f64(raw, -50.0, 50.0)
```

Named-argument form (common for stdlib module calls):

```polydat
v := dist_normal(mean: 72.0, stddev: 5.0)
weights := combinations(seed: 0, charset: "A-Z0-9", length: 8)
```

An argument that is a bare literal is **promoted** to an anonymous
constant node, which is why calls mix wires and literals freely:
`pow(x, 2.0)` and `exp := 2.0` followed by `pow(x, exp)` compile to the
same graph. The promoted node is compile-constant, so the fold removes
it before any engine runs and the literal costs nothing per cycle.
Promotion is one of the closed set of parse-time intrinsics
(§[18](#sec-gaxioms), G6.i) — a library cannot add another.

---

<a id="sec-interpolation"></a>
## 9. String interpolation

A string literal containing `{…}` placeholders is **desugared at parse
time** into a `printf` call. The placeholder body is parsed as a full
expression; `{{`/`}}` are literal braces; a malformed placeholder (e.g. a
printf format spec like `{:05}`) silently stays a plain `StringLit`.

```polydat compile
input cycle: u64
(tenant, device, reading) := mixed_radix(cycle, 100, 1000, 0)
tenant_code := mod(hash(tenant), 10000)
device_seq := mod(hash(interleave(tenant, device)), 100000)
device_id := "{tenant_code}-{device_seq}"
label := "sensor_reading"
```

> **Projection note (important).** `device_id := "{tenant_code}-{device_seq}"`
> parses to a `printf` call, so it **projects back as**
> `device_id := printf("{}-{}", tenant_code, device_seq)`. The
> placeholder-free `label` stays `"sensor_reading"`. This is the clearest
> demonstration of projection-as-canonicalization: the runtime gives
> back the desugared form, and the round-trip is idempotent from there.

---

<a id="sec-casts"></a>
## 10. `as` casts

`<expr> as <type>` declares the type a wire should have and inserts the
adapter that gets it there. The rule, in order:

- same type → a no-op passthrough;
- `str as u64` → the string-to-number parse (`StrToU64`): a declared
  reading of text, not a narrowing;
- `f64 as u64` → a **hard compile error**. Narrowing a float to an
  integer under `as` is disallowed because the rounding is a semantic
  choice the author must make by name: `f64_to_u64` (truncate),
  `round_to_u64`, `floor_to_u64`, or `ceil_to_u64`;
- any other pair → exactly the adapter the assembler would insert
  between a wire of the expression's type and a port of the target type,
  from the whole adapter catalog ([Type System](type_system.md) §3):
  every lossless widening (`u64 as f64` is `to_f64`), every register
  retag, and the JSON and byte-string conversions the catalog defines. A
  pair the catalog has no adapter for is a compile error naming both
  types.

Casts may chain (`x as u64 as f64`). A hole's declared type in a tile
(`${x: u64}`, §[17](#sec-tiles)) is the same rule, so a tile can reach
every catalog adapter a binding can.

```polydat compile
input cycle: u64
x := hash(cycle)
ratio := (x as f64) / 4294967296.0
```

`ratio` projects as `((x as f64) / 4294967296.0)`. The cast binds tighter
than `/`, so the parenthesization here matches what you wrote.

```text
narrowed := some_f64 as u64    # REJECTED: use round_to_u64 / f64_to_u64 / …
```

---

<a id="sec-fields-cursors"></a>
## 11. Field access and cursors  **[↔ programmatic]**

<a id="sec-field-access"></a>
### 11.1 Field access

`base.field` projects a field from a source-typed base. The base must be
a bare identifier (or soft keyword); field access does **not** apply to
call results or parenthesized groups. **Chained** dots are accepted and
flattened with `__`: `q.cursor.idx` becomes
`FieldAccess { source: "q__cursor", field: "idx" }`, reading wire
`q__cursor__idx`.

<a id="sec-cursors"></a>
### 11.2 Cursors

A **cursor** is a named `u64` ordinal position tracker driving data
access. Its declaration uses `=` (not `:=`):
`cursor <name> = <constructor> [over <expr>]`. A cursor has no fields or
schema of its own; data is read via accessor functions that take the
cursor's ordinal. The optional `over <expr>` clause
([Cursor Partitions](cursor_partitions.md)) supplies a partition source.

```polydat
cursor users = range(0, 1000000)
```

```polydat
cursor q = range(0, 100) over p
i := q.cursor.idx
ratio := (i as f64) / 100.0
```

> The `over` clause is part of the canonical projection — `pp_cursor`
> emits it, so a cursor with `over` round-trips faithfully. The paired
> AST builder is in the
> [programmatic guide](polydat_grammar_programmatic.md#p-cursor-over).

The **constructor** is an ordinary expression. `range(start, end)` is the
finite-ordinal form. The vectordata cursor-sugar forms rewrite to a
synthetic `range(…)` plus auxiliary projections at compile time (the
cursor-sugar registry is **open**, extensible by node modules via
`inventory::submit!`):

```polydat
cursor row = vectordata_base("example", "label_00")
```

(`vectordata_base`/`_query`/`_source` parse without any feature flag but
only *compile* under the `vectordata` feature, so this block is
round-tripped, not compiled, by the test harness.)

---

<a id="sec-externs"></a>
## 12. Externs  **[↔ programmatic]**

`extern name: type [= default]` declares a slot fixed per scope-init
(via Context Fusion) rather than advancing per cycle like an `input`. The
default is optional.

```polydat compile
input cycle: u64
extern scale: u64
result := cycle * scale
```

`result` projects as `(cycle * scale)`. Externs are the typed,
host-written counterpart to coordinate inputs; the paired builder is in
the [programmatic guide](polydat_grammar_programmatic.md#p-extern).

```polydat
extern balance: f64 = 0.0
extern session_id: u64 = 0
```

---

<a id="sec-modules"></a>
## 13. Module definitions  **[↔ programmatic]**

A module is a named, typed, reusable computation unit:
`name(params) -> (outputs) := { body }`. Parameters are input slots,
outputs are output ports, the body is a statement block. Modules infer
inputs from unbound references and outputs from terminal bindings; the
explicit signature pins the contract.

```polydat compile
sine_wave(input: u64, period: u64) -> (value: f64) := {
    pos := to_f64(input % period)
    per := to_f64(period)
    value := sin((pos / per) * 6.283185307179586)
}
```

The body uses `input` (a soft keyword) as a parameter name and type-aware
infix that resolves to `f64` once operands are `f64`. The paired builder
is in the [programmatic guide](polydat_grammar_programmatic.md#p-module).

> **Projection note.** Module bodies project with four-space-indented
> statements inside `{ … }`, and inner binops are parenthesized while
> calls are not — so the body projects as
> `pos := to_f64((input % period))` / `per := to_f64(period)` /
> `value := sin(((pos / per) * 6.283185307179586))`. Idempotent from there.

---

<a id="sec-pragmas"></a>
## 14. Pragmas

`pragma <name>` is a first-class, module-level compile-time directive
(bare name only, no arguments). Recognized pragmas include
`strict_types`, `strict_values`, and `strict`. **Unknown pragmas are
forward-compatible** — a warning, not an error.

```polydat
pragma strict_types
```

---

<a id="sec-types"></a>
## 15. Types nameable in syntax

The complete set of type keywords usable in `input x: T`, `extern x: T`,
`<expr> as T`, and module signatures is fixed by `PortType::from_keyword`
(`polydat-grammar/src/port_type.rs`, re-exported as `polydat::ast::PortType`):

```text
u64 f64  u32 i32  i64 f32  u8 i8  u16 i16  f16  u128 i128
reg128  reg_i8x16 reg_i16x8 reg_i32x4 reg_i64x2 reg_f16x8 reg_f32x4 reg_f64x2
bool
str | Str | String           (three spellings; Display renders "String")
bytes
json | Json
ext | Ext
handle
vec_f32 vec_i32 vec_f64 vec_i64 vec_f16 vec_i16 vec_i8
```

> **Spelling gotchas.** Vector keywords are **underscored**: `vec_f32`,
> not `vecf32`. Register-lane keywords are `reg_i8x16` … `reg_f64x2` plus
> raw `reg128`. There is **no `f128`** (stable Rust cannot carry it) and
> **no `none`** keyword (`None` is a runtime `Value` sentinel, not a
> type). An unknown keyword is a loud diagnostic, never a silent default.

```polydat
input cycle: u64
extern threshold: f64 = 0.5
extern tag: String
extern embedding: vec_f32
```

---

<a id="sec-for"></a>
<a id="sec-comprehension"></a>
## 16. The `for` construct

`for` is one keyword with two readings that share one compiled form
([The `for` Construct](for_traversal.md)). As an **expression**,
`name := for <comprehension>` binds the comprehension as a value: a
**producer** wire of type `Streamer`. As a **statement**,
`for <source> { body }` with no l-value is a **traversal**: one child
scope activates per tuple, and the comprehension's element names are
wires inside the body. The comprehension algebra itself (constructors,
validity axioms, optimizer rewrites, IR) is owned by
[`comprehension_forms.md`](comprehension_forms.md); this section is the
statement-language surface.

```text
statement          ::= ... | for_stmt
binding            ::= modifier* ident ":=" expr        (* expr may be for_expr *)
expr               ::= ... | for_expr

for_expr           ::= "for" comprehension_text
                    |  "for" ident ("where" predicate)? ("order" strategy ("/" int)?)?   (* derivation *)
for_stmt           ::= "for" for_source "{" statement* "}"
for_source         ::= comprehension_text                (* inline comprehension *)
                    |  ident                             (* a bound producer *)

comprehension_text ::= clause ("," clause)* ("where" predicate)? ("order" strategy ("/" int)?)?
                    |  "[" ("for" comprehension_text ","?)+ "]" ("where" predicate)? ("order" strategy ("/" int)?)?   (* union *)
clause             ::= ident "in" source
                    |  "(" ident ("," ident)+ ")" "in" source
```

<a id="sec-for-capture"></a>
### 16.1 The captured text

The lexer captures everything after `for` as **one token**: up to the end
of the line at bracket depth zero or a `{` at bracket depth zero, whichever
comes first (a bracketed union runs across lines, one member per line), with
string literals skipped whole. The captured text is handed to the
comprehension parser unchanged, so the comprehension grammar has exactly
one owner and `pp_file` prints the text back as written. Three
consequences:

- A traversal's `{` must open **on the same line** as its comprehension;
  `for k in 1..4` followed by a newline is a parse error naming the
  missing block.
- A `{name}` reference inside a `where` predicate is part of the text
  (a block brace is never immediately followed by an identifier and a
  closing brace), so `for k in 1..9 where {k} > 3 {` reads as intended.
- A trailing `//` or `#` comment on the line is not captured.

<a id="sec-for-sources"></a>
### 16.2 Sources, predicates, and order

`source` is the comprehension's source surface: literal lists
(`10,20,30`, `load,verify`, `true,false`), `lo..hi` ranges (`a..=b`
inclusive, `a..b step c`), generator calls such as
`partitions("*/4", n)` and `subdivide(p, n)`, string comprehensions, and
`{name}` references to wires of the enclosing scope, which are resolved
when the traversal is opened. A continuous source is a float range
(`0.0..1.0`, uniform over the interval) or a named measure,
`normal(0, 1)`, drawing from the measure's own support, and
`exponential(1) on 0.0..1.0` for the measure restricted to an interval;
the measures are the closed set `normal`, `exponential`, `pareto`,
`beta`, `log_normal`, `gamma`, and `uniform01`, each taking its own
parameters in order or none for the standard ones. A continuous source
projects only under a sampling order, per the order rules below.
A depth-0 comma starts a new clause only
when followed by `<ident> in`, so value-list and argument commas stay
inside one clause. Tuple clauses `(a, b) in (…)` zip in parallel;
`zip_truncate(…)` and `zip_cycle(…)` choose the zip mode.

`where <predicate>` filters tuples; the predicate names elements as
`{name}`. `order <strategy>[/<n>]` permutes and optionally truncates;
the strategy names the text grammar accepts are `lex`, `reverse_lex`,
`diagonal`, `antidiagonal`, `extrema`, `shells`, `halton`, `sobol`, `lhs`,
and `shuffle`, plus the meta-form `space_filling(<halton|sobol|lhs>, …)`, as a
bare `name`, terse `name/N`, or keyword `name(arg=val, …)`. The keyword
form carries `seed=<u64>` for `shuffle` and `lhs`, the authored seed those
strategies derive their permutation from; any other strategy refuses a
seed. `custom(fn)`
parses but is rejected when the text is lowered to the algebra: the strategy
set is closed.

<a id="sec-for-elements"></a>
### 16.3 Element types

Each element name is typed at compile time from its source:

| Source form | Element type |
|---|---|
| Integer literal list or `lo..hi` range | `u64` |
| Float literal list or continuous interval | `f64` |
| String literal list or string comprehension | `str` |
| Boolean literal list | `bool` |
| `partitions(...)`, `subdivide(...)`, `<name>.partitions` | `ext` carrying `Partition` |
| A generator node call | the node's declared return type |
| A bound `Streamer` used as a source | `Streamer` |

An integer among floats widens the element to `f64`; a list mixing
numbers and strings is a compile error.

```polydat compile
input cycle: u64
for a in 1..4, b in 10,20,30, c in 1.5,2.5, d in load,verify, e in true,false, p in partitions("*/2", 100) {
    x := hash(a)
}
```

<a id="sec-for-traversal"></a>
### 16.4 Traversals

The body is a statement list compiled once, at parent compile time, into
a child program keyed by the statement's lexical position. Every
statement kind is allowed inside it: bindings, `const`, cursors, module
calls, tiles, nested `for` statements, and nested producers. Element
names are typed inputs of the child; outer wires the body references
cascade in with the parent's types; the only coordinate a body may
declare is `cycle`. The projection indents the body four spaces.

```polydat compile
input cycle: u64
for k in 1..4, limit in 10,20,30 {
    f := hash(k)
    g := u64_add(limit, k)
}
```

A cursor inside a body may be declared `over` an element; at activation
the cursor is narrowed to that partition and the body iterates the
slice:

```polydat compile
input cycle: u64
for p in partitions("*/4", 1000) {
    cursor rows = range(0, 1000) over p
    row := mod_in(cycle, rows.cursor)
}
```

Bodies nest; each level is one program, however many tuples flow, and a
nested comprehension may name the enclosing tuple's elements:

```polydat compile
input cycle: u64
for p in partitions("*/4", 1000) {
    outer := cardinality(p)
    for t in 0..20 {
        mid := u64_add(outer, t)
        for d in 0..50 {
            leaf := u64_add(mid, d)
        }
    }
}
```

<a id="sec-for-producer"></a>
### 16.5 Producers and derivations

`name := for <comprehension>` binds a `Streamer`. The binding is
`const`: the compiler lowers it to `const name := streamer("<payload>")`
over the resolved comprehension, so the wire carries the comprehension
as an `ext` value. A traversal may name a producer instead of inline
text, and reads the producer's element names. A **derivation**,
`for <producer> where …` or `for <producer> order …` (or both), applies
the algebra's filter and order to a producer bound earlier in the same
scope; derivations chain. A bare producer name alone in expression
position (`x := for sweep`) is a parse error, since it modifies nothing.

```polydat compile
input cycle: u64
sweep := for k in 1..4, limit in 10, 20, 30 order halton/5
for sweep {
    f := hash(k)
    g := u64_add(limit, k)
}
```

The full surface, with both readings, a producer traversal, an inline
traversal, and nesting, round-trips through `pp_file` byte for byte
(this block is not compiled: `myfunc` and `otherfunc` are not defined
here):

```polydat
sweep := for k in 1..4, limit in 10, 20, 30 order halton/5
for sweep {
    f := myfunc(k)
    g := otherfunc(limit, k)
}
for phase in load, verify, p in partitions("*/4", 1000000) {
    row := mod_in(cycle, p)
    for q in 1..2 {
        z := hash(q)
    }
}
```

```text
base     := for k in 1..100, limit in 1..100
boundary := for base where {k} == 1 || {k} == 100
sampled  := for base order halton/50
```

---

<a id="sec-tiles"></a>
## 17. Tiles

A **tile** is a compiled variate template: a skeleton of static bytes
with typed holes bound to wires ([Polytile](polytile.md)). A tile
statement has the shape of every other wire binding,
`modifier name : type := value`: `tile` is the modifier, the encoding
is the type of the document on the wire (its port type is `str`), and
`:=` binds the wire, so another tile reads it as `${doc!}` and a
binding as `f(doc)`. `:=` is mandatory in every body form; a header
without it is a parse error naming the tile.

```text
statement   ::= ... | tile_def

tile_def    ::= "tile" ident (":" encoding)? options? ":=" tile_body
encoding    ::= "json" | "text" | "csv"                (* default text *)
options     ::= "(" option ("," option)* ")"
option      ::= "delims" string string                  (* hole delimiters *)
             |  "sigil" string                          (* directive prefix *)
             |  "strict"
             |  "instring"

tile_body   ::= json_block                              (* json: balanced { } or [ ] *)
             |  heredoc                                 (* <<< ... >>> *)
             |  string_literal                          (* one-line tiles *)
```

The lexer captures a tile body **raw** after `:=`, as it captures the
text after `for`: a `json` body is a brace- or bracket-balanced block,
string-aware; a heredoc is everything between `<<<` and `>>>`; a string
literal is an ordinary Polydat string. A block body keeps the author's
layout but not the indentation of the statement around it (the common
leading whitespace of the lines after the first is removed), so a tile
inside a `for` body renders the same bytes as at top level; heredoc and
string bodies are exact.

<a id="sec-tile-template"></a>
### 17.1 The template grammar

Inside a body:

```text
hole        ::= open expr (":" type)? ("|" format)? ("!")? close
projection  ::= sigil "for" for_source ("sep" string)? "{" body "}"
branch      ::= sigil "if" expr "{" body "}" (sigil "else" "{" body "}")?
splice      ::= open tile_name close
escape      ::= open open                               (* a literal open delimiter *)
```

`open` and `close` default to `${` and `}`; `sigil` defaults to `@`;
all three are overridable per tile with `delims` and `sigil`. A **hole**
is a Polydat expression in the enclosing scope; `:type` declares the
hole's type by the `as` rule (§[10](#sec-casts)), `|format` is a printf
format spec applied before encoding, and `!` marks the hole raw (no
escaping). Named arguments inside a hole expression are not type
declarations. A **projection** repeats its body once per tuple of a
comprehension; `for_source` is the `for` construct's source surface
(§[16](#sec-for)), inline text or a bound producer, and the element
names are wires inside the body. A **branch** renders one of two bodies
by a `u64` condition. A **splice** names another tile of the same
encoding and inlines its skeleton at compile time. A doubled open
delimiter is a literal open delimiter. The braces of a directive block
delimit it and whitespace padding them is not body; braces inside body
text are balanced, so JSON objects sit in a block unescaped; a hole
cannot appear in a directive header, but `{name}` interpolation may.

```polydat compile
input cycle: u64
tile t := "n=${cycle}"
```

Every body form, with a projection, a raw hole, and per-tile options,
round-trips through `pp_file` byte for byte (not compiled here: the
holes name wires this block does not define):

```polydat
tile doc : json := {
    "tenant": ${tenant_id},
    "samples": [ @for s in 0..4 { { "n": ${s} } } ]
}
tile load : text := <<<
INSERT ${keyspace} '${doc!}'
>>>
tile row : csv := "${a},${b}"
tile odd : text (delims "<%" "%>", sigil "#", strict) := "<%x%> #if c { y }"
```

<a id="sec-tile-typing"></a>
### 17.2 Typing, encodings, and the host forms

Every hole has a type known at compile time, from its declaration, else
its expression's inferred type, else the position's contextual
expectation, and the encoder for the hole is chosen by that type: in a
`json` value position a `str` wire is quoted and a number bare; inside a
string literal or an object key any type renders as escaped text; `csv`
fields are quoted when needed; `text` is the display form. A `json` tile
is checked at compile time as JSON with `0` in every hole. A projection's
comprehension must have bounded cardinality; a continuous source projects
only under a sampling order (`halton`, `sobol`, `lhs`, `shuffle`, or
`extrema`) with a count. Under `(strict)` or the compiler's strict mode, implicit adapters
at holes are rejected as they are on wires. The rules in full are
[Polytile](polytile.md) §4 and §5.

`name := polytile(encoding, body, options...)` and
`name := polytile_json(body, options...)` are binding forms the parser
rewrites into `tile` statements, so a host that holds only a string
lowers it to a tile as a program transform; the body is taken raw and
never evaluated. `tile` is a hard keyword; directives and delimiters have
no meaning outside a tile body.

---

<a id="sec-gaxioms"></a>
## 18. Foundations: the six G-axioms

The grammar is small but does an unusual amount of load-bearing work.
These six structural commitments (stated in full, with what breaks
without each, in [`grammar.md`](grammar.md) §4) are the basis the
substrate, compiler, runtime, and embedding docs rest on. They are
**not** optimizations — without them those layers' contracts would not
hold.

- **G1 — Auto-extern as syntactic discovery.** An identifier reference is
  classified local-or-outer with the *same* syntax; an unresolved local
  reference is searched up the scope chain and synthesised as an extern
  slot. (Type rules `T-LocalIdent`, then `T-OuterIdent`.)
- **G2 — Lifecycle declared at the surface.** `const`/`shared`/`volatile`
  declare lifecycle; the compiler verifies, it does not infer
  (§[5](#sec-modifiers)).
- **G3 — Scope-chain transparency.** An outer-scope reference uses the
  same syntax as a local one — no `outer` keyword or qualifier.
- **G4 — Port-typed expressions.** Every well-formed expression has a
  compile-time-derivable `PortType`; the type rules are total
  (§[15](#sec-types)).
- **G5 — Two-lifecycle structural classification.** Every wire is
  classifiable Effectively-const or Dynamic from the wire chain alone,
  independent of runtime state.
- **G6 — One grammar for expressions and programs.** An expression is a
  program of one anonymous output; a program is a sequence of named
  bindings. Sub-axiom **G6.i**: compiler intrinsics (`if(…)` and its
  block form, literal promotion, interpolation→`printf`, and the
  `polytile`/`polytile_json` rewrites into `tile` statements) are a
  **closed** parse-time set, not extensible by library code. Sub-axiom
  **G6.p**: infix precedence is a stable, grammar-structural commitment
  (§[6.1](#sec-precedence)).

This section summarizes the G-axioms. The full type-inference rules
(`T-IntLit`, `T-Call-OverloadResolve`, `T-BinOp-Add`, `T-FieldAccess`, …)
and the G-axiom composition diagram are in the
[`grammar.md`](grammar.md) formal appendix.

---

<a id="sec-rejections"></a>
## 19. Rejection rules (summary)

The parser/compiler reject, with diagnostics:

- `const volatile` together, and any duplicate modifier
  (§[5.1](#sec-modifier-combos)).
- A type annotation on a binding that is not `shared`, and a `shared`
  initializer whose type does not match its annotation
  (§[5.2](#sec-shared-typed)).
- `input ()` — the empty input tuple (§[4.1](#sec-input-tuple)).
- `f64 as u64`, and any `as` pair the adapter catalog has no adapter for
  (§[10](#sec-casts)) — narrowing requires an explicit rounding node.
- Ordered comparison (`< > <= >=`) on strings
  (§[7](#sec-comparison)).
- An `if` block without `else`, or with anything but `{` or `if` after
  `else` (§[7.1](#sec-if-block)).
- A `for` statement whose `{` is not on the comprehension's line; a bare
  producer name as a `for` expression; a traversal over a producer that
  is not bound in scope; a body declaring an `input` other than `cycle`;
  a literal list mixing numbers and strings (§[16](#sec-for)).
- A tile header without `:=`; an unknown encoding; an unterminated hole,
  block, or heredoc; a projection over a source of unbounded
  cardinality (§[17](#sec-tiles)).
- A bare expression that is not a complete statement (every top-level
  construct must be a statement).
- Undefined wire / unknown function / forward reference (at validate).

> There is **no** “reserved word used as a name” error beyond the seven
> hard-unusable keywords, and **no** “chained dot rejected” error —
> chained field access is accepted (§[11.1](#sec-field-access)). Do not
> assume rejections this spec does not list.

---

<a id="sec-projection-summary"></a>
## 20. Appendix: projection canonicalization, at a glance

What “the syntax the runtime gives back” changes, relative to your input:

| You write | It projects as | Why |
|---|---|---|
| `y := x + 1` | `y := (x + 1)` | BinOps fully parenthesized |
| `name := "{a}-{b}"` | `name := printf("{}-{}", a, b)` | interpolation desugared at parse |
| `m := 0xFF` | `m := 255` | canonical integer rendering |
| `f := 60` (into f64 ctx) / `60.0` | `60.0` | integral floats keep `.0` |
| `input (a: u64, b: u64)` | `input a: u64` / `input b: u64` | tuple input desugared |
| `cursor q = range(0,1) over p` | `cursor q = range(0, 1) over p` | `over` retained |
| `out := if n > 0 { 1 } else { 2 }` | `out := if((n > 0), 1, 2)` | block `if` desugared at parse |
| `for k in 1..4 {` … | `for k in 1..4 {` … | the captured text is printed as written; the body indents four spaces |
| `tile t := "n=${cycle}"` | `tile t := "n=${cycle}"` | the raw body is printed as captured |

Everything in this table is exercised by the suite, which extracts
every <code>```polydat</code> block above, asserts idempotent
round-trip, compiles the <code>compile</code>-tagged ones, and proves
the **[↔ programmatic]** examples project identically to their
hand-built ASTs in
[`polydat_grammar_programmatic.md`](polydat_grammar_programmatic.md).
