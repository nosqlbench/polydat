# Native Tile Rendering — Plan

**Status:** SRD 117, landed 2026-09-11. Steps 0 through 4 are in; §5
records each step and §6 its measurements. The plan set out to close
the refinement SRD 115 §12 and SRD 114 §13 recorded: the P3 tile
renderer was the interpreter's walk behind a helper, not the
straight-line skeleton code the first Polytile drafts sketched. Nothing
here changed what a tile means or the bytes it renders; every step was
gated on the tier differential and on a bench recorded before the first
change. The straight-line code itself was not taken, for the reason
step 3 records: once the allocations and lookups were gone, the walk
was not where the time was.

**Depends on:** [Polytile](polytile.md) (SRD 114) for the skeleton and
its semantics, [Compiled By-Reference Slots](compiled_handles.md)
(SRD 115) for the reference-pair slots a rendered document rides in, and
[Engine Parity](engine_parity.md) (SRD 116) for the `Kernel` trait every
engine shares, which the projection bodies use.

*Since SRD 115's third revision the render node is one closure on
every compiled engine, which native code calls in place through the
slot-call helper (compiled_handles.md §6): what this plan measured as
"P3" is now that closure inside the native segment or cone, writing
into the step's own scratch, and the tile-specific helper it describes
is gone. The measurements below are kept as the record of what each
step bought; the mechanism they describe is the closure's.*

## 1. The problem

A tile renders the same bytes on every engine, and on P3 it rendered
inside native code: `tile_encode` and `tile_render` lowered to helpers
that wrote straight into a thread-local arena. That was the correctness
result SRD 115 set out to reach. The cost result was not there,
because the render path was assembled from the P1 pieces and each
piece still did what it did on the interpreter:

1. **A hole is encoded twice.** The compiler lowers every hole
   `${expr}` to its own binding, `__tile_<name>_hN := tile_encode(expr,
   spec)`, and the tile to `tile_render(spec, h0, h1, ...)` over those
   texts (SRD 114 §6 pass 6). On P3 each `tile_encode` is a helper call
   that encodes the value into the arena and returns a handle; the
   render helper then decodes every handle back into an owned `Value`
   (`decode_args` builds a `Vec<Value>`, copying each arena string into
   an `Arc<str>`), and the walk turns each of those back into text with
   `to_display_string()`, allocating a `String` per hole, before copying
   it into the arena writer. A hole that costs one encode at P1 costs
   one encode, two allocations, and two copies at P3.
2. **The walk is the interpreter's.** `TileProgram::render_into` is one
   body shared by every tier, matching on a `Vec<RtOp>` per op. The
   match is cheap; what it carries is not: `text_of` allocates for every
   `Hole` and `Branch` condition, and a `Repeat` calls
   `evaluate_for_iteration` on every render, allocating the tuple list
   for a comprehension whose tuples were the same last cycle and will
   be the same next cycle.
3. **Projection bodies run interpreted.** A `Repeat` binds each tuple
   into a `PolydatState` over the body program with `set_input(v.clone())`
   per element, cascades the outer wires the same way, and pulls the
   body's holes as `Value`s through `to_display_string()`. The body
   program is the interpreter's, whatever engine the enclosing kernel
   is on, so a tile with a projection is the one place a P3 kernel
   still runs a program on P1 (SRD 116 lifted every other such place).

None of this is visible in the differential, which is what it should be:
the bytes agree. It is visible in a per-cycle cost that scales with the
hole count and, for a projection, with the tuple count times the body's
P1 cost, where the rest of the program pays P3 prices. The engine ladder
does not render a tile, so the cost had never been recorded until step 0
of this plan.

**Where the time goes.** Step 0 timed the pieces on their own (release
build, the machine of §6). One `u64` hole in a two-byte skeleton costs
about 330 ns on P3 and 860 ns on P1 beyond the graph that feeds it, and
each further `u64` hole adds about the same. The pieces: parsing the
hole's spec string, which the P1 and P2 `tile_encode` bodies do on
every call, 115 ns; encoding a `u64` through `Value` display, 90 ns;
turning the arena handle back into an owned `Value` and that into a
`String` in the render helper, two allocations and two copies, about
160 ns; the arena writer itself, 18 ns; the panic guard, 1 ns. A
formatted `f64` hole (`${f | .2}`) costs 530 ns to encode alone because
`formatted_text` renders the value's display text first and then
formats it again with the precision, so the number is formatted twice
and the first result thrown away. A string hole costs 50 ns to encode
but 780 ns end to end on P3, the rest being the handle round trip. So
the cost is not the skeleton walk (99 ns for a whole one-hole render
into a `String`); it is that every hole's bytes are produced twice and
carried through owned values between the two.

## 2. What "straight-line skeleton code" means here

SRD 114 §6 defines the skeleton: `Copy`, `Hole`, `Repeat`, `Branch`, in
order. The first drafts imagined P3 emitting one native function per
tile, a `memcpy` per static run and a typed encoder call per hole, so
that rendering is a sequence with no interpretation at all. That is
still the end state, but it is the last step, not the first, because
the measurable cost in §1 is allocation and double encoding, not the
op dispatch. The plan therefore reaches the end state in three moves,
each measured:

1. **One node, encoded at the hole.** The tile lowers to a single
   `tile_render` whose inputs are the hole *values*, with each hole's
   encoding carried in the skeleton (`HoleSource::Wire(i)` gains its
   `HoleEncoding`). The renderer encodes at the hole, from a borrowed
   `ValueRef` straight into the sink, on every tier: P1 into its
   `String`, P2 and P3 into the step's own scratch entry through a
   `BytesSink`. The `tile_encode` node
   stays in the library (a host may call it) but the compiler no longer
   emits it. On P3 the helper reads its arguments as `ValueRef` views
   over the slots (`marshal::arg_ref`, which the encoder already takes)
   and never builds a `Value`. A hole then costs one encode and one
   copy on every engine, and the graph has one node per tile instead of
   one per hole plus one. Two defects the probe found are fixed on the
   way: `formatted_text` formats a number once, under its precision or
   width, and no engine parses a spec string at render time.

   *Landed 2026-09-11.* `HoleSource::Wire` and `Child` carry the hole's
   spec; `RtOp::Hole` holds the parsed encoding; `render_into` takes
   `&[ValueRef]` and encodes at the hole, and a branch reads its
   condition's truth. The compiler binds each hole expression
   (`__tile_<name>_hN := expr`, or the wire itself when the hole names
   one) and passes the values to one `tile_render`. On P3 the helper
   builds views with `arg_refs`; on P2 and as a closure step in a
   hybrid kernel, `tile_render` supplies its own closure through the
   new `compiled_slot = <path>` override of the node macro, reading
   each slot as a view by the wire type the kernel fixed. The `None`
   rule is unchanged: a hole naming a kernel input reads it through the
   input passthrough, so the render node still fuses beside it. Bench
   after the step, same machine: `flat` 3459 ns on P3 (was 6241) and
   5476 on P1 (was 10715), so a hole costs about 175 ns on P3 and 370
   on P1; `wide` 7857 on P3 (was 13460), 280 ns per hole; `one_hole`
   466 on P3 (was 878); `projected` 8496 on P3 (was 14851), now no
   better than P1's 8360, because the projection's body runs
   interpreted inside the helper, which is step 2's subject. The
   remaining per-hole cost on P3 is the encoder's `Display` path for
   numbers and the read that copies the document out, which step 3
   measures against.
2. **Projections as compiled bodies with memoized tuples.** A `Repeat`
   whose comprehension names no generator clause and no cascaded wire
   in its sources has the same tuples every render; they are evaluated
   once, when the `TileProgram` is constructed, and reused. A body runs
   on the engine of the enclosing kernel: the `TileProgram` holds the
   body's program per engine, as `Traversal::program_on` does for a
   `for` body, and the walk creates a body kernel through the `Kernel`
   trait, binds the tuple and the cascade with `set_input`, and reads
   each body hole with `pull`, encoding from the returned value. Body
   kernels are created once per rendering state and reused, in the
   render node's own scratch. The comprehension evaluator is
   engine-neutral already (SRD 116, step 8 addendum), so a
   comprehension that does read a cascaded wire evaluates against the
   body kernel on any engine.

   *Landed 2026-09-11, with two corrections to the design above.* The
   body's compiled program is built at `TileProgram` construction, not
   on first render, for the first render's cost. Every
   compiled kernel renders its bodies on
   `Engine::default()`, not on its own engine: the render node's closure
   serves the closure tier and a hybrid kernel's closure steps alike
   and has no engine to ask, and the default is the fastest engine the
   build has. `TileProgram::interned` builds outside its lock, since a
   body with a tile of its own interns through the same table; holding
   the lock deadlocked the tutorial. The tuples of a comprehension with
   no generator clause and no placeholder are evaluated once at
   construction; the walk binds each tuple and the cascade through the
   `Kernel` trait into one body kernel per body program and engine,
   owned by the rendering state, and reads each body hole with `pull`.
   `tests/tile_projections.rs` checks the body's engine and
   `library::tile_render::tests` the reuse. Bench: `projected` 8150 ns
   on P3 (was 8496), 8366 on P1 (was 8360): the step is functionally
   complete and nearly free of effect, because the cost was not where
   §1 put it. A probe of one nested body kernel (two holes) driven the
   way the walk drives it costs about 450 ns per tuple on P3 and 375
   on P1; the two hole encodes about 300; the rest of the 1.2 µs per
   tuple is the binding around them. Of the body's 450 ns, a clean
   `pull` alone is 110 to 160 ns on a compiled kernel against 82 on
   the interpreter: the name-keyed trait path is four string-hashed
   lookups, an `Arc` clone of the plan, and a cone walk per pull, which
   a host pulling a few outputs never notices and a projection pulling
   per tuple does. Resolving a body's inputs and holes to slots once
   per kernel, and driving the body through an index-keyed path, is
   the refinement step 3 measures alongside the skeleton code.
3. **Generated skeleton code.** With the allocations gone, the
   remaining interpretation is the `RtOp` match and the `HoleEncoding`
   dispatch per hole. The classifier lowers a `tile_render` whose
   skeleton has no `Repeat` to Cranelift code that calls a `memcpy`
   helper per static run with the interned pointer and length as
   immediates, and a typed encoder per hole (`jit_encode_u64_json`,
   `jit_encode_str_json_quoted`, and so on, one per encoding, position,
   and wire type the classifier has already fixed), each appending to
   the step's scratch entry, with integer and float text produced by the
   fast formatters rather than `Display`. A `Branch` is a native
   conditional over the condition slot. A skeleton with a `Repeat` keeps the helper of step
   2, since a projection's body is a program, not a sequence. This step
   lands only if the bench of §4 shows it wins by more than the drift
   the performance guide records; if step 1 and step 2 already reach the
   floor set by encoding and copying, the plan closes there and says so.

   *Landed 2026-09-11, in two halves, the second measured against the
   first.* The first half is the index-keyed body path step 2 named:
   `Kernel` gains `input_index`, `set_input_at`, `output_index`, and
   `pull_at`, with defaults over the named calls; the interpreter maps
   them to its index APIs, and the compiled kernels resolve an output
   once to its slot, type, and cone (`pull_at` on the cores) and set an
   extern by input index (`Externs::set_at`). A body kernel's entry
   resolves the tuple's elements, the cascade, and the body's holes (now
   numbered within their body) to indices on the first tuple and keeps
   them, so a tuple is bound and read with no string lookup. The
   binary's fibers pull by index too. Paired bench, baseline worktree at
   the step 2 commit against the working tree, same hour: `projected`
   7907 to 6693 ns on P3, 8248 to 7104 on P2, 8334 to 7949 on P1; every
   other case within drift. The projection's per-tuple cost on P3 fell
   from about 1050 ns to 750, and P3 leads the projection case for the
   first time. `tests/kernel_api.rs` checks the index-keyed calls
   against the named ones on every engine.

   The second half is not generated skeleton code. With the allocations
   and lookups gone, what remains per hole on P3 is the encoder itself
   and, per render, the read that copies the document out; the walk is
   about ten nanoseconds an op, which generated code would save at the
   cost of a Cranelift function per tile, and no measurement here
   justifies that. The encoder is where the time is: an integer went
   through `Display` into a `String` and then into the sink, so an
   integer hole with no format now writes its digits straight into the
   sink, on every engine, the same bytes. Floats keep `Display` and the
   precision formats keep `format!`, because a faster float writer must
   produce byte-identical text to Rust's for every value, which the
   shortest-representation writers do not promise, and a tile's bytes
   are the contract. Measured in the same hour as the paired run:
   `one_hole` 493 to 395 ns on P3, 483 to 371 on pure native code, 653
   to 536 on P1; `flat`, with two integer holes among seven, within
   drift.

   *Float writer, landed 2026-09-11.* A float hole with no format and a
   float or integer hole under a `.N` precision now write their text
   through `library::support::float_text`, byte-identical to Rust's
   `Debug` form and to `format!("{:.N}")`: the shortest form takes
   ryu's digits with Rust's layout, falling back to `format!` on an
   exact tie between two shortest candidates, where ryu rounds to even
   and Rust rounds up; the fixed form rounds the exact binary value in
   `u128` arithmetic, half to even, falling back where the value does
   not fit. `tests/float_text.rs` is the proof: edge values, arithmetic
   series, and a million seeded bit patterns for the shortest form and
   every precision 0 through 9, with a sixteen-million sweep behind
   `--ignored`. Its paired measurement is pending a quiet machine and
   will be recorded in §6 when taken.

## 3. What does not change

- **The bytes.** Every step runs under `tests/handle_tiers.rs`, whose
  generator covers every hole shape, encoding, format, declared type,
  raw hole, splice, branch, and projection that has landed, on every
  tier against the interpreter (H7), and under the tutorial's quoted
  output (`tests/guide_output.rs`), which must not change by a byte.
- **The `None` rule.** `tile_encode` writes `null` for a `None` input,
  and a render node fed straight by a kernel input therefore stays a
  closure step on the hybrid kernel and takes its value through the
  `None` mask (SRD 115 §9, SRD 114 §7.2). With encoding at the hole the
  same rule applies to the fused node: the classifier keeps a
  `tile_render` with a possibly-`None` input off native segments, and
  the P2 form renders it with the mask.
- **The typing.** Hole typing, adapters, the `TileHoleTyped` and
  `TileCompiled` events, and `explain tiles` are compile-time and are
  untouched; only the lowering after pass 5 of SRD 114 §6 changes.
- **Ownership.** The renderer writes into the render step's own string
  scratch, and a projection's body kernels live in the same step's
  scratch, so a render touches no storage but its own state's
  (SRD 115 §3).
- **The host surfaces.** `polytile`, `polytile_json`, the `polydat::tile`
  functions, `apply_tile_defaults`, and `--emit tile:<name>` compile
  through the same lowering and are unaffected.

## 4. The measurement

Step 0 adds `benches/tile_render.rs`, a Criterion ladder over one
document, the reading of the toy test definition
(`examples/toy_test_definition.polydat`): a static `meta` arm, seven
holes of four types with two formats, a nested object, one projection
of four tuples with a formatted hole, and a declared boolean. Three
cases per engine, each one complete cycle that pulls the rendered tile:

| Case | What it isolates |
| --- | --- |
| `reading` | the reading's eight wires read directly, no tile: the render cost of every other case is that case less this one |
| `one_hole` | one numeric hole in a two-byte skeleton: the floor for a render |
| `flat` | the document without its projection: encode and copy cost per hole |
| `projected` | the document as written: the projection's tuple and body cost |
| `wide` | the flat document with twenty holes of three types: how cost scales with hole count |

The engines are the four of the engine ladder: P1 with `JitMode::Off`,
P2, P3 (the hybrid kernel), and pure native code, the last only where
every node of the case lowers (`hashed_uuid` and `weighted_strings`
keep the reading off it, so `one_hole` is its only case). The bench records
nanoseconds per cycle and is run against a same-hour baseline worktree
before and after every step, as the performance guide does for the
ladder. The numbers before step 1 are the baseline this plan is judged
against and are recorded in §6 when step 0 lands. The target is stated
in the same terms as the ladder: after step 2, the P3 `flat` case costs
no more than the hole count times the cost of one `str_concat` of the
same bytes plus the static copies, and `projected` costs `flat` plus the
tuple count times the body's P3 cost; step 3 is judged against step 2.

## 5. Steps

Each step lands with its tests, its bench comparison, and leaves the
previous surfaces working.

0. **The bench.** `benches/tile_render.rs` per §4, the three programs
   it compiles, and the baseline table in §6.
1. **Encode at the hole.** `HoleSource::Wire` carries its
   `HoleEncoding`; `dsl::tile_lower` emits one `tile_render` over the
   hole expressions; `TileProgram::render_into` encodes from `ValueRef`
   into the sink; the P2 closure form takes its values as the kit
   already passes them; the P3 helper decodes to `ValueRef`, not
   `Value`, and `decode_args` goes. `tile_encode` stays as a library
   node. `explain tiles` and the skeleton counts in `TileCompiled` are
   checked unchanged. Bench: `flat` and `wide` on P2 and P3.
2. **Projections.** Memoized tuples for a constant comprehension;
   body programs per engine on the `TileProgram`, created through the
   `Kernel` trait and reused per render node; cascade and elements by
   `set_input`; body holes by `pull`. The P1 renderer uses the same
   path with the interpreter's program, so there is one walk still.
   Bench: `projected` on every engine. `tests/tile_projections.rs`
   gains a check that a body's kernel is on the enclosing engine.
3. **Generated skeleton code**, conditional per §2. Bench: `flat` and
   `wide` on P3 and pure native against step 2. If it does not win, the
   step records the numbers and closes without landing code.
4. **Docs.** SRD 114 §6 pass 6, §7.1, and §7.2 rewritten to the landed
   form; SRD 115 §12 and SRD 114 §13 note the refinement closed; the
   Polytile tutorial's engine section keeps its quoted output; the
   performance guide gains the tile ladder beside the engine ladder.

   *Landed 2026-09-11.* The tutorial's lowering description and its
   quoted `explain tiles` output were re-captured at step 1; the rest
   here.

## 6. Baseline

Recorded 2026-09-11 at commit e6ab238, before any change, on the
machine of the performance guide (AMD Ryzen 9 3900X, Windows,
`x86_64-pc-windows-msvc`), with `cargo bench --bench tile_render`.
Each entry is Criterion's point estimate in nanoseconds per cycle, with
its confidence interval; pure native code runs `one_hole` only, since
the reading's two string nodes have no native lowering.

| Case | P1 interpreter | P2 closures | P3 native segments | pure native |
| --- | ---: | ---: | ---: | ---: |
| `reading` | 2861 `[2787, 2949]` | 2565 `[2498, 2660]` | 2147 `[2101, 2195]` | – |
| `one_hole` | 988 `[971, 1007]` | 1043 `[1026, 1064]` | 878 `[815, 960]` | 597 `[583, 613]` |
| `flat` | 10715 `[10407, 11027]` | 10230 `[9902, 10593]` | 6241 `[5995, 6501]` | – |
| `projected` | 18810 `[18593, 19058]` | 17479 `[17222, 17748]` | 14851 `[14562, 15174]` | – |
| `wide` | 20626 `[20401, 20887]` | 20826 `[20494, 21202]` | 13460 `[13154, 13861]` | – |

What the table says, taking each case less `reading`: rendering the
seven-hole `flat` document costs about 4.1 µs on P3 and 7.9 µs on P1,
roughly 590 ns and 1.1 µs per hole; the twenty-hole `wide` document
costs 11.3 µs on P3, 565 ns per hole, so the cost is linear in the hole
count and the static skeleton is not where it goes; and the four-tuple
projection adds 8.6 µs on P3 over `flat`, about 2.2 µs per tuple, for a
body with two holes. For scale, the whole eleven-node engine ladder
graph evaluates in about 60 ns on P3, so one hole costs ten of those
graphs and one projection tuple thirty-five. The engine ratio P3/P1 is
about 1.7 on `flat` and 1.5 on `wide`, against 5.8 on the engine
ladder: a tile is the part of a program that native code speeds up
least, which is the refinement this plan exists for.

**After step 1** (2026-09-11, same machine, working tree at the step's
commit):

| Case | P1 interpreter | P2 closures | P3 native segments | pure native |
| --- | ---: | ---: | ---: | ---: |
| `reading` | 2760 | 2470 | 2243 | – |
| `one_hole` | 588 | 449 | 466 | 551 |
| `flat` | 5476 | 3701 | 3459 | – |
| `projected` | 8360 | 7764 | 8496 | – |
| `wide` | 8393 | 8194 | 7857 | – |

Every engine roughly halved its render cost; P3's `flat` fell from
4.1 µs to 1.2 µs. The projection is the one case P3 does not lead,
which step 2 addresses.

**After step 2** (2026-09-11, same machine):

| Case | P1 interpreter | P2 closures | P3 native segments | pure native |
| --- | ---: | ---: | ---: | ---: |
| `reading` | 2749 | 2380 | 2038 | – |
| `one_hole` | 574 | 448 | 452 | 413 |
| `flat` | 4717 | 3760 | 3424 | – |
| `projected` | 8366 | 7904 | 8150 | – |
| `wide` | 9133 | 8976 | 8654 | – |

The machine drifted a few percent slower than the step 1 run (the
`reading` and `wide` rows moved without a change to their code); the
projection moved 4% on P3 and not at all on P1. The step's record in
§5 says where the per-tuple cost actually is.

**Step 3, index-keyed body path** (2026-09-11, paired: baseline worktree
at the step 2 commit, then the working tree, same hour):

| Case | P1 before | P1 after | P2 before | P2 after | P3 before | P3 after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `reading` | 2641 | 2833 | 2222 | 2617 | 2155 | 2230 |
| `one_hole` | 605 | 653 | 426 | 478 | 422 | 493 |
| `flat` | 4745 | 5136 | 3943 | 3696 | 3719 | 3682 |
| `projected` | 8334 | 7949 | 8248 | 7104 | 7907 | 6693 |
| `wide` | 10071 | 9109 | 8870 | 8341 | 9057 | 8286 |

The unchanged cases (`reading`, `one_hole`) show the pair's drift, about
5 to 15% slower in the second run; against that, the projection fell
15% on P3 and 14% on P2.

**Current record** (2026-09-14, the 0.3.0 tree on the Intel machine of the
performance guide): the table lives in the
[performance guide](../guides/performance.md) beside the engine ladder.
Two things changed since the step 3 pair: every node with a kit lowers
natively, so pure native code runs every case rather than `one_hole`
alone, and the machine is a different one, so the absolute times are not
comparable with the tables above.

## 7. Boundaries

- A tile whose hole is a vector-typed wire, or whose body contains a
  node with no closure form, follows the engine's ordinary rules for
  that node; this plan changes the renderer, not what an engine accepts.
- Two renders of one tile in one program, or the same tile inside a
  projection body and outside it, keep their own body kernels; the
  plan never shares dispense state (SRD 114 §7.1).
- Step 3 generates code per tile skeleton, so a program with many
  distinct tiles compiles more native code; the bench's `wide` case is
  the only place that cost is measured, and a skeleton above a fixed
  size keeps the step 2 helper.
