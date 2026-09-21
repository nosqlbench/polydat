# Architecture review, September 2026

Six read-only reviews of the crate (now the
`polydat-core`/`polydat-nodes`/`polydat-grammar`/`polydat` workspace) as it stood after
engine parity and native tile rendering landed: execution engines and the native boundary; the compiler front end and
the graph it builds; the kernel runtime, cells, scopes and cursors; the language and its
constructs (DSL, tiles, comprehensions, traversals); the node library and the node macro;
and the host-facing surface (entry points, assembler, binary, guides). Each review read
its area of the code and every design document that claims it, and answered three
questions: what is wrong or duplicated, what the documents say that is no longer so, and
what the documents should say instead.

This file is the digest for decision. The full reports, with file and line citations for
every claim, were scratch under `target/review/` and are not retained.

Recorded 2026-09-11. Status as of 2026-09-20: the step-1 fixes (F-C1, F-N3, F-N11, F-E4,
F-E5, F-E13 stated, F-C12 documented, F-K2), F-K1, F-E3, F-E6, groups B, C, I, L,
and the `Lookup` half of group E have landed, and the runtime, node library, and language
have since been split into their own crates; section 3's weeding map is closed, every
document kept-and-weeded or deleted with its disposition in its row; Group B is closed as of 2026-09-18, and the nmbrs node-library request was
answered on 2026-09-19 (host_request_running_counts.md).

On 2026-09-20 the surface work continued: F-L11 and F-E5 are closed, group L
landed as `transform::add_tiles`, `Engine::PureNative` made the fourth tier
nameable (turning up F-E15 and F-E16, both fixed the same day), the extended
kernel API became the `SlotKernel` subtrait rather than `#[doc(hidden)]`, and
the native tier is now built in every configuration rather than refused without
the `jit` feature. The engine ladder was run against a same-hour baseline and
found one real regression, seven `#[inline]` attributes dropped when the shared
core macro landed; it is fixed and every tier is back at or under baseline.
Group D closed with it: one write rule checked on every engine, and the interpreter's borrowing readers renamed to `pull_ref`/`pull_ref_at` so they no longer shadow the trait. The functional table's rows were re-checked against the code rather than trusted, and the nine that had landed without being marked are now marked. Groups A (second half), F, G, H, J, K (in part), M remain open. Of section 5's decisions, 4 and 6 were
Of section 5's decisions, 1, 4, 5, 6, 7 and 8 are taken; 2 (`tile_encode`) and 3 (the
engine projection bodies use) remain open, both inside group F.
Section 1 lists the findings ranked; section 2 proposes
an order of work; section 3 is the weeding map for every design document; section 4 lists
the rules the documents must newly state; section 5 lists the decisions that are the
project's to make before the work starts.

## 1. Findings

Severity, in order: **functional** (observable wrong behaviour or a contract the code does
not keep), **architectural** (one thing written twice, or owned in the wrong place),
**drift** (a document states something the code no longer does), **weed** (dead code or
stale text). Size: XS under an hour, S a half day to a day, M two to three days, L a week.

### 1.1 Functional

| ID | Area | Size | Defect |
|---|---|---|---|
| F-K1 | kernel | S | Constructing or compiling an interpreter kernel opened a root cycle (the build's constant fold seeded inputs through `set_inputs`), which reset the thread's arena. Superseded: the cycle and the arena were removed with SRD 115's third revision (runtime_model.md R4). |
| F-E3 | engines | S | The P3 kernel's provenance mask is one `u64`; a program whose inputs span 64 or more slots (ten cursors plus a coordinate) shifts out of range at build. The closure tier and pure tier use `ProvMask`. **Verified landed 2026-09-20:** the hybrid kernel carries `ProvMask`, not a bare `u64`.  |
| F-E7 | engines | M | The None rule has two predicates: the cone planner's (a None-tolerant node joins a cone only when every input is intra-cone) and the segment batcher's (a node downstream of an extern unset at build is a closure). A host that clears an extern after build reaches the runtime panic in a native segment where the interpreter produces a value. **Fixed 2026-09-20.** There is one predicate now, `compile::none_rule_admits`, called by both fusers: fused code answers a `None` on a boundary input with `None` on all of its outputs, and a node that would have consumed the `None` and kept going may join only when every input comes from inside, where none can arrive. The hybrid's panic is gone — a cleared extern reads `None` there as it does on the interpreter and the closure tier. The build-time taint stays as the narrower case it always was. The pure tier still refuses, and now says why: it is native code with no closure to propagate a `None` through, and its message names both ways an extern ends up without a value and what to run instead. `kernel_api::an_extern_cleared_after_the_build_reads_none` pins all four. |
| F-C1 | compiler | XS | `pragma strict_values` and `strict_types` are honoured on every compiled engine and on the logged interpreter path, and silently dropped on the plain interpreter path. **Verified landed 2026-09-20:** `set_strict_wires` is on the shared assembler build, which every engine and every entry point goes through.  |
| F-H3 | host | S | The assembler's strict checks (implicit adapters, config-wire violations) run only in `compile_strict`, which only the interpreter path reaches. `strict: true` on a compiled engine accepts what the interpreter rejects. |
| F-C4 | compiler | S | Deferred cursor extents and the retained AST are resolved on the interpreter path only; `cursor_schemas()` reports `extent: None` on P2/P3 for a computed extent. **Verified landed 2026-09-20:** the deferred extents resolve on the shared build path, on every engine.  |
| F-H4 | host | S | The two diagnostic entry points build a different program from the one they diagnose (one rejects `for`, one skips pragmas and source text). |
| F-E5 | engines | S | `into_program` on a compiled kernel clones host-set extern values into every created kernel; the interpreter's starts from defaults. The trait does not say which. **Closed 2026-09-20.** The behaviour half had already landed in step 1: `SharedKernel::create_kernel` resets to the program, so all four engines start a created kernel at the declared defaults, verified. What remained was the contract. `KernelProgram::create_kernel` stated it; `Kernel::into_program` did not, and that is where a host converting a kernel it has been writing to actually looks, so the rule and its reason are now there — an extern is per-kernel state, in the same family as the coordinates, and neither is part of the compiled program. `kernel_api::a_kernel_s_own_writes_do_not_travel_into_its_program` pins it on every engine, including that the created kernel is writable in its turn. |
| F-E4 | engines | S | `Native(Auto)` never runs the provenance selector, so the default engine always pays push bookkeeping; `engine()` reports a mode the host did not ask for. **Verified landed 2026-09-20:** `provenance_for` runs the selector for `Auto` on the native engine and the closure tier alike.  |
| F-L3 | dsl | S | A module defined in the program resolves inside a `for` or projection body only because the child compiler re-reads the parent file from disk; compiled from a string, the body cannot see it. **Closed 2026-09-20.** `entry_points::a_program_module_is_visible_inside_a_traversal_body` compiles from a string with no source directory and drives every engine, asserting the module runs inside the body; it passes, so the body no longer depends on the parent file being on disk. |
| F-K5a | kernel | S | A `for` body that reads an outer shared wire sees a snapshot at open time, not the cell; the design promises the cell. Untested on any engine. **Closed 2026-09-20, as documented rather than as filed.** The premise is stale: both `for_traversal.md` and `scope_model.md` §4 already say the body receives the cell's value at open and not the cell, which is what the code does — verified identical on the interpreter, the closure tier and the native tier. The cascade is a snapshot, a `shared` wire included, which is the capture-not-freeze rule rather than a defect. What was accurate is the last sentence, so `for_engines::a_body_reading_an_outer_shared_wire_sees_the_value_at_open` now pins it: a write to the cell mid-traversal reaches no activation of the open stream, re-opening picks it up, and the parent goes on reading its cell live. |
| F-E13 | engines | XS/M | One refusal remains on compiled engines (an extern wider than one slot) while the documents and the trait doc say every engine accepts every program. State it or close it. |
| F-N3 | library | XS | `file_line_at` and `regex_replace` are registered twice (legacy sig and macro); which diagnostic a bad pattern produces depends on link order. **Verified landed 2026-09-20:** neither is registered twice.  |
| F-N11 | library | XS | Link order leaks into the pinned parity matrix text; adding a module reorders rows and fails the test with empty diffs. **Verified landed 2026-09-20:** the matrix names are sorted before comparison.  |
| F-E6 | engines | S | The public `P2Engine`/`P3Engine` selectors hand a host a kernel that never resets the arena. No in-tree caller; deletion. **Verified landed 2026-09-20:** both selectors are gone.  |
| F-H7 | host | M | Every binary command compiles the program twice (interpreter for names and manifest, run engine for the run); `run --emit` four times. **Reduced, not closed, as of 2026-09-20:** `describe` reuses the run engine`s program when that engine is the interpreter, so `run --emit` compiles twice rather than four times. The probe compile remains.  |
| F-N8 | library | S | `dynamic_weighted_select` re-parses its spec every evaluation; `random_string` re-parses its charset per call; every `Const<Vec<C>>` node clones its list per cycle. |
| F-N6 | library | S | Const-argument constraints cannot be declared through the macro; the substitutes are panics caught at construction and the legacy `validate_node`. **Closed 2026-09-20.** `#[constraint(...)]` now takes any `ConstConstraint` expression and emits it into the argument's `ParamSpec`, and a node states a rule between two of its constants with `validate = <path>`, which the factory calls after the per-parameter constraints pass. `chance`'s range and `n_of`'s non-zero `m` moved from body panics to declarations; `n_of`'s `n <= m` and `in_range`'s `lo <= hi` moved to node validators, which retired `param_helpers::validate_node`'s name-keyed table. The check turned up a parity defect while it was being written: `n_of(cycle, 9, 3)` panicked on the interpreter and the closure tier and **built and returned `U64(1)` under native lowering**, because the body that held the rule is never run there — the same class as F-E7. A rule read by the factory holds on every engine, which `a_relation_between_parameters_is_refused_on_every_engine` now pins across all three. |
| F-C12 | compiler | XS | Plan B (a scope-init const that yields None is a hard error) is documented and not enforced; the materializer warns and continues. Doc or code, one of them changes. **Closed 2026-09-20: the doc, and it already had.** `evaluation_model.md` §"Scope-Activation Pull (Plan B)" states the warn-and-continue behaviour and gives the reason — a `const` may depend on resolution that is not ready until the workload runs (`dataset_prebuffer` and its kind), so a warning at activation plus the failure in context at first use is the diagnostic pair an operator can act on. No design document claims the hard error; only this row did. The code matches: catch, warn naming the binding and the panic text, leave the buffer at `Value::None`, activate. One real defect turned up while checking: the warning went to `eprintln!` rather than the audit sink, so a host that installed a log function never received the one diagnostic the design leans on. It goes through `audit::warn` now. |
| F-L11 | dsl | XS | An array literal in argument position is dropped during lowering instead of refused. `out := printf("{}", [1, 2])` compiles, supplies zero wire inputs, and panics at eval ("format references input #0 but only 0 wire input(s) supplied"); bound first (`w := [1, 2]`) it works and yields `Str("1, 2")`. A list literal is a binding-position form (polydat_grammar.md §18.1), so the argument position should be a compile error naming the form. Found 2026-09-18 while merging the type rules. **Fixed 2026-09-19.** |
| F-E15 | engines | S | The pure native tier runs no compile-constant fold at build, where the closure tier and the hybrid each run their constant steps once (`run_steps(&constants)`) so that what is knowable at build is known at build. The pure tier is one compiled function with no step list, so it has no subset to run. A deferred cursor extent is read from the kernel's folded constants, so `cursor_schemas()[0].extent` is `Some(0)` on this engine where every other engine reports `Some(15)`. Against the fail-fast flattening rule. Found 2026-09-19, when `Engine::PureNative` made the tier reachable and the shared engine list drove it for the first time. **Fixed the same day**: the constant steps are compiled a second time into an entry of their own and run once over the kernel's buffer, which is the hybrid's semantics reached the only way a tier with no step list can reach them. `every_engine_folds_its_constants_at_build` pins the failure half. |
| F-E16 | engines | S | A compile-constant step that cannot be computed was reported four different ways. The interpreter caught the panic, warned "skipping fold", and continued, so the same failure arrived on the first pull instead; the three compiled engines panicked out of the builder rather than returning an error; and once they returned one, it read as `KernelError::Refused`, which says one engine declined a program the others accept. Found 2026-09-19 while fixing F-E15. **Fixed the same day**: it is `KernelError::ConstantFold` on every engine, an error and never a panic, carrying the node's own enriched message. A step no input reaches will do at every pull what it does at build, so there is nothing a later evaluation could supply; deferring only moved the failure. `entry_points::every_engine_folds_its_constants_at_build` and `handle_boundaries::a_failed_parse_over_a_literal_is_a_build_error` pin it, and the latter's dynamic twin pins that the *timing* follows the step's lifecycle and not the engine. |

### 1.2 Architectural

Grouped where the findings are one change.

| Group | Findings | Size | One change |
|---|---|---|---|
| A. One compiled core | F-E1, F-E2, F-E10, F-E8, F-E11 | L | The closure tier and the hybrid kernel are one engine written twice (same thirty fields, same twenty methods; the no-jit hybrid is a third copy). Eleven provenance-mode types carry one flag. One core, one step enum `{Native, Closure, Copy}`, provenance as a field with an optional cone guard; the closure engine is the same builder with every node classified fallback. The pure tier keeps a raw kernel for the differential and Tier-1 and drops the rest. Net deletion around 1,500 lines. Gated by the ladder. **Measured 2026-09-19**, which splits the group in two. Of the 23 methods the two cores share by name, **17 bodies are byte-identical** and two more differ only cosmetically (`ref_entry` by an import path, `plan` because the hybrid counts native segments the closure tier has none of). The genuine divergence is four methods: `run_fresh`, `run_guarded`, `run_order`, and `validate_refs`, and the first three are the hot loops, which differ because the step type does. So the safe half is sharing ~19 bookkeeping methods and the ~24 fields they read, which touches no evaluation path; the half that needs the ladder is reconciling three loops. `Kernel::engine` was made state rather than type on 2026-09-19, which is what lets one type serve both tiers. |
| B. One entry point | F-H1, F-H2, F-C3, F-C2, F-C15, F-H9 | M | Twenty-four `compile_*` functions, six of them full pipelines with different settings; twenty-two assembler constructors with a fallback that returns an empty kernel or panics. One `Compiler::from_options`, one `compile_file` shared by the interpreter and compiled paths, four public conveniences, one `KernelError`; everything else a deprecated two-line wrapper. **Mostly done 2026-09-18**: the entry points are sixteen dividing by what they return, each named for it, every one on `KernelError`, the fourteen deprecated wrappers deleted rather than left to be found. The empty-kernel fallback and the `expect` on a public method are gone. The assembler's builders followed: the four with no caller and the one that was a pure alias of `compile_hybrid` are deleted, leaving eight, and the family is `#[doc(hidden)]` throughout, which is what §1 already said it was. Deleting them made `JitKernelPush` and `JitKernelPull` unconstructible, so the pure tier's two unreachable provenance variants went with them — two of the eleven mode-carrying types group A names, removed by the group that exposed them. |
| C. Cones belong to the interpreter engine | F-H5, F-H6 | S | `JitMode` is a process global that changes what `Engine::Interpreter` builds. Make it part of the variant (`Interpreter { cones }`), delete the global; the binary's `--engine off/auto/force` becomes `--engine interpreter/closures/native` with `--cones` for the interpreter. **Done.** The global went with the variant change; the binary's flags split on 2026-09-18, and the two hidden aliases that kept the old spelling working were deleted 2026-09-20, so `--engine off` is now an error listing the five values rather than a synonym. With them went the last place where one flag set two things: `cones()` reads `--cones` and nothing else. |
| D. One trait, one meaning | F-K2, F-K3, F-H8, F-E9, F-K6 | M | `invalidate_all` means two things; three write rules on the interpreter (heal, reject, widen) and a fourth on compiled kernels; inherent methods shadow the trait with different return types; four hidden construction hooks on the host trait; cycle ownership spelled four ways with no assertion. One write rule at every engine's `set_input`; one `invalidate_all`; borrowing readers renamed; internals behind a sealed trait; cycle ownership stated as an R-axiom with a debug tripwire. **Closed 2026-09-20.** `invalidate_all` has one documented meaning and `KernelInternals` is a sealed `pub(crate)` trait a host neither sees nor implements, both from earlier passes. The write rules turned out already unified by the structured-`WriteError` work — driving seven bad writes through all four engines gives the same variant with the same facts every time, and nothing heals, widens or narrows at the boundary. One divergence remained and is fixed: `UnknownWire.known` listed externs only on the compiled tiers and every input slot on the interpreter, and the compiled tiers' own indexed write disagreed with their named one; all three now report `Kernel::input_names`. `WriteError` joins `KernelError` at the crate root, since they are the pair a host handles. `every_engine_writes_by_one_rule` pins the whole shape. F-K6 was superseded by SRD 115 rev 3, which removed the cycle. F-E9 closed the same day: the borrowing readers are `pull_ref` and `pull_ref_at`, so neither shadows a trait method. The rename could not be a sed — dropping the inherent `pull` lets the concrete-typed call sites fall through to the trait.s, which clones, with no compile error — so the old names became `#[deprecated]` shims first, the compiler named all 423 sites that resolved to them, only those lines were rewritten, and the shims then went. Measured flat on the ladder, every tier. |
| E. One binder for child kernels | F-K5, F-K11, F-K12, F-L9, F-C6 | L | Subcontext materialization (interpreter only, cells and transit) and traversal activation (any engine, snapshot) are two mechanisms for one idea. Express binding over `dyn Kernel` plus a `Lookup` view; streams remember the engine that opened them; `Lookup` implemented for a layered constant view so opening a traversal allocates no interpreter state; the kernel-bound evaluation surfaces take `&dyn Lookup`. Part (b), a `ParentView` instead of a transient parent kernel, is M on its own and removes F-K1's second trigger. |
| F. One body carrier | F-L2, F-L4, F-L5, F-C13 | M | Tile projection bodies and `for` bodies are two lowerings that have diverged (settings, engine choice, error path, three compiles per tile body). Tile lowering builds a `PolydatFile` plus `BodySource` as traversals do; bodies live in a program-owned table with `program_on(engine)`; the compiler hands the node an `Arc<TileProgram>` through an opaque const instead of JSON text; `tile_encode` is retired or documented as host-callable. |
| G. One macro kit | F-N1, F-N2, F-N7 | L | Three closure kits with three eligibility rules over six shape vocabularies; the Rust-type-to-slot mapping in four places, two dead; native lowering a by-name match with a positional constant contract and eleven names that are not nodes. One kit from one argument-read enum; slot I/O on the `Wire` trait; a node declares its lowering the way it declares its closure. |
| H. One classifier, one tripwire | F-C5, F-E8, F-E14, F-K6 | S | Lifecycle classification is duplicated in the cone extractor and again in the hybrid batcher; nondeterminism has four encodings; H4's validation is written three times; `EnginePlan` on the interpreter is computed by node-name prefix. **F-E14 and most of F-C5 done 2026-09-20.** `engine_plan` counts native segments by asking each node whether it stands in for a subgraph (`fusion_subgraph`), not by a prefix on its display name. The cone extractor's second lifecycle walk is deleted and calls `PolydatProgram::classify_lifecycle`; the copy knew nothing of the `volatile` modifier or of volatility's contagion. `first_dynamic_wire`'s five-name list of nondeterministic sources is replaced by the declaration it was standing in for. **One real defect fell out of the third encoding**: the never-current set is computed by the inventory when the program is built, and the output modifiers are installed after it, so a node feeding a `volatile` output was excluded from the fold (which reads the modifiers) and then cached per cycle at runtime (which did not) — half of what `evaluation_model.md` §"Non-Deterministic Nodes" states. `volatile_is_never_current.rs` shows a volatile binding returning its first value on every later cycle, and passes on the fix. Two items are left and neither is what the finding says: `scalar_ok` is now three lines over `slot_color` rather than a restatement of marshal's rules, and its `Imm2` exclusion is the cone boundary's own rule; the "second eligibility check" is the codegen attempt, which extraction is designed to be able to fail. |
| I. Legacy scaffolding out | F-N4, F-N5, F-N10, F-N9 | M | About 840 lines of empty `register_nodes!` bodies; `limit` and fourteen vector nodes hand-written where the attribute would do; `instantiate` with no user; placeholder port types in the static graph against the alignment axiom. |
| J. Compiler tidy | F-C7, F-C8, F-C9, F-C16 | S each | Error variants reconstructed from message strings; a hand losslessness table disagreeing with the catalog; a dead extern-type inference; 128-bit types documented as interpreter-only with no parity case. **F-C9 done 2026-09-20**: `infer_auto_extern_type` and its name table are deleted. It was reached only when the assembler had no output type for a const binding whose right-hand side had just been compiled, and a probe that panicked there instead ran the whole suite — 2542 tests — without firing once. The auto-extern slot now takes the compiled node's own `PortType`, and the assembler failing to carry one is an internal error rather than a second guess from the surface AST followed by `Ext`. **F-C8 done 2026-09-20**: `is_lossless_adapter`'s eleven-pair table is gone; the answer is computed from `PortType::numeric_domain`, a description of what numbers each type holds, so the whole catalog is classified by one rule. The table was wrong in both directions — it refused `U8 → U64` and every other narrow widening as lossy, and it called `U64 → F64` and `I64 → F64` lossless where both round above `2^53`, which is the correction the strict surfaces now carry (two tests moved to a `Str` target, and one new test states the refusal). The finding's premise that this contradicts `type_system.md` §3 does not hold: class A is *always-defined*, which `u64 → f64` and `u64 → bool` both are and neither is lossless. Both documents now say so. **F-C7 done 2026-09-20**: `classify_compile_error` takes the `KernelError` rather than its rendered text, so `AssemblyError::TypeMismatch` hands its four fields across instead of `(unknown)` and `U64 → U64` — `hash(json_object())` now reports `Json → U64` between the two nodes. `UnknownNode` carries `registry::suggest_function`'s answer, and `LifecycleMismatch` carries the compiled kernel's inputs, which are what the expression waits on. `ResultMissing` and `Timeout` are deleted; nothing but the display smoke test ever built one. A fourth arm turned up dead on inspection: the classifier tested for a `"not a const expression"` message that no compiler path emits — it is the text of `LifecycleMismatch`'s own `Display`. Three tests in `core_with_library` pin one field each. **F-C16 done 2026-09-20, closing group J**: the drift half was already gone — no document still calls the 128-bit types interpreter-only — and the last claim of it, a comment on the `u128` `Wire` impl, now says what `JIT: None` means: no named native lowering, the carrier crossing as a limb pair. The test gap is closed by `the_128_bit_carriers_agree_on_every_engine`, which widens a cycle into each carrier, reads it back through all six projections, and asserts the four engines return the same `Value` — built through the assembler, since these adapters are `__`-prefixed and no DSL program in the matrix reaches one. It also asserts that all three compiled tiers built the program, the pure one included, so it cannot pass by being skipped. |
| K. One grammar | F-L1, F-L7, F-L8 | M | Four grammar documents, two claiming sole authority; the tested one omits `for`, `tile`, the `if` block and `shared x: T`. One normative, test-checked file, with a test that fails when an AST variant has no example. The comprehension text front end renamed from "legacy" and `custom(fn)` removed. |
| L. Host tiles as a transform | F-H12 | S | The `_with_tiles` entry points and `polydat::tile`'s exposed internals become a public AST transform, per the standing rule that host features are program transforms. **Done 2026-09-19**: `transform::add_tiles` adds host-built tiles to a parsed program and refuses a name the program already declares, reaching module and `for` bodies to do it. The two `_with_tiles` entry points are deleted, and `parse_polydat` / `parse_polydat_with_tile_defaults` make the transform path first class, so a host takes parse, transform, compile without reaching into `dsl::lexer`. Adding tiles now composes with `assign_values` and with any rewrite of the host's own, which an entry point per feature could not. |
| M. Generated node reference | F-N12, F-N13 | S | The hand-curated half of `nodes.md` is wrong about the library (counts, missing files, host nodes documented as polydat's); `library_catalog.md` is an older second catalog. Generate the listing from the registry once the macro carries help text. |

### 1.3 Drift and weed

Listed in section 3 by document, and in the reports by file. The ones that are code, not
documents: F-E12 and F-C11 (module docs and comments in plan vocabulary), F-C10 (four
user-facing messages naming a plan step), F-L6 (the pretty-printer's doc states a retired
purpose while it is now the projection contract), F-K4 and F-K10 (interpreter-state
helpers the trait path covers; removed APIs named in module docs; dead compatibility
variants), F-N10 (dead macro attributes; the macro emits empty help text).

## 2. Proposed order

Functional first, then architectural in dependency order. Every step gated on the full
commit gate; steps touching closures, hybrid or JIT kernels also gated on the two ladders
against a same-hour baseline.

1. **Small functional fixes, one commit each**: F-C1, F-N3, F-N11, F-E4, F-E5, F-E11,
   F-C12 (doc), F-E13 (state), F-K2 (rename). Half a day together.
2. **F-K1 with the H5 tripwire** (group H's debug run-depth counter lands here). One day.
3. **F-E3** mask width, with group H's single provenance computation. Half a day.
4. **F-E7** one None predicate. Two days.
5. **Group B** entry points, taking F-H3, F-H4, F-C4, F-L3 with it since they are all
   "the two paths do different things". Three days.
6. **Group C** cones on the interpreter variant and the binary's flags; F-H7 once the
   `Kernel` trait carries what the binary reads. Two days.
7. **Group D** trait meaning. Two days.
8. **Group A** one compiled core. One week, mostly deletion; the ladder is the gate.
9. **Group F** one body carrier, then **group E** one binder (E depends on F for the
   tile side). Two weeks together.
10. **Groups G and I** the macro. One to two weeks; parity matrix is the gate.
11. **Groups J, K, L, M** as they fit.

The design-document rewrite (section 3) runs alongside from step 5, since groups A to F
change what the documents must say; the weeding of status and plan text can start now.

## 3. Weeding map

Rule applied throughout: the reasons a design is the way it is stay; status lines,
revision logs, step lists, landing records, "since step N", dates of measurement, links
to host-project SRDs and test-file names go. Every kept document states the present
mechanism on every engine.

| Document | Role after weeding | Action |
|---|---|---|
| engines.md | The engine lattice and the rules every engine follows (the parity rules) | **Rewrite.** Keep the mode table, §3.1-3.3, §5, §6 properties, §7 placements. Absorb engine_parity.md's rules. Add: the one evaluation rule, the cycle rule, the None rule (single statement), the failure contract, `invalidate_all`, declaration-order outputs, `into_program` and per-thread kernels, where the selector applies, `EnginePlan`. Drop "since step N", `try_compile*`/`auto_compile_*` as API, the "no branching" rationale that is no longer true. |
| engine_parity.md | none | **Deleted** 2026-09-17: its rules are engines.md §3, §6, §7, §8 and for_traversal.md §5; its method is engines.md §7. The code cited it by step and aberration number; each citation now names the rule's section. |
| jit_boundary.md | The Rust-to-Cranelift call and failure boundary, helper ABI, slot-state axioms | **Rewrite** the overview to the three sites native code runs (embedded cone, P3 segment, pure function), all through `invoke_with_catch`; add `guarded`, the tracker slot, the Windows jmp_buf form. Delete the ratification narrative, "since step 7", the Tests section, the `into_parts` paragraph, "128-bit integers do not ride". |
| compiled_handles.md | The Hdl1 contract: format, lifetimes, value table, helper ABI, axioms | **Weed.** Keep §1 (three limits only), §2-§10. Fold §12's four rules into §4/§7/§8 and the `compiled_handle = <path>` override into §7. Rewrite §6.1-6.2 to the landed projection form. Delete status, "before this document", §11, §12. |
| tile_native_rendering.md | none | **Deleted** 2026-09-18. polytile.md §7 already carried the landed forms, both corrections with their reasons, and the float writer contract, from its own weeding; §7.2 gains the rule that rendering changes a tile's cost on an engine and never what the engine accepts. The helper-reads-borrowed-views rule is compiled_handles.md §6, stated as the general rule for slot-call helpers with why a decode-to-`Value` helper is invisible in a differential. The ladder's five cases and what each isolates are the performance guide, which already held the current record and the paired before-and-after. What did not move: the per-step baselines, the probe timings, and the step list, which are the record of doing the work rather than reasons the design is what it is. |
| simd_isa_autopromotion.md | Tier-1 scalar-flow promotion | **Weed.** Drop status, the benchmark table and environment block; SRD names to titles; note the register kernel is the pure tier's reason to exist. |
| runtime_model.md | The runtime contract (R-axioms, D-axioms) on every engine | **Rewrite** §3-§4 as the one rule with a two-row realization table (interpreter; compiled). Add the `Kernel` trait in R-terms, R4 cycle ownership, shared cells on every engine, provenance as an optimization. Delete the SRD table, the enforcement file maps, the ownership declaration. |
| scope_model.md | What a scope is and how one is constructed | **Rewrite** §1, §2, §4, §8 to kernels on any engine, `create_kernel`, the binder's steps in present tense. Add the traversal cascade rule once decided. |
| cross_fiber_invalidation.md | The cell protocol (publish value, revision, intent bit) | **Rewrite** §3.1 (three creation sites), §5 (provenance-wide invalidation and why), §12 (costs); state the compiled consumer's 64-cell word bound or fix it. Delete §10.1-10.3, 10.5. |
| subcontext_construction.md | The host-composed child protocol | **Weed.** Delete §9, the compatibility-variant paragraph, the status line. Add one paragraph placing it beside traversal activation and stating the intent that they share one binder. |
| composition_substrate.md | S/T/L axioms | **Weed.** Rewrite S3, T3, §8.4, §10.3, §12; fold §12.1/12.3 in. Delete the ownership declaration, §9 cross-reference table, "shipped" status, nbrs tier names. |
| cursor_partitions.md | Cursor partition axioms CP1-CP9 | **Weed.** Rewrite §7.2's last two paragraphs to `cursor_schemas` on every kernel; delete status header; point the partition table at nodes.md. |
| wire_materialization.md | The cross-scope read invariant and materialization gradient | **Rewrite** the gradient bullets to the code's order; strip nbrs vocabulary and SRD-18 references; state the gradient holds on compiled engines. Delete the preamble link block and "what this SRD covers". |
| evaluation_model.md | Program/state split, two lifecycles, input spaces | **Rewrite** the folding phases, scope-init pull, diagnostics, compilation levels ("the host chooses an Engine; default is P3"). Delete the prior-model paragraph, the extends-list, the code reference points. Overlaps runtime_model §3-§5; consider merging into it. |
| graph_compiler.md | The compiler's construction contract (H, CF, NF axioms) | **Rewrite** §2 table and §6 diagram to the actual pass order; §5 to two mechanisms (adapter insertion at wire resolution; subgraph fusion); §4.2 to `materialize_subscope`. Delete the ownership declaration and SRD list, §3.5, §7, §9, NF3's unimplemented sentence, the "engine selection" pass. State the pipeline is engine-neutral to `ResolvedDag`. |
| ir_architecture.md | Reference for the comprehension IR | **Weed** (landed). Rewrite the boundary paragraph as the rule; delete the file-layout table; note the evaluator is engine-neutral. |
| expression_engine.md | Host-facing evaluation contract (E-axioms) | **Rewrite** §3.2-3.4 signatures to the real ones; fix E4 vs §5.5; §6 to emitted variants. Delete §7, §10, §11, the ownership declaration. |
| type_system.md | Value types, adapters, strictness | **Weed.** Fix U128/I128 claims; drop nbrs-runtime reference; SRD refs to links; §8 to four anchors. |
| type_system_alignment.md | The four-plane alignment and slot-color contract | **Weed** §7 only: closure form derived from signature, native form declared by the node; P2 total, P3 native-where-lowered. State that `EnginePlan` makes eligibility observable. |
| none_semantics.md | The None rule at the language level | **Weed.** Fix citations to files that no longer exist; drop SRD tags and the incident narrative; state Rule 1 holds on every engine. |
| library_catalog.md | Design rationale for the node library (authoring contract, cost classes, open registry) | **Rewrite.** Keep cost classes, host-registered nodes, the `pick` and `exactly_one_value` rationales, registration's first paragraphs. Replace registration bullets with the shape table. Delete every signature table, Polydat Modules, Node Fusion, the vectordata history, host SRD references. |
| module_system.md | Discovery, resolution, inlining | **Keep.** Add what a body inherits, program-local modules inside bodies (after F-L3), tiles in module bodies. |
| polydat_grammar.md | The normative, test-checked language spec | **Rewrite** to absorb grammar.md §3-§5, `for` (both readings), tiles, the `if` block, `shared x: T`, the `as` catalog; fix the keyword count, the `as` rule, the shuffle claim. |
| grammar.md | none | **Deleted** 2026-09-18. Its three unique contributions merged into polydat_grammar.md: the type rules and the six G-axioms with what breaks without each are §18, the composition diagram §18.4, the EBNF productions §21. Two claims died as stale rather than moving: `true`/`false` are identifiers, not a `Bool` literal tier, and `Cast` counts as a non-sugar constructor (nine + three, not eight + four). One died as false: `T-ArrayLit` did not yield `Vec<T>` — a list literal binds to a constant string of its rendered elements, which §18.1 now states with the reason. |
| language_spec.md | none | **Deleted** 2026-09-18. Most of it had already been superseded: the conditional, the `as` rule, precedence, comparisons, and the type keywords were all in polydat_grammar.md, the carriers and the adapter grid in type_system.md, the pipeline in graph_compiler.md, and the invalidation rule in runtime_model.md §3–§4. What it alone held moved: the dispatch table's two live facts (bitwise on an `f64` operand, the widening advisory) to polydat_grammar.md §6.2, literal promotion to §8, implicit inputs and the `cycle` demystification to §4; the `PolydatNode` surface to library_catalog.md; the compiled-program layout and the compile log's three levels to graph_compiler.md §2.1–§2.2; the "braces are not a Polydat expression form" boundary to expression_engine.md §3.2.1. One claim died with it as false: `///` does not attach a doc comment to the next binding, the lexer strips it like any line comment. |
| polydat_grammar_programmatic.md | Proof that the builder path projects to the parsed path | **Keep** (landed; every example paired 2026-09-17). Drop the `pp_cursor` remark. |
| for_traversal.md | The `for` construct: readings, semantics, compiled form, activation, axioms | **Weed.** Keep §1-§3, §5.1-5.4, §6, §7, §9. Rewrite §4 and §5.2 to present tense on every engine. Fold three §11 rationale bullets into §3/§5. Delete status, §8, §10's option list, §11. |
| polytile.md | Tile grammar, structure, typing, semantics, skeleton, runtime, axioms | **Weed.** Keep §1-§9, §11. Fold §13's rules into §3.2, §5.4, §7.3. Delete status, revision log, §10, §12, §13. State the SRD 117 outcome (hole values, precompiled bodies on `Engine::default()` and why, memoized tuples, index-keyed binding, byte-identical writers, interning lifetime). |
| comprehension_forms.md | The comprehension algebra and its verification rule | **Rewrite** the preamble and §9.5 (the surfaces that exist). §8's `Ext` and `for base where` are fixed; the plan's invariants, the cutover's error ownership, and the gate's oracle are absorbed (§9.6, §9.7, §10.6). §15 stays: it records an open question, by decision 2026-09-17. |
| comprehension_implementation_plan.md | none | **Deleted** 2026-09-17: its stages diagram is forms §9.0, its invariants §9.6, its verification §9.8, its error contract §9.7, its ownership of the flat form §14.8. |
| comprehension_cutover_contact_surfaces.md | none | **Deleted** 2026-09-17: §5's one-resolution rule is expression_engine.md §3.2 (its mechanism was already there, not in the runtime doc); §6's ownership split is subcontext_construction.md §9; §7 is forms §14.8.1; §8 is forms §14.6; §9 is forms §9.7. |
| comprehension_migration_gate.md | none | **Deleted** 2026-09-17: the oracle in forms §10.6, the invariants in §9.6, the verification surfaces in §9.8, the parser and activation boundaries in §14.8–§14.9. |

Outside `docs/design`:

- **docs/README.md**: the Design paragraph stops naming plans and records; the
  comprehension plan links go; `documentation_links_resolve` catches the moves.
- **guides/compilation.md**: add "Choosing an engine" mapping `Engine::{Interpreter,
  Closures(p), Native(p)}` to the rows and stating the default; one naming scheme (P1/P2/P3).
- **guides/performance.md**: one reference table with one date line; construction column
  in `compile_with(Engine)` terms; the tile ladder keeps its command and one current pair.
  The number refresh itself is deferred until the machine is quiet.
- **guides/embedding.md**: engine definitions move to compilation.md; §13 leads with the
  logged default-engine compile and `plan()`; the §11 recap and the F-E4 quote go.
- **tutorials/illustrations.md**: two stale paths, one stale count, the nbrs sentence.
- **tutorials/polytile_tutorial.md**: SRD citations; §16 changes with group C.
- **README.md (root)**: one engines section; the documentation map reduced to the index
  link; the ladder table pointer (already applied).
- **README.md (crate)** and **src/lib.rs**: "Phase 1 (default)" and the hard-coded ratios
  go; the quick starts end on the default engine; the module list is completed.
- **docs/reference/nodes.md**: generated between markers (group M); host nodes move out.

## 4. What the documents must newly state

The engine work of the last month changed rules that no document states as a rule. Each
belongs in exactly one place; others cite it.

1. `Engine::default()` is `Native(Auto)` with the `jit` feature and `Closures(Auto)`
   without; the four entry points returning `Box<dyn Kernel>` with no engine argument
   build it, as does the binary without `--engine`, while the fourteen returning the
   concrete `PolydatKernel` are the interpreter by their return type (engines.md §1).
   *Stated 2026-09-18. The first wording, "every engine-less entry point and the binary
   build it", is false: fourteen engine-less entry points build the interpreter. The
   return type decides, not the absent argument.*
2. One evaluation rule on every engine: a step is current until an input in its
   provenance changes; nondeterministic never current; compile-constant folded at build;
   no step exempt; provenance mode is an optimization that never changes a result
   (runtime_model.md §3, engines.md §3).
3. Outputs are owned by their provenance: every output, immediate or by reference,
   stands until an input in its provenance is written; each state owns the storage
   behind its outputs and no storage belongs to a thread (runtime_model.md R4;
   compiled_handles.md §3). The cycle, the arena, the value table, and the nested
   kernel of the earlier design were removed for contradicting this.
4. The `Kernel` trait in those terms: the writes, `pull` runs one cone,
   `eval` runs every step, `invalidate_all` keeps inputs, `into_program` and the created
   kernel's starting state, `traverse` on every engine, the index-keyed calls,
   `cursor_schemas` on every kernel (runtime_model.md §6; the host-facing half is the
   [embedding guide](../guides/embedding.md), there being no separate kernel API doc).
5. The None rule, once: which nodes native code may see, how the planner and the batcher
   apply it, the runtime panic as tripwire (engines.md §3.3).
6. The failure contract: one message on every engine, `enrich_panic`, the tracker slot
   (engines.md §3.4, jit_boundary.md).
7. Compile events are engine-independent: `ConstantFolded`, `ExternWithoutDefault`, tile
   events, assertion counts are identical on every engine (engines.md §3.5).
8. `EnginePlan`: what each engine reports and that eligibility is observed, not inferred
   (engines.md §3.5, type_system_alignment.md §7).
9. Traversal activation on every engine: opening evaluates the comprehension in the body's
   scope through `Lookup`/`Layered`; `BodySource` carries settings; one program per engine
   per position; the cascade rule for shared wires (for_traversal.md §4-§5).
10. Tiles after SRD 117: hole values as the render node's inputs; bodies are kernels
    precompiled at construction on `Engine::default()` because a closure has no engine to
    ask; memoized tuples and their condition; integers and floats written directly,
    byte-identical to Rust's formatting; one body kernel per body program and engine,
    owned by the rendering state in the render step's scratch (polytile.md §7).
11. The hybrid is P3: segments are runs of consecutive native-eligible nodes of one
    lifecycle; a constant node never joins a non-constant segment; the pure tier is the
    differential oracle and Tier-1's carrier, not a host surface (engines.md §1, §8).
12. Cells on every engine: the publish contract (value, revision, intent bit), three
    creation sites, the compiled consumer's poll (cross_fiber_invalidation.md §1, §3.1, §5.2).


## 5. Decisions before the work starts

1. ~~**Shared wires cascaded into a `for` body**: attach the parent's cell (the lowering must
   declare the cascaded wire shared) or state that the cascade is a snapshot at open. The
   design promises the cell; the code snapshots. (F-K5a)~~ **Decided 2026-09-20: the
   snapshot, which both design documents already state.** The premise was wrong — neither
   `for_traversal.md` nor `scope_model.md` §4 promises the cell. A traversal materialises
   its tuples when it opens, so the body must capture the scope it opened against or the
   two would disagree about what that scope was. Pinned on every engine by
   `for_engines::a_body_reading_an_outer_shared_wire_sees_the_value_at_open`.
   design promises the cell; the code snapshots. (F-K5a)
2. **`tile_encode`**: keep as a host-callable library node with its lowering, or delete
   with `JitOp::TileEncode`. Nothing in the DSL emits it. (F-L5)
3. **Projection bodies' engine**: on `Engine::default()` always (today, with the H5
   reason) or on the enclosing kernel's engine as `for` bodies are. (F-L2 consequence 3)
4. **The wide-extern refusal**: close it (two-slot extern seeding) or state it as the one
   exception. (F-E13)
5. ~~**The pure native tier**: reduce to a raw kernel without the `Kernel` impl, or keep it
   as a fourth rung of the matrix. (F-E10)~~ **Decided 2026-09-20: keep it, and name it.**
   `Engine::PureNative(provenance)` makes the tier a host can ask for, which is the only
   way to ask whether a program is fully native — `Native` cannot answer, because it
   falls back to a closure and so cannot fail for that reason.
6. ~~**Plan B**: enforce (materialization returns an error under strict) or document the
   warn-and-continue behaviour. (F-C12)~~ **Decided 2026-09-20: document, and it already
   was.** `evaluation_model.md` carries the rule and the reason for it; no document ever
   claimed the hard error. Enforcing would refuse a scope whose `const` depends on
   resolution that is not ready until the workload runs, which is the case the
   warn-and-continue exists for.
7. ~~**Grammar documents**: merge four into `polydat_grammar.md` as proposed, or keep
   `grammar.md` as a formal appendix under test. (F-L1)~~ **Decided 2026-09-18: merge.**
   `polydat_grammar.md` now carries the type rules and G-axioms (§18) and the
   productions (§21), so the appendix material is under the same example test as the
   rest of the spec rather than in a second document that nothing checked.
8. ~~**The comprehension plan, cutover and gate documents**: delete after their invariants
   move into `comprehension_forms.md`. (section 3)~~ **Done 2026-09-17**, and with the
   language spec, the grammar appendix, and the tile plan deleted on 2026-09-18 the
   weeding map of section 3 is closed: every row is either kept-and-weeded or deleted
   with its disposition recorded.
