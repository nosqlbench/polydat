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

Recorded 2026-09-11. Status as of 2026-09-16: the step-1 fixes (F-C1, F-N3, F-N11, F-E4,
F-E5, F-E13 stated, F-C12 documented, F-K2), F-K1, F-E3, F-E6, groups B (in part), C, I,
and the `Lookup` half of group E have landed, and the runtime, node library, and language
have since been split into their own crates; section 3's weeding pass landed except the
deletions; groups A, D (in part), F, G, H, J, K, L, M remain open. Of section 5's
decisions, 4 and 6 were taken by stating the behaviour as it is; the rest remain open.
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
| F-E3 | engines | S | The P3 kernel's provenance mask is one `u64`; a program whose inputs span 64 or more slots (ten cursors plus a coordinate) shifts out of range at build. The closure tier and pure tier use `ProvMask`. |
| F-E7 | engines | M | The None rule has two predicates: the cone planner's (a None-tolerant node joins a cone only when every input is intra-cone) and the segment batcher's (a node downstream of an extern unset at build is a closure). A host that clears an extern after build reaches the runtime panic in a native segment where the interpreter produces a value. |
| F-C1 | compiler | XS | `pragma strict_values` and `strict_types` are honoured on every compiled engine and on the logged interpreter path, and silently dropped on the plain interpreter path. |
| F-H3 | host | S | The assembler's strict checks (implicit adapters, config-wire violations) run only in `compile_strict`, which only the interpreter path reaches. `strict: true` on a compiled engine accepts what the interpreter rejects. |
| F-C4 | compiler | S | Deferred cursor extents and the retained AST are resolved on the interpreter path only; `cursor_schemas()` reports `extent: None` on P2/P3 for a computed extent. |
| F-H4 | host | S | The two diagnostic entry points build a different program from the one they diagnose (one rejects `for`, one skips pragmas and source text). |
| F-E5 | engines | S | `into_program` on a compiled kernel clones host-set extern values into every created kernel; the interpreter's starts from defaults. The trait does not say which. |
| F-E4 | engines | S | `Native(Auto)` never runs the provenance selector, so the default engine always pays push bookkeeping; `engine()` reports a mode the host did not ask for. |
| F-L3 | dsl | S | A module defined in the program resolves inside a `for` or projection body only because the child compiler re-reads the parent file from disk; compiled from a string, the body cannot see it. |
| F-K5a | kernel | S | A `for` body that reads an outer shared wire sees a snapshot at open time, not the cell; the design promises the cell. Untested on any engine. |
| F-E13 | engines | XS/M | One refusal remains on compiled engines (an extern wider than one slot) while the documents and the trait doc say every engine accepts every program. State it or close it. |
| F-N3 | library | XS | `file_line_at` and `regex_replace` are registered twice (legacy sig and macro); which diagnostic a bad pattern produces depends on link order. |
| F-N11 | library | XS | Link order leaks into the pinned parity matrix text; adding a module reorders rows and fails the test with empty diffs. |
| F-E6 | engines | S | The public `P2Engine`/`P3Engine` selectors hand a host a kernel that never resets the arena. No in-tree caller; deletion. |
| F-H7 | host | M | Every binary command compiles the program twice (interpreter for names and manifest, run engine for the run); `run --emit` four times. |
| F-N8 | library | S | `dynamic_weighted_select` re-parses its spec every evaluation; `random_string` re-parses its charset per call; every `Const<Vec<C>>` node clones its list per cycle. |
| F-N6 | library | S | Const-argument constraints cannot be declared through the macro; the substitutes are panics caught at construction and the legacy `validate_node`. |
| F-C12 | compiler | XS | Plan B (a scope-init const that yields None is a hard error) is documented and not enforced; the materializer warns and continues. Doc or code, one of them changes. |

### 1.2 Architectural

Grouped where the findings are one change.

| Group | Findings | Size | One change |
|---|---|---|---|
| A. One compiled core | F-E1, F-E2, F-E10, F-E8, F-E11 | L | The closure tier and the hybrid kernel are one engine written twice (same thirty fields, same twenty methods; the no-jit hybrid is a third copy). Eleven provenance-mode types carry one flag. One core, one step enum `{Native, Closure, Copy}`, provenance as a field with an optional cone guard; the closure engine is the same builder with every node classified fallback. The pure tier keeps a raw kernel for the differential and Tier-1 and drops the rest. Net deletion around 1,500 lines. Gated by the ladder. |
| B. One entry point | F-H1, F-H2, F-C3, F-C2, F-C15, F-H9 | M | Twenty-four `compile_*` functions, six of them full pipelines with different settings; twenty-two assembler constructors with a fallback that returns an empty kernel or panics. One `Compiler::from_options`, one `compile_file` shared by the interpreter and compiled paths, four public conveniences, one `KernelError`; everything else a deprecated two-line wrapper. |
| C. Cones belong to the interpreter engine | F-H5, F-H6 | S | `JitMode` is a process global that changes what `Engine::Interpreter` builds. Make it part of the variant (`Interpreter { cones }`), delete the global; the binary's `--engine off/auto/force` becomes `--engine interpreter/closures/native` with `--cones` for the interpreter. |
| D. One trait, one meaning | F-K2, F-K3, F-H8, F-E9, F-K6 | M | `invalidate_all` means two things; three write rules on the interpreter (heal, reject, widen) and a fourth on compiled kernels; inherent methods shadow the trait with different return types; four hidden construction hooks on the host trait; cycle ownership spelled four ways with no assertion. One write rule at every engine's `set_input`; one `invalidate_all`; borrowing readers renamed; internals behind a sealed trait; cycle ownership stated as an R-axiom with a debug tripwire. |
| E. One binder for child kernels | F-K5, F-K11, F-K12, F-L9, F-C6 | L | Subcontext materialization (interpreter only, cells and transit) and traversal activation (any engine, snapshot) are two mechanisms for one idea. Express binding over `dyn Kernel` plus a `Lookup` view; streams remember the engine that opened them; `Lookup` implemented for a layered constant view so opening a traversal allocates no interpreter state; the kernel-bound evaluation surfaces take `&dyn Lookup`. Part (b), a `ParentView` instead of a transient parent kernel, is M on its own and removes F-K1's second trigger. |
| F. One body carrier | F-L2, F-L4, F-L5, F-C13 | M | Tile projection bodies and `for` bodies are two lowerings that have diverged (settings, engine choice, error path, three compiles per tile body). Tile lowering builds a `PolydatFile` plus `BodySource` as traversals do; bodies live in a program-owned table with `program_on(engine)`; the compiler hands the node an `Arc<TileProgram>` through an opaque const instead of JSON text; `tile_encode` is retired or documented as host-callable. |
| G. One macro kit | F-N1, F-N2, F-N7 | L | Three closure kits with three eligibility rules over six shape vocabularies; the Rust-type-to-slot mapping in four places, two dead; native lowering a by-name match with a positional constant contract and eleven names that are not nodes. One kit from one argument-read enum; slot I/O on the `Wire` trait; a node declares its lowering the way it declares its closure. |
| H. One classifier, one tripwire | F-C5, F-E8, F-E14, F-K6 | S | Lifecycle classification is duplicated in the cone extractor and again in the hybrid batcher; nondeterminism has four encodings; H4's validation is written three times; `EnginePlan` on the interpreter is computed by node-name prefix. |
| I. Legacy scaffolding out | F-N4, F-N5, F-N10, F-N9 | M | About 840 lines of empty `register_nodes!` bodies; `limit` and fourteen vector nodes hand-written where the attribute would do; `instantiate` with no user; placeholder port types in the static graph against the alignment axiom. |
| J. Compiler tidy | F-C7, F-C8, F-C9, F-C16 | S each | Error variants reconstructed from message strings; a hand losslessness table disagreeing with the catalog; a dead extern-type inference; 128-bit types documented as interpreter-only with no parity case. |
| K. One grammar | F-L1, F-L7, F-L8 | M | Four grammar documents, two claiming sole authority; the tested one omits `for`, `tile`, the `if` block and `shared x: T`. One normative, test-checked file, with a test that fails when an AST variant has no example. The comprehension text front end renamed from "legacy" and `custom(fn)` removed. |
| L. Host tiles as a transform | F-H12 | S | The `_with_tiles` entry points and `polydat::tile`'s exposed internals become a public AST transform, per the standing rule that host features are program transforms. |
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
| tile_native_rendering.md | none | **Delete** after: the landed forms and the two corrections with reasons move to polytile.md §7; the helper-reads-views rule to compiled_handles §6; the float writer contract to polytile.md; the ladder cases and one paired table to the performance guide. |
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
| grammar.md | none | **Delete** after its G-axioms with rationale and its type rules merge into polydat_grammar.md. |
| language_spec.md | none | **Delete** after Conditional Selection, the `as` paragraph, the dispatch table, and the pipeline move to polydat_grammar.md and graph_compiler.md. |
| polydat_grammar_programmatic.md | Proof that the builder path projects to the parsed path | **Keep** (landed; every example paired 2026-09-17). Drop the `pp_cursor` remark. |
| for_traversal.md | The `for` construct: readings, semantics, compiled form, activation, axioms | **Weed.** Keep §1-§3, §5.1-5.4, §6, §7, §9. Rewrite §4 and §5.2 to present tense on every engine. Fold three §11 rationale bullets into §3/§5. Delete status, §8, §10's option list, §11. |
| polytile.md | Tile grammar, structure, typing, semantics, skeleton, runtime, axioms | **Weed.** Keep §1-§9, §11. Fold §13's rules into §3.2, §5.4, §7.3. Delete status, revision log, §10, §12, §13. State the SRD 117 outcome (hole values, precompiled bodies on `Engine::default()` and why, memoized tuples, index-keyed binding, byte-identical writers, interning lifetime). |
| comprehension_forms.md | The comprehension algebra and its verification rule | **Rewrite** the preamble, §8 (Polydat's own surface, fix `Ext` and `for base where`), §9.5 (the surfaces that exist); absorb the plan's six invariants, the cutover's error-ownership table, the gate's oracle. Delete §15. |
| comprehension_implementation_plan.md | none | **Deleted** 2026-09-17: its stages diagram is forms §9.0, its invariants §9.6, its verification §9.8, its error contract §9.7, its ownership of the flat form §14.8. |
| comprehension_cutover_contact_surfaces.md | none | **Delete** (open) after §5 moves to the runtime doc and §8 to forms §14. |
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
   without; every engine-less entry point and the binary build it (engines.md §1).
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
   `cursor_schemas` on every kernel (runtime_model.md, kernel API doc).
5. The None rule, once: which nodes native code may see, how the planner and the batcher
   apply it, the runtime panic as tripwire (engines.md §6).
6. The failure contract: one message on every engine, `enrich_panic`, the tracker slot
   (engines.md §6, jit_boundary.md).
7. Compile events are engine-independent: `ConstantFolded`, `ExternWithoutDefault`, tile
   events, assertion counts are identical on every engine (engines.md §7).
8. `EnginePlan`: what each engine reports and that eligibility is observed, not inferred
   (engines.md §7, type_system_alignment.md §7).
9. Traversal activation on every engine: opening evaluates the comprehension in the body's
   scope through `Lookup`/`Layered`; `BodySource` carries settings; one program per engine
   per position; the cascade rule for shared wires (for_traversal.md §5).
10. Tiles after SRD 117: hole values as the render node's inputs; bodies are kernels
    precompiled at construction on `Engine::default()` because a closure has no engine to
    ask; memoized tuples and their condition; integers and floats written directly,
    byte-identical to Rust's formatting; one body kernel per body program and engine,
    owned by the rendering state in the render step's scratch (polytile.md §7).
11. The hybrid is P3: segments are runs of consecutive native-eligible nodes of one
    lifecycle; a constant node never joins a non-constant segment; the pure tier is the
    differential oracle and Tier-1's carrier, not a host surface (engines.md §1-§2).
12. Cells on every engine: the publish contract (value, revision, intent bit), three
    creation sites, the compiled consumer's poll (cross_fiber_invalidation.md §3, §5).

## 5. Decisions before the work starts

1. **Shared wires cascaded into a `for` body**: attach the parent's cell (the lowering must
   declare the cascaded wire shared) or state that the cascade is a snapshot at open. The
   design promises the cell; the code snapshots. (F-K5a)
2. **`tile_encode`**: keep as a host-callable library node with its lowering, or delete
   with `JitOp::TileEncode`. Nothing in the DSL emits it. (F-L5)
3. **Projection bodies' engine**: on `Engine::default()` always (today, with the H5
   reason) or on the enclosing kernel's engine as `for` bodies are. (F-L2 consequence 3)
4. **The wide-extern refusal**: close it (two-slot extern seeding) or state it as the one
   exception. (F-E13)
5. **The pure native tier**: reduce to a raw kernel without the `Kernel` impl, or keep it
   as a fourth rung of the matrix. (F-E10)
6. **Plan B**: enforce (materialization returns an error under strict) or document the
   warn-and-continue behaviour. (F-C12)
7. **Grammar documents**: merge four into `polydat_grammar.md` as proposed, or keep
   `grammar.md` as a formal appendix under test. (F-L1)
8. **The comprehension plan, cutover and gate documents**: delete after their invariants
   move into `comprehension_forms.md`. (section 3)
