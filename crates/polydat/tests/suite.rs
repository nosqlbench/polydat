// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The integration suite as one test binary. Every file under `tests/`
//! is a module here, so the suite links once against the library
//! rather than once per file (sixty-odd links after every library
//! change, the largest cost of the gate); nextest runs each test in
//! its own process regardless. The five files that define nodes of
//! their own (`ext_tiers`, `handle_tiers`, `kernel_api`,
//! `polydat_node_macro`, `srd80b_invariant`) stay binaries of their
//! own: a node registers into the inventory at link time, and the
//! tests here that walk the registry must see the library alone. A
//! test in the suite is addressed by its module:
//!
//! ```sh
//! cargo nextest run -p polydat --test suite engine_parity::
//! ```

mod common;

#[path = "adapter_catalog_invariants.rs"]
mod adapter_catalog_invariants;
#[path = "adversarial_polydat.rs"]
mod adversarial_polydat;
#[path = "argv_assignment.rs"]
mod argv_assignment;
#[path = "bench_graphs.rs"]
mod bench_graphs;
#[path = "binary_run.rs"]
mod binary_run;
#[path = "body_carrier.rs"]
mod body_carrier;
#[path = "compile_events.rs"]
mod compile_events;
#[path = "core_with_library.rs"]
mod core_with_library;
#[path = "cursor_tiers.rs"]
mod cursor_tiers;
#[path = "declared_wires.rs"]
mod declared_wires;
#[path = "doc_examples_test.rs"]
mod doc_examples_test;
#[path = "end_to_end.rs"]
mod end_to_end;
#[path = "engine_ladder_equivalence.rs"]
mod engine_ladder_equivalence;
#[path = "engine_parity.rs"]
mod engine_parity;
#[path = "entry_points.rs"]
mod entry_points;
#[path = "equivalence_corners.rs"]
mod equivalence_corners;
#[path = "equivalence_harness.rs"]
mod equivalence_harness;
#[path = "examples_test.rs"]
mod examples_test;
#[path = "failure_parity.rs"]
mod failure_parity;
#[path = "float_text.rs"]
mod float_text;
#[path = "for_compile.rs"]
mod for_compile;
#[path = "for_engines.rs"]
mod for_engines;
#[path = "for_flatten.rs"]
mod for_flatten;
#[path = "for_invariance.rs"]
mod for_invariance;
#[path = "for_producers.rs"]
mod for_producers;
#[path = "for_runtime.rs"]
mod for_runtime;
#[path = "for_sampling.rs"]
mod for_sampling;
#[path = "for_syntax.rs"]
mod for_syntax;
#[path = "function_coverage.rs"]
mod function_coverage;
#[path = "fuzz_conversions.rs"]
mod fuzz_conversions;
#[path = "fuzz_for_syntax.rs"]
mod fuzz_for_syntax;
#[path = "fuzz_tile_syntax.rs"]
mod fuzz_tile_syntax;
#[path = "fuzz_type_adapters.rs"]
mod fuzz_type_adapters;
#[path = "guide_output.rs"]
mod guide_output;
#[path = "handle_boundaries.rs"]
mod handle_boundaries;
#[path = "hybrid_test.rs"]
mod hybrid_test;
#[path = "ir_end_to_end.rs"]
mod ir_end_to_end;
#[path = "local_modules.rs"]
mod local_modules;
#[path = "nodes_reference.rs"]
mod nodes_reference;
#[path = "optimizer_worked_examples.rs"]
mod optimizer_worked_examples;
#[path = "polydat_examples_test.rs"]
mod polydat_examples_test;
#[path = "polydat_files_test.rs"]
mod polydat_files_test;
#[path = "predicate_analyzer_soundness.rs"]
mod predicate_analyzer_soundness;
#[path = "resource_bounds_verification.rs"]
mod resource_bounds_verification;
#[path = "rustdoc_examples.rs"]
mod rustdoc_examples;
#[path = "sampling_test.rs"]
mod sampling_test;
#[path = "scope_composition.rs"]
mod scope_composition;
#[path = "shared_tiers.rs"]
mod shared_tiers;
#[path = "slot_state_axioms.rs"]
mod slot_state_axioms;
#[path = "source_tests.rs"]
mod source_tests;
#[path = "spec_section_11_worked_examples.rs"]
mod spec_section_11_worked_examples;
#[path = "spec_surface_parity.rs"]
mod spec_surface_parity;
#[path = "surfaces_independence.rs"]
mod surfaces_independence;
#[path = "tier_concurrency_bench.rs"]
mod tier_concurrency_bench;
#[path = "tile_host_surfaces.rs"]
mod tile_host_surfaces;
#[path = "tile_projections.rs"]
mod tile_projections;
#[path = "tile_render.rs"]
mod tile_render;
#[path = "tile_structural.rs"]
mod tile_structural;
#[path = "tile_syntax.rs"]
mod tile_syntax;
#[path = "tile_typing.rs"]
mod tile_typing;
#[path = "ui.rs"]
mod ui;
#[path = "variadic_lowering.rs"]
mod variadic_lowering;
#[path = "vector_set_ops.rs"]
mod vector_set_ops;
#[path = "vectordata_concurrency.rs"]
mod vectordata_concurrency;
#[path = "vectordata_integration.rs"]
mod vectordata_integration;
#[path = "volatile_is_never_current.rs"]
mod volatile_is_never_current;
#[path = "wire_type_fidelity.rs"]
mod wire_type_fidelity;
