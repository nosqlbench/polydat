// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A focused comparison of the same typed graph through P1, P2, and P3.
//!
//! One iteration advances the cycle, evaluates the complete eleven-node graph,
//! and reads its four named outputs. Compilation and output-name resolution are
//! deliberately outside the timed region.

use std::time::Duration;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use polydat::JitMode;
use polydat::compile::assembly::PolydatAssembler;
use polydat::dsl::compile::compile_polydat_to_assembler;

const GRAPH: &str = include_str!("../examples/engine_ladder.polydat");
const OUTPUTS: [&str; 4] = ["account_id", "shard", "payload_class", "event_token"];
const TENANT_SEED: u64 = 0x5445_4e41_4e54;
const OPERATION_SEED: u64 = 0x4f50_4552_4154_494f;

fn assembler() -> PolydatAssembler {
    compile_polydat_to_assembler(GRAPH).expect("engine-ladder graph must assemble")
}

fn bench_engine_ladder(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_ladder");
    group.throughput(Throughput::Elements(1));
    group.sample_size(60);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(3));

    let mut p1_asm = assembler();
    p1_asm.set_jit_mode(JitMode::Off);
    let mut p1 = p1_asm.compile().expect("P1 compilation must succeed");
    let p1_outputs = OUTPUTS.map(|name| {
        p1.program()
            .output_index(name)
            .expect("P1 output must resolve")
    });
    group.bench_function("p1_interpreter", |b| {
        let mut cycle = 1u64;
        b.iter(|| {
            p1.set_inputs(&[cycle, TENANT_SEED, OPERATION_SEED]);
            for &output in &p1_outputs {
                black_box(p1.pull_ref_at(output).as_u64());
            }
            cycle = cycle.wrapping_add(1);
        });
    });

    // Each compiled tier is measured twice, from one kernel type and
    // one contract, held two ways.
    //
    // `<tier>` names the kernel by its own type, so the call is
    // monomorphized: what a host pays to run the program. `<tier>_dyn`
    // holds the same kernel as `Box<dyn SlotKernel>`, the way
    // `compile_slots` and every engine-by-name entry point hand one
    // back: what a host pays to run the program *and* to have chosen
    // the engine at run time. The gap between the two is the price of
    // that choice, and it is a number worth watching rather than
    // rediscovering — measured once at about a fifth of the native
    // tier's per-cycle cost.
    //
    // The monomorphized column is why this bench needs `bench-tiers`:
    // naming a kernel type is the one thing the normative surface
    // deliberately does not let a caller do.
    macro_rules! tier {
        ($name:literal, $typed:expr, $boxed:expr) => {
            let mut typed = $typed;
            let slots = OUTPUTS.map(|n| typed.resolve_output(n).expect("output must resolve"));
            group.bench_function($name, |b| {
                let mut cycle = 1u64;
                b.iter(|| {
                    typed.eval(&[cycle, TENANT_SEED, OPERATION_SEED]);
                    for &slot in &slots {
                        black_box(typed.get_slot(slot));
                    }
                    cycle = cycle.wrapping_add(1);
                });
            });
            let mut boxed = $boxed;
            let slots = OUTPUTS.map(|n| boxed.resolve_output(n).expect("output must resolve"));
            group.bench_function(concat!($name, "_dyn"), |b| {
                let mut cycle = 1u64;
                b.iter(|| {
                    boxed.eval_at(&[cycle, TENANT_SEED, OPERATION_SEED]);
                    for &slot in &slots {
                        black_box(boxed.get_slot(slot));
                    }
                    cycle = cycle.wrapping_add(1);
                });
            });
        };
    }

    tier!(
        "p2_closures",
        assembler()
            .compile_closures_raw()
            .expect("every engine-ladder node must support P2"),
        assembler()
            .compile_slots(polydat::Engine::Closures(polydat::Provenance::Raw))
            .expect("every engine-ladder node must support P2")
    );

    #[cfg(feature = "jit")]
    {
        tier!(
            "p3_native",
            assembler()
                .compile_native_raw()
                .expect("every engine-ladder node must support P3"),
            assembler()
                .compile_slots(polydat::Engine::Native(polydat::Provenance::Raw))
                .expect("every engine-ladder node must support P3")
        );
        tier!(
            "pure_native",
            assembler()
                .compile_pure_native_raw()
                .expect("every engine-ladder node lowers to native code"),
            assembler()
                .compile_slots(polydat::Engine::PureNative(polydat::Provenance::Raw))
                .expect("every engine-ladder node lowers to native code")
        );
    }

    group.finish();
}

/// The conversion adapters through the compiled tiers.
///
/// The ladder graph above is all `u64` and inserts no adapter, so it
/// cannot see how conversions are lowered. This graph is conversions:
/// checked narrowings, `u64` to and from `f32`, and widenings, over a
/// value small enough that every narrowing passes, so it runs the inline
/// domain of each conversion (engines.md §7.1). The native tier derives
/// those lowerings from the conversion table and the two port types; a
/// conversion that fell back to a slot call would show here as the
/// closure tier's time. The closure rung reads the same graph through
/// code the native lowering does not touch, so it is the canary.
const CONVERSIONS: &str = "input cycle: u64
m := mod(cycle, 1000)
a := __u64_to_u32(m)
b := __u32_to_i32(a)
c := __i32_to_i64(b)
f := __u64_to_f32(m)
g := __f32_to_f64(f)
h := __u64_to_i64(m)
k := __i64_to_u64(h)
w := __u32_to_u64(a)
";
const CONVERSION_OUTPUTS: [&str; 5] = ["c", "g", "k", "w", "f"];

fn bench_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversions");
    group.throughput(Throughput::Elements(1));
    group.sample_size(60);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(3));
    let asm = || compile_polydat_to_assembler(CONVERSIONS).expect("conversion graph must assemble");

    macro_rules! rung {
        ($name:literal, $kernel:expr) => {
            let mut k = $kernel;
            let slots =
                CONVERSION_OUTPUTS.map(|n| k.resolve_output(n).expect("output must resolve"));
            group.bench_function($name, |b| {
                let mut cycle = 1u64;
                b.iter(|| {
                    k.eval(&[cycle]);
                    for &slot in &slots {
                        black_box(k.get_slot(slot));
                    }
                    cycle = cycle.wrapping_add(1);
                });
            });
        };
    }
    rung!(
        "p2_closures",
        asm()
            .compile_closures_raw()
            .expect("every conversion supports P2")
    );
    #[cfg(feature = "jit")]
    rung!(
        "p3_native",
        asm()
            .compile_native_raw()
            .expect("every conversion supports P3")
    );
    group.finish();
}

criterion_group!(benches, bench_engine_ladder, bench_conversions);
criterion_main!(benches);
