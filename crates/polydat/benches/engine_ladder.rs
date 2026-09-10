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
                black_box(p1.pull_by_index(output).as_u64());
            }
            cycle = cycle.wrapping_add(1);
        });
    });

    let mut p2 = match assembler().try_compile_raw() {
        Ok(kernel) => kernel,
        Err(_) => panic!("every engine-ladder node must support P2"),
    };
    let p2_outputs = OUTPUTS.map(|name| p2.resolve_output(name).expect("P2 output must resolve"));
    group.bench_function("p2_closures", |b| {
        let mut cycle = 1u64;
        b.iter(|| {
            p2.eval(&[cycle, TENANT_SEED, OPERATION_SEED]);
            for &output in &p2_outputs {
                black_box(p2.get_slot(output));
            }
            cycle = cycle.wrapping_add(1);
        });
    });

    #[cfg(feature = "jit")]
    {
        let mut p3 = assembler()
            .try_compile_jit_raw()
            .expect("every engine-ladder node must support P3");
        let p3_outputs =
            OUTPUTS.map(|name| p3.resolve_output(name).expect("P3 output must resolve"));
        group.bench_function("p3_native", |b| {
            let mut cycle = 1u64;
            b.iter(|| {
                p3.eval(&[cycle, TENANT_SEED, OPERATION_SEED]);
                for &output in &p3_outputs {
                    black_box(p3.get_slot(output));
                }
                cycle = cycle.wrapping_add(1);
            });
        });

        let mut pure = assembler()
            .try_compile_pure_jit_raw()
            .expect("every engine-ladder node lowers to native code");
        let pure_outputs = OUTPUTS.map(|name| {
            pure.resolve_output(name)
                .expect("pure native output must resolve")
        });
        group.bench_function("pure_native", |b| {
            let mut cycle = 1u64;
            b.iter(|| {
                pure.eval(&[cycle, TENANT_SEED, OPERATION_SEED]);
                for &output in &pure_outputs {
                    black_box(pure.get_slot(output));
                }
                cycle = cycle.wrapping_add(1);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_engine_ladder);
criterion_main!(benches);
