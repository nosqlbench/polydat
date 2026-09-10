// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Local study for offset-stamped `i32x4` scalar-flow promotion.
//!
//! The scalar arm executes the same wrapping affine stages as one P3 scalar
//! Polydat graph evaluation per item. The SIMD arm synthesizes four affine
//! ordinal values, executes an explicit register-typed P3 graph once, and
//! drains its stamped packet through arbitrarily sized scalar bursts.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

use polydat::ast::Bits128;
use polydat::compile::jit::JitKernelRaw;
use polydat::iteration::simd_ordinal::{AffineI32Source, OrdinalI32x4Stream, PacketFragmentPolicy};

const FACTOR: u64 = 3;
const ADDEND: u64 = 17;

fn scalar_graph_source(depth: usize) -> String {
    let mut source = String::from(
        "input scalar: u64\n\
         input factor: u64\n\
         input addend: u64\n",
    );
    let mut previous = "scalar".to_string();
    for stage in 0..depth {
        let product = format!("product_{stage}");
        let output = if stage + 1 == depth {
            "out".to_string()
        } else {
            format!("stage_{stage}")
        };
        source.push_str(&format!(
            "{product} := u64_mul({previous}, factor)\n\
             {output} := u64_add({product}, addend)\n"
        ));
        if stage + 1 < depth {
            previous = output;
        }
    }
    source
}

fn vector_graph_source(depth: usize) -> String {
    let mut source = String::from(
        "input packet: reg_i32x4\n\
         input factor: u64\n\
         input addend: u64\n\
         factor_v := reg_splat_i32(factor)\n\
         addend_v := reg_splat_i32(addend)\n",
    );
    let mut previous = "packet".to_string();
    for stage in 0..depth {
        let product = format!("product_{stage}");
        let output = if stage + 1 == depth {
            "out".to_string()
        } else {
            format!("stage_{stage}")
        };
        source.push_str(&format!(
            "{product} := reg_mul_i32({previous}, factor_v)\n\
             {output} := reg_add_i32({product}, addend_v)\n"
        ));
        if stage + 1 < depth {
            previous = output;
        }
    }
    source
}

fn compile_raw(source: &str) -> JitKernelRaw {
    polydat::dsl::compile::compile_polydat_to_assembler(source)
        .expect("assemble benchmark graph")
        .try_compile_pure_jit_raw()
        .expect("benchmark graph must lower completely to P3")
}

fn scalar_oracle(depth: usize, mut value: i32) -> i32 {
    for _ in 0..depth {
        value = value
            .wrapping_mul(FACTOR as i32)
            .wrapping_add(ADDEND as i32);
    }
    value
}

fn bench_scalar_p3(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    depth: usize,
    burst: usize,
) {
    let mut kernel = compile_raw(&scalar_graph_source(depth));
    let output_slot = kernel.resolve_output("out").unwrap();
    let mut ordinal = 0u64;
    let mut output = vec![0i32; burst];

    group.throughput(Throughput::Elements(burst as u64));
    group.bench_with_input(BenchmarkId::new("scalar_p3", burst), &burst, |b, _| {
        b.iter(|| {
            for value in &mut output {
                kernel.eval(&[ordinal, FACTOR, ADDEND]);
                *value = kernel.get_slot(output_slot) as u32 as i32;
                ordinal = ordinal.wrapping_add(1);
            }
            black_box(&output);
        });
    });
}

fn bench_offset_simd_p3(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    depth: usize,
    burst: usize,
) {
    let mut kernel = compile_raw(&vector_graph_source(depth));
    let output_slot = kernel.resolve_output("out").unwrap();
    let vector = move |lanes: [i32; 4]| {
        let input = Bits128::from_lanes_i32(lanes);
        kernel.eval(&[input.0[0], input.0[1], FACTOR, ADDEND]);
        Bits128([
            kernel.get_slot(output_slot),
            kernel.get_slot(output_slot + 1),
        ])
        .lanes_i32()
    };
    let scalar = move |value| scalar_oracle(depth, value);
    let mut stream = OrdinalI32x4Stream::new(
        AffineI32Source::new(1, 0, 0, 0, 1),
        0..u64::MAX,
        0,
        0,
        vector,
        scalar,
    )
    .unwrap()
    .with_fragment_policy(PacketFragmentPolicy::PadAndVectorize);
    let mut output = vec![0i32; burst];

    group.throughput(Throughput::Elements(burst as u64));
    group.bench_with_input(BenchmarkId::new("offset_simd_p3", burst), &burst, |b, _| {
        b.iter(|| {
            let written = stream.drain_into(&mut output);
            debug_assert_eq!(written, output.len());
            black_box(&output);
        });
    });
}

fn bench_simd_autopromotion(c: &mut Criterion) {
    for depth in [1usize, 4, 16] {
        let mut group = c.benchmark_group(format!("simd_autopromotion/depth_{depth}"));
        group.sample_size(30);
        for burst in [1usize, 3, 4, 16, 256] {
            bench_scalar_p3(&mut group, depth, burst);
            bench_offset_simd_p3(&mut group, depth, burst);
        }
        group.finish();
    }
}

criterion_group!(benches, bench_simd_autopromotion);
criterion_main!(benches);
