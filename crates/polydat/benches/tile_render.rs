// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The tile ladder (docs/guides/performance.md, "The tile ladder"): one
//! document rendered through P1, P2, P3, and pure native code, in five
//! cases. `reading` is the toy test definition's reading with no tile,
//! its eight wires read directly, so the render cost of every other
//! case is that case less this one. `one_hole` is the floor: one
//! numeric hole in a two-byte skeleton. `flat` is the reading document
//! without its projection, so the cost is encoding and copying per
//! hole; `projected` is the document as written, adding a four-tuple
//! projection with a formatted hole; `wide` is the flat document with
//! twenty holes of three types. One iteration advances the cycle,
//! evaluates the graph, and reads the outputs, which on the compiled
//! engines copies a document out of the arena as any host read does.
//! Compilation is outside the timed region.

use std::time::Duration;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use polydat::JitMode;
use polydat::compile::assembly::PolydatAssembler;
use polydat::dsl::compile::compile_polydat_to_assembler;

/// The reading of `examples/toy_test_definition.polydat`, self-contained.
const READING: &str = r#"
input cycle: u64
(tenant, device, reading) := mixed_radix(cycle, 20, 50, 0)
tenant_id    := mod(hash(tenant), 1000000)
device_key   := interleave(tenant, device)
device_id    := hashed_uuid(device_key)
device_kind  := weighted_strings(device_key, "sensor:0.7;gateway:0.2;controller:0.1")
reading_seed := hash(interleave(device_key, reading))
temp_c       := unit_interval(reading_seed) * 30.0 + 10.0
humidity     := unit_interval(hash(reading_seed)) * 60.0 + 20.0
status       := select_str(u64_gt(mod(reading_seed, 10), 8), "warn", "ok")
ts           := 1700000000000 + reading * 1000
flagged      := u64_gt(mod(reading_seed, 4), 2)
"#;

const READING_OUTPUTS: [&str; 8] = [
    "tenant_id",
    "device_id",
    "device_kind",
    "ts",
    "temp_c",
    "humidity",
    "status",
    "flagged",
];

const ONE_HOLE: &str = r#"
input cycle: u64
k := mod(hash(cycle), 1000)
tile doc : json := {"k": ${k}}
"#;

const FLAT_TILE: &str = r#"
tile doc : json := {
    "meta": { "schema": 3, "source": "polydat", "units": { "temp": "C", "rh": "%" } },
    "tenant": ${tenant_id},
    "device": ${device_id},
    "kind": ${device_kind},
    "ts": ${ts},
    "reading": { "temp": ${temp_c | .2}, "rh": ${humidity | .1}, "status": ${status} },
    "flagged": ${flagged: bool}
}
"#;

const PROJECTED_TILE: &str = r#"
tile doc : json := {
    "meta": { "schema": 3, "source": "polydat", "units": { "temp": "C", "rh": "%" } },
    "tenant": ${tenant_id},
    "device": ${device_id},
    "kind": ${device_kind},
    "ts": ${ts},
    "reading": { "temp": ${temp_c | .2}, "rh": ${humidity | .1}, "status": ${status} },
    "samples": [ @for s in 0..4 { { "n": ${s}, "temp": ${temp_c + s | .2} } } ],
    "flagged": ${flagged: bool}
}
"#;

/// Twenty holes: ten `u64`, five formatted `f64`, five `str`.
fn wide_source() -> String {
    let mut src = String::from(READING);
    let mut holes = Vec::new();
    for i in 0..20u64 {
        match i % 4 {
            0 | 1 => {
                src.push_str(&format!("k{i} := mod(hash(cycle + {i}), 1000)\n"));
                holes.push(format!("\"k{i}\": ${{k{i}}}"));
            }
            2 => {
                src.push_str(&format!(
                    "k{i} := unit_interval(hash(cycle + {i})) * 100.0\n"
                ));
                holes.push(format!("\"k{i}\": ${{k{i} | .3}}"));
            }
            _ => {
                src.push_str(&format!(
                    "k{i} := \"item-{{mod(hash(cycle + {i}), 1000)}}\"\n"
                ));
                holes.push(format!("\"k{i}\": ${{k{i}}}"));
            }
        }
    }
    src.push_str(&format!(
        "tile doc : json := {{ \"meta\": {{ \"schema\": 3 }}, {} }}\n",
        holes.join(", ")
    ));
    src
}

fn assembler(src: &str) -> PolydatAssembler {
    compile_polydat_to_assembler(src).expect("tile-ladder program must assemble")
}

/// The byte length of a value: a document's text or a scalar's width,
/// which is what a host reading it would touch.
fn weight(v: &polydat::ast::Value) -> usize {
    match v {
        polydat::ast::Value::Str(s) => s.len(),
        _ => 8,
    }
}

fn bench_case(c: &mut Criterion, case: &str, src: &str, outputs: &[&str]) {
    let mut group = c.benchmark_group(format!("tile_render/{case}"));
    group.throughput(Throughput::Elements(1));
    group.sample_size(60);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(3));

    let mut p1_asm = assembler(src);
    p1_asm.set_jit_mode(JitMode::Off);
    let mut p1 = p1_asm.compile().expect("P1 compilation must succeed");
    let p1_outputs: Vec<usize> = outputs
        .iter()
        .map(|name| {
            p1.program()
                .output_index(name)
                .expect("P1 output must resolve")
        })
        .collect();
    group.bench_function("p1_interpreter", |b| {
        let mut cycle = 1u64;
        b.iter(|| {
            p1.set_inputs(&[cycle]);
            for &output in &p1_outputs {
                black_box(weight(p1.pull_ref_at(output)));
            }
            cycle = cycle.wrapping_add(1);
        });
    });

    let mut p2 =
        match assembler(src).compile_slots(polydat::Engine::Closures(polydat::Provenance::Raw)) {
            Ok(kernel) => kernel,
            Err(_) => panic!("every tile-ladder node must support P2"),
        };
    group.bench_function("p2_closures", |b| {
        let mut cycle = 1u64;
        b.iter(|| {
            p2.eval_at(&[cycle]);
            for name in outputs {
                black_box(weight(&p2.get_value(name)));
            }
            cycle = cycle.wrapping_add(1);
        });
    });

    #[cfg(feature = "jit")]
    {
        let mut p3 = assembler(src)
            .compile_slots(polydat::Engine::Native(polydat::Provenance::Raw))
            .expect("every tile-ladder node must support P3");
        group.bench_function("p3_native", |b| {
            let mut cycle = 1u64;
            b.iter(|| {
                p3.eval_at(&[cycle]);
                for name in outputs {
                    black_box(weight(&p3.get_value(name)));
                }
                cycle = cycle.wrapping_add(1);
            });
        });

        // Pure native code runs a program only when every node lowers;
        // a case a node keeps off it is recorded as absent.
        if let Ok(mut pure) =
            assembler(src).compile_slots(polydat::Engine::PureNative(polydat::Provenance::Raw))
        {
            group.bench_function("pure_native", |b| {
                let mut cycle = 1u64;
                b.iter(|| {
                    pure.eval_at(&[cycle]);
                    for name in outputs {
                        black_box(weight(&pure.get_value(name)));
                    }
                    cycle = cycle.wrapping_add(1);
                });
            });
        }
    }

    group.finish();
}

fn bench_tile_ladder(c: &mut Criterion) {
    bench_case(c, "reading", READING, &READING_OUTPUTS);
    bench_case(c, "one_hole", ONE_HOLE, &["doc"]);
    bench_case(c, "flat", &format!("{READING}{FLAT_TILE}"), &["doc"]);
    bench_case(
        c,
        "projected",
        &format!("{READING}{PROJECTED_TILE}"),
        &["doc"],
    );
    bench_case(c, "wide", &wide_source(), &["doc"]);
}

criterion_group!(benches, bench_tile_ladder);
criterion_main!(benches);
