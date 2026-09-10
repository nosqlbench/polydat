// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Cross-engine semantic gate for the graph used by `engine_ladder` benchmark.

use polydat::JitMode;
use polydat::dsl::compile::compile_polydat_to_assembler;

const GRAPH: &str = include_str!("../examples/engine_ladder.polydat");
const OUTPUTS: [&str; 4] = ["account_id", "shard", "payload_class", "event_token"];

#[test]
fn engine_ladder_has_identical_results_at_each_level() {
    let mut p1_asm = compile_polydat_to_assembler(GRAPH).expect("assemble P1 graph");
    p1_asm.set_jit_mode(JitMode::Off);
    let mut p1 = p1_asm.compile().expect("compile P1 graph");
    let p1_outputs =
        OUTPUTS.map(|name| p1.program().output_index(name).expect("resolve P1 output"));

    let mut p2 = match compile_polydat_to_assembler(GRAPH)
        .expect("assemble P2 graph")
        .try_compile_raw()
    {
        Ok(kernel) => kernel,
        Err(_) => panic!("engine-ladder graph must be entirely P2-compatible"),
    };
    let p2_outputs = OUTPUTS.map(|name| p2.resolve_output(name).expect("resolve P2 output"));

    #[cfg(feature = "jit")]
    let mut p3 = compile_polydat_to_assembler(GRAPH)
        .expect("assemble P3 graph")
        .try_compile_jit_raw()
        .expect("engine-ladder graph must be entirely P3-compatible");
    #[cfg(feature = "jit")]
    let p3_outputs = OUTPUTS.map(|name| p3.resolve_output(name).expect("resolve P3 output"));
    #[cfg(feature = "jit")]
    let mut pure = compile_polydat_to_assembler(GRAPH)
        .expect("assemble pure native graph")
        .try_compile_pure_jit_raw()
        .expect("engine-ladder graph must lower entirely to native code");
    #[cfg(feature = "jit")]
    let pure_outputs = OUTPUTS.map(|name| {
        pure.resolve_output(name)
            .expect("resolve pure native output")
    });

    let cases = [
        [0, 0, 0],
        [1, 0x5445_4e41_4e54, 0x4f50_4552_4154_494f],
        [42, 7, 11],
        [1_000_000, u32::MAX as u64, 17],
        [u64::MAX, u64::MAX - 1, u64::MAX - 2],
    ];

    for inputs in cases {
        p1.set_inputs(&inputs);
        let p1_values = p1_outputs.map(|output| p1.pull_by_index(output).as_u64());

        p2.eval(&inputs);
        let p2_values = p2_outputs.map(|output| p2.get_slot(output));
        assert_eq!(p2_values, p1_values, "P2 differs for inputs {inputs:?}");

        #[cfg(feature = "jit")]
        {
            p3.eval(&inputs);
            let p3_values = p3_outputs.map(|output| p3.get_slot(output));
            assert_eq!(p3_values, p1_values, "P3 differs for inputs {inputs:?}");
            pure.eval(&inputs);
            let pure_values = pure_outputs.map(|output| pure.get_slot(output));
            assert_eq!(
                pure_values, p1_values,
                "pure native code differs for inputs {inputs:?}"
            );
        }

        assert!(p1_values[0] < 10_000_000, "account_id bound");
        assert!(p1_values[1] < 64, "shard bound");
        assert!(p1_values[2] < 8, "payload_class bound");
    }
}
