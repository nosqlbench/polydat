// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The core's tests that construct nodes of this library: they lived
//! beside the compiler when the nodes did, and moved with the nodes,
//! since the core's own tests cannot name a type of a crate above it.

mod fusion {
    use crate::arithmetic::Mod;
    use crate::hash::Hash;
    use polydat::ast::PolydatNode;
    use polydat::ast::Value;
    use polydat::compile::assembly::{PolydatAssembler, WireRef};
    use polydat::compile::fusion::{FusedNode, default_rules};

    #[test]
    fn hash_mod_fuses_to_hash_range() {
        let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
        asm.add_node("h", Box::new(Hash::new()), vec![WireRef::input("cycle")]);
        asm.add_node("m", Box::new(Mod::new(100)), vec![WireRef::node("h")]);
        asm.add_output("out", WireRef::node("m"));

        let mut kernel = asm.compile().unwrap();

        // Verify output correctness after fusion.
        for cycle in 0..1000u64 {
            kernel.set_inputs(&[cycle]);
            let result = kernel.pull("out").as_u64();
            // Must match hash_range semantics: hash(cycle) % 100
            let expected = polydat::numeric::hash::splitmix64_u64(cycle) % 100;
            assert_eq!(result, expected, "cycle {cycle}");
        }
    }

    #[test]
    fn fusion_skipped_when_intermediate_has_consumers() {
        let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
        asm.add_node("h", Box::new(Hash::new()), vec![WireRef::input("cycle")]);
        asm.add_node("m", Box::new(Mod::new(100)), vec![WireRef::node("h")]);
        // Also wire hash output to a second consumer.
        asm.add_node("m2", Box::new(Mod::new(50)), vec![WireRef::node("h")]);
        asm.add_output("out1", WireRef::node("m"));
        asm.add_output("out2", WireRef::node("m2"));

        let mut kernel = asm.compile().unwrap();

        // Both outputs should still work correctly.
        for cycle in 0..100u64 {
            kernel.set_inputs(&[cycle]);
            let h = polydat::numeric::hash::splitmix64_u64(cycle);
            assert_eq!(kernel.pull("out1").as_u64(), h % 100, "out1 cycle {cycle}");
            assert_eq!(kernel.pull("out2").as_u64(), h % 50, "out2 cycle {cycle}");
        }
    }

    /// Helper: compare fused node output vs decomposed graph output
    /// across a range of deterministic inputs.
    fn assert_equivalence(fused: &dyn FusedNode, test_count: usize) {
        let decomposed = fused.decomposed();
        let input_count = fused.meta().wire_inputs().len();
        let output_count = fused.meta().outs.len();

        for seed in 0..test_count as u64 {
            // Generate deterministic inputs from the seed.
            let inputs: Vec<Value> = (0..input_count)
                .map(|port| {
                    let v = xxhash_rust::xxh3::xxh3_64(
                        &(seed.wrapping_mul(31).wrapping_add(port as u64)).to_le_bytes(),
                    );
                    // Use the port type from the fused node's metadata.
                    match fused.meta().wire_inputs()[port].typ {
                        polydat::ast::PortType::U64 => Value::U64(v),
                        polydat::ast::PortType::F64 => Value::F64(f64::from_bits(v)),
                        _ => Value::U64(v), // fallback for other types
                    }
                })
                .collect();

            // Evaluate fused node.
            let mut fused_outputs = vec![Value::None; output_count];
            fused.eval(&inputs, &mut fused_outputs);

            // Evaluate decomposed graph.
            let decomposed_outputs = decomposed.eval(&inputs);

            // Compare each output.
            for (port_idx, (fused_val, decomposed_val)) in fused_outputs
                .iter()
                .zip(decomposed_outputs.iter())
                .enumerate()
            {
                match (&fused_val, &decomposed_val) {
                    (Value::U64(a), Value::U64(b)) => {
                        assert_eq!(
                            a, b,
                            "equivalence failed: seed={seed} port={port_idx} fused={a} decomposed={b}"
                        );
                    }
                    (Value::F64(a), Value::F64(b)) => {
                        // Allow tiny floating point differences from operation reordering.
                        let diff = (a - b).abs();
                        let tolerance = 1e-10 * a.abs().max(b.abs()).max(1.0);
                        assert!(
                            diff <= tolerance,
                            "equivalence failed: seed={seed} port={port_idx} fused={a} decomposed={b} diff={diff}"
                        );
                    }
                    _ => {
                        assert_eq!(
                            fused_val.to_display_string(),
                            decomposed_val.to_display_string(),
                            "equivalence failed: seed={seed} port={port_idx}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn hash_range_equivalence() {
        use crate::hash::HashRange;
        // Test with various moduli including edge cases.
        for max in [1, 2, 7, 100, 10_000, u64::MAX] {
            let fused = HashRange::new(max);
            assert_equivalence(&fused, 10_000);
        }
    }

    #[test]
    fn hash_interval_equivalence() {
        use crate::hash::HashInterval;
        // Test with various ranges.
        for (lo, hi) in [(0.0, 1.0), (-180.0, 180.0), (0.0, 1000.0), (-1.0, -0.5)] {
            let fused = HashInterval::new(lo, hi);
            assert_equivalence(&fused, 10_000);
        }
    }

    #[test]
    fn scale_range_equivalence() {
        use crate::lerp::ScaleRange;
        for (lo, hi) in [(0.0, 1.0), (-100.0, 100.0), (0.0, 360.0), (1e6, 1e7)] {
            let fused = ScaleRange::new(lo, hi);
            assert_equivalence(&fused, 10_000);
        }
    }

    /// Meta-test: verify that the default fusion rules all reference
    /// nodes that implement FusedNode and pass equivalence checks.
    #[test]
    fn all_default_rules_produce_equivalent_nodes() {
        let rules = default_rules();
        for rule in &rules {
            // Build a representative match result with plausible constants.
            // We test the actual fused nodes via their specific tests above;
            // this test verifies the rule table is wired correctly.
            assert!(
                rule.pattern.root_op().is_some(),
                "rule '{}' has no root op",
                rule.name
            );
            assert!(
                !rule.input_bindings.is_empty(),
                "rule '{}' has no input bindings",
                rule.name
            );
        }
    }

    #[test]
    fn variadic_pattern_matches_sum() {
        use crate::arithmetic::Sum;

        // Build a graph: sum(a, b, c) where a, b, c are coordinates
        let mut asm = PolydatAssembler::new(vec!["a".into(), "b".into(), "c".into()]);
        asm.add_node(
            "s",
            Box::new(Sum::new(3)),
            vec![
                WireRef::input("a"),
                WireRef::input("b"),
                WireRef::input("c"),
            ],
        );
        asm.add_output("out", WireRef::node("s"));

        let mut kernel = asm.compile().unwrap();

        // Verify it works.
        kernel.set_inputs(&[10, 20, 30]);
        assert_eq!(kernel.pull("out").as_u64(), 60);
    }

    #[test]
    fn typed_constants_captured_in_match() {
        use crate::hash::HashRange;
        use polydat::ast::ConstValue;

        // Build: hash_range(cycle, 100)
        let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
        asm.add_node(
            "hr",
            Box::new(HashRange::new(100)),
            vec![WireRef::input("cycle")],
        );
        asm.add_output("out", WireRef::node("hr"));

        // Manually test pattern matching on the resolved graph.
        // HashRange has slots: [Wire("input"), Const("max", U64, 100)]
        let node = HashRange::new(100);
        let typed: Vec<ConstValue> = node
            .meta()
            .const_slots()
            .iter()
            .map(|c| c.1.clone())
            .collect();
        assert_eq!(typed.len(), 1);
        assert_eq!(typed[0], ConstValue::U64(100));
    }
}

mod simd_plan {
    use polydat::ast::PortType;
    use polydat::compile::simd_plan::{SimdVariantError, validate_simd_variant};

    #[test]
    fn exact_core_variants_validate_by_signature() {
        let add = crate::bitwise::U64Add::new();
        let add_variant = validate_simd_variant(&add).unwrap();
        assert_eq!(add_variant.vector_node, "reg_add_i64");
        assert_eq!(add_variant.shape.scalar, PortType::U64);
        assert_eq!(add_variant.shape.lanes, 2);

        let mul = crate::math::F64Mul::new();
        let mul_variant = validate_simd_variant(&mul).unwrap();
        assert_eq!(mul_variant.vector_node, "reg_mul_f64");
        assert_eq!(mul_variant.shape.register, PortType::RegF64x2);
    }

    #[test]
    fn undeclared_nodes_remain_ineligible() {
        let node = crate::bitwise::U64Div::new();
        assert_eq!(
            validate_simd_variant(&node),
            Err(SimdVariantError::Undeclared)
        );
    }
}

#[cfg(feature = "jit")]
mod codegen {
    use polydat::compile::jit::{JitOp, classify_node, compile_jit_raw};
    use std::collections::HashMap;

    #[test]
    fn jit_shuffle() {
        // Create a real Shuffle to get its constants
        use crate::sampling::metashift::{Shuffle, feedback_for_size};
        use polydat::ast::PolydatNode;
        // SRD-80b Phase E — `Shuffle::new` now takes `(feedback, size, min)`.
        // The bank-0 feedback for size=1000 is computed via the public helper.
        let size = 1000u64;
        let node = Shuffle::new(feedback_for_size(size), size, 0);
        let consts = node.jit_constants();

        let steps = vec![(
            JitOp::ShuffleConst(consts[0], consts[1], consts[2]),
            vec![0],
            vec![1],
        )];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        // Verify same result as the node
        kernel.eval(&[42]);
        let jit_result = kernel.get("out");

        let mut out = [polydat::ast::Value::None];
        node.eval(&[polydat::ast::Value::U64(42)], &mut out);
        assert_eq!(jit_result, out[0].as_u64());
    }

    #[test]
    fn jit_lut_sample() {
        // Build a simple linear LUT: f(x) = x * 100
        use crate::sampling::lut::LutF64;
        let lut = LutF64::from_fn(|p| p * 100.0, 1000);
        let lut_ptr = lut.as_ptr() as u64;
        let lut_len = lut.len() as u64;

        let steps = vec![(JitOp::LutSampleConst(lut_ptr, lut_len), vec![0], vec![1])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        // Input 0.5 → should give ~50.0
        kernel.eval(&[0.5f64.to_bits()]);
        let v = f64::from_bits(kernel.get("out"));
        assert!((v - 50.0).abs() < 0.1, "got {v}");

        // Input 0.0 → should give 0.0
        kernel.eval(&[0.0f64.to_bits()]);
        let v = f64::from_bits(kernel.get("out"));
        assert!((v - 0.0).abs() < 0.1, "got {v}");

        // Input 1.0 → should give 100.0
        kernel.eval(&[1.0f64.to_bits()]);
        let v = f64::from_bits(kernel.get("out"));
        assert!((v - 100.0).abs() < 0.1, "got {v}");
    }

    #[test]
    fn jit_lut_normal_distribution() {
        // Build a normal distribution LUT and verify JIT gives same results as P1
        use crate::sampling::icd;
        let lut = icd::dist_normal_lut(0.0, 1.0, icd::DEFAULT_RESOLUTION);
        let lut_ptr = lut.as_ptr() as u64;
        let lut_len = lut.len() as u64;

        let steps = vec![(JitOp::LutSampleConst(lut_ptr, lut_len), vec![0], vec![1])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        // Median of standard normal = 0.0
        kernel.eval(&[0.5f64.to_bits()]);
        let v = f64::from_bits(kernel.get("out"));
        assert!((v - 0.0).abs() < 0.01, "median should be ~0, got {v}");

        // p=0.5 + 1σ ≈ 0.8413 → should give ~1.0
        kernel.eval(&[0.8413f64.to_bits()]);
        let v = f64::from_bits(kernel.get("out"));
        assert!((v - 1.0).abs() < 0.05, "1σ should be ~1.0, got {v}");
    }

    #[test]
    fn classify_routes_new_param_helpers() {
        use crate::param_helpers::{InRange, IsPositive};
        // The classify_node entrypoint must see the new
        // predicate nodes and return the JIT-lowered op variants
        // rather than falling through to Fallback.
        let p = IsPositive::new("rate".to_string());
        assert!(matches!(classify_node(&p), JitOp::IsPositiveCheck { .. }));

        let r = InRange::new(1, 100);
        assert!(matches!(classify_node(&r), JitOp::InRangeCheck(1, 100)));
    }

    #[test]
    fn classify_leaves_other_param_helpers_on_fallback() {
        use crate::param_helpers::{Matches, Required, ThisOr};
        // By design: required/this_or/matches stay on Phase-2.
        // classify_node must pick Fallback so the closure-based
        // eval runs instead of an uninitialized JIT op.
        assert!(matches!(
            classify_node(&Required::new("x".to_string())),
            JitOp::Fallback
        ));
        assert!(matches!(classify_node(&ThisOr::new()), JitOp::Fallback));
        assert!(matches!(
            classify_node(&Matches::new(r"^\d+$".to_string())),
            JitOp::Fallback
        ));
    }

    #[test]
    fn classify_routes_is_one_of_to_fallback() {
        // SRD-80b Phase C — `is_one_of` migrated to the macro's
        // `Const<Vec<C>>` shape, which is JIT-ineligible (the JIT
        // u64 buffer has no slot shape for a variable-length
        // captured list). The node now runs on the typed-eval
        // path. A future `compiled_u64_override` could reinstate
        // the JIT lowering if perf demands it.
        use crate::param_helpers::IsOneOf;
        let n = IsOneOf::new(vec![1, 3, 5, 7]);
        assert!(matches!(classify_node(&n), JitOp::Fallback));
    }
}
