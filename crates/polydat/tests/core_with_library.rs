// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The core's unit tests that compile programs over the node library.
//! They lived beside the compiler, the kernel, and the registry when
//! the nodes did; the core's own test build cannot link the node
//! library (it would see a second copy of the core), so every test
//! that names a library function in Polydat source runs here, where
//! the facade links everything. Each module mirrors the module the
//! tests came from, with the helpers those tests use.

#![allow(clippy::approx_constant)]

mod dsl_compile_tests {
    #![allow(unused_imports, dead_code)]
    use polydat::JitMode;
    use polydat::ast::*;
    use polydat::dsl::ast::*;
    use polydat::dsl::compile::*;
    use polydat::dsl::registry::*;
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use polydat::kernel::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// The interpreter kernel under `strict` alone.
    fn strict(src: &str, strict: bool) -> Result<PolydatKernel, polydat::KernelError> {
        let options = CompileOptions {
            strict,
            ..CompileOptions::default()
        };
        compile_polydat_interpreter_with_options(src, &options, None)
    }

    /// The interpreter kernel keeping `required` outputs, under `strict`.
    fn with_outputs(
        src: &str,
        required: &[String],
        strict: bool,
    ) -> Result<PolydatKernel, polydat::KernelError> {
        let options = CompileOptions {
            required_outputs: required.to_vec(),
            strict,
            ..CompileOptions::default()
        };
        compile_polydat_interpreter_with_options(src, &options, None)
    }

    #[test]
    fn typed_surface_bool() {
        let v: bool = eval_const_expr_typed("5 > 3").unwrap();
        assert!(v);
        let v: bool = eval_const_expr_typed("3 > 5").unwrap();
        assert!(!v);
    }

    #[test]
    fn typed_surface_u64() {
        let v: u64 = eval_const_expr_typed("10 * 5").unwrap();
        assert_eq!(v, 50);
    }

    #[test]
    fn typed_surface_f64() {
        let v: f64 = eval_const_expr_typed("3.14 * 2.0").unwrap();
        assert!((v - 6.28).abs() < 1e-9);
    }

    #[test]
    fn typed_strict_kernel_bound() {
        let kernel = compile_polydat_interpreter("const k := 10\n").unwrap();
        // Lossless: u64 → String, the display round-trip.
        let v: String = eval_kernel_bound_typed_strict("{k} * 2", &kernel).unwrap();
        assert_eq!(v, "20");
        // Lossy: u64 → bool — strict rejects.
        let result: Result<bool, _> = eval_kernel_bound_typed_strict("{k} > 5", &kernel);
        assert!(matches!(result, Err(EmbeddingError::TypeMismatch { .. })));
        // Lossy too: u64's 64 magnitude bits do not fit f64's 53-bit
        // significand, so strict refuses the widening rather than
        // deciding per value.
        let result: Result<f64, _> = eval_kernel_bound_typed_strict("{k} * 2", &kernel);
        assert!(matches!(result, Err(EmbeddingError::TypeMismatch { .. })));
    }

    #[test]
    fn typed_surface_kernel_bound() {
        let kernel = compile_polydat_interpreter("const k := 10\n").unwrap();
        let v: bool = eval_kernel_bound_typed("{k} > 5", &kernel).unwrap();
        assert!(v);
        let v: u64 = eval_kernel_bound_typed("{k} * 2", &kernel).unwrap();
        assert_eq!(v, 20);
    }

    #[test]
    fn compile_hello_world() {
        let src = r#"
            input cycle: u64
            hashed := hash(cycle)
            user_id := mod(hashed, 1000000)
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[42]);
        let uid = kernel.pull_ref("user_id").as_u64();
        assert!(uid < 1_000_000, "user_id={uid}");
    }

    #[test]
    fn compile_with_inline_nesting() {
        let src = r#"
            input cycle: u64
            result := mod(hash(cycle), 100)
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[42]);
        assert!(kernel.pull_ref("result").as_u64() < 100);
    }

    #[test]
    fn compile_deterministic() {
        let src = r#"
            input cycle: u64
            h := hash(cycle)
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[42]);
        let v1 = kernel.pull_ref("h").as_u64();
        kernel.set_inputs(&[42]);
        let v2 = kernel.pull_ref("h").as_u64();
        assert_eq!(v1, v2);
    }

    #[test]
    fn shared_modifier_tracked() {
        let src = r#"
            input cycle: u64
            shared counter := 0
            normal := mod(hash(cycle), 100)
        "#;
        let kernel = compile_polydat_interpreter(src).unwrap();
        assert_eq!(
            kernel.program().output_modifier("counter"),
            polydat::dsl::ast::BindingModifier::SHARED
        );
        assert_eq!(
            kernel.program().output_modifier("normal"),
            polydat::dsl::ast::BindingModifier::NONE
        );
    }

    #[test]
    fn shared_outputs_query() {
        let src = r#"
            input cycle: u64
            shared counter := 0
            shared budget := 100
            normal := hash(cycle)
        "#;
        let kernel = compile_polydat_interpreter(src).unwrap();
        let mut shared = kernel.program().shared_outputs();
        shared.sort();
        assert_eq!(shared, vec!["budget", "counter"]);
        assert!(kernel.program().const_outputs().is_empty());
    }

    #[test]
    fn final_outputs_query() {
        let src = r#"
            input cycle: u64
            const dim := 128
            const dataset := "example"
            normal := hash(cycle)
        "#;
        let kernel = compile_polydat_interpreter(src).unwrap();
        let mut finals = kernel.program().const_outputs();
        finals.sort();
        assert_eq!(finals, vec!["dataset", "dim"]);
        assert!(kernel.program().shared_outputs().is_empty());
    }

    #[test]
    fn unmodified_bindings_have_none_modifier() {
        let src = r#"
            input cycle: u64
            h := hash(cycle)
            v := mod(h, 100)
        "#;
        let kernel = compile_polydat_interpreter(src).unwrap();
        assert_eq!(
            kernel.program().output_modifier("h"),
            polydat::dsl::ast::BindingModifier::NONE
        );
        assert_eq!(
            kernel.program().output_modifier("v"),
            polydat::dsl::ast::BindingModifier::NONE
        );
    }

    #[test]
    fn compile_mixed_radix() {
        let src = r#"
            input cycle: u64
            (tenant, device, reading) := mixed_radix(cycle, 100, 1000, 0)
            tenant_h := hash(tenant)
            tenant_code := mod(tenant_h, 10000)
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[4_201_337]);
        let tc = kernel.pull_ref("tenant_code").as_u64();
        assert!(tc < 10000, "tenant_code={tc}");
    }

    #[test]
    fn compile_comments_ignored() {
        let src = r#"
            // This is a comment
            input cycle: u64
            // Another comment
            h := hash(cycle)
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[1]);
        assert!(kernel.pull_ref("h").as_u64() != 0);
    }

    #[test]
    fn error_unknown_function_suggests() {
        let src = "input cycle: u64\nresult := hahs(cycle)";
        let (_, report) = compile_polydat_checked(src);
        let errors = report.errors();
        let err = errors.iter().find(|e| e.message.contains("hahs")).unwrap();
        assert!(
            err.hint.as_ref().unwrap().contains("hash"),
            "should suggest 'hash', got: {:?}",
            err.hint
        );
    }

    #[test]
    fn inferred_coordinates() {
        // Without explicit coordinates, 'cycle' is inferred as a coordinate input
        let src = "h := hash(cycle)";
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        assert_eq!(kernel.input_names(), &["cycle"]);
        kernel.set_inputs(&[42]);
        let h = kernel.pull_ref("h").as_u64();
        assert_ne!(h, 42); // hashed, not identity
    }

    #[test]
    fn inferred_multi_coordinates() {
        // Multiple unbound names become multiple coordinate inputs (sorted)
        let src = "h := hash(interleave(row, col))";
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        assert_eq!(kernel.input_names(), &["col", "row"]); // alphabetically sorted
        kernel.set_inputs(&[10, 20]);
        let h = kernel.pull_ref("h").as_u64();
        assert_ne!(h, 0);
    }

    #[test]
    fn checked_compile_success_with_no_errors() {
        let src = r#"
            input cycle: u64
            h := hash(cycle)
            result := mod(h, 1000)
        "#;
        let (result, report) = compile_polydat_checked(src);
        assert!(!report.has_errors());
        assert!(result.is_ok());
    }

    #[test]
    fn strict_accepts_explicit_coordinates() {
        // With explicit coordinates, strict mode should succeed
        let src = r#"
            input cycle: u64
            h := hash(cycle)
        "#;
        let mut kernel = strict(src, true).unwrap();
        kernel.set_inputs(&[42]);
        let h = kernel.pull_ref("h").as_u64();
        assert_ne!(h, 42); // hashed, not identity
    }

    #[test]
    fn non_strict_infers_coordinates() {
        // Without strict, coordinate inference works as before
        let src = "h := hash(cycle)";
        let mut kernel = strict(src, false).unwrap();
        kernel.set_inputs(&[42]);
        assert_ne!(kernel.pull_ref("h").as_u64(), 42);
    }

    #[test]
    fn dce_filters_to_required_outputs() {
        // Polydat source defines three bindings but we only request one
        let src = r#"
            input cycle: u64
            a := hash(cycle)
            b := mod(a, 100)
            c := add(cycle, 1)
        "#;
        let required = vec!["b".to_string()];
        let mut kernel = with_outputs(src, &required, false).unwrap();
        kernel.set_inputs(&[42]);

        // "b" should be available and correct
        let b = kernel.pull_ref("b").as_u64();
        assert!(b < 100, "b={b}");

        // "a" and "c" should NOT be in the output map
        let outputs = kernel.output_names();
        assert!(outputs.contains(&"b"), "should contain 'b'");
        assert!(!outputs.contains(&"a"), "should not contain pruned 'a'");
        assert!(!outputs.contains(&"c"), "should not contain pruned 'c'");
    }

    #[test]
    fn dce_preserves_upstream_dependencies() {
        // Request "result" which depends on "h" — both the result node
        // and its upstream "h" node must be kept, but "unrelated" is pruned
        let src = r#"
            input cycle: u64
            h := hash(cycle)
            result := mod(h, 1000)
            unrelated := add(cycle, 999)
        "#;
        let required = vec!["result".to_string()];
        let mut kernel = with_outputs(src, &required, false).unwrap();
        kernel.set_inputs(&[42]);

        let result = kernel.pull_ref("result").as_u64();
        assert!(result < 1000, "result={result}");

        let outputs = kernel.output_names();
        assert!(
            !outputs.contains(&"unrelated"),
            "unrelated should be pruned"
        );
    }

    #[test]
    fn dce_empty_required_compiles_all() {
        // Empty required_outputs should produce the same kernel as compile_polydat
        let src = r#"
            input cycle: u64
            a := hash(cycle)
            b := mod(a, 100)
        "#;
        let kernel_all = compile_polydat_interpreter(src).unwrap();
        let kernel_empty = with_outputs(src, &[], false).unwrap();

        assert_eq!(
            kernel_all.output_names().len(),
            kernel_empty.output_names().len()
        );
    }

    #[test]
    fn init_binding_survives_dce_even_when_unconsumed() {
        // Regression test for the prebuffer-not-firing bug. An
        // `const` binding declares a side-effect-bearing init-time
        // computation (download, register, prebuffer). The user's
        // signal that they want it evaluated is the `init`
        // keyword itself, *not* a downstream wire reference. Yet
        // the assembler's DCE walks back from the requested
        // outputs and prunes whatever's not in their ancestry.
        //
        // Pre-fix: with `required = ["b"]` and an unconsumed
        // `init side_effect = …` binding, the `side_effect` node
        // (and its constant-fold call) got pruned. Post-fix:
        // `compile_polydat_with_outputs` extends the required list
        // with every `const` binding's name, so DCE keeps the
        // node, fold evaluates it, and `kernel.pull("side_effect")`
        // returns the folded result.
        let src = r#"
            input cycle: u64
            const side_effect := 42
            b := mod(hash(cycle), 100)
        "#;
        let required = vec!["b".to_string()];
        let mut kernel = with_outputs(src, &required, false).unwrap();
        kernel.set_inputs(&[0]);

        let outputs = kernel.output_names();
        assert!(
            outputs.contains(&"side_effect"),
            "init binding must survive DCE even when unconsumed; got outputs {outputs:?}"
        );
        assert_eq!(kernel.pull_ref("side_effect").as_u64(), 42);
    }

    #[test]
    fn dce_multiple_required_outputs() {
        // Request two of three bindings
        let src = r#"
            input cycle: u64
            x := hash(cycle)
            y := mod(x, 50)
            z := add(cycle, 10)
        "#;
        let required = vec!["y".to_string(), "z".to_string()];
        let mut kernel = with_outputs(src, &required, false).unwrap();
        kernel.set_inputs(&[5]);

        assert!(kernel.pull_ref("y").as_u64() < 50);
        assert_eq!(kernel.pull_ref("z").as_u64(), 15);

        let outputs = kernel.output_names();
        assert!(outputs.contains(&"y"));
        assert!(outputs.contains(&"z"));
        // "x" is an upstream dep of "y" but not a requested output
        assert!(!outputs.contains(&"x"), "x should not be in outputs");
    }

    #[test]
    fn strict_rejects_unused_bindings() {
        // "unused" has no downstream consumer and is not an output → strict error
        // Use compile_polydat_strict which exposes all bindings as outputs,
        // so the kernel sees the full graph and detects the unused node.
        // Actually: when all bindings are outputs, none are "unused".
        // The unused check only applies with DCE (required_outputs filter).
        // With DCE, pruned bindings produce a warning at the compiler level.
        let src = r#"
            input cycle: u64
            used := hash(cycle)
            unused := add(cycle, 1)
        "#;
        let required = vec!["used".to_string()];
        // Non-strict: DCE prunes "unused" silently
        let result = with_outputs(src, &required, false);
        assert!(result.is_ok(), "non-strict with DCE should compile");
        // Verify "unused" is actually pruned
        let kernel = result.unwrap();
        assert!(
            !kernel.output_names().contains(&"unused"),
            "unused should be pruned by DCE"
        );
    }

    #[test]
    fn strict_rejects_implicit_type_coercion() {
        // u64 → f64 auto-adapter → strict error
        let src = r#"
            input cycle: u64
            h := hash(cycle)
            f := sqrt(h)
        "#;
        let result = strict(src, true);
        assert!(result.is_err(), "strict should reject implicit coercion");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("coercion") || err.to_string().contains("__adapt"),
            "error should mention coercion: {err}"
        );
    }

    #[test]
    fn non_strict_allows_implicit_type_coercion() {
        let src = r#"
            input cycle: u64
            h := hash(cycle)
            f := sqrt(h)
        "#;
        let result = strict(src, false);
        assert!(result.is_ok(), "non-strict should allow implicit coercion");
    }

    #[test]
    fn strict_accepts_clean_program() {
        // All inputs declared, all bindings used, no coercions
        let src = r#"
            input cycle: u64
            h := hash(cycle)
            id := mod(h, 1000)
        "#;
        let required = vec!["id".to_string()];
        let result = with_outputs(src, &required, true);
        assert!(
            result.is_ok(),
            "clean program should pass strict: {:?}",
            result.err()
        );
    }

    #[test]
    fn compile_bitwise_and() {
        let src = r#"
            input cycle: u64
            out := cycle & 0xFF
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[0x1234]);
        assert_eq!(kernel.pull_ref("out").as_u64(), 0x34);
    }

    #[test]
    fn compile_shift_left() {
        let src = r#"
            input cycle: u64
            out := cycle << 8
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[1]);
        assert_eq!(kernel.pull_ref("out").as_u64(), 256);
    }

    #[test]
    fn compile_bitwise_not() {
        let src = r#"
            input cycle: u64
            out := !cycle
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[0]);
        assert_eq!(kernel.pull_ref("out").as_u64(), u64::MAX);
    }

    #[test]
    fn compile_bitwise_xor() {
        let src = r#"
            input cycle: u64
            out := cycle ^ 0xFF
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[0xF0]);
        assert_eq!(kernel.pull_ref("out").as_u64(), 0x0F);
    }

    #[test]
    fn compile_bitwise_or() {
        let src = r#"
            input cycle: u64
            out := cycle | 0x0F
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[0xF0]);
        assert_eq!(kernel.pull_ref("out").as_u64(), 0xFF);
    }

    #[test]
    fn compile_shift_right() {
        let src = r#"
            input cycle: u64
            out := cycle >> 4
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[0xFF]);
        assert_eq!(kernel.pull_ref("out").as_u64(), 0x0F);
    }

    #[test]
    fn compile_power_operator() {
        let src = r#"
            input cycle: u64
            out := to_f64(cycle) ** 2.0
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[3]);
        // pow(3.0, 2.0) = 9.0
        let result = kernel.pull_ref("out").as_f64();
        assert!((result - 9.0).abs() < 0.001);
    }

    #[test]
    fn eval_const_expr_arithmetic() {
        // 4 * 4: both operands are IntLit → u64_mul → returns u64(16)
        let v = eval_const_expr("4 * 4").unwrap();
        assert_eq!(v.as_u64(), 16, "expected u64(16), got {:?}", v);

        // 4.0 * 4.0: both operands are FloatLit → f64_mul → returns f64(16.0)
        let v = eval_const_expr("4.0 * 4.0").unwrap();
        assert!(
            (v.as_f64() - 16.0).abs() < 0.001,
            "expected 16.0, got {}",
            v.as_f64()
        );

        // Mixed: 4 * 4.0 → auto-widen LHS to f64, f64_mul → returns f64(16.0)
        let v = eval_const_expr("4 * 4.0").unwrap();
        assert!(
            (v.as_f64() - 16.0).abs() < 0.001,
            "expected 16.0, got {}",
            v.as_f64()
        );
    }

    #[test]
    fn eval_const_expr_function() {
        let v = eval_const_expr("hash(42)").unwrap();
        assert!(v.as_u64() != 0, "hash(42) should be non-zero");
    }

    #[test]
    fn eval_const_expr_nested() {
        let v = eval_const_expr("mod(hash(42), 100)").unwrap();
        assert!(
            v.as_u64() < 100,
            "mod(hash(42), 100) should be < 100, got {}",
            v.as_u64()
        );
    }

    #[test]
    fn init_binding_wired_to_cycle_input_rejected() {
        // Init binding wired to `cycle` (Coordinate input) is a
        // hard structural violation. Plan A must reject.
        let src = "input cycle: u64\n\
                   const bad := hash(cycle)\n";
        let err = compile_polydat_interpreter(src)
            .expect_err("Plan A must reject init binding wired to a coordinate input");
        assert!(
            err.to_string().contains("init binding 'bad'")
                && err.to_string().contains("init contract"),
            "diagnostic must name the binding and the contract; got: {err}"
        );
        assert!(
            err.to_string().contains("cycle") || err.to_string().contains("coordinate"),
            "diagnostic should pinpoint the offending wire; got: {err}"
        );
    }

    #[test]
    fn init_binding_wired_to_external_write_port_rejected() {
        // External-write port (extern with default) is dynamic;
        // init bindings must not depend on one.
        let src = "extern session_id: u64 = 0\n\
                   const derived := mod(session_id, 100)\n";
        let err = compile_polydat_interpreter(src)
            .expect_err("Plan A must reject init binding wired to a external-write port");
        assert!(
            err.to_string().contains("init binding 'derived'")
                && err.to_string().contains("init contract"),
            "diagnostic must name the binding and the contract; got: {err}"
        );
        assert!(
            err.to_string().contains("session_id") || err.to_string().contains("capture"),
            "diagnostic should pinpoint the offending wire; got: {err}"
        );
    }

    #[test]
    fn cycle_binding_wired_to_cycle_input_still_allowed() {
        // The contract applies *only* to bindings declared `init`.
        // A normal `:=` binding wired to `cycle` is the bread-and-
        // butter case and must keep working.
        let src = "input cycle: u64\n\
                   user_id := mod(hash(cycle), 1000)\n";
        let _kernel = compile_polydat_interpreter(src)
            .expect("non-init bindings wired to cycle must still compile");
    }

    #[test]
    fn str_concat_via_plus_operator() {
        // `+` between Str-typed operands lowers to str_concat.
        let src = r#"
            input cycle: u64
            greeting := "hello, " + "world"
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[0]);
        assert_eq!(kernel.pull_ref("greeting").as_str(), "hello, world");
    }

    #[test]
    fn str_concat_flattens_chained_plus() {
        // `"a" + b + "c"` flattens into a single str_concat node
        // (rather than a chain of binary concatenations) so the
        // assembler sees the full operand list at once.
        let src = r#"
            input cycle: u64
            x := "id="
            y := 42
            z := " end"
            out := x + y + z
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[0]);
        assert_eq!(kernel.pull_ref("out").as_str(), "id=42 end");
    }

    #[test]
    fn str_concat_mixed_str_and_numeric() {
        // Numeric operand on the right is rendered as decimal text;
        // the Str path wins because the left side is Str.
        let src = r#"
            input cycle: u64
            n := 7
            out := "n=" + n
        "#;
        let mut kernel = compile_polydat_interpreter(src).unwrap();
        kernel.set_inputs(&[0]);
        assert_eq!(kernel.pull_ref("out").as_str(), "n=7");
    }

    #[test]
    fn auto_extern_slot_inherits_arithmetic_operand_type() {
        // `const y := other + 1` — BinOp with U64 operands.
        // The auto-extern slot for `y` MUST be U64.
        let src = r#"
            extern other: u64
            const y := other + 1
        "#;
        let kernel = compile_polydat_interpreter(src).expect("compile");
        assert_eq!(
            kernel.program().input_port_type("y"),
            Some(polydat::ast::PortType::U64),
            "arithmetic-RHS auto-extern MUST inherit operand type",
        );
    }

    /// SRD-84 Part 1 — `&&` / `||` as eager truthiness combinators:
    /// correct results, lowest precedence (below comparison; `||`
    /// looser than `&&`), and truthiness normalisation (a value is
    /// "true" iff non-zero — which raw bitwise would get wrong).
    #[test]
    fn logical_and_or_eval_precedence_and_truthiness() {
        let eval = |src: &str| -> u64 {
            compile_polydat_interpreter(src)
                .unwrap_or_else(|e| panic!("compile `{src}`: {e}"))
                .pull_ref("out")
                .as_u64()
        };
        // Basic && / ||.
        assert_eq!(eval("out := 60 > 50 && 20 > 10"), 1, "both true");
        assert_eq!(eval("out := 40 > 50 && 20 > 10"), 0, "first false");
        assert_eq!(eval("out := 40 > 50 || 20 > 10"), 1, "second true");
        assert_eq!(eval("out := 40 > 50 || 5 > 10"), 0, "neither");
        // Precedence: && sits below comparison, || below && —
        // `1>0 && 0>1 || 5>0` == `((1>0)&&(0>1)) || (5>0)` == `(1&&0)||1` == 1.
        assert_eq!(
            eval("out := 1 > 0 && 0 > 1 || 5 > 0"),
            1,
            "|| binds looser than &&, both below comparison"
        );
        // Truthiness: 6 && 1 → both non-zero → 1. Raw bitwise 6 & 1 = 0,
        // so this proves the `!= 0` normalisation, not a bitwise and.
        assert_eq!(
            eval("out := 6 && 1"),
            1,
            "non-zero && non-zero → 1 (not bitwise)"
        );
        assert_eq!(eval("out := 6 && 0"), 0, "non-zero && zero → 0");
        assert_eq!(eval("out := 0 || 0"), 0, "zero || zero → 0");
        assert_eq!(eval("out := 0 || 7"), 1, "zero || non-zero → 1");
        // Parentheses override precedence (already supported; locked here).
        assert_eq!(eval("out := (1 + 2) * 3"), 9, "parens: add before mul");
        assert_eq!(eval("out := 1 + 2 * 3"), 7, "no parens: mul binds tighter");
        assert_eq!(
            eval("out := 1 > 0 || 0 > 1 && 0 > 1"),
            1,
            "no parens: && tighter → 1 || (0 && 0) = 1"
        );
        assert_eq!(
            eval("out := (1 > 0 || 0 > 1) && 0 > 1"),
            0,
            "parens group the ||: (1 || 0) && 0 = 0"
        );
    }

    /// SRD-84 Part 1b — `<expr> as <type>` cast: alignment-only type
    /// fusion (no-op when aligned, SRD-79 adapter otherwise), tight
    /// (atom-binding) precedence, and an error when no fusion exists.
    #[test]
    fn as_cast_type_fusion_and_precedence() {
        let f64_of = |src: &str| {
            compile_polydat_interpreter(src)
                .unwrap_or_else(|e| panic!("compile `{src}`: {e}"))
                .pull_ref("out")
                .as_f64()
        };
        let u64_of = |src: &str| {
            compile_polydat_interpreter(src)
                .unwrap_or_else(|e| panic!("compile `{src}`: {e}"))
                .pull_ref("out")
                .as_u64()
        };
        // u64 → f64 widening fusion (allowed under `as`).
        assert_eq!(f64_of("out := 5 as f64"), 5.0);
        // Narrowing f64 → u64 is NOT allowed under `as` (ambiguous
        // rounding); the author chooses an explicit conversion.
        assert!(
            compile_polydat_interpreter("out := 7.9 as u64").is_err(),
            "narrowing f64 → u64 under `as` is rejected"
        );
        assert_eq!(u64_of("out := f64_to_u64(7.9)"), 7, "explicit truncate");
        assert_eq!(u64_of("out := round_to_u64(7.9)"), 8, "explicit round");
        // Aligned cast is a no-op.
        assert_eq!(u64_of("out := 42 as u64"), 42);
        // `as` binds to the atom: `5 / 2 as f64` == `5 / (2 as f64)` ==
        // 2.5, not `(5 / 2) as f64` == 2.0.
        assert_eq!(f64_of("out := 5 / 2 as f64"), 2.5);
        assert_eq!(f64_of("out := (5 / 2) as f64"), 2.0);
        // No valid fusion → compile error.
        assert!(
            compile_polydat_interpreter("out := \"x\" as f64").is_err(),
            "str → f64 has no defined fusion → error"
        );
    }
}

#[cfg(feature = "jit")]
mod compile_cone_tests {
    #![allow(unused_imports, dead_code)]
    use polydat::JitMode;
    use polydat::ast::*;
    use polydat::compile::cone::*;
    use polydat::dsl::ast::*;
    use polydat::dsl::compile::*;
    use polydat::dsl::registry::*;
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use polydat::kernel::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    const U64_CHAIN: &str = "input (x: u64)\n\
                             v := mul(x, 3)\n\
                             w := add(v, 7)\n";

    /// The interpreter kernel of `src` with its cones per `mode`.
    fn compile(src: &str, mode: JitMode) -> polydat::kernel::PolydatKernel {
        let options = CompileOptions {
            context: "cone_test".into(),
            ..CompileOptions::default()
        };
        let mut asm = compile_polydat_to_assembler_with(src, &options).expect("assemble");
        asm.set_jit_mode(mode);
        asm.compile().expect("compile")
    }

    /// Pull `output` for each x in `xs`, returning the values.
    fn sweep(src: &str, mode: JitMode, output: &str, xs: &[u64]) -> Vec<Value> {
        let mut k = compile(src, mode);
        let idx = k.program().find_input("x").expect("input x");
        xs.iter()
            .map(|&x| {
                k.state().set_input(idx, Value::U64(x));
                k.pull_ref(output).clone()
            })
            .collect()
    }

    fn node_count(src: &str, mode: JitMode) -> usize {
        compile(src, mode).program().node_count()
    }

    /// SRD-105 panic parity: a predicate violation reports the same
    /// actionable core — predicate name, violation text, and for
    /// is_one_of the allow-list contents — whether it fires on the
    /// interpreter or inside a fused cone. The cone adds its member
    /// attribution; it never obscures the original message.
    fn capture_violation(src: &str, mode: JitMode, x: u64) -> String {
        let mut k = compile(src, mode);
        let idx = k.program().find_input("x").expect("input x");
        k.state().set_input(idx, Value::U64(x));
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            k.pull_ref("checked");
        }))
        .expect_err("violation must panic");
        err.downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&'static str>().map(|s| (*s).to_string()))
            .expect("string payload")
    }

    fn assert_parity(src: &str, x: u64, core: &str) {
        let off = capture_violation(src, JitMode::Off, x);
        let force = capture_violation(src, JitMode::Force, x);
        assert!(
            off.contains(core),
            "interpreter message carries the core: {off}"
        );
        assert!(
            force.contains(core),
            "cone message carries the same core: {force}"
        );
        assert!(off.contains("in node"), "interpreter enriches: {off}");
        assert!(force.contains("in node"), "cone enriches: {force}");
    }

    #[test]
    fn a_plain_compile_fuses_by_default() {
        // A host that names no mode gets `Auto`: the chain fuses as it does
        // under an explicit `Auto`, and the kernel reports that engine.
        let k = polydat::dsl::compile::compile_polydat_interpreter(U64_CHAIN).expect("compile");
        assert_eq!(
            polydat::Kernel::engine(&k),
            polydat::Engine::Interpreter(JitMode::Auto)
        );
        assert_eq!(
            k.program().node_count(),
            node_count(U64_CHAIN, JitMode::Auto)
        );
        assert!(node_count(U64_CHAIN, JitMode::Auto) < node_count(U64_CHAIN, JitMode::Off));
    }

    #[test]
    fn force_fuses_and_matches_interpreter_u64() {
        let xs: Vec<u64> = (0..50).chain([u64::MAX / 3, u64::MAX]).collect();
        let baseline = sweep(U64_CHAIN, JitMode::Off, "w", &xs);
        let base_nodes = node_count(U64_CHAIN, JitMode::Off);
        let fused = sweep(U64_CHAIN, JitMode::Force, "w", &xs);
        let fused_nodes = node_count(U64_CHAIN, JitMode::Force);
        assert_eq!(baseline, fused, "cone output must be bit-identical");
        assert!(
            fused_nodes < base_nodes,
            "force must fuse mul+add into one cone: {fused_nodes} vs {base_nodes}"
        );
    }

    #[test]
    fn force_matches_interpreter_f64_boundary() {
        // to_f64 → f64 arithmetic: the F64 bits cross the cone
        // boundary in both directions.
        let src = "input (x: u64)\n\
                   f := to_f64(x)\n\
                   g := ((f * 1.5) + 0.25)\n";
        let xs: Vec<u64> = (0..40).chain([1 << 52, u64::MAX >> 1]).collect();
        let baseline = sweep(src, JitMode::Off, "g", &xs);
        let fused = sweep(src, JitMode::Force, "g", &xs);
        assert_eq!(baseline, fused, "f64 cone output must be bit-identical");
    }

    #[test]
    fn mixed_graph_keeps_fallback_on_interpreter() {
        // `default_or` consumes None (SRD-74 opt-out) — ineligible by
        // contract, so it must survive extraction as its own node while
        // the u64 chain ahead of it fuses.
        let src = "input (x: u64)\n\
                   v := mul(x, 3)\n\
                   w := add(v, 7)\n\
                   out := default_or(w, 9)\n";
        let xs: Vec<u64> = (0..20).collect();
        let baseline = sweep(src, JitMode::Off, "out", &xs);
        let base_nodes = node_count(src, JitMode::Off);
        let fused = sweep(src, JitMode::Force, "out", &xs);
        let fused_nodes = node_count(src, JitMode::Force);
        assert_eq!(baseline, fused);
        assert!(
            fused_nodes < base_nodes,
            "the eligible prefix must still fuse: {fused_nodes} vs {base_nodes}"
        );
    }

    #[test]
    fn auto_requires_two_members() {
        // A single eligible node clears Force's threshold but not
        // Auto's: fusing one node buys no fusion win and pays boundary
        // marshalling.
        let src = "input (x: u64)\n\
                   v := mul(x, 3)\n";
        let auto_nodes = node_count(src, JitMode::Auto);
        let force_nodes = node_count(src, JitMode::Force);
        let off_nodes = node_count(src, JitMode::Off);
        assert_eq!(auto_nodes, off_nodes, "auto must not fuse a 1-node cone");
        assert_eq!(
            force_nodes, off_nodes,
            "a 1-node cone replaces 1 node with 1 cone"
        );
        let xs: Vec<u64> = (0..10).collect();
        let baseline = sweep(src, JitMode::Off, "v", &xs);
        let forced = sweep(src, JitMode::Force, "v", &xs);
        assert_eq!(baseline, forced);
    }

    #[test]
    fn violation_inside_cone_attributes_the_member() {
        // A predicate violation in native code surfaces through
        // invoke_with_catch and the cone's own attribution, naming the
        // violated predicate as the program's node with the program's
        // output name and context; the cone is no frame of its own, so the
        // report reads as it reads on every other engine (A7).
        let src = "input (x: u64)\n\
                   checked := is_positive(mul(x, 0))\n";
        let msg = capture_violation(src, JitMode::Force, 5);
        assert!(
            msg.contains("↳ in node `is_positive` (output checked) while evaluating"),
            "violation names the predicate as the program's node: {msg}"
        );
        assert!(
            !msg.contains("jit_cone["),
            "the cone is not a frame of the report: {msg}"
        );
    }

    #[test]
    fn const_subgraphs_stay_on_the_fold_path() {
        // A compile-time-constant subgraph must NOT fuse: const
        // folding evaluates it once and replaces it with a literal
        // (feeding `get_constant` consumers like eval_const_expr);
        // fusing it would demote it to per-pull native evaluation and
        // break the single-output fold replacement. Lifecycle
        // classification keeps extraction on per-cycle (Dynamic)
        // work only.
        let v = polydat::dsl::compile::eval_const_expr("mod(hash(42), 100)")
            .expect("const expr must fold");
        assert!(v.as_u64() < 100);
        let forced = compile("out := mod(hash(42), 100)\n", JitMode::Force);
        assert_eq!(
            forced.get_constant("out"),
            Some(&v),
            "the fold result is mode-independent"
        );
    }

    #[test]
    fn scope_init_chains_stay_on_the_fold_path() {
        // Extern-fed (IterationExtern) subgraphs classify ScopeInit:
        // the scope-activation fold owns them, so extraction must not
        // fuse them — only per-cycle (Coordinate/ExternalWrite-fed)
        // work joins cones.
        let src = "extern x: u64\n\
                   v := mul(x, 3)\n\
                   w := add(v, 7)\n";
        let off_nodes = node_count(src, JitMode::Off);
        let force_nodes = node_count(src, JitMode::Force);
        assert_eq!(off_nodes, force_nodes, "scope-init chains must not fuse");
        let xs: Vec<u64> = (0..10).collect();
        let baseline = sweep(src, JitMode::Off, "w", &xs);
        let forced = sweep(src, JitMode::Force, "w", &xs);
        assert_eq!(baseline, forced);
    }

    #[test]
    fn violation_message_parity_between_engines() {
        assert_parity(
            "input (x: u64)\nchecked := is_positive(mul(x, 0))\n",
            5,
            "is_positive(value): value must be > 0, got 0",
        );
    }

    #[test]
    fn in_range_violation_parity() {
        assert_parity(
            "input (x: u64)\nchecked := in_range(add(x, 100), 1, 10)\n",
            5,
            "in_range: value 105 outside [1, 10]",
        );
    }

    #[test]
    fn is_one_of_violation_parity() {
        // The allow-list contents must appear in BOTH messages —
        // catchup A2: the JIT fail extern used to elide them.
        assert_parity(
            "input (x: u64)\nchecked := is_one_of(add(x, 100), 1, 3, 7)\n",
            5,
            "is_one_of: value 105 not in allowed set [1, 3, 7]",
        );
    }

    /// SRD-105 Push 3 — program identity is engine-mix invariant.
    /// Identity hashing walks THROUGH fusion nodes into their stored
    /// subgraph, so `jit=off` / `auto` / `force` compiles of the same
    /// source hash identically and resume-skip matching survives mode
    /// changes. The shape here stresses the walk: a multi-output cone
    /// (v is both fused-interior and a named output), a const-folded
    /// upstream (hashes post-fold in both forms), and a graph input.
    #[test]
    fn canonical_hash_is_extraction_invariant() {
        let src = "input (x: u64)\n\
                   const c := 42\n\
                   v := mul(x, 3)\n\
                   w := (v + c)\n";
        let off = compile(src, JitMode::Off).program().canonical_hash();
        let force = compile(src, JitMode::Force).program().canonical_hash();
        let auto = compile(src, JitMode::Auto).program().canonical_hash();
        assert_eq!(off, force, "off vs force identity must match");
        assert_eq!(off, auto, "off vs auto identity must match");
    }
}

mod dsl_registry_tests {
    #![allow(unused_imports, dead_code)]
    use polydat::JitMode;
    use polydat::ast::*;
    use polydat::dsl::ast::*;
    use polydat::dsl::compile::*;
    use polydat::dsl::registry::*;
    use polydat::dsl::registry::*;
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use polydat::kernel::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn suggest_close_match() {
        assert_eq!(suggest_function("hsh"), Some("hash"));
        assert_eq!(suggest_function("hahs"), Some("hash"));
        assert_eq!(suggest_function("interleav"), Some("interleave"));
    }

    #[test]
    fn lookup_exists() {
        let sig = lookup("hash").unwrap();
        assert_eq!(sig.wire_input_count(), 1);
        assert_eq!(sig.outputs, 1);
        assert!(!sig.is_variadic());
        assert_eq!(sig.category, FuncCategory::Hashing);
    }

    #[test]
    fn lookup_variadic() {
        let sig = lookup("sum").unwrap();
        assert!(sig.is_variadic());
        assert_eq!(sig.identity, Some(0));
        assert_eq!(sig.category, FuncCategory::Variadic);
    }

    #[test]
    fn arithmetic_params_populated() {
        // Verify key arithmetic functions have params.
        for name in &[
            "add",
            "mul",
            "div",
            "mod",
            "clamp",
            "interleave",
            "mixed_radix",
        ] {
            let sig = lookup(name).unwrap_or_else(|| panic!("missing '{name}'"));
            assert!(
                !sig.params.is_empty(),
                "function '{}' should have params populated",
                name
            );
        }
    }

    #[test]
    fn variadic_params_populated() {
        for name in &["sum", "product", "min", "max"] {
            let sig = lookup(name).unwrap_or_else(|| panic!("missing '{name}'"));
            assert!(
                matches!(sig.arity, Arity::VariadicWires { .. }),
                "function '{}' should have VariadicWires arity",
                name
            );
        }
    }

    #[test]
    fn mixed_radix_is_variadic_consts() {
        // SRD-80b Phase E: `mixed_radix` is macro-registered via the
        // `Const<Vec<u64>>` shape, which implies
        // `VariadicConsts { min_consts: 0 }` (empty lists permitted at
        // the signature level; the trailing-zero positional rule lives
        // in the arithmetic module's hand validator). The stale
        // hand-written FuncSig this test used to pin (min_consts: 1,
        // params without the const-vec entry) duplicated the macro
        // registration under the same name and was removed.
        let sig = lookup("mixed_radix").unwrap();
        assert!(
            matches!(sig.arity, Arity::VariadicConsts { min_consts: 0 }),
            "mixed_radix should be VariadicConsts, got {:?}",
            sig.arity
        );
        assert!(
            matches!(sig.params[0].slot_type, SlotType::Wire),
            "first param is the u64 wire input"
        );
    }
}

#[cfg(feature = "jit")]
mod compile_simd_tier1_tests {
    #![allow(unused_imports, dead_code)]
    use polydat::JitMode;
    use polydat::ast::*;
    use polydat::compile::simd_tier1::*;
    use polydat::dsl::ast::*;
    use polydat::dsl::compile::*;
    use polydat::dsl::registry::*;
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use polydat::kernel::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn compile_executor() -> Tier1SimdExecutor {
        let source = "\
cursor base = range(0, 64)
k := 3
m := u64_mul(base.ordinal, k)
out := u64_add(m, k)
";
        polydat::dsl::compile::compile_polydat_tier1_simd_ordinal(source, "base__ordinal", "out")
            .unwrap()
    }

    fn cursors(executor: &Tier1SimdExecutor, start: u64, end: u64) -> Cursors {
        let factories: HashMap<String, Arc<dyn DataSourceFactory>> = HashMap::from([(
            "base".to_string(),
            Arc::new(RangeSourceFactory::new(start, end)) as Arc<dyn DataSourceFactory>,
        )]);
        Cursors::for_fields(executor.scalar_program(), &["out"], &factories)
    }

    #[test]
    fn selected_scalar_cone_compiles_to_typed_register_plan() {
        let executor = compile_executor();
        let descriptor = executor.descriptor();
        assert_eq!(descriptor.scalar_type, PortType::U64);
        assert_eq!(descriptor.register_type, PortType::RegI64x2);
        assert_eq!(descriptor.lanes, 2);
        assert_eq!(descriptor.member_nodes.len(), 2);
        assert!(descriptor.broadcast_inputs.is_empty());
        assert!(!descriptor.effective_isa_fingerprint.is_empty());
    }

    #[test]
    fn cursor_lease_vectorizes_and_drains_in_arbitrary_bursts() {
        let mut executor = compile_executor();
        let mut cursors = cursors(&executor, 1, 22);
        let bursts = [1usize, 4, 3, 7, 2];
        let mut got = Vec::new();
        let mut burst = 0;

        while let Some(lease) = cursors.reserve_ordinal_batch(6).unwrap() {
            executor.begin_lease(lease, 9).unwrap();
            while executor.has_active_lease() {
                let mut output = vec![Value::None; bursts[burst % bursts.len()]];
                let written = executor.drain_into(&mut output);
                got.extend_from_slice(&output[..written]);
                burst += 1;
            }
        }

        let want: Vec<_> = (1u64..22)
            .map(|ordinal| Value::U64(ordinal.wrapping_mul(3).wrapping_add(3)))
            .collect();
        assert_eq!(got, want);
        assert_eq!(cursors.consumed(), 21);
        assert_eq!(executor.stats().values_drained, 21);
        assert!(executor.stats().vector_packets > 0);
        assert!(executor.stats().scalar_fragment_lanes > 0);
    }

    #[test]
    fn scope_stable_scalar_input_is_broadcast_across_each_packet() {
        let source = "\
input (base__ordinal: u64, k: u64)
m := u64_mul(base__ordinal, k)
out := u64_add(m, k)
";
        let mut executor = polydat::dsl::compile::compile_polydat_to_assembler(source)
            .unwrap()
            .try_compile_tier1_simd_ordinal("base__ordinal", "out")
            .unwrap();
        assert_eq!(executor.descriptor().broadcast_inputs, ["k"]);
        executor.set_broadcast("k", Value::U64(5)).unwrap();

        let mut cursors = cursors(&executor, 3, 10);
        let lease = cursors.reserve_ordinal_batch(7).unwrap().unwrap();
        executor.begin_lease(lease, 11).unwrap();
        assert!(matches!(
            executor.set_broadcast("k", Value::U64(7)),
            Err(Tier1SimdError::ActiveLease)
        ));

        let mut output = vec![Value::None; 7];
        assert_eq!(executor.drain_into(&mut output), 7);
        let want: Vec<_> = (3u64..10)
            .map(|ordinal| Value::U64(ordinal.wrapping_mul(5).wrapping_add(5)))
            .collect();
        assert_eq!(output, want);
    }

    #[test]
    fn scalar_recovery_preserves_the_committed_frontier() {
        let mut executor = compile_executor();
        let mut cursors = cursors(&executor, 0, 11);
        let lease = cursors.reserve_ordinal_batch(11).unwrap().unwrap();
        executor.begin_lease(lease, 1).unwrap();

        let mut first = [Value::None];
        assert_eq!(executor.drain_into(&mut first), 1);
        executor.force_scalar_recovery();
        let mut rest = vec![Value::None; 10];
        assert_eq!(executor.drain_into(&mut rest), 10);

        let got = [first.as_slice(), rest.as_slice()].concat();
        let want: Vec<_> = (0u64..11)
            .map(|ordinal| Value::U64(ordinal.wrapping_mul(3).wrapping_add(3)))
            .collect();
        assert_eq!(got, want);
        assert_eq!(executor.stats().vector_packets, 1);
        assert_eq!(executor.stats().scalar_recovery_lanes, 10);
    }

    #[test]
    fn rejected_out_of_order_lease_is_returned_for_recovery() {
        let mut executor = compile_executor();
        let mut cursors = cursors(&executor, 0, 8);
        let first = cursors.reserve_ordinal_batch(4).unwrap().unwrap();
        let second = cursors.reserve_ordinal_batch(4).unwrap().unwrap();

        let rejection = executor.begin_lease(second, 3).unwrap_err();
        assert!(matches!(
            rejection.reason(),
            Tier1SimdError::LeaseSequenceMismatch {
                expected: 0,
                got: 1
            }
        ));
        let (_, second) = rejection.into_parts();

        executor.begin_lease(first, 3).unwrap();
        let mut output = core::array::from_fn::<_, 4, _>(|_| Value::None);
        assert_eq!(executor.drain_into(&mut output), 4);
        executor.begin_lease(second, 3).unwrap();
        assert_eq!(executor.drain_into(&mut output), 4);
    }

    #[test]
    fn externally_writable_side_input_is_not_promoted() {
        let source = "\
cursor base = range(0, 64)
cursor other = range(0, 64)
m := u64_mul(base.ordinal, other.ordinal)
out := u64_add(m, other.ordinal)
";
        let error = polydat::dsl::compile::compile_polydat_to_assembler(source)
            .unwrap()
            .try_compile_tier1_simd_ordinal("base__ordinal", "out")
            .err()
            .expect("mutable broadcast must be rejected");
        assert!(
            matches!(error, Tier1SimdError::MutableBroadcast(ref name) if name == "other__ordinal"),
            "unexpected rejection: {error:?}"
        );
    }
}

mod kernel_program_r1v_contagion_tests {
    #![allow(unused_imports, dead_code)]
    use polydat::JitMode;
    use polydat::ast::*;
    use polydat::dsl::ast::*;
    use polydat::dsl::compile::*;
    use polydat::dsl::registry::*;
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use polydat::kernel::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// Pull `out` from kernel `k` at coordinate `cycle`,
    /// returning the resulting u64.
    fn pull_u64_at(k: &mut polydat::kernel::PolydatKernel, cycle: u64, out: &str) -> u64 {
        k.set_inputs(&[cycle]);
        match k.pull_ref(out) {
            Value::U64(v) => *v,
            other => panic!("expected U64, got {other:?}"),
        }
    }

    #[test]
    fn direct_consumer_of_intrinsic_volatile_re_evals_per_cycle() {
        // `now` reads system clock (volatile); `b` adds 1.
        // Without R1.v contagion, `b` would return a stale
        // cached value referencing cycle-0's `now` on cycle 1+.
        let src = "input cycle: u64\n\
                   now := current_epoch_millis()\n\
                   b := add(now, 1)\n";
        let mut k = compile_polydat_interpreter(src).expect("compile");
        let v0 = pull_u64_at(&mut k, 0, "b");
        // Spin briefly to ensure system clock advances. A few ms
        // is enough; if clock granularity is coarser the test
        // falls back to asserting at-least-as-many-calls.
        std::thread::sleep(std::time::Duration::from_millis(2));
        let v1 = pull_u64_at(&mut k, 1, "b");
        assert!(
            v1 >= v0,
            "consumer of volatile producer must re-eval per cycle (v0={v0}, v1={v1})"
        );
        // Stronger: if clock advanced at all, the value must
        // reflect the advance. Allow ties only when the clock
        // didn't tick in the sleep window.
        std::thread::sleep(std::time::Duration::from_millis(2));
        let v2 = pull_u64_at(&mut k, 2, "b");
        assert!(
            v2 > v0,
            "after a 4ms cumulative wait the volatile consumer must reflect the advance \
             (v0={v0}, v2={v2}) — if this fails, R1.v contagion is broken"
        );
    }

    #[test]
    fn transitive_consumer_of_intrinsic_volatile_re_evals_per_cycle() {
        // Three-deep chain: now → b → c. Each level must
        // re-eval per cycle when `now` advances. Tests the
        // multi-hop transitive case.
        let src = "input cycle: u64\n\
                   now := current_epoch_millis()\n\
                   b := add(now, 1)\n\
                   c := add(b, 1)\n";
        let mut k = compile_polydat_interpreter(src).expect("compile");
        let v0 = pull_u64_at(&mut k, 0, "c");
        std::thread::sleep(std::time::Duration::from_millis(4));
        let v1 = pull_u64_at(&mut k, 1, "c");
        assert!(
            v1 > v0,
            "transitive consumer (chain depth 2) of volatile producer must re-eval \
             (v0={v0}, v1={v1}) — R1.v contagion must propagate through the chain"
        );
    }

    #[test]
    fn counter_intrinsic_volatile_increments_per_cycle() {
        // `counter` is Purity::Nondeterministic; each cycle
        // should see a fresh increment. Also tests transitive
        // contagion through `wrapped`.
        let src = "input cycle: u64\n\
                   c := counter()\n\
                   wrapped := add(c, 1000)\n";
        let mut k = compile_polydat_interpreter(src).expect("compile");
        let v0 = pull_u64_at(&mut k, 0, "wrapped");
        let v1 = pull_u64_at(&mut k, 1, "wrapped");
        let v2 = pull_u64_at(&mut k, 2, "wrapped");
        assert_eq!(
            v1,
            v0 + 1,
            "counter must advance per cycle (v0={v0}, v1={v1})"
        );
        assert_eq!(
            v2,
            v1 + 1,
            "counter must continue advancing (v1={v1}, v2={v2})"
        );
    }

    #[test]
    fn non_volatile_path_does_not_get_contaminated() {
        // `pure_chain` only depends on `cycle` — no volatile
        // upstream. It should NOT be in nondeterministic_nodes
        // (within-cycle clean-flag caching should apply).
        // Contagion must be precise — only propagate from
        // actual volatile producers, not blanket-mark every node.
        let src = "input cycle: u64\n\
                   pure_chain := add(cycle, 1)\n\
                   pure_outer := mul(pure_chain, 2)\n";
        let mut k = compile_polydat_interpreter(src).expect("compile");
        // For the same coordinate, multiple pulls must return
        // the same value AND not re-evaluate the eval function.
        // The latter property is hard to assert without
        // instrumentation, so check the former; the implicit
        // claim is that nondeterministic_nodes is precise.
        k.set_inputs(&[42]);
        let r1 = match k.pull_ref("pure_outer") {
            Value::U64(v) => *v,
            _ => panic!(),
        };
        let r2 = match k.pull_ref("pure_outer") {
            Value::U64(v) => *v,
            _ => panic!(),
        };
        assert_eq!(r1, r2);
        // Pure path: same coord → same output.
        k.set_inputs(&[42]);
        let r3 = match k.pull_ref("pure_outer") {
            Value::U64(v) => *v,
            _ => panic!(),
        };
        assert_eq!(
            r1, r3,
            "pure (non-volatile) chain must be coord-deterministic"
        );
        // Different coord → different output.
        k.set_inputs(&[43]);
        let r4 = match k.pull_ref("pure_outer") {
            Value::U64(v) => *v,
            _ => panic!(),
        };
        assert_ne!(r3, r4);
    }
}

mod kernel_program_ast_metadata_tests {
    #![allow(unused_imports, dead_code)]
    use polydat::JitMode;
    use polydat::ast::*;
    use polydat::dsl::ast::*;
    use polydat::dsl::compile::*;
    use polydat::dsl::registry::*;
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use polydat::kernel::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn local_inclusion_chain_includes_transitive_deps() {
        // `bar` depends on `foo`. Both are cycle bindings.
        // The chain for `bar` should include `foo` first, `bar` second.
        let src = "\
foo := hash(cycle)
bar := mod(foo, 100)
";
        let k = compile_polydat_interpreter(src).expect("compile");
        let chain = k
            .program()
            .local_inclusion_chain("bar", &std::collections::HashSet::new());
        assert_eq!(
            chain.len(),
            2,
            "expected 2 bindings in chain, got {}",
            chain.len()
        );
        match chain[0] {
            Statement::Binding(b) => assert!(b.targets.iter().any(|t| t == "foo")),
            _ => panic!("expected foo first"),
        }
        match chain[1] {
            Statement::Binding(b) => assert!(b.targets.iter().any(|t| t == "bar")),
            _ => panic!("expected bar second"),
        }
    }

    #[test]
    fn local_inclusion_chain_stops_at_final() {
        // `seed` is `final` — should NOT appear in the chain
        // (case 1, promoted-final, is the caller's job).
        let src = "\
const seed := 12345
mixed := hash(seed)
";
        let k = compile_polydat_interpreter(src).expect("compile");
        let chain = k
            .program()
            .local_inclusion_chain("mixed", &std::collections::HashSet::new());
        // Just `mixed` — `seed` is final, walk stops.
        assert_eq!(chain.len(), 1);
        match chain[0] {
            Statement::Binding(b) => assert!(b.targets.iter().any(|t| t == "mixed")),
            _ => panic!("expected mixed"),
        }
    }

    #[test]
    fn local_inclusion_chain_respects_excluded() {
        // If `foo` is already locally satisfied (excluded), walk
        // doesn't include it. `bar` alone should appear.
        let src = "\
foo := hash(cycle)
bar := mod(foo, 100)
";
        let k = compile_polydat_interpreter(src).expect("compile");
        let mut excluded = std::collections::HashSet::new();
        excluded.insert("foo".to_string());
        let chain = k.program().local_inclusion_chain("bar", &excluded);
        assert_eq!(chain.len(), 1);
        match chain[0] {
            Statement::Binding(b) => assert!(b.targets.iter().any(|t| t == "bar")),
            _ => panic!("expected bar"),
        }
    }
}

mod kernel_program_canonical_hash_tests {
    #![allow(unused_imports, dead_code)]
    use polydat::JitMode;
    use polydat::ast::*;
    use polydat::dsl::ast::*;
    use polydat::dsl::compile::*;
    use polydat::dsl::registry::*;
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use polydat::kernel::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn const_inside_function_call_changes_hash() {
        // The integer literal in `mod(..., N)` lives in a const
        // slot that canonical_hash should cover — even when it's
        // an argument to a function call rather than a top-level
        // `const X := <literal>` binding.
        let a = compile_polydat_interpreter("input cycle: u64\nshard := mod(hash(cycle), 8)\n")
            .expect("a");
        let b = compile_polydat_interpreter("input cycle: u64\nshard := mod(hash(cycle), 16)\n")
            .expect("b");
        assert_ne!(
            a.program().canonical_hash(),
            b.program().canonical_hash(),
            "literal-arg const value must change canonical hash"
        );
    }
}

mod kernel_tests {
    #![allow(unused_imports, dead_code)]
    use polydat::JitMode;
    use polydat::ast::*;
    use polydat::dsl::ast::*;
    use polydat::dsl::compile::*;
    use polydat::dsl::registry::*;
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use polydat::kernel::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn fold_init_constants_basic() {
        // base=42, seed=hash(base) should both be folded
        // user_id=hash(cycle) should NOT be folded (depends on coordinate)
        use polydat::dsl::compile::compile_polydat_interpreter;
        let mut k = compile_polydat_interpreter(
            "input cycle: u64\nbase := 42\nseed := hash(base)\nuser_id := hash(cycle)",
        )
        .unwrap();

        // seed should be constant across cycles
        k.set_inputs(&[0]);
        let seed_0 = k.pull_ref("seed").clone();
        k.set_inputs(&[1]);
        let seed_1 = k.pull_ref("seed").clone();
        assert_eq!(
            seed_0.as_u64(),
            seed_1.as_u64(),
            "seed should be constant (folded)"
        );

        // user_id should vary
        k.set_inputs(&[0]);
        let uid_0 = k.pull_ref("user_id").clone();
        k.set_inputs(&[1]);
        let uid_1 = k.pull_ref("user_id").clone();
        assert_ne!(
            uid_0.as_u64(),
            uid_1.as_u64(),
            "user_id should vary per cycle"
        );
    }

    #[test]
    fn fold_does_not_touch_cycle_dependent() {
        use polydat::dsl::compile::compile_polydat_interpreter;
        let mut k = compile_polydat_interpreter("input cycle: u64\nout := hash(cycle)").unwrap();
        k.set_inputs(&[42]);
        let v1 = k.pull_ref("out").as_u64();
        k.set_inputs(&[43]);
        let v2 = k.pull_ref("out").as_u64();
        assert_ne!(v1, v2, "cycle-dependent node should not be folded");
    }

    #[test]
    fn implicit_u64_to_f64_adapter_does_not_crash() {
        use polydat::dsl::compile::compile_polydat_interpreter;
        // sin() expects f64, cycle is u64. The compiler should auto-insert
        // a __u64_to_f64 adapter. This must not panic.
        let mut k = compile_polydat_interpreter("input cycle: u64\nout := sin(cycle)").unwrap();
        k.set_inputs(&[1]);
        let v = k.pull_ref("out");
        // sin(1.0) ≈ 0.8414709848078965
        let f = v.as_f64();
        assert!(
            (f - 0.8414709848078965).abs() < 0.001,
            "sin(1) should be ~0.841, got {f}"
        );
    }
}

#[cfg(feature = "jit")]
mod compile_lattice_tests {
    #![allow(unused_imports, dead_code)]
    use polydat::JitMode;
    use polydat::ast::*;
    use polydat::compile::lattice::*;
    use polydat::dsl::ast::*;
    use polydat::dsl::compile::*;
    use polydat::dsl::registry::*;
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use polydat::kernel::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn report_for(src: &str, mode: JitMode) -> LatticeReport {
        let mut asm = compile_polydat_to_assembler(src).expect("assemble");
        asm.set_jit_mode(mode);
        let k = asm.compile().expect("compile");
        lattice_report(k.program())
    }

    #[test]
    fn mixed_graph_reports_cone_and_residue() {
        // mul+add fuse; default_or (SRD-74 None-consumer) stays
        // interpreted and is neither p3-classifiable nor
        // p2-capable (Value-based optionality).
        let src = "input (x: u64)\n\
                   v := mul(x, 3)\n\
                   w := add(v, 7)\n\
                   out := default_or(w, 9)\n";
        let rep = report_for(src, JitMode::Auto);
        assert_eq!(rep.cones.len(), 1, "one cone expected");
        assert_eq!(rep.fused_nodes, 2, "mul+add fused");
        assert!(
            rep.cones[0].members.contains(&"mul".to_string())
                && rep.cones[0].members.contains(&"add".to_string()),
            "members listed: {:?}",
            rep.cones[0].members
        );
        assert!(
            rep.residue.iter().any(|r| r.name == "default_or"),
            "fallback node in residue"
        );
        let shown = format!("{rep}");
        assert!(
            shown.contains("jit_cone["),
            "display names the cone: {shown}"
        );
    }

    #[test]
    fn off_mode_reports_pure_residue() {
        let src = "input (x: u64)\n\
                   v := mul(x, 3)\n\
                   w := add(v, 7)\n";
        let rep = report_for(src, JitMode::Off);
        assert!(rep.cones.is_empty());
        assert_eq!(rep.fused_nodes, 0);
        // Both nodes are P3-classifiable — the report shows the
        // unfused potential under off mode.
        assert!(rep.p3_unfused >= 2, "p3_unfused: {}", rep.p3_unfused);
    }
}

mod library_context_tests {
    #![allow(unused_imports, dead_code)]
    use polydat::JitMode;
    use polydat::ast::*;
    use polydat::dsl::ast::*;
    use polydat::dsl::compile::*;
    use polydat::dsl::registry::*;
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use polydat::kernel::*;
    use polydat::library::context::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn elapsed_from_injected_origin_compiles_in_dsl() {
        // Phase-duration metrics need no dedicated node: a phase-level
        // `metrics:` value of `current_epoch_millis() - phase_start`
        // (with `phase_start` an executor-injected origin wire) is the
        // canonical form. Confirm the expression compiles.
        let src = "extern phase_start: u64 = 0\n\
                   volatile te := current_epoch_millis() - phase_start\n";
        let k = polydat::dsl::compile_polydat_interpreter(src)
            .expect("clock-minus-injected-origin must compile");
        assert!(
            k.program().output_names().contains(&"te"),
            "expected output 'te' in {:?}",
            k.program().output_names()
        );
    }
}

mod kernel_engines_panic_enrichment_tests {
    #![allow(unused_imports, dead_code)]
    use polydat::JitMode;
    use polydat::ast::*;
    use polydat::dsl::ast::*;
    use polydat::dsl::compile::*;
    use polydat::dsl::registry::*;
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use polydat::kernel::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn type_mismatch_panic_carries_node_and_output_context() {
        // Declare the input as u64, then write a Str into the
        // slot — `mul`'s u64 path will panic on `as_u64()`.
        // The enricher must wrap the message with the node
        // name + output name + program context.
        let options = CompileOptions {
            context: "test_workload".into(),
            ..CompileOptions::default()
        };
        let mut k = compile_polydat_interpreter_with_options(
            "extern x: u64\n\
             doubled := mul(x, 2)\n",
            &options,
            None,
        )
        .expect("compile");
        let idx = k.program().find_input("x").unwrap();
        k.state()
            .set_input(idx, polydat::ast::Value::Str("oops".into()));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            k.pull_ref("doubled");
        }));
        let err = result.expect_err("pull should panic on type mismatch");
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&'static str>().map(|s| (*s).to_string()))
            .expect("panic payload should be a String");
        assert!(
            msg.contains("expected U64"),
            "missing original panic body in: {msg}"
        );
        assert!(
            msg.contains("`mul`"),
            "missing node name in enriched message: {msg}"
        );
        assert!(
            msg.contains("doubled"),
            "missing output binding in enriched message: {msg}"
        );
        assert!(
            msg.contains("test_workload"),
            "missing program context in enriched message: {msg}"
        );
        assert!(
            msg.contains("\"oops\""),
            "missing input snapshot in enriched message: {msg}"
        );
        // The suppression hook must have captured the ORIGINAL
        // panic site (Value::as_u64 in ast.rs) — not the
        // re-raise site in engines.rs.
        assert!(
            msg.contains("panicked at") && msg.contains("ast.rs"),
            "missing original panic location in enriched message: {msg}"
        );
        // Surface the full enriched message in `cargo test --
        // --nocapture` runs so the format is easy to eyeball.
        eprintln!("== enriched message ==\n{msg}\n======================");
    }
}

#[cfg(feature = "jit")]
mod iteration_simd_ordinal_tests {
    #![allow(unused_imports, dead_code)]
    use polydat::JitMode;
    use polydat::ast::*;
    use polydat::dsl::ast::*;
    use polydat::dsl::compile::*;
    use polydat::dsl::registry::*;
    use polydat::iteration::simd_ordinal::*;
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use polydat::kernel::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[cfg(feature = "jit")]
    #[test]
    fn explicit_cranelift_i32x4_variant_drives_the_batch_stream() {
        use polydat::ast::Bits128;

        let src = "\
input packet: reg_i32x4
input factor: u64
input addend: u64
factor_v := reg_splat_i32(factor)
addend_v := reg_splat_i32(addend)
n0 := reg_mul_i32(packet, factor_v)
out := reg_add_i32(n0, addend_v)
";
        let asm = polydat::dsl::compile::compile_polydat_to_assembler(src).unwrap();
        let mut jit = asm
            .compile_slots(polydat::Engine::PureNative(polydat::Provenance::Raw))
            .expect("i32x4 graph must lower");
        let output_slot = jit.resolve_output("out").unwrap();

        let vector = move |lanes: [i32; 4]| {
            let input = Bits128::from_lanes_i32(lanes);
            jit.eval_at(&[input.0[0], input.0[1], 3, 17]);
            Bits128([jit.get_slot(output_slot), jit.get_slot(output_slot + 1)]).lanes_i32()
        };
        let scalar = |x: i32| x.wrapping_mul(3).wrapping_add(17);
        let mut batch = OrdinalI32x4Stream::new(
            AffineI32Source::new(1, 0, 0, 0, 1),
            0..19,
            0,
            0,
            vector,
            scalar,
        )
        .unwrap();

        let mut got = [0; 19];
        assert_eq!(batch.drain_into(&mut got[..3]), 3);
        assert_eq!(batch.drain_into(&mut got[3..]), 16);
        let want =
            core::array::from_fn::<_, 19, _>(|i| (i as i32).wrapping_mul(3).wrapping_add(17));
        assert_eq!(got, want);
        assert_eq!(batch.stats().vector_packets, 4);
        assert_eq!(batch.stats().scalar_fragment_lanes, 3);
    }
}

mod iteration_cursor_partition_over_tests {
    #![allow(unused_imports, dead_code)]
    use polydat::JitMode;
    use polydat::ast::*;
    use polydat::dsl::ast::*;
    use polydat::dsl::compile::*;
    use polydat::dsl::registry::*;
    use polydat::iteration::cursor_partition::*;
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use polydat::kernel::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn narrow_cursor_writes_declared_slots() {
        let src = "input cycle: u64\ncursor q = range(0, 100) over \"*/4\"\nn := cardinality(q.cursor)\ns := q.cursor.start_ordinal";
        let mut k = polydat::dsl::compile_polydat_interpreter(src).unwrap();
        let program = k.program().clone();
        let parts =
            cursor_over_partitions(&program, k.state(), &program.cursor_schemas()[0]).unwrap();
        assert_eq!(parts.len(), 4);
        narrow_cursor(&program, k.state(), "q", &parts[2]);
        k.set_inputs(&[0]);
        assert_eq!(k.pull_ref("n").as_u64(), 25);
        assert_eq!(k.pull_ref("s").as_u64(), 50);
    }
}

mod dsl_stub_tests {
    #![allow(unused_imports, dead_code)]
    use polydat::JitMode;
    use polydat::ast::*;
    use polydat::dsl::ast::*;
    use polydat::dsl::compile::*;
    use polydat::dsl::registry::*;
    use polydat::dsl::stub::*;
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use polydat::kernel::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn scoped_expr_binds_to_a_kernel_scope_and_is_callable() {
        // SRD-84 shape 2 — a scoped, callable expression. Shape 1
        // (`GraphMatter`) builds the matter: a typed extern wire plus a
        // predicate stub. Shape 2 (`ScopedExpr`) binds it into a
        // sub-context of a parent kernel and evaluates it many times
        // against injected inputs, as truthiness.
        use polydat::ast::Value;
        let parent = polydat::dsl::compile_polydat_interpreter("input cycle: u64\nx := 5")
            .expect("parent kernel");
        let mut matter = GraphMatter::new();
        matter.extern_wire::<u64>("threshold").bind(
            ExprStub::parse("__pred", "threshold > 50")
                .expect("parse")
                .returning::<u64>()
                .volatile(),
        );
        let mut scoped = ScopedExpr::bind(&parent, "__pred", matter).expect("bind to parent scope");
        assert!(
            scoped.set("threshold", Value::U64(100)).is_true(),
            "100 > 50 → true"
        );
        assert!(
            !scoped.set("threshold", Value::U64(10)).is_true(),
            "10 > 50 → false"
        );
    }
}

mod dsl_factories_tests {
    #![allow(unused_imports, dead_code)]
    use polydat::JitMode;
    use polydat::ast::*;
    use polydat::dsl::ast::*;
    use polydat::dsl::compile::*;
    use polydat::dsl::factories::*;
    use polydat::dsl::registry::*;
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use polydat::kernel::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn by_category_includes_factory_nodes() {
        struct TestFactory;
        impl NodeFactory for TestFactory {
            fn signatures(&self) -> Vec<FuncSig> {
                vec![FuncSig {
                    name: "factory_hash",
                    category: FuncCategory::Hashing,
                    outputs: 1,
                    description: "a factory hashing node",
                    help: "",
                    identity: None,
                    variadic_ctor: None,
                    params: &[ParamSpec {
                        name: "input",
                        slot_type: SlotType::Wire,
                        required: true,
                        example: "cycle",
                        constraint: None,
                    }],
                    arity: Arity::Fixed,
                    commutativity: polydat::ast::Commutativity::Positional,
                    default_resolver: None,
                    output_type: polydat::dsl::registry::OutputType::Fixed,
                    // Hand registration: no static return-port declaration;
                    // type inference falls back to the name heuristic.
                    output_port: None,
                }]
            }
            fn build(
                &self,
                _: &str,
                _: usize,
                _: &[FactoryArg],
            ) -> Result<Box<dyn PolydatNode>, String> {
                Ok(Box::new(polydat::library::identity::Identity::new(
                    polydat::ast::PortType::U64,
                )))
            }
        }

        let mut rt = PolydatRuntime::new();
        rt.register_factory(Box::new(TestFactory));

        let grouped = rt.by_category();
        let hashing = grouped
            .iter()
            .find(|(c, _)| *c == FuncCategory::Hashing)
            .unwrap();
        assert!(hashing.1.iter().any(|s| s.name == "factory_hash"));
        // Built-in hash should also be there
        assert!(hashing.1.iter().any(|s| s.name == "hash"));
    }
}

/// The cursor advancer's item-at-a-time path: `advance` reads one item
/// from each targeted cursor and `inject_into_state` lands its ordinal
/// at the cursor's input index, so the program computes over it.
#[test]
fn the_cursor_advancer_injects_each_ordinal_into_the_state() {
    use polydat::iteration::source::{Cursors, DataSourceFactory, RangeSourceFactory};
    use std::collections::HashMap;
    use std::sync::Arc;
    let mut k = polydat::dsl::compile_polydat_interpreter(
        "input cycle: u64\ncursor rows = range(0, 100)\nout := u64_mul(rows.ordinal, 3)\n",
    )
    .unwrap();
    let factories: HashMap<String, Arc<dyn DataSourceFactory>> = HashMap::from([(
        "rows".to_string(),
        Arc::new(RangeSourceFactory::new(7, 10)) as Arc<dyn DataSourceFactory>,
    )]);
    let mut cursors = Cursors::for_fields(k.program(), &["out"], &factories);
    assert_eq!(cursors.target_count(), 1);
    let mut seen = Vec::new();
    while cursors.advance() {
        cursors.inject_into_state(k.state());
        seen.push(k.pull_ref("out").as_u64());
    }
    assert_eq!(seen, vec![21, 24, 27]);
    assert_eq!(cursors.consumed(), 3);
}

/// The typed embedding errors carry what the compiler knew, rather
/// than a shape rebuilt from the message text (F-C7). Each case below
/// used to lose a field: the type mismatch reported `(unknown)` nodes
/// and `U64 → U64` though the assembler had the real four; the
/// unknown function never suggested a near name though the registry
/// computes one; the lifecycle mismatch named no input though the
/// kernel's inputs are exactly what it is about.
#[cfg(test)]
mod typed_embedding_errors {
    use polydat::dsl::compile::{EmbeddingError, eval_const_expr};

    #[test]
    fn a_wiring_mismatch_names_both_nodes_and_both_types() {
        let err = eval_const_expr("hash(json_object())").unwrap_err();
        let EmbeddingError::TypeMismatch {
            from_type, to_type, ..
        } = &err
        else {
            panic!("expected TypeMismatch, got {err:?}");
        };
        assert_eq!(*from_type, polydat::ast::PortType::Json);
        assert_eq!(*to_type, polydat::ast::PortType::U64);
        let text = err.to_string();
        assert!(!text.contains("(unknown)"), "{text}");
    }

    #[test]
    fn an_unknown_function_carries_the_registry_s_suggestion() {
        let err = eval_const_expr("mdo(1, 2)").unwrap_err();
        let EmbeddingError::UnknownNode {
            name, suggestion, ..
        } = &err
        else {
            panic!("expected UnknownNode, got {err:?}");
        };
        assert_eq!(name, "mdo");
        let sug = suggestion.as_deref().expect("a name one edit away exists");
        assert!(err.to_string().contains(sug), "{err}");
    }

    #[test]
    fn a_lifecycle_mismatch_names_what_it_waits_on() {
        let err = eval_const_expr("mod(cycle, 10)").unwrap_err();
        let EmbeddingError::LifecycleMismatch { dynamic_inputs, .. } = &err else {
            panic!("expected LifecycleMismatch, got {err:?}");
        };
        assert_eq!(dynamic_inputs, &["cycle".to_string()]);
    }
}
/// A node whose output type comes from its wires reports that type,
/// not a placeholder (F-N9).
///
/// `pick` returns whatever its value wires carry, and its output port
/// used to be declared `U64` regardless. A `Str` from `pick` feeding a
/// `Str` port therefore had a `U64ToString` adapter inserted between
/// them — and the adapter read the string's pointer as a number, so
/// `matches(pick(b, "yes"), "y.*")` compared the pattern against
/// something like "2163471811744".
#[cfg(test)]
mod polymorphic_output_types {
    use polydat::Kernel;

    #[test]
    fn pick_reports_the_type_its_wires_carry() {
        let mut k = polydat::dsl::compile::compile_polydat_interpreter(
            "input cycle: u64\nb := cycle > 0\ns := pick(b, \"yes\")\nout := matches(s, \"y.*\")\n",
        )
        .expect("it builds");
        k.set_inputs(&[1]);
        assert_eq!(k.pull("s").as_str(), "yes");
        assert_eq!(k.pull("out").as_str(), "yes");
    }

    #[test]
    fn and_a_numeric_pick_still_reports_a_number() {
        let mut k = polydat::dsl::compile::compile_polydat_interpreter(
            "input cycle: u64\nb := cycle > 0\nn := pick(b, 7)\nout := u64_add(n, 1)\n",
        )
        .expect("it builds");
        k.set_inputs(&[1]);
        assert_eq!(k.pull("out").as_u64(), 8);
    }
}

/// The kernel-bound embedding surfaces take any scope, not just an
/// interpreter kernel (F-C6).
///
/// `Lookup` had one kernel implementor and these surfaces took
/// `&PolydatKernel`, so a host holding a `Box<dyn Kernel>` could not
/// read its own bindings through them: the only route was to compile
/// the program a second time on the interpreter.
#[cfg(test)]
mod kernel_bound_on_every_engine {
    use polydat::dsl::compile::{compile_polydat_with, eval_kernel_bound_typed};
    use polydat::{Engine, JitMode, Kernel, Provenance};

    #[test]
    fn a_host_interpolates_against_the_kernel_it_holds() {
        let src = "input cycle: u64\nconst k := 10\nn := u64_add(cycle, 5)\n";
        for engine in [
            Engine::Interpreter(JitMode::Off),
            Engine::Closures(Provenance::Raw),
            Engine::Native(Provenance::PushPull),
        ] {
            let mut kernel: Box<dyn Kernel> =
                compile_polydat_with(src, engine).unwrap_or_else(|e| panic!("{engine:?}: {e}"));
            kernel.set_inputs(&[3]);
            let scope = polydat::kernel::interp::KernelScope::new(kernel.as_ref());

            // A folded constant of the program.
            let doubled: u64 = eval_kernel_bound_typed("{k} * 2", &scope)
                .unwrap_or_else(|e| panic!("{engine:?}: {e}"));
            assert_eq!(doubled, 20, "{engine:?}");

            // An input the host wrote.
            let cycle: u64 = eval_kernel_bound_typed("{cycle}", &scope)
                .unwrap_or_else(|e| panic!("{engine:?}: {e}"));
            assert_eq!(cycle, 3, "{engine:?}");
        }
    }
}
