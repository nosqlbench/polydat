// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Trigonometric and mathematical function nodes.
//!
//! Standard math operations on f64 values. Use after `unit_interval`
//! or `scale_range` to transform normalized values into waveforms,
//! angles, or other mathematical shapes.

#[polydat::polydat_node(category = Math)]
fn sin(input: f64) -> f64 {
    input.sin()
}

#[polydat::polydat_node(category = Math)]
fn cos(input: f64) -> f64 {
    input.cos()
}

#[polydat::polydat_node(category = Math)]
fn tan(input: f64) -> f64 {
    input.tan()
}

#[polydat::polydat_node(category = Math)]
fn asin(input: f64) -> f64 {
    input.asin()
}

#[polydat::polydat_node(category = Math)]
fn acos(input: f64) -> f64 {
    input.acos()
}

#[polydat::polydat_node(category = Math)]
fn atan(input: f64) -> f64 {
    input.atan()
}

#[polydat::polydat_node(category = Math)]
fn sqrt(input: f64) -> f64 {
    input.sqrt()
}

#[polydat::polydat_node(category = Math)]
fn abs_f64(input: f64) -> f64 {
    input.abs()
}

#[polydat::polydat_node(category = Math)]
fn ln(input: f64) -> f64 {
    input.ln()
}

#[polydat::polydat_node(category = Math)]
fn exp(input: f64) -> f64 {
    input.exp()
}

#[polydat::polydat_node(category = Math, simd = "reg_add_f64", simd_total)]
fn f64_add(a: f64, b: f64) -> f64 {
    a + b
}

#[polydat::polydat_node(category = Math, simd = "reg_sub_f64", simd_total)]
fn f64_sub(a: f64, b: f64) -> f64 {
    a - b
}

#[polydat::polydat_node(category = Math, simd = "reg_mul_f64", simd_total)]
fn f64_mul(a: f64, b: f64) -> f64 {
    a * b
}

#[polydat::polydat_node(category = Math)]
fn f64_div(a: f64, b: f64) -> f64 {
    if b != 0.0 { a / b } else { 0.0 }
}

#[polydat::polydat_node(category = Math)]
fn f64_mod(a: f64, b: f64) -> f64 {
    if b != 0.0 { a % b } else { 0.0 }
}

// --- Binary f64 math functions ---

/// Two-argument arc tangent: atan2(y, x).
///
/// Signature: `atan2(y: f64, x: f64) -> (f64)`
///
/// Returns the angle in radians between the positive x-axis and the
/// point (x, y). Output in (-pi, pi]. Use for converting Cartesian
/// coordinates to polar angle.
///
/// JIT level: P2.
#[polydat::polydat_node(category = Math)]
fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// Power: base^exponent.
#[polydat::polydat_node(category = Math)]
fn pow(base: f64, exponent: f64) -> f64 {
    base.powf(exponent)
}

#[cfg(any())]
#[cfg(test)]
mod tests {
    use super::*;
    use polydat::ast::{PolydatNode, Value};
    use std::f64::consts::PI;

    #[test]
    fn sin_known_values() {
        let node = Sin::new();
        let mut out = [Value::None];
        node.eval(&[Value::F64(0.0)], &mut out);
        assert!((out[0].as_f64() - 0.0).abs() < 1e-10);
        node.eval(&[Value::F64(PI / 2.0)], &mut out);
        assert!((out[0].as_f64() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cos_known_values() {
        let node = Cos::new();
        let mut out = [Value::None];
        node.eval(&[Value::F64(0.0)], &mut out);
        assert!((out[0].as_f64() - 1.0).abs() < 1e-10);
        node.eval(&[Value::F64(PI)], &mut out);
        assert!((out[0].as_f64() + 1.0).abs() < 1e-10);
    }

    #[test]
    fn sqrt_known() {
        let node = Sqrt::new();
        let mut out = [Value::None];
        node.eval(&[Value::F64(4.0)], &mut out);
        assert!((out[0].as_f64() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn atan2_quadrants() {
        let node = Atan2::new();
        let mut out = [Value::None];
        // atan2(1, 0) = pi/2
        node.eval(&[Value::F64(1.0), Value::F64(0.0)], &mut out);
        assert!((out[0].as_f64() - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn pow_known() {
        let node = Pow::new();
        let mut out = [Value::None];
        node.eval(&[Value::F64(2.0), Value::F64(10.0)], &mut out);
        assert!((out[0].as_f64() - 1024.0).abs() < 1e-10);
    }

    #[test]
    fn ln_exp_roundtrip() {
        let node_ln = Ln::new();
        let node_exp = Exp::new();
        let mut out = [Value::None];
        node_exp.eval(&[Value::F64(3.0)], &mut out);
        let e3 = out[0].as_f64();
        node_ln.eval(&[Value::F64(e3)], &mut out);
        assert!((out[0].as_f64() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn compiled_matches_eval() {
        let node = Sin::new();
        let compiled = node.compiled_u64().unwrap();
        let input = PI / 4.0;
        let mut eval_out = [Value::None];
        node.eval(&[Value::F64(input)], &mut eval_out);
        let mut comp_out = [0u64];
        compiled(&[input.to_bits()], &mut comp_out);
        assert_eq!(eval_out[0].as_f64(), f64::from_bits(comp_out[0]));
    }
}
