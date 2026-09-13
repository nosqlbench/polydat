// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Comparison and selection nodes.
//!
//! Two families:
//!
//! - **Comparison** (`u64_eq`, `u64_lt`, `f64_lt`, …): two-input
//!   nodes that produce a u64 truth value (0 or 1). The DSL's
//!   `==`, `!=`, `<`, `>`, `<=`, `>=` infix operators desugar to
//!   these — type-aware dispatch in `compile_binding` picks the
//!   `u64_*` or `f64_*` variant based on operand types.
//!
//! - **Selection** (`select_u64`, `select_f64`): three-input nodes
//!   that pick between two operand values based on a u64 condition
//!   (any nonzero → first arg, zero → second). Used to desugar
//!   `if(cond, a, b)` once the compiler knows the result type.
//!   Both branches always evaluate — no short-circuit. JIT level:
//!   P2 (compiled closure; could become a P3 conditional select
//!   in a future pass).
//!
//! Output of every comparison node is u64 so downstream code can
//! mix them with bitwise operators (`a < b & c < d`) without
//! widening, and pass them as the `cond` input to `select_*`.

// ---------------------------------------------------------------------------
// Comparison nodes
// ---------------------------------------------------------------------------

#[crate::polydat_node(category = Comparison)]
fn u64_eq(a: u64, b: u64) -> u64 {
    if a == b { 1 } else { 0 }
}

#[crate::polydat_node(category = Comparison)]
fn u64_ne(a: u64, b: u64) -> u64 {
    if a != b { 1 } else { 0 }
}

#[crate::polydat_node(category = Comparison)]
fn u64_lt(a: u64, b: u64) -> u64 {
    if a < b { 1 } else { 0 }
}

#[crate::polydat_node(category = Comparison)]
fn u64_gt(a: u64, b: u64) -> u64 {
    if a > b { 1 } else { 0 }
}

#[crate::polydat_node(category = Comparison)]
fn u64_le(a: u64, b: u64) -> u64 {
    if a <= b { 1 } else { 0 }
}

#[crate::polydat_node(category = Comparison)]
fn u64_ge(a: u64, b: u64) -> u64 {
    if a >= b { 1 } else { 0 }
}

// f64 comparisons follow IEEE 754 — NaN compares unequal to
// itself and is neither <, >, <=, nor >=. Tests for NaN should
// use `a != a`.

#[crate::polydat_node(category = Comparison)]
fn f64_eq(a: f64, b: f64) -> u64 {
    if a == b { 1 } else { 0 }
}

#[crate::polydat_node(category = Comparison)]
fn f64_ne(a: f64, b: f64) -> u64 {
    if a != b { 1 } else { 0 }
}

#[crate::polydat_node(category = Comparison)]
fn f64_lt(a: f64, b: f64) -> u64 {
    if a < b { 1 } else { 0 }
}

#[crate::polydat_node(category = Comparison)]
fn f64_gt(a: f64, b: f64) -> u64 {
    if a > b { 1 } else { 0 }
}

#[crate::polydat_node(category = Comparison)]
fn f64_le(a: f64, b: f64) -> u64 {
    if a <= b { 1 } else { 0 }
}

#[crate::polydat_node(category = Comparison)]
fn f64_ge(a: f64, b: f64) -> u64 {
    if a >= b { 1 } else { 0 }
}

// ---------------------------------------------------------------------------
// Selection nodes (the desugar target for `if(cond, a, b)`)
// ---------------------------------------------------------------------------

/// Pick between two u64 inputs based on a u64 condition.
#[crate::polydat_node(category = Comparison)]
fn select_u64(cond: u64, a: u64, b: u64) -> u64 {
    if cond != 0 { a } else { b }
}

// ---------------------------------------------------------------------------
// String comparisons
// ---------------------------------------------------------------------------
//
// Strings live on the heap; the compiled-u64 fast path can't carry
// them in raw u64 buffers, so these are eval-only. The DSL desugar
// in `binding.rs` picks `str_eq` / `str_ne` over the u64 / f64
// variants when either operand has `PortType::Str`.

/// Equality of two String wires. Returns 1 if equal else 0.
///
/// Signature: `str_eq(a: String, b: String) -> (u64)`
#[crate::polydat_node(category = Comparison)]
fn str_eq(a: &str, b: &str) -> u64 {
    if a == b { 1 } else { 0 }
}

/// Inequality of two String wires.
///
/// Signature: `str_ne(a: String, b: String) -> (u64)`
#[crate::polydat_node(category = Comparison)]
fn str_ne(a: &str, b: &str) -> u64 {
    if a != b { 1 } else { 0 }
}

/// Pick between two f64 inputs based on a u64 condition.
/// f64s travel as raw u64 bit patterns through the compiled buffer.
#[crate::polydat_node(category = Comparison)]
fn select_f64(cond: u64, a: f64, b: f64) -> f64 {
    if cond != 0 { a } else { b }
}

/// Pick between two String inputs based on a u64 condition.
/// Any nonzero `cond` → `a`; zero → `b`.
#[crate::polydat_node(category = Comparison)]
fn select_str(cond: u64, a: String, b: String) -> String {
    if cond != 0 { a } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{PolydatNode, Value};

    fn run(node: &dyn PolydatNode, ins: Vec<Value>) -> Value {
        let mut outs = vec![Value::U64(0)];
        node.eval(&ins, &mut outs);
        outs.into_iter().next().unwrap()
    }

    #[test]
    fn u64_lt_gt_eq_basics() {
        assert_eq!(
            run(&U64Lt::new(), vec![Value::U64(1), Value::U64(2)]).as_u64(),
            1
        );
        assert_eq!(
            run(&U64Lt::new(), vec![Value::U64(2), Value::U64(2)]).as_u64(),
            0
        );
        assert_eq!(
            run(&U64Gt::new(), vec![Value::U64(3), Value::U64(2)]).as_u64(),
            1
        );
        assert_eq!(
            run(&U64Eq::new(), vec![Value::U64(5), Value::U64(5)]).as_u64(),
            1
        );
        assert_eq!(
            run(&U64Ne::new(), vec![Value::U64(5), Value::U64(5)]).as_u64(),
            0
        );
        assert_eq!(
            run(&U64Le::new(), vec![Value::U64(2), Value::U64(2)]).as_u64(),
            1
        );
        assert_eq!(
            run(&U64Ge::new(), vec![Value::U64(2), Value::U64(2)]).as_u64(),
            1
        );
    }

    #[test]
    fn f64_comparisons_basics() {
        assert_eq!(
            run(&F64Lt::new(), vec![Value::F64(0.1), Value::F64(0.2)]).as_u64(),
            1
        );
        assert_eq!(
            run(&F64Gt::new(), vec![Value::F64(0.2), Value::F64(0.1)]).as_u64(),
            1
        );
        assert_eq!(
            run(&F64Eq::new(), vec![Value::F64(0.1), Value::F64(0.1)]).as_u64(),
            1
        );
        // NaN: f64_eq of NaN with itself is 0 (IEEE 754).
        assert_eq!(
            run(
                &F64Eq::new(),
                vec![Value::F64(f64::NAN), Value::F64(f64::NAN)]
            )
            .as_u64(),
            0
        );
    }

    #[test]
    fn select_u64_picks_by_cond() {
        let mut outs = vec![Value::U64(0)];
        SelectU64::new().eval(&[Value::U64(1), Value::U64(10), Value::U64(20)], &mut outs);
        assert_eq!(outs[0].as_u64(), 10);
        SelectU64::new().eval(&[Value::U64(0), Value::U64(10), Value::U64(20)], &mut outs);
        assert_eq!(outs[0].as_u64(), 20);
    }

    #[test]
    fn select_f64_picks_by_cond() {
        let mut outs = vec![Value::F64(0.0)];
        SelectF64::new().eval(
            &[Value::U64(1), Value::F64(0.5), Value::F64(1.05)],
            &mut outs,
        );
        assert_eq!(outs[0].as_f64(), 0.5);
        SelectF64::new().eval(
            &[Value::U64(0), Value::F64(0.5), Value::F64(1.05)],
            &mut outs,
        );
        assert_eq!(outs[0].as_f64(), 1.05);
    }

    #[test]
    fn str_eq_ne_basics() {
        let mut out = vec![Value::U64(0)];
        StrEq::new().eval(
            &[Value::Str("LATENCY".into()), Value::Str("LATENCY".into())],
            &mut out,
        );
        assert_eq!(out[0].as_u64(), 1);
        StrEq::new().eval(
            &[Value::Str("LATENCY".into()), Value::Str("RECALL".into())],
            &mut out,
        );
        assert_eq!(out[0].as_u64(), 0);
        StrNe::new().eval(
            &[Value::Str("LATENCY".into()), Value::Str("RECALL".into())],
            &mut out,
        );
        assert_eq!(out[0].as_u64(), 1);
        StrNe::new().eval(&[Value::Str("a".into()), Value::Str("a".into())], &mut out);
        assert_eq!(out[0].as_u64(), 0);
    }

    #[test]
    fn select_str_picks_by_cond() {
        let mut out = vec![Value::Str(String::new().into())];
        SelectStr::default().eval(
            &[
                Value::U64(1),
                Value::Str("yes".into()),
                Value::Str("no".into()),
            ],
            &mut out,
        );
        assert_eq!(out[0].as_str(), "yes");
        SelectStr::default().eval(
            &[
                Value::U64(0),
                Value::Str("yes".into()),
                Value::Str("no".into()),
            ],
            &mut out,
        );
        assert_eq!(out[0].as_str(), "no");
    }
}
