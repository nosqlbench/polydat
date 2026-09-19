// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Every function computes the same bits on every engine at the
//! corners of its inputs.
//!
//! The engine parity test pins that the engines agree on each node at
//! a few ordinary cycles, by display string. This suite drives each
//! node's program through the corner values of its varying input, the
//! integer edges of a `u64` coordinate and the special values of an
//! `f64` extern, and compares every output bit for bit: an `f64` by
//! its bits (so `-0.0` differs from `0.0` and a NaN's payload counts),
//! a vector element by element, a register word by its two words, a
//! string or byte string by its bytes. A failure the interpreter
//! raises must be raised alike, location aside, by every engine that
//! runs the program.
//!
//! The suite is optional: it is ignored by default and runs with
//!
//! ```sh
//! cargo test -p polydat --test suite equivalence_corners:: -- --ignored
//! cargo nextest run -p polydat --test suite equivalence_corners:: --run-ignored ignored-only
//! ```

#![cfg(feature = "jit")]

use super::common;

use polydat::ast::{Purity, Slot, Value};
use polydat::dsl::compile::compile_polydat_to_assembler;

/// The corners of a `u64` coordinate: zero, the small integers a
/// node treats specially, each power of two and its neighbours, the
/// widths of every narrower integer, the largest exactly
/// representable `f64` integer, the signed boundary, and the top.
const U64_CORNERS: &[u64] = &[
    0,
    1,
    2,
    3,
    4,
    7,
    8,
    9,
    15,
    16,
    31,
    32,
    63,
    64,
    100,
    127,
    128,
    255,
    256,
    1000,
    1023,
    1024,
    65535,
    65536,
    (1 << 31) - 1,
    1 << 31,
    (1 << 32) - 1,
    1 << 32,
    (1 << 32) + 1,
    (1 << 53) - 1,
    1 << 53,
    (1 << 53) + 1,
    (1 << 63) - 1,
    1 << 63,
    (1 << 63) + 1,
    u64::MAX - 1,
    u64::MAX,
];

/// The corners of an `f64` extern: both zeros, both infinities, a
/// quiet NaN, the smallest and largest normals, a subnormal, one and
/// its neighbours, the halves that rounding rules turn on, the
/// integer-precision edge, and the integer bounds as floats.
fn f64_corners() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.5,
        2.5,
        -2.5,
        0.1,
        1.0 / 3.0,
        1.0 - f64::EPSILON,
        1.0 + f64::EPSILON,
        f64::EPSILON,
        f64::MIN_POSITIVE,
        5e-324,
        f64::MAX,
        f64::MIN,
        1e308,
        1e-308,
        9007199254740992.0,
        9007199254740993.0,
        -9007199254740992.0,
        4294967296.0,
        9223372036854775808.0,
        -9223372036854775808.0,
        18446744073709551616.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
    ]
}

/// A value's bits as text, so that two values compare equal exactly
/// when every bit of them does.
fn canonical(v: &Value) -> String {
    match v {
        Value::U64(x) => format!("u64:{x}"),
        Value::I64(x) => format!("i64:{x}"),
        Value::F64(x) => format!("f64:{:016x}", x.to_bits()),
        Value::Bool(b) => format!("bool:{b}"),
        Value::Str(s) => format!("str:{s:?}"),
        Value::Bytes(b) => format!("bytes:{b:02x?}"),
        Value::Json(j) => format!("json:{j}"),
        Value::Reg128(b, lanes) => format!("reg:{lanes:?}:{:016x}:{:016x}", b.0[0], b.0[1]),
        Value::U128(b) => format!("u128:{:016x}:{:016x}", b.0[0], b.0[1]),
        Value::I128(b) => format!("i128:{:016x}:{:016x}", b.0[0], b.0[1]),
        Value::VecF32(v) => format!(
            "vec_f32:{:?}",
            v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>()
        ),
        Value::VecF64(v) => format!(
            "vec_f64:{:?}",
            v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>()
        ),
        Value::VecF16(v) => format!(
            "vec_f16:{:?}",
            v.iter().map(|x| x.to_bits()).collect::<Vec<u16>>()
        ),
        Value::VecI8(v) => format!("vec_i8:{:?}", v.as_slice()),
        Value::VecI16(v) => format!("vec_i16:{:?}", v.as_slice()),
        Value::VecI32(v) => format!("vec_i32:{:?}", v.as_slice()),
        Value::VecI64(v) => format!("vec_i64:{:?}", v.as_slice()),
        Value::Ext(_) | Value::Handle(_) => format!("{}:{}", v.port_type(), v.to_display_string()),
        Value::None => "none".into(),
    }
}

fn payload_text(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// A failure message without its `panicked at` line, which names the
/// code that raised the panic and so differs by engine.
fn without_location(msg: &str) -> String {
    msg.lines()
        .filter(|l| !l.trim_start().starts_with("↳ panicked at"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// One evaluation's outputs, or the failure it raised.
type Outcome = Result<Vec<String>, String>;

/// The engines that accept a program, and its output names.
type Engines = (Vec<Box<dyn Engine>>, Vec<String>);

/// One engine over one program: writes a coordinate or an extern,
/// evaluates, and reads every output.
trait Engine {
    fn name(&self) -> &'static str;
    fn run(&mut self, coords: &[u64], extern_value: Option<&Value>, outs: &[String]) -> Outcome;
}

struct Interpreter(polydat::kernel::PolydatKernel);
struct Closures(polydat::compile::closures::CompiledKernelRaw);
struct Hybrid(polydat::compile::hybrid::HybridKernelPushPull);
struct Pure(polydat::compile::jit::JitKernelPushPull);

fn caught<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(payload_text)
}

impl Engine for Interpreter {
    fn name(&self) -> &'static str {
        "P1"
    }
    fn run(&mut self, coords: &[u64], extern_value: Option<&Value>, outs: &[String]) -> Outcome {
        let k = &mut self.0;
        caught(|| {
            k.set_inputs(coords);
            if let Some(v) = extern_value {
                k.set_input("x", v.clone()).unwrap();
            }
            outs.iter().map(|o| canonical(k.pull(o))).collect()
        })
    }
}

impl Engine for Closures {
    fn name(&self) -> &'static str {
        "P2"
    }
    fn run(&mut self, coords: &[u64], extern_value: Option<&Value>, outs: &[String]) -> Outcome {
        let k = &mut self.0;
        caught(|| {
            if let Some(v) = extern_value {
                k.set_input("x", v.clone()).unwrap();
            }
            k.eval(coords);
            outs.iter().map(|o| canonical(&k.get_value(o))).collect()
        })
    }
}

impl Engine for Hybrid {
    fn name(&self) -> &'static str {
        "P3"
    }
    fn run(&mut self, coords: &[u64], extern_value: Option<&Value>, outs: &[String]) -> Outcome {
        let k = &mut self.0;
        caught(|| {
            if let Some(v) = extern_value {
                k.set_input("x", v.clone()).unwrap();
            }
            k.eval(coords);
            outs.iter().map(|o| canonical(&k.get_value(o))).collect()
        })
    }
}

impl Engine for Pure {
    fn name(&self) -> &'static str {
        "pure"
    }
    fn run(&mut self, coords: &[u64], extern_value: Option<&Value>, outs: &[String]) -> Outcome {
        let k = &mut self.0;
        caught(|| {
            if let Some(v) = extern_value {
                k.set_input("x", v.clone()).unwrap();
            }
            k.eval(coords);
            outs.iter().map(|o| canonical(&k.get_value(o))).collect()
        })
    }
}

/// The interpreter and every compiled engine that accepts `src`, with
/// the program's output names; `None` when the interpreter refuses
/// the program (the coverage test owns that) or the program holds a
/// node declared nondeterministic (a clock, entropy, a counter).
fn engines(src: &str) -> Option<Engines> {
    let mut asm = compile_polydat_to_assembler(src).ok()?;
    asm.set_jit_mode(polydat::JitMode::Off);
    let p1 = asm.compile().ok()?;
    let program = p1.program();
    if (0..program.node_count()).any(|i| {
        matches!(
            program.node_ref(i).purity(),
            Purity::Nondeterministic { .. }
        )
    }) {
        return None;
    }
    let outs: Vec<String> = p1.output_names().iter().map(|s| s.to_string()).collect();
    let mut engines: Vec<Box<dyn Engine>> = vec![Box::new(Interpreter(p1))];
    if let Ok(k) = compile_polydat_to_assembler(src).unwrap().try_compile_raw() {
        engines.push(Box::new(Closures(k)));
    }
    if let Ok(k) = compile_polydat_to_assembler(src).unwrap().compile_hybrid() {
        engines.push(Box::new(Hybrid(k)));
    }
    if let Ok(k) = compile_polydat_to_assembler(src)
        .unwrap()
        .try_compile_pure_jit()
    {
        engines.push(Box::new(Pure(k)));
    }
    Some((engines, outs))
}

/// Run every engine at one point and record how any of them differs
/// from the interpreter.
fn compare_at(
    name: &str,
    engines: &mut [Box<dyn Engine>],
    outs: &[String],
    coords: &[u64],
    extern_value: Option<&Value>,
    label: &str,
    disagreements: &mut Vec<String>,
) {
    let (oracle, rest) = engines
        .split_first_mut()
        .expect("the interpreter runs first");
    let want = oracle.run(coords, extern_value, outs);
    for engine in rest {
        let got = engine.run(coords, extern_value, outs);
        match (&want, got) {
            (Ok(want), Ok(got)) => {
                for (i, o) in outs.iter().enumerate() {
                    if want[i] != got[i] {
                        disagreements.push(format!(
                            "  {name} on {}: `{o}` at {label}: interpreter {} / {} {}",
                            engine.name(),
                            want[i],
                            engine.name(),
                            got[i]
                        ));
                    }
                }
            }
            (Err(want), Err(got)) => {
                if without_location(&got) != without_location(want) {
                    disagreements.push(format!(
                        "  {name} on {}: at {label} fails differently:\n    interpreter: {}\n    {}: {}",
                        engine.name(),
                        want.replace('\n', "\n      "),
                        engine.name(),
                        got.replace('\n', "\n      ")
                    ));
                }
            }
            (Ok(_), Err(got)) => disagreements.push(format!(
                "  {name} on {}: at {label} fails where the interpreter computes: {}",
                engine.name(),
                got.lines().next().unwrap_or("")
            )),
            (Err(want), Ok(_)) => disagreements.push(format!(
                "  {name} on {}: at {label} computes where the interpreter fails: {}",
                engine.name(),
                want.lines().next().unwrap_or("")
            )),
        }
    }
}

/// The programs whose coordinate feeds a size: the synthesized call
/// puts the coordinate on every wire, and a corner of `u64` on a
/// count or a width is not a value to compute over but an allocation
/// to refuse. Each gets the coordinate on its value wires and a
/// literal on its size.
fn sized_overrides() -> Vec<(&'static str, &'static str)> {
    vec![
        ("hash_vec", "input cycle: u64\nout := hash_vec(cycle, 8)"),
        (
            "xxhash3_vec",
            "input cycle: u64\nout := xxhash3_vec(cycle, 8)",
        ),
        (
            "random_vector",
            "input cycle: u64\nout := random_vector(cycle, 8)",
        ),
        (
            "char_buf",
            "input cycle: u64\nout := char_buf(cycle, \"a-z\", 12)",
        ),
    ]
}

/// A silent panic hook while the suite runs: every failure it provokes
/// is caught and compared, so the default hook's report is noise.
fn quietly<T>(f: impl FnOnce() -> T) -> T {
    let prior = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = f();
    std::panic::set_hook(prior);
    r
}

/// The float functions whose coverage program is written by hand (it
/// derives its float from the coordinate), each over the extern
/// instead, so their corners are the float corners too.
fn float_overrides() -> Vec<(&'static str, &'static str)> {
    vec![
        // Uneven constants, so a reciprocal in place of a division or
        // a bound's zero in place of the input's would show.
        ("quantize", "extern x: f64 = 0.0\nout := quantize(x, 0.3)"),
        (
            "clamp_f64",
            "extern x: f64 = 0.0\nout := clamp_f64(x, -0.0, 0.5)",
        ),
        ("lerp", "extern x: f64 = 0.0\nout := lerp(x, 0.1, 0.7)"),
        (
            "inv_lerp",
            "extern x: f64 = 0.0\nout := inv_lerp(x, -1.0, 3.0)",
        ),
        (
            "remap",
            "extern x: f64 = 0.0\nout := remap(x, -1.0, 3.0, 10.0, 40.0)",
        ),
        (
            "discretize",
            "extern x: f64 = 0.0\nout := discretize(x, 7.0, 3)",
        ),
        (
            "scale_range",
            "input cycle: u64\nout := scale_range(cycle, 0.1, 0.7)",
        ),
        (
            "vec_scale",
            "extern x: f64 = 0.0\nout := vec_scale(hash_vec(7, 8), x)",
        ),
        (
            "lid_mle",
            "extern x: f64 = 0.0\nout := lid_mle(hash_vec(7, 8), x)",
        ),
        (
            "reg_with_lane_f32",
            "extern x: f64 = 0.0\nout := reg_with_lane_f32(reg_splat_f32(x), 1, x)",
        ),
        (
            "reg_lane_f32",
            "extern x: f64 = 0.0\nout := reg_lane_f32(reg_splat_f32(x), 2)",
        ),
        (
            "reg_dot_f32",
            "extern x: f64 = 0.0\nout := reg_dot_f32(reg_splat_f32(x), reg_splat_f32(x))",
        ),
        (
            "reg_to_vec_f32",
            "extern x: f64 = 0.0\nout := reg_to_vec_f32(reg_splat_f32(x))",
        ),
        (
            "dist_empirical",
            "extern x: f64 = 0.0\nout := dist_empirical(x, \"1.0 3.0 5.0 7.0 9.0\")",
        ),
        ("blend", "extern x: f64 = 0.0\nout := blend(x, 1.0, 0.5)"),
    ]
}

fn report(disagreements: &[String], family: &str) {
    assert!(
        disagreements.is_empty(),
        "{} disagreement(s) with the interpreter over the {family} corners:\n{}",
        disagreements.len(),
        disagreements.join("\n")
    );
}

/// Every node's program at every corner of its `u64` coordinate.
#[test]
#[ignore = "the optional corner-case suite; run with `-- --ignored`"]
fn every_function_computes_identical_bits_at_the_integer_corners() {
    let sized = sized_overrides();
    let mut disagreements = Vec::new();
    let mut programs = 0usize;
    for (name, src) in common::coverage_cases::programs() {
        let src = sized
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, s)| s.to_string())
            .unwrap_or(src);
        let Some((mut engines, outs)) = engines(&src) else {
            continue;
        };
        programs += 1;
        if std::env::var_os("CORNERS_TRACE").is_some() {
            eprintln!("{name}");
        }
        quietly(|| {
            for &c in U64_CORNERS {
                compare_at(
                    &name,
                    &mut engines,
                    &outs,
                    &[c],
                    None,
                    &format!("cycle {c}"),
                    &mut disagreements,
                );
            }
        });
    }
    eprintln!(
        "{programs} programs at {} integer corners each",
        U64_CORNERS.len()
    );
    report(&disagreements, "integer");
}

/// Every node whose first wire is an `f64` port, at every corner of an
/// `f64` extern on that wire.
#[test]
#[ignore = "the optional corner-case suite; run with `-- --ignored`"]
fn every_float_function_computes_identical_bits_at_the_float_corners() {
    let corners = f64_corners();
    let mut disagreements = Vec::new();
    let mut programs = 0usize;
    let mut programs_over =
        common::coverage_cases::programs_over("extern x: f64 = 0.0", "x", false);
    programs_over.extend(
        float_overrides()
            .into_iter()
            .map(|(n, s)| (n.to_string(), s.to_string())),
    );
    for (name, src) in programs_over {
        let Some((mut engines, outs)) = engines(&src) else {
            continue;
        };
        // Only a node that reads the extern as an f64 is a float
        // function; the others read it through an adapter, which the
        // integer corners cover.
        let mut asm = compile_polydat_to_assembler(&src).unwrap();
        asm.set_jit_mode(polydat::JitMode::Off);
        let p1 = asm.compile().unwrap();
        let program = p1.program();
        let first_port_is_f64 = (0..program.node_count()).any(|i| {
            let meta = program.node_meta(i);
            meta.name == name
                && meta.ins.iter().any(|s| match s {
                    Slot::Wire(p) => p.typ == polydat::ast::PortType::F64,
                    Slot::Const { .. } => false,
                })
        });
        if !first_port_is_f64 {
            continue;
        }
        programs += 1;
        if std::env::var_os("CORNERS_TRACE").is_some() {
            eprintln!("{name}");
        }
        quietly(|| {
            for &f in &corners {
                compare_at(
                    &name,
                    &mut engines,
                    &outs,
                    &[],
                    Some(&Value::F64(f)),
                    &format!("x = {f:?} ({:016x})", f.to_bits()),
                    &mut disagreements,
                );
            }
        });
    }
    eprintln!(
        "{programs} programs at {} float corners each",
        corners.len()
    );
    report(&disagreements, "float");
}

/// The canonical form tells apart what a display string does not.
#[test]
fn the_canonical_form_is_bit_exact() {
    assert_ne!(canonical(&Value::F64(0.0)), canonical(&Value::F64(-0.0)));
    assert_ne!(
        canonical(&Value::F64(f64::NAN)),
        canonical(&Value::F64(-f64::NAN))
    );
    assert_eq!(
        canonical(&Value::F64(f64::NAN)),
        canonical(&Value::F64(f64::NAN))
    );
    assert_ne!(
        canonical(&Value::F64(1.0)),
        canonical(&Value::F64(1.0 + f64::EPSILON))
    );
}
