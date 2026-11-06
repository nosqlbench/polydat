// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The vector names `vec_add`, `vec_scale`, `vec_norm`, `vec_dot`,
//! `vec_l2`, `vec_cosine` and `vec_distance` are overloads of an f32
//! and an f64 form (`polydat::dsl::registry::Overload`). A call builds
//! the form its operands' types agree with most, so f64 vectors are
//! never narrowed to f32 on the way in, and operands agreeing with
//! neither form reach the f32 form through the assembler's adapters, as
//! the names did before they had f64 forms.

use polydat::ast::{PortType, SliceArc, Value};
use polydat::compile::assembly::WireRef;
use polydat::dsl::factory::{BuildContext, build_node};
use polydat::dsl::registry;

/// One engine of each kind.
fn engines() -> Vec<polydat::Engine> {
    use polydat::{Engine, JitMode, Provenance};
    let mut all = vec![
        Engine::Interpreter(JitMode::Off),
        Engine::Closures(Provenance::Auto),
        Engine::Native(Provenance::Auto),
    ];
    if cfg!(feature = "jit") {
        all.push(Engine::PureNative(Provenance::PushPull));
    }
    all
}

/// The registered function a call to `name` over wires of `types`
/// builds.
fn built(name: &str, types: &[PortType]) -> String {
    let wires: Vec<WireRef> = (0..types.len())
        .map(|i| WireRef::input(format!("w{i}")))
        .collect();
    build_node(&BuildContext::default(), name, &wires, types, &[])
        .unwrap_or_else(|e| panic!("{name}{types:?}: {e}"))
        .meta()
        .name
        .clone()
}

#[test]
fn each_vector_name_builds_the_form_its_operands_match() {
    use PortType::{F64, Str, U64, VecF32, VecF64, VecI32};
    for name in ["vec_add", "vec_dot", "vec_l2", "vec_cosine"] {
        assert_eq!(built(name, &[VecF32, VecF32]), format!("{name}_f32"));
        assert_eq!(built(name, &[VecF64, VecF64]), format!("{name}_f64"));
        // No form takes i32 lists: the first form, reached by adapters.
        assert_eq!(built(name, &[VecI32, VecI32]), format!("{name}_f32"));
    }
    assert_eq!(
        built("vec_distance", &[Str, VecF32, VecF32]),
        "vec_distance_f32"
    );
    assert_eq!(
        built("vec_distance", &[Str, VecF64, VecF64]),
        "vec_distance_f64"
    );
    assert_eq!(built("vec_norm", &[VecF32]), "vec_norm_f32");
    assert_eq!(built("vec_norm", &[VecF64]), "vec_norm_f64");
    assert_eq!(built("vec_scale", &[VecF32, F64]), "vec_scale_f32");
    assert_eq!(built("vec_scale", &[VecF64, F64]), "vec_scale_f64");
    // An integer factor agrees with neither form; the vector decides,
    // and the factor widens, rather than the vector narrowing.
    assert_eq!(built("vec_scale", &[VecF64, U64]), "vec_scale_f64");
    assert_eq!(built("vec_scale", &[VecF32, U64]), "vec_scale_f32");
}

/// Every form of every overload declares the wire types its node
/// takes, and every form takes the name's parameters, which is what
/// lets the compiler read them from the first form before the
/// operands' types are known.
#[test]
fn every_overload_form_is_the_node_it_declares() {
    let mut seen = 0;
    for overload in registry::overloads() {
        let first = registry::lookup_registered(overload.forms[0].func).unwrap_or_else(|| {
            panic!(
                "{}: {} is not registered",
                overload.name, overload.forms[0].func
            )
        });
        for form in overload.forms {
            let sig = registry::lookup_registered(form.func)
                .unwrap_or_else(|| panic!("{}: {} is not registered", overload.name, form.func));
            let params = |s: &registry::FuncSig| {
                s.params
                    .iter()
                    .map(|p| (p.name, p.slot_type))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                params(sig),
                params(first),
                "{}: {} parameters",
                overload.name,
                form.func
            );
            let ports: Vec<PortType> = {
                let wires: Vec<WireRef> = (0..form.wires.len())
                    .map(|i| WireRef::input(format!("w{i}")))
                    .collect();
                build_node(&BuildContext::default(), form.func, &wires, form.wires, &[])
                    .unwrap_or_else(|e| panic!("{}: {e}", form.func))
                    .meta()
                    .wire_inputs()
                    .iter()
                    .map(|p| p.typ)
                    .collect()
            };
            assert_eq!(
                ports, form.wires,
                "{}: {} wire types",
                overload.name, form.func
            );
            assert!(
                registry::lookup_registered(overload.name).is_none(),
                "{} is both an overload and a registered function",
                overload.name
            );
            seen += 1;
        }
    }
    assert!(seen >= 14, "the vector overloads are not registered");
}

/// Over f64 vectors whose elements f32 cannot hold, each name computes
/// in f64 on every engine, from extern operands and from parsed ones.
#[test]
fn the_distance_names_keep_f64_operands_in_f64_on_every_engine() {
    let (a, b) = ([0.1f64, 0.2, 0.7], [0.3f64, -0.4, 0.5]);
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let sq: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
    let norms =
        a.iter().map(|x| x * x).sum::<f64>().sqrt() * b.iter().map(|y| y * y).sum::<f64>().sqrt();
    let src = concat!(
        "input cycle: u64\n",
        "extern a: vec_f64\n",
        "extern b: vec_f64\n",
        "dot := vec_dot(a, b)\n",
        "l2 := vec_l2(a, b)\n",
        "cos := vec_cosine(a, b)\n",
        "sq := vec_distance(\"EUCLIDEAN\", a, b)\n",
        "neg := vec_distance(\"DOT_PRODUCT\", a, b)\n",
        "parsed := vec_dot(str_to_vec_f64(\"[0.1, 0.2, 0.7]\"), str_to_vec_f64(\"[0.3, -0.4, 0.5]\"))\n",
        "twice := vec_dot(a, b) * 2.0\n",
    );
    for engine in engines() {
        let asm = polydat::dsl::compile::compile_polydat_to_assembler(src)
            .unwrap_or_else(|e| panic!("{e}\n{src}"));
        let mut k = asm
            .compile_with(engine)
            .unwrap_or_else(|e| panic!("{engine}: {e}\n{src}"));
        k.set_inputs(&[0]);
        k.set_input("a", Value::VecF64(SliceArc::from_vec(a.to_vec())))
            .unwrap_or_else(|e| panic!("{engine}: set a: {e}"));
        k.set_input("b", Value::VecF64(SliceArc::from_vec(b.to_vec())))
            .unwrap_or_else(|e| panic!("{engine}: set b: {e}"));
        assert_eq!(k.pull("dot").as_f64(), dot, "{engine}");
        assert_eq!(k.pull("l2").as_f64(), sq.sqrt(), "{engine}");
        assert_eq!(k.pull("cos").as_f64(), dot / norms, "{engine}");
        assert_eq!(k.pull("sq").as_f64(), sq, "{engine}");
        assert_eq!(k.pull("neg").as_f64(), -dot, "{engine}");
        assert_eq!(k.pull("parsed").as_f64(), dot, "{engine}");
        assert_eq!(k.pull("twice").as_f64(), 2.0 * dot, "{engine}");
    }
}

/// Over f64 vectors each producer computes and returns f64 on every
/// engine, an integer factor to `vec_scale` widening to meet the vector.
#[test]
fn the_vector_producers_keep_f64_operands_in_f64_on_every_engine() {
    let (a, b) = ([0.1f64, 0.2, 0.7], [0.3f64, -0.4, 0.5]);
    let mag = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let src = concat!(
        "input cycle: u64\n",
        "extern a: vec_f64\n",
        "extern b: vec_f64\n",
        "sum := vec_add(a, b)\n",
        "scaled := vec_scale(a, 3)\n",
        "unit := vec_norm(a)\n",
        "chained := vec_dot(vec_add(a, b), vec_norm(b))\n",
    );
    for engine in engines() {
        let asm = polydat::dsl::compile::compile_polydat_to_assembler(src)
            .unwrap_or_else(|e| panic!("{e}\n{src}"));
        let mut k = asm
            .compile_with(engine)
            .unwrap_or_else(|e| panic!("{engine}: {e}\n{src}"));
        k.set_inputs(&[0]);
        k.set_input("a", Value::VecF64(SliceArc::from_vec(a.to_vec())))
            .unwrap_or_else(|e| panic!("{engine}: set a: {e}"));
        k.set_input("b", Value::VecF64(SliceArc::from_vec(b.to_vec())))
            .unwrap_or_else(|e| panic!("{engine}: set b: {e}"));
        let want_sum: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert_eq!(k.pull("sum").as_vec_f64(), &want_sum[..], "{engine}");
        let want_scaled: Vec<f64> = a.iter().map(|x| x * 3.0).collect();
        assert_eq!(k.pull("scaled").as_vec_f64(), &want_scaled[..], "{engine}");
        let want_unit: Vec<f64> = a.iter().map(|x| x / mag).collect();
        assert_eq!(k.pull("unit").as_vec_f64(), &want_unit[..], "{engine}");
        let bmag = b.iter().map(|y| y * y).sum::<f64>().sqrt();
        let want_chained: f64 = want_sum.iter().zip(&b).map(|(s, y)| s * (y / bmag)).sum();
        assert_eq!(k.pull("chained").as_f64(), want_chained, "{engine}");
    }
}

/// Over f32 vectors each name is its f32 form, and over integer lists
/// it is still the f32 form, reached through the widening adapter.
#[test]
fn the_vector_names_read_f32_and_widened_operands_as_before() {
    let src = concat!(
        "input cycle: u64\n",
        "a := hash_vec(cycle, 8)\n",
        "b := hash_vec(cycle + 1, 8)\n",
        "same_dot := f64_eq(vec_dot(a, b), vec_dot_f32(a, b))\n",
        "same_l2 := f64_eq(vec_l2(a, b), vec_l2_f32(a, b))\n",
        "same_cos := f64_eq(vec_cosine(a, b), vec_cosine_f32(a, b))\n",
        "same_dist := f64_eq(vec_distance(\"COSINE\", a, b), vec_distance_f32(\"COSINE\", a, b))\n",
        "same_add := vec_eq_f32(vec_add(a, b), vec_add_f32(a, b))\n",
        "same_scale := vec_eq_f32(vec_scale(a, 2), vec_scale_f32(a, 2.0))\n",
        "same_norm := vec_eq_f32(vec_norm(a), vec_norm_f32(a))\n",
        "ints := vec_dot(str_to_vec_i32(\"[1, 2]\"), str_to_vec_i32(\"[3, 4]\"))\n",
    );
    for engine in engines() {
        let asm = polydat::dsl::compile::compile_polydat_to_assembler(src)
            .unwrap_or_else(|e| panic!("{e}\n{src}"));
        let mut k = asm
            .compile_with(engine)
            .unwrap_or_else(|e| panic!("{engine}: {e}\n{src}"));
        for cycle in [0u64, 7, 1234] {
            k.set_inputs(&[cycle]);
            for name in ["same_dot", "same_l2", "same_cos", "same_dist"] {
                assert_eq!(k.pull(name).as_u64(), 1, "{engine} {name} at cycle {cycle}");
            }
            for name in ["same_add", "same_scale", "same_norm"] {
                assert!(k.pull(name).as_bool(), "{engine} {name} at cycle {cycle}");
            }
            assert_eq!(k.pull("ints").as_f64(), 11.0, "{engine}");
        }
    }
}

/// A host asks `lookup` whether a program can call a name, so it
/// answers for every overloaded name, with the first form's signature,
/// while `lookup_registered` answers only for the forms; and the
/// overloaded names are listed beside the registered ones, under their
/// first form's category.
#[test]
fn a_host_finds_and_lists_the_overloaded_names_as_programs_call_them() {
    for name in [
        "vec_add",
        "vec_scale",
        "vec_norm",
        "vec_dot",
        "vec_l2",
        "vec_cosine",
        "vec_distance",
    ] {
        let sig = registry::lookup(name).unwrap_or_else(|| panic!("{name} is not callable"));
        assert_eq!(sig.name, format!("{name}_f32"));
        assert!(registry::lookup_registered(name).is_none(), "{name}");
        let listed = polydat::dsl::factories::PolydatRuntime::new()
            .overloads_by_category()
            .into_iter()
            .find(|(_, names)| names.iter().any(|o| o.name == name))
            .unwrap_or_else(|| panic!("{name} is not listed"));
        assert_eq!(listed.0, sig.category, "{name}");
    }
    let listed: usize = registry::overloads_by_category()
        .iter()
        .map(|(_, names)| names.len())
        .sum();
    assert_eq!(listed, registry::overloads().count());
}
