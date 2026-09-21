// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `transform::declared_wires` answers the same question the compiler
//! does, from the source instead of from a compiled program (F-H7).
//!
//! `run --emit` needs the names of a scope's wires before the program
//! is compiled, because the emit binding it appends names them. It
//! used to get them by compiling the program once as a probe, reading
//! a kernel, and compiling the transformed program again — so the run
//! paid for two compiles and reported their sum as its compile time.
//!
//! A second reading of "what does this scope bind" is only safe if it
//! agrees with the first. This checks that it does, over the example
//! programs and the shapes the binary's own tests use.

use polydat::dsl::compile::compile_polydat_interpreter;
use polydat::dsl::parse_polydat;
use polydat::dsl::transform::declared_wires;

/// The compiler's answer for the root scope: the outputs a program
/// declares, less the inputs it declares and the compiler's own.
fn compiled_root_names(src: &str) -> Vec<String> {
    let kernel = compile_polydat_interpreter(src).expect("it compiles");
    let program = kernel.program();
    let inputs = program.input_names();
    program
        .own_output_names()
        .into_iter()
        .filter(|n| !n.starts_with("__") && !inputs.iter().any(|i| i == n))
        .map(str::to_string)
        .collect()
}

fn check(label: &str, src: &str) {
    let declared = declared_wires(&parse_polydat(src).expect("it parses"));
    assert_eq!(
        declared.root.names,
        compiled_root_names(src),
        "{label}: the source's reading and the compiler's disagree"
    );
}

#[test]
fn the_root_scope_reads_the_same_from_source_as_from_the_compiler() {
    check(
        "scalar chain",
        "input cycle: u64\nh := hash(cycle)\nn := mod(h, 100)\n",
    );
    check(
        "const and volatile",
        "input cycle: u64\nconst k := 7\nvolatile v := hash(cycle)\nout := u64_add(v, k)\n",
    );
    check(
        "tuple destructuring",
        "input cycle: u64\n(a, b) := mixed_radix(cycle, 10, 0)\n",
    );
    check(
        "a tile",
        "input cycle: u64\nn := hash(cycle)\ntile row : csv := <<<${n}>>>\n",
    );
    check(
        "a producer binding",
        "input cycle: u64\nsweep := for k in 1..4\nn := hash(cycle)\n",
    );
}

/// A traversal body's wires include the element names its source
/// binds, which are not bindings of the body and appear nowhere in it
/// — they come from the comprehension, or from the producer a bare
/// name or a derivation refers to.
#[test]
fn a_traversal_body_reads_its_element_names() {
    let src = "input cycle: u64\n\
               for phase in load,verify, k in 1..4 {\n\
               \x20   n := hash(k)\n\
               }\n";
    let declared = declared_wires(&parse_polydat(src).expect("it parses"));
    let body = &declared.traversal_bodies[0];
    assert!(body.binds("phase"), "{:?}", body.names);
    assert!(body.binds("k"), "{:?}", body.names);
    assert!(body.binds("n"), "{:?}", body.names);

    let via_producer = "input cycle: u64\n\
                        sweep := for phase in load,verify, k in 1..4\n\
                        for sweep {\n\
                        \x20   n := hash(k)\n\
                        }\n";
    let declared = declared_wires(&parse_polydat(via_producer).expect("it parses"));
    let body = &declared.traversal_bodies[0];
    assert!(body.binds("phase"), "{:?}", body.names);
    assert!(body.binds("k"), "{:?}", body.names);

    let via_derivation = "input cycle: u64\n\
                          sweep := for phase in load,verify, k in 1..4\n\
                          some := for sweep where {k} > 2\n\
                          for some {\n\
                          \x20   n := hash(k)\n\
                          }\n";
    let declared = declared_wires(&parse_polydat(via_derivation).expect("it parses"));
    let body = &declared.traversal_bodies[0];
    assert!(body.binds("phase"), "{:?}", body.names);
    assert!(body.binds("k"), "{:?}", body.names);
}

/// The example programs the repository ships, which are the largest
/// real ones available: every traversal body's wires must cover what
/// the compiler resolves for it.
#[test]
fn the_shipped_examples_agree() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let mut checked = 0;
    for entry in std::fs::read_dir(&root).expect("examples/ exists") {
        let path = entry.expect("a directory entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("polydat") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("readable");
        let Ok(file) = parse_polydat(&src) else {
            continue;
        };
        let declared = declared_wires(&file);
        let Ok(kernel) = compile_polydat_interpreter(&src) else {
            continue;
        };
        let program = kernel.program();
        for (i, t) in program.traversals().iter().enumerate() {
            let body = &declared.traversal_bodies[i];
            for name in t.program.own_output_names() {
                // `__`-carrying names are the compiler's own — a
                // cursor's `rows__ordinal`, an adapter, the emit
                // binding. A program never writes one and an operator
                // never names one.
                if name.contains("__") {
                    continue;
                }
                assert!(
                    body.binds(name),
                    "{}: traversal {i} binds {name}, which the source's reading missed: {:?}",
                    path.display(),
                    body.names
                );
            }
        }
        checked += 1;
    }
    assert!(checked > 0, "no example programs were checked");
}
