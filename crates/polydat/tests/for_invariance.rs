// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 113 step 5: program invariance under coordinates. A compiled
//! program is a property of the lexical position of each `for` body,
//! not of the tuples that reach it. A three-level traversal compiles
//! exactly four programs (the root and one per body) and activating
//! thousands of tuples builds none. Every build is recorded in the
//! compile ledger of the tree it belongs to, so each test reads the
//! ledger of the kernel it holds and no other test's compiles can
//! reach it, whatever harness runs them.

use std::sync::Arc;

use polydat::dsl::compile_polydat_interpreter;
use polydat::kernel::{PolydatKernel, program_count};

const THREE_LEVELS: &str = "input cycle: u64\n\
for p in partitions(\"*/4\", 100000) {\n\
    slice := cardinality(p)\n\
    for tenant in 0..20 {\n\
        tid := u64_add(u64_mul(tenant, 1000), slice)\n\
        for device in 0..50 {\n\
            leaf := u64_add(tid, device)\n\
        }\n\
    }\n\
}\n";

#[test]
fn compiling_three_levels_builds_exactly_four_programs() {
    let k = compile_polydat_interpreter(THREE_LEVELS).unwrap();
    // Element-type probes and the body compiles themselves build
    // intermediate programs, all recorded in the tree's ledger, but
    // the compiled result holds exactly one per lexical position.
    assert_eq!(program_count(k.program()), 4);
    assert!(k.program().ledger().programs() >= 4);
    let level1 = &k.program().traversals()[0].program;
    let level2 = &level1.traversals()[0].program;
    let level3 = &level2.traversals()[0].program;
    assert!(level3.traversals().is_empty());
    assert!(level3.output_names().contains(&"leaf"));
}

/// Walk every level of the three-level traversal, counting innermost
/// activations and checking they share the leaf program.
fn walk_three_levels(
    k: &mut PolydatKernel,
    leaf_program: &Arc<polydat::kernel::PolydatProgram>,
) -> (u64, u64) {
    let mut leaves = 0u64;
    let mut checksum = 0u64;
    let mut outer = k.traverse(0).unwrap();
    assert_eq!(outer.len(), 4);
    while let Some(mut a1) = outer.advance().unwrap() {
        a1.cycle(0);
        let mut mid = a1.kernel.traverse(0).unwrap();
        assert_eq!(mid.len(), 20);
        while let Some(mut a2) = mid.advance().unwrap() {
            a2.cycle(0);
            let mut inner = a2.kernel.traverse(0).unwrap();
            assert_eq!(inner.len(), 50);
            while let Some(mut a3) = inner.advance().unwrap() {
                assert!(Arc::ptr_eq(a3.kernel.program(), leaf_program));
                checksum = checksum.wrapping_add(a3.cycle(0).pull("leaf").as_u64());
                leaves += 1;
            }
        }
    }
    (leaves, checksum)
}

#[test]
fn activating_thousands_of_tuples_builds_no_programs() {
    let mut k = compile_polydat_interpreter(THREE_LEVELS).unwrap();
    k.set_inputs(&[0]);
    let ledger = k.program().ledger().clone();
    let leaf_program = k.program().traversals()[0].program.traversals()[0]
        .program
        .traversals()[0]
        .program
        .clone();
    assert!(
        Arc::ptr_eq(leaf_program.ledger(), &ledger),
        "one ledger per tree"
    );

    // Opening a traversal whose source is a generator call evaluates
    // that call once through the constant-expression path, which is
    // cached by text. The first walk pays that; activation itself
    // never compiles, so the second walk builds nothing at all.
    let (leaves, checksum) = walk_three_levels(&mut k, &leaf_program);
    assert_eq!(leaves, 4 * 20 * 50);
    assert_ne!(checksum, 0);

    let before = ledger.programs();
    let (leaves, again) = walk_three_levels(&mut k, &leaf_program);
    let after = ledger.programs();
    assert_eq!(leaves, 4 * 20 * 50);
    assert_eq!(again, checksum);
    assert_eq!(
        after,
        before,
        "activation must not compile: {} programs were built",
        after - before
    );
}

#[test]
fn opening_a_generator_sourced_traversal_compiles_its_source_once() {
    let mut k = compile_polydat_interpreter(THREE_LEVELS).unwrap();
    k.set_inputs(&[0]);
    let ledger = k.program().ledger().clone();
    let before = ledger.programs();
    let first = k.traverse(0).unwrap();
    let after_first = ledger.programs();
    let second = k.traverse(0).unwrap();
    let after_second = ledger.programs();
    assert_eq!(first.len(), second.len());
    // The `partitions(...)` source evaluates through the cached
    // constant-expression path: at most a couple of programs on the
    // first open, none on the second.
    assert!(
        after_first - before <= 2,
        "first open built {} programs",
        after_first - before
    );
    assert_eq!(after_second, after_first, "re-opening must not compile");
}

#[test]
fn a_second_host_shares_the_same_programs_and_builds_none() {
    let k = compile_polydat_interpreter(THREE_LEVELS).unwrap();
    let program = k.into_program();
    let ledger = program.ledger().clone();
    let before = ledger.programs();
    let mut a = PolydatKernel::over(program.clone());
    let mut b = PolydatKernel::over(program.clone());
    a.set_inputs(&[0]);
    b.set_inputs(&[0]);
    let sa = a.traverse(0).unwrap();
    let sb = b.traverse(0).unwrap();
    let act_a = sa.activation(3).unwrap();
    let act_b = sb.activation(3).unwrap();
    assert!(Arc::ptr_eq(act_a.kernel.program(), act_b.kernel.program()));
    // Two hosts over one program share its ledger, and neither built.
    assert!(Arc::ptr_eq(a.program().ledger(), b.program().ledger()));
    assert_eq!(ledger.programs(), before);
}

/// Activation cost is affine: a constant per activation that does not
/// depend on how many tuples precede it. Measured, not asserted tightly,
/// because absolute numbers depend on the build profile. Run with
/// `-- --ignored --nocapture` to see them.
#[test]
#[ignore = "cost measurement; run deliberately with --nocapture"]
fn activation_cost_is_flat_across_the_tuple_index() {
    use std::time::Instant;
    let src = "input cycle: u64\nfor k in 0..4000 {\n    v := hash(k)\n}\n";
    let mut k = compile_polydat_interpreter(src).unwrap();
    k.set_inputs(&[0]);
    let stream = k.traverse(0).unwrap();
    assert_eq!(stream.len(), 4000);

    let measure = |range: std::ops::Range<usize>| -> f64 {
        let start = Instant::now();
        let mut sink = 0u64;
        for i in range.clone() {
            let mut act = stream.activation(i).unwrap();
            sink = sink.wrapping_add(act.cycle(0).pull("v").as_u64());
        }
        assert_ne!(sink, 0);
        start.elapsed().as_nanos() as f64 / range.len() as f64
    };
    // Warm once so allocator state is comparable.
    let _ = measure(0..200);
    let head = measure(0..500);
    let tail = measure(3500..4000);
    // Baseline: state allocation alone over the same program.
    let program = stream.traversal().program.clone();
    let start = Instant::now();
    for _ in 0..500 {
        let _ = program.create_state();
    }
    let alloc = start.elapsed().as_nanos() as f64 / 500.0;
    eprintln!(
        "activation ns: first 500 = {head:.0}, last 500 = {tail:.0}, bare state allocation = {alloc:.0}"
    );
    // Flatness: the last block costs no more than twice the first, and
    // the state allocation is the dominant term of each.
    assert!(
        tail <= head * 2.0,
        "activation cost grew with index: head {head:.0} ns, tail {tail:.0} ns"
    );
    assert!(
        alloc <= head,
        "state allocation {alloc:.0} ns should not exceed a full activation {head:.0} ns"
    );
}

#[test]
fn producers_and_derivations_do_not_add_programs_per_tuple() {
    let k = compile_polydat_interpreter(
        "input cycle: u64\nbase := for k in 1..200, limit in 1..50\nedges := for base where {k} == 1 || {k} == 199\nfor edges {\n    f := u64_add(k, limit)\n}\n",
    )
    .unwrap();
    assert_eq!(program_count(k.program()), 2);
    let mut k = k;
    k.set_inputs(&[0]);
    let ledger = k.program().ledger().clone();
    // The filter evaluates in the comprehension grammar directly, so
    // neither opening nor activating compiles anything, on the first
    // pass or any later one.
    let before = ledger.programs();
    for _ in 0..2 {
        let stream = k.traverse(0).unwrap();
        assert_eq!(stream.len(), 98);
        for i in 0..stream.len() {
            let mut act = stream.activation(i).unwrap();
            act.cycle(0).pull("f");
        }
    }
    assert_eq!(
        ledger.programs(),
        before,
        "{} programs were built",
        ledger.programs() - before
    );
}

/// A compile charged to a ledger the host holds: two trees built under
/// one ledger record into it, and a tree built under none has one of
/// its own.
#[test]
fn a_host_ledger_collects_every_tree_compiled_under_it() {
    use polydat::dsl::compile::{CompileOptions, compile_polydat_with_options};
    use polydat::kernel::CompileLedger;
    let ledger = CompileLedger::new();
    let options = CompileOptions {
        ledger: Some(ledger.clone()),
        ..CompileOptions::default()
    };
    let a = compile_polydat_with_options(THREE_LEVELS, &options, None).unwrap();
    let after_a = ledger.programs();
    assert!(after_a >= 4, "{after_a} programs after the first tree");
    let b = compile_polydat_with_options(THREE_LEVELS, &options, None).unwrap();
    let after_b = ledger.programs();
    assert!(
        after_b >= after_a + 4,
        "{after_b} programs after the second tree"
    );
    assert!(Arc::ptr_eq(a.program().ledger(), b.program().ledger()));
    let alone = compile_polydat_interpreter(THREE_LEVELS).unwrap();
    assert!(!Arc::ptr_eq(alone.program().ledger(), &ledger));
    assert_eq!(ledger.programs(), after_b);
    assert!(alone.program().ledger().programs() >= 4);
}
