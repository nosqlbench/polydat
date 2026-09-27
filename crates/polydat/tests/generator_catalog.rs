// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Every named generator has a row in comprehension_forms.md §3.1.3,
//! "Named generators", and every row names a generator. The generator
//! set is `NamedGenerator::all()`, the walk the dispatch itself looks
//! names up in, so a generator the dispatch accepts is one this test
//! requires a row for.

use polydat::iteration::comprehension::eval::NamedGenerator;

/// The first cells of the named-generator table.
fn table_calls() -> Vec<String> {
    let doc = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/docs/design/comprehension_forms.md"
    ))
    .expect("comprehension_forms.md is readable from the crate dir");
    let section = doc
        .split("#### 3.1.3 ")
        .nth(1)
        .and_then(|s| s.split("\n#### ").next())
        .expect("the spec has a §3.1.3");
    let table = section
        .split("**Named generators.**")
        .nth(1)
        .expect("§3.1.3 has a named-generator table");
    table
        .lines()
        .skip_while(|l| !l.starts_with('|'))
        .take_while(|l| l.starts_with('|'))
        .filter(|l| l.starts_with("| `"))
        .map(|l| {
            l.trim_start_matches("| `")
                .split('`')
                .next()
                .unwrap_or_default()
                .to_string()
        })
        .collect()
}

#[test]
fn every_named_generator_has_a_spec_row() {
    let calls = table_calls();
    let signatures: Vec<&str> = NamedGenerator::all().map(|g| g.signature()).collect();
    for sig in &signatures {
        assert!(
            calls.iter().any(|c| c == sig),
            "the named generator `{sig}` has no row in comprehension_forms.md §3.1.3"
        );
    }
    for call in &calls {
        assert!(
            signatures.contains(&call.as_str()),
            "comprehension_forms.md §3.1.3 has a row for `{call}`, which is not a named generator"
        );
    }
}

/// A call's name finds its generator, and the walk visits each once.
#[test]
fn every_named_generator_is_found_by_its_name() {
    let all: Vec<NamedGenerator> = NamedGenerator::all().collect();
    for g in &all {
        assert_eq!(NamedGenerator::from_name(g.name()), Some(*g));
        assert_eq!(all.iter().filter(|h| *h == g).count(), 1);
    }
}
