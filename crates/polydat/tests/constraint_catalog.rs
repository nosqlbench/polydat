// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The constant-constraint vocabulary is the set the library catalog
//! lists (`library_catalog.md`, "Parameter resolution and validation").
//! The variant set comes from the enum itself: [`every_variant`] walks
//! an exhaustive match, so a new variant does not compile until it has
//! a place in the walk, and the walk then requires its catalog row.

use polydat::dsl::const_constraints::ConstConstraint;
use polydat::dsl::factory::ConstArg;

/// One instance of every `ConstConstraint` variant. `after` maps each
/// variant to the next and the last to `None`; its match is exhaustive,
/// so adding a variant breaks the build here until it joins the walk.
fn every_variant() -> Vec<ConstConstraint> {
    fn accept_all(_: &str) -> Result<(), String> {
        Ok(())
    }
    fn after(c: Option<&ConstConstraint>) -> Option<ConstConstraint> {
        use ConstConstraint as C;
        match c {
            None => Some(C::RangeU64 { min: 1, max: 2 }),
            Some(C::RangeU64 { .. }) => Some(C::RangeF64 { min: 0.0, max: 1.0 }),
            Some(C::RangeF64 { .. }) => Some(C::AllowedU64(&[2, 8])),
            Some(C::AllowedU64(_)) => Some(C::NonZeroU64),
            Some(C::NonZeroU64) => Some(C::NonEmptyStr),
            Some(C::NonEmptyStr) => Some(C::StrParser(accept_all)),
            Some(C::StrParser(_)) => Some(C::PositiveFiniteF64),
            Some(C::PositiveFiniteF64) => Some(C::FiniteF64),
            Some(C::FiniteF64) => None,
        }
    }
    std::iter::successors(after(None), |c| after(Some(c))).collect()
}

/// The variant's name, read from its derived `Debug` rendering.
fn variant_name(c: &ConstConstraint) -> String {
    format!("{c:?}")
        .chars()
        .take_while(|ch| ch.is_ascii_alphanumeric())
        .collect()
}

/// The rows of the constraint table in the catalog section.
fn catalog_rows() -> Vec<String> {
    let doc = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/docs/design/library_catalog.md"
    ))
    .expect("library_catalog.md is readable from the crate dir");
    let section = doc
        .split("### Parameter resolution and validation")
        .nth(1)
        .expect("the catalog has a parameter validation section");
    let section = section.split("\n### ").next().unwrap_or(section);
    section
        .lines()
        .filter(|l| l.starts_with("| `"))
        .map(str::to_string)
        .collect()
}

#[test]
fn every_constraint_variant_has_a_catalog_row() {
    let rows = catalog_rows();
    let variants = every_variant();
    let names: Vec<String> = variants.iter().map(variant_name).collect();
    for name in &names {
        let row_prefix = format!("| `{name}");
        assert!(
            rows.iter().any(|r| r.starts_with(&row_prefix)
                && r[row_prefix.len()..].starts_with(|c: char| !c.is_ascii_alphanumeric())),
            "ConstConstraint::{name} has no row in library_catalog.md, \
             \"Parameter resolution and validation\""
        );
    }
    assert_eq!(
        rows.len(),
        names.len(),
        "the catalog lists a constraint the enum lacks: {rows:#?}"
    );
}

/// The catalog's worked example fails the build with the message it
/// quotes.
#[test]
fn the_catalog_example_fails_with_its_quoted_message() {
    let err = match polydat::dsl::compile::compile_polydat_to_assembler(
        "input cycle: u64\nout := n_of(cycle, 1, 0)\n",
    ) {
        Ok(_) => panic!("n_of with m = 0 builds"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("bad constant n_of: m must be in [1, 65536], got 0"),
        "{err}"
    );
}

/// Each variant's rejection names the parameter first, as the catalog's
/// error texts show.
#[test]
fn every_constraint_rejection_names_the_parameter() {
    fn reject(_: &str) -> Result<(), String> {
        Err("no".into())
    }
    for c in every_variant() {
        let c = match c {
            ConstConstraint::StrParser(_) => ConstConstraint::StrParser(reject),
            other => other,
        };
        let bad = match c {
            ConstConstraint::RangeU64 { .. }
            | ConstConstraint::AllowedU64(_)
            | ConstConstraint::NonZeroU64 => ConstArg::Int(0),
            ConstConstraint::RangeF64 { .. }
            | ConstConstraint::PositiveFiniteF64
            | ConstConstraint::FiniteF64 => ConstArg::Float(f64::NAN),
            ConstConstraint::NonEmptyStr | ConstConstraint::StrParser(_) => {
                ConstArg::Str(" ".into())
            }
        };
        let err = c
            .check(&bad, "p")
            .expect_err(&format!("{} rejects {bad:?}", variant_name(&c)));
        assert!(
            err.starts_with("p must ") || err.starts_with("p: "),
            "{}: {err}",
            variant_name(&c)
        );
    }
}
