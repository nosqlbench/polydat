// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Illustration: one two-dimensional comprehension dispensed under several
//! traversal strategies and a filter, as coordinate tuples.

use polydat::iteration::comprehension::ast::Comprehension;
use polydat::iteration::comprehension::ir::{compile, interpret};
use polydat::iteration::comprehension::optimize::optimize;
use polydat::iteration::comprehension::source::Source;
use polydat::iteration::comprehension::strategy::StrategyName;

fn show(label: &str, ast: &Comprehension) {
    let prog = compile(&optimize(ast.clone()));
    let mut stream = interpret(&prog);
    let mut out = Vec::new();
    while let Some(t) = stream.advance() {
        let cells: Vec<String> = t.bindings.iter().map(|(_, v)| format!("{v:?}")).map(|s| s.trim_start_matches("I64(").trim_end_matches(')').to_string()).collect();
        out.push(format!("({})", cells.join(",")));
    }
    println!("{label:<14} {:>2}  {}", out.len(), out.join(" "));
}

fn main() {
    let base = Comprehension::cartesian(vec![
        Comprehension::clause("k", Source::IntRange { lo: 1, hi: 4, step: 1 }),
        Comprehension::clause("limit", Source::IntRange { lo: 10, hi: 40, step: 10 }),
    ]);
    show("lex", &base);
    show("reverse", &Comprehension::order(base.clone(), StrategyName::ReverseLex, None));
    show("diagonal", &Comprehension::order(base.clone(), StrategyName::Diagonal, None));
    show("shells", &Comprehension::order(base.clone(), StrategyName::Shells, None));
    show("extrema/1", &Comprehension::order(base.clone(), StrategyName::Extrema, Some(1)));
    show("halton/5", &Comprehension::order(base.clone(), StrategyName::Halton, Some(5)));
    show("where", &Comprehension::filter(base.clone(), "{k} >= 2 && {limit} != 20"));
}
