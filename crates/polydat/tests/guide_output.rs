// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Every fenced `text` block in a tutorial or guide is real output. The
//! documents promise that every quoted line was produced by running
//! something; this test makes the promise fail loudly when an example,
//! the binary, or the compiler drifts from the document.
//!
//! How a block is attributed to its producer:
//!
//! - A block whose first line starts with `$ polydat ` is a binary run.
//!   The command (with `\` continuations joined) is executed from the
//!   `examples/` directory and the remaining lines are its output.
//! - Otherwise the block belongs to the first `examples/<name>.rs` link
//!   in its `##` section, or to the document's default example.
//!
//! Every non-blank quoted line must appear in the producer's output, in
//! order. A line consisting of `...` marks an elision and matches
//! nothing. Output quoted from a source this test cannot run is fenced
//! as `console` and skipped.
//!
//! Guides that quote files and facts instead of output get their own
//! checks below: a quoted grammar or graph must equal its file, counts
//! and names stated in prose must describe that file, feature names must
//! exist in the manifest, and every relative link must resolve.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Run the crate's binary or one of its examples. The binary's progress
/// lines go to stderr and precede its output on a terminal, so a
/// binary run is quoted as stderr followed by stdout.
fn run(args: &[&str], cwd: &Path, with_stderr: bool) -> String {
    let manifest = manifest_dir().join("Cargo.toml");
    let out = Command::new(env!("CARGO"))
        .args(["run", "--quiet", "--all-features", "--manifest-path"])
        .arg(&manifest)
        .args(args)
        .current_dir(cwd)
        .output()
        .unwrap_or_else(|e| panic!("cargo run {args:?}: {e}"));
    assert!(
        out.status.success(),
        "cargo run {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut text = String::new();
    if with_stderr {
        text.push_str(&String::from_utf8_lossy(&out.stderr));
    }
    text.push_str(&String::from_utf8_lossy(&out.stdout));
    text.replace("\r\n", "\n")
}

/// What produced a block.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Producer {
    Example(String),
    Binary(Vec<String>),
}

impl Producer {
    fn output(&self) -> String {
        match self {
            Producer::Example(name) => run(&["--example", name], &manifest_dir(), false),
            Producer::Binary(args) => {
                let mut argv = vec!["--"];
                argv.extend(args.iter().map(String::as_str));
                run(&argv, &manifest_dir().join("examples"), true)
            }
        }
    }
}

struct Block {
    line: usize,
    producer: Option<Producer>,
    lines: Vec<String>,
}

/// The first `examples/<name>.rs` link inside each `##` section, keyed
/// by the section's starting line.
fn section_examples(lines: &[&str]) -> Vec<(usize, Option<String>)> {
    let mut sections: Vec<(usize, Option<String>)> = vec![(0, None)];
    for (i, line) in lines.iter().enumerate() {
        if line.starts_with("## ") {
            sections.push((i, None));
        } else if let Some(rest) = line.split("examples/").nth(1) {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if rest[name.len()..].starts_with(".rs") {
                let current = sections.last_mut().unwrap();
                if current.1.is_none() {
                    current.1 = Some(name);
                }
            }
        }
    }
    sections
}

fn text_blocks(doc: &str, default: Option<&str>) -> Vec<Block> {
    let lines: Vec<&str> = doc.lines().collect();
    let sections = section_examples(&lines);
    // A document with a default example is quoted from that example
    // alone; links to other examples in its prose are cross-references.
    let example_for = |line: usize| -> Option<String> {
        default.map(str::to_string).or_else(|| {
            let section = sections
                .iter()
                .rev()
                .find(|(start, _)| *start <= line)
                .unwrap();
            section.1.clone()
        })
    };
    let mut blocks = Vec::new();
    let mut open: Option<(usize, Vec<String>)> = None;
    for (i, line) in lines.iter().enumerate() {
        match &mut open {
            None if line.trim_end() == "```text" => open = Some((i, Vec::new())),
            None => {}
            Some((start, body)) if line.starts_with("```") => {
                let start = *start;
                let body = std::mem::take(body);
                let block = if body.first().is_some_and(|l| l.starts_with("$ polydat ")) {
                    let mut command = String::new();
                    let mut rest = Vec::new();
                    let mut continuing = true;
                    for l in body {
                        if continuing {
                            let l = l.trim_start_matches("$ polydat ");
                            continuing = l.ends_with('\\');
                            command.push(' ');
                            command.push_str(l.trim_end_matches('\\'));
                        } else {
                            rest.push(l);
                        }
                    }
                    let args = command.split_whitespace().map(str::to_string).collect();
                    Block {
                        line: start + 1,
                        producer: Some(Producer::Binary(args)),
                        lines: rest,
                    }
                } else {
                    Block {
                        line: start + 1,
                        producer: example_for(start).map(Producer::Example),
                        lines: body,
                    }
                };
                blocks.push(block);
                open = None;
            }
            Some((_, body)) => body.push(line.trim_end().to_string()),
        }
    }
    assert!(open.is_none(), "unterminated fence");
    blocks
}

/// The quoted lines that do not appear, in order, in the output.
fn missing_lines(block: &[String], output: &str) -> Vec<String> {
    let out: Vec<&str> = output.lines().map(str::trim_end).collect();
    let mut cursor = 0;
    let mut missing = Vec::new();
    for line in block {
        if line.trim().is_empty() || line.trim() == "..." {
            continue;
        }
        match out[cursor..].iter().position(|o| o == line) {
            Some(p) => cursor += p + 1,
            None => missing.push(line.clone()),
        }
    }
    missing
}

fn check(doc: &str, default: Option<&str>) {
    let path = manifest_dir().join(doc);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .replace("\r\n", "\n");
    let blocks = text_blocks(&text, default);
    assert!(!blocks.is_empty(), "{doc} has no text blocks");
    let mut outputs: HashMap<Producer, String> = HashMap::new();
    let mut failures = Vec::new();
    for block in &blocks {
        let Some(producer) = &block.producer else {
            failures.push(format!(
                "{doc}:{}: no example link in this section and no default",
                block.line
            ));
            continue;
        };
        let output = outputs
            .entry(producer.clone())
            .or_insert_with(|| producer.output());
        let missing = missing_lines(&block.lines, output);
        if !missing.is_empty() {
            failures.push(format!(
                "{doc}:{}: lines not produced by {producer:?}:\n  {}",
                block.line,
                missing.join("\n  ")
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} blocks are not real output:\n\n{}\n",
        failures.len(),
        blocks.len(),
        failures.join("\n\n")
    );
}

#[test]
fn embedding_guide_quotes_its_example() {
    check("docs/guides/embedding.md", Some("embedding_guide"));
}

#[test]
fn polytile_tutorial_quotes_its_example() {
    check(
        "docs/tutorials/polytile_tutorial.md",
        Some("polytile_tutorial"),
    );
}

#[test]
fn illustrations_quote_their_examples() {
    check("docs/tutorials/illustrations.md", None);
}

#[test]
fn toy_tutorial_quotes_the_binary() {
    check("docs/tutorials/toy_test_definition.md", None);
}

/// The toy tutorial reproduces its grammar file in full; the copy must
/// be the file.
#[test]
fn toy_tutorial_quotes_the_grammar_file() {
    let doc = std::fs::read_to_string(manifest_dir().join("docs/tutorials/toy_test_definition.md"))
        .unwrap()
        .replace("\r\n", "\n");
    let file = std::fs::read_to_string(manifest_dir().join("examples/toy_test_definition.polydat"))
        .unwrap()
        .replace("\r\n", "\n");
    let after_heading = doc
        .split("\n## The grammar\n")
        .nth(1)
        .expect("grammar section");
    let block = after_heading
        .split("```polydat\n")
        .nth(1)
        .expect("polydat fence")
        .split("\n```")
        .next()
        .unwrap();
    let normalize = |s: &str| {
        s.lines()
            .map(str::trim_end)
            .collect::<Vec<_>>()
            .join("\n")
            .trim_end()
            .to_string()
    };
    assert_eq!(
        normalize(block),
        normalize(&file),
        "the quoted grammar differs from examples/toy_test_definition.polydat"
    );
}

// ── Guides that quote files and facts rather than program output ──

fn read(rel: &str) -> String {
    let path = manifest_dir().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .replace("\r\n", "\n")
}

/// The fenced block of the given language that follows a heading.
fn block_after<'a>(doc: &'a str, heading: &str, lang: &str) -> &'a str {
    let after = doc
        .split(&format!("\n{heading}\n"))
        .nth(1)
        .unwrap_or_else(|| panic!("heading {heading:?}"));
    after
        .split(&format!("```{lang}\n"))
        .nth(1)
        .unwrap_or_else(|| panic!("{lang} fence after {heading:?}"))
        .split("\n```")
        .next()
        .unwrap()
}

fn normalize(s: &str) -> String {
    s.lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .trim_end()
        .to_string()
}

/// Backticked spans in a line, in order.
fn code_spans(line: &str) -> Vec<&str> {
    line.split('`').skip(1).step_by(2).collect()
}

/// The performance guide reproduces its benchmark graph; the copy must
/// be the file, less the comment header that the prose around the
/// quote already paraphrases.
#[test]
fn performance_guide_quotes_the_graph_file() {
    let doc = read("docs/guides/performance.md");
    let file = read("examples/engine_ladder.polydat");
    let body: Vec<&str> = file
        .lines()
        .skip_while(|l| l.starts_with("//") || l.trim().is_empty())
        .collect();
    assert_eq!(
        normalize(block_after(&doc, "## The graph", "polydat")),
        normalize(&body.join("\n")),
        "the quoted graph differs from examples/engine_ladder.polydat"
    );
}

/// The measurement contract names the graph's inputs, node count, and
/// consumed outputs in prose. Those must describe the graph file.
#[test]
fn performance_guide_describes_the_graph() {
    let doc = read("docs/guides/performance.md");
    let file = read("examples/engine_ladder.polydat");
    let inputs = file.lines().filter(|l| l.starts_with("input ")).count();
    let nodes = file.lines().filter(|l| l.contains(":=")).count();
    let words = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve",
    ];
    let word = |n: usize| {
        words
            .get(n)
            .copied()
            .unwrap_or_else(|| panic!("no number word for {n}; extend the table"))
    };
    let contract = doc
        .split("\n## Measurement contract\n")
        .nth(1)
        .expect("contract section")
        .split("\n## ")
        .next()
        .unwrap();
    assert!(
        contract.contains(&format!("all {} inputs", word(inputs))),
        "the graph has {inputs} inputs; the contract says otherwise"
    );
    assert!(
        contract.contains(&format!("the {} graph nodes", word(nodes))),
        "the graph has {nodes} nodes; the contract says otherwise"
    );
    let consume = contract
        .lines()
        .find(|l| l.contains("Consume `"))
        .expect("a Consume step");
    let outputs = code_spans(consume);
    assert_eq!(outputs.len(), 4, "the contract consumes four outputs");
    for name in outputs {
        assert!(
            file.lines().any(|l| l.starts_with(&format!("{name} :="))),
            "consumed output `{name}` is not a binding in the graph"
        );
    }
    let manifest = read("Cargo.toml");
    let cranelift = manifest
        .lines()
        .find(|l| l.starts_with("cranelift-jit = "))
        .expect("cranelift-jit dependency");
    let version = cranelift
        .split("version = \"")
        .nth(1)
        .expect("version")
        .split('"')
        .next()
        .unwrap();
    assert!(
        doc.contains(&format!("Cranelift {version}")),
        "the reference environment names a Cranelift version other than the manifest's {version}"
    );
}

/// The compilation guide's feature names are real Cargo features.
#[test]
fn compilation_guide_names_real_features() {
    let doc = read("docs/guides/compilation.md");
    let manifest = read("Cargo.toml");
    let features: Vec<&str> = manifest
        .split("\n[features]\n")
        .nth(1)
        .expect("[features] table")
        .split("\n[")
        .next()
        .unwrap()
        .lines()
        .filter_map(|l| l.split_once(" = ").map(|(k, _)| k.trim()))
        .collect();
    let mut named = Vec::new();
    for line in doc.lines() {
        if line.starts_with("| Phase") || line.starts_with("| Hybrid") {
            let cell = line
                .trim_end_matches('|')
                .rsplit('|')
                .next()
                .unwrap()
                .trim();
            if cell != "always" {
                named.extend(code_spans(cell));
            }
        } else if line.starts_with("- **`") {
            named.push(code_spans(line)[0]);
        }
    }
    assert!(!named.is_empty(), "no feature names found in the guide");
    for name in named {
        assert!(
            features.contains(&name),
            "`{name}` is not a feature in Cargo.toml (features: {features:?})"
        );
    }
}

/// Every relative link in the documentation resolves to a file or
/// directory in the repository.
#[test]
fn documentation_links_resolve() {
    fn markdown_files(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                markdown_files(&path, out);
            } else if path.extension().is_some_and(|e| e == "md") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    markdown_files(&manifest_dir().join("docs"), &mut files);
    files.push(manifest_dir().join("README.md"));
    files.push(manifest_dir().join("../../README.md"));
    let mut broken = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file).unwrap();
        let dir = file.parent().unwrap();
        for (i, line) in text.lines().enumerate() {
            for piece in line.split("](").skip(1) {
                let target = piece.split(')').next().unwrap_or("");
                let target = target.split('#').next().unwrap();
                if target.is_empty() || target.contains("://") || target.starts_with("mailto:") {
                    continue;
                }
                if !dir.join(target).exists() {
                    broken.push(format!(
                        "{}:{}: {target}",
                        file.strip_prefix(manifest_dir()).unwrap_or(file).display(),
                        i + 1
                    ));
                }
            }
        }
    }
    assert!(
        broken.is_empty(),
        "{} broken links:\n{}",
        broken.len(),
        broken.join("\n")
    );
}

/// The compilation guide's measured column cites the performance
/// guide; every figure it quotes must appear in that guide's reference
/// table, so the two cannot drift apart.
#[test]
fn compilation_guide_cites_the_reference_run() {
    let doc = read("docs/guides/compilation.md");
    let reference = read("docs/guides/performance.md");
    let mut quoted = 0;
    for line in doc.lines().filter(|l| l.starts_with("| Phase")) {
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        let measured = cells[3];
        for token in measured
            .split([' ', ','])
            .filter(|t| t.chars().next().is_some_and(|c| c.is_ascii_digit()))
        {
            assert!(
                reference.contains(token),
                "`{token}` from the compilation guide is not in the performance guide's reference table"
            );
            quoted += 1;
        }
    }
    assert!(quoted >= 3, "the compilation guide quotes no measurements");
}
