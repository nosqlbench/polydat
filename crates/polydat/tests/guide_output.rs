// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Every fenced `text` block in a guide is verbatim output of the
//! guide's companion example. The guides promise that every quoted
//! line is real; this test makes the promise fail loudly when the
//! example or the compiler drifts from the document.
//!
//! Output that a guide quotes from somewhere other than its example
//! (the binary, another tool) is fenced as `console` and not checked.

use std::path::Path;
use std::process::Command;

/// Run a companion example and return its stdout with normalized
/// line endings.
fn example_output(example: &str) -> String {
    let out = Command::new(env!("CARGO"))
        .args(["run", "--quiet", "--all-features", "--example", example])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap_or_else(|e| panic!("run example {example}: {e}"));
    assert!(out.status.success(), "example {example} failed:\n{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n")
}

/// The `text` fenced blocks of a document, each with the line number
/// of its opening fence.
fn text_blocks(doc: &str) -> Vec<(usize, String)> {
    let mut blocks = Vec::new();
    let mut open: Option<(usize, Vec<&str>)> = None;
    for (i, line) in doc.lines().enumerate() {
        match &mut open {
            None if line.trim_end() == "```text" => open = Some((i + 1, Vec::new())),
            None => {}
            Some((start, lines)) if line.starts_with("```") => {
                blocks.push((*start, lines.join("\n")));
                open = None;
            }
            Some((_, lines)) => lines.push(line.trim_end()),
        }
    }
    assert!(open.is_none(), "unterminated fence");
    blocks
}

fn check(doc: &str, example: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(doc);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let text = text.replace("\r\n", "\n");
    let blocks = text_blocks(&text);
    assert!(!blocks.is_empty(), "{doc} has no text blocks");
    let output: String = example_output(example).lines().map(str::trim_end).collect::<Vec<_>>().join("\n");
    let missing: Vec<String> = blocks
        .iter()
        .filter(|(_, block)| !output.contains(block.as_str()))
        .map(|(line, block)| format!("{doc}:{line}: not in the output of examples/{example}.rs:\n{block}\n"))
        .collect();
    assert!(missing.is_empty(), "{} of {} blocks are not verbatim example output:\n\n{}", missing.len(), blocks.len(), missing.join("\n"));
}

#[test]
fn embedding_guide_quotes_its_example() {
    check("docs/guides/embedding.md", "embedding_guide");
}

#[test]
fn polytile_tutorial_quotes_its_example() {
    check("docs/tutorials/polytile_tutorial.md", "polytile_tutorial");
}
