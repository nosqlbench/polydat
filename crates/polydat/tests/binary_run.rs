// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The `polydat` binary's run loop: only coordinate inputs advance per
//! cycle, so an input fixed by a `name=value` argument keeps its value,
//! and emitted rows carry typed values.

use std::process::Command;

fn write_temp(name: &str, contents: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "polydat-binary-run-{}-{}",
        std::process::id(),
        name
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("program.polydat");
    std::fs::write(&path, contents).unwrap();
    path
}

fn run_binary(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_polydat"))
        .args(args)
        .output()
        .expect("polydat binary runs");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn rows(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .filter(|l| !l.starts_with("DBG"))
        .map(str::to_string)
        .collect()
}

#[test]
fn an_assigned_input_keeps_its_value_across_cycles() {
    let path = write_temp("assign", "input cycle: u64\nn := cycle * 2\n");
    let (ok, stdout, stderr) = run_binary(&[
        "run",
        path.to_str().unwrap(),
        "--cycles",
        "3",
        "--emit",
        "map",
        "--outputs",
        "cycle,n",
        "-q",
        "cycle=3",
    ]);
    assert!(ok, "{stderr}");
    assert_eq!(rows(&stdout), vec!["cycle=3 n=6"; 3], "{stdout}");
    // Unassigned, the coordinate advances.
    let (ok, stdout, stderr) = run_binary(&[
        "run",
        path.to_str().unwrap(),
        "--cycles",
        "3",
        "--emit",
        "map",
        "--outputs",
        "cycle,n",
        "-q",
    ]);
    assert!(ok, "{stderr}");
    assert_eq!(
        rows(&stdout),
        vec!["cycle=0 n=0", "cycle=1 n=2", "cycle=2 n=4"],
        "{stdout}"
    );
}

#[test]
fn jsonl_rows_carry_typed_values() {
    let path = write_temp(
        "jsonl",
        "input cycle: u64\nf := to_f64(cycle) / 2.0\ns := \"x{cycle}\"\nb := u64_gt(cycle, 0)\n",
    );
    let (ok, stdout, stderr) = run_binary(&[
        "run",
        path.to_str().unwrap(),
        "--cycles",
        "2",
        "--emit",
        "jsonl",
        "--outputs",
        "cycle,f,s,b",
        "-q",
    ]);
    assert!(ok, "{stderr}");
    let rows = rows(&stdout);
    assert_eq!(rows.len(), 2, "{stdout}");
    let second: serde_json::Value = serde_json::from_str(&rows[1]).unwrap();
    assert_eq!(second["cycle"], 1);
    assert_eq!(second["f"], 0.5);
    assert_eq!(second["s"], "x1");
    assert_eq!(second["b"], 1);
}
