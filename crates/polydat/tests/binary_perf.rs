// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `polydat perf`: the suite runs from the built-in definition and from
//! a TOML file, reports grouped results, writes and reads saved results,
//! and names what it cannot run. Timings are kept tiny; these check the
//! command, not the numbers.

use std::process::Command;

fn perf(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_polydat"))
        .arg("perf")
        .args(["--rounds", "1", "--warmup-ms", "1", "--measure-ms", "5"])
        .args(args)
        .output()
        .expect("polydat binary runs");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn temp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("polydat-perf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

#[test]
fn the_builtin_suite_reports_each_group_and_engine() {
    let (ok, out, err) = perf(&["--engine", "interpreter", "--engine", "closures"]);
    assert!(ok, "{err}");
    for want in [
        "ladder",
        "conversions",
        "interpreter",
        "closures",
        "ns/cycle",
    ] {
        assert!(out.contains(want), "no `{want}` in:\n{out}");
    }
    assert!(
        out.contains("ladder ok") || out.contains("ladder BROKEN"),
        "{out}"
    );
}

#[test]
fn a_printed_config_runs_as_a_suite_file() {
    let (ok, toml, err) = perf(&["--print-config", "--group", "conversions"]);
    assert!(ok, "{err}");
    assert!(
        toml.contains("[[group]]") && toml.contains("conversions"),
        "{toml}"
    );
    let path = temp("suite.toml");
    std::fs::write(&path, &toml).unwrap();
    let (ok, out, err) = perf(&["--config", path.to_str().unwrap(), "--engine", "closures"]);
    assert!(ok, "{err}");
    assert!(
        out.contains("conversions") && out.contains("closures"),
        "{out}"
    );
}

#[test]
fn a_program_file_is_read_relative_to_the_suite() {
    let program = temp("prog.polydat");
    std::fs::write(&program, "input cycle: u64\nout := hash(cycle)\n").unwrap();
    let suite = temp("relative.toml");
    std::fs::write(
        &suite,
        "[[group]]\nname = \"hashing\"\nprogram = \"prog.polydat\"\nengines = [\"closures\"]\n",
    )
    .unwrap();
    let (ok, out, err) = perf(&["--config", suite.to_str().unwrap()]);
    assert!(ok, "{err}");
    assert!(
        out.contains("hashing") && out.contains("prog.polydat"),
        "{out}"
    );
}

#[test]
fn a_round_is_reported_as_json() {
    let (ok, out, err) = perf(&["--round-json", "--group", "ladder", "--engine", "closures"]);
    assert!(ok, "{err}");
    let v: serde_json::Value = serde_json::from_str(&out).expect("one JSON report");
    let rungs = v["results"].as_array().unwrap();
    assert_eq!(rungs.len(), 1);
    assert_eq!(rungs[0]["engine"], "closures");
    assert_eq!(rungs[0]["rounds"].as_array().unwrap().len(), 1);
}

#[test]
fn saved_results_compare_against_a_new_run() {
    let saved = temp("saved.json");
    let args = ["--group", "ladder", "--engine", "closures"];
    let (ok, _, err) = perf(&[&args[..], &["--save", saved.to_str().unwrap()]].concat());
    assert!(ok, "{err}");
    let (ok, out, err) = perf(&[&args[..], &["--compare", saved.to_str().unwrap()]].concat());
    assert!(ok, "{err}");
    assert!(out.contains("saved") && out.contains("delta"), "{out}");
}

#[test]
fn an_unknown_engine_is_an_error_that_names_it() {
    let suite = temp("bad.toml");
    std::fs::write(
        &suite,
        "[[group]]\nname = \"g\"\nsource = \"input cycle: u64\\nout := hash(cycle)\"\nengines = [\"turbo\"]\n",
    )
    .unwrap();
    let (ok, _, err) = perf(&["--config", suite.to_str().unwrap()]);
    assert!(!ok);
    assert!(err.contains("turbo"), "{err}");
}

/// The checked-in cone spectrum suite runs: every generated group
/// builds, and the report says each graph's shape and size.
#[test]
fn the_cone_spectrum_suite_runs() {
    let suite = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/perf/cone_spectrum.toml"
    );
    let (ok, out, err) = perf(&["--config", suite, "--engine", "closures"]);
    assert!(ok, "{err}");
    for want in [
        "chains-64-one",
        "trunk-16-all",
        "lattice-16x8-one",
        "chain-256",
    ] {
        assert!(out.contains(want), "no `{want}` in:\n{out}");
    }
    assert!(out.contains("bindings, pulling 1 output(s)"), "{out}");
}

/// A group names its program exactly once.
#[test]
fn a_group_with_two_programs_is_an_error() {
    let suite = temp("two.toml");
    std::fs::write(
        &suite,
        "[[group]]\nname = \"g\"\nsource = \"input cycle: u64\\nout := hash(cycle)\"\ngenerate = { shape = \"chains\" }\n",
    )
    .unwrap();
    let (ok, _, err) = perf(&["--config", suite.to_str().unwrap()]);
    assert!(!ok);
    assert!(err.contains("more than once"), "{err}");
}
