// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Repository tasks, run through the cargo aliases in
//! `.cargo/config.toml`.
//!
//! `cargo ws-release <bump> [cargo-workspaces options]` runs
//! `xtask release`: every check CI runs, and the rustdoc examples, and
//! then, only when all of them pass, `cargo ws publish` with the
//! release's fixed options and the ones given
//! (`crates/polydat/docs/guides/releasing.md`). `--dry-run` passes
//! through to `cargo ws publish`, so a rehearsal is gated the same way.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

/// One check of the release gate: what it is called and the cargo
/// arguments that run it.
struct Check {
    name: &'static str,
    args: &'static [&'static str],
    /// Set `RUSTDOCFLAGS=-D warnings`, as CI's rustdoc steps do.
    rustdoc_warnings_are_errors: bool,
}

const fn check(name: &'static str, args: &'static [&'static str]) -> Check {
    Check {
        name,
        args,
        rustdoc_warnings_are_errors: false,
    }
}

const fn rustdoc(name: &'static str, args: &'static [&'static str]) -> Check {
    Check {
        name,
        args,
        rustdoc_warnings_are_errors: true,
    }
}

/// The release gate: the steps of `.github/workflows/test.yml`, for
/// the three feature sets CI covers, and the rustdoc examples. Kept in
/// the order a failure is cheapest to find.
const GATE: &[Check] = &[
    check("rustfmt", &["fmt", "--all", "--check"]),
    check(
        "clippy (default features)",
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
    ),
    check(
        "clippy (no default features)",
        &[
            "clippy",
            "--workspace",
            "--no-default-features",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
    ),
    check(
        "clippy (all features)",
        &[
            "clippy",
            "--workspace",
            "--all-features",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
    ),
    rustdoc(
        "rustdoc (default features)",
        &["doc", "--workspace", "--no-deps"],
    ),
    rustdoc(
        "rustdoc (no default features)",
        &["doc", "--workspace", "--no-deps", "--no-default-features"],
    ),
    rustdoc(
        "rustdoc (all features)",
        &["doc", "--workspace", "--no-deps", "--all-features"],
    ),
    check(
        "tests (default features)",
        &["nextest", "run", "--workspace", "--profile", "ci"],
    ),
    check(
        "tests (no default features)",
        &[
            "nextest",
            "run",
            "--workspace",
            "--no-default-features",
            "--profile",
            "ci",
        ],
    ),
    check(
        "tests (all features)",
        &[
            "nextest",
            "run",
            "--workspace",
            "--all-features",
            "--profile",
            "ci",
        ],
    ),
    check(
        "doc examples",
        &["test", "--doc", "--workspace", "--all-features"],
    ),
    check(
        "release build",
        &[
            "check",
            "--release",
            "--workspace",
            "--all-features",
            "--all-targets",
        ],
    ),
    check(
        "node attribute diagnostics",
        &[
            "nextest",
            "run",
            "-p",
            "polydat",
            "--all-features",
            "--test",
            "suite",
            "--run-ignored",
            "ignored-only",
            "ui::",
        ],
    ),
];

/// The options every release passes to `cargo ws publish`: version
/// every crate together (`--force '*'`, `--all`), pin the internal
/// requirements to the released version (`--exact`), and release only
/// from `main`.
const PUBLISH: &[&str] = &[
    "ws",
    "publish",
    "--force",
    "*",
    "--all",
    "--exact",
    "--allow-branch",
    "main",
];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.split_first() {
        Some((task, rest)) if task == "release" => release(rest),
        _ => {
            eprintln!(
                "usage: cargo ws-release <patch|minor|major|custom X.Y.Z> [cargo-workspaces options]"
            );
            ExitCode::FAILURE
        }
    }
}

/// Run the gate, then publish with `publish_args` after the fixed
/// options.
fn release(publish_args: &[String]) -> ExitCode {
    if publish_args.is_empty() {
        eprintln!("cargo ws-release: name the bump, such as `patch` or `minor`");
        return ExitCode::FAILURE;
    }
    let root = workspace_root();
    for (index, check) in GATE.iter().enumerate() {
        eprintln!(
            "\n==> release gate {}/{}: {}",
            index + 1,
            GATE.len(),
            check.name
        );
        let mut command = cargo(&root);
        command.args(check.args);
        if check.rustdoc_warnings_are_errors {
            command.env("RUSTDOCFLAGS", "-D warnings");
        }
        if !succeeds(&mut command) {
            eprintln!(
                "\ncargo ws-release: the release gate failed at {}; nothing was versioned or published",
                check.name
            );
            return ExitCode::FAILURE;
        }
    }
    eprintln!(
        "\n==> release gate passed; publishing with cargo ws publish {}",
        publish_args.join(" ")
    );
    let mut command = cargo(&root);
    command.args(PUBLISH).args(publish_args);
    if succeeds(&mut command) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// The cargo that runs this task, so the gate builds with the same
/// toolchain, in the workspace root.
fn cargo(root: &Path) -> Command {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = Command::new(cargo);
    command.current_dir(root);
    command
}

/// Whether `command` ran and exited successfully; a command that
/// cannot start, such as a missing `cargo nextest`, fails by name.
fn succeeds(command: &mut Command) -> bool {
    match command.status() {
        Ok(status) => status.success(),
        Err(e) => {
            eprintln!("cargo ws-release: could not run {command:?}: {e}");
            false
        }
    }
}

/// The workspace root: the directory above this crate's.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives in the workspace root")
        .to_path_buf()
}
