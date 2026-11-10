---
type: guide
title: Releasing polydat
description: How a release chooses its version, the porting guide every minor release ships, and the steps of a release, including raising the internal dependency requirements so a host's lockfile picks up every crate the release publishes.
tags: [release]
---

# Releasing polydat

A release publishes the crates that changed since the last one, with
`cargo-workspaces`, which bumps their versions, tags the release, and
records it as a `Release <version>` commit listing each crate it
published. The facade `polydat` re-exports the other crates, and a host
depends on the facade alone, so a release is correct only when the
facade it publishes requires the versions of the other crates it was
built and tested with.

## Which version a release takes

A release that contains any breaking change bumps the minor version
(0.x.0). A patch release (0.x.y) carries only compatible fixes: a host
that updates to it compiles unchanged, and its behavior changes only by
the corrections the release names. A change is breaking when a host
that compiled against the previous release stops compiling, or when a
program or call that succeeded gives a different result for a reason
other than a named correction.

Every minor release ships a porting guide,
`guides/porting_to_X_Y_0.md`, that lists every change a host sees since
the previous minor release, patch releases included, in the structure
of the existing guides: what stops compiling and its fix, what changes
quietly, the new capabilities, and the known issues.

## Why the requirements must be raised

The root `Cargo.toml` declares every internal crate once, under
`[workspace.dependencies]`, with a path and a version requirement.
Cargo reads a requirement such as `version = "0.5.0"` as "any
compatible 0.5 version", so a host whose lockfile already holds
`polydat-core` 0.5.2 keeps it when it updates to a facade 0.5.3 that
declares that requirement. The host then runs the new facade over the
old runtime, without the fixes that the release carried in the
runtime. A requirement that names the newest published version of its
crate makes Cargo move the host's lockfile to that version.

## Steps

1. **Write the porting guide** (a minor release only). Write
   `crates/polydat/docs/guides/porting_to_X_Y_0.md`, add it to the
   Guides list in `crates/polydat/docs/README.md`, newest first,
   regenerate the indexes with `ct okf gen-index --recursive`, and
   commit it.
2. **Run the engine ladder** after a change to the closure-tier,
   native, or pure-native evaluation loops, against a same-hour
   baseline ([Engine-ladder performance](performance.md)). The release
   gate does not run it.
3. **Release.** From a clean `main`, `cargo ws-release minor` for a
   release with a breaking change and `cargo ws-release patch`
   otherwise (`custom X.Y.Z` names the version); add `--dry-run` to
   rehearse it. The alias runs `xtask release`, which:
   - **gates the tree** on every check CI runs, and stops before
     versioning anything at the first that fails:
     - `cargo fmt --all --check`;
     - `cargo clippy --workspace --all-targets -- -D warnings`, and
       `cargo doc --workspace --no-deps` with `RUSTDOCFLAGS=-D warnings`,
       each with default features, `--no-default-features` and
       `--all-features`;
     - `cargo nextest run --workspace --profile ci` with the same three
       feature sets, which includes the node reference and the engine
       matrix checked against the whole library;
     - `cargo test --doc --workspace --all-features`, the rustdoc
       examples, which nextest does not run;
     - `cargo check --release --workspace --all-features --all-targets`;
     - the node attribute diagnostics (`ui::`, run with
       `--run-ignored ignored-only`);
   - then runs `cargo ws publish --force '*' --all --exact
     --allow-branch main` with the options given. It versions every
     crate together, at the version the workspace inherits; raises
     every internal requirement under `[workspace.dependencies]` to
     that version, exactly (`--exact`); commits `Release <version>`,
     tags it, and pushes both; and publishes the crates in dependency order: `polydat-derive`,
     `polydat-grammar`, `polydat-core`, `polydat-nodes`, `polydat`.
     `xtask` is versioned with them and never published.
4. **If a publish stops** on a crate whose dependency crates.io does
   not list yet (`failed to select a version for the requirement`),
   the index has not caught up with the crate published just before
   it; wait a minute and publish the rest with `cargo publish -p
   <crate>`, in the same order.
5. **Verify from a host's side.** In a project that already depends on
   polydat, run `cargo update -p polydat` and then `cargo tree -i
   polydat-core`; every internal crate must resolve to the version this
   release published or, for a crate it did not publish, to that
   crate's newest published version.
