---
type: guide
title: Releasing polydat
description: The steps of a release, including raising the internal dependency requirements so a host's lockfile picks up every crate the release publishes.
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

1. **Gate the tree.** From the repository root:
   - `cargo fmt --all --check`
   - `cargo clippy --workspace --all-targets`
   - `cargo nextest run --workspace`
   - `cargo nextest run -p polydat --all-features -E 'test(/nodes_reference|engine_parity/)'`
     (the node reference and the engine matrix are checked against the
     whole library)
   - `cargo check -p polydat --no-default-features`
   - After a change to the closure-tier, native, or pure-native
     evaluation loops, the engine ladder against a same-hour baseline
     ([Engine-ladder performance](performance.md)).
2. **Bump the versions.** `cargo workspaces version` bumps the crates
   that changed and the workspace version they inherit.
3. **Raise the internal requirements.** In the root `Cargo.toml`, under
   `[workspace.dependencies]`, set the `version` of every internal crate
   this release publishes to the version it is being published at. A
   crate this release does not publish keeps the requirement that names
   its newest published version. This has to follow step 2: a
   requirement may not name a version the crate in the tree does not
   yet have.
4. **Check the packages.** `cargo publish --dry-run -p <crate>` for each
   crate being published, in dependency order: `polydat-derive`,
   `polydat-grammar`, `polydat-core`, `polydat-nodes`, `polydat`.
5. **Publish.** `cargo workspaces publish --from-git` publishes the
   bumped crates in dependency order and commits `Release <version>`.
6. **Verify from a host's side.** In a project that already depends on
   polydat, run `cargo update -p polydat` and then `cargo tree -i
   polydat-core`; every internal crate must resolve to the version this
   release published or, for a crate it did not publish, to that
   crate's newest published version.
