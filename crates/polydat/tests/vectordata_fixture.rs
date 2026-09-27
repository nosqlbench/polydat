// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The dataset accessors against a real facet on disk.
//!
//! `tests/fixtures/vectordata` holds a catalog with one dataset,
//! `polydat-tiny`, in the on-disk format the `vectordata` crate reads:
//! six base vectors and three query vectors of dimension 3 (`fvec`),
//! a `u8` metadata facet with one value per base vector, and an `i32`
//! predicates facet with one value per query. The tests point the
//! vectordata client configuration at that catalog and open the
//! dataset by name, so every read goes through the production path:
//! catalog resolution, `dataset.yaml`, and the typed and uniform
//! readers over the files. The answers are the constants below, and
//! every engine that builds the program gives them.
//!
//! The binary facets are the output of [`facet_files`];
//! `the_fixture_is_what_its_generator_writes` holds the checked-in
//! bytes to it, and the ignored `write_the_fixture` writes them.

#![cfg(feature = "vectordata")]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use polydat::ast::{SliceArc, Value};
use polydat::dsl::compile::{compile_polydat_interpreter, compile_polydat_to_assembler};
use polydat::{Engine, JitMode, KernelError, Provenance};

/// The dataset and profile the fixture catalog names.
const SOURCE: &str = "polydat-tiny:default";
/// The metadata value of each base vector.
const METADATA: [u8; 6] = [2, 0, 2, 5, 2, 0];
/// The predicate value of each query.
const PREDICATES: [i32; 3] = [2, 5, 2];
/// The dimension of every vector.
const DIM: usize = 3;

/// Base vector `i`: `[i, i + 0.5, -i]`, exact in `f32`.
fn base_vector(i: usize) -> Vec<f32> {
    let x = i as f32;
    vec![x, x + 0.5, -x]
}

/// Query vector `q`: `[q + 0.25, 1, -0.5]`, exact in `f32`.
fn query_vector(q: usize) -> Vec<f32> {
    vec![q as f32 + 0.25, 1.0, -0.5]
}

/// Records in the `fvec` layout: each one a little-endian `i32`
/// dimension followed by that many little-endian `f32` values.
fn fvec(records: impl Iterator<Item = Vec<f32>>) -> Vec<u8> {
    let mut out = Vec::new();
    for r in records {
        out.extend_from_slice(&(r.len() as i32).to_le_bytes());
        for x in r {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

/// The fixture's binary facets, by file name under `tiny/`. A scalar
/// facet is its values back to back, little-endian, with the element
/// type named by the extension.
fn facet_files() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        (
            "base_vectors.fvec",
            fvec((0..METADATA.len()).map(base_vector)),
        ),
        (
            "query_vectors.fvec",
            fvec((0..PREDICATES.len()).map(query_vector)),
        ),
        ("metadata_content.u8", METADATA.to_vec()),
        (
            "metadata_predicates.i32",
            PREDICATES.iter().flat_map(|p| p.to_le_bytes()).collect(),
        ),
    ]
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vectordata")
}

/// Point the vectordata client at the fixture catalog, once per
/// process.
///
/// The client reads its catalog list from `catalogs.yaml` under
/// `$VECTORDATA_HOME`, which is the crate's seam for isolating its
/// configuration; the file goes in a directory of this process's own
/// under the target's scratch space, naming the fixture catalog by
/// absolute path. The suite runs one process per test under nextest,
/// so the variable is set before any thread of this process reads it.
fn use_fixture_catalog() {
    static HOME: OnceLock<PathBuf> = OnceLock::new();
    HOME.get_or_init(|| {
        let home = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("vectordata-home-{}", std::process::id()));
        std::fs::create_dir_all(&home).expect("create the vectordata home");
        let catalog = fixture_dir().to_string_lossy().replace('\\', "/");
        std::fs::write(
            home.join("catalogs.yaml"),
            format!("fixture: '{catalog}'\n"),
        )
        .expect("write catalogs.yaml");
        // SAFETY: set once, before the first catalog read of this
        // process; nothing else in the process writes the variable.
        unsafe { std::env::set_var("VECTORDATA_HOME", &home) };
        home
    });
}

/// Every engine a program can be built on in this build.
fn engines() -> Vec<Engine> {
    let mut all = vec![
        Engine::Interpreter(JitMode::Off),
        Engine::Interpreter(JitMode::Force),
    ];
    for m in [Provenance::Raw, Provenance::PushPull, Provenance::Auto] {
        all.push(Engine::Closures(m));
        all.push(Engine::Native(m));
    }
    if cfg!(feature = "jit") {
        all.push(Engine::PureNative(Provenance::PushPull));
    }
    all
}

/// The dataset nodes these programs call. None of them has a compiled
/// form in this build, so the compiled engines refuse a program that
/// calls one, naming the node, and the program runs on the
/// interpreter. docs/design/engines.md §8 says every registered node
/// has a kit; the refusal is tolerated here, by these names only, so
/// that the checks below cover each compiled engine as soon as it
/// accepts the program.
const DATASET_NODES: [&str; 12] = [
    "dataset_open",
    "dataset_prebuffer",
    "metadata_value_at",
    "predicate_value_at",
    "metadata_content_count",
    "metadata_count_of",
    "predicate_count_of",
    "vector_at",
    "query_vector_at",
    "vector_count",
    "query_count",
    "vector_dim",
];

/// Handles a host opens, by name: each `(name, call)` pair is a
/// binding pulled from an interpreter kernel, and the handle it holds
/// is the one the host passes to another kernel as an extern.
fn opened(bindings: &[(&str, String)]) -> Vec<(String, Value)> {
    use_fixture_catalog();
    let src: String = bindings
        .iter()
        .map(|(name, call)| format!("{name} := {call}\n"))
        .collect();
    let mut k = compile_polydat_interpreter(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    bindings
        .iter()
        .map(|(name, _)| {
            let h = k.pull_ref(name).clone();
            assert!(matches!(h, Value::Handle(_)), "{name} is a handle: {h:?}");
            (name.to_string(), h)
        })
        .collect()
}

/// Build `src` on every engine, set `externs`, and check each output
/// against `expected(output, cycle)` for cycles `0..cycles`, as display
/// text. Both interpreter modes run the program. A compiled engine that
/// refuses it names one of the [`DATASET_NODES`]; any other refusal
/// fails the check.
fn check_on_every_engine(
    src: &str,
    externs: &[(String, Value)],
    outputs: &[&str],
    cycles: u64,
    expected: impl Fn(&str, u64) -> String,
) {
    use_fixture_catalog();
    for engine in engines() {
        let asm = compile_polydat_to_assembler(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
        let mut k = match asm.compile_with(engine) {
            Ok(k) => k,
            Err(KernelError::Refused { reason, .. })
                if !matches!(engine, Engine::Interpreter(_))
                    && DATASET_NODES
                        .iter()
                        .any(|n| reason.contains(&format!("node '{n}'"))) =>
            {
                continue;
            }
            Err(e) => panic!("{engine}: {e}\n{src}"),
        };
        for (name, value) in externs {
            k.set_input(name, value.clone())
                .unwrap_or_else(|e| panic!("{engine}: set {name}: {e}\n{src}"));
        }
        for c in 0..cycles {
            k.set_inputs(&[c]);
            for out in outputs {
                assert_eq!(
                    k.pull(out).to_display_string(),
                    expected(out, c),
                    "{engine}: {out} at cycle {c}\n{src}"
                );
            }
        }
    }
}

fn metadata_at(i: u64) -> String {
    METADATA[i as usize % METADATA.len()].to_string()
}

fn predicate_at(q: u64) -> String {
    PREDICATES[q as usize % PREDICATES.len()].to_string()
}

fn metadata_count(v: u64) -> String {
    METADATA
        .iter()
        .filter(|m| **m as u64 == v)
        .count()
        .to_string()
}

fn predicate_count(v: u64) -> String {
    PREDICATES
        .iter()
        .filter(|p| **p as u64 == v)
        .count()
        .to_string()
}

/// The metadata and predicate answers, by output name.
fn facet_answer(out: &str, c: u64) -> String {
    match out {
        "meta" => metadata_at(c),
        "pred" => predicate_at(c),
        "meta_n" => METADATA.len().to_string(),
        "meta_of" => metadata_count(c),
        "pred_of" => predicate_count(c),
        other => panic!("no answer for {other}"),
    }
}

const FACET_OUTPUTS: [&str; 5] = ["meta", "pred", "meta_n", "meta_of", "pred_of"];

/// The accessors called with the dataset's source string, which
/// promotes to the facet each one names. The value index wraps modulo
/// the record count, and the counted values run past every value the
/// facets hold, so a value no record carries counts zero.
#[test]
fn facet_accessors_read_the_fixture_through_a_source_string() {
    let src = format!(
        "input cycle: u64\n\
         meta := metadata_value_at(\"{SOURCE}\", cycle)\n\
         pred := predicate_value_at(\"{SOURCE}\", cycle)\n\
         meta_n := metadata_content_count(\"{SOURCE}\")\n\
         meta_of := metadata_count_of(\"{SOURCE}\", to_i64(cycle))\n\
         pred_of := predicate_count_of(\"{SOURCE}\", to_i64(cycle))\n"
    );
    check_on_every_engine(&src, &[], &FACET_OUTPUTS, 8, facet_answer);
}

/// The same answers through handles `dataset_open` resolves in the
/// program.
#[test]
fn facet_accessors_read_the_fixture_through_opened_handles() {
    let src = format!(
        "input cycle: u64\n\
         content := dataset_open(\"{SOURCE}\", \"metadata_content\")\n\
         preds := dataset_open(\"{SOURCE}\", \"metadata_predicates\")\n\
         meta := metadata_value_at(content, cycle)\n\
         pred := predicate_value_at(preds, cycle)\n\
         meta_n := metadata_content_count(content)\n\
         meta_of := metadata_count_of(content, to_i64(cycle))\n\
         pred_of := predicate_count_of(preds, to_i64(cycle))\n"
    );
    check_on_every_engine(&src, &[], &FACET_OUTPUTS, 8, facet_answer);
}

/// The same answers with the facet handles opened by the host and
/// passed in as externs, so the program calls no resolver.
#[test]
fn facet_accessors_read_the_fixture_through_handle_externs() {
    let externs = opened(&[
        (
            "content",
            format!("dataset_open(\"{SOURCE}\", \"metadata_content\")"),
        ),
        (
            "preds",
            format!("dataset_open(\"{SOURCE}\", \"metadata_predicates\")"),
        ),
    ]);
    let src = "input cycle: u64\n\
               extern content: handle\n\
               extern preds: handle\n\
               meta := metadata_value_at(content, cycle)\n\
               pred := predicate_value_at(preds, cycle)\n\
               meta_n := metadata_content_count(content)\n\
               meta_of := metadata_count_of(content, to_i64(cycle))\n\
               pred_of := predicate_count_of(preds, to_i64(cycle))\n";
    check_on_every_engine(src, &externs, &FACET_OUTPUTS, 8, facet_answer);
}

/// The value and count-of accessors through a prebuffered handle,
/// which each one resolves to its own facet, in the program and as an
/// extern. `metadata_content_count` does not resolve a prebuffered
/// handle to its facet and answers 0 for one, so it is not checked
/// here.
#[test]
fn facet_accessors_read_the_fixture_through_a_prebuffered_handle() {
    let body = "meta := metadata_value_at(ds, cycle)\n\
                pred := predicate_value_at(ds, cycle)\n\
                meta_of := metadata_count_of(ds, to_i64(cycle))\n\
                pred_of := predicate_count_of(ds, to_i64(cycle))\n";
    let outputs = ["meta", "pred", "meta_of", "pred_of"];
    let src = format!("input cycle: u64\nds := dataset_prebuffer(\"{SOURCE}\")\n{body}");
    check_on_every_engine(&src, &[], &outputs, 8, facet_answer);
    let externs = opened(&[("ds", format!("dataset_prebuffer(\"{SOURCE}\")"))]);
    let src = format!("input cycle: u64\nextern ds: handle\n{body}");
    check_on_every_engine(&src, &externs, &outputs, 8, facet_answer);
}

/// The vector answers, by output name.
fn vector_answer(out: &str, c: u64) -> String {
    let vec_text = |v: Vec<f32>| Value::VecF32(SliceArc::from_vec(v)).to_display_string();
    match out {
        "base" => vec_text(base_vector(c as usize % METADATA.len())),
        "query" => vec_text(query_vector(c as usize % PREDICATES.len())),
        "base_n" => METADATA.len().to_string(),
        "query_n" => PREDICATES.len().to_string(),
        "dim" => DIM.to_string(),
        other => panic!("no answer for {other}"),
    }
}

const VECTOR_OUTPUTS: [&str; 5] = ["base", "query", "base_n", "query_n", "dim"];

/// The vectors beside the facets: the same catalog path serves the
/// uniform readers, through source strings and through handle externs.
#[test]
fn vector_accessors_read_the_fixture() {
    let src = format!(
        "input cycle: u64\n\
         base := vector_at(\"{SOURCE}\", cycle)\n\
         query := query_vector_at(\"{SOURCE}\", cycle)\n\
         base_n := vector_count(\"{SOURCE}\")\n\
         query_n := query_count(\"{SOURCE}\")\n\
         dim := vector_dim(\"{SOURCE}\")\n"
    );
    check_on_every_engine(&src, &[], &VECTOR_OUTPUTS, 8, vector_answer);
    let externs = opened(&[
        ("b", format!("dataset_open(\"{SOURCE}\", \"base\")")),
        ("q", format!("dataset_open(\"{SOURCE}\", \"query\")")),
    ]);
    let src = "input cycle: u64\n\
               extern b: handle\n\
               extern q: handle\n\
               base := vector_at(b, cycle)\n\
               query := query_vector_at(q, cycle)\n\
               base_n := vector_count(b)\n\
               query_n := query_count(q)\n\
               dim := vector_dim(b)\n";
    check_on_every_engine(src, &externs, &VECTOR_OUTPUTS, 8, vector_answer);
}

/// The checked-in facets are the bytes [`facet_files`] writes.
#[test]
fn the_fixture_is_what_its_generator_writes() {
    let dir = fixture_dir().join("tiny");
    for (name, bytes) in facet_files() {
        let on_disk = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"));
        assert_eq!(
            on_disk, bytes,
            "{name} differs from its generator; run write_the_fixture"
        );
    }
}

/// Write the fixture's binary facets. Run once after changing the
/// constants above:
/// `cargo nextest run -p polydat --features vectordata --run-ignored only -E 'test(write_the_fixture)'`.
#[test]
#[ignore]
fn write_the_fixture() {
    let dir = fixture_dir().join("tiny");
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    for (name, bytes) in facet_files() {
        std::fs::write(dir.join(name), bytes).unwrap_or_else(|e| panic!("write {name}: {e}"));
    }
}
