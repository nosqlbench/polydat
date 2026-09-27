// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The dataset accessors against a real facet on disk.
//!
//! `tests/fixtures/vectordata` holds a catalog with one dataset,
//! `polydat-tiny`, in the on-disk format the `vectordata` crate reads:
//! six base vectors and three query vectors of dimension 3 (`fvec`),
//! a `u8` metadata facet with one value per base vector, an `i32`
//! predicates facet with one value per query, the ground truth of
//! each query (its two nearest base vectors and their squared
//! Euclidean distances, and its nearest base vector among those whose
//! metadata value is the query's predicate), and the variable-length
//! `ivvec` metadata results naming, per query, every base vector whose
//! metadata value is its predicate. The tests point the vectordata
//! client configuration at a per-process copy of that catalog and open
//! the dataset by name, so every read goes through the production
//! path: catalog resolution, `dataset.yaml`, and the typed, uniform,
//! and variable-length readers over the files. The answers are the
//! constants below, and every engine that builds the program gives
//! them.
//!
//! The binary facets are the output of [`facet_files`];
//! `the_fixture_is_what_its_generator_writes` holds the checked-in
//! bytes to it, and the ignored `write_the_fixture` writes them.
//! `the_ground_truth_is_the_true_nearest_neighbors` holds the
//! ground-truth constants to a brute-force search over the vectors.

#![cfg(feature = "vectordata")]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use polydat::ast::{SliceArc, Value};
use polydat::dsl::compile::{compile_polydat_interpreter, compile_polydat_to_assembler};
use polydat::dsl::registry::DefaultResolver;
use polydat::{Engine, JitMode, Kernel, Provenance};

/// The dataset and profile the fixture catalog names.
const SOURCE: &str = "polydat-tiny:default";
/// The metadata value of each base vector.
const METADATA: [u8; 6] = [2, 0, 2, 5, 2, 0];
/// The predicate value of each query.
const PREDICATES: [i32; 3] = [2, 5, 2];
/// The dimension of every vector.
const DIM: usize = 3;
/// The two nearest base vectors of each query, nearest first.
const NEIGHBORS: [[i32; 2]; 3] = [[0, 1], [1, 0], [1, 2]];
/// The squared Euclidean distance from each query to each of its
/// [`NEIGHBORS`].
const DISTANCES: [[f32; 2]; 3] = [[0.5625, 1.0625], [0.5625, 2.0625], [2.0625, 4.5625]];
/// The nearest base vector of each query among those whose metadata
/// value is the query's predicate.
const FILTERED_NEIGHBORS: [[i32; 1]; 3] = [[0], [3], [2]];
/// The squared Euclidean distance from each query to its
/// [`FILTERED_NEIGHBORS`].
const FILTERED_DISTANCES: [[f32; 1]; 3] = [[0.5625], [15.5625], [4.5625]];
/// The base vectors whose metadata value is each query's predicate.
const METADATA_RESULTS: [&[i32]; 3] = [&[0, 2, 4], &[3], &[0, 2, 4]];

/// Base vector `i`: `[i, i + 0.5, -i]`, exact in `f32`.
fn base_vector(i: usize) -> Vec<f32> {
    let x = i as f32;
    vec![x, x + 0.5, -x]
}

/// Query vector `q`: `[q + 0.25, 1, -0.5]`, exact in `f32`.
fn query_vector(q: usize) -> Vec<f32> {
    vec![q as f32 + 0.25, 1.0, -0.5]
}

/// Records in the `xvec` layout: each one a little-endian `i32`
/// dimension followed by that many 4-byte little-endian values. The
/// `fvec`, `ivec`, and `ivvec` files all have it; an `fvec` or `ivec`
/// has one dimension throughout, and an `ivvec` a dimension per record.
fn xvec<T: Copy, R: AsRef<[T]>>(
    records: impl IntoIterator<Item = R>,
    le: fn(T) -> [u8; 4],
) -> Vec<u8> {
    let mut out = Vec::new();
    for r in records {
        let r = r.as_ref();
        out.extend_from_slice(&(r.len() as i32).to_le_bytes());
        for x in r {
            out.extend_from_slice(&le(*x));
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
            xvec((0..METADATA.len()).map(base_vector), f32::to_le_bytes),
        ),
        (
            "query_vectors.fvec",
            xvec((0..PREDICATES.len()).map(query_vector), f32::to_le_bytes),
        ),
        ("metadata_content.u8", METADATA.to_vec()),
        (
            "metadata_predicates.i32",
            PREDICATES.iter().flat_map(|p| p.to_le_bytes()).collect(),
        ),
        ("neighbor_indices.ivec", xvec(NEIGHBORS, i32::to_le_bytes)),
        ("neighbor_distances.fvec", xvec(DISTANCES, f32::to_le_bytes)),
        (
            "prefiltered_neighbor_indices.ivec",
            xvec(FILTERED_NEIGHBORS, i32::to_le_bytes),
        ),
        (
            "prefiltered_neighbor_distances.fvec",
            xvec(FILTERED_DISTANCES, f32::to_le_bytes),
        ),
        (
            "metadata_results.ivvec",
            xvec(METADATA_RESULTS, i32::to_le_bytes),
        ),
    ]
}

/// The squared Euclidean distance between two vectors.
fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// The base vectors a query's ground truth ranks, `(distance, ordinal)`
/// nearest first, among the ordinals `admit` accepts.
fn ranked(q: usize, admit: impl Fn(usize) -> bool) -> Vec<(f32, i32)> {
    let query = query_vector(q);
    let mut all: Vec<(f32, i32)> = (0..METADATA.len())
        .filter(|&i| admit(i))
        .map(|i| (squared_l2(&base_vector(i), &query), i as i32))
        .collect();
    all.sort_by(|a, b| a.partial_cmp(b).expect("distances are finite"));
    all
}

/// The ground-truth constants are what a brute-force search over the
/// fixture's vectors finds, with no ties to break.
#[test]
fn the_ground_truth_is_the_true_nearest_neighbors() {
    for q in 0..PREDICATES.len() {
        let all = ranked(q, |_| true);
        assert!(all[1].0 < all[2].0, "query {q}: a tie at the second place");
        let top: Vec<(f32, i32)> = DISTANCES[q].iter().copied().zip(NEIGHBORS[q]).collect();
        assert_eq!(all[..2], top[..], "query {q}");

        let matching = |i: usize| i32::from(METADATA[i]) == PREDICATES[q];
        let filtered = ranked(q, matching);
        assert!(
            filtered.len() < 2 || filtered[0].0 < filtered[1].0,
            "query {q}: a tie at the first filtered place"
        );
        assert_eq!(
            filtered[0],
            (FILTERED_DISTANCES[q][0], FILTERED_NEIGHBORS[q][0]),
            "query {q}"
        );

        let results: Vec<i32> = (0..METADATA.len())
            .filter(|&i| matching(i))
            .map(|i| i as i32)
            .collect();
        assert_eq!(results, METADATA_RESULTS[q], "query {q}");
    }
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vectordata")
}

/// Copy the directory tree at `from` to `to`.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap_or_else(|e| panic!("create {}: {e}", to.display()));
    for entry in std::fs::read_dir(from).unwrap_or_else(|e| panic!("read {}: {e}", from.display()))
    {
        let entry = entry.expect("a directory entry");
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target)
                .unwrap_or_else(|e| panic!("copy {}: {e}", entry.path().display()));
        }
    }
}

/// Point the vectordata client at a copy of the fixture catalog, once
/// per process.
///
/// The client reads its catalog list from `catalogs.yaml` under
/// `$VECTORDATA_HOME`, which is the crate's seam for isolating its
/// configuration; the file goes in a directory of this process's own
/// under the target's scratch space, naming by absolute path a copy of
/// the fixture catalog in the same directory. The copy is there
/// because the variable-length reader writes an offset index
/// (`IDXFOR__<file>.i32`) beside the file it opens, and a test process
/// writes that only into its own copy. The suite runs one process per
/// test under nextest, so the variable is set before any thread of
/// this process reads it.
fn use_fixture_catalog() {
    static HOME: OnceLock<PathBuf> = OnceLock::new();
    HOME.get_or_init(|| {
        let home = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("vectordata-home-{}", std::process::id()));
        let copy = home.join("catalog");
        if copy.exists() {
            std::fs::remove_dir_all(&copy).expect("clear the catalog copy");
        }
        copy_tree(&fixture_dir(), &copy);
        let catalog = copy.to_string_lossy().replace('\\', "/");
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
/// text. Every engine builds the program, each dataset node running
/// its kit on the compiled engines, so a refusal fails the check.
fn check_on_every_engine(
    src: &str,
    externs: &[(String, Value)],
    outputs: &[&str],
    cycles: u64,
    expected: impl Fn(&str, u64) -> String,
) {
    use_fixture_catalog();
    for engine in engines() {
        let mut k = kernel_on(engine, src, externs);
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

/// `src` built on `engine`, with `externs` set.
fn kernel_on(engine: Engine, src: &str, externs: &[(String, Value)]) -> Box<dyn Kernel> {
    let asm = compile_polydat_to_assembler(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    let mut k = asm
        .compile_with(engine)
        .unwrap_or_else(|e| panic!("{engine}: {e}\n{src}"));
    for (name, value) in externs {
        k.set_input(name, value.clone())
            .unwrap_or_else(|e| panic!("{engine}: set {name}: {e}\n{src}"));
    }
    k
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

/// A source string into a `Handle` port of a function with a default
/// resolver compiles to the resolver call (type_system.md §1.8): the
/// compiled graph holds a `dataset_open` node between the string and
/// the accessor, reading the string and the facet the accessor names.
#[test]
fn a_source_string_compiles_to_the_resolver_call() {
    use polydat::kernel::WireSource;
    let src = "input cycle: u64\n\
               extern source: str\n\
               meta := metadata_value_at(source, cycle)\n";
    let kernel = compile_polydat_interpreter(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    let program = kernel.program();
    let find = |name: &str| {
        (0..program.node_count())
            .find(|&i| program.node_meta(i).name == name)
            .unwrap_or_else(|| panic!("the compiled graph has no {name} node"))
    };
    let accessor = find("metadata_value_at");
    let WireSource::NodeOutput(resolver, 0) = program.node_wiring(accessor)[0] else {
        panic!(
            "the accessor's handle is not a node output: {:?}",
            program.node_wiring(accessor)
        );
    };
    assert_eq!(program.node_meta(resolver).name, "dataset_open");
    let wiring = program.node_wiring(resolver);
    assert!(
        matches!(wiring[0], WireSource::Input(i) if program.input_name_by_idx(i) == Some("source")),
        "the resolver reads the source string: {wiring:?}"
    );
    let WireSource::NodeOutput(facet, 0) = wiring[1] else {
        panic!("the facet is not a node output: {wiring:?}");
    };
    let mut out = vec![polydat::ast::Value::None];
    program.node_ref(facet).eval(&[], &mut out);
    assert_eq!(out[0].to_display_string(), "metadata_content");
}

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

/// The facets the fixture's profile declares: every facet a dataset
/// node reads.
const FIXTURE_FACETS: [&str; 9] = [
    "base",
    "query",
    "neighbor_indices",
    "neighbor_distances",
    "filtered_neighbor_indices",
    "filtered_neighbor_distances",
    "metadata_results",
    "metadata_content",
    "metadata_predicates",
];

/// Every facet accessor: its name, the facet it resolves, and the
/// arguments after the handle.
const ACCESSORS: [(&str, &str, &str); 18] = [
    ("vector_at", "base", ", cycle"),
    ("vector_count", "base", ""),
    ("vector_dim", "base", ""),
    ("query_vector_at", "query", ", cycle"),
    ("query_count", "query", ""),
    ("neighbor_indices_at", "neighbor_indices", ", cycle"),
    ("neighbor_count", "neighbor_indices", ""),
    ("neighbor_distances_at", "neighbor_distances", ", cycle"),
    (
        "filtered_neighbor_indices_at",
        "filtered_neighbor_indices",
        ", cycle",
    ),
    (
        "filtered_neighbor_distances_at",
        "filtered_neighbor_distances",
        ", cycle",
    ),
    ("metadata_results_at", "metadata_results", ", cycle"),
    ("metadata_results_len_at", "metadata_results", ", cycle"),
    ("metadata_results_count", "metadata_results", ""),
    ("metadata_value_at", "metadata_content", ", cycle"),
    ("metadata_content_count", "metadata_content", ""),
    ("metadata_count_of", "metadata_content", ", to_i64(cycle)"),
    ("predicate_value_at", "metadata_predicates", ", cycle"),
    (
        "predicate_count_of",
        "metadata_predicates",
        ", to_i64(cycle)",
    ),
];

/// Every group accessor: its name and the arguments after the handle.
const GROUP_ACCESSORS: [(&str, &str); 10] = [
    ("dataset_distance_function", ""),
    ("dataset_facets", ""),
    ("dataset_profile_count", ""),
    ("dataset_profile_names", ""),
    ("matching_profiles", ", \"\""),
    ("dataset_profile_name_at", ", cycle"),
    ("profile_base_count", ", cycle"),
    ("profile_facets", ", cycle"),
    ("profile_partitions", ", \"*\""),
    ("matching_profile_name_at", ", \"*\", cycle"),
];

/// Every registered node that resolves a source string to a facet is
/// in [`ACCESSORS`] with that facet, the facet is one the fixture
/// declares, and every node that resolves one to the dataset group is
/// in [`GROUP_ACCESSORS`], so the parity check below covers an
/// accessor as soon as it is registered.
#[test]
fn every_dataset_accessor_is_checked() {
    for sig in polydat::dsl::registry::registry() {
        match sig.default_resolver {
            Some(DefaultResolver::Facet(facet)) => {
                assert!(
                    FIXTURE_FACETS.contains(&facet),
                    "{} resolves the {facet} facet, which the fixture lacks",
                    sig.name
                );
                assert!(
                    ACCESSORS
                        .iter()
                        .any(|(name, f, _)| *name == sig.name && *f == facet),
                    "{} resolves the {facet} facet and is not in ACCESSORS",
                    sig.name
                );
            }
            Some(DefaultResolver::Group) => assert!(
                GROUP_ACCESSORS.iter().any(|(name, _)| *name == sig.name),
                "{} resolves the dataset group and is not in GROUP_ACCESSORS",
                sig.name
            ),
            _ => {}
        }
    }
}

/// Every accessor answers the same on a prebuffered handle as on the
/// handle `dataset_open` gives for its own facet, or for a group
/// accessor the handle `dataset_group_open` gives, on every engine,
/// with the handles resolved in the program and passed in as externs,
/// and every engine gives the interpreter's answer. A prebuffered
/// handle names the dataset rather than a facet, and each accessor
/// resolves it to the facet or the group it reads.
#[test]
fn every_accessor_agrees_on_a_prebuffered_and_an_opened_handle() {
    let facet_handle = |facet: &str| format!("f_{facet}");
    let calls = ACCESSORS
        .iter()
        .map(|(name, facet, rest)| (*name, facet_handle(facet), *rest))
        .chain(
            GROUP_ACCESSORS
                .iter()
                .map(|(name, rest)| (*name, "group".to_string(), *rest)),
        );
    let mut body = String::new();
    let mut outputs: Vec<(String, String)> = Vec::new();
    for (name, handle, rest) in calls {
        let (pre, open) = (format!("pre_{name}"), format!("open_{name}"));
        body.push_str(&format!("{pre} := {name}(pre{rest})\n"));
        body.push_str(&format!("{open} := {name}({handle}{rest})\n"));
        outputs.push((pre, open));
    }
    let handles: Vec<(String, String)> = [
        (
            "pre".to_string(),
            format!("dataset_prebuffer(\"{SOURCE}\")"),
        ),
        (
            "group".to_string(),
            format!("dataset_group_open(\"{SOURCE}\")"),
        ),
    ]
    .into_iter()
    .chain(FIXTURE_FACETS.iter().map(|facet| {
        (
            facet_handle(facet),
            format!("dataset_open(\"{SOURCE}\", \"{facet}\")"),
        )
    }))
    .collect();

    let in_program: String = handles
        .iter()
        .map(|(name, call)| format!("{name} := {call}\n"))
        .collect();
    let as_externs: String = handles
        .iter()
        .map(|(name, _)| format!("extern {name}: handle\n"))
        .collect();
    let refs: Vec<(&str, String)> = handles
        .iter()
        .map(|(name, call)| (name.as_str(), call.clone()))
        .collect();
    let externs = opened(&refs);

    use_fixture_catalog();
    for (decls, externs) in [(in_program, Vec::new()), (as_externs, externs)] {
        let src = format!("input cycle: u64\n{decls}{body}");
        let mut oracle = kernel_on(Engine::Interpreter(JitMode::Off), &src, &externs);
        let expected: Vec<Vec<String>> = (0..8)
            .map(|c| {
                oracle.set_inputs(&[c]);
                outputs
                    .iter()
                    .map(|(_, open)| oracle.pull(open).to_display_string())
                    .collect()
            })
            .collect();
        for engine in engines() {
            let mut k = kernel_on(engine, &src, &externs);
            for (c, answers) in expected.iter().enumerate() {
                k.set_inputs(&[c as u64]);
                for ((pre, open), answer) in outputs.iter().zip(answers) {
                    assert_eq!(
                        &k.pull(open).to_display_string(),
                        answer,
                        "{engine}: {open} at cycle {c}\n{src}"
                    );
                    assert_eq!(
                        &k.pull(pre).to_display_string(),
                        answer,
                        "{engine}: {pre} at cycle {c}\n{src}"
                    );
                }
            }
        }
    }
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

/// The ground-truth and metadata-results answers, by output name.
fn ground_truth_answer(out: &str, c: u64) -> String {
    let q = c as usize % PREDICATES.len();
    let f32s = |v: &[f32]| Value::VecF32(SliceArc::from_vec(v.to_vec())).to_display_string();
    let i32s = |v: &[i32]| Value::VecI32(SliceArc::from_vec(v.to_vec())).to_display_string();
    match out {
        "nn" => i32s(&NEIGHBORS[q]),
        "nd" => f32s(&DISTANCES[q]),
        "k" => NEIGHBORS[0].len().to_string(),
        "fnn" => i32s(&FILTERED_NEIGHBORS[q]),
        "fnd" => f32s(&FILTERED_DISTANCES[q]),
        "results" => i32s(METADATA_RESULTS[q]),
        "results_len" => METADATA_RESULTS[q].len().to_string(),
        "results_n" => METADATA_RESULTS.len().to_string(),
        other => panic!("no answer for {other}"),
    }
}

const GROUND_TRUTH_OUTPUTS: [&str; 8] = [
    "nn",
    "nd",
    "k",
    "fnn",
    "fnd",
    "results",
    "results_len",
    "results_n",
];

/// The ground truth and the metadata results against the constants,
/// through source strings and through handle externs. The query index
/// wraps modulo the record count.
#[test]
fn ground_truth_accessors_read_the_fixture() {
    let src = format!(
        "input cycle: u64\n\
         nn := neighbor_indices_at(\"{SOURCE}\", cycle)\n\
         nd := neighbor_distances_at(\"{SOURCE}\", cycle)\n\
         k := neighbor_count(\"{SOURCE}\")\n\
         fnn := filtered_neighbor_indices_at(\"{SOURCE}\", cycle)\n\
         fnd := filtered_neighbor_distances_at(\"{SOURCE}\", cycle)\n\
         results := metadata_results_at(\"{SOURCE}\", cycle)\n\
         results_len := metadata_results_len_at(\"{SOURCE}\", cycle)\n\
         results_n := metadata_results_count(\"{SOURCE}\")\n"
    );
    check_on_every_engine(&src, &[], &GROUND_TRUTH_OUTPUTS, 6, ground_truth_answer);
    let externs = opened(&[
        (
            "ni",
            format!("dataset_open(\"{SOURCE}\", \"neighbor_indices\")"),
        ),
        (
            "nd_h",
            format!("dataset_open(\"{SOURCE}\", \"neighbor_distances\")"),
        ),
        (
            "fni",
            format!("dataset_open(\"{SOURCE}\", \"filtered_neighbor_indices\")"),
        ),
        (
            "fnd_h",
            format!("dataset_open(\"{SOURCE}\", \"filtered_neighbor_distances\")"),
        ),
        (
            "mr",
            format!("dataset_open(\"{SOURCE}\", \"metadata_results\")"),
        ),
    ]);
    let src = "input cycle: u64\n\
               extern ni: handle\n\
               extern nd_h: handle\n\
               extern fni: handle\n\
               extern fnd_h: handle\n\
               extern mr: handle\n\
               nn := neighbor_indices_at(ni, cycle)\n\
               nd := neighbor_distances_at(nd_h, cycle)\n\
               k := neighbor_count(ni)\n\
               fnn := filtered_neighbor_indices_at(fni, cycle)\n\
               fnd := filtered_neighbor_distances_at(fnd_h, cycle)\n\
               results := metadata_results_at(mr, cycle)\n\
               results_len := metadata_results_len_at(mr, cycle)\n\
               results_n := metadata_results_count(mr)\n";
    check_on_every_engine(src, &externs, &GROUND_TRUTH_OUTPUTS, 6, ground_truth_answer);
}

/// The group answers the fixture's one profile determines, by output
/// name.
fn group_answer(out: &str, _c: u64) -> String {
    match out {
        "distance" => "EUCLIDEAN".to_string(),
        "names" | "matching" | "name_at" | "tier_name" => "default".to_string(),
        "profile_n" | "tiers" => "1".to_string(),
        "base_n" | "tier_end" => METADATA.len().to_string(),
        other => panic!("no answer for {other}"),
    }
}

const GROUP_OUTPUTS: [&str; 9] = [
    "distance",
    "names",
    "matching",
    "name_at",
    "tier_name",
    "profile_n",
    "base_n",
    "tiers",
    "tier_end",
];

/// The group accessors against the fixture's one profile, through a
/// source string, a `dataset_group_open` handle, and a prebuffered
/// handle.
#[test]
fn group_accessors_read_the_fixture() {
    for group in [
        format!("\"{SOURCE}\""),
        format!("dataset_group_open(\"{SOURCE}\")"),
        format!("dataset_prebuffer(\"{SOURCE}\")"),
    ] {
        let src = format!(
            "input cycle: u64\n\
             g := {group}\n\
             distance := dataset_distance_function(g)\n\
             names := dataset_profile_names(g)\n\
             matching := matching_profiles(g, \"\")\n\
             name_at := dataset_profile_name_at(g, cycle)\n\
             tier_name := matching_profile_name_at(g, \"*\", cycle)\n\
             profile_n := dataset_profile_count(g)\n\
             base_n := profile_base_count(g, cycle)\n\
             tiers := partition_count(profile_partitions(g, \"*\"))\n\
             tier_end := end_of(partition_at(profile_partitions(g, \"*\"), 0))\n"
        );
        check_on_every_engine(&src, &[], &GROUP_OUTPUTS, 3, group_answer);
    }
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
