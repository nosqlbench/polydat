// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! One program per registered public node, shared by the coverage test
//! (every node compiles on the interpreter) and the engine parity test
//! (which engines accept it). A node with an explicit case here uses
//! it; every other node gets a call synthesized from its signature.

use std::collections::HashMap;

/// Fixture files the file-reading nodes need, created under the temp
/// directory: `(csv, jsonl, txt)` paths. Written once per process,
/// under names that carry the process id: test binaries run
/// concurrently under nextest, and a shared name would let one
/// process truncate a file while another reads it.
pub fn fixtures() -> (String, String, String) {
    static FIXTURES: std::sync::OnceLock<(String, String, String)> = std::sync::OnceLock::new();
    FIXTURES
        .get_or_init(|| {
            let pid = std::process::id();
            let path = |ext: &str| {
                std::env::temp_dir().join(format!("_polydat_coverage_test_{pid}.{ext}"))
            };
            let csv_path = path("csv");
            std::fs::write(&csv_path, "name,age\nalice,30\nbob,25\n").unwrap();
            let jsonl_path = path("jsonl");
            std::fs::write(&jsonl_path, "{\"name\":\"alice\"}\n{\"name\":\"bob\"}\n").unwrap();
            let txt_path = path("txt");
            std::fs::write(&txt_path, "hello\nworld\n").unwrap();
            (
                csv_path.to_str().unwrap().to_string(),
                jsonl_path.to_str().unwrap().to_string(),
                txt_path.to_str().unwrap().to_string(),
            )
        })
        .clone()
}

/// The explicit cases: nodes whose synthesized call would not type or
/// would need a file.
#[allow(clippy::too_many_lines)]
pub fn overrides(csv: &str, jsonl: &str, txt: &str) -> HashMap<&'static str, String> {
    let mut overrides: std::collections::HashMap<&str, String> = [
        // Bytes input
        ("to_hex", "input cycle: u64\nb := u64_to_bytes(cycle)\nout := to_hex(b)".into()),
        ("from_hex", "input cycle: u64\nb := u64_to_bytes(cycle)\nh := to_hex(b)\nout := from_hex(h)".into()),
        ("sha256", "input cycle: u64\nb := u64_to_bytes(cycle)\nout := sha256(b)".into()),
        ("md5", "input cycle: u64\nb := u64_to_bytes(cycle)\nout := md5(b)".into()),
        ("to_base64", "input cycle: u64\nb := u64_to_bytes(cycle)\nout := to_base64(b)".into()),
        ("from_base64", "input cycle: u64\nb := u64_to_bytes(cycle)\ne := to_base64(b)\nout := from_base64(e)".into()),
        // JSON input
        ("json_to_str", "input cycle: u64\nj := to_json(cycle)\nout := json_to_str(j)".into()),
        ("json_merge", "input cycle: u64\na := to_json(cycle)\nb := to_json(cycle)\nout := json_merge(a, b)".into()),
        ("escape_json", "input cycle: u64\ns := format_u64(cycle, 10)\nout := escape_json(s)".into()),
        // Distributions
        ("dist_normal", "input cycle: u64\nout := dist_normal(hash(cycle), 0.0, 1.0)".into()),
        ("dist_exponential", "input cycle: u64\nout := dist_exponential(hash(cycle), 1.0)".into()),
        ("dist_uniform", "input cycle: u64\nout := dist_uniform(hash(cycle), 0.0, 1.0)".into()),
        ("dist_pareto", "input cycle: u64\nout := dist_pareto(hash(cycle), 1.0, 1.0)".into()),
        ("dist_zipf", "input cycle: u64\nout := dist_zipf(hash(cycle), 100, 1.0)".into()),
        ("histribution", "input cycle: u64\nout := histribution(hash(cycle), \"50 25 13 12\")".into()),
        ("dist_empirical", "input cycle: u64\nf := unit_interval(hash(cycle))\nout := dist_empirical(f, \"1.0 3.0 5.0 7.0 9.0\")".into()),
        // Weighted
        ("weighted_strings", "input cycle: u64\nout := weighted_strings(hash(cycle), \"a:0.5;b:0.5\")".into()),
        ("weighted_u64", "input cycle: u64\nout := weighted_u64(hash(cycle), \"10:0.5;20:0.5\")".into()),
        ("weighted_pick", "input cycle: u64\nout := weighted_pick(hash(cycle), \"10:0.5;20:0.5\")".into()),
        ("one_of_weighted", "input cycle: u64\nout := one_of_weighted(hash(cycle), \"a:0.5;b:0.5\")".into()),
        // String input
        ("html_encode", "input cycle: u64\ns := format_u64(cycle, 10)\nout := html_encode(s)".into()),
        ("html_decode", "input cycle: u64\ns := format_u64(cycle, 10)\nout := html_decode(s)".into()),
        ("url_encode", "input cycle: u64\ns := format_u64(cycle, 10)\nout := url_encode(s)".into()),
        ("url_decode", "input cycle: u64\ns := format_u64(cycle, 10)\nout := url_decode(s)".into()),
        ("regex_replace", "input cycle: u64\ns := format_u64(cycle, 10)\nout := regex_replace(s, \"[0-9]\", \"x\")".into()),
        ("regex_match", "input cycle: u64\ns := format_u64(cycle, 10)\nout := regex_match(s, \"[0-9]+\")".into()),
        // Multi-input
        ("select", "input cycle: u64\nout := select(fair_coin(hash(cycle)), cycle, cycle)".into()),
        ("blend", "input cycle: u64\nout := blend(hash(cycle), hash(cycle), 0.5)".into()),
        ("date_components", "input cycle: u64\n(y, mo, d, h, mi, s, ms) := date_components(cycle)".into()),
        ("perlin_2d", "input cycle: u64\nout := perlin_2d(cycle, cycle, 42, 0.01)".into()),
        ("simplex_2d", "input cycle: u64\nout := simplex_2d(cycle, cycle, 42, 0.01)".into()),
        ("fractal_noise_2d", "input cycle: u64\nout := fractal_noise_2d(cycle, cycle, 42, 0.02)".into()),
        ("pcg_stream", "input cycle: u64\nout := pcg_stream(cycle, cycle, 42)".into()),
        ("format_u64", "input cycle: u64\nout := format_u64(cycle, 16)".into()),
        // Context (no inputs)
        ("current_epoch_millis", "input cycle: u64\nout := current_epoch_millis()".into()),
        ("counter", "input cycle: u64\nout := counter()".into()),
        // Random nodes — auto-gen would feed both min/max the same
        // value (100, 100), causing range=0 division panics.
        ("random_range", "input cycle: u64\nout := random_range(0, 1000)".into()),
        ("random_f64",   "input cycle: u64\nout := random_f64(0.0, 1.0)".into()),
        ("session_start_millis", "input cycle: u64\nout := session_start_millis()".into()),
        ("elapsed_millis", "input cycle: u64\nout := elapsed_millis()".into()),
        ("thread_id", "input cycle: u64\nout := thread_id()".into()),
        // f64 input
        ("clamp_f64", "input cycle: u64\nf := unit_interval(hash(cycle))\nout := clamp_f64(f, 0.0, 0.5)".into()),
        ("quantize", "input cycle: u64\nf := unit_interval(hash(cycle))\nout := quantize(f, 0.1)".into()),
        ("lerp", "input cycle: u64\nf := unit_interval(hash(cycle))\nout := lerp(f, 0.0, 100.0)".into()),
        ("inv_lerp", "input cycle: u64\nf := unit_interval(hash(cycle))\nout := inv_lerp(f, 0.0, 1.0)".into()),
        ("remap", "input cycle: u64\nf := unit_interval(hash(cycle))\nout := remap(f, 0.0, 1.0, 0.0, 100.0)".into()),
        // FFT (creates output file)
        ("fft_analyze", "input cycle: u64\nf := unit_interval(hash(cycle))\nout := fft_analyze(f, \"/tmp/_polydat_fft_test.jsonl\", 8)".into()),
        // `env(name)` errors if the named var isn't set —
        // use `PATH` which is universally present in test
        // environments. The auto-generated `env("test")`
        // would otherwise fail unless someone happens to
        // export TEST.
        ("env", "input cycle: u64\nout := env(\"PATH\")".into()),
        // body_column_i32 needs a Json input — the auto-generator
        // wires `cycle` (u64) which trips the type adapter. Override
        // with `to_json(cycle)` so the signature smoke compiles.
        ("body_column_i32",
            "input cycle: u64\nout := body_column_i32(to_json(cycle), \"key\")".into()),
        // SRD 71: partition-typed inputs come from a cursor's
        // `.cursor` projection — declare a small cursor and
        // pull from there. The cursor's `over` clause takes a
        // string-literal spec (parsed at phase setup); the
        // generic `cycle` input slot is the wrong type.
        ("cardinality",
            "input cycle: u64\ncursor q = range(0, 100) over \"0..50%\"\nout := cardinality(q.cursor)".into()),
        ("start_of",
            "input cycle: u64\ncursor q = range(0, 100) over \"0..50%\"\nout := start_of(q.cursor)".into()),
        ("end_of",
            "input cycle: u64\ncursor q = range(0, 100) over \"0..50%\"\nout := end_of(q.cursor)".into()),
        ("idx_of",
            "input cycle: u64\ncursor q = range(0, 100) over \"0..50%\"\nout := idx_of(q.cursor)".into()),
        ("mod_in",
            "input cycle: u64\ncursor q = range(0, 100) over \"0..50%\"\nout := mod_in(cycle, q.cursor)".into()),
        ("at",
            "input cycle: u64\ncursor q = range(0, 100) over \"0..50%\"\nout := at(q.cursor, cycle)".into()),
        ("clamp_in",
            "input cycle: u64\ncursor q = range(0, 100) over \"0..50%\"\nout := clamp_in(cycle, q.cursor)".into()),
        ("count_of",
            "input cycle: u64\ncursor q = range(0, 100) over \"0..50%\"\nout := count_of(q.cursor)".into()),
        ("random_in",
            "input cycle: u64\ncursor q = range(0, 100) over \"0..50%\"\nout := random_in(q.cursor, cycle)".into()),
        ("subdivide",
            "input cycle: u64\ncursor q = range(0, 100) over \"0..50%\"\nout := subdivide(q.cursor, 2)".into()),
        ("partitions",
            "input cycle: u64\nout := partitions(\"linear:4\", 1000)".into()),
        ("partition_count",
            "input cycle: u64\nout := partition_count(partitions(\"linear:4\", 1000))".into()),
        ("partition_at",
            "input cycle: u64\nout := partition_at(partitions(\"linear:4\", 1000), u64_mod(cycle, 4))".into()),
        // Vector-math nodes need vec_f32 operands — hash_vec is
        // the workload-callable synthetic generator.
        ("vec_add",
            "input cycle: u64\nout := vec_add(hash_vec(cycle, 8), hash_vec(cycle + 1, 8))".into()),
        ("vec_dot",
            "input cycle: u64\nout := vec_dot(hash_vec(cycle, 8), hash_vec(cycle + 1, 8))".into()),
        ("vec_l2",
            "input cycle: u64\nout := vec_l2(hash_vec(cycle, 8), hash_vec(cycle + 1, 8))".into()),
        ("vec_cosine",
            "input cycle: u64\nout := vec_cosine(hash_vec(cycle, 8), hash_vec(cycle + 1, 8))".into()),
        ("vec_scale",
            "input cycle: u64\nout := vec_scale(hash_vec(cycle, 8), 2.0)".into()),
        ("vec_norm",
            "input cycle: u64\nout := vec_norm(hash_vec(cycle, 8))".into()),
        ("lid_mle",
            "input cycle: u64\nout := lid_mle(hash_vec(cycle, 8), 4.0)".into()),
        // Register-plane nodes need reg operands — splats are the
        // workload-callable constructors.
        ("reg_gather_f32",
            "input cycle: u64\nout := reg_gather_f32(hash_vec(cycle, 8), 0)".into()),
        ("vec_to_reg_f32",
            "input cycle: u64\nout := vec_to_reg_f32(hash_vec(cycle, 4))".into()),
        ("reg_to_vec_f32",
            "input cycle: u64\nout := reg_to_vec_f32(reg_splat_f32(cycle))".into()),
        ("reg_lane_f32",
            "input cycle: u64\nout := reg_lane_f32(reg_splat_f32(cycle), 0)".into()),
        ("reg_with_lane_f32",
            "input cycle: u64\nout := reg_with_lane_f32(reg_splat_f32(cycle), 0, 1.5)".into()),
        ("reg_lane_i16",
            "input cycle: u64\nout := reg_lane_i16(reg_splat_i16(cycle), 0)".into()),
        ("reg_lane_i64",
            "input cycle: u64\nout := reg_lane_i64(reg_splat_i64(cycle), 0)".into()),
        ("reg_add_f32",
            "input cycle: u64\nout := reg_add_f32(reg_splat_f32(cycle), reg_splat_f32(cycle))".into()),
        ("reg_sub_f32",
            "input cycle: u64\nout := reg_sub_f32(reg_splat_f32(cycle), reg_splat_f32(cycle))".into()),
        ("reg_mul_f32",
            "input cycle: u64\nout := reg_mul_f32(reg_splat_f32(cycle), reg_splat_f32(cycle))".into()),
        ("reg_add_f64",
            "input cycle: u64\nout := reg_add_f64(reg_splat_f64(cycle), reg_splat_f64(cycle))".into()),
        ("reg_sub_f64",
            "input cycle: u64\nout := reg_sub_f64(reg_splat_f64(cycle), reg_splat_f64(cycle))".into()),
        ("reg_mul_f64",
            "input cycle: u64\nout := reg_mul_f64(reg_splat_f64(cycle), reg_splat_f64(cycle))".into()),
        ("reg_add_i8",
            "input cycle: u64\nout := reg_add_i8(reg_splat_i8(cycle), reg_splat_i8(cycle))".into()),
        ("reg_sub_i8",
            "input cycle: u64\nout := reg_sub_i8(reg_splat_i8(cycle), reg_splat_i8(cycle))".into()),
        ("reg_mul_i8",
            "input cycle: u64\nout := reg_mul_i8(reg_splat_i8(cycle), reg_splat_i8(cycle))".into()),
        ("reg_add_i16",
            "input cycle: u64\nout := reg_add_i16(reg_splat_i16(cycle), reg_splat_i16(cycle))".into()),
        ("reg_sub_i16",
            "input cycle: u64\nout := reg_sub_i16(reg_splat_i16(cycle), reg_splat_i16(cycle))".into()),
        ("reg_mul_i16",
            "input cycle: u64\nout := reg_mul_i16(reg_splat_i16(cycle), reg_splat_i16(cycle))".into()),
        ("reg_add_i32",
            "input cycle: u64\nout := reg_add_i32(reg_splat_i32(cycle), reg_splat_i32(cycle))".into()),
        ("reg_sub_i32",
            "input cycle: u64\nout := reg_sub_i32(reg_splat_i32(cycle), reg_splat_i32(cycle))".into()),
        ("reg_mul_i32",
            "input cycle: u64\nout := reg_mul_i32(reg_splat_i32(cycle), reg_splat_i32(cycle))".into()),
        ("reg_add_i64",
            "input cycle: u64\nout := reg_add_i64(reg_splat_i64(cycle), reg_splat_i64(cycle))".into()),
        ("reg_sub_i64",
            "input cycle: u64\nout := reg_sub_i64(reg_splat_i64(cycle), reg_splat_i64(cycle))".into()),
        ("reg_mul_i64",
            "input cycle: u64\nout := reg_mul_i64(reg_splat_i64(cycle), reg_splat_i64(cycle))".into()),
        ("reg_dot_f32",
            "input cycle: u64\nout := reg_dot_f32(reg_splat_f32(cycle), reg_splat_f32(cycle + 1))".into()),
        ("reg_shuffle_bytes",
            "input cycle: u64\nout := reg_shuffle_bytes(reg_splat_i8(cycle), 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)".into()),
    ].into_iter().collect();

    // File I/O nodes — use real fixture files
    overrides.insert(
        "csv_field",
        format!("input cycle: u64\nout := csv_field(cycle, \"{csv}\", \"name\")"),
    );
    overrides.insert(
        "csv_row",
        format!("input cycle: u64\nout := csv_row(cycle, \"{csv}\")"),
    );
    overrides.insert(
        "csv_row_count",
        format!("input cycle: u64\nout := csv_row_count(\"{csv}\")"),
    );
    overrides.insert(
        "jsonl_field",
        format!("input cycle: u64\nout := jsonl_field(cycle, \"{jsonl}\", \"name\")"),
    );
    overrides.insert(
        "jsonl_row",
        format!("input cycle: u64\nout := jsonl_row(cycle, \"{jsonl}\")"),
    );
    overrides.insert(
        "jsonl_row_count",
        format!("input cycle: u64\nout := jsonl_row_count(\"{jsonl}\")"),
    );
    overrides.insert(
        "file_line_at",
        format!("input cycle: u64\nout := file_line_at(cycle, \"{txt}\")"),
    );

    // SRD-66 `pick(b0..bN-1, v0..vN-1)` — needs Bool
    // selectors and uniform values. The auto-generated
    // single-wire form fails the min_wires=2 check, and a
    // comparison such as `cycle == 0` is a u64 truth value, not
    // a Bool wire. The two regex selectors are complementary
    // (the digits contain a 3, or contain no 3), so exactly one
    // is true for every cycle.
    overrides.insert(
        "pick",
        "input cycle: u64\ns := format_u64(cycle, 10)\nout := pick(regex_match(s, \"3\"), regex_match(s, \"^[^3]*$\"), 100, 200)".into(),
    );
    // Assertion nodes pass their input through only when the
    // precondition holds; the generic argument shapes (a `[100, 100]`
    // range, the allowed set `{100}`, the pattern `test` against a
    // digit string, a `test` weight spec) fail it for every cycle, so
    // each gets a program the interpreter can actually run.
    overrides.insert(
        "in_range",
        "input cycle: u64\nout := in_range(cycle, 0, 1000)".into(),
    );
    overrides.insert(
        "is_one_of",
        "input cycle: u64\nout := is_one_of(cycle, 0, 1, 2, 3)".into(),
    );
    overrides.insert(
        "matches",
        "input cycle: u64\ns := format_u64(cycle, 10)\nout := matches(s, \"[0-9]+\")".into(),
    );
    overrides.insert(
        "dynamic_weighted_select",
        "input cycle: u64\nout := dynamic_weighted_select(cycle, \"alpha:0.3;beta:0.5;gamma:0.2\")"
            .into(),
    );
    // SRD 113: `streamer` takes comprehension text (or the compiler's
    // JSON payload); the generic string example is neither.
    overrides.insert(
        "streamer",
        "input cycle: u64\nout := streamer(\"k in 1..4, limit in 10,20,30\")".into(),
    );
    // SRD 114: `tile_render` takes a compiled skeleton the compiler
    // emits for a `tile` statement; exercise it through one.
    overrides.insert(
        "tile_render",
        "input cycle: u64\ntile out : text := \"n=${cycle}\"".into(),
    );
    overrides.insert(
        "tile_encode",
        "input cycle: u64\nout := tile_encode(cycle, \"json|value|u64||\")".into(),
    );
    overrides
}

/// Every registered public node with the program that exercises it:
/// `(name, source)`. Adapters (`__*`) and dataset-backed nodes
/// (`RealData`) are excluded, as the coverage test excludes them.
pub fn programs() -> Vec<(String, String)> {
    programs_over("input cycle: u64", "cycle", true)
}

/// The call synthesized from a node's signature with `wire` on every
/// wire parameter, or `None` for a node whose call is written by hand
/// (an override).
pub fn synthesized_call(sig: &polydat::dsl::registry::FuncSig, wire: &str) -> Option<String> {
    use polydat::ast::SlotType;
    let (csv, jsonl, txt) = fixtures();
    if overrides(&csv, &jsonl, &txt).contains_key(sig.name) {
        return None;
    }
    let mut args: Vec<String> = Vec::new();
    for p in sig.params {
        match p.slot_type {
            SlotType::Wire => args.push(wire.into()),
            SlotType::ConstU64 => args.push("100".into()),
            SlotType::ConstF64 => args.push("1.0".into()),
            SlotType::ConstStr => args.push("\"test\"".into()),
            SlotType::ConstVecU64 => args.push("100".into()),
            SlotType::ConstVecF64 => args.push("1.0".into()),
            SlotType::ConstVec => args.push("100".into()),
        }
    }
    if args.is_empty() && sig.is_variadic() {
        args.push(wire.into());
    }
    Some(format!("{}({})", sig.name, args.join(", ")))
}

/// The programs over a declared input: `decl` declares it, `wire`
/// names it in every synthesized call, and the hand-written cases
/// (which are over `cycle`) are included only when asked.
pub fn programs_over(decl: &str, wire: &str, with_overrides: bool) -> Vec<(String, String)> {
    use polydat::dsl::registry;
    let (csv, jsonl, txt) = fixtures();
    let overrides = overrides(&csv, &jsonl, &txt);
    let mut out = Vec::new();
    for sig in &registry::registry() {
        if sig.category == registry::FuncCategory::RealData || sig.name.starts_with("__") {
            continue;
        }
        let src = match (overrides.get(sig.name), synthesized_call(sig, wire)) {
            (Some(s), _) if with_overrides => s.clone(),
            (Some(_), _) => continue,
            (None, Some(call)) => format!("{decl}\nout := {call}"),
            (None, None) => continue,
        };
        out.push((sig.name.to_string(), src));
    }
    // By name, whatever order the modules linked in, so the pinned
    // matrix does not move when a module is added.
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}
