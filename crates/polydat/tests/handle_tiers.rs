// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The P1↔P2↔P3 differential suite for by-reference values (strings,
//! byte strings, JSON, extension values, handles: the `Ref2` kinds of
//! jit_boundary.md). A grammar-directed generator emits random programs over the
//! string, JSON, tile, partition, and streamer nodes with typed wires,
//! extension values, a fallible construction, a mixed tuple return
//! destructured through the compiler's copy steps, explicit copies of
//! table kinds, externs with defaults and externs the harness sets on
//! every kernel (a JSON document and a partition), casts, formats,
//! declared hole types, raw holes, splices, branches, projections
//! (nested, in string position, and empty), and structural templates;
//! every program compiles on the interpreter (the oracle), with forced
//! cone extraction, as a P2 closure kernel, as a hybrid kernel, and
//! where every node lowers as a pure-P3 kernel, and every output agrees
//! across the tiers by type and text for a run of coordinates. The P2
//! push-pull kernel is driven over a repeating coordinate sequence as
//! well, so a reference output that stays current across repeated
//! writes is under the same oracle. The P2 and hybrid kernels run the
//! S9 reference validator after every run in these debug builds.
//!
//! `FUZZ_SEED` and `FUZZ_ITERATIONS` follow the other fuzzers.

#![cfg(feature = "jit")]

use polydat::JitMode;
use polydat::ast::{PortType, Value};
use polydat::dsl::compile::compile_polydat_to_assembler;
use polydat::kernel::PolydatKernel;

// Two shapes the library has no instance of with a handle element: a
// fallible construction whose cached value is a string, and a tuple
// return mixing a carrier, a string, and JSON. Both ride the closure
// kits through their cached-value and per-element writes.

/// A fallible node: the body runs once at construction and its value
/// is replayed every cycle.
#[polydat::polydat_node(category = String)]
fn fixed_prefix(prefix: polydat::derive_support::Const<&str>) -> Result<String, String> {
    if prefix.is_empty() {
        Err("fixed_prefix: empty".into())
    } else {
        Ok(format!("{}#", *prefix))
    }
}

/// A tuple return with a carrier, a string, and a JSON element.
#[polydat::polydat_node(category = Math, output_names(low, text, doc))]
fn triple_of(n: u64) -> (u64, String, std::sync::Arc<serde_json::Value>) {
    (
        n % 1000,
        format!("t{}", n % 97),
        std::sync::Arc::new(serde_json::json!({ "n": n % 1000, "odd": n % 2 == 1 })),
    )
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() as usize) % n
        }
    }
    fn coin(&mut self, pct: usize) -> bool {
        self.range(100) < pct
    }
}

/// A wire the generator has bound, with the type it carries and
/// whether its text is a decimal number (so it may be parsed or cast).
#[derive(Clone)]
struct Wire {
    name: String,
    ty: PortType,
    numeric_text: bool,
    /// A `json` tile: its bare name in a hole of another `json` tile is
    /// a splice, so it may only fill a value position there.
    json_tile: bool,
    /// An extension value's runtime type, for wires the consumers of
    /// that type may take: `Partition`, `PartitionList`, or `Streamer`.
    ext: Option<&'static str>,
    /// How many times a partition has been subdivided, which bounds
    /// its cardinality from below so the next subdivision is legal.
    depth: u8,
}

struct Gen {
    rng: Rng,
    wires: Vec<Wire>,
    /// Names of `json` tiles, for splicing.
    json_tiles: Vec<String>,
    lines: Vec<String>,
    outputs: Vec<String>,
    /// Externs without a default, with the value the harness sets on
    /// every kernel before it runs.
    host_externs: Vec<(String, Value)>,
    next: usize,
}

impl Gen {
    fn new(seed: u64) -> Self {
        Gen {
            rng: Rng::new(seed),
            wires: vec![Wire {
                name: "cycle".into(),
                ty: PortType::U64,
                numeric_text: false,
                json_tile: false,
                ext: None,
                depth: 0,
            }],
            json_tiles: Vec::new(),
            lines: vec!["input cycle: u64".into()],
            outputs: Vec::new(),
            host_externs: Vec::new(),
            next: 0,
        }
    }

    /// An extern declaration goes before the bindings; its wire is
    /// usable like any other.
    fn declare_extern(&mut self, ty: PortType, decl: String, ext: Option<&'static str>) -> Wire {
        let name = self.name();
        self.lines.insert(1, format!("extern {name}{decl}"));
        let w = Wire {
            name: name.clone(),
            ty,
            numeric_text: false,
            json_tile: false,
            ext,
            depth: 0,
        };
        self.wires.push(w.clone());
        self.outputs.push(name);
        w
    }

    fn name(&mut self) -> String {
        self.next += 1;
        format!("w{}", self.next)
    }

    fn pick(&mut self, ty: PortType) -> Option<Wire> {
        let candidates: Vec<Wire> = self.wires.iter().filter(|w| w.ty == ty).cloned().collect();
        if candidates.is_empty() {
            None
        } else {
            Some(candidates[self.rng.range(candidates.len())].clone())
        }
    }

    fn numeric_text(&mut self) -> Option<Wire> {
        let candidates: Vec<Wire> = self
            .wires
            .iter()
            .filter(|w| w.numeric_text)
            .cloned()
            .collect();
        if candidates.is_empty() {
            None
        } else {
            Some(candidates[self.rng.range(candidates.len())].clone())
        }
    }

    fn pick_ext(&mut self, kind: &'static str) -> Option<Wire> {
        let candidates: Vec<Wire> = self
            .wires
            .iter()
            .filter(|w| w.ext == Some(kind))
            .cloned()
            .collect();
        if candidates.is_empty() {
            None
        } else {
            Some(candidates[self.rng.range(candidates.len())].clone())
        }
    }

    /// An extension-valued binding: an output like any other, compared
    /// across tiers by type name and display.
    fn bind_ext(&mut self, kind: &'static str, depth: u8, expr: String) -> Wire {
        let name = self.name();
        self.lines.push(format!("{name} := {expr}"));
        let w = Wire {
            name: name.clone(),
            ty: PortType::Ext,
            numeric_text: false,
            json_tile: false,
            ext: Some(kind),
            depth,
        };
        self.wires.push(w.clone());
        self.outputs.push(name);
        w
    }

    fn any(&mut self) -> Wire {
        let i = self.rng.range(self.wires.len());
        self.wires[i].clone()
    }

    fn bind(&mut self, ty: PortType, expr: String) -> Wire {
        self.bind_with(ty, expr, false)
    }

    fn bind_with(&mut self, ty: PortType, expr: String, numeric_text: bool) -> Wire {
        let name = self.name();
        self.lines.push(format!("{name} := {expr}"));
        let w = Wire {
            name: name.clone(),
            ty,
            numeric_text,
            json_tile: false,
            ext: None,
            depth: 0,
        };
        self.wires.push(w.clone());
        self.outputs.push(name);
        w
    }

    /// One binding from the grammar.
    fn step(&mut self) {
        let u = self.pick(PortType::U64).expect("cycle is always u64");
        match self.rng.range(44) {
            0 => {
                self.bind(PortType::U64, format!("hash({})", u.name));
            }
            1 => {
                self.bind(PortType::F64, format!("to_f64({}) / 7.0", u.name));
            }
            2 => {
                self.bind(PortType::Bool, format!("u64_gt({}, 1000)", u.name));
            }
            3 => {
                self.bind_with(PortType::Str, format!("__u64_to_string({})", u.name), true);
            }
            4 => {
                if let Some(s) = self.pick(PortType::Str) {
                    let f = ["str_upper", "str_lower", "escape_json"][self.rng.range(3)];
                    self.bind(PortType::Str, format!("{f}({})", s.name));
                }
            }
            5 => {
                let n = 1 + self.rng.range(4);
                let mut specs = Vec::new();
                let mut args = Vec::new();
                for _ in 0..n {
                    let w = self.any();
                    let spec = match w.ty {
                        PortType::U64 => ["{}", "{:05}", "{:x}", "{:X}", "{:b}", "{:o}", "{:8}"]
                            [self.rng.range(7)],
                        PortType::F64 => ["{}", "{:.2}", "{:.0}", "{:10.3}"][self.rng.range(4)],
                        PortType::Str => ["{}", "{:>12}", "{:3}"][self.rng.range(3)],
                        _ => "{}",
                    };
                    specs.push(spec.to_string());
                    args.push(w.name);
                }
                let fmt = specs.join(["-", " ", "|", "{{}}"][self.rng.range(4)]);
                self.bind(
                    PortType::Str,
                    format!("printf(\"{fmt}\", {})", args.join(", ")),
                );
            }
            6 => {
                self.bind(PortType::Json, format!("__u64_to_json({})", u.name));
            }
            7 => {
                let w = self.any();
                self.bind(PortType::Json, format!("to_json({})", w.name));
            }
            8 => {
                let n = self.rng.range(4);
                let args: Vec<String> = (0..n).map(|_| self.any().name).collect();
                self.bind(PortType::Json, format!("json_array({})", args.join(", ")));
            }
            9 => {
                let n = 1 + self.rng.range(3);
                let parts: Vec<String> = (0..n)
                    .map(|i| {
                        let w = self.any();
                        format!("json_with(\"k{i}\", {})", w.name)
                    })
                    .collect();
                self.bind(PortType::Json, format!("json_object({})", parts.join(", ")));
            }
            10 => {
                if let Some(j) = self.pick(PortType::Json) {
                    let f = ["json_to_str", "json_text", "json_to_str_pretty"][self.rng.range(3)];
                    self.bind(PortType::Str, format!("{f}({})", j.name));
                }
            }
            11 => {
                if let Some(s) = self.pick(PortType::Str) {
                    self.bind(PortType::Json, format!("str_to_json({})", s.name));
                }
            }
            12 => {
                if let Some(s) = self.pick(PortType::Str) {
                    let w = self.any();
                    self.bind(PortType::Str, format!("str_concat({}, {})", s.name, w.name));
                }
            }
            13 => {
                // The typed JSON adapters and their inverse text forms.
                match self.rng.range(4) {
                    0 => {
                        if let Some(f) = self.pick(PortType::F64) {
                            self.bind(PortType::Json, format!("__f64_to_json({})", f.name));
                        }
                    }
                    1 => {
                        if let Some(b) = self.pick(PortType::Bool) {
                            self.bind(PortType::Json, format!("__bool_to_json({})", b.name));
                        }
                    }
                    2 => {
                        if let Some(i) = self.pick(PortType::I64) {
                            self.bind(PortType::Json, format!("__i64_to_json({})", i.name));
                        }
                    }
                    _ => {
                        if let Some(d) = self.numeric_text() {
                            self.bind(PortType::Json, format!("__str_to_json({})", d.name));
                        }
                    }
                }
            }
            14 => {
                // Bounded first: a hash can exceed i64::MAX, which P1 diagnoses.
                self.bind(
                    PortType::I64,
                    format!("__u64_to_i64(mod({}, 1000003))", u.name),
                );
            }
            15 => {
                if let Some(i) = self.pick(PortType::I64) {
                    self.bind_with(PortType::Str, format!("__i64_to_string({})", i.name), true);
                }
            }
            16 => {
                if let Some(f) = self.pick(PortType::F64) {
                    self.bind(PortType::Str, format!("__f64_to_string({})", f.name));
                }
            }
            17 => {
                if let Some(b) = self.pick(PortType::Bool) {
                    self.bind(PortType::Str, format!("__bool_to_str({})", b.name));
                }
            }
            18 => {
                // Parses of decimal text, which every tier must read alike.
                if let Some(d) = self.numeric_text() {
                    match self.rng.range(3) {
                        0 => {
                            self.bind(PortType::U64, format!("__str_to_u64({})", d.name));
                        }
                        1 => {
                            self.bind(PortType::F64, format!("__str_to_f64({})", d.name));
                        }
                        _ => {
                            // Bounded text: a hash rendered as text exceeds i64::MAX,
                            // which P1 diagnoses (as for `__u64_to_i64` above).
                            self.bind(
                                PortType::I64,
                                format!("__str_to_i64(__u64_to_string(mod({}, 1000003)))", u.name),
                            );
                        }
                    }
                }
            }
            19 => {
                if let (Some(a), Some(b)) = (self.pick(PortType::Json), self.pick(PortType::Json)) {
                    self.bind(
                        PortType::Json,
                        format!("json_merge({}, {})", a.name, b.name),
                    );
                }
            }
            20 => {
                // `as` casts reach the whole adapter catalog.
                match self.rng.range(4) {
                    0 => {
                        self.bind_with(PortType::Str, format!("{} as str", u.name), true);
                    }
                    1 => {
                        if let Some(f) = self.pick(PortType::F64) {
                            self.bind(PortType::Str, format!("{} as str", f.name));
                        }
                    }
                    2 => {
                        if let Some(d) = self.numeric_text() {
                            self.bind(PortType::U64, format!("{} as u64", d.name));
                        }
                    }
                    _ => {
                        self.bind(PortType::F64, format!("{} as f64", u.name));
                    }
                }
            }
            21 => {
                if let (Some(a), Some(b)) = (self.pick(PortType::Str), self.pick(PortType::Str)) {
                    self.bind(PortType::U64, format!("str_eq({}, {})", a.name, b.name));
                }
            }
            22 => {
                if let Some(s) = self.pick(PortType::Str) {
                    self.bind(
                        PortType::Str,
                        format!("str_concat({}, \",q\\\"z\")", s.name),
                    );
                }
            }
            23 => self.structural(),
            24 => self.tile(),
            25 => self.tile(),
            26 => {
                // A partition list from a spec, the root of every partition.
                let spec = ["*/4", "20%,30%,*", "*/2", "10%,*"][self.rng.range(4)];
                let extent = [100u64, 500, 1000, 1024][self.rng.range(4)];
                self.bind_ext(
                    "PartitionList",
                    0,
                    format!("partitions(\"{spec}\", {extent})"),
                );
            }
            27 => {
                if let Some(l) = self.pick_ext("PartitionList") {
                    if self.rng.coin(30) {
                        self.bind(PortType::U64, format!("partition_count({})", l.name));
                    } else {
                        self.bind_ext(
                            "Partition",
                            l.depth,
                            format!(
                                "partition_at({}, u64_mod({}, partition_count({})))",
                                l.name, u.name, l.name
                            ),
                        );
                    }
                }
            }
            28 => {
                if let Some(p) = self.pick_ext("Partition") {
                    match self.rng.range(10) {
                        0 => self.bind(PortType::U64, format!("cardinality({})", p.name)),
                        1 => self.bind(PortType::U64, format!("start_of({})", p.name)),
                        2 => self.bind(PortType::U64, format!("end_of({})", p.name)),
                        3 => self.bind(PortType::U64, format!("idx_of({})", p.name)),
                        4 => self.bind(PortType::U64, format!("count_of({})", p.name)),
                        5 => self.bind(PortType::U64, format!("mod_in({}, {})", u.name, p.name)),
                        6 => self.bind(PortType::U64, format!("clamp_in({}, {})", u.name, p.name)),
                        7 => self.bind(PortType::U64, format!("random_in({}, {})", p.name, u.name)),
                        8 => self.bind(
                            PortType::U64,
                            format!(
                                "at({}, u64_mod({}, cardinality({})))",
                                p.name, u.name, p.name
                            ),
                        ),
                        _ => {
                            // Every spec above leaves at least 25 ordinals per
                            // partition, so two halvings are always legal.
                            if p.depth < 2 {
                                self.bind_ext(
                                    "PartitionList",
                                    p.depth + 1,
                                    format!("subdivide({}, 2)", p.name),
                                )
                            } else {
                                self.bind(PortType::U64, format!("cardinality({})", p.name))
                            }
                        }
                    };
                }
            }
            29 => {
                // A fallible construction: a fixed string per program.
                let prefix = ["job", "row", "k"][self.rng.range(3)];
                self.bind(PortType::Str, format!("fixed_prefix(\"{prefix}\")"));
            }
            30 => {
                // A tuple return destructured into three wires, one of
                // them a table kind, so the copy steps the compiler
                // inserts for the elements carry a table handle.
                let low = self.name();
                let text = self.name();
                let doc = self.name();
                self.lines
                    .push(format!("({low}, {text}, {doc}) := triple_of({})", u.name));
                for (name, ty) in [
                    (&low, PortType::U64),
                    (&text, PortType::Str),
                    (&doc, PortType::Json),
                ] {
                    self.wires.push(Wire {
                        name: name.clone(),
                        ty,
                        numeric_text: false,
                        json_tile: false,
                        ext: None,
                        depth: 0,
                    });
                    self.outputs.push(name.clone());
                }
            }
            31 => {
                // An explicit copy of a table-kind wire.
                let candidates: Vec<Wire> = self
                    .wires
                    .iter()
                    .filter(|w| w.ty == PortType::Json || w.ty == PortType::Ext)
                    .cloned()
                    .collect();
                if !candidates.is_empty() {
                    let w = candidates[self.rng.range(candidates.len())].clone();
                    let name = self.name();
                    self.lines.push(format!("{name} := identity({})", w.name));
                    self.wires.push(Wire {
                        name: name.clone(),
                        ty: w.ty,
                        numeric_text: false,
                        json_tile: false,
                        ext: w.ext,
                        depth: w.depth,
                    });
                    self.outputs.push(name);
                }
            }
            32 => {
                // An extern with a default: a carrier or a string the
                // kernel seeds. Its passthrough is a copy step on every
                // engine.
                if self.rng.coin(50) {
                    let n = [3u64, 42, 1000, 65535][self.rng.range(4)];
                    self.declare_extern(PortType::U64, format!(": u64 = {n}"), None);
                } else {
                    let s = ["us-east", "row", "k,q\\\"z"][self.rng.range(3)];
                    self.declare_extern(PortType::Str, format!(": str = \"{s}\""), None);
                }
            }
            33 => {
                // An extern the host sets: a JSON document or a
                // partition, so a table kind crosses the input boundary
                // and its passthrough copies a table handle.
                if self.rng.coin(50) {
                    let w = self.declare_extern(PortType::Json, ": json".into(), None);
                    let n = self.rng.range(1000) as u64;
                    self.host_externs.push((
                        w.name,
                        Value::Json(std::sync::Arc::new(
                            serde_json::json!({ "n": n, "tags": ["a", "b"] }),
                        )),
                    ));
                } else {
                    let w = self.declare_extern(PortType::Ext, ": ext".into(), Some("Partition"));
                    let idx = self.rng.range(4) as u64;
                    self.host_externs.push((w.name, host_partition(idx)));
                }
            }
            34 => {
                let spec = [
                    "k in 1..4",
                    "k in 1..3, side in left,right",
                    "n in 0..10 order halton/3",
                ][self.rng.range(3)];
                self.bind_ext("Streamer", 0, format!("streamer(\"{spec}\")"));
            }
            // The shapes the general closure kit took on in engine
            // parity step 2: every one of these ran on the interpreter
            // only before it.
            35 => {
                // Byte strings through the arena: a `&[u8]` argument and
                // a `Vec<u8>` result, rendered as text at the end.
                let bytes = format!("u64_to_bytes({})", u.name);
                let expr = match self.rng.range(3) {
                    0 => format!("to_hex({bytes})"),
                    1 => format!("to_hex(sha256({bytes}))"),
                    _ => format!("to_hex(byte_slice({bytes}, 2, 4))"),
                };
                self.bind(PortType::Str, expr);
            }
            36 => {
                // A setup derived from consts on a scalar-or-string node.
                match self.rng.range(4) {
                    0 => {
                        if let Some(s) = self.pick(PortType::Str) {
                            self.bind(
                                PortType::Str,
                                format!("regex_replace({}, \"[0-9]\", \"x\")", s.name),
                            );
                        }
                    }
                    1 => {
                        if let Some(s) = self.pick(PortType::Str) {
                            self.bind(
                                PortType::Bool,
                                format!("regex_match({}, \"[0-9]+\")", s.name),
                            );
                        }
                    }
                    2 => {
                        self.bind(
                            PortType::Str,
                            format!("weighted_strings(hash({}), \"a:0.5;b:0.3;c:0.2\")", u.name),
                        );
                    }
                    _ => {
                        self.bind(
                            PortType::F64,
                            format!("dist_normal(hash({}), 0.0, 1.0)", u.name),
                        );
                    }
                }
            }
            37 => {
                // A const list.
                match self.rng.range(3) {
                    0 => self.bind(
                        PortType::U64,
                        format!("fixed_values_u64({}, 7, 9, 11)", u.name),
                    ),
                    1 => self.bind(
                        PortType::Str,
                        format!("one_of({}, \"x\", \"y\", \"z\")", u.name),
                    ),
                    _ => self.bind(
                        PortType::U64,
                        format!("is_one_of(u64_mod({}, 3), 0, 1, 2)", u.name),
                    ),
                };
            }
            38 => {
                // An `Option<u64>` argument, always present on a compiled
                // engine, and a `Config` wire.
                if self.rng.coin(50) {
                    self.bind(PortType::U64, format!("this_or({}, 7)", u.name));
                } else {
                    self.bind(
                        PortType::Str,
                        format!(
                            "dynamic_weighted_select(hash({}), \"alpha:0.3;beta:0.5;gamma:0.2\")",
                            u.name
                        ),
                    );
                }
            }
            39 => {
                // The split-halves variadic: two complementary selectors
                // over the digits of a number, two u64 values.
                let a = self.pick(PortType::U64).expect("cycle is always u64");
                let b = self.pick(PortType::U64).expect("cycle is always u64");
                self.bind(
                    PortType::U64,
                    format!(
                        "pick(regex_match(__u64_to_string({u}), \"3\"), \
                         regex_match(__u64_to_string({u}), \"^[^3]*$\"), {}, {})",
                        a.name,
                        b.name,
                        u = u.name
                    ),
                );
            }
            40 => {
                // A polymorphic return, encoded by the resolved output type.
                let w = self.any();
                if w.ext.is_none() && w.ty != PortType::Json {
                    let fallback = self.pick(w.ty).expect("the wire itself is a candidate");
                    self.bind(w.ty, format!("default_or({}, {})", w.name, fallback.name));
                }
            }
            41 => {
                // A cursor whose literal `over` clause denotes one
                // partition: resolved and seeded at build, so no kernel
                // needs a host call (engine parity step 3). Its slot is a
                // partition like any other for the family above.
                if !self.lines.iter().any(|l| l.starts_with("cursor ")) {
                    let spec = ["0..50%", "25%..75%", "*", "10%..90%"][self.rng.range(4)];
                    let name = self.name();
                    self.lines
                        .insert(1, format!("cursor {name} = range(0, 1000) over \"{spec}\""));
                    self.bind_ext("Partition", 0, format!("{name}.cursor"));
                }
            }
            42 => {
                // A side channel: the row it emits is counted per engine
                // per cycle, so a compiled kernel fires it exactly when
                // the interpreter does.
                let a = self.any();
                let b = self.any();
                if a.ext.is_none() && b.ext.is_none() {
                    self.bind(
                        PortType::U64,
                        format!("emit_row(\"csv\", \"a,b\", {}, {})", a.name, b.name),
                    );
                }
            }
            _ => {
                // A session-static setup, captured from the node.
                self.bind(PortType::Str, "tmp_dir()".to_string());
            }
        }
    }

    /// A hole over some wire, with a declared type or a format sometimes,
    /// raw sometimes when the tile allows it.
    fn hole(&mut self, raw_ok: bool, splice_ok: bool) -> String {
        let mut w = self.any();
        if (w.json_tile && !splice_ok) || w.ext.is_some() {
            w = self.pick(PortType::U64).expect("cycle is always u64");
        }
        // A declaration reaches only the catalog's lossless adapters; a
        // parse is written in the expression as a cast.
        let expr = if w.numeric_text && self.rng.coin(30) {
            format!("{} as u64", w.name)
        } else {
            w.name.clone()
        };
        let ty = if expr.ends_with(" as u64") {
            PortType::U64
        } else {
            w.ty
        };
        let decl = match ty {
            PortType::U64 if self.rng.coin(25) => [": str", ": f64", ": u64"][self.rng.range(3)],
            PortType::F64 if self.rng.coin(20) => ": str",
            PortType::Bool if self.rng.coin(40) => ": bool",
            _ => "",
        };
        let fmt = match ty {
            PortType::U64 => ["", " | 05", " | x", " | >8", " | X"][self.rng.range(5)],
            PortType::F64 => ["", " | .2", " | .0"][self.rng.range(3)],
            PortType::Str => ["", " | >6", " | <4"][self.rng.range(3)],
            _ => "",
        };
        let raw = if raw_ok && self.rng.coin(15) { "!" } else { "" };
        format!("${{{expr}{decl}{fmt}{raw}}}")
    }

    fn tile(&mut self) {
        let name = self.name();
        let kind = self.rng.range(3);
        let projection = self.rng.coin(35);
        let body = match kind {
            0 => {
                let mut fields = Vec::new();
                for i in 0..1 + self.rng.range(3) {
                    fields.push(format!("\"f{i}\": {}", self.hole(false, true)));
                }
                fields.push(format!("\"s\": \"x-{}-y\"", self.hole(false, false)));
                if let Some(b) = self.pick(PortType::Bool) {
                    fields.push(format!(
                        "\"b\": @if {} {{ {{\"on\": {}}} }} @else {{ false }}",
                        b.name,
                        self.hole(false, false)
                    ));
                }
                if projection {
                    fields.push(format!(
                        "\"xs\": [@for i in 0..3 sep \",\" {{ {{\"i\": ${{i}}, \"h\": {}}} }}]",
                        self.hole(false, false)
                    ));
                }
                if self.rng.coin(25) {
                    // An empty projection leaves the object valid.
                    fields.push("\"none\": [@for k in 1..1 { ${k} }]".to_string());
                }
                if self.rng.coin(25) {
                    // A projection in string position escapes as text.
                    fields.push(format!(
                        "\"sp\": \"@for k in 1..3 sep \\\"-\\\" {{${{k}}:{}}}\"",
                        self.hole(false, false)
                    ));
                }
                if self.rng.coin(30) && projection {
                    fields.push(format!(
                        "\"nest\": [@for r in 0..2 {{ {{\"r\": ${{r}}, \"c\": [@for c in 0..2 {{ ${{r * 10 + c}} }}], \"o\": {}}} }}]",
                        self.hole(false, false)
                    ));
                }
                if let Some(inner) = self.json_tiles.last().cloned()
                    && self.rng.coin(40)
                {
                    fields.push(format!("\"inner\": ${{{inner}}}"));
                }
                self.json_tiles.push(name.clone());
                format!("tile {name} : json := {{{}}}", fields.join(", "))
            }
            1 => {
                let mut parts = vec![format!("n={}", self.hole(true, false))];
                if let Some(b) = self.pick(PortType::Bool) {
                    parts.push(format!(
                        "@if {} {{yes {}}} @else {{no}}",
                        b.name,
                        self.hole(true, false)
                    ));
                }
                if projection {
                    parts.push(format!(
                        "[@for k in 1..4 sep \\\"-\\\" {{${{k}}:{}}}]",
                        self.hole(true, false)
                    ));
                }
                if self.rng.coin(20) {
                    // A literal open delimiter.
                    parts.push("${${lit".to_string());
                }
                format!("tile {name} : text := \"{}\"", parts.join(" "))
            }
            _ => {
                let cols: Vec<String> = (0..1 + self.rng.range(3))
                    .map(|_| self.hole(true, false))
                    .collect();
                format!(
                    "tile {name} : csv := \"{},{}\"",
                    cols.join(","),
                    self.hole(false, false)
                )
            }
        };
        self.lines.push(body);
        self.wires.push(Wire {
            name: name.clone(),
            ty: PortType::Str,
            numeric_text: false,
            json_tile: kind == 0,
            ext: None,
            depth: 0,
        });
        self.outputs.push(name);
    }

    /// Any wire that is not a `json` tile, for positions where a splice
    /// would not belong.
    fn non_tile(&mut self) -> Wire {
        let w = self.any();
        if w.json_tile || w.ext.is_some() {
            self.pick(PortType::U64).expect("cycle is always u64")
        } else {
            w
        }
    }

    /// A structural template handed in as JSON text.
    fn structural(&mut self) {
        let name = self.name();
        let v = self.non_tile();
        let s = self.non_tile();
        let mut members = vec![
            format!("\"id\": \"${{{}}}\"", v.name),
            format!("\"label\": \"row-${{{}}}\"", s.name),
            format!(
                "\"samples\": [ \"@for k in 0..2\", {{ \"k\": \"${{k}}\", \"v\": \"${{{}}}\" }} ]",
                v.name
            ),
        ];
        if let Some(b) = self.pick(PortType::Bool) {
            members.push(format!(
                "\"audit\": [ \"@if {}\", {{ \"by\": \"ops\" }} ]",
                b.name
            ));
        }
        if self.rng.coin(50) {
            members.push(format!(
                "\"tags\": {{ \"@for t in a,b\": {{ \"${{t}}-${{{}}}\": true }} }}",
                v.name
            ));
        }
        let text = format!("{{ {} }}", members.join(", "));
        self.lines
            .push(format!("{name} := polytile_json(<<<\n{text}\n>>>)"));
        self.wires.push(Wire {
            name: name.clone(),
            ty: PortType::Str,
            numeric_text: false,
            json_tile: true,
            ext: None,
            depth: 0,
        });
        self.outputs.push(name);
    }

    fn program(mut self, steps: usize) -> (String, Vec<String>, Vec<(String, Value)>) {
        for _ in 0..steps {
            self.step();
        }
        let mut src = self.lines.join("\n");
        src.push('\n');
        (src, self.outputs, self.host_externs)
    }
}

/// One of the four quarters of `[0, 1000)`, as a host would hand a
/// partition to an extern.
fn host_partition(idx: u64) -> Value {
    let list = polydat::iteration::cursor_partition::resolve(
        &polydat::iteration::cursor_partition::parse("*/4").expect("spec"),
        0,
        1000,
    )
    .expect("resolve");
    Value::from_partition(list[idx as usize % list.len()])
}

fn kernel(src: &str, mode: JitMode) -> PolydatKernel {
    let mut asm = compile_polydat_to_assembler(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    asm.set_jit_mode(mode);
    asm.compile().unwrap_or_else(|e| panic!("{e}\n{src}"))
}

fn same(a: &Value, b: &Value, tier: &str, out: &str, c: u64, src: &str) {
    assert_eq!(
        a.port_type(),
        b.port_type(),
        "{tier}: `{out}` at cycle {c}: type\n{src}"
    );
    assert_eq!(
        a.to_display_string(),
        b.to_display_string(),
        "{tier}: `{out}` at cycle {c}\n{src}"
    );
}

/// Every tier agrees with the interpreter on every output. Returns
/// whether a pure-P3 kernel was available for the program.
fn check(src: &str, outputs: &[&str], cycles: u64) -> bool {
    check_with(src, outputs, cycles, &[])
}

/// `check` for a program with externs the host sets: every kernel gets
/// the same values before it runs.
fn check_with(src: &str, outputs: &[&str], cycles: u64, externs: &[(String, Value)]) -> bool {
    // A constant side channel fires when its step is flattened: at
    // construction on a compiled kernel, at the first pull on the
    // interpreter (engines.md §3.1). Each engine's construction
    // rows count toward its first cycle.
    let _ = polydat::library::emit::take_rows();
    let built = || polydat::library::emit::take_rows().len();
    let mut p1 = kernel(src, JitMode::Off);
    let p1_built = built();
    let mut cones = kernel(src, JitMode::Force);
    let cones_built = built();
    let mut p2 = compile_polydat_to_assembler(src)
        .unwrap()
        .compile_slots(polydat::Engine::Closures(polydat::Provenance::Raw))
        .unwrap_or_else(|e| {
            panic!("every node here has a P2 form, but the closure tier refused it: {e}\n{src}")
        });
    let p2_built = built();
    let mut p2pp = compile_polydat_to_assembler(src)
        .unwrap()
        .compile_slots(polydat::Engine::Closures(polydat::Provenance::PushPull))
        .unwrap_or_else(|_| panic!("P2 push-pull\n{src}"));
    let _ = built();
    let mut p3 = compile_polydat_to_assembler(src)
        .unwrap()
        .compile_slots(polydat::Engine::PureNative(polydat::Provenance::PushPull))
        .ok();
    let p3_built = built();
    let mut hybrid = compile_polydat_to_assembler(src)
        .unwrap()
        .compile_slots(polydat::Engine::Native(polydat::Provenance::PushPull))
        .unwrap_or_else(|e| panic!("hybrid: {e}\n{src}"));
    let hybrid_built = built();
    for (name, value) in externs {
        p1.set_input(name, value.clone())
            .unwrap_or_else(|e| panic!("P1 set_input: {e}\n{src}"));
        cones
            .set_input(name, value.clone())
            .unwrap_or_else(|e| panic!("cones set_input: {e}\n{src}"));
        p2.set_input(name, value.clone())
            .unwrap_or_else(|e| panic!("P2 set_input: {e}\n{src}"));
        p2pp.set_input(name, value.clone())
            .unwrap_or_else(|e| panic!("P2 push-pull set_input: {e}\n{src}"));
        if let Some(k) = p3.as_mut() {
            k.set_input(name, value.clone())
                .unwrap_or_else(|e| panic!("P3 set_input: {e}\n{src}"));
        }
        hybrid
            .set_input(name, value.clone())
            .unwrap_or_else(|e| panic!("hybrid set_input: {e}\n{src}"));
    }
    // Every engine is driven through the `Kernel` trait, output by
    // output, so a compiled kernel runs each output's cone as the
    // interpreter does (engine parity step 5), and the side channel's
    // rows are counted per engine per cycle: what `emit_row` observes
    // is what it observes on the interpreter.
    let _ = polydat::library::emit::take_rows();
    for c in 0..cycles {
        let first = |n: usize| if c == 0 { n } else { 0 };
        p1.set_inputs(&[c]);
        let want: Vec<Value> = outputs.iter().map(|o| p1.pull(o).clone()).collect();
        let want_rows = built() + first(p1_built);
        cones.set_inputs(&[c]);
        let got_cones: Vec<Value> = outputs.iter().map(|o| cones.pull(o).clone()).collect();
        let cones_rows = built() + first(cones_built);
        p2.set_inputs(&[c]);
        let got_p2: Vec<Value> = outputs.iter().map(|o| p2.pull(o)).collect();
        let p2_rows = built() + first(p2_built);
        let got_p3: Option<Vec<Value>> = p3.as_mut().map(|k| {
            k.set_inputs(&[c]);
            outputs.iter().map(|o| k.pull(o)).collect()
        });
        let p3_rows = built() + first(p3_built);
        hybrid.set_inputs(&[c]);
        let got_hybrid: Vec<Value> = outputs.iter().map(|o| hybrid.pull(o)).collect();
        let hybrid_rows = built() + first(hybrid_built);
        for (i, out) in outputs.iter().enumerate() {
            same(&want[i], &got_cones[i], "cones", out, c, src);
            same(&want[i], &got_p2[i], "P2", out, c, src);
            if let Some(g) = got_p3.as_ref() {
                same(&want[i], &g[i], "P3", out, c, src);
            }
            same(&want[i], &got_hybrid[i], "hybrid", out, c, src);
        }
        assert_eq!(
            cones_rows, want_rows,
            "cones: rows emitted at cycle {c}\n{src}"
        );
        assert_eq!(p2_rows, want_rows, "P2: rows emitted at cycle {c}\n{src}");
        if p3.is_some() {
            assert_eq!(p3_rows, want_rows, "P3: rows emitted at cycle {c}\n{src}");
        }
        assert_eq!(
            hybrid_rows, want_rows,
            "hybrid: rows emitted at cycle {c}\n{src}"
        );
    }
    // The push-pull kernel over a repeating sequence: a reference
    // output skipped as current must still be the value its inputs
    // produced (axioms S3, S5).
    for &c in &[0u64, 1, 1, 2, 2, 2, 0, 3, 3, 1] {
        p1.set_inputs(&[c]);
        let want: Vec<Value> = outputs.iter().map(|o| p1.pull(o).clone()).collect();
        p2pp.eval_at(&[c]);
        let got: Vec<Value> = outputs.iter().map(|o| p2pp.get_value(o)).collect();
        for (i, out) in outputs.iter().enumerate() {
            same(&want[i], &got[i], "P2 push-pull", out, c, src);
        }
    }
    p3.is_some()
}

fn iterations() -> usize {
    std::env::var("FUZZ_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(150)
}

fn seed() -> u64 {
    std::env::var("FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(115)
}

#[test]
fn random_programs_agree_across_every_tier() {
    let base = seed();
    let mut p3_available = 0usize;
    let mut ext_seen = 0usize;
    for i in 0..iterations() {
        let (src, outputs, externs) = Gen::new(base.wrapping_add(i as u64)).program(4 + i % 7);
        if src.contains("partitions(") || src.contains("streamer(") || src.contains(": ext") {
            ext_seen += 1;
        }
        let outs: Vec<&str> = outputs.iter().map(String::as_str).collect();
        // A panic inside a kernel (a validator, a diagnostic) carries no
        // program text; attach it so the case can be reproduced.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_with(&src, &outs, 6, &externs)
        })) {
            Ok(true) => p3_available += 1,
            Ok(false) => {}
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "<non-string panic>".to_string());
                panic!(
                    "iteration {i} (seed {}):\n{msg}\n--- program ---\n{src}",
                    base.wrapping_add(i as u64)
                );
            }
        }
    }
    assert!(
        p3_available > 0,
        "some generated programs lower whole to P3"
    );
    assert!(
        ext_seen > 0,
        "some generated programs carry extension values"
    );
}

/// The hand-written corpus from the earlier steps, through the same
/// tiers.
#[test]
fn the_corpus_agrees_across_every_tier() {
    const WIRES: &str = "input cycle: u64\nh := hash(cycle)\nf := to_f64(h) / 7.0\nb := u64_gt(h, 5)\ns := __u64_to_string(h)\n";
    let corpus = [
        format!("{WIRES}out := printf(\"id={{:05}} hex={{:x}} f={{:.3}} b={{}} s={{:>24}}|{{}}\", cycle, h, f, b, s, h)\n"),
        format!("{WIRES}arr := json_array(h, f, b, s, cycle)\ntxt := json_to_str(arr)\n"),
        format!("{WIRES}obj := json_object(json_with(\"id\", h), json_with(\"name\", s), json_with(\"ok\", b))\nt := json_to_str(obj)\n"),
        "input cycle: u64\nj := to_json(hash(cycle))\nt := json_text(j)\nback := str_to_json(t)\n".to_string(),
        "input cycle: u64\nt := \"{{\\\"n\\\": {cycle}}}\"\nj := __str_to_json(t)\nback := json_to_str(j)\n".to_string(),
        format!("{WIRES}tile d : json := {{\"n\": ${{h}}, \"f\": ${{f | .2}}, \"s\": ${{s}}, \"t\": ${{b}}, \"in\": \"x-${{h}}-${{s}}\"}}\n"),
        format!("{WIRES}tile t : text := \"hello ${{s}} #${{h | 06}} @if b {{yes}} @else {{no}}\"\n"),
        "input cycle: u64\ntile t : text := \"${cycle}: @for k in 1..4 sep \\\",\\\" {${k}}\"\n".to_string(),
        "input cycle: u64\nh := hash(cycle)\ntile d : json := {\"h\": ${h}, \"xs\": [@for i in 0..3 sep \",\" { {\"i\": ${i}, \"h\": ${h}} }]}\n".to_string(),
        format!("{WIRES}c := str_concat(s, \",q\\\"z\")\ntile r : csv := \"${{h}},${{c}},${{s!}},${{f | .1}}\"\n"),
        format!("{WIRES}tile inner : json := {{\"n\": ${{h}}}}\ntile outer : json := {{\"a\": ${{inner}}, \"b\": ${{inner}}, \"e\": [@for k in 1..1 {{ ${{k}} }}], \"sp\": \"@for k in 1..3 sep \\\"-\\\" {{${{k}}:${{s}}}}\"}}\n"),
        format!("{WIRES}d := polytile_json(<<<\n{{ \"id\": \"${{h}}\", \"label\": \"row-${{s}}\", \"samples\": [ \"@for k in 0..2\", {{ \"k\": \"${{k}}\" }} ], \"audit\": [ \"@if b\", {{ \"by\": \"ops\" }} ], \"tags\": {{ \"@for t in a,b\": {{ \"${{t}}\": true }} }} }}\n>>>)\n"),
        format!("{WIRES}a := h as str\nc := __str_to_u64(s)\ne := escape_json(s)\nm := json_merge(json_object(json_with(\"x\", h)), json_object(json_with(\"y\", s)))\np := json_to_str_pretty(m)\n"),
        // Extension values: the partition family and the streamer, read
        // through every consumer and rendered as JSON, text, and a string.
        format!("{WIRES}q := partitions(\"20%,30%,*\", 500)\nn := partition_count(q)\np := partition_at(q, u64_mod(h, n))\nc := cardinality(p)\nm := mod_in(h, p)\nl := subdivide(p, 4)\nk := streamer(\"k in 1..4\")\nr := random_in(p, h)\na := at(p, u64_mod(h, cardinality(p)))\nj := json_object(json_with(\"p\", p), json_with(\"k\", k))\nd := printf(\"{{}} {{}} {{}}\", p, l, k)\ne := \"{{p}}/{{l}}\"\nx := str_concat(\"x\", p, k)\n"),
    ];
    for src in &corpus {
        let asm = compile_polydat_to_assembler(src).unwrap();
        let outputs: Vec<String> = asm.output_names().iter().map(|s| s.to_string()).collect();
        let outs: Vec<&str> = outputs.iter().map(String::as_str).collect();
        check(src, &outs, 8);
    }
}

/// A reference output is owned by its step and stands until an input
/// in its provenance changes: with repeating coordinates the P2
/// push-pull kernel and the hybrid kernel skip the current string and
/// JSON steps and still read the values their inputs produced.
#[test]
fn provenance_kernels_keep_reference_outputs_current_on_repeated_coordinates() {
    let src = "input cycle: u64\nh := hash(cycle)\ns := __u64_to_string(h)\nj := to_json(s)\nt := json_to_str(j)\nn := __str_to_u64(s)\n";
    let mut p1 = kernel(src, JitMode::Off);
    let mut p2 = compile_polydat_to_assembler(src)
        .unwrap()
        .compile_slots(polydat::Engine::Closures(polydat::Provenance::PushPull))
        .unwrap_or_else(|_| panic!("P2 push-pull"));
    let mut hybrid = compile_polydat_to_assembler(src)
        .unwrap()
        .compile_slots(polydat::Engine::Native(polydat::Provenance::PushPull))
        .expect("hybrid");
    let coords = [3u64, 3, 3, 4, 4, 3, 3, 5, 5, 5];
    for &c in &coords {
        p1.set_inputs(&[c]);
        let want: Vec<Value> = ["s", "j", "t", "n"]
            .iter()
            .map(|o| p1.pull(o).clone())
            .collect();
        p2.eval_at(&[c]);
        let got: Vec<Value> = ["s", "j", "t", "n"]
            .iter()
            .map(|o| p2.get_value(o))
            .collect();
        assert_eq!(
            got.iter().map(Value::to_display_string).collect::<Vec<_>>(),
            want.iter()
                .map(Value::to_display_string)
                .collect::<Vec<_>>(),
            "P2 at {c}"
        );
        hybrid.eval_at(&[c]);
        let got: Vec<Value> = ["s", "j", "t", "n"]
            .iter()
            .map(|o| hybrid.get_value(o))
            .collect();
        assert_eq!(
            got.iter().map(Value::to_display_string).collect::<Vec<_>>(),
            want.iter()
                .map(Value::to_display_string)
                .collect::<Vec<_>>(),
            "hybrid at {c}"
        );
    }
}

/// A P2 kernel's JSON outputs are each owned by their step's scratch
/// and replaced in place on every run; the S9 validator ran after
/// every eval, in this debug build.
#[test]
fn a_p2_kernel_replaces_reference_outputs_in_place() {
    let src = "input cycle: u64\nh := hash(cycle)\nj := __u64_to_json(h)\nk := to_json(h)\nt := json_to_str(k)\n";
    let mut p2 = compile_polydat_to_assembler(src)
        .unwrap()
        .compile_slots(polydat::Engine::Closures(polydat::Provenance::Raw))
        .unwrap_or_else(|_| panic!("P2"));
    for c in 0..50u64 {
        p2.eval_at(&[c]);
        assert_eq!(
            p2.get_value("t").as_str(),
            p2.get_value("j").to_display_string()
        );
    }
}

/// Externs of every kind the compiled engines carry, through every
/// tier: two with defaults the kernels seed, and a JSON document and a
/// partition the harness sets on every kernel, read through the
/// partition consumers, JSON text, interpolation, and a tile.
#[test]
fn externs_agree_across_every_tier() {
    let src = "input cycle: u64\n\
        extern scale: u64 = 100\n\
        extern label: str = \"lbl\"\n\
        extern doc: json\n\
        extern part: ext\n\
        h := hash(cycle)\n\
        id := mod_wire(h, scale)\n\
        m := mod_in(h, part)\n\
        c := cardinality(part)\n\
        s := start_of(part)\n\
        t := json_to_str(doc)\n\
        copy := identity(doc)\n\
        line := \"{label}/{id}/{m}/{c}/{s}/{t}/{part}\"\n\
        tile d : json := {\"label\": ${label}, \"id\": ${id}, \"m\": ${m}, \"doc\": ${t}}\n";
    let externs = vec![
        (
            "doc".to_string(),
            Value::Json(std::sync::Arc::new(
                serde_json::json!({ "k": 7, "tags": ["a", "b"] }),
            )),
        ),
        ("part".to_string(), host_partition(2)),
    ];
    let asm = compile_polydat_to_assembler(src).unwrap();
    let outputs: Vec<String> = asm.output_names().iter().map(|s| s.to_string()).collect();
    let outs: Vec<&str> = outputs.iter().map(String::as_str).collect();
    check_with(src, &outs, 8, &externs);
}
