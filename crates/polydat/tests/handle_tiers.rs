// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 115 §7, step 7: the P1↔P2↔P3 differential suite for handle
//! slots. A grammar-directed generator emits random programs over the
//! string, JSON, and tile nodes with typed wires, casts, formats,
//! declared hole types, raw holes, splices, branches, projections
//! (nested, in string position, and empty), and structural templates;
//! every program compiles on the interpreter (the oracle), with forced
//! cone extraction, as a P2 closure kernel, as a hybrid kernel, and
//! where every node lowers as a pure-P3 kernel, and every output agrees
//! across the tiers by type and text for a run of cycles (axiom H7).
//! The P2 push-pull kernel is driven over a repeating coordinate
//! sequence as well, so the never-clean rule for handle steps is under
//! the same oracle. The P2, P3, and hybrid kernels run the H4 validator
//! after every run in these debug builds.
//!
//! `FUZZ_SEED` and `FUZZ_ITERATIONS` follow the other fuzzers.

#![cfg(feature = "jit")]

use polydat::ast::{PortType, Value};
use polydat::dsl::compile::compile_polydat_to_assembler;
use polydat::kernel::PolydatKernel;
use polydat::JitMode;

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
        if n == 0 { 0 } else { (self.next_u64() as usize) % n }
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
}

struct Gen {
    rng: Rng,
    wires: Vec<Wire>,
    /// Names of `json` tiles, for splicing.
    json_tiles: Vec<String>,
    lines: Vec<String>,
    outputs: Vec<String>,
    next: usize,
}

impl Gen {
    fn new(seed: u64) -> Self {
        Gen {
            rng: Rng::new(seed),
            wires: vec![Wire { name: "cycle".into(), ty: PortType::U64, numeric_text: false, json_tile: false }],
            json_tiles: Vec::new(),
            lines: vec!["input cycle: u64".into()],
            outputs: Vec::new(),
            next: 0,
        }
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
        let candidates: Vec<Wire> = self.wires.iter().filter(|w| w.numeric_text).cloned().collect();
        if candidates.is_empty() {
            None
        } else {
            Some(candidates[self.rng.range(candidates.len())].clone())
        }
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
        let w = Wire { name: name.clone(), ty, numeric_text, json_tile: false };
        self.wires.push(w.clone());
        self.outputs.push(name);
        w
    }

    /// One binding from the grammar.
    fn step(&mut self) {
        let u = self.pick(PortType::U64).expect("cycle is always u64");
        match self.rng.range(26) {
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
                        PortType::U64 => ["{}", "{:05}", "{:x}", "{:X}", "{:b}", "{:o}", "{:8}"][self.rng.range(7)],
                        PortType::F64 => ["{}", "{:.2}", "{:.0}", "{:10.3}"][self.rng.range(4)],
                        PortType::Str => ["{}", "{:>12}", "{:3}"][self.rng.range(3)],
                        _ => "{}",
                    };
                    specs.push(spec.to_string());
                    args.push(w.name);
                }
                let fmt = specs.join(["-", " ", "|", "{{}}"][self.rng.range(4)]);
                self.bind(PortType::Str, format!("printf(\"{fmt}\", {})", args.join(", ")));
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
                self.bind(PortType::I64, format!("__u64_to_i64(mod({}, 1000003))", u.name));
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
                            self.bind(PortType::I64, format!("__str_to_i64({})", d.name));
                        }
                    }
                }
            }
            19 => {
                if let (Some(a), Some(b)) = (self.pick(PortType::Json), self.pick(PortType::Json)) {
                    self.bind(PortType::Json, format!("json_merge({}, {})", a.name, b.name));
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
                    self.bind(PortType::Str, format!("str_concat({}, \",q\\\"z\")", s.name));
                }
            }
            23 => self.structural(),
            _ => self.tile(),
        }
    }

    /// A hole over some wire, with a declared type or a format sometimes,
    /// raw sometimes when the tile allows it.
    fn hole(&mut self, raw_ok: bool, splice_ok: bool) -> String {
        let mut w = self.any();
        if w.json_tile && !splice_ok {
            w = self.pick(PortType::U64).expect("cycle is always u64");
        }
        // A declaration reaches only the catalog's lossless adapters; a
        // parse is written in the expression as a cast.
        let expr = if w.numeric_text && self.rng.coin(30) { format!("{} as u64", w.name) } else { w.name.clone() };
        let ty = if expr.ends_with(" as u64") { PortType::U64 } else { w.ty };
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
                    fields.push(format!("\"b\": @if {} {{ {{\"on\": {}}} }} @else {{ false }}", b.name, self.hole(false, false)));
                }
                if projection {
                    fields.push(format!("\"xs\": [@for i in 0..3 sep \",\" {{ {{\"i\": ${{i}}, \"h\": {}}} }}]", self.hole(false, false)));
                }
                if self.rng.coin(25) {
                    // An empty projection leaves the object valid.
                    fields.push("\"none\": [@for k in 1..1 { ${k} }]".to_string());
                }
                if self.rng.coin(25) {
                    // A projection in string position escapes as text.
                    fields.push(format!("\"sp\": \"@for k in 1..3 sep \\\"-\\\" {{${{k}}:{}}}\"", self.hole(false, false)));
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
                    parts.push(format!("@if {} {{yes {}}} @else {{no}}", b.name, self.hole(true, false)));
                }
                if projection {
                    parts.push(format!("[@for k in 1..4 sep \\\"-\\\" {{${{k}}:{}}}]", self.hole(true, false)));
                }
                if self.rng.coin(20) {
                    // A literal open delimiter.
                    parts.push("${${lit".to_string());
                }
                format!("tile {name} : text := \"{}\"", parts.join(" "))
            }
            _ => {
                let cols: Vec<String> = (0..1 + self.rng.range(3)).map(|_| self.hole(true, false)).collect();
                format!("tile {name} : csv := \"{},{}\"", cols.join(","), self.hole(false, false))
            }
        };
        self.lines.push(body);
        self.wires.push(Wire { name: name.clone(), ty: PortType::Str, numeric_text: false, json_tile: kind == 0 });
        self.outputs.push(name);
    }

    /// Any wire that is not a `json` tile, for positions where a splice
    /// would not belong.
    fn non_tile(&mut self) -> Wire {
        let w = self.any();
        if w.json_tile {
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
            format!("\"samples\": [ \"@for k in 0..2\", {{ \"k\": \"${{k}}\", \"v\": \"${{{}}}\" }} ]", v.name),
        ];
        if let Some(b) = self.pick(PortType::Bool) {
            members.push(format!("\"audit\": [ \"@if {}\", {{ \"by\": \"ops\" }} ]", b.name));
        }
        if self.rng.coin(50) {
            members.push(format!("\"tags\": {{ \"@for t in a,b\": {{ \"${{t}}-${{{}}}\": true }} }}", v.name));
        }
        let text = format!("{{ {} }}", members.join(", "));
        self.lines.push(format!("{name} := polytile_json(<<<\n{text}\n>>>)"));
        self.wires.push(Wire { name: name.clone(), ty: PortType::Str, numeric_text: false, json_tile: true });
        self.outputs.push(name);
    }

    fn program(mut self, steps: usize) -> (String, Vec<String>) {
        for _ in 0..steps {
            self.step();
        }
        let mut src = self.lines.join("\n");
        src.push('\n');
        (src, self.outputs)
    }
}

fn kernel(src: &str, mode: JitMode) -> PolydatKernel {
    let mut asm = compile_polydat_to_assembler(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    asm.set_jit_mode(mode);
    asm.compile().unwrap_or_else(|e| panic!("{e}\n{src}"))
}

fn same(a: &Value, b: &Value, tier: &str, out: &str, c: u64, src: &str) {
    assert_eq!(a.port_type(), b.port_type(), "{tier}: `{out}` at cycle {c}: type\n{src}");
    assert_eq!(a.to_display_string(), b.to_display_string(), "{tier}: `{out}` at cycle {c}\n{src}");
}

/// Every tier agrees with the interpreter on every output. Returns
/// whether a pure-P3 kernel was available for the program.
fn check(src: &str, outputs: &[&str], cycles: u64) -> bool {
    let mut p1 = kernel(src, JitMode::Off);
    let mut cones = kernel(src, JitMode::Force);
    let mut p2 = compile_polydat_to_assembler(src)
        .unwrap()
        .try_compile_raw()
        .unwrap_or_else(|k| {
            let p = k.program();
            let names: Vec<String> = (0..p.node_count()).map(|i| p.node_meta(i).name.clone()).collect();
            panic!("every node here has a P2 form, but the kernel fell back; nodes: {names:?}\n{src}")
        });
    let mut p2pp = compile_polydat_to_assembler(src)
        .unwrap()
        .try_compile()
        .unwrap_or_else(|_| panic!("P2 push-pull\n{src}"));
    let mut p3 = compile_polydat_to_assembler(src).unwrap().try_compile_jit().ok();
    let mut hybrid = compile_polydat_to_assembler(src)
        .unwrap()
        .compile_hybrid()
        .unwrap_or_else(|e| panic!("hybrid: {e}\n{src}"));
    // Every kernel here is a root on this thread, and a root's cycle
    // advance resets the arena (SRD 115 §4), so each kernel's outputs
    // are copied out right after its own run, before the next root
    // kernel runs. Reading them later would read through handles into
    // storage another kernel has reused.
    for c in 0..cycles {
        p1.set_inputs(&[c]);
        let want: Vec<Value> = outputs.iter().map(|o| p1.pull(o).clone()).collect();
        cones.set_inputs(&[c]);
        let got_cones: Vec<Value> = outputs.iter().map(|o| cones.pull(o).clone()).collect();
        p2.eval(&[c]);
        let got_p2: Vec<Value> = outputs.iter().map(|o| p2.get_value(o)).collect();
        let got_p3: Option<Vec<Value>> = p3.as_mut().map(|k| {
            k.eval(&[c]);
            outputs.iter().map(|o| k.get_value(o)).collect()
        });
        hybrid.eval(&[c]);
        let got_hybrid: Vec<Value> = outputs.iter().map(|o| hybrid.get_value(o)).collect();
        for (i, out) in outputs.iter().enumerate() {
            same(&want[i], &got_cones[i], "cones", out, c, src);
            same(&want[i], &got_p2[i], "P2", out, c, src);
            if let Some(g) = got_p3.as_ref() {
                same(&want[i], &g[i], "P3", out, c, src);
            }
            same(&want[i], &got_hybrid[i], "hybrid", out, c, src);
        }
    }
    // The push-pull kernel over a repeating sequence: clean-step
    // skipping must never leave a handle slot pointing into storage the
    // next root cycle reused (SRD 115 §4).
    for &c in &[0u64, 1, 1, 2, 2, 2, 0, 3, 3, 1] {
        p1.set_inputs(&[c]);
        let want: Vec<Value> = outputs.iter().map(|o| p1.pull(o).clone()).collect();
        p2pp.eval(&[c]);
        let got: Vec<Value> = outputs.iter().map(|o| p2pp.get_value(o)).collect();
        for (i, out) in outputs.iter().enumerate() {
            same(&want[i], &got[i], "P2 push-pull", out, c, src);
        }
    }
    p3.is_some()
}

fn iterations() -> usize {
    std::env::var("FUZZ_ITERATIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(150)
}

fn seed() -> u64 {
    std::env::var("FUZZ_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(115)
}

#[test]
fn random_programs_agree_across_every_tier() {
    let base = seed();
    let mut p3_available = 0usize;
    for i in 0..iterations() {
        let (src, outputs) = Gen::new(base.wrapping_add(i as u64)).program(4 + i % 7);
        let outs: Vec<&str> = outputs.iter().map(String::as_str).collect();
        if check(&src, &outs, 6) {
            p3_available += 1;
        }
    }
    assert!(p3_available > 0, "some generated programs lower whole to P3");
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
    ];
    for src in &corpus {
        let asm = compile_polydat_to_assembler(src).unwrap();
        let outputs: Vec<String> = asm.output_names().iter().map(|s| s.to_string()).collect();
        let outs: Vec<&str> = outputs.iter().map(String::as_str).collect();
        check(src, &outs, 8);
    }
}

/// A step that writes a handle slot is never marked clean in the
/// provenance kernels (SRD 115 §4): with repeating coordinates the P2
/// push-pull kernel and the hybrid kernel still recompute their string
/// and JSON outputs each run, so no slot holds a handle into storage the
/// root cycle has reset.
#[test]
fn provenance_kernels_rerun_handle_steps_on_repeated_coordinates() {
    let src = "input cycle: u64\nh := hash(cycle)\ns := __u64_to_string(h)\nj := to_json(s)\nt := json_to_str(j)\nn := __str_to_u64(s)\n";
    let mut p1 = kernel(src, JitMode::Off);
    let mut p2 = compile_polydat_to_assembler(src).unwrap().try_compile().unwrap_or_else(|_| panic!("P2 push-pull"));
    let mut hybrid = compile_polydat_to_assembler(src).unwrap().compile_hybrid().expect("hybrid");
    let coords = [3u64, 3, 3, 4, 4, 3, 3, 5, 5, 5];
    for &c in &coords {
        p1.set_inputs(&[c]);
        let want: Vec<Value> = ["s", "j", "t", "n"].iter().map(|o| p1.pull(o).clone()).collect();
        p2.eval(&[c]);
        let got: Vec<Value> = ["s", "j", "t", "n"].iter().map(|o| p2.get_value(o)).collect();
        assert_eq!(got.iter().map(Value::to_display_string).collect::<Vec<_>>(), want.iter().map(Value::to_display_string).collect::<Vec<_>>(), "P2 at {c}");
        hybrid.eval(&[c]);
        let got: Vec<Value> = ["s", "j", "t", "n"].iter().map(|o| hybrid.get_value(o)).collect();
        assert_eq!(got.iter().map(Value::to_display_string).collect::<Vec<_>>(), want.iter().map(Value::to_display_string).collect::<Vec<_>>(), "hybrid at {c}");
    }
}

/// A P2 kernel owns a fixed table sized to its table-kind slots and
/// replaces entries in place; the H4 validator ran after every eval
/// above, in this debug build.
#[test]
fn a_p2_kernel_owns_a_fixed_table() {
    let src = "input cycle: u64\nh := hash(cycle)\nj := __u64_to_json(h)\nk := to_json(h)\nt := json_to_str(k)\n";
    let mut p2 = compile_polydat_to_assembler(src).unwrap().try_compile_raw().unwrap_or_else(|_| panic!("P2"));
    assert_eq!(p2.table_len(), 2);
    for c in 0..50u64 {
        p2.eval(&[c]);
        assert_eq!(p2.get_value("t").as_str(), p2.get_value("j").to_display_string());
    }
    assert_eq!(p2.table_len(), 2);
}
