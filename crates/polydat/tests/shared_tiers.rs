// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Engine parity, step 9 (A10): a `shared` binding runs on every engine
//! with the interpreter's cell protocol. The binding's slot is bound to
//! a `SharedCell`; a write publishes through the cell; a read takes the
//! cell's current value, inside a cycle too; and a host attaches one
//! kernel's cell to another through the `Kernel` trait so both read
//! and write one register.

use polydat::ast::Value;
use polydat::dsl::compile::compile_polydat_with;
use polydat::{Engine, JitMode, Kernel, Provenance};

const SRC: &str = "input cycle: u64\nshared counter := 10\nout := cycle + counter\n";

fn engines() -> Vec<Engine> {
    let mut all = vec![Engine::Interpreter(JitMode::Auto)];
    for m in [
        Provenance::Raw,
        Provenance::Push,
        Provenance::Pull,
        Provenance::PushPull,
    ] {
        all.push(Engine::Closures(m));
        #[cfg(feature = "jit")]
        all.push(Engine::Native(m));
    }
    all
}

fn kernel(engine: Engine) -> Box<dyn Kernel> {
    compile_polydat_with(SRC, engine).unwrap_or_else(|e| panic!("{engine}: {e}"))
}

#[test]
fn a_shared_binding_runs_on_every_engine_and_writes_through() {
    for engine in engines() {
        let mut k = kernel(engine);
        k.set_inputs(&[3]);
        assert_eq!(k.pull("out"), Value::U64(13), "{engine}: the default");
        assert_eq!(k.pull("counter"), Value::U64(10), "{engine}: the binding");
        k.set_input("counter", Value::U64(4)).unwrap();
        assert_eq!(k.pull("out"), Value::U64(7), "{engine}: after a write");
        let cells = k.shared_cells();
        assert_eq!(cells.len(), 1, "{engine}");
        assert_eq!(cells[0].name, "counter");
        assert_eq!(
            cells[0].cell.snapshot().0,
            Value::U64(4),
            "{engine}: the cell holds the write"
        );
    }
}

#[test]
fn kernels_attached_to_one_cell_read_and_write_one_register() {
    for engine in engines() {
        let mut a = kernel(engine);
        let mut b = kernel(engine);
        a.set_inputs(&[3]);
        b.set_inputs(&[100]);
        assert_eq!(a.pull("out"), Value::U64(13), "{engine}");
        assert_eq!(b.pull("out"), Value::U64(110), "{engine}");
        // Their cells are their own until attached.
        a.set_input("counter", Value::U64(1)).unwrap();
        assert_eq!(b.pull("out"), Value::U64(110), "{engine}: unattached");
        let cell = a.shared_cells()[0].cell.clone();
        b.attach_shared_cell("counter", cell).unwrap();
        // The attached kernel reads the register at once.
        assert_eq!(b.pull("out"), Value::U64(101), "{engine}: attached");
        // A write on either is what the other reads next, inside its
        // open cycle, without new coordinates.
        b.set_input("counter", Value::U64(20)).unwrap();
        assert_eq!(a.pull("out"), Value::U64(23), "{engine}: b's write on a");
        a.set_input("counter", Value::U64(30)).unwrap();
        assert_eq!(b.pull("out"), Value::U64(130), "{engine}: a's write on b");
        // The cell itself holds the last write.
        assert_eq!(a.shared_cells()[0].cell.snapshot().0, Value::U64(30));
        // A write outside both kernels, through the cell, reaches both.
        a.shared_cells()[0].cell.publish(Value::U64(40));
        assert_eq!(a.pull("out"), Value::U64(43), "{engine}: published to a");
        assert_eq!(b.pull("out"), Value::U64(140), "{engine}: published to b");
    }
}

#[test]
fn kernels_created_from_a_shared_program_have_their_own_cells() {
    for engine in engines() {
        let program = kernel(engine).into_program();
        let mut a = program.clone().create_kernel();
        let mut b = program.create_kernel();
        a.set_inputs(&[0]);
        b.set_inputs(&[0]);
        a.set_input("counter", Value::U64(7)).unwrap();
        assert_eq!(a.pull("out"), Value::U64(7), "{engine}");
        assert_eq!(
            b.pull("out"),
            Value::U64(10),
            "{engine}: b keeps its own cell"
        );
    }
}

#[test]
fn a_shared_write_is_type_stable_and_a_non_shared_name_takes_no_cell() {
    for engine in engines() {
        let mut k = kernel(engine);
        assert!(
            k.set_input("counter", Value::Str("no".into())).is_err(),
            "{engine}: a cell keeps one type"
        );
        let cell = k.shared_cells()[0].cell.clone();
        let err = k.attach_shared_cell("out", cell).unwrap_err();
        assert!(err.contains("shared"), "{engine}: {err}");
    }
}
