// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Trait implementations on the interpreter kernel: the
//! engine-independent [`Kernel`](crate::kernel::Kernel) surface every
//! engine shares, and the three interpreter-only traits ([`Metadata`],
//! [`Dataflow`], [`Construction`]).

use crate::ast::{PortType, Value};
use crate::kernel::{Construction, Dataflow, Metadata, PolydatKernel};

impl Metadata for PolydatKernel {
    #[inline]
    fn find_input(&self, name: &str) -> Option<usize> {
        self.program().find_input(name)
    }

    #[inline]
    fn input_names(&self) -> Vec<String> {
        self.program().input_names()
    }

    #[inline]
    fn output_names(&self) -> Vec<String> {
        self.program()
            .output_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[inline]
    fn coord_count(&self) -> usize {
        self.program().coord_count()
    }

    #[inline]
    fn input_port_type(&self, name: &str) -> Option<PortType> {
        self.program().input_port_type(name)
    }

    #[inline]
    fn input_port_type_by_idx(&self, idx: usize) -> Option<PortType> {
        self.program().input_port_type_by_idx(idx)
    }

    #[inline]
    fn output_port_type(&self, name: &str) -> Option<PortType> {
        self.program().output_port_type(name)
    }
}

impl Dataflow for PolydatKernel {
    fn set_wire_idx(
        &mut self,
        idx: usize,
        value: Value,
    ) -> Result<(), crate::kernel::api::WriteError> {
        use crate::kernel::api::WriteError;

        // Look up the slot's declared port type. An out-of-range
        // index is an unknown-wire error rather than a panic —
        // the typed boundary surfaces the diagnostic uniformly
        // for callers who computed the index from external
        // metadata.
        let slot_type = match self.program().input_port_type_by_idx(idx) {
            Some(t) => t,
            None => {
                return Err(WriteError::UnknownWire {
                    key: format!("wire[{idx}]"),
                    known: Vec::new(),
                });
            }
        };

        let slot_name = self
            .program()
            .input_name_by_idx(idx)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("wire[{idx}]"));

        // Per S4 / T2: try direct write, fall back to the
        // boundary auto-adapter, then report a TypeMismatch
        // diagnostic if no adapter healed the mismatch. The
        // `adapt_boundary_value` helper currently passes
        // unhealable mismatches through with a warning; here we
        // detect that case by checking whether the adapted value
        // still has the wrong port type, and surface a typed
        // error instead of letting silent corruption propagate
        // to downstream readers.
        let got = value.port_type();
        let adapted = crate::kernel::state::adapt_boundary_value(&slot_name, slot_type, value);
        // `Value::None` is the absent sentinel — always permitted
        // regardless of slot type (per none_semantics.md / SRD-74
        // Rule 1). For non-None values, the residual check uses
        // the bit-stuffing equivalence helper
        // (`Value::satisfies_slot`) so a narrowing adapter that
        // outputs `Value::U64` for a U32 slot — the runtime
        // bit-stuffed form per type_system.md §1 — passes
        // validation. The pre-adapter check in
        // `adapt_boundary_value` remains strict, so an unadapted
        // Value::U64 cannot silently truncate into a U32 slot.
        if !adapted.satisfies_slot(slot_type) {
            return Err(WriteError::TypeMismatch {
                slot: slot_name,
                expected: slot_type,
                got,
            });
        }
        self.state().set_input(idx, adapted);
        Ok(())
    }

    #[inline]
    fn get_wire_idx(&self, idx: usize) -> Value {
        self.state_ref().get_input(idx)
    }
}

impl Construction for PolydatKernel {
    type Error = crate::kernel::subcontext::ContractViolation;

    fn root(matter: crate::kernel::subcontext::PolydatMatter<'_>) -> Result<Self, Self::Error> {
        use crate::kernel::subcontext::PolydatMatterInner;
        match matter.inner {
            PolydatMatterInner::Source(s) => {
                let options = crate::dsl::compile::CompileOptions {
                    source_dir: s.options.workload_dir.clone(),
                    lib_paths: s.options.polydat_lib_paths,
                    required_outputs: s.options.required_outputs.clone(),
                    strict: s.options.strict,
                    context: s
                        .options
                        .context_label
                        .clone()
                        .unwrap_or_else(|| s.label.clone()),
                    cursor_limit: s.options.cursor_limit,
                    ledger: None,
                    engine: crate::Engine::default(),
                };
                crate::dsl::compile::compile_polydat_with_options(&s.body, &options, None).map_err(
                    |e| crate::kernel::subcontext::ContractViolation::Compile(e.to_string()),
                )
            }
            PolydatMatterInner::Statements(s) => {
                // Pre-parsed AST — go through the compile-from-AST
                // path. The `PolydatFile` AST root takes the statements
                // verbatim; the same options surface as the source
                // path.
                let file = crate::dsl::ast::PolydatFile {
                    statements: s.statements,
                };
                let options = crate::dsl::compile::CompileOptions {
                    source_dir: s.options.workload_dir.clone(),
                    lib_paths: s.options.polydat_lib_paths,
                    required_outputs: s.options.required_outputs.clone(),
                    strict: s.options.strict,
                    context: s
                        .options
                        .context_label
                        .clone()
                        .unwrap_or_else(|| s.label.clone()),
                    cursor_limit: None,
                    ledger: None,
                    engine: crate::Engine::default(),
                };
                crate::dsl::compile::compile_ast_with_options(&file, "", &options, None).map_err(
                    |e| crate::kernel::subcontext::ContractViolation::Compile(e.to_string()),
                )
            }
            PolydatMatterInner::Program(p) => {
                let mut k = PolydatKernel::from_program(p.program);
                for (var, value) in p.iter_bindings {
                    if let Some(idx) = k.program().find_input(var) {
                        k.state().set_input(idx, value.clone());
                    }
                }
                Ok(k)
            }
        }
    }

    fn subscope(
        &self,
        matter: crate::kernel::subcontext::PolydatMatter<'_>,
    ) -> Result<Self, Self::Error> {
        // Delegate to PolydatKernel's existing typed subscope path.
        PolydatKernel::build_subscope(self, matter)
    }
}

// ── The interpreter kernel on the engine-independent surface ────────

impl crate::kernel::Kernel for PolydatKernel {
    fn engine(&self) -> crate::compile::select::Engine {
        crate::compile::select::Engine::Interpreter(self.program().cone_mode())
    }
    fn set_inputs(&mut self, coords: &[u64]) {
        PolydatKernel::set_inputs(self, coords);
    }
    fn set_input(&mut self, name: &str, value: Value) -> Result<(), crate::kernel::WriteError> {
        PolydatKernel::set_input(self, name, value)
    }
    fn set_cursor(
        &mut self,
        name: &str,
        partition: &crate::iteration::cursor_partition::Partition,
    ) -> Result<(), crate::kernel::WriteError> {
        PolydatKernel::set_cursor(self, name, partition)
    }
    /// Every output is pulled, so what a side channel observes is what
    /// it observes on a compiled kernel's run.
    fn eval(&mut self) {
        for name in Metadata::output_names(self) {
            let _ = PolydatKernel::pull(self, &name);
        }
    }
    fn pull(&mut self, name: &str) -> Value {
        PolydatKernel::pull(self, name).clone()
    }
    fn input_names(&self) -> Vec<String> {
        Metadata::input_names(self)
    }
    fn output_names(&self) -> Vec<String> {
        Metadata::output_names(self)
    }
    fn output_type(&self, name: &str) -> Option<PortType> {
        Metadata::output_port_type(self, name)
    }
    fn externs(&self) -> Vec<(String, PortType)> {
        let program = self.program();
        Metadata::input_names(self)
            .into_iter()
            .enumerate()
            .filter(|(i, _)| program.input_kind(*i) != Some(crate::kernel::InputKind::Coordinate))
            .filter_map(|(i, name)| Metadata::input_port_type_by_idx(self, i).map(|t| (name, t)))
            .collect()
    }
    fn cursor_schemas(&self) -> &[crate::iteration::source::SourceSchema] {
        self.program().cursor_schemas()
    }
    fn input_value(&self, name: &str) -> Option<Value> {
        let idx = self.program().find_input(name)?;
        Some(self.state_ref().get_input(idx))
    }
    fn input_index(&self, name: &str) -> Option<usize> {
        self.program().find_input(name)
    }
    fn set_input_at(
        &mut self,
        index: usize,
        value: Value,
    ) -> Result<(), crate::kernel::WriteError> {
        PolydatKernel::set_input_at(self, index, value)
    }
    fn output_index(&self, name: &str) -> Option<usize> {
        self.program().output_index(name)
    }
    fn pull_at(&mut self, index: usize) -> Value {
        PolydatKernel::pull_by_index(self, index).clone()
    }
    fn traversals(&self) -> &[crate::dsl::traversal::Traversal] {
        self.program().traversals()
    }
    fn plan(&self) -> crate::EnginePlan {
        self.program().engine_plan()
    }
    fn traverse(&mut self, index: usize) -> Result<crate::kernel::TraversalStream, String> {
        PolydatKernel::traverse(self, index)
    }
    fn invalidate_all(&mut self) {
        self.state().invalidate_all();
    }
    fn shared_cells(&self) -> Vec<crate::kernel::SharedCellEntry> {
        self.shared_cells_in_scope()
    }
    fn attach_shared_cell(
        &mut self,
        name: &str,
        cell: crate::kernel::SharedCell,
    ) -> Result<(), String> {
        let program = self.program().clone();
        let shared = program.shared_outputs();
        let idx = program.find_input(name).filter(|_| shared.contains(&name));
        let Some(idx) = idx else {
            return Err(format!(
                "no `shared` binding named '{name}'; this kernel's shared bindings are {shared:?}"
            ));
        };
        self.state().attach_shared_cell(idx, cell);
        Ok(())
    }
    fn into_program(self: Box<Self>) -> std::sync::Arc<dyn crate::kernel::KernelProgram> {
        PolydatKernel::into_program(*self)
    }
    fn ledger(&self) -> &std::sync::Arc<crate::kernel::CompileLedger> {
        self.program().ledger()
    }
}

impl crate::kernel::KernelInternals for PolydatKernel {
    fn set_traversals(
        &mut self,
        traversals: Vec<crate::dsl::traversal::Traversal>,
        producers: Vec<crate::dsl::traversal::Producer>,
    ) {
        PolydatKernel::set_traversals(self, traversals, producers);
    }
    fn folded_value(&self, name: &str) -> Option<Value> {
        self.get_constant(name).cloned()
    }
    fn set_cursor_extent(&mut self, index: usize, extent: u64) {
        let mut schemas = self.program().cursor_schemas().to_vec();
        if let Some(schema) = schemas.get_mut(index) {
            schema.extent = Some(extent);
            self.set_cursor_schemas(schemas);
        }
    }
}

impl crate::kernel::KernelProgram for crate::kernel::PolydatProgram {
    fn engine(&self) -> crate::compile::select::Engine {
        crate::compile::select::Engine::Interpreter(self.cone_mode())
    }
    fn as_interpreter(self: std::sync::Arc<Self>) -> Option<std::sync::Arc<Self>> {
        Some(self)
    }
    fn create_kernel(self: std::sync::Arc<Self>) -> Box<dyn crate::kernel::Kernel> {
        Box::new(PolydatKernel::from_program(self))
    }
    fn ledger(&self) -> &std::sync::Arc<crate::kernel::CompileLedger> {
        crate::kernel::PolydatProgram::ledger(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::compile::compile_polydat_interpreter;

    /// Indexed wire access works.
    #[test]
    fn dataflow_indexed_set_get() {
        let mut k = compile_polydat_interpreter("input cycle: u64\nconst x := 7\n").unwrap();
        // cycle is index 0
        k.set_wire(0_usize, Value::U64(42)).expect("typed write");
        assert_eq!(k.get_wire(0_usize), Some(Value::U64(42)));
    }

    /// Named wire access resolves through metadata.
    #[test]
    fn dataflow_named_set_get() {
        let mut k = compile_polydat_interpreter("input cycle: u64\nextern n: u64\n").unwrap();
        k.set_wire("n", Value::U64(5)).expect("typed write");
        match k.get_wire("n") {
            Some(Value::U64(5)) => {}
            other => panic!("expected U64(5), got {other:?}"),
        }
    }

    /// String key works alongside &str.
    #[test]
    fn dataflow_string_key() {
        let mut k = compile_polydat_interpreter("input cycle: u64\nextern n: u64\n").unwrap();
        let name = String::from("n");
        k.set_wire(&name, Value::U64(99)).expect("typed write");
        assert_eq!(k.get_wire(name.clone()), Some(Value::U64(99)));
    }

    /// Unknown name returns Err(UnknownWire) / None — no panic.
    #[test]
    fn dataflow_unknown_name_safe() {
        let mut k = compile_polydat_interpreter("input cycle: u64\n").unwrap();
        let err = k.set_wire("nonexistent", Value::U64(1)).unwrap_err();
        assert!(matches!(
            err,
            crate::kernel::api::WriteError::UnknownWire { .. }
        ));
        assert!(k.get_wire("nonexistent").is_none());
    }

    /// S4 type-check: writing the wrong Value variant to a typed
    /// slot returns Err(TypeMismatch) when no boundary adapter
    /// can heal the mismatch.
    ///
    /// `VecF32 → U64` is intentionally absent from the polyfill
    /// matrix (type_system.md §3 — collection → scalar requires
    /// explicit choice), so it is a stable "no adapter exists"
    /// pair for testing the diagnostic.
    #[test]
    fn dataflow_type_mismatch_rejected() {
        let mut k = compile_polydat_interpreter("input cycle: u64\nextern n: u64\n").unwrap();
        let err = k
            .set_wire(
                "n",
                Value::VecF32(crate::ast::SliceArc::from_vec(vec![1.0_f32, 2.0])),
            )
            .unwrap_err();
        match err {
            crate::kernel::api::WriteError::TypeMismatch {
                slot,
                expected,
                got,
            } => {
                assert_eq!(slot, "n");
                assert_eq!(expected, PortType::U64);
                assert_eq!(got, PortType::VecF32);
            }
            other => panic!("expected TypeMismatch, got {other:?}"),
        }
    }

    /// The `WriteError::TypeMismatch` Display impl includes a
    /// vec → scalar hint naming the reduction the program needs
    /// when the rejected `got` is a Vec type and the `expected` is
    /// not a collection-compatible type.
    #[test]
    fn vec_to_scalar_diagnostic_mentions_explicit_helpers() {
        let err = crate::kernel::api::WriteError::TypeMismatch {
            slot: "score".into(),
            expected: PortType::F64,
            got: PortType::VecF32,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("reduction node"),
            "missing reduction hint: {msg}"
        );
        assert!(msg.contains("vec_dot"), "missing vec_dot hint: {msg}");
    }

    /// S4 type-adapt: a healable mismatch (u64 → f64) routes
    /// through the boundary auto-adapter rather than rejecting.
    #[test]
    fn dataflow_healable_mismatch_adapts() {
        let mut k = compile_polydat_interpreter("input cycle: u64\nextern x: f64\n").unwrap();
        // u64 → f64 has an auto-adapter (lossless widening); the
        // typed-write API should accept this transparently.
        k.set_wire("x", Value::U64(42))
            .expect("u64→f64 boundary adapter");
        match k.get_wire("x") {
            Some(Value::F64(42.0)) => {}
            other => panic!("expected adapted F64(42.0), got {other:?}"),
        }
    }

    /// S4 None pass-through: Value::None is the absent sentinel
    /// and always permitted at the boundary regardless of slot
    /// type (per none_semantics.md).
    #[test]
    fn dataflow_none_passes_through_any_slot() {
        let mut k = compile_polydat_interpreter("input cycle: u64\nextern n: u64\n").unwrap();
        k.set_wire("n", Value::None).expect("None always permitted");
    }

    /// Metadata trait surfaces names + types.
    #[test]
    fn metadata_listings() {
        let k = compile_polydat_interpreter(
            "input (cycle: u64, thread: u64)\nextern n: u64\nconst x := 7\n",
        )
        .unwrap();
        let inputs: Vec<String> = k.input_names();
        assert!(inputs.iter().any(|s| s == "cycle"));
        assert!(inputs.iter().any(|s| s == "n"));
        assert_eq!(k.coord_count(), 2); // cycle + thread
        assert!(k.find_input("n").is_some());
        assert_eq!(k.input_port_type("n"), Some(PortType::U64));
    }

    /// Construction trait — both paths take the same polydat
    /// matter type. Verify symmetry: root from source, then
    /// subscope from source against the root.
    #[test]
    fn construction_symmetric_paths() {
        let root_opts = crate::kernel::subcontext::CompileOptions {
            workload_dir: None,
            polydat_lib_paths: Vec::new(),
            strict: false,
            required_outputs: Vec::new(),
            context_label: Some("root".to_string()),
            cursor_limit: None,
            ..Default::default()
        };
        let root_matter = crate::kernel::subcontext::PolydatMatter::builder()
            .label("root")
            .source("input cycle: u64\nshared flag := 0\n")
            .options(root_opts)
            .build()
            .expect("matter build");
        let root =
            <PolydatKernel as Construction>::root(root_matter).expect("root from source matter");

        let sub_opts = crate::kernel::subcontext::CompileOptions {
            workload_dir: None,
            polydat_lib_paths: Vec::new(),
            strict: false,
            required_outputs: Vec::new(),
            context_label: Some("sub".to_string()),
            cursor_limit: None,
            ..Default::default()
        };
        let sub_matter = crate::kernel::subcontext::PolydatMatter::builder()
            .label("sub")
            .source("input cycle: u64\n")
            .options(sub_opts)
            .build()
            .expect("matter build");
        let _sub = root
            .subscope(sub_matter)
            .expect("subscope from source matter");
    }

    /// Root construction also accepts pre-compiled program
    /// matter (re-instance with fresh state). Verifies via
    /// the input slot — `n` is an extern input.
    #[test]
    fn construction_root_from_program() {
        let template = compile_polydat_interpreter("input cycle: u64\nextern n: u64\n").unwrap();
        let program = template.program().clone();
        let matter = crate::kernel::subcontext::PolydatMatter::builder()
            .program(program)
            .build()
            .expect("matter build");
        let mut root =
            <PolydatKernel as Construction>::root(matter).expect("root from program matter");
        root.set_wire("n", Value::U64(13)).expect("set_wire");
        assert_eq!(root.get_wire("n"), Some(Value::U64(13)));
    }

    /// Builder rejects ambiguous matter (multiple input forms).
    #[test]
    fn builder_rejects_multiple_forms() {
        let template = compile_polydat_interpreter("input cycle: u64\n").unwrap();
        match crate::kernel::subcontext::PolydatMatter::builder()
            .source("input cycle: u64\n")
            .program(template.program().clone())
            .build()
        {
            Err(msg) => assert!(
                msg.contains("multiple"),
                "expected multiple-forms error, got: {msg}"
            ),
            Ok(_) => panic!("multiple forms must error"),
        }
    }

    /// Builder rejects empty matter.
    #[test]
    fn builder_rejects_empty() {
        match crate::kernel::subcontext::PolydatMatter::builder().build() {
            Err(msg) => assert!(
                msg.contains("no input form"),
                "expected no-form error, got: {msg}"
            ),
            Ok(_) => panic!("empty matter must error"),
        }
    }
}
