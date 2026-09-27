// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Node factories, the host-side registry view, and extern resolvers.
//!
//! A [`NodeFactory`] is what the registry builds every node through
//! (`factory::build_node`). `PolydatRuntime` lists the registry and
//! carries module search paths. The extern-resolver registry is what
//! the kernel consults at runtime.

use std::path::PathBuf;
use std::sync::Mutex;

use crate::ast::{PolydatNode, PortType, Value};
use crate::compile::assembly::WireRef;
use crate::dsl::factory::{BuildContext, ConstArg};
use crate::dsl::registry::{FuncCategory, FuncSig};

// ───── Virtual-wire resolver registry (γ-8) ─────

/// Host-mediated extern resolver per
/// `expression_engine.md` §5.6 (virtual wires). A resolver
/// is a callback that fires at Context Fusion scope-init
/// time when a kernel's extern slot can't be satisfied
/// from the outer scope's direct bindings.
///
/// Arguments: slot name + declared slot type + the kernel
/// being initialised. The resolver returns `Some(value)` to
/// fill the slot or `None` to fall through to ordinary
/// resolution (typed error if no other source exists).
///
/// Per §5.6.2's contract: the returned `Value`'s type must
/// match the slot's declared `PortType` (boundary adapter
/// applies otherwise per γ-5); the resolver fires once per
/// scope-init (per S3); and the resolver is responsible for
/// its own determinism.
pub type ExternResolver = Box<dyn Fn(&str, PortType) -> Option<Value> + Send + Sync>;

/// Process-level registry of virtual-wire resolvers.
///
/// Resolvers are registered via [`register_extern_resolver`]
/// and consulted by Context Fusion's
/// `materialize_wiring_from_outer` when an outer-chain
/// lookup yields nothing. Multiple resolvers iterate in
/// registration order; the first matching resolver wins.
///
/// Test isolation: tests that register a resolver should
/// call [`clear_extern_resolvers`] in a teardown block to
/// avoid leaking state across tests.
static RESOLVERS: Mutex<Vec<ExternResolver>> = Mutex::new(Vec::new());

/// Register a virtual-wire resolver. Resolvers stay
/// registered for the process's lifetime unless
/// [`clear_extern_resolvers`] is called.
///
/// Per `expression_engine.md` §5.6 PLANNED → γ-8 SHIPPED.
pub fn register_extern_resolver(resolver: ExternResolver) {
    let mut r = RESOLVERS.lock().unwrap();
    r.push(resolver);
}

/// Clear every registered resolver. Primarily for tests
/// that want clean teardown.
pub fn clear_extern_resolvers() {
    let mut r = RESOLVERS.lock().unwrap();
    r.clear();
}

/// Try every registered resolver in order; return the
/// first `Some(value)` whose type matches `slot_type`
/// (or whose type the catalog can adapt to `slot_type`).
///
/// Called by `PolydatKernel::materialize_wiring_from_outer`
/// as a fall-through after outer-chain lookup.
pub(crate) fn resolve_extern(slot_name: &str, slot_type: PortType) -> Option<Value> {
    let r = RESOLVERS.lock().unwrap();
    for resolver in r.iter() {
        if let Some(value) = resolver(slot_name, slot_type) {
            return Some(value);
        }
    }
    None
}

// ───── End virtual-wire resolver registry ─────

/// What the registry builds a node through: a set of function
/// signatures and the constructor for each.
///
/// Every node the compiler builds comes from a factory's [`Self::build`]
/// ([`crate::dsl::factory::build_node`]). A node module's
/// [`NodeRegistration`](crate::dsl::registry::NodeRegistration), the
/// form `#[polydat_node]` and `register_nodes!` emit, is a factory whose
/// `build` is its registered [`NodeBuildFn`](crate::dsl::registry::NodeBuildFn).
/// A host that wants a constructor of its own implements this trait and
/// links the factory into the registry with a
/// [`FactoryRegistration`](crate::dsl::registry::FactoryRegistration):
///
/// ```ignore
/// static COUNTERS: MyFactory = MyFactory::new();
/// polydat::inventory::submit! {
///     polydat::dsl::registry::FactoryRegistration { factory: &COUNTERS }
/// }
/// ```
///
/// Its nodes are then indistinguishable from built-in ones: same
/// registry, same listings, same type checking, on every engine.
pub trait NodeFactory: Send + Sync {
    /// The functions this factory builds. The registry lists them, and
    /// a call to one of them is built by this factory.
    fn signatures(&self) -> &[FuncSig];

    /// Check a call's constant arguments before [`Self::build`]; an
    /// `Err` fails the compile as a bad constant, so `build` never sees
    /// a malformed literal. The registry has already applied each
    /// parameter's declared `ConstConstraint`. Accepts everything
    /// unless overridden.
    fn validate(&self, _name: &str, _consts: &[ConstArg]) -> Result<(), String> {
        Ok(())
    }

    /// Build the node `name`, one of [`Self::signatures`], for the given
    /// wires, their resolved port types, and its constant arguments.
    /// `ctx` says which bindings are under construction and which
    /// resource scope the program tree has
    /// ([`BuildContext`](crate::dsl::factory::BuildContext)).
    fn build(
        &self,
        ctx: &BuildContext,
        name: &str,
        wires: &[WireRef],
        wire_types: &[PortType],
        consts: &[ConstArg],
    ) -> Result<Box<dyn PolydatNode>, String>;
}

/// A host-side registry view: the linked registry for listing, and the
/// module search paths.
pub struct PolydatRuntime {
    /// Additional module search paths (the `--lib` search paths).
    polydat_lib_paths: Vec<PathBuf>,
}

impl PolydatRuntime {
    /// A runtime with no module search paths.
    pub fn new() -> Self {
        Self {
            polydat_lib_paths: Vec::new(),
        }
    }

    /// Add a module search path (one of the `--lib` search paths).
    pub fn add_polydat_lib(&mut self, path: PathBuf) {
        self.polydat_lib_paths.push(path);
    }

    /// Every function the linked registry builds: the built-in library,
    /// `#[polydat_node]` and `register_nodes!` registrations, and every
    /// registered [`NodeFactory`].
    pub fn registry(&self) -> Vec<FuncSig> {
        crate::dsl::registry::registry()
    }

    /// [`Self::registry`] grouped by category, in display order.
    pub fn by_category(&self) -> Vec<(FuncCategory, Vec<FuncSig>)> {
        crate::dsl::registry::by_category()
    }

    /// The `--lib` search paths.
    pub fn polydat_lib_paths(&self) -> &[PathBuf] {
        &self.polydat_lib_paths
    }
}

impl Default for PolydatRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: serialise resolver-registry tests via a
    /// process-wide mutex so two tests don't race the static
    /// `RESOLVERS`.
    static RESOLVER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn extern_resolver_register_and_lookup() {
        let _guard = RESOLVER_TEST_LOCK.lock().unwrap();
        clear_extern_resolvers();
        register_extern_resolver(Box::new(|name, _typ| {
            if name == "region" {
                Some(Value::Str("us-east-1".into()))
            } else {
                None
            }
        }));
        let v = resolve_extern("region", PortType::Str);
        assert_eq!(v, Some(Value::Str("us-east-1".into())));
        let v = resolve_extern("missing", PortType::Str);
        assert_eq!(v, None);
        clear_extern_resolvers();
    }

    #[test]
    fn extern_resolver_first_match_wins() {
        let _guard = RESOLVER_TEST_LOCK.lock().unwrap();
        clear_extern_resolvers();
        register_extern_resolver(Box::new(|name, _typ| {
            if name == "k" {
                Some(Value::U64(1))
            } else {
                None
            }
        }));
        register_extern_resolver(Box::new(|name, _typ| {
            if name == "k" {
                Some(Value::U64(2))
            } else {
                None
            }
        }));
        let v = resolve_extern("k", PortType::U64);
        // First registered wins.
        assert_eq!(v, Some(Value::U64(1)));
        clear_extern_resolvers();
    }

    #[test]
    fn extern_resolver_clear_removes_all() {
        let _guard = RESOLVER_TEST_LOCK.lock().unwrap();
        register_extern_resolver(Box::new(|_, _| Some(Value::U64(99))));
        assert!(resolve_extern("anything", PortType::U64).is_some());
        clear_extern_resolvers();
        assert!(resolve_extern("anything", PortType::U64).is_none());
    }

    #[test]
    fn default_runtime_has_builtins() {
        let rt = PolydatRuntime::new();
        let reg = rt.registry();
        assert!(reg.len() >= 50);
    }
}
