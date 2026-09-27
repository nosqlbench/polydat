// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Node factory: maps Polydat function names to runtime node instances.
//!
//! `build_node` is the single construction point: the compiler turns
//! every call expression into a `Box<dyn PolydatNode>` through it, and it
//! builds through the [`NodeFactory`](crate::dsl::factories::NodeFactory)
//! that owns the function name. `ConstArg` captures assembly-time
//! constant arguments extracted from the AST.

use crate::ast::PolydatNode;
use crate::compile::assembly::WireRef;
use crate::library::identity::ConstU64;

use crate::dsl::registry;

/// Constant arguments extracted from the AST.
///
/// Holds assembly-time values (integers, floats, strings, float arrays)
/// that are baked into node constructors rather than passed as wire inputs.
///
/// `pub` visibility is required so that `NodeRegistration::build` function
/// pointers (which are `pub` fields) can name this type.
#[derive(Clone)]
pub enum ConstArg {
    /// An integer literal.
    Int(u64),
    /// A float literal.
    Float(f64),
    /// A string literal.
    Str(String),
    /// Workload-list const carrier for the
    /// `Const<Vec<C>>` shape. Each inner [`ConstArg`] is one
    /// element; the macro emits the walk over the list and the
    /// per-element extraction for the element type it read out of
    /// the signature.
    List(Vec<ConstArg>),
    /// A value the compiler built and hands the node as it is.
    ///
    /// The other variants are what a literal in the source parses to.
    /// This one is for what the compiler makes: a tile's skeleton with
    /// its projection bodies already lowered, for instance. Carried as
    /// the value itself rather than as serialized text, it cannot fail
    /// to parse at node construction, and it keeps everything the
    /// compiler knows that JSON cannot carry.
    ///
    /// The receiving node names the concrete type, which
    /// [`ConstArg::as_opaque`] downcasts to.
    Opaque(std::sync::Arc<dyn std::any::Any + Send + Sync>),
}

impl std::fmt::Debug for ConstArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstArg::Int(v) => f.debug_tuple("Int").field(v).finish(),
            ConstArg::Float(v) => f.debug_tuple("Float").field(v).finish(),
            ConstArg::Str(s) => f.debug_tuple("Str").field(s).finish(),
            ConstArg::List(l) => f.debug_tuple("List").field(l).finish(),
            // A compiler-built value has no text form, and the point
            // of carrying it opaquely is that this layer does not know
            // what it is.
            ConstArg::Opaque(_) => f.write_str("Opaque(..)"),
        }
    }
}

impl ConstArg {
    /// The value as `T`, when this is an opaque const the compiler
    /// built and `T` is the type it built.
    pub fn as_opaque<T: std::any::Any + Send + Sync>(&self) -> Option<std::sync::Arc<T>> {
        match self {
            ConstArg::Opaque(v) => v.clone().downcast::<T>().ok(),
            _ => None,
        }
    }

    /// Return the value as a `u64`, or 0 if incompatible.
    pub fn as_u64(&self) -> u64 {
        match self {
            ConstArg::Int(v) => *v,
            _ => 0,
        }
    }

    /// Return the value as an `f64`, or 0.0 if incompatible.
    ///
    /// Integer literals are widened to f64.
    pub fn as_f64(&self) -> f64 {
        match self {
            ConstArg::Float(v) => *v,
            ConstArg::Int(v) => *v as f64,
            _ => 0.0,
        }
    }

    /// Return the value as a `&str`, or `""` if incompatible.
    pub fn as_str(&self) -> &str {
        match self {
            ConstArg::Str(s) => s,
            _ => "",
        }
    }
}

/// What the compiler tells a node factory about the node it is building:
/// the bindings under construction when the node was asked for, and the
/// program tree's resource scope.
///
/// Every factory receives one ([`crate::dsl::registry::NodeBuildFn`]).
/// A factory that records attribution reads [`Self::binding`]:
/// `rate_adj := control_set("rate", target)` builds `control_set` with
/// `binding()` equal to `Some("rate_adj")`. A node built inside another
/// binding's construction sees the whole chain in [`Self::bindings`],
/// outermost first: an argument that is itself a call compiles as an
/// intermediate binding the compiler names after its enclosing one
/// (`x := f(control_set("r", t))` builds `control_set` under
/// `["x", "x__anon_0"]`), and a module body's bindings compile under the
/// binding that called the module. The context is a value the compiler
/// hands down, so attribution is exact under nesting and on any thread.
///
/// A node that looks up a host resource when it evaluates keeps a clone
/// of [`Self::resources`] (resource.rs).
///
/// A `#[polydat_node]` node reads the context through a setup that
/// names `ctx` first, `#[poly_const(setup, from = (ctx, key))]`, whose
/// function takes `&BuildContext` and returns what the node keeps;
/// the macro's `new()` then takes the context as its first argument.
/// A caller building such a node directly passes
/// [`BuildContext::with_binding`] or [`BuildContext::new`].
#[derive(Clone, Debug, Default)]
pub struct BuildContext {
    bindings: Vec<String>,
    resources: crate::resource::ResourceScope,
}

impl BuildContext {
    /// A context for a node built under `bindings` (outermost first) in
    /// the program tree whose resource scope is `resources`.
    pub fn new(bindings: Vec<String>, resources: crate::resource::ResourceScope) -> Self {
        Self {
            bindings,
            resources,
        }
    }

    /// A context for a node built by the binding `name` alone, in a
    /// tree of its own with no accessor installed: what a host's test
    /// passes to build a node that records its binding.
    pub fn with_binding(name: impl Into<String>) -> Self {
        Self::new(vec![name.into()], crate::resource::ResourceScope::new())
    }

    /// The binding whose construction built the node: the innermost of
    /// [`Self::bindings`]. `None` for a node built outside any binding,
    /// such as one a caller builds directly with [`build_node`].
    pub fn binding(&self) -> Option<&str> {
        self.bindings.last().map(String::as_str)
    }

    /// Every binding under construction when the node was built,
    /// outermost first.
    pub fn bindings(&self) -> &[String] {
        &self.bindings
    }

    /// The resource scope of the program tree the node belongs to.
    pub fn resources(&self) -> &crate::resource::ResourceScope {
        &self.resources
    }
}

/// Build the node `func` takes for the given wires, their types, and
/// constant arguments, through the registry; an unknown function or a
/// mismatched signature is an error naming it. `ctx` reaches the
/// factory unchanged.
///
/// Every node is built here, and built by the factory whose signatures
/// list `func` ([`registry::factories`]): the parameters' declared
/// constraints are checked, then the factory's
/// [`validate`](crate::dsl::factories::NodeFactory::validate), then its
/// [`build`](crate::dsl::factories::NodeFactory::build).
pub fn build_node(
    ctx: &BuildContext,
    func: &str,
    wires: &[WireRef],
    wire_types: &[crate::ast::PortType],
    consts: &[ConstArg],
) -> Result<Box<dyn PolydatNode>, String> {
    for factory in registry::factories() {
        // The signatures say which factory owns `func`; validation runs
        // before construction, so a constructor never sees a constant
        // its validator would refuse.
        let Some(sig) = factory.signatures().iter().find(|s| s.name == func) else {
            continue;
        };
        // Per-parameter constraints ("must be in [0,1]", "must be one
        // of {2,8,10,16}", "spec must parse").
        if let Err(msg) = check_param_constraints(sig, consts) {
            return Err(format!("bad constant {func}: {msg}"));
        }
        // The factory's own relational and cross-parameter rules
        // (`n_of`'s n ≤ m).
        if let Err(reason) = factory.validate(func, consts) {
            return Err(format!("bad constant {func}: {reason}"));
        }
        return factory.build(ctx, func, wires, wire_types, consts);
    }

    let mut msg = format!("unknown function: '{func}'\n");
    if let Some(suggestion) = registry::suggest_function(func) {
        msg.push_str(&format!("\n  Did you mean '{suggestion}'?"));
    }
    msg.push_str("\n\n  This function is not registered in the wiring function library.");
    msg.push_str("\n  See the registered wiring functions for the available names.");
    Err(msg)
}

/// The node a variadic signature builds for `wire_count` wires: its
/// identity element for none, its variadic constructor otherwise. A
/// signature that is not variadic, or has neither, builds nothing and
/// is an error naming the function.
pub(crate) fn variadic_node(
    sig: &registry::FuncSig,
    wire_count: usize,
) -> Result<Box<dyn PolydatNode>, String> {
    let func = sig.name;
    if !sig.is_variadic() {
        return Err(format!(
            "'{func}' is registered, but its registration builds no node for it"
        ));
    }
    if wire_count == 0 {
        return match sig.identity {
            Some(id) => Ok(Box::new(ConstU64::new(id))),
            None => Err(format!(
                "variadic function '{func}' requires at least one input"
            )),
        };
    }
    match sig.variadic_ctor {
        Some(ctor) => Ok(ctor(wire_count)),
        None => Err(format!(
            "variadic function '{func}' is registered without a variadic constructor"
        )),
    }
}

/// Walk `sig.params`, applying every declared `ConstConstraint`
/// to the corresponding positional `ConstArg`. Parameters with no
/// constraint are skipped; missing optional arguments are skipped
/// (the `required` flag handles mandatory presence elsewhere).
fn check_param_constraints(
    sig: &crate::dsl::registry::FuncSig,
    consts: &[ConstArg],
) -> Result<(), String> {
    use crate::ast::SlotType;
    // Const args appear in positional order, but `sig.params`
    // mixes wire and const slots. Walk both in lockstep, pulling
    // const args from a separate counter.
    let mut const_idx = 0usize;
    for spec in sig.params {
        if matches!(spec.slot_type, SlotType::Wire) {
            continue;
        }
        if let Some(constraint) = &spec.constraint
            && let Some(arg) = consts.get(const_idx)
        {
            constraint.check(arg, spec.name)?;
        }
        const_idx += 1;
    }
    Ok(())
}
