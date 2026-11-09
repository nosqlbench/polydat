// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Function registry: known function signatures for DSL validation.
//!
//! Each registered function declares its name, category, expected wire
//! inputs, constant parameters, output count, and variadic behavior.
//! The compiler uses this to validate calls at parse time and to
//! generically dispatch variadic functions.
//!
//! Categories are a type-safe enum — every function must declare one.
//! Categories group the registry for listings (`by_category`). The
//! stdlib files carry a `// @category: Name` line for readers;
//! `FuncCategory::parse` maps that text but no compile path consumes
//! it.
//!
//! Signatures are owned by their respective node modules. This file
//! defines the shared types and the collector function.

pub use crate::ast::CompileLevel;
use crate::compile::assembly::WireRef;
use crate::dsl::factories::NodeFactory;

/// Builder for a node module: `(build context, name, wires, resolved
/// wire port types, const args) -> Some(Ok(node)) / Some(Err(msg))`,
/// or `None` when the name isn't handled by this module. The build
/// context carries the bindings under construction and the program
/// tree's resource scope ([`crate::dsl::factory::BuildContext`]).
pub type NodeBuildFn = fn(
    &crate::dsl::factory::BuildContext,
    &str,
    &[WireRef],
    &[crate::ast::PortType],
    &[crate::dsl::factory::ConstArg],
) -> Option<Result<Box<dyn crate::ast::PolydatNode>, String>>;

/// A node module's registration: signatures + builder.
///
/// Each node module submits one of these at link time via `inventory::submit!`.
/// The runtime collects all submissions to build the function registry and
/// dispatch table without any explicit module list.
///
/// A registration is a [`NodeFactory`]: its `build` is the factory's
/// body, and the registry builds its nodes through that factory as it
/// builds every other node ([`crate::dsl::factory::build_node`]).
pub struct NodeRegistration {
    /// Returns the static slice of `FuncSig` entries for this module.
    pub signatures: fn() -> &'static [FuncSig],
    /// Builds a node for the given function name: the body of this
    /// registration's [`NodeFactory::build`].
    ///
    /// Returns `Some(Ok(node))` / `Some(Err(msg))` for a name it
    /// builds, and `None` for one it leaves to its signature: a
    /// variadic function then gets its signature's identity element or
    /// variadic constructor, and any other name is an unknown function.
    ///
    /// `wire_types[i]` is the resolved [`crate::ast::PortType`] of
    /// `wires[i]` — the output type of the upstream node feeding
    /// that wire input. Modules that build type-polymorphic nodes
    /// (e.g. `log_info`, whose output type equals its input type)
    /// read this to construct the node with the correct port
    /// types. Modules whose nodes have type-fixed signatures can
    /// ignore the slice. When the assembler can't resolve a
    /// wire's type (forward reference, dangling), the slot
    /// defaults to [`crate::ast::PortType::U64`].
    pub build: NodeBuildFn,
    /// Optional assembly-time validator for this module's constants.
    ///
    /// The factory calls this **before** `build` whenever the name
    /// matches one of this module's functions. Returning `Err` makes
    /// the compile fail with a structured `bad constant` error, so
    /// the node itself never sees a malformed literal and can keep
    /// its constructor and `eval()` branch-free.
    pub validate: Option<crate::dsl::const_constraints::NodeValidator>,
}

inventory::collect!(NodeRegistration);

impl NodeFactory for NodeRegistration {
    fn signatures(&self) -> &[FuncSig] {
        (self.signatures)()
    }

    fn validate(&self, name: &str, consts: &[crate::dsl::factory::ConstArg]) -> Result<(), String> {
        match self.validate {
            Some(validator) => validator(name, consts),
            None => Ok(()),
        }
    }

    /// The registered builder's node. A builder that declines a name its
    /// signatures list leaves a variadic function to its signature: the
    /// identity element for no wires, the variadic constructor otherwise.
    fn build(
        &self,
        ctx: &crate::dsl::factory::BuildContext,
        name: &str,
        wires: &[WireRef],
        wire_types: &[crate::ast::PortType],
        consts: &[crate::dsl::factory::ConstArg],
    ) -> Result<Box<dyn crate::ast::PolydatNode>, String> {
        if let Some(result) = (self.build)(ctx, name, wires, wire_types, consts) {
            return result;
        }
        let sig = (self.signatures)().iter().find(|s| s.name == name);
        match sig {
            Some(sig) => crate::dsl::factory::variadic_node(sig, wires.len()),
            None => Err(format!("unknown function: '{name}'")),
        }
    }
}

/// A host's own [`NodeFactory`], linked into the registry.
///
/// A host submits one at link time, as a node module submits its
/// [`NodeRegistration`]:
///
/// ```ignore
/// static FACTORY: MyFactory = MyFactory;
/// polydat::inventory::submit! {
///     polydat::dsl::registry::FactoryRegistration { factory: &FACTORY }
/// }
/// ```
///
/// The registry lists the factory's signatures and builds every call to
/// one of them through its `build`, on every engine.
pub struct FactoryRegistration {
    /// The factory.
    pub factory: &'static dyn NodeFactory,
}

inventory::collect!(FactoryRegistration);

/// Every factory linked into the registry: each node module's
/// [`NodeRegistration`], then each host [`FactoryRegistration`]. A
/// function name belongs to the first factory whose signatures list it.
pub fn factories() -> impl Iterator<Item = &'static dyn NodeFactory> {
    inventory::iter::<NodeRegistration>
        .into_iter()
        .map(|reg| reg as &'static dyn NodeFactory)
        .chain(
            inventory::iter::<FactoryRegistration>
                .into_iter()
                .map(|reg| reg.factory),
        )
}

/// A function name that stands for one node per argument type, as
/// `vec_dot` stands for `vec_dot_f32` and `vec_dot_f64`
/// (polydat_grammar.md §18.1, `T-Call-OverloadResolve`).
///
/// A call to the name builds the form that agrees with the most of the
/// call's wire types, position by position, and the earliest such form
/// when several agree with as many: the form matching every type when
/// one does. The assembler's adapters then reach the form's other
/// ports, as they reach any node's. So `vec_scale(v, 2)` over a
/// `vec_f64` builds the f64 form and widens the integer, rather than
/// narrowing the vector to reach a form that takes neither operand as
/// given. The first form is the one the name meant before it had others,
/// so operands no form agrees with build it, and a program that
/// compiled against the name still does. A node module submits one at
/// link time:
///
/// ```ignore
/// polydat::inventory::submit! {
///     polydat::dsl::registry::Overload {
///         name: "vec_dot",
///         forms: &[
///             OverloadForm { wires: &[PortType::VecF32, PortType::VecF32], func: "vec_dot_f32" },
///             OverloadForm { wires: &[PortType::VecF64, PortType::VecF64], func: "vec_dot_f64" },
///         ],
///     }
/// }
/// ```
///
/// Every form takes the same parameters in the same order, so the name's
/// parameters are those of any form; only the wires' types differ.
pub struct Overload {
    /// The name programs call.
    pub name: &'static str,
    /// The forms the name resolves to, the default first.
    pub forms: &'static [OverloadForm],
}

/// One form of an [`Overload`]: the registered function a call builds
/// when its wires have these types.
pub struct OverloadForm {
    /// The port types of the form's wire inputs, in order.
    pub wires: &'static [crate::ast::PortType],
    /// The registered function this form builds.
    pub func: &'static str,
}

inventory::collect!(Overload);

impl Overload {
    /// The function a call whose wires have `wire_types` builds: the
    /// form agreeing with the most of them, the earliest among equals.
    pub fn resolve(&self, wire_types: &[crate::ast::PortType]) -> &'static str {
        let agreement = |form: &OverloadForm| {
            form.wires
                .iter()
                .zip(wire_types)
                .filter(|(declared, given)| declared == given)
                .count()
        };
        let mut best = &self.forms[0];
        for form in &self.forms[1..] {
            if agreement(form) > agreement(best) {
                best = form;
            }
        }
        best.func
    }

    /// The signature of the first form, which a call to the name is
    /// checked against before its wire types are known: every form
    /// shares its parameters. Its `name` is the form's, such as
    /// `vec_dot_f32`, not the overload's. `None` only when the first
    /// form names no registered function.
    pub fn signature(&self) -> Option<&'static FuncSig> {
        lookup_registered(self.forms[0].func)
    }

    /// The category the name is listed under: its first form's.
    pub fn category(&self) -> Option<FuncCategory> {
        self.signature().map(|sig| sig.category)
    }
}

/// Every overloaded name linked into the registry.
pub fn overloads() -> impl Iterator<Item = &'static Overload> {
    inventory::iter::<Overload>.into_iter()
}

/// The overloaded name `name`, if it is one.
pub fn overload(name: &str) -> Option<&'static Overload> {
    overloads().find(|o| o.name == name)
}

/// The function a call to `name` with wires of `wire_types` builds:
/// the resolved form when `name` is an [`Overload`], `name` itself
/// otherwise.
pub fn resolve_call<'a>(name: &'a str, wire_types: &[crate::ast::PortType]) -> &'a str {
    overload(name).map_or(name, |o| o.resolve(wire_types))
}

/// The signature a call to `name` is checked against before its wire
/// types are known; the same as [`lookup`].
#[deprecated(
    since = "0.8.1",
    note = "`lookup` answers for overloaded names; use it"
)]
pub fn lookup_call(name: &str) -> Option<&'static FuncSig> {
    lookup(name)
}

/// Every overloaded name, grouped by its [`Overload::category`] in
/// display order and sorted by name within a category: the names
/// [`by_category`] leaves out, since it lists registered functions.
/// A listing of what a program can call shows both.
///
/// An overload whose first form names no registered function has no
/// category and is left out; nothing can call it either. Those are
/// registration errors, and [`unlisted_overloads`] names them.
pub fn overloads_by_category() -> Vec<(FuncCategory, Vec<&'static Overload>)> {
    let mut groups: std::collections::HashMap<FuncCategory, Vec<&'static Overload>> =
        std::collections::HashMap::new();
    for o in overloads() {
        if let Some(category) = o.category() {
            groups.entry(category).or_default().push(o);
        }
    }
    FuncCategory::display_order()
        .iter()
        .filter_map(|cat| {
            groups.remove(cat).map(|mut names| {
                names.sort_unstable_by_key(|o| o.name);
                (*cat, names)
            })
        })
        .collect()
}

/// Every overload whose first form names no registered function,
/// sorted by name: a registration error, such as a misspelled form or
/// a form whose module is not linked. [`lookup`] answers `None` for
/// these names and [`overloads_by_category`] leaves them out, so a
/// host checks that this is empty, in a test or at startup, to find
/// its own broken registrations.
pub fn unlisted_overloads() -> Vec<&'static Overload> {
    let mut unlisted: Vec<_> = overloads().filter(|o| o.signature().is_none()).collect();
    unlisted.sort_unstable_by_key(|o| o.name);
    unlisted
}

/// Register a node module's signatures and builder with the Polydat runtime.
///
/// Place this call at module scope in each node module. The inventory crate
/// arranges for the registration to run before `main` so that `registry()`
/// and `build_node()` see all entries.
///
/// Two forms:
///
/// - `register_nodes!(signatures, build_node)` — no assembly-time
///   validation. The builder is responsible for handling any bad
///   input itself (usually by trusting the caller or panicking).
/// - `register_nodes!(signatures, build_node, validate_node)` — the
///   factory calls `validate_node(name, consts)` before `build_node`.
///   Use this to declare [`ConstConstraint`]-style checks so
///   constructors can stay infallible.
///
/// [`ConstConstraint`]: crate::dsl::const_constraints::ConstConstraint
#[macro_export]
macro_rules! register_nodes {
    ($sigs:expr, $builder:expr) => {
        inventory::submit! {
            $crate::dsl::registry::NodeRegistration {
                signatures: $sigs,
                build: $builder,
                validate: None,
            }
        }
    };
    ($sigs:expr, $builder:expr, $validator:expr) => {
        inventory::submit! {
            $crate::dsl::registry::NodeRegistration {
                signatures: $sigs,
                build: $builder,
                validate: Some($validator),
            }
        }
    };
}

/// Functional category for a Polydat node function.
///
/// Every native node and stdlib module belongs to exactly one category.
/// Categories group the registry for listings (`by_category`) and provide
/// semantic organization for documentation and discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FuncCategory {
    /// Core deterministic hashing.
    Hashing,
    /// Integer arithmetic with constant parameters.
    Arithmetic,
    /// Comparison and selection: ==, !=, <, >, <=, >=, if(...).
    /// Comparison nodes return u64 truth values (0 or 1); select
    /// nodes pick between two operand values based on a u64 cond.
    Comparison,
    /// Variadic N-ary operations (sum, product, min, max).
    Variadic,
    /// Type conversions between u64, f64, String, etc.
    Conversions,
    /// Statistical distribution LUT builders and samplers.
    Distributions,
    /// Date and time generation and decomposition.
    Datetime,
    /// HTML, URL, hex, base64 encoding/decoding.
    Encoding,
    /// Linear interpolation, range mapping, quantization.
    Interpolation,
    /// Trigonometric and mathematical functions (sin, cos, sqrt, etc.).
    Math,
    /// Probability modeling: coins, selection, conditionals.
    Probability,
    /// Weighted categorical selection.
    Weighted,
    /// Printf-style and structured string formatting.
    Formatting,
    /// String generation: combinations, number words.
    String,
    /// JSON construction, serialization, merging.
    Json,
    /// Byte buffer construction and manipulation.
    ByteBuffers,
    /// Cryptographic and non-cryptographic digests.
    Digest,
    /// Coherent noise: Perlin, simplex.
    Noise,
    /// Regular expression matching and substitution.
    Regex,
    /// Bijective permutations and shuffles.
    Permutation,
    /// Real-world data: names, places, codes.
    RealData,
    /// Non-deterministic context: wall clock, counters.
    Context,
    /// Debugging and introspection.
    Diagnostic,
    /// File-based data access: CSV, JSONL, text files.
    Data,
}

impl FuncCategory {
    /// Display name for the category (used in describe output).
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Hashing => "Hashing",
            Self::Arithmetic => "Arithmetic",
            Self::Comparison => "Comparison",
            Self::Variadic => "Variadic",
            Self::Conversions => "Conversions",
            Self::Distributions => "Distributions",
            Self::Datetime => "Datetime",
            Self::Encoding => "Encoding",
            Self::Interpolation => "Interpolation",
            Self::Math => "Math",
            Self::Probability => "Probability",
            Self::Weighted => "Weighted",
            Self::Formatting => "Formatting",
            Self::String => "String",
            Self::Json => "JSON",
            Self::ByteBuffers => "Byte Buffers",
            Self::Digest => "Digest",
            Self::Noise => "Noise",
            Self::Regex => "Regex",
            Self::Permutation => "Permutation",
            Self::RealData => "Real Data",
            Self::Context => "Context",
            Self::Diagnostic => "Diagnostic",
            Self::Data => "Data",
        }
    }

    /// Parse a category name from a string (case-insensitive): the
    /// text of a stdlib file's `// @category: Name` line. No compile
    /// path consumes it.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "hashing" => Some(Self::Hashing),
            "arithmetic" => Some(Self::Arithmetic),
            "comparison" | "compare" => Some(Self::Comparison),
            "variadic" => Some(Self::Variadic),
            "conversions" | "conversion" => Some(Self::Conversions),
            "distributions" | "distribution" => Some(Self::Distributions),
            "datetime" | "date" | "time" => Some(Self::Datetime),
            "encoding" => Some(Self::Encoding),
            "interpolation" | "lerp" => Some(Self::Interpolation),
            "math" | "trig" | "trigonometry" => Some(Self::Math),
            "probability" => Some(Self::Probability),
            "weighted" => Some(Self::Weighted),
            "formatting" | "format" | "printf" => Some(Self::Formatting),
            "string" | "strings" => Some(Self::String),
            "json" => Some(Self::Json),
            "byte buffers" | "bytebuffers" | "bytes" => Some(Self::ByteBuffers),
            "digest" => Some(Self::Digest),
            "noise" => Some(Self::Noise),
            "regex" => Some(Self::Regex),
            "permutation" | "shuffle" => Some(Self::Permutation),
            "real data" | "realdata" | "realer" => Some(Self::RealData),
            "context" => Some(Self::Context),
            "diagnostic" | "diagnostics" | "debug" => Some(Self::Diagnostic),
            "data" | "datafile" | "csv" | "jsonl" => Some(Self::Data),
            _ => None,
        }
    }

    /// Canonical ordering for display (same order as the enum definition).
    pub fn display_order() -> &'static [Self] {
        &[
            Self::Hashing,
            Self::Arithmetic,
            Self::Comparison,
            Self::Variadic,
            Self::Conversions,
            Self::Distributions,
            Self::Datetime,
            Self::Encoding,
            Self::Interpolation,
            Self::Math,
            Self::Probability,
            Self::Weighted,
            Self::Formatting,
            Self::String,
            Self::Json,
            Self::ByteBuffers,
            Self::Digest,
            Self::Noise,
            Self::Regex,
            Self::Permutation,
            Self::RealData,
            Self::Context,
            Self::Diagnostic,
            Self::Data,
        ]
    }
}

// ---------------------------------------------------------------------------
// Unified parameter specification (library_catalog.md, "Shapes")
// ---------------------------------------------------------------------------

use crate::ast::SlotType;

/// Describes one parameter in a function's call signature.
///
/// A "slot template" — the type-level version of a `Slot` without
/// a concrete value. Parameters are listed in positional order
/// matching the DSL syntax.
///
/// A parameter is built with [`ParamSpec::new`] or [`ParamSpec::wire`]
/// and the `with_` methods, all `const`, so a signature can live in a
/// `static`. The struct is `#[non_exhaustive]`: a field added later is
/// not a breaking change to a host that builds one.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ParamSpec {
    /// Parameter name (for error messages and describe output).
    pub name: &'static str,
    /// Wire or constant, and if constant, what type.
    pub slot_type: SlotType,
    /// Whether this parameter must be provided.
    pub required: bool,
    /// Example value for this parameter, as program text, used for
    /// probing compile level and for documentation. Wire params use
    /// `"cycle"`; a const param with a declared default uses that
    /// default, which passes validation; a const param without one is
    /// empty, since the signature offers no value to show.
    pub example: &'static str,
    /// Optional assembly-time validation rule. The factory enforces this before
    /// `build_node` so node constructors can stay infallible and
    /// branch-free at runtime. `None` = no constraint declared
    /// (default for wires and unconstrained constants).
    pub constraint: Option<crate::dsl::const_constraints::ConstConstraint>,
}

impl ParamSpec {
    /// A required parameter `name` of the slot type `slot_type`, with no
    /// constraint. Its example is `cycle` for a wire, which every
    /// program can pass, and empty for a constant, which has none to
    /// offer until [`Self::with_example`] gives one.
    pub const fn new(name: &'static str, slot_type: SlotType) -> Self {
        Self {
            name,
            slot_type,
            required: true,
            example: if matches!(slot_type, SlotType::Wire) {
                "cycle"
            } else {
                ""
            },
            constraint: None,
        }
    }

    /// A required wire parameter `name`.
    pub const fn wire(name: &'static str) -> Self {
        Self::new(name, SlotType::Wire)
    }

    /// The parameter, but one a call may leave out: a constant with a
    /// default, or a variadic tail that may be empty.
    pub const fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    /// The parameter with `example` as its example value, written as
    /// program text.
    pub const fn with_example(mut self, example: &'static str) -> Self {
        self.example = example;
        self
    }

    /// The parameter with the constraint `c`, which the factory checks
    /// before it builds the node.
    pub const fn with_constraint(
        mut self,
        c: crate::dsl::const_constraints::ConstConstraint,
    ) -> Self {
        self.constraint = Some(c);
        self
    }
}

/// Arity specification for a function signature.
///
/// Describes which parts of the parameter list are fixed vs repeatable.
#[derive(Debug, Clone, Default)]
pub enum Arity {
    /// Exactly the parameters declared in `params`.
    #[default]
    Fixed,
    /// Trailing wire parameters repeat (sum, product, min, max).
    VariadicWires {
        /// The fewest trailing wires allowed.
        min_wires: usize,
    },
    /// Trailing constant parameters repeat (mixed_radix).
    VariadicConsts {
        /// The fewest trailing constants allowed.
        min_consts: usize,
    },
    /// A repeating group of slot types (weighted_sum).
    VariadicGroup {
        /// The slot types of one repetition, in order.
        group: &'static [SlotType],
        /// The fewest repetitions allowed.
        min_repeats: usize,
    },
}

/// Output-type contract for a registered function.
///
/// Most nodes have fixed port types declared by their constructor's
/// `NodeMeta`. Some — `log_info`, `identity`, anything documented
/// as "pass-through" — produce an output whose type matches one
/// of their inputs. Declaring this here makes the contract visible
/// to the assembler, the build-node dispatch path, registry
/// listings, and any future static analysis, instead of being
/// buried inside an `eval` that silently passes values through a
/// wire whose declared type lies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    /// Output types are whatever the constructor's `NodeMeta`
    /// declares — independent of input wire types. The default;
    /// covers the vast majority of nodes (`hash`, `regex_match`,
    /// `mod`, …) whose I/O contract is type-fixed.
    Fixed,
    /// The function's single output port has the same type as the
    /// input wire at the given index. The build dispatch resolves
    /// the wire's type from the assembler and the module's
    /// `build_node` reads it from the supplied `wire_types` slice
    /// to construct a port-typed node. Used by pass-through
    /// nodes (`log_info`, `log_debug`, …).
    SameAsInput(usize),
}

/// Description of a registered function's signature.
///
/// A signature is built with [`FuncSig::new`] and the `with_` methods,
/// all `const`, so a module's signatures can live in a `static`:
///
/// ```
/// use polydat_core::ast::{PortType, SlotType};
/// use polydat_core::dsl::registry::{FuncCategory, FuncSig, ParamSpec};
///
/// static SIGS: &[FuncSig] = &[FuncSig::new(
///     "counted_mod",
///     FuncCategory::Arithmetic,
///     &[ParamSpec::wire("x"), ParamSpec::new("m", SlotType::ConstU64).with_example("7")],
/// )
/// .with_description("mod, built by a host factory")
/// .with_output_port(PortType::U64)];
/// # assert_eq!(SIGS[0].outputs, 1);
/// ```
///
/// The struct is `#[non_exhaustive]`: a field added later is not a
/// breaking change to a host that builds one, since the new field
/// arrives with its default.
#[non_exhaustive]
pub struct FuncSig {
    /// Function name as used in the DSL.
    pub name: &'static str,
    /// Functional category.
    pub category: FuncCategory,
    /// Number of output ports (0 = dynamic, determined at compile time).
    pub outputs: usize,
    /// Short description for help/error messages.
    pub description: &'static str,
    /// Detailed help text: theory, usage examples, parameter meanings.
    /// Long-form help text for listings and documentation.
    pub help: &'static str,
    /// For variadic functions: the identity element for zero inputs.
    pub identity: Option<u64>,
    /// Factory for variadic nodes: takes wire count, returns node.
    pub variadic_ctor: Option<fn(usize) -> Box<dyn crate::ast::PolydatNode>>,
    /// Positional parameter list: wires and constants in call order.
    pub params: &'static [ParamSpec],
    /// Arity specification.
    pub arity: Arity,
    /// Input commutativity for this function.
    pub commutativity: crate::ast::Commutativity,
    /// Optional resolver hint for `Handle`-typed input ports. When
    /// the binding compiler emits this function and a `Handle`
    /// input is wired to a `Str`-producing source, it splices in
    /// the named resolver to convert the string into a handle. This
    /// is the "string-conversion node insertion" mechanism of
    /// source-string call-site sugar. `None` means no
    /// auto-promotion — the caller must pass a `Handle` directly.
    pub default_resolver: Option<DefaultResolver>,
    /// Output-type contract — `Fixed` for the vast majority of
    /// nodes; `SameAsInput(idx)` for type-polymorphic pass-throughs
    /// (e.g. `log_info` whose output type tracks its sole input).
    pub output_type: OutputType,
    /// Concrete port type of the single output, when statically
    /// known (`#[polydat_node]` emits it from the return type's
    /// `Wire::PORT`). `None` for tuple/dynamic/polymorphic outputs
    /// and for hand registrations that don't declare one. The DSL
    /// type inference (`binding::infer_expr_type`) reads this
    /// FIRST — the name-prefix heuristic is only the fallback —
    /// so call-expression operand typing flows from the symbol
    /// registry, not from a hand-maintained list.
    pub output_port: Option<crate::ast::PortType>,
}

/// Auto-resolver kind attached to handle-taking functions. Tells the
/// binding compiler which resolver to splice in when a string source
/// is wired to a handle input port.
#[derive(Debug, Clone, Copy)]
pub enum DefaultResolver {
    /// Insert `dataset_open(<source_wire>, "<facet>")` between the
    /// string source and the handle input.
    Facet(&'static str),
    /// Insert `dataset_group_open(<source_wire>)` between the string
    /// source and the handle input.
    Group,
}

impl FuncSig {
    /// The signature of `name` in `category` taking `params`: one
    /// output of a fixed type not stated here, a fixed arity, positional
    /// inputs, no description or help, and none of the optional parts.
    /// The `with_` methods change each of these.
    pub const fn new(
        name: &'static str,
        category: FuncCategory,
        params: &'static [ParamSpec],
    ) -> Self {
        Self {
            name,
            category,
            outputs: 1,
            description: "",
            help: "",
            identity: None,
            variadic_ctor: None,
            params,
            arity: Arity::Fixed,
            commutativity: crate::ast::Commutativity::Positional,
            default_resolver: None,
            output_type: OutputType::Fixed,
            output_port: None,
        }
    }

    /// The signature with `outputs` output ports (0 for a count fixed at
    /// compile time).
    pub const fn with_outputs(mut self, outputs: usize) -> Self {
        self.outputs = outputs;
        self
    }

    /// The signature with the short description `description`.
    pub const fn with_description(mut self, description: &'static str) -> Self {
        self.description = description;
        self
    }

    /// The signature with the long-form help text `help`.
    pub const fn with_help(mut self, help: &'static str) -> Self {
        self.help = help;
        self
    }

    /// The signature of a variadic function whose identity element, for
    /// zero inputs, is `identity`.
    pub const fn with_identity(mut self, identity: u64) -> Self {
        self.identity = Some(identity);
        self
    }

    /// The signature of a variadic function whose node for a given
    /// wire count `ctor` builds.
    pub const fn with_variadic_ctor(
        mut self,
        ctor: fn(usize) -> Box<dyn crate::ast::PolydatNode>,
    ) -> Self {
        self.variadic_ctor = Some(ctor);
        self
    }

    /// The signature with the arity `arity`.
    pub const fn with_arity(mut self, arity: Arity) -> Self {
        self.arity = arity;
        self
    }

    /// The signature with the input commutativity `commutativity`.
    pub const fn with_commutativity(mut self, commutativity: crate::ast::Commutativity) -> Self {
        // `Commutativity::Groups` owns a `Vec`, so a `const fn` cannot
        // drop the value it replaces and forgets it instead. A static
        // signature cannot hold a `Groups` (no `Vec` is built in a
        // `const` context), so what is forgotten there owns nothing;
        // a runtime caller replacing a `Groups` leaks its lists.
        let replaced = std::mem::replace(&mut self.commutativity, commutativity);
        std::mem::forget(replaced);
        self
    }

    /// The signature whose `Handle` inputs, wired from a `Str`, are
    /// resolved by `resolver`.
    pub const fn with_default_resolver(mut self, resolver: DefaultResolver) -> Self {
        self.default_resolver = Some(resolver);
        self
    }

    /// The signature with the output-type contract `output_type`.
    pub const fn with_output_type(mut self, output_type: OutputType) -> Self {
        self.output_type = output_type;
        self
    }

    /// The signature whose single output has the port type `port`.
    pub const fn with_output_port(mut self, port: crate::ast::PortType) -> Self {
        self.output_port = Some(port);
        self
    }

    /// Number of wire inputs in the fixed parameter list.
    pub fn wire_input_count(&self) -> usize {
        self.params.iter().filter(|p| p.slot_type.is_wire()).count()
    }

    /// Whether this function accepts variadic arguments.
    pub fn is_variadic(&self) -> bool {
        !matches!(self.arity, Arity::Fixed)
    }

    /// Constant parameter names and whether they're required.
    pub fn const_param_info(&self) -> Vec<(&'static str, bool)> {
        self.params
            .iter()
            .filter(|p| p.slot_type.is_const())
            .map(|p| (p.name, p.required))
            .collect()
    }
}

impl std::fmt::Debug for FuncSig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuncSig")
            .field("name", &self.name)
            .field("category", &self.category)
            .field("params", &self.params)
            .field("arity", &self.arity)
            .finish()
    }
}

impl Clone for FuncSig {
    fn clone(&self) -> Self {
        Self {
            name: self.name,
            category: self.category,
            outputs: self.outputs,
            description: self.description,
            help: self.help,
            identity: self.identity,
            variadic_ctor: self.variadic_ctor,
            params: self.params,
            arity: self.arity.clone(),
            commutativity: self.commutativity.clone(),
            default_resolver: self.default_resolver,
            output_type: self.output_type,
            output_port: self.output_port,
        }
    }
}

/// Return the full registry of known functions: the signatures of every
/// linked factory ([`factories`]).
pub fn registry() -> Vec<FuncSig> {
    let mut funcs = Vec::new();
    for factory in factories() {
        funcs.extend_from_slice(factory.signatures());
    }
    funcs
}

/// Return functions grouped by category in display order.
pub fn by_category() -> Vec<(FuncCategory, Vec<FuncSig>)> {
    let reg = registry();
    let mut groups: std::collections::HashMap<FuncCategory, Vec<FuncSig>> =
        std::collections::HashMap::new();
    for sig in reg {
        groups.entry(sig.category).or_default().push(sig);
    }
    FuncCategory::display_order()
        .iter()
        .filter_map(|cat| groups.remove(cat).map(|funcs| (*cat, funcs)))
        .collect()
}

/// Find the closest function name to a misspelling, among the
/// registered functions and the [`Overload`] names.
pub fn suggest_function(name: &str) -> Option<&'static str> {
    let reg = registry();
    let names = reg
        .iter()
        .map(|sig| sig.name)
        .chain(overloads().map(|o| o.name));
    let mut best: Option<(&str, usize)> = None;
    for candidate in names {
        let dist = edit_distance(name, candidate);
        if dist <= 3 && (best.is_none() || dist < best.unwrap().1) {
            best = Some((candidate, dist));
        }
    }
    best.map(|(name, _)| name)
}

/// The signature of what a program calls by `name`: the registered
/// function `name`, or, for an [`Overload`] such as `vec_dot`, its
/// first form's ([`Overload::signature`]), whose parameters every form
/// shares. `None` means no call to `name` can build. For an overload
/// the signature's `name` is the form's, such as `vec_dot_f32`;
/// [`resolve_call`] names the form a call with given wire types builds,
/// and [`lookup_registered`] answers only for registered functions.
/// Every factory is `'static`, so its signatures are too, and no
/// allocation is needed.
pub fn lookup(name: &str) -> Option<&'static FuncSig> {
    lookup_registered(name).or_else(|| overload(name).and_then(Overload::signature))
}

/// Find a registered function by its exact name: the signature of the
/// factory that builds it. Unlike [`lookup`], it answers `None` for an
/// overloaded name, which no factory builds itself.
pub fn lookup_registered(name: &str) -> Option<&'static FuncSig> {
    factories().find_map(|factory| factory.signatures().iter().find(|sig| sig.name == name))
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut matrix = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for (i, row) in matrix.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in matrix[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            matrix[i][j] = (matrix[i - 1][j] + 1)
                .min(matrix[i][j - 1] + 1)
                .min(matrix[i - 1][j - 1] + cost);
        }
    }
    matrix[a.len()][b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggest_no_match() {
        assert_eq!(suggest_function("zzzzzzzzz"), None);
    }

    #[test]
    fn lookup_missing() {
        assert!(lookup("nonexistent").is_none());
    }

    #[test]
    fn every_function_has_category() {
        let reg = registry();
        for sig in &reg {
            // Just verify the category display name is non-empty
            assert!(
                !sig.category.display_name().is_empty(),
                "function '{}' has no category display name",
                sig.name
            );
        }
    }

    #[test]
    fn lookup_answers_for_an_overloaded_name_and_lookup_registered_does_not() {
        for o in overloads().filter(|o| lookup_registered(o.forms[0].func).is_some()) {
            let sig = lookup(o.name).unwrap_or_else(|| panic!("{} is not callable", o.name));
            assert_eq!(sig.name, o.forms[0].func, "{}", o.name);
            assert!(lookup_registered(o.name).is_none(), "{}", o.name);
        }
    }

    #[test]
    fn overloads_by_category_lists_every_overload_but_the_unlisted_once() {
        let listed: usize = overloads_by_category().iter().map(|(_, o)| o.len()).sum();
        assert_eq!(listed + unlisted_overloads().len(), overloads().count());
    }

    // An overload whose first form is not registered, as a host's
    // misspelled registration would be.
    inventory::submit! {
        Overload {
            name: "registry_test_misspelled_overload",
            forms: &[OverloadForm {
                wires: &[crate::ast::PortType::U64],
                func: "registry_test_no_such_function",
            }],
        }
    }

    #[test]
    fn an_overload_of_no_registered_function_is_unlisted_and_uncallable() {
        let name = "registry_test_misspelled_overload";
        let unlisted: Vec<_> = unlisted_overloads().iter().map(|o| o.name).collect();
        assert_eq!(unlisted, [name]);
        assert!(lookup(name).is_none());
        assert!(
            overloads_by_category()
                .iter()
                .all(|(_, names)| names.iter().all(|o| o.name != name))
        );
    }

    #[test]
    fn by_category_covers_all() {
        let grouped = by_category();
        let total: usize = grouped.iter().map(|(_, funcs)| funcs.len()).sum();
        let reg = registry();
        assert_eq!(
            total,
            reg.len(),
            "by_category must cover all registered functions"
        );
    }

    #[test]
    fn category_parse_roundtrip() {
        for cat in FuncCategory::display_order() {
            let name = cat.display_name();
            let parsed = FuncCategory::parse(name);
            assert_eq!(parsed, Some(*cat), "failed to parse category '{name}'");
        }
    }

    #[test]
    fn registry_has_entries() {
        let reg = registry();
        assert!(reg.len() > 50, "registry should have 50+ functions");
    }

    // --- Unified param model tests ---

    #[test]
    fn printf_has_const_str_param() {
        // printf is a `#[polydat_node]` with a `Const<&str> format`
        // arg + `&[Value] parts` variadic. The macro lists both in
        // `params` (the variadic arg appears as a SlotType::Wire
        // entry whose count is governed by the node's
        // `Arity::VariadicWires`). The load-bearing assertion is that
        // the FIRST param is the format ConstStr and the arity is
        // variadic.
        let sig = lookup("printf").unwrap();
        assert!(
            !sig.params.is_empty(),
            "printf must have at least the format param"
        );
        assert!(matches!(sig.params[0].slot_type, SlotType::ConstStr));
        assert!(matches!(sig.arity, Arity::VariadicWires { .. }));
    }
}
