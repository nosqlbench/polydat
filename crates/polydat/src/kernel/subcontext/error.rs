// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic types: [`SourceContext`] and [`ContractViolation`].

use crate::ast::PortType;

use super::name::ChildName;

/// Diagnostic context attached to a [`super::ScopeModule`] —
/// where the module's source came from. Used in error messages
/// when a contract violation surfaces at spawn or finalize.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceContext {
    /// Logical label — workload phase / op-template name / SRD
    /// reference. Free-form; appears verbatim in diagnostics.
    pub label: String,
    /// Source file path, if applicable.
    pub file: Option<String>,
    /// Line range `(start, end)` if known.
    pub line_range: Option<(usize, usize)>,
}

impl SourceContext {
    /// A context with a label and no file or lines.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            file: None,
            line_range: None,
        }
    }

    /// The context of a phase, labelled `phase:<name>`.
    pub fn for_phase(name: &str) -> Self {
        Self::new(format!("phase:{name}"))
    }

    /// The context of an op, labelled `op:<name>`.
    pub fn for_op(name: &str) -> Self {
        Self::new(format!("op:{name}"))
    }

    /// The same context with its source file.
    pub fn with_file(mut self, file: impl Into<String>) -> Self {
        self.file = Some(file.into());
        self
    }

    /// The same context with its line range.
    pub fn with_lines(mut self, start: usize, end: usize) -> Self {
        self.line_range = Some((start, end));
        self
    }

    /// Render as a single line for error messages.
    pub fn display(&self) -> String {
        let mut s = self.label.clone();
        if let Some(f) = &self.file {
            s.push_str(&format!(" ({f}"));
            if let Some((a, b)) = self.line_range {
                s.push_str(&format!(":{a}-{b}"));
            }
            s.push(')');
        } else if let Some((a, b)) = self.line_range {
            s.push_str(&format!(" ({a}-{b})"));
        }
        s
    }
}

/// Contract violation surfaced at finalize or spawn.
///
/// Variants per SRD-67 §"Cross-binding rules" plus the umbrella
/// [`Self::Compile`] for errors raised by the Polydat compiler when
/// the body fragment is converted into a program (typically an
/// unbound identifier in the body, which the compiler catches
/// after `finalize`'s name-closure check on declared imports).
///
/// The active set is the design doc's §7 error contract:
/// [`Self::UnboundImport`], [`Self::FinalShadow`],
/// [`Self::DuplicateChild`], [`Self::Compile`], and
/// [`Self::StrictNonePropagation`]. [`Self::Type`],
/// [`Self::Modifier`], and [`Self::Phase2WriteThrough`] are
/// retained as compatibility surface and are not emitted by the
/// builder (design doc §2.2).
#[derive(Debug, Clone)]
pub enum ContractViolation {
    /// Rule 1 — Import resolution: an artifact import has no
    /// matching parent export.
    UnboundImport {
        /// The import's name.
        import: String,
        /// Where the import is declared.
        site: SourceContext,
    },
    /// Rule 1 — Type mismatch on import.
    Type {
        /// The import's name.
        import: String,
        /// The type the import requires.
        required: PortType,
        /// The type the parent exports.
        parent_export: PortType,
        /// Where the import is declared.
        site: SourceContext,
    },
    /// Rule 1 — Modifier mismatch (e.g. shared import against a
    /// non-shared parent export).
    Modifier {
        /// The import's name.
        import: String,
        /// What differs.
        detail: String,
        /// Where the import is declared.
        site: SourceContext,
    },
    /// Rule 2 — Final-shadow on export: a child can't redefine
    /// an immutable parent export.
    FinalShadow {
        /// The export shadowed.
        export: String,
        /// Where the child redefines it.
        site: SourceContext,
    },
    /// Rule 2 — Shared write-through rewrite was required but
    /// could not be performed.
    ///
    /// Never emitted. The rewrite is implemented in
    /// [`super::SubcontextBuilder::finalize`] (design doc §3.1),
    /// which reports a rewrite that fails to produce its input
    /// slot or synthetic output as [`Self::Compile`]; success is
    /// visible as [`super::ScopeModule::write_throughs`]. The
    /// variant is kept as compatibility surface for callers that
    /// pattern-match on it (design doc §2.2).
    Phase2WriteThrough {
        /// The export the rewrite targeted.
        export: String,
        /// Where the export is declared.
        site: SourceContext,
        /// What the rewrite could not do.
        note: &'static str,
    },
    /// Named-child registry: a duplicate spawn under the same
    /// name (SRD-67 §"Named-child registry"). Reports both spawn
    /// sites.
    DuplicateChild {
        /// The child's name.
        name: ChildName,
        /// Boxed: this is the only variant carrying two
        /// `SourceContext`s — boxing one keeps the whole enum (and
        /// every `Result<_, ContractViolation>`) small.
        prior_site: Box<SourceContext>,
        /// The second spawn site.
        this_site: SourceContext,
    },
    /// Polydat compile-time error — the body failed to compile (most
    /// commonly: unbound identifier; corresponds to Rule 1's
    /// closure-binding economy detecting a free identifier with
    /// no matching import).
    Compile(String),
    /// L2.f strict-mode hardening: an intermediate-layer
    /// `const` binding's Plan B materialisation yielded
    /// `Value::None`, and the build was running with strict
    /// mode enabled. Per composition_substrate.md L2.f's
    /// strict-mode hardening clause, silent fall-through to
    /// the outer scope's binding is rejected in strict mode —
    /// the author must either ensure the const yields a
    /// defined value or remove the binding and declare an
    /// explicit `extern <name>` if fall-through to outer was
    /// intended. The `bindings` field carries every const
    /// output that materialised to None.
    StrictNonePropagation {
        /// Every const output that materialised to `None`.
        bindings: Vec<String>,
        /// Where the bindings are declared.
        site: SourceContext,
    },
}

impl std::fmt::Display for ContractViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnboundImport { import, site } => write!(
                f,
                "unbound import `{import}` (parent does not export it) at {}",
                site.display()
            ),
            Self::Type {
                import,
                required,
                parent_export,
                site,
            } => write!(
                f,
                "type mismatch on import `{import}`: required {required:?}, parent exports {parent_export:?} at {}",
                site.display()
            ),
            Self::Modifier {
                import,
                detail,
                site,
            } => write!(
                f,
                "modifier mismatch on import `{import}`: {detail} at {}",
                site.display()
            ),
            Self::FinalShadow { export, site } => write!(
                f,
                "child export `{export}` shadows parent's `final` export at {}",
                site.display()
            ),
            Self::Phase2WriteThrough { export, site, note } => write!(
                f,
                "write-through rewrite for shared export `{export}` could not be performed at {} — {note}",
                site.display()
            ),
            Self::DuplicateChild {
                name,
                prior_site,
                this_site,
            } => write!(
                f,
                "duplicate spawn of child `{name}`: prior at {}, this at {}",
                prior_site.display(),
                this_site.display()
            ),
            Self::Compile(msg) => write!(f, "compile error: {msg}"),
            Self::StrictNonePropagation { bindings, site } => {
                let names = bindings.join(", ");
                write!(
                    f,
                    "L2.f strict-mode violation: intermediate-layer const \
                    binding(s) [{names}] yielded `Value::None` at scope-init \
                    at {}; strict mode rejects silent fall-through to the \
                    outer scope. Either ensure the binding yields a defined \
                    value, or remove the binding and declare \
                    `extern <name>` explicitly if fall-through to outer was \
                    intended.",
                    site.display()
                )
            }
        }
    }
}

impl std::error::Error for ContractViolation {}
