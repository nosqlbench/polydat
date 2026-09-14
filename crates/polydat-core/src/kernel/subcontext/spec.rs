// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Typed import / export contracts (per SRD-13e §1.2).
//!
//! The specs carry the SRD-13e taxonomy (lifecycle
//! classifications, port types, binding modifiers) as data.
//! What [`crate::kernel::subcontext::SubcontextBuilder::finalize`]
//! enforces from them is fixed by
//! `docs/design/subcontext_construction.md` §2.2: every import
//! name must exist on the parent (`UnboundImport`), a child
//! export may not shadow a parent `const` output (`FinalShadow`),
//! and a child export matching an in-scope shared cell becomes a
//! write-through binding. `port_type` and `classification` are
//! preserved in the public `ScopeContract` but are not compared
//! against a typed parent manifest; the child's input slots and
//! shared-cell writes are protected by the compiler's slot type
//! checks and by `kernel::state::check_write_through_type`.

use crate::ast::PortType;
use crate::dsl::ast::BindingModifier;

/// Lifecycle classification for an import — taxonomically what
/// SRD-13e §1.2 specifies. Drives the spawn-time validation
/// decisions per SRD-67 Rule 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportClassification {
    /// `final X: T` — fold the parent's value into the child at
    /// compile/init time.
    CompileConst,
    /// `extern X: T` — wire to the parent's value via an input
    /// slot. Most common shape.
    Extern,
    /// `shared X: T` — share-cell-attach against a parent's
    /// `shared`-modifier export.
    Shared,
    /// Iteration extern: parent's coordinate / iteration variable
    /// (SRD-67 Rule 4 routes through the parent's coordinate
    /// buffer).
    IterationExtern,
}

/// Lifecycle classification for an export — what kind of
/// downstream contract this export carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportClassification {
    /// Locally-defined output of the body. The default shape.
    Local,
    /// `const` modifier on the body's binding.
    Final,
    /// `shared` modifier on the body's binding — shared cell
    /// available to descendants.
    Shared,
    /// Iteration variable (comprehension coordinate). Routed
    /// through the parent's coord buffer at spawn (Rule 4).
    Coordinate,
    /// `volatile` modifier — excluded from const-fold identity.
    Volatile,
}

/// Typed import declaration: a name the child's body will
/// reference, expecting the parent to export it.
#[derive(Debug, Clone)]
pub struct ImportSpec {
    /// Name as referenced inside the child body.
    pub name: String,
    /// Expected port type. Spawn validates against the parent's
    /// matching export (Rule 1).
    pub port_type: PortType,
    /// Lifecycle classification.
    pub classification: ImportClassification,
}

impl ImportSpec {
    /// An import written to by the host, as an `extern` port.
    pub fn extern_(name: impl Into<String>, port_type: PortType) -> Self {
        Self {
            name: name.into(),
            port_type,
            classification: ImportClassification::Extern,
        }
    }

    /// An import fixed at compile time, from a parent `final` export.
    pub fn final_(name: impl Into<String>, port_type: PortType) -> Self {
        Self {
            name: name.into(),
            port_type,
            classification: ImportClassification::CompileConst,
        }
    }

    /// An import bound to a shared cell.
    pub fn shared(name: impl Into<String>, port_type: PortType) -> Self {
        Self {
            name: name.into(),
            port_type,
            classification: ImportClassification::Shared,
        }
    }

    /// An import rebound per activation of an enclosing iteration.
    pub fn iter_var(name: impl Into<String>, port_type: PortType) -> Self {
        Self {
            name: name.into(),
            port_type,
            classification: ImportClassification::IterationExtern,
        }
    }
}

/// Typed export declaration: a named value the child produces,
/// available to its own descendants.
#[derive(Debug, Clone)]
pub struct ExportSpec {
    /// Name as it appears in the child's body.
    pub name: String,
    /// Port type the child binds.
    pub port_type: PortType,
    /// Modifier (`final` / `shared` / none) — the standard GK
    /// modifier set; spawn uses it to apply Rule 2.
    pub modifier: BindingModifier,
    /// Lifecycle classification.
    pub classification: ExportClassification,
}

impl ExportSpec {
    /// A local export with no modifier.
    pub fn local(name: impl Into<String>, port_type: PortType) -> Self {
        Self {
            name: name.into(),
            port_type,
            modifier: BindingModifier::NONE,
            classification: ExportClassification::Local,
        }
    }

    /// A `final` export: fixed once bound.
    pub fn final_(name: impl Into<String>, port_type: PortType) -> Self {
        Self {
            name: name.into(),
            port_type,
            modifier: BindingModifier::CONST,
            classification: ExportClassification::Final,
        }
    }

    /// A `shared` export: a cell the descendants write through.
    pub fn shared(name: impl Into<String>, port_type: PortType) -> Self {
        Self {
            name: name.into(),
            port_type,
            modifier: BindingModifier::SHARED,
            classification: ExportClassification::Shared,
        }
    }

    /// An export rebound per activation of an enclosing iteration.
    pub fn iter_var(name: impl Into<String>, port_type: PortType) -> Self {
        Self {
            name: name.into(),
            port_type,
            modifier: BindingModifier::NONE,
            classification: ExportClassification::Coordinate,
        }
    }
}
