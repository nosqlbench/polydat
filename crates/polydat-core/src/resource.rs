// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Dependency-inverted resource-accessor bridge.
//!
//! A polydat node sometimes needs a **live, host-owned resource** (the
//! first consumer is a CQL `Session`) addressed by its configuration
//! **fingerprint**. polydat is the dependency floor and does not depend
//! on the host runtime, so the bridge is a type-erased trait,
//! [`ResourceAccessor`], which the host implements over its own store.
//!
//! The accessor belongs to a program tree, not to the process. A
//! [`ResourceScope`] is the tree's slot for it: a host installs an
//! accessor when it compiles (`CompileOptions::resources`) or later, on
//! the kernel it holds ([`crate::Kernel::resources`]). Every program of
//! the tree shares the one scope: `for` bodies, subscopes built under a
//! kernel of the tree, and every kernel created from or forked off one
//! of its programs. A node reaches the scope through the
//! [`BuildContext`](crate::dsl::factory::BuildContext) its factory
//! receives, keeps it, and looks a resource up when it evaluates. Two
//! trees in one process each see their own accessor.
//!
//! A lookup is a synchronous read of what the host has already
//! attached: it never blocks and never connects, and the payload is
//! erased to `Arc<dyn Any + Send + Sync>`, which the consuming node
//! downcasts to its own concrete handle type.

use std::any::Any;
use std::sync::{Arc, OnceLock};

/// Type-erased accessor over the host's live resource store.
///
/// Implemented by the host runtime (nbrs-runtime's resource pool) and
/// installed into a program tree's [`ResourceScope`]. The trait is
/// deliberately minimal and free of host types so polydat stays the
/// dependency floor.
pub trait ResourceAccessor: Send + Sync {
    /// Synchronous lookup of an already-attached resource's accessor
    /// payload by fingerprint `key`. Returns `None` when no live entry
    /// matches that key (never blocks, never connects).
    ///
    /// The `key` is the host's stable rendering of a resource fingerprint;
    /// a single canonical rendering is shared by whoever installs the
    /// payload and whoever looks it up, so the string round-trips exactly.
    fn lookup(&self, key: &str) -> Option<Arc<dyn Any + Send + Sync>>;
}

/// A program tree's slot for its host's [`ResourceAccessor`].
///
/// Cloning a scope yields another handle to the same slot, which is how
/// every program and kernel of one tree shares it. An accessor is
/// installed at most once per tree; a tree with none installed answers
/// every lookup with `None`, which is the norm for a program that runs
/// without a host.
#[derive(Clone, Default)]
pub struct ResourceScope(Arc<OnceLock<Arc<dyn ResourceAccessor>>>);

impl ResourceScope {
    /// A scope with no accessor installed, for a new program tree.
    pub fn new() -> Self {
        Self::default()
    }

    /// A scope with `accessor` already installed.
    pub fn with_accessor(accessor: Arc<dyn ResourceAccessor>) -> Self {
        let scope = Self::new();
        let _ = scope.0.set(accessor);
        scope
    }

    /// Install `accessor` for every program and kernel of this tree.
    /// A tree has one accessor for its life: when one is already
    /// installed the call returns `accessor` back as the error and the
    /// installed one stays.
    pub fn install(
        &self,
        accessor: Arc<dyn ResourceAccessor>,
    ) -> Result<(), Arc<dyn ResourceAccessor>> {
        self.0.set(accessor)
    }

    /// Whether an accessor is installed.
    pub fn is_installed(&self) -> bool {
        self.0.get().is_some()
    }

    /// Look up an already-attached resource's accessor payload by
    /// fingerprint `key` through the installed accessor.
    ///
    /// Returns `None` when no accessor is installed or when no live
    /// entry matches `key`. The lookup is synchronous and never blocks.
    pub fn lookup(&self, key: &str) -> Option<Arc<dyn Any + Send + Sync>> {
        self.0.get()?.lookup(key)
    }

    /// Whether `self` and `other` are handles to one tree's slot.
    pub fn same_scope(&self, other: &ResourceScope) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::fmt::Debug for ResourceScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceScope")
            .field("installed", &self.is_installed())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed(&'static str, u64);

    impl ResourceAccessor for Fixed {
        fn lookup(&self, key: &str) -> Option<Arc<dyn Any + Send + Sync>> {
            (key == self.0).then(|| Arc::new(self.1) as Arc<dyn Any + Send + Sync>)
        }
    }

    #[test]
    fn an_empty_scope_answers_none() {
        assert!(ResourceScope::new().lookup("k").is_none());
    }

    #[test]
    fn clones_share_one_slot_and_install_once() {
        let scope = ResourceScope::new();
        let child = scope.clone();
        assert!(scope.install(Arc::new(Fixed("k", 7))).is_ok());
        let got = child.lookup("k").and_then(|v| v.downcast::<u64>().ok());
        assert_eq!(got.as_deref(), Some(&7));
        assert!(scope.install(Arc::new(Fixed("k", 8))).is_err());
        assert!(child.same_scope(&scope));
        assert!(!ResourceScope::new().same_scope(&scope));
    }
}
