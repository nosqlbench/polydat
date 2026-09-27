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
//! A program compiled on its own has a scope of its own. Binding one of
//! its kernels under a parent ([`crate::kernel::bind_under`]) joins that
//! scope to the parent's ([`ResourceScope::join`]): while the child's
//! scope has no accessor installed, its lookups resolve through the
//! parent's. The link is on the scope, so every kernel of the child's
//! program shares it, and an accessor installed once at the root serves
//! every image bound beneath it.
//!
//! A lookup is a synchronous read of what the host has already
//! attached: it never blocks and never connects, and the payload is
//! erased to `Arc<dyn Any + Send + Sync>`, which the consuming node
//! downcasts to its own concrete handle type.

use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
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
/// installed at most once per scope. A scope with none installed
/// resolves a lookup through the parent it joined
/// ([`Self::join`]), and answers `None` when it joined none, which is
/// the norm for a program that runs without a host.
#[derive(Clone, Default)]
pub struct ResourceScope(Arc<Slot>);

/// The shared state behind every handle to one scope.
#[derive(Default)]
struct Slot {
    accessor: OnceLock<Arc<dyn ResourceAccessor>>,
    parent: OnceLock<Link>,
}

/// A scope's delegation to the parent it joined. A link that lost a
/// race to close a cycle is severed: it stays set, and lookups and
/// joins treat the scope as joined to nothing.
struct Link {
    parent: ResourceScope,
    live: AtomicBool,
}

/// Why [`ResourceScope::join`] left a scope joined to nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScopeJoinError {
    /// Another thread joined scopes at the same moment in a way that,
    /// with this join, would have made a scope delegate to itself. This
    /// join is severed and the scope delegates to nothing.
    Cycle,
}

impl std::fmt::Display for ScopeJoinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScopeJoinError::Cycle => write!(
                f,
                "joining the child's resource scope to its parent's, concurrently with \
                 another join, would have made a scope delegate to itself"
            ),
        }
    }
}

impl std::error::Error for ScopeJoinError {}

impl ResourceScope {
    /// A scope with no accessor installed, for a new program tree.
    pub fn new() -> Self {
        Self::default()
    }

    /// A scope with `accessor` already installed.
    pub fn with_accessor(accessor: Arc<dyn ResourceAccessor>) -> Self {
        let scope = Self::new();
        let _ = scope.0.accessor.set(accessor);
        scope
    }

    /// Install `accessor` for every program and kernel of this tree.
    /// A scope has one accessor for its life: when one is already
    /// installed the call returns `accessor` back as the error and the
    /// installed one stays. An accessor installed here takes precedence
    /// over the parent the scope joined.
    pub fn install(
        &self,
        accessor: Arc<dyn ResourceAccessor>,
    ) -> Result<(), Arc<dyn ResourceAccessor>> {
        self.0.accessor.set(accessor)
    }

    /// Whether an accessor is installed on this scope itself. A scope
    /// that resolves through its parent reports `false`.
    pub fn is_installed(&self) -> bool {
        self.0.accessor.get().is_some()
    }

    /// The parent this scope delegates to, when it joined one.
    pub fn parent(&self) -> Option<&ResourceScope> {
        self.0
            .parent
            .get()
            .filter(|link| link.live.load(Ordering::Acquire))
            .map(|link| &link.parent)
    }

    /// Join this scope to `parent`, so that while no accessor is
    /// installed here, lookups resolve through `parent`. Binding a
    /// kernel under a parent kernel ([`crate::kernel::bind_under`])
    /// joins the child program's scope to the parent's.
    ///
    /// A scope joins at most one parent, the first it is joined to, and
    /// keeps it for its life. The join changes nothing when:
    ///
    /// - this scope has an accessor installed, which it keeps;
    /// - `parent` is this scope, or already resolves through it (this
    ///   scope is `parent`'s ancestor), so joining would make a scope
    ///   delegate to itself;
    /// - this scope already joined a parent, `parent` or another. The
    ///   first join wins: a node holds its program's one scope, so a
    ///   program whose kernels are bound under two trees resolves
    ///   through the first. A host that binds one program under trees
    ///   with different accessors installs one on the program itself.
    ///
    /// Otherwise this scope joins `parent`. The one error is a join
    /// that raced others into a cycle ([`ScopeJoinError::Cycle`]).
    pub fn join(&self, parent: &ResourceScope) -> Result<(), ScopeJoinError> {
        if self.is_installed() || parent.reaches(self) {
            return Ok(());
        }
        let link = Link {
            parent: parent.clone(),
            live: AtomicBool::new(true),
        };
        if self.0.parent.set(link).is_err() {
            return Ok(());
        }
        // A concurrent join elsewhere may have closed a cycle through
        // this link after the check above. The last link a cycle needs
        // sees the whole cycle here, so severing it keeps every scope's
        // chain acyclic.
        if parent.reaches(self)
            && let Some(link) = self.0.parent.get()
        {
            link.live.store(false, Ordering::Release);
            return Err(ScopeJoinError::Cycle);
        }
        Ok(())
    }

    /// Whether `target` is this scope or one it delegates to.
    fn reaches(&self, target: &ResourceScope) -> bool {
        let mut scope = self;
        loop {
            if scope.same_scope(target) {
                return true;
            }
            match scope.parent() {
                Some(parent) => scope = parent,
                None => return false,
            }
        }
    }

    /// Look up an already-attached resource's accessor payload by
    /// fingerprint `key`: through this scope's accessor when one is
    /// installed, and otherwise through the parent it joined.
    ///
    /// Returns `None` when no scope on the way has an accessor or when
    /// the accessor that answers holds no live entry under `key`. The
    /// lookup is synchronous and never blocks.
    pub fn lookup(&self, key: &str) -> Option<Arc<dyn Any + Send + Sync>> {
        let mut scope = self;
        loop {
            if let Some(accessor) = scope.0.accessor.get() {
                return accessor.lookup(key);
            }
            scope = scope.parent()?;
        }
    }

    /// Whether `self` and `other` are handles to one scope.
    pub fn same_scope(&self, other: &ResourceScope) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::fmt::Debug for ResourceScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceScope")
            .field("installed", &self.is_installed())
            .field("joined", &self.parent().is_some())
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

    fn read(scope: &ResourceScope) -> Option<u64> {
        scope
            .lookup("k")
            .and_then(|v| v.downcast::<u64>().ok())
            .map(|v| *v)
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
        assert_eq!(read(&child), Some(7));
        assert!(scope.install(Arc::new(Fixed("k", 8))).is_err());
        assert!(child.same_scope(&scope));
        assert!(!ResourceScope::new().same_scope(&scope));
    }

    #[test]
    fn a_joined_scope_resolves_through_its_parent_until_it_installs_its_own() {
        let root = ResourceScope::new();
        let child = ResourceScope::new();
        let grandchild = ResourceScope::new();
        child.join(&root).unwrap();
        grandchild.join(&child).unwrap();
        assert_eq!(read(&grandchild), None, "no accessor anywhere yet");
        root.install(Arc::new(Fixed("k", 1))).ok();
        assert_eq!(read(&grandchild), Some(1), "installed once at the root");
        child.install(Arc::new(Fixed("k", 2))).ok();
        assert_eq!(read(&grandchild), Some(2), "the nearest accessor answers");
        assert_eq!(read(&root), Some(1), "a parent never reads its child's");
    }

    #[test]
    fn a_scope_with_its_own_accessor_keeps_it() {
        let root = ResourceScope::with_accessor(Arc::new(Fixed("k", 1)));
        let own = ResourceScope::with_accessor(Arc::new(Fixed("k", 2)));
        own.join(&root).unwrap();
        assert!(own.parent().is_none());
        assert_eq!(read(&own), Some(2));
    }

    #[test]
    fn a_scope_never_delegates_to_itself_or_a_descendant() {
        let root = ResourceScope::new();
        root.join(&root).unwrap();
        assert!(root.parent().is_none(), "joining itself changes nothing");
        let child = ResourceScope::new();
        child.join(&root).unwrap();
        let grandchild = ResourceScope::new();
        grandchild.join(&child).unwrap();
        root.join(&grandchild).unwrap();
        assert!(
            root.parent().is_none(),
            "an ancestor bound under its descendant stays a root"
        );
    }

    #[test]
    fn the_first_join_wins() {
        let a = ResourceScope::with_accessor(Arc::new(Fixed("k", 1)));
        let b = ResourceScope::with_accessor(Arc::new(Fixed("k", 2)));
        let child = ResourceScope::new();
        child.join(&a).unwrap();
        child.join(&b).unwrap();
        assert!(child.parent().is_some_and(|p| p.same_scope(&a)));
        assert_eq!(read(&child), Some(1), "the second join changed nothing");
    }

    #[test]
    fn concurrent_joins_never_close_a_cycle() {
        for _ in 0..200 {
            let scopes: Vec<ResourceScope> = (0..4).map(|_| ResourceScope::new()).collect();
            std::thread::scope(|s| {
                for i in 0..4 {
                    let (from, to) = (scopes[i].clone(), scopes[(i + 1) % 4].clone());
                    s.spawn(move || from.join(&to));
                }
            });
            // Every chain ends: at most three links of four stand.
            let live = scopes.iter().filter(|s| s.parent().is_some()).count();
            assert!(live < 4, "a cycle of joins stood");
            scopes[3].install(Arc::new(Fixed("k", 9))).ok();
            for s in &scopes {
                let _ = read(s);
            }
        }
    }
}
