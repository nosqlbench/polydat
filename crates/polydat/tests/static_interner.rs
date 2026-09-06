// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 115 step 3: string constants are interned at kernel build. The
//! interner deduplicates by content, resolves whole handles, and holds
//! every static run and separator of a tile skeleton, so constants never
//! enter the cycle arena.

use polydat::dsl::compile_polydat;
use polydat::kernel::{cycle_arena_used, StaticInterner, TAG_MASK, TAG_STATIC};

#[test]
fn interning_deduplicates_by_content_and_resolves_whole_handles() {
    let a = StaticInterner::intern("srd-115 interner text");
    let b = StaticInterner::intern("srd-115 interner text");
    let c = StaticInterner::intern("srd-115 other text");
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_eq!(a & TAG_MASK, TAG_STATIC);
    assert_eq!(StaticInterner::resolve_handle(a), Some("srd-115 interner text"));
    assert_eq!(StaticInterner::resolve_handle(c), Some("srd-115 other text"));
    // An arena or table handle is not a static handle.
    assert_eq!(StaticInterner::resolve_handle(polydat::kernel::TAG_ARENA | 5), None);
}

#[test]
fn tile_static_runs_and_separators_are_interned_at_build() {
    let before = StaticInterner::len();
    let src = "input cycle: u64\n\
        tile doc : json := {\"meta\": {\"schema\": 3, \"unique-static-run-9f3a\": true}, \"n\": ${cycle}, \"xs\": [@for i in 0..2 sep \"; \" { ${i} }]}\n";
    let mut k = compile_polydat(src).unwrap();
    k.set_inputs(&[4]);
    let text = k.pull("doc").as_str().to_string();
    assert!(text.contains("unique-static-run-9f3a"), "{text}");
    assert_eq!(text, "{\"meta\": {\"schema\": 3, \"unique-static-run-9f3a\": true}, \"n\": 4, \"xs\": [0; 1]}");
    // The static runs and the separator now resolve from the interner.
    assert!(StaticInterner::len() > before);
    let run = StaticInterner::intern("{\"meta\": {\"schema\": 3, \"unique-static-run-9f3a\": true}, \"n\": ");
    assert_eq!(StaticInterner::resolve_handle(run).map(|s| s.contains("9f3a")), Some(true));
    assert_eq!(StaticInterner::len(), StaticInterner::len(), "interning an existing run adds nothing");
    let sep = StaticInterner::intern("; ");
    assert_eq!(StaticInterner::resolve_handle(sep), Some("; "));
    // Rendering copies statics from the interner and puts nothing in the arena.
    k.set_inputs(&[5]);
    let _ = k.pull("doc");
    assert_eq!(cycle_arena_used(), 0);
}

/// A string literal node classifies to a static handle whose text is the
/// literal's, so a cone stores an immediate rather than touching the arena.
#[cfg(feature = "jit")]
#[test]
fn a_string_literal_node_classifies_to_a_static_handle() {
    use polydat::compile::jit::{classify_node, JitOp};
    let node = polydat::library::identity::ConstStr::new("literal-2b7c".to_string());
    match classify_node(&node) {
        JitOp::StaticStr(handle) => {
            assert_eq!(StaticInterner::resolve_handle(handle), Some("literal-2b7c"));
            assert_eq!(StaticInterner::intern("literal-2b7c"), handle, "interning again yields the same handle");
        }
        other => panic!("expected StaticStr, got {other:?}"),
    }
}
