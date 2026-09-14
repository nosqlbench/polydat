// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! One source of truth for the effective native Cranelift ISA.
//!
//! `isa::lookup(Triple::host())` selects an architecture but does not infer
//! optional host features. JIT code and the SIMD planner must instead consume
//! the same `cranelift_native` builder, whose detection includes the OS-enabled
//! architectural state checked by Rust's feature-detection macros.

use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::settings;

/// Effective code-generation capabilities advertised by the native Cranelift
/// builder used for this process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectiveIsa {
    /// Full target triple of the JIT host.
    pub triple: String,
    /// Width of Polydat's fixed register type plane. This is deliberately not
    /// inferred from AVX/AVX-512 feature names.
    pub polydat_register_bits: u16,
    /// Cranelift version which produced this capability record.
    pub cranelift_version: &'static str,
    /// Enabled target-specific Cranelift boolean flags, sorted by name.
    enabled_flags: Vec<&'static str>,
}

impl EffectiveIsa {
    /// Probe the same native builder used by production JIT compilation.
    pub fn detect() -> Result<Self, String> {
        let isa = build_host_isa(settings::builder())?;
        let mut enabled_flags: Vec<_> = isa
            .isa_flags()
            .into_iter()
            .filter(|flag| flag.as_bool() == Some(true))
            .map(|flag| flag.name)
            .collect();
        enabled_flags.sort_unstable();

        Ok(Self {
            triple: isa.triple().to_string(),
            polydat_register_bits: 128,
            cranelift_version: cranelift_native::VERSION,
            enabled_flags,
        })
    }

    /// Whether an ISA flag such as `has_avx2` is enabled in the builder.
    pub fn has_flag(&self, flag: &str) -> bool {
        self.enabled_flags.binary_search(&flag).is_ok()
    }

    /// Stable ordered flag list for diagnostics and plan-cache identity.
    pub fn enabled_flags(&self) -> &[&'static str] {
        &self.enabled_flags
    }

    /// Text fingerprint suitable for inclusion in a compiled-plan cache key.
    pub fn fingerprint(&self) -> String {
        format!(
            "{}|clif={}|reg={}|{}",
            self.triple,
            self.cranelift_version,
            self.polydat_register_bits,
            self.enabled_flags.join(",")
        )
    }
}

/// Build a host ISA with native feature inference and caller-selected shared
/// Cranelift flags.
pub(crate) fn build_host_isa(
    shared_flag_builder: settings::Builder,
) -> Result<OwnedTargetIsa, String> {
    let isa_builder =
        cranelift_native::builder().map_err(|e| format!("native ISA detection failed: {e}"))?;
    isa_builder
        .finish(settings::Flags::new(shared_flag_builder))
        .map_err(|e| format!("native ISA build failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_record_matches_native_x86_detection() {
        let caps = EffectiveIsa::detect().expect("host ISA");
        assert_eq!(caps.polydat_register_bits, 128);
        assert!(!caps.fingerprint().is_empty());

        #[cfg(target_arch = "x86_64")]
        {
            assert_eq!(
                caps.has_flag("has_avx"),
                std::is_x86_feature_detected!("avx")
            );
            assert_eq!(
                caps.has_flag("has_avx2"),
                std::is_x86_feature_detected!("avx2")
            );
            assert_eq!(
                caps.has_flag("has_fma"),
                std::is_x86_feature_detected!("fma")
            );
        }
    }
}
