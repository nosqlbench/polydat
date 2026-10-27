// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Cycle-time data-source audit log.
//!
//! Emits one structured line per significant data-source event so
//! the operator can spot mismatches between what
//! `crate::library::vectors::do_dataset_prebuffer` *covered* and
//! what cycle-time accessors *opened*. The typical failure mode
//! this catches: prebuffer reports success but readers still hit
//! HTTP per cycle, because the facets the workload reads aren't
//! in the active profile's manifest.
//!
//! Output is routed through [`set_log_fn`] when the caller (the
//! activity runner, the test harness, …) installs a sink. With no
//! sink installed, lines go to stderr — preserves visibility from
//! contexts that don't carry an `observer::log` plumbing path
//! (unit tests, the `dryrun=` paths).

use std::sync::OnceLock;

/// Severity for audit-channel events. A conventional severity
/// ladder so a host's installed sink can map it 1:1 to its own
/// logger levels without reformatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// The finest detail.
    Trace,
    /// Developer detail.
    Debug,
    /// Ordinary events.
    Info,
    /// Something is off but the run continues.
    Warn,
    /// A failure.
    Error,
}

type LogFn = Box<dyn Fn(LogLevel, &str) + Send + Sync>;

static LOG_FN: OnceLock<LogFn> = OnceLock::new();

/// Install the audit sink. A host installs this once so audit lines
/// flow through its own logger alongside the rest of the run output.
/// Subsequent calls are no-ops.
pub fn set_log_fn<F>(f: F)
where
    F: Fn(LogLevel, &str) + Send + Sync + 'static,
{
    let _ = LOG_FN.set(Box::new(f));
}

/// Emit a leveled message through the configured sink, falling
/// back to stderr when no sink is installed (unit tests, dryrun
/// paths, pre-init).
pub fn log(level: LogLevel, msg: &str) {
    if let Some(f) = LOG_FN.get() {
        f(level, msg);
    } else {
        let tag = match level {
            LogLevel::Trace => "TRC",
            LogLevel::Debug => "DBG",
            LogLevel::Info => "INF",
            LogLevel::Warn => "WRN",
            LogLevel::Error => "ERR",
        };
        eprintln!("{tag} {msg}");
    }
}

impl From<log::Level> for LogLevel {
    fn from(level: log::Level) -> Self {
        match level {
            log::Level::Error => LogLevel::Error,
            log::Level::Warn => LogLevel::Warn,
            log::Level::Info => LogLevel::Info,
            log::Level::Debug => LogLevel::Debug,
            log::Level::Trace => LogLevel::Trace,
        }
    }
}

/// A [`log`](mod@log) logger that forwards vectordata's records into the audit
/// channel, so its warnings reach the host's sink beside polydat's own
/// lines. The records forwarded are those at `Info` and above from the
/// `vectordata` crate and its `veks_*` crates, each as
/// `<target>: <message>` at the matching [`LogLevel`].
///
/// polydat never installs it: a process has one `log` logger, and
/// choosing it is the host's call. A host with no logger of its own
/// installs this one with [`LogBridge::install`]; a host that has one
/// already receives vectordata's records through it.
pub struct LogBridge;

impl LogBridge {
    /// The least severe level forwarded.
    const LEVEL: log::LevelFilter = log::LevelFilter::Info;

    /// Make this the process's `log` logger. Fails if a logger is
    /// already installed, which is then left in place.
    pub fn install() -> Result<(), log::SetLoggerError> {
        static BRIDGE: LogBridge = LogBridge;
        log::set_logger(&BRIDGE)?;
        log::set_max_level(Self::LEVEL);
        Ok(())
    }

    /// Whether `target` names a module of vectordata or of one of its
    /// `veks_*` crates.
    fn forwards(target: &str) -> bool {
        let krate = target.split("::").next().unwrap_or(target);
        krate == "vectordata" || krate.starts_with("veks_")
    }
}

impl log::Log for LogBridge {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= Self::LEVEL && Self::forwards(metadata.target())
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            log(
                record.level().into(),
                &format!("{}: {}", record.target(), record.args()),
            );
        }
    }

    fn flush(&self) {}
}

/// Convenience helpers for callsite ergonomics.
pub fn debug(msg: &str) {
    log(LogLevel::Debug, msg);
}
/// Log at `Info`.
pub fn info(msg: &str) {
    log(LogLevel::Info, msg);
}
/// Log at `Warn`.
pub fn warn(msg: &str) {
    log(LogLevel::Warn, msg);
}
/// Log at `Error`.
pub fn error(msg: &str) {
    log(LogLevel::Error, msg);
}

/// Record that `dataset_prebuffer(...)` was invoked. Emitted at
/// the top of `do_dataset_prebuffer` *unconditionally* — fires
/// even if the function bails on a resolve / profile-missing
/// error, so the absence of this line in `session.log` is
/// definitive evidence that `init prebuffer = ...` never
/// evaluated. Pairs with `record_prebuffered` (per-facet) and
/// `log_prebuffer_summary` (tail).
pub fn record_prebuffer_entered(source: &str) {
    debug(&format!("prebuffer: entered dataset_prebuffer({source:?})"));
}

/// Record that the prebuffer pass covered `facet` of `profile` of the
/// dataset named `dataset`. Emitted once per facet in the report of
/// the prebuffer's `view.fetch`.
pub fn record_prebuffered(dataset: &str, profile: &str, facet: &str) {
    debug(&format!("prebuffer: covered {dataset}:{profile}/{facet}"));
}

/// Record that a reader was opened for `facet` of `profile` of the
/// dataset named `dataset`. Emitted from
/// every `vectors::*` reader-open path *before* the actual
/// `view.<facet>_as()` / `view.facet_as()` call so the line lands
/// even if the open errors. `kind` distinguishes the open shape
/// (`uniform`, `ivvec32`, `generic-typed`, …) for at-a-glance
/// debugging.
pub fn record_opened(dataset: &str, profile: &str, facet: &str, kind: &str) {
    debug(&format!(
        "vectordata: opened {dataset}:{profile}/{facet} (kind={kind})"
    ));
}

/// One-line summary at the end of `dataset_prebuffer` over `profile`
/// of the dataset named `dataset`. Pairs
/// with the per-facet `prebuffer: covered …` lines above and
/// the `vectordata: opened …` lines below to make the
/// covered-vs-opened delta easy to read.
pub fn log_prebuffer_summary(dataset: &str, profile: &str, facet_count: u64) {
    debug(&format!(
        "prebuffer: done {dataset}:{profile} (covered {facet_count} facet(s))"
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use log::Log;

    fn enabled(target: &str, level: log::Level) -> bool {
        LogBridge.enabled(&log::Metadata::builder().target(target).level(level).build())
    }

    #[test]
    fn the_bridge_forwards_vectordata_and_its_veks_crates() {
        for target in ["vectordata", "vectordata::fetch", "veks_io::reader::vvec", "veks_apicore"] {
            assert!(enabled(target, log::Level::Warn), "{target}");
        }
    }

    #[test]
    fn the_bridge_leaves_other_crates_to_the_host() {
        for target in ["reqwest::connect", "hyper", "vectordatax", "veks", "polydat_core"] {
            assert!(!enabled(target, log::Level::Error), "{target}");
        }
    }

    #[test]
    fn the_bridge_forwards_info_and_above() {
        for level in [log::Level::Error, log::Level::Warn, log::Level::Info] {
            assert!(enabled("vectordata::fetch", level), "{level}");
        }
        for level in [log::Level::Debug, log::Level::Trace] {
            assert!(!enabled("vectordata::fetch", level), "{level}");
        }
    }

    #[test]
    fn each_log_level_maps_to_its_audit_level() {
        let pairs = [
            (log::Level::Error, LogLevel::Error),
            (log::Level::Warn, LogLevel::Warn),
            (log::Level::Info, LogLevel::Info),
            (log::Level::Debug, LogLevel::Debug),
            (log::Level::Trace, LogLevel::Trace),
        ];
        for (level, audit) in pairs {
            assert_eq!(LogLevel::from(level), audit);
        }
    }
}
