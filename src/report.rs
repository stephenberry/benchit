//! Result types and the two reporters.
//!
//! Keeping the reporter behind a trait is what keeps `text` and `tsv` two
//! implementations rather than a set of branches threaded through the
//! scheduler.

use std::io::Write;

use crate::Throughput;
use crate::baseline::Row;
use crate::cli::Config;
use crate::stats::{Ratio, Stats};

/// One measured case, as handed back by [`Group::finish`](crate::Group::finish).
///
/// Everything the report prints is derived from these fields.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CaseResult {
    /// The case name, without the group prefix.
    pub name: String,
    /// Iterations per sample, as calibrated.
    pub iters: u64,
    /// Per-iteration nanoseconds, one entry per sample, in round order.
    ///
    /// Round order is load-bearing: index `i` here was measured in the same
    /// round as index `i` of every other case in the group, which is what makes
    /// the two comparable pair by pair.
    pub samples: Vec<f64>,
    /// Order statistics over [`samples`](Self::samples), in nanoseconds per
    /// iteration.
    pub stats: Stats,
    /// The group's reference case: the first one that passed the filter.
    ///
    /// False on every case of a group that declared
    /// [`no_reference`](crate::Group::no_reference), which has none.
    pub is_reference: bool,
    /// This case's cost relative to the reference case.
    ///
    /// `None` for the reference case, for a case whose ratio could not be
    /// computed because the reference measured as zero, and for every case of a
    /// group that declared [`no_reference`](crate::Group::no_reference).
    pub ratio: Option<Ratio>,
    /// What `--baseline` had recorded for this case, if anything.
    pub baseline: Option<Stats>,
    /// What this case declared it processes per iteration.
    pub throughput: Option<Throughput>,
}

impl CaseResult {
    /// Items per second, if this case declared a
    /// [`Throughput`](crate::Throughput).
    ///
    /// The unit is whatever the declaration counted; pair it with
    /// [`throughput`](Self::throughput) to know which.
    ///
    /// Computed from [`stats.min`](Stats::min), matching the printed column.
    /// Derive it from `stats.p50` instead when the question is whether a budget
    /// holds in practice rather than at best: something that fits only at its
    /// best-case time does not fit.
    pub fn rate(&self) -> Option<f64> {
        let amount = self.throughput?.amount() as f64;
        if self.stats.min <= 0.0 {
            return None;
        }
        Some(amount / (self.stats.min * 1e-9))
    }

    /// Change in min against the loaded baseline, as a fraction: `0.04` is 4%
    /// slower than the baseline, `-0.04` is 4% faster.
    pub fn delta(&self) -> Option<f64> {
        let base = self.baseline?.min;
        if base <= 0.0 {
            return None;
        }
        Some((self.stats.min - base) / base)
    }

    pub(crate) fn to_row(&self, group: &str) -> Row {
        Row {
            group: group.to_string(),
            case: self.name.clone(),
            iters: self.iters,
            min_ns: self.stats.min,
            p50_ns: self.stats.p50,
            p90_ns: self.stats.p90,
            samples: self.samples.len(),
        }
    }
}

/// One measured group, as returned by [`Group::finish`](crate::Group::finish).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GroupResult {
    /// The group name, which prefixes every case's full name.
    pub name: String,
    /// The measured cases, in registration order. The first is the reference.
    /// Empty if the group did not run; see [`Group::finish`](crate::Group::finish).
    pub cases: Vec<CaseResult>,
    /// The amount for the text header, set only when every case agrees. A
    /// formatting detail rather than a result: per-case amounts are on
    /// [`CaseResult::throughput`].
    pub(crate) throughput: Option<Throughput>,
    /// Set by [`Group::no_reference`](crate::Group::no_reference), and a
    /// formatting detail for the same reason: the cases already carry no ratio,
    /// this is what also removes the column they would have sat in.
    pub(crate) no_reference: bool,
}

/// Where results go.
pub(crate) trait Reporter {
    /// Called once, before the first group.
    fn header(&mut self, config: &Config, timer_ns: f64, iter_with_floor_ns: f64);
    /// Called once per group, as soon as that group has finished measuring.
    fn group(&mut self, group: &GroupResult);
    /// Called when the `Bench` is dropped.
    fn finish(&mut self) {}
}

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

/// `v` to `digits` significant figures, in plain decimal notation.
///
/// Never fewer digits than the integer part needs, so a large value stays exact
/// rather than being rounded to a power of ten: `34567` to 3 figures is
/// `34567`, not `34600`.
pub(crate) fn sig(v: f64, digits: usize) -> String {
    if !v.is_finite() {
        return "-".to_string();
    }
    // Below 1.0 the integer-digit count goes negative, which is the point:
    // 0.00042 needs more decimals than 0.42, not the same three. Getting this
    // wrong printed a case that won by 1000x as `0.00x`.
    let decimals = |a: f64| {
        let int_digits = if a > 0.0 {
            a.log10().floor() as i32 + 1
        } else {
            1
        };
        (digits as i32 - int_digits).max(0) as usize
    };
    let d = decimals(v.abs());
    let formatted = format!("{v:.d$}");
    // Rounding can carry into another integer digit (0.9996 to three figures is
    // 1.00, not 1.000). Recompute from the rounded value so a column of ratios
    // does not mix widths.
    let carried = decimals(formatted.parse().unwrap_or(v).abs());
    if carried < d {
        format!("{v:.carried$}")
    } else {
        formatted
    }
}

/// A duration in nanoseconds, scaled per value rather than per column, to four
/// significant figures.
pub(crate) fn time(ns: f64) -> String {
    if !ns.is_finite() {
        return "-".to_string();
    }
    let (v, unit) = match ns.abs() {
        n if n < 1e3 => (ns, "ns"),
        n if n < 1e6 => (ns / 1e3, "us"),
        n if n < 1e9 => (ns / 1e6, "ms"),
        _ => (ns / 1e9, "s"),
    };
    format!("{} {}", sig(v, 4), unit)
}

/// A byte count in binary units.
fn bytes(n: f64, suffix: &str) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < UNITS.len() {
        v /= 1024.0;
        i += 1;
    }
    format!("{} {}{}", sig(v, 4), UNITS[i], suffix)
}

/// An element count in SI units.
fn elements(n: f64, suffix: &str) -> String {
    const UNITS: [&str; 5] = ["", "K", "M", "G", "T"];
    let mut v = n;
    let mut i = 0;
    while v >= 1000.0 && i + 1 < UNITS.len() {
        v /= 1000.0;
        i += 1;
    }
    format!("{} {}elem{}", sig(v, 4), UNITS[i], suffix)
}

/// The amount a group declared, for its header line. Exact counts, so small
/// ones print as integers rather than as `1.000`.
fn amount(t: Throughput) -> String {
    match t {
        Throughput::Bytes(n) if n < 1024 => format!("{n} B"),
        Throughput::Bytes(n) => bytes(n as f64, ""),
        Throughput::Elements(n) if n < 1000 => format!("{n} elem"),
        Throughput::Elements(n) => elements(n as f64, ""),
    }
}

/// A rate, for a case's throughput column.
fn rate(per_second: f64, t: Throughput) -> String {
    match t {
        Throughput::Bytes(_) => bytes(per_second, "/s"),
        Throughput::Elements(_) => elements(per_second, "/s"),
    }
}

/// A ratio to three significant figures, so `1.00x`, `0.0421x`, and `34567x`
/// all read well.
fn ratio(v: f64) -> String {
    format!("{}x", sig(v, 3))
}

fn percent(fraction: f64) -> String {
    format!("{:+.1}%", fraction * 100.0)
}

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

/// Aligned text on stdout.
pub(crate) struct TextReporter {
    out: Box<dyn Write>,
}

impl TextReporter {
    pub fn new(out: Box<dyn Write>) -> Self {
        Self { out }
    }
}

impl Reporter for TextReporter {
    fn header(&mut self, config: &Config, timer_ns: f64, iter_with_floor_ns: f64) {
        let schedule = if config.interleave {
            if config.block == 1 {
                "interleaved".to_string()
            } else {
                format!("interleaved, {} per visit", config.block)
            }
        } else {
            "sequential".to_string()
        };
        let _ = writeln!(
            self.out,
            "benchit: timer {}/call, iter_with floor {}/iter",
            time(timer_ns),
            time(iter_with_floor_ns),
        );
        let _ = writeln!(
            self.out,
            "         up to {} sample{} in {} ms per case, {}",
            config.samples,
            if config.samples == 1 { "" } else { "s" },
            config.time_ms,
            schedule,
        );
        let _ = writeln!(self.out);
    }

    fn group(&mut self, group: &GroupResult) {
        let cells: Vec<Cells> = group
            .cases
            .iter()
            .map(|c| Cells::new(c, group.no_reference))
            .collect();
        let w = Widths::of(&cells);

        // The declared amount, right-aligned over the min column.
        let mut head = group.name.clone();
        if let Some(t) = group.throughput {
            let a = amount(t);
            let column = 2 + w.name + 2 + w.min;
            let pad = column
                .saturating_sub(head.chars().count() + a.chars().count())
                .max(2);
            head.push_str(&" ".repeat(pad));
            head.push_str(&a);
        }
        let _ = writeln!(self.out, "{head}");

        for c in &cells {
            let mut line = format!(
                "  {:<name$}  {:>min$}   p50 {:>p50$}   p90 {:>p90$}",
                c.name,
                c.min,
                c.p50,
                c.p90,
                name = w.name,
                min = w.min,
                p50 = w.p50,
                p90 = w.p90,
            );
            if w.rate > 0 {
                line.push_str(&format!("   {:>rate$}", c.rate, rate = w.rate));
            }
            if w.ratio > 0 {
                line.push_str(&format!("   {:>ratio$}", c.ratio, ratio = w.ratio));
            }
            // Padded even when empty, so a case without a band does not shift
            // the columns after it.
            if w.band > 0 {
                line.push_str(&format!("  {:<band$}", c.band, band = w.band));
            }
            if w.delta > 0 {
                line.push_str(&format!("  base {:>delta$}", c.delta, delta = w.delta));
            }
            let _ = writeln!(self.out, "{}", line.trim_end());
        }
        let _ = writeln!(self.out);
        let _ = self.out.flush();
    }

    fn finish(&mut self) {
        let _ = self.out.flush();
    }
}

/// One case's cells, pre-formatted so the widths can be measured.
struct Cells {
    name: String,
    min: String,
    p50: String,
    p90: String,
    rate: String,
    ratio: String,
    band: String,
    delta: String,
}

impl Cells {
    fn new(c: &CaseResult, no_reference: bool) -> Self {
        Self {
            name: c.name.clone(),
            min: time(c.stats.min),
            p50: time(c.stats.p50),
            p90: time(c.stats.p90),
            rate: c
                .rate()
                .zip(c.throughput)
                .map(|(v, t)| rate(v, t))
                .unwrap_or_default(),
            // A group with no reference has nothing to compare, so the cell is
            // empty rather than a dash: an empty cell is what collapses the
            // column in `Widths`, and a dash would read as a failed ratio.
            ratio: if no_reference {
                String::new()
            } else {
                match (&c.ratio, c.is_reference) {
                    (Some(r), _) => ratio(r.point),
                    (None, true) => "1.00x".to_string(),
                    // The reference measured as zero, so there is no ratio to
                    // report. Saying "1.00x" here would be a lie.
                    (None, false) => "-".to_string(),
                }
            },
            band: match c.ratio.and_then(|r| r.iqr) {
                Some((lo, hi)) => format!("[{} .. {}]", sig(lo, 3), sig(hi, 3)),
                None => String::new(),
            },
            delta: c.delta().map(percent).unwrap_or_default(),
        }
    }
}

#[derive(Default)]
struct Widths {
    name: usize,
    min: usize,
    p50: usize,
    p90: usize,
    rate: usize,
    ratio: usize,
    band: usize,
    delta: usize,
}

impl Widths {
    fn of(cells: &[Cells]) -> Self {
        let mut w = Widths::default();
        for c in cells {
            w.name = w.name.max(c.name.chars().count());
            w.min = w.min.max(c.min.chars().count());
            w.p50 = w.p50.max(c.p50.chars().count());
            w.p90 = w.p90.max(c.p90.chars().count());
            w.rate = w.rate.max(c.rate.chars().count());
            w.band = w.band.max(c.band.chars().count());
            w.delta = w.delta.max(c.delta.chars().count());
        }
        // A group with one case has nothing to compare against, so it prints no
        // ratio column at all.
        if cells.len() > 1 {
            w.ratio = cells
                .iter()
                .map(|c| c.ratio.chars().count())
                .max()
                .unwrap_or(0);
        }
        w
    }
}

// ---------------------------------------------------------------------------
// TSV
// ---------------------------------------------------------------------------

/// One machine-readable row per case.
pub(crate) struct TsvReporter {
    out: Box<dyn Write>,
    header_written: bool,
}

impl TsvReporter {
    pub fn new(out: Box<dyn Write>) -> Self {
        Self {
            out,
            header_written: false,
        }
    }
}

impl Reporter for TsvReporter {
    fn header(&mut self, config: &Config, timer_ns: f64, iter_with_floor_ns: f64) {
        let _ = writeln!(
            self.out,
            "# benchit v{}  timer_ns={:.3}  iter_with_floor_ns={:.3}  profile={}  interleave={}  block={}",
            env!("CARGO_PKG_VERSION"),
            timer_ns,
            iter_with_floor_ns,
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
            config.interleave,
            config.block,
        );
    }

    fn group(&mut self, group: &GroupResult) {
        if !self.header_written {
            let _ = writeln!(
                self.out,
                "group\tcase\titers\tsamples\tmin_ns\tp50_ns\tp90_ns\tratio\tratio_lo\tratio_hi\tper_second\tbaseline_min_ns"
            );
            self.header_written = true;
        }
        for c in &group.cases {
            let (point, lo, hi) = match (&c.ratio, c.is_reference) {
                (Some(r), _) => {
                    let (lo, hi) = r.iqr.map(|(l, h)| (num(l), num(h))).unwrap_or_default();
                    (num(r.point), lo, hi)
                }
                (None, true) => (num(1.0), String::new(), String::new()),
                (None, false) => (String::new(), String::new(), String::new()),
            };
            let _ = writeln!(
                self.out,
                "{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}\t{}",
                group.name,
                c.name,
                c.iters,
                c.samples.len(),
                c.stats.min,
                c.stats.p50,
                c.stats.p90,
                point,
                lo,
                hi,
                c.rate().map(num).unwrap_or_default(),
                c.baseline.map(|b| num(b.min)).unwrap_or_default(),
            );
        }
        let _ = self.out.flush();
    }

    fn finish(&mut self) {
        let _ = self.out.flush();
    }
}

fn num(v: f64) -> String {
    format!("{v:.6}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn significant_figures() {
        assert_eq!(sig(412.34, 4), "412.3");
        assert_eq!(sig(1.2345, 4), "1.234");
        assert_eq!(sig(34567.0, 3), "34567");
        assert_eq!(sig(0.5, 4), "0.5000");
        assert_eq!(sig(f64::NAN, 4), "-");
    }

    #[test]
    fn values_below_one_keep_their_significant_digits() {
        // A case that wins by 1000x has a ratio of 0.001; rounding that to two
        // decimals reports `0.00x` and throws the whole result away.
        assert_eq!(ratio(0.000421), "0.000421x");
        assert_eq!(ratio(0.0421), "0.0421x");
        assert_eq!(ratio(0.421), "0.421x");
        assert_eq!(sig(0.0, 3), "0.00");
    }

    #[test]
    fn rounding_that_carries_does_not_widen_the_column() {
        // 0.9996 to three figures is 1.00; without the carry check it printed
        // 1.000 and sat one character wider than every other ratio.
        assert_eq!(sig(0.9996, 3), "1.00");
        assert_eq!(sig(9.996, 3), "10.0");
        assert_eq!(sig(999.6, 3), "1000");
    }

    #[test]
    fn durations_scale_per_value() {
        assert_eq!(time(412.3), "412.3 ns");
        assert_eq!(time(412_300.0), "412.3 us");
        assert_eq!(time(1_284_000.0), "1.284 ms");
        assert_eq!(time(2_000_000_000.0), "2.000 s");
    }

    #[test]
    fn rates_use_binary_bytes_and_si_elements() {
        assert_eq!(
            rate(1024.0 * 1024.0 * 1024.0, Throughput::Bytes(1)),
            "1.000 GiB/s"
        );
        assert_eq!(rate(2_000_000.0, Throughput::Elements(1)), "2.000 Melem/s");
    }

    #[test]
    fn declared_amounts_are_exact_counts() {
        assert_eq!(amount(Throughput::Bytes(1 << 20)), "1.000 MiB");
        assert_eq!(amount(Throughput::Bytes(64)), "64 B");
        assert_eq!(amount(Throughput::Elements(1)), "1 elem");
        assert_eq!(amount(Throughput::Elements(1_000_000)), "1.000 Melem");
    }

    #[test]
    fn ratios_stay_readable_across_magnitudes() {
        assert_eq!(ratio(1.0), "1.00x");
        assert_eq!(ratio(2.357), "2.36x");
        assert_eq!(ratio(34600.0), "34600x");
    }
}
