//! A minimal benchmarking harness: no dependencies, no configuration, and a
//! comparison you can trust.
//!
//! `benchit` measures in-process, CPU-bound Rust functions that take somewhere
//! between a nanosecond and a few hundred milliseconds. It is not a load
//! generator, not a profiler, and not a system benchmark.
//!
//! # The comparison is the product
//!
//! Most benchmarking compares alternatives within a single run: two
//! implementations of the same operation, a fast path against its fallback, a
//! new data structure against the one it replaces. A harness that runs case A
//! to completion and then case B puts every bit of frequency scaling, thermal
//! throttling, and background load that occurred in between directly onto the
//! ratio you came to read.
//!
//! So a group's cases are run in **interleaved rounds**: one sample of each,
//! then the next round. Round `i` yields `A_i` and `B_i` measured milliseconds
//! apart, under the same thermal state and the same background load. Drift
//! cancels inside every pair, and the interquartile range of the per-round
//! ratios is an honest noise band, measured rather than resampled.
//!
//! # Usage
//!
//! Add it as a dev-dependency and declare the bench target with
//! `harness = false`:
//!
//! ```toml
//! [dev-dependencies]
//! benchit = "0.1"
//!
//! [[bench]]
//! name = "decode"
//! harness = false
//! ```
//!
//! `harness = false` already means `main` is yours, so there are no macros:
//!
//! ```no_run
//! use benchit::{Bench, Throughput};
//! use std::hint::black_box;
//!
//! fn main() {
//!     let mut bench = Bench::from_args();
//!     let input = vec![7u8; 1 << 20];
//!
//!     let mut group = bench.group("decode/1MiB");
//!     group.throughput(Throughput::Bytes(input.len() as u64));
//!     group.bench("sum", |b| b.iter(|| black_box(&input).iter().map(|&x| x as u64).sum::<u64>()));
//!     group.bench("fold", |b| b.iter(|| black_box(&input).iter().fold(0u64, |a, &x| a + x as u64)));
//!     group.finish();
//! }
//! ```
//!
//! ```text
//! lookup/10000_keys  1 elem
//!   hashmap      6.018 ns   p50 6.258 ns   p90 6.358 ns   166.2 Melem/s   1.00x
//!   btreemap     19.70 ns   p50 20.28 ns   p90 20.70 ns   50.77 Melem/s   3.27x  [3.24 .. 3.29]
//! ```
//!
//! The leading column is the minimum, which is the best estimator of true cost
//! when noise is one-sided. The ratio and its bracket are the median and
//! interquartile range of the per-round paired ratios, so the point always sits
//! inside its own band. See [`Bencher::iter`] for what falls inside the timed
//! span.
//!
//! # Command line
//!
//! Bench binaries accept a substring filter on `group/case`, plus:
//!
//! ```text
//! --quick               ~200ms per benchmark
//! --samples N           cap on samples per case (default 50)
//! --time MS             per-benchmark budget in ms (default 1000)
//! --block N             samples per visit when interleaving (default 1)
//! --no-interleave       run each case to completion instead of in rounds
//! --save-baseline NAME  write benchit/NAME.tsv beside the built binary
//! --baseline NAME       load benchit/NAME.tsv from there, show a delta column
//! --format=text|tsv     output format (default text)
//! --list                list matching benchmarks without running them
//! ```
//!
//! The full text is in [`USAGE`].
//!
//! # Reading the numbers back
//!
//! [`Group::finish`] returns the [`GroupResult`] it just printed, so a
//! benchmark that needs a metric this crate has no opinion about (cost per
//! audio second, fraction of a core, cycles per byte) computes it from the
//! samples rather than parsing `--format=tsv` back out of a pipe.
//!
//! # What it does not do
//!
//! No HTML reports or plots. No bootstrap resampling, confidence intervals, or
//! p-values: everything reported is an order statistic over samples that were
//! actually collected. No automatic regression verdicts, because a verdict on a
//! 2% cross-run change is a false-confidence machine.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod baseline;
mod bencher;
mod cli;
mod report;
mod runner;
mod stats;

pub use bencher::Bencher;
pub use cli::{ArgError, Config, Format, USAGE};
pub use report::{CaseResult, GroupResult};
pub use stats::{Ratio, Stats};

use std::fmt::Display;

use report::{Reporter, TextReporter, TsvReporter};
use runner::Case;

/// What a benchmark processes per iteration, so the reporter can print a rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Throughput {
    /// Bytes per iteration; reported in binary units.
    Bytes(u64),
    /// Elements per iteration; reported in SI units.
    Elements(u64),
}

impl Throughput {
    /// How much one iteration processes, whatever the unit. Reading the count
    /// through this rather than by matching keeps a new variant from breaking
    /// the caller.
    pub fn amount(self) -> u64 {
        match self {
            Throughput::Bytes(n) | Throughput::Elements(n) => n,
        }
    }
}

/// A benchmark run: configuration, the loaded baseline, and the reporter.
///
/// Create one at the top of `main`, take [`group`](Bench::group)s from it, and
/// let it drop at the end. Dropping it writes `--save-baseline` if one was
/// requested.
pub struct Bench {
    config: Config,
    reporter: Box<dyn Reporter>,
    /// `Some` once the run has started, which is also what keeps the header to
    /// one line.
    timer_ns: Option<f64>,
    pub(crate) iter_with_floor_ns: f64,
    loaded: baseline::Baseline,
    to_save: Vec<baseline::Row>,
}

impl Bench {
    /// Build from the process arguments.
    ///
    /// Exits the process on `--help` (status 0) or on an unusable argument
    /// (status 2), which is the behaviour a bench binary wants. Use
    /// [`Config::from_args`] with [`with_config`](Bench::with_config) to handle
    /// those yourself.
    pub fn from_args() -> Self {
        match Config::from_args() {
            Ok(config) => Self::with_config(config),
            Err(ArgError::HelpRequested) => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("benchit: {e}\n");
                eprint!("{USAGE}");
                std::process::exit(2);
            }
        }
    }

    /// Build from an explicit configuration.
    ///
    /// `Config`'s fields are public, so counts of zero are clamped to one here
    /// rather than trusted: the command line rejects them, but a caller
    /// building a `Config` by hand bypasses that check, and a zero would
    /// otherwise reach the scheduler and collect no samples at all.
    pub fn with_config(mut config: Config) -> Self {
        config.samples = config.samples.max(1);
        config.block = config.block.max(1);
        config.time_ms = config.time_ms.max(1);
        let reporter = reporter_for(config.format, Box::new(std::io::stdout()));

        let mut loaded = baseline::Baseline::new();
        if let Some(name) = &config.baseline {
            match baseline::path(name).and_then(|p| baseline::load(&p)) {
                Ok(rows) => loaded = rows,
                // A missing or unreadable baseline should not sink the run: the
                // measurements are still worth having, just without a delta.
                Err(e) => eprintln!("benchit: cannot read baseline `{name}`: {e}"),
            }
        }

        Self {
            config,
            reporter,
            timer_ns: None,
            iter_with_floor_ns: 0.0,
            loaded,
            to_save: Vec::new(),
        }
    }

    /// Send the report somewhere other than stdout.
    ///
    /// The format still follows [`Config::format`]. Must be called before the
    /// first group, since the run header is written when the first group is
    /// measured.
    pub fn report_to(mut self, out: Box<dyn std::io::Write>) -> Self {
        self.reporter = reporter_for(self.config.format, out);
        self
    }

    /// Open a group of cases to be compared against each other.
    ///
    /// Cases are registered, not run: the group cannot build an interleaved
    /// schedule until it has seen all of them, so the measuring happens when
    /// the group is dropped.
    pub fn group(&mut self, name: impl Display) -> Group<'_> {
        Group {
            name: name.to_string(),
            bench: self,
            throughput: None,
            cases: Vec::new(),
        }
    }

    /// Measure this machine's clock costs and print the run header, once.
    fn begin_run(&mut self) {
        if self.timer_ns.is_some() {
            return;
        }
        if cfg!(debug_assertions) {
            // On stderr, so it survives a pipe and never lands inside a TSV
            // stream. `cargo bench` uses the bench profile, but people run
            // bench binaries directly and then report the numbers.
            eprintln!(
                "benchit: !! debug build. These numbers are not worth reporting; build with --release."
            );
        }
        let timer_ns = bencher::timer_round_trip();
        self.iter_with_floor_ns = bencher::iter_with_floor();
        self.timer_ns = Some(timer_ns);
        self.reporter
            .header(&self.config, timer_ns, self.iter_with_floor_ns);
    }

    /// What the loaded baseline recorded for a case. The saved row's iteration
    /// and sample counts describe how that number was obtained rather than what
    /// it was, and nothing reports them.
    fn saved_stats(&self, full_name: &str) -> Option<Stats> {
        let row = self.loaded.get(full_name)?;
        Some(Stats {
            min: row.min_ns,
            p50: row.p50_ns,
            p90: row.p90_ns,
        })
    }

    fn record(&mut self, group: &GroupResult) {
        self.to_save
            .extend(group.cases.iter().map(|c| c.to_row(&group.name)));
    }
}

impl Drop for Bench {
    fn drop(&mut self) {
        // Nothing measured means nothing to save. Without this, `--list
        // --save-baseline main` would write an empty file over a real baseline.
        if let Some(name) = self
            .config
            .save_baseline
            .take()
            .filter(|_| !self.to_save.is_empty())
        {
            let rows = std::mem::take(&mut self.to_save);
            match baseline::path(&name).and_then(|p| baseline::save(&p, rows).map(|()| p)) {
                Ok(p) => eprintln!("benchit: baseline saved to {}", p.display()),
                Err(e) => eprintln!("benchit: cannot save baseline `{name}`: {e}"),
            }
        }
        self.reporter.finish();
    }
}

fn reporter_for(format: Format, out: Box<dyn std::io::Write>) -> Box<dyn Reporter> {
    match format {
        Format::Text => Box::new(TextReporter::new(out)),
        Format::Tsv => Box::new(TsvReporter::new(out)),
    }
}

/// A set of cases measured against each other in one interleaved schedule.
///
/// The group runs when it is dropped, so results appear in registration order
/// as each group finishes.
pub struct Group<'b> {
    bench: &'b mut Bench,
    name: String,
    /// Applied to cases registered from here on, so a parameterized group can
    /// declare a different amount per size.
    throughput: Option<Throughput>,
    cases: Vec<Case<'b>>,
}

impl<'b> Group<'b> {
    /// Declare what one iteration processes, so the report carries a rate.
    ///
    /// As in criterion, this applies to every case registered *after* the call,
    /// so a parameterized group can declare its own amount per case:
    ///
    /// ```no_run
    /// # use benchit::{Bench, Throughput};
    /// # let mut bench = Bench::from_args();
    /// # let mut g = bench.group("sort");
    /// for n in [64u64, 1024] {
    ///     g.throughput(Throughput::Elements(n));
    ///     g.bench(format!("n={n}"), move |b| b.iter(|| n * 2));
    /// }
    /// ```
    pub fn throughput(&mut self, throughput: Throughput) -> &mut Self {
        self.throughput = Some(throughput);
        self
    }

    /// Register a case.
    ///
    /// `name` is anything [`Display`], so a parameterized name is just
    /// `format!("encode/{n}")`. The closure may borrow from the enclosing
    /// scope, which is how nearly every benchmark is written.
    ///
    /// Ratios are relative to the first case registered (more precisely, the
    /// first that passes the filter), which is conventionally the reference
    /// implementation.
    ///
    /// # Sharing a scratch buffer between cases
    ///
    /// Every case in a group is live at once, because an interleaved schedule
    /// cannot exist until all of them are registered. Two closures that each
    /// capture `&mut scratch` therefore do not compile, which is the one thing
    /// that catches people converting from a harness that runs each case as it
    /// is declared. Share it through a [`RefCell`](std::cell::RefCell) instead:
    ///
    /// ```no_run
    /// # use benchit::Bench;
    /// use std::cell::RefCell;
    ///
    /// # let mut bench = Bench::from_args();
    /// let scratch = RefCell::new(vec![0u64; 4096]);
    /// let mut group = bench.group("solve");
    /// group.bench("naive", |b| b.iter(|| scratch.borrow_mut()[0] += 1));
    /// group.bench("tuned", |b| b.iter(|| scratch.borrow_mut()[1] += 2));
    /// ```
    ///
    /// Borrow outside the [`iter`](Bencher::iter) call rather than inside it if
    /// the operation is small enough for a borrow-flag check to register.
    pub fn bench(&mut self, name: impl Display, f: impl FnMut(&mut Bencher) + 'b) -> &mut Self {
        self.cases.push(Case {
            name: name.to_string(),
            throughput: self.throughput,
            body: Box::new(f),
        });
        self
    }

    /// Run the group now and hand back what it measured.
    ///
    /// Dropping the group runs it too, so calling this is only necessary when
    /// you want the numbers. Everything the report prints is in the
    /// [`GroupResult`], which is how a benchmark computes a metric the harness
    /// has no business knowing about:
    ///
    /// ```no_run
    /// # use benchit::Bench;
    /// # const SECONDS_PER_BUFFER: f64 = 1024.0 / 48_000.0;
    /// # let mut bench = Bench::from_args();
    /// let mut group = bench.group("synth/saw");
    /// group.bench("scalar", |b| b.iter(|| 1));
    /// let result = group.finish();
    ///
    /// for case in &result.cases {
    ///     let cpu_seconds = case.stats.min * 1e-9;
    ///     // On stderr, so `--format=tsv` stays machine-readable.
    ///     eprintln!("{}: {:.1}x realtime", case.name, SECONDS_PER_BUFFER / cpu_seconds);
    /// }
    /// ```
    ///
    /// An empty [`cases`](GroupResult::cases) means the group did not run: a
    /// filter excluded it, or this was `--list`. A benchmark that gates on its
    /// own results should check for that, since "nothing was measured" and
    /// "nothing measured badly" are the same empty list.
    pub fn finish(mut self) -> GroupResult {
        self.run()
    }

    /// Takes the cases out, which is also what leaves `Drop` nothing to do.
    fn run(&mut self) -> GroupResult {
        let mut cases = std::mem::take(&mut self.cases);
        runner::run(self.bench, &self.name, &mut cases)
    }
}

impl Drop for Group<'_> {
    fn drop(&mut self) {
        // Running a full benchmark schedule while the stack unwinds would bury
        // the user's panic under seconds of measurement, and a body that then
        // panicked itself would abort the process outright.
        if self.cases.is_empty() || std::thread::panicking() {
            return;
        }
        self.run();
    }
}
