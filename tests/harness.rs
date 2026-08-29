//! End-to-end tests of the public API: the schedule the runner actually
//! produces, and the report it actually prints.

use std::cell::RefCell;
use std::io::Write;
use std::sync::{Arc, Mutex};

use benchit::{Bench, Config, Format, Throughput};

/// A report destination the test can read back.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Capture {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().expect("not poisoned").clone()).expect("utf-8")
    }
}

impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("not poisoned").extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A configuration that exercises the real calibrate-then-round path while
/// finishing quickly.
///
/// The budget is deliberately far larger than 12 batches of these bodies need,
/// so the sample cap is what binds and every case collects exactly 12 samples.
/// That keeps the visit-order assertions below deterministic even when the test
/// binary is running its cases in parallel on a loaded machine.
fn fast(format: Format) -> Config {
    Config {
        time_ms: 1000,
        samples: 12,
        format,
        ..Config::default()
    }
}

/// A body with an easily distinguished cost.
///
/// Each step depends nonlinearly on the last, so the optimizer cannot collapse
/// the loop into a closed form and make every case cost the same.
fn work(rounds: u64) -> u64 {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..rounds {
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    }
    x
}

/// Run two cases and return the order in which the scheduler visited them,
/// alongside the report.
fn schedule(config: Config) -> (Vec<&'static str>, String) {
    let log = RefCell::new(Vec::new());
    let capture = Capture::default();
    {
        let mut bench = Bench::with_config(config).report_to(Box::new(capture.clone()));
        let mut g = bench.group("g");
        g.bench("a", |b| {
            log.borrow_mut().push("a");
            b.iter(|| work(std::hint::black_box(200)))
        });
        g.bench("b", |b| {
            log.borrow_mut().push("b");
            b.iter(|| work(std::hint::black_box(400)))
        });
        g.finish();
    }
    (log.into_inner(), capture.text())
}

#[test]
fn interleaving_alternates_between_cases() {
    let (visits, _) = schedule(fast(Format::Text));
    // Calibration runs first and is not interleaved, so look at the tail,
    // which is entirely rounds.
    let tail = &visits[visits.len() - 12..];
    let expected: Vec<&str> = (0..12)
        .map(|i| if i % 2 == 0 { "a" } else { "b" })
        .collect();
    assert_eq!(tail, expected.as_slice(), "full visit order: {visits:?}");
}

#[test]
fn a_block_size_trades_pairing_for_warmer_caches() {
    let config = Config {
        block: 4,
        ..fast(Format::Text)
    };
    let (visits, _) = schedule(config);
    let tail = &visits[visits.len() - 8..];
    assert_eq!(
        tail,
        ["a", "a", "a", "a", "b", "b", "b", "b"],
        "visit order: {visits:?}"
    );
}

#[test]
fn no_interleave_runs_each_case_to_completion() {
    let config = Config {
        interleave: false,
        ..fast(Format::Text)
    };
    let (visits, _) = schedule(config);
    let tail = &visits[visits.len() - 12..];
    assert!(tail.iter().all(|&v| v == "b"), "visit order: {visits:?}");
}

#[test]
fn the_report_carries_a_ratio_and_a_measured_band() {
    let (_, report) = schedule(fast(Format::Text));
    let case_b = report
        .lines()
        .find(|l| l.trim_start().starts_with("b "))
        .unwrap_or_else(|| panic!("no line for case b in:\n{report}"));

    assert!(case_b.contains("p50"), "{case_b}");
    assert!(case_b.contains("p90"), "{case_b}");
    // b does twice the work of a, and the band is the paired IQR around it.
    assert!(case_b.contains('['), "expected an IQR band: {case_b}");
    let ratio: f64 = case_b
        .split_whitespace()
        .find_map(|t| t.strip_suffix('x')?.parse().ok())
        .unwrap_or_else(|| panic!("no ratio in: {case_b}"));
    assert!(
        (1.4..3.0).contains(&ratio),
        "b should cost about 2x a, got {ratio}x"
    );
}

#[test]
fn a_lone_case_gets_no_ratio_column() {
    let capture = Capture::default();
    {
        let mut bench = Bench::with_config(fast(Format::Text)).report_to(Box::new(capture.clone()));
        let mut g = bench.group("solo");
        g.bench("only", |b| b.iter(|| work(std::hint::black_box(100))));
        g.finish();
    }
    let report = capture.text();
    assert!(report.contains("solo"), "{report}");
    assert!(
        !report.contains('x'),
        "a lone case has nothing to compare against:\n{report}"
    );
}

#[test]
fn tsv_is_one_row_per_case() {
    let (_, report) = schedule(fast(Format::Tsv));
    let mut lines = report.lines();
    assert!(lines.next().expect("comment").starts_with("# benchit"));
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    assert_eq!(header[0], "group");
    assert_eq!(header[1], "case");

    let rows: Vec<Vec<&str>> = lines.map(|l| l.split('\t').collect()).collect();
    assert_eq!(rows.len(), 2, "{report}");
    for row in &rows {
        assert_eq!(row.len(), header.len(), "row width must match the header");
        assert_eq!(row[0], "g");
    }
    assert_eq!(rows[0][1], "a");
    assert_eq!(rows[1][1], "b");
}

#[test]
fn the_filter_is_a_substring_of_group_slash_case() {
    let config = Config {
        filter: Some("g/b".into()),
        ..fast(Format::Text)
    };
    let (visits, report) = schedule(config);
    assert!(visits.iter().all(|&v| v == "b"), "{visits:?}");
    assert!(!report.contains("\n  a "), "{report}");
    // The surviving case becomes the reference, so it gets no ratio.
    assert!(!report.contains('['), "{report}");
}

#[test]
fn list_mode_measures_nothing() {
    let (visits, report) = schedule(Config {
        list: true,
        ..Config::default()
    });
    assert!(
        visits.is_empty(),
        "list mode must not call any benchmark body"
    );
    assert_eq!(
        report, "",
        "list mode writes names to stdout, not to the reporter"
    );
}

#[test]
fn throughput_is_reported_as_a_rate() {
    let capture = Capture::default();
    {
        let mut bench = Bench::with_config(fast(Format::Text)).report_to(Box::new(capture.clone()));
        let mut g = bench.group("thr");
        g.throughput(Throughput::Bytes(1 << 20));
        g.bench("case", |b| b.iter(|| work(std::hint::black_box(100))));
        g.finish();
    }
    let report = capture.text();
    assert!(
        report.contains("1.000 MiB"),
        "the declared amount:\n{report}"
    );
    assert!(report.contains("iB/s"), "the measured rate:\n{report}");
}

#[test]
fn throughput_applies_to_the_cases_registered_after_it() {
    let capture = Capture::default();
    let result = {
        let mut bench = Bench::with_config(fast(Format::Tsv)).report_to(Box::new(capture.clone()));
        let mut g = bench.group("sized");
        for n in [1_024u64, 4_096] {
            g.throughput(Throughput::Bytes(n));
            g.bench(format!("n={n}"), move |b| {
                b.iter(|| work(std::hint::black_box(100)).wrapping_add(n))
            });
        }
        g.finish()
    };

    // Each case keeps the amount declared before it: the second must not
    // inherit the first one's, nor the first the second's. Asserted on the
    // declarations rather than on the rates they produce, since the two bodies
    // are deliberately identical and a ratio of their timings would be a
    // statement about the machine.
    let declared: Vec<Option<Throughput>> = result.cases.iter().map(|c| c.throughput).collect();
    assert_eq!(
        declared,
        [
            Some(Throughput::Bytes(1_024)),
            Some(Throughput::Bytes(4_096))
        ]
    );

    // The rate column follows from that: same time, four times the amount.
    let report = capture.text();
    for case in &result.cases {
        let amount = case.throughput.expect("declared").amount() as f64;
        let rate = case.rate().expect("declared");
        assert!(
            (rate * case.stats.min * 1e-9 - amount).abs() < 1e-6,
            "{} reports {rate}/s for {amount}:\n{report}",
            case.name
        );
    }
}

#[test]
fn a_hand_built_config_with_zero_counts_does_not_panic() {
    // `Config`'s fields are public, so these bypass the command line's
    // validation. A zero block once spun forever collecting no samples and then
    // panicked on an empty slice.
    let capture = Capture::default();
    {
        let config = Config {
            block: 0,
            samples: 0,
            time_ms: 0,
            ..Config::default()
        };
        let mut bench = Bench::with_config(config).report_to(Box::new(capture.clone()));
        let mut g = bench.group("zeroes");
        g.bench("a", |b| b.iter(|| work(std::hint::black_box(50))));
        g.bench("b", |b| b.iter(|| work(std::hint::black_box(50))));
        g.finish();
    }
    assert!(capture.text().contains("zeroes"), "{}", capture.text());
}

#[test]
fn a_panicking_body_does_not_drag_the_whole_schedule_through_the_unwind() {
    // `Group::drop` runs the schedule, so a panic in user code between
    // registration and the end of scope must not kick off the whole run during
    // the unwind - and a second panic there would abort the process.
    let ran = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = ran.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let mut bench = Bench::with_config(fast(Format::Text));
        let mut g = bench.group("unwound");
        g.bench("a", move |b| {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            b.iter(|| work(std::hint::black_box(50)))
        });
        panic!("user code failed after registering");
    }));
    assert!(result.is_err(), "the user's panic must propagate");
    assert_eq!(
        ran.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "no benchmark body should run while unwinding"
    );
}

#[test]
fn the_ratio_sits_inside_its_own_band() {
    let (_, report) = schedule(fast(Format::Tsv));
    let row = report
        .lines()
        .find(|l| l.split('\t').nth(1) == Some("b"))
        .unwrap_or_else(|| panic!("no row for b in:\n{report}"));
    let field = |n: usize| -> f64 {
        row.split('\t')
            .nth(n)
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("field {n} in: {row}"))
    };
    let (point, lo, hi) = (field(7), field(8), field(9));
    assert!(
        lo <= point && point <= hi,
        "the point estimate must lie inside the band it is printed with: {lo} .. {point} .. {hi}"
    );
}

#[test]
fn a_case_that_beats_the_reference_keeps_its_digits() {
    // A sub-1.0 ratio once lost every significant digit and printed `0.00x`.
    let capture = Capture::default();
    {
        let mut bench = Bench::with_config(fast(Format::Text)).report_to(Box::new(capture.clone()));
        let mut g = bench.group("wins");
        g.bench("slow_reference", |b| {
            b.iter(|| work(std::hint::black_box(4_000)))
        });
        g.bench("fast", |b| b.iter(|| work(std::hint::black_box(4))));
        g.finish();
    }
    let report = capture.text();
    let line = report
        .lines()
        .find(|l| l.trim_start().starts_with("fast "))
        .unwrap_or_else(|| panic!("no line for fast in:\n{report}"));
    let ratio: f64 = line
        .split_whitespace()
        .find_map(|t| t.strip_suffix('x')?.parse().ok())
        .unwrap_or_else(|| panic!("no ratio in: {line}"));
    assert!(ratio > 0.0, "a winning case must not report 0.00x: {line}");
    assert!(ratio < 0.5, "expected a large win, got {ratio}x: {line}");
}

#[test]
fn iter_with_subtracts_its_own_clock_overhead() {
    // Timing each iteration separately costs a clock read per iteration. Left
    // in, it inflated a 1024-byte fill by ~17ns, which is more than the work.
    let capture = Capture::default();
    {
        // A near-zero timed span never reaches the 1ms calibration floor, so
        // only the wall-time cap stops the batch growing. A tight budget keeps
        // that quick.
        let config = Config {
            time_ms: 100,
            ..fast(Format::Tsv)
        };
        let mut bench = Bench::with_config(config).report_to(Box::new(capture.clone()));
        let mut g = bench.group("floor");
        g.bench("nothing_at_all", |b| b.iter_with(|| 1u64, |x| x));
        g.finish();
    }
    let report = capture.text();
    let min_ns: f64 = report
        .lines()
        .find(|l| l.split('\t').nth(1) == Some("nothing_at_all"))
        .and_then(|l| l.split('\t').nth(4))
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no min in:\n{report}"));
    // An empty body should land near zero, not near one clock read.
    assert!(
        min_ns < 8.0,
        "iter_with on an empty body reported {min_ns} ns; the clock overhead is not being subtracted"
    );
}

#[test]
fn iter_with_excludes_setup_from_the_timed_span() {
    let capture = Capture::default();
    {
        // Expensive setup, so this one gets a tight budget: `iter_with` pays
        // for setup in wall time on every iteration of every batch.
        let config = Config {
            time_ms: 50,
            ..fast(Format::Tsv)
        };
        let mut bench = Bench::with_config(config).report_to(Box::new(capture.clone()));
        let mut g = bench.group("setup");
        // Setup dwarfs the measured body; if it were being timed, the two
        // cases would look identical.
        g.bench("timed_body_only", |b| {
            b.iter_with(
                || work(std::hint::black_box(2_000)),
                |seed| seed.wrapping_add(1),
            )
        });
        g.bench("setup_included", |b| {
            b.iter(|| work(std::hint::black_box(2_000)).wrapping_add(1))
        });
        g.finish();
    }
    let report = capture.text();
    let min_ns = |case: &str| -> f64 {
        report
            .lines()
            .find(|l| l.split('\t').nth(1) == Some(case))
            .and_then(|l| l.split('\t').nth(4))
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("no min for {case} in:\n{report}"))
    };
    assert!(
        min_ns("timed_body_only") < min_ns("setup_included") / 2.0,
        "setup must fall outside the timed span:\n{report}"
    );
}

#[test]
fn finish_hands_back_the_numbers_it_printed() {
    let capture = Capture::default();
    let result = {
        let mut bench = Bench::with_config(fast(Format::Tsv)).report_to(Box::new(capture.clone()));
        let mut g = bench.group("g");
        g.throughput(Throughput::Elements(100));
        g.bench("cheap", |b| b.iter(|| work(std::hint::black_box(50))));
        g.bench("dear", |b| b.iter(|| work(std::hint::black_box(400))));
        g.finish()
    };

    assert_eq!(result.name, "g");
    let names: Vec<&str> = result.cases.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["cheap", "dear"]);

    let find = |name: &str| {
        result
            .cases
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("no case {name}"))
    };

    let cheap = find("cheap");
    assert!(cheap.is_reference);
    assert_eq!(cheap.ratio, None, "the reference has nothing to compare to");
    assert_eq!(cheap.samples.len(), 12);
    assert_eq!(cheap.throughput, Some(Throughput::Elements(100)));
    // A rate times the time one iteration took is the amount it declared.
    let per_iteration = cheap.rate().expect("declared a throughput") * cheap.stats.min * 1e-9;
    assert!((per_iteration - 100.0).abs() < 1e-9, "{per_iteration}");
    assert_eq!(cheap.delta(), None, "no baseline was loaded");

    let dear = find("dear");
    assert!(!dear.is_reference);
    assert!(dear.ratio.expect("a ratio against the reference").point > 1.0);

    // The same numbers reached the report, rather than the result carrying one
    // set and the printed table another.
    let report = capture.text();
    for c in &result.cases {
        assert!(
            report.contains(&format!("{:.3}", c.stats.min)),
            "{} is missing from:\n{report}",
            c.name
        );
    }
}

#[test]
fn a_group_that_measures_nothing_still_hands_back_a_result() {
    for config in [
        Config {
            filter: Some("no-such-case".to_string()),
            ..fast(Format::Text)
        },
        Config {
            list: true,
            ..fast(Format::Text)
        },
    ] {
        let capture = Capture::default();
        let result = {
            let mut bench = Bench::with_config(config).report_to(Box::new(capture.clone()));
            let mut g = bench.group("g");
            g.bench("a", |b| b.iter(|| work(std::hint::black_box(50))));
            g.finish()
        };
        assert_eq!(result.name, "g");
        assert!(result.cases.is_empty(), "{} cases", result.cases.len());
        assert_eq!(capture.text(), "", "nothing should have been measured");
    }
}

#[test]
fn finish_does_not_run_the_schedule_twice() {
    let capture = Capture::default();
    let result = {
        let mut bench = Bench::with_config(fast(Format::Tsv)).report_to(Box::new(capture.clone()));
        let mut g = bench.group("g");
        g.bench("a", |b| b.iter(|| work(std::hint::black_box(50))));
        // `g` is dropped at the end of this block, and must not measure again.
        g.finish()
    };
    assert_eq!(result.cases.len(), 1);

    let report = capture.text();
    let rows = report.lines().filter(|l| l.starts_with("g\t")).count();
    assert_eq!(rows, 1, "the group was reported twice:\n{report}");
}

#[test]
fn a_group_without_a_reference_prints_no_ratio_column() {
    let capture = Capture::default();
    let result = {
        let mut bench = Bench::with_config(fast(Format::Text)).report_to(Box::new(capture.clone()));
        let mut g = bench.group("codec");
        g.no_reference();
        g.bench("encode", |b| b.iter(|| work(std::hint::black_box(100))));
        g.bench("decode", |b| b.iter(|| work(std::hint::black_box(400))));
        g.finish()
    };
    let report = capture.text();

    // The timings are the reason the group exists, so they all stay.
    assert!(report.contains("encode"), "{report}");
    assert!(report.contains("decode"), "{report}");
    assert!(report.contains("p50"), "{report}");
    assert!(
        !report.contains('x'),
        "cases that are not alternatives must not be compared:\n{report}"
    );
    assert!(!report.contains('['), "no ratio means no band:\n{report}");

    // And the result says what the report says, rather than carrying a
    // comparison the printed table declined to make.
    for case in &result.cases {
        assert_eq!(case.ratio, None, "{} carries a ratio", case.name);
        assert!(!case.is_reference, "{} is a reference", case.name);
    }
}

#[test]
fn a_group_without_a_reference_leaves_the_tsv_ratio_columns_empty() {
    let capture = Capture::default();
    {
        let mut bench = Bench::with_config(fast(Format::Tsv)).report_to(Box::new(capture.clone()));
        let mut g = bench.group("codec");
        g.no_reference();
        g.bench("encode", |b| b.iter(|| work(std::hint::black_box(100))));
        g.bench("decode", |b| b.iter(|| work(std::hint::black_box(400))));
        g.finish();
    }
    let report = capture.text();

    let mut lines = report.lines();
    assert!(lines.next().expect("comment").starts_with("# benchit"));
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let ratio_columns = ["ratio", "ratio_lo", "ratio_hi"].map(|name| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("no {name} column in: {header:?}"))
    });
    let min_ns = header
        .iter()
        .position(|h| *h == "min_ns")
        .expect("a min_ns column");

    let rows: Vec<Vec<&str>> = lines.map(|l| l.split('\t').collect()).collect();
    assert_eq!(rows.len(), 2, "{report}");
    for row in &rows {
        assert_eq!(row.len(), header.len(), "row width must match the header");
        assert!(
            !row[min_ns].is_empty(),
            "the timings still print:\n{report}"
        );
        for column in ratio_columns {
            assert_eq!(
                row[column], "",
                "{} must report no ratio:\n{report}",
                row[1]
            );
        }
    }
}
