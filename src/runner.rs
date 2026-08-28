//! The scheduler: calibrate every case in a group, then measure them in
//! interleaved rounds.
//!
//! Why rounds rather than one case at a time is the crate-level documentation's
//! argument; what matters here is the consequence. Samples must stay in round
//! order, and index `i` of one case must be the same round as index `i` of
//! another, because that is what [`crate::stats::paired_ratio`] pairs on.

use std::io::Write;
use std::time::Duration;

use crate::bencher::{self, Body, MIN_SAMPLES};
use crate::report::{CaseResult, GroupResult};
use crate::stats::{self, Ratio, Stats};
use crate::{Bench, Throughput};

/// A registered case, waiting for the group to build a schedule.
pub(crate) struct Case<'b> {
    pub name: String,
    pub throughput: Option<Throughput>,
    pub body: Body<'b>,
}

/// What calibration decided for one case, and where its samples land.
struct Plan<'a, 'b> {
    case: &'a mut Case<'b>,
    full_name: String,
    iters: u64,
    target_samples: usize,
    /// Per-iteration nanoseconds, in round order.
    samples: Vec<f64>,
}

impl Plan<'_, '_> {
    fn take_sample(&mut self, floor_ns: f64) {
        let batch = bencher::run_batch(&self.full_name, &mut self.case.body, self.iters, floor_ns);
        self.samples
            .push(batch.timed.as_nanos() as f64 / self.iters as f64);
    }

    fn wants_more(&self) -> bool {
        self.samples.len() < self.target_samples
    }
}

pub(crate) fn run(bench: &mut Bench, group_name: &str, cases: &mut [Case<'_>]) -> GroupResult {
    // A group that measured nothing still returns a result; see `Group::finish`.
    let nothing = || GroupResult {
        name: group_name.to_string(),
        throughput: None,
        cases: Vec::new(),
    };

    let matches = |case: &Case<'_>| bench.config.matches(&full_name(group_name, &case.name));
    if !cases.iter().any(matches) {
        return nothing();
    }

    if bench.config.list {
        let mut out = std::io::stdout().lock();
        for case in cases.iter().filter(|c| matches(c)) {
            let _ = writeln!(out, "{}", full_name(group_name, &case.name));
        }
        return nothing();
    }

    bench.begin_run();
    let floor_ns = bench.iter_with_floor_ns;

    let budget = Duration::from_millis(bench.config.time_ms);
    // Cap a single batch so even a slow operation yields several samples.
    let max_batch = budget / MIN_SAMPLES as u32;
    // Both the cap and the sample count are in wall time, so a case with
    // expensive `iter_with` setup still finishes inside its budget.

    let mut plans: Vec<Plan> = cases
        .iter_mut()
        .filter(|case| bench.config.matches(&full_name(group_name, &case.name)))
        .map(|case| {
            let full = full_name(group_name, &case.name);
            let (iters, wall) = bencher::calibrate(&full, &mut case.body, max_batch, floor_ns);
            let target = target_samples(wall, budget, bench.config.samples);
            Plan {
                case,
                full_name: full,
                iters,
                target_samples: target,
                // Reserved up front so the run does no allocation it can avoid.
                // A pathological `--samples` is capped here rather than trusted
                // to `Vec`, which would abort on a capacity overflow.
                samples: Vec::with_capacity(target.min(4096)),
            }
        })
        .collect();

    if bench.config.interleave {
        measure_interleaved(&mut plans, bench.config.block, floor_ns);
    } else {
        measure_sequentially(&mut plans, floor_ns);
    }

    let result = summarize(bench, group_name, plans);
    if bench.config.save_baseline.is_some() {
        bench.record(&result);
    }
    bench.reporter.group(&result);
    result
}

fn full_name(group: &str, case: &str) -> String {
    format!("{group}/{case}")
}

/// How many samples fit in the budget, floored so a slow case still gets
/// enough for order statistics to mean anything.
///
/// The floor gives way to an explicit `--samples`: someone who asked for 5 gets
/// 5, and overrunning the budget they set would be worse than a thin sample.
fn target_samples(batch: Duration, budget: Duration, cap: usize) -> usize {
    let cap = cap.max(1);
    let floor = MIN_SAMPLES.min(cap);
    let batch_ns = batch.as_nanos().max(1);
    // Saturating rather than `as usize`, which would wrap on a 32-bit target
    // and silently thin the sampling.
    let fits = usize::try_from(budget.as_nanos() / batch_ns).unwrap_or(usize::MAX);
    fits.clamp(floor, cap)
}

/// One sample of every case, then the next round.
fn measure_interleaved(plans: &mut [Plan], block: usize, floor_ns: f64) {
    // `block` is clamped rather than trusted: `Config`'s fields are public, and
    // a zero here would spin without ever taking a sample.
    let block = block.max(1);
    while plans.iter().any(Plan::wants_more) {
        for plan in plans.iter_mut() {
            for _ in 0..block {
                if !plan.wants_more() {
                    break;
                }
                plan.take_sample(floor_ns);
            }
        }
    }
}

/// Criterion's ordering: each case run to completion. Unpaired, and exposed to
/// every bit of drift between one case and the next, but it is the right answer
/// when the benchmark is deliberately measuring hot-cache behaviour.
fn measure_sequentially(plans: &mut [Plan], floor_ns: f64) {
    for plan in plans.iter_mut() {
        while plan.wants_more() {
            plan.take_sample(floor_ns);
        }
    }
}

fn summarize(bench: &Bench, group_name: &str, plans: Vec<Plan<'_, '_>>) -> GroupResult {
    // The group header can only carry one amount, so it carries the declared
    // amount when every case agrees and nothing when they differ. A
    // parameterized group still gets a per-case rate column.
    let throughput = match plans.first().map(|p| p.case.throughput) {
        Some(first) if plans.iter().all(|p| p.case.throughput == first) => first,
        _ => None,
    };
    // The reference is the first case that passed the filter, which is
    // conventionally the implementation the others are being compared against.
    let reference: Vec<f64> = plans.first().map(|p| p.samples.clone()).unwrap_or_default();
    let reference_min = plans
        .first()
        .map(|p| Stats::from_sorted(&stats::sorted_copy(&p.samples)).min)
        .unwrap_or(f64::NAN);

    let cases = plans
        .into_iter()
        .enumerate()
        .map(|(i, plan)| {
            let sorted = stats::sorted_copy(&plan.samples);
            let case_stats = Stats::from_sorted(&sorted);
            // Pairing only means something when the samples were interleaved;
            // sequential samples share an index but nothing else, so those fall
            // back to a ratio of the two reported minima.
            let ratio = (i > 0)
                .then(|| {
                    bench
                        .config
                        .interleave
                        .then(|| stats::paired_ratio(&plan.samples, &reference))
                        .flatten()
                        .or_else(|| {
                            (reference_min > 0.0).then(|| Ratio {
                                point: case_stats.min / reference_min,
                                iqr: None,
                            })
                        })
                })
                .flatten();
            let name = std::mem::take(&mut plan.case.name);
            CaseResult {
                baseline: bench.saved_stats(&plan.full_name),
                is_reference: i == 0,
                throughput: plan.case.throughput,
                name,
                iters: plan.iters,
                samples: plan.samples,
                stats: case_stats,
                ratio,
            }
        })
        .collect();

    GroupResult {
        name: group_name.to_string(),
        throughput,
        cases,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_fill_the_budget_but_respect_the_floor_and_cap() {
        let budget = Duration::from_millis(1000);
        // A 1ms batch could run 1000 times; the cap holds it to 50.
        assert_eq!(target_samples(Duration::from_millis(1), budget, 50), 50);
        // A 100ms batch fits 10 times, which is also the floor.
        assert_eq!(target_samples(Duration::from_millis(100), budget, 50), 10);
        // A batch longer than the whole budget still gets the floor.
        assert_eq!(
            target_samples(Duration::from_millis(5000), budget, 50),
            MIN_SAMPLES
        );
        // A zero-length batch must not divide by zero.
        assert_eq!(target_samples(Duration::ZERO, budget, 50), 50);
    }

    #[test]
    fn an_explicit_sample_cap_below_the_floor_is_honoured() {
        let budget = Duration::from_millis(1000);
        assert_eq!(target_samples(Duration::from_millis(1), budget, 5), 5);
        assert_eq!(target_samples(Duration::from_millis(500), budget, 5), 5);
        assert_eq!(target_samples(Duration::from_millis(1), budget, 0), 1);
    }
}
