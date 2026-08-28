//! The measurement core: the `Bencher` protocol, timer calibration, and batch
//! size calibration.

use std::hint::black_box;
use std::time::{Duration, Instant};

/// The floor a calibrated batch must clear. About 25000x a timer round trip,
/// which puts timer error under 0.01% of the reported number.
const BATCH_FLOOR: Duration = Duration::from_millis(1);

/// The fewest samples a case is allowed to contribute. A batch is capped so
/// that even a slow operation yields at least this many samples per case.
pub(crate) const MIN_SAMPLES: usize = 10;

/// The handle passed to a benchmark body.
///
/// The scheduler presets the iteration count, calls the registered closure
/// once, and reads the elapsed time back out. A benchmark body's only job is
/// to call [`Bencher::iter`] (or [`Bencher::iter_with`]) exactly once:
///
/// ```
/// # let mut bench = benchit::Bench::with_config(benchit::Config { list: true, ..Default::default() });
/// # let mut group = bench.group("example");
/// group.bench("sum", |b| b.iter(|| (0..100u64).sum::<u64>()));
/// ```
///
/// This is the same protocol criterion uses, which is what keeps benchmark
/// bodies portable between the two.
pub struct Bencher {
    iters: u64,
    /// The per-iteration cost of the clock reads `iter_with` performs inside
    /// its own timed span, measured on this machine. See [`iter_with_floor`].
    floor_ns: f64,
    elapsed: Option<Duration>,
}

impl Bencher {
    pub(crate) fn new(iters: u64, floor_ns: f64) -> Self {
        Self {
            iters,
            floor_ns,
            elapsed: None,
        }
    }

    /// Time `f` over the whole batch and divide.
    ///
    /// The return value is passed through [`black_box`] so the body cannot be
    /// eliminated as dead code; benchmark bodies do not have to remember to do
    /// this themselves. Black-boxing the *input* is still the caller's job.
    ///
    /// The drop of `f`'s return value happens inside the timed span. A closure
    /// that allocates and returns a `Vec` is timing the free as well; use
    /// [`iter_with`](Self::iter_with) when that is not what you meant to
    /// measure.
    pub fn iter<T>(&mut self, mut f: impl FnMut() -> T) {
        let start = Instant::now();
        for _ in 0..self.iters {
            black_box(f());
        }
        self.elapsed = Some(start.elapsed());
    }

    /// Time `f` with per-iteration setup, excluding the setup, the drop of the
    /// input, and the drop of the result from the timed span.
    ///
    /// ```
    /// # let mut bench = benchit::Bench::with_config(benchit::Config { list: true, ..Default::default() });
    /// # let mut group = bench.group("example");
    /// group.bench("encode", |b| {
    ///     b.iter_with(|| vec![0u8; 1024], |input| input.len())
    /// });
    /// ```
    ///
    /// Excluding those costs requires timing each iteration separately rather
    /// than the batch as a whole, so every iteration pays for the clock reads
    /// that bracket it. That cost is measured at startup and subtracted (the
    /// run header prints it), but a correction is not a measurement: the
    /// residual error is a nanosecond or two, which is nothing at a microsecond
    /// and everything at ten nanoseconds. [`iter`](Self::iter) amortizes one
    /// clock pair over the whole batch and needs no correction, so prefer it
    /// whenever the drops are not what you are trying to exclude.
    pub fn iter_with<S, T>(&mut self, mut setup: impl FnMut() -> S, mut f: impl FnMut(S) -> T) {
        let mut total = Duration::ZERO;
        for _ in 0..self.iters {
            let input = black_box(setup());
            let start = Instant::now();
            let out = f(input);
            total += start.elapsed();
            drop(black_box(out));
        }
        let overhead = Duration::from_nanos((self.floor_ns * self.iters as f64) as u64);
        self.elapsed = Some(total.saturating_sub(overhead));
    }

    fn take(self, case: &str) -> Duration {
        self.elapsed.unwrap_or_else(|| {
            panic!(
                "benchmark `{case}` never timed anything: its body must call `iter` or `iter_with`"
            )
        })
    }
}

/// A benchmark body: anything the scheduler can call with a preset `Bencher`.
pub(crate) type Body<'b> = Box<dyn FnMut(&mut Bencher) + 'b>;

/// The two costs of one batch.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Batch {
    /// What the body reported, and the only thing that reaches the report.
    pub timed: Duration,
    /// What the batch cost the run. With [`Bencher::iter_with`] this includes
    /// the setup, which is excluded from `timed` but very much not free; the
    /// scheduler budgets against this so that a benchmark with expensive setup
    /// still finishes in the time it was given.
    pub wall: Duration,
}

/// Run one batch of `iters` iterations.
pub(crate) fn run_batch(
    case: &str,
    body: &mut dyn FnMut(&mut Bencher),
    iters: u64,
    floor_ns: f64,
) -> Batch {
    let mut b = Bencher::new(iters, floor_ns);
    let start = Instant::now();
    body(&mut b);
    let wall = start.elapsed();
    Batch {
        timed: b.take(case),
        wall,
    }
}

/// Measure the cost of one `Instant::now()` + `elapsed()` round trip.
///
/// This is the noise floor of everything the harness reports, and it varies
/// enough by platform and clock source that assuming a number is not safe.
/// Measured as a minimum over batches, for the same reason benchmarks report
/// one: the noise here is one-sided.
pub(crate) fn timer_round_trip() -> f64 {
    const BATCH: u64 = 1000;
    let mut best = f64::INFINITY;
    for _ in 0..20 {
        let start = Instant::now();
        for _ in 0..BATCH {
            let t = Instant::now();
            black_box(t.elapsed());
        }
        let per_call = start.elapsed().as_nanos() as f64 / BATCH as f64;
        if per_call < best {
            best = per_call;
        }
    }
    best
}

/// Measure what [`Bencher::iter_with`] spends on clock reads inside its own
/// timed span, per iteration.
///
/// This is *not* the full round trip from [`timer_round_trip`]. `iter_with`
/// starts its span in the middle of one `Instant::now()` and ends it in the
/// middle of the next, so it captures roughly one clock read rather than two,
/// and on this machine the two numbers differ by more than a factor of two.
/// Correcting with the wrong one would be worse than not correcting at all,
/// which is why this is measured with the loop shape it is correcting.
pub(crate) fn iter_with_floor() -> f64 {
    const BATCH: u64 = 1000;
    let mut best = f64::INFINITY;
    for _ in 0..20 {
        let mut total = Duration::ZERO;
        for _ in 0..BATCH {
            // The span `iter_with` opens, with nothing at all inside it.
            let start = Instant::now();
            black_box(());
            total += start.elapsed();
        }
        let per_iteration = total.as_nanos() as f64 / BATCH as f64;
        if per_iteration < best {
            best = per_iteration;
        }
    }
    best
}

/// Choose a batch size for one case, and warm it up in the process.
///
/// Doubles the iteration count until a batch clears [`BATCH_FLOOR`], then
/// re-times that batch: a closure that gets faster as caches, branch
/// predictors, and the allocator warm up can clear the floor on a cold batch
/// and fall back under it once warm, and keeping the smaller `k` would leave
/// timer error in every sample that follows.
///
/// `max_batch` caps the result so a slow operation still yields multiple
/// samples rather than one enormous one.
///
/// Returns the batch size and how long that batch took, which is what the
/// scheduler divides the time budget by.
pub(crate) fn calibrate(
    case: &str,
    body: &mut dyn FnMut(&mut Bencher),
    max_batch: Duration,
    floor_ns: f64,
) -> (u64, Duration) {
    let mut iters: u64 = 1;
    loop {
        let batch = run_batch(case, body, iters, floor_ns);
        // The cap is on wall time, since that is what the run has to pay for.
        if batch.wall >= max_batch {
            return (iters, batch.wall);
        }
        // The floor is on the timed span, since that is what timer error is
        // measured against.
        if batch.timed >= BATCH_FLOOR {
            // Confirm now that the case is warm.
            let confirm = run_batch(case, body, iters, floor_ns);
            if confirm.timed >= BATCH_FLOOR {
                return (iters, confirm.wall);
            }
        }
        let Some(next) = iters.checked_mul(2) else {
            return (iters, batch.wall);
        };
        iters = next;
    }
}
