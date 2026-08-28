# benchit

A minimal benchmarking harness for Rust: no dependencies, no configuration, and a comparison you can trust.

```toml
[dev-dependencies]
benchit = "0.1"

[[bench]]
name = "decode"
harness = false
```

`harness = false` already means `main` is yours, so there are no macros:

```rust
use benchit::{Bench, Throughput};
use std::hint::black_box;

fn main() {
    let mut bench = Bench::from_args();
    let input = make_input(1 << 20);

    let mut g = bench.group("decode/1MiB");
    g.throughput(Throughput::Bytes(input.len() as u64));
    g.bench("mine", |b| b.iter(|| mycrate::decode(black_box(&input))));
    g.bench("theirs", |b| b.iter(|| othercrate::decode(black_box(&input))));
    g.finish();
}
```

```
benchit: timer 42.17 ns/call, iter_with floor 17.62 ns/iter
         up to 50 samples in 1000 ms per case, interleaved

lookup/10000_keys  1 elem
  hashmap      6.018 ns   p50 6.258 ns   p90 6.358 ns   166.2 Melem/s   1.00x
  btreemap     19.70 ns   p50 20.28 ns   p90 20.70 ns   50.77 Melem/s   3.27x  [3.24 .. 3.29]
  linear_scan  1.615 us   p50 1.672 us   p90 1.714 us   619.3 Kelem/s    269x  [264 .. 273]
```

## What makes it different

**The cases in a group are interleaved.** Most benchmarking compares alternatives within a single run: two implementations of the same operation, a fast path against its fallback, your crate against the one you are trying to beat. A harness that runs case A to completion and then case B puts every bit of frequency scaling, thermal throttling, and background load that happened in between directly onto the ratio you came to read.

So a group is run in rounds: one sample of each case, then the next round. Round `i` yields `A_i` and `B_i` measured milliseconds apart, under the same thermal state and the same background load. Drift cancels inside every pair.

The reported ratio and the bracket after it are the median and interquartile range of those per-round paired ratios: a noise band that was measured rather than modelled, with no resampling and no distributional assumption. Both are order statistics of the same paired quantity, so the point always sits inside its own band. (A ratio of the two displayed minima would be a different estimator, and in practice lands outside the band a few percent of the time — the same size as the effects the tool exists to detect.) Under `--no-interleave` there is nothing to pair, so the ratio falls back to `min / min` and the bracket is omitted.

**The minimum leads.** For deterministic CPU-bound code, measurement noise is one-sided: interrupts, preemption, frequency transitions, and cache pollution all add time and none subtract it. Under that model the minimum is the best estimator of true cost and the most sensitive detector of a real change. That assumption does not always hold, which is why p50 and p90 sit beside it and never replace it. A wide min-to-p90 spread is the signal that the workload itself is variable and the median is the number to read.

**Nothing needs configuring.** The batch size is calibrated per case by doubling until a batch clears 1ms, which simultaneously sets the batch size, warms caches and branch predictors, warms the allocator, and gets the CPU off its idle frequency. A nanosecond benchmark and a hundred-millisecond benchmark both work with no per-benchmark settings.

**It is fast enough to run in the edit loop.** The demo's eleven cases finish in about two seconds. Add `--quick` when even that is too slow.

## What falls inside the timed span

`iter` times the closure including the drop of its return value, so a closure that allocates and returns a `Vec` is timing the free as well. When that is not what you meant to measure, `iter_with` puts the setup, the drop of the input, and the drop of the result all outside the timed span:

```rust
g.bench("encode", |b| b.iter_with(|| make_input(), |input| encode(&input)));
```

Excluding those costs means timing each iteration separately instead of the batch as a whole, so every iteration pays for the clock reads that bracket it. That cost is measured at startup and subtracted, and the run header prints it:

```
benchit: timer 42.17 ns/call, iter_with floor 17.62 ns/iter
```

A correction is not a measurement, though. The residual is a nanosecond or two — nothing at a microsecond, everything at ten nanoseconds. `iter` amortizes one clock pair over a whole batch and needs no correction, so prefer it unless the drops are the thing you are excluding.

`iter` passes the closure's return value through `black_box` for you. Black-boxing the *input*, to defeat constant folding, is still yours to do.

## Parameterized cases

`BenchmarkId::new("sort", n)` is just `format!("sort/{n}")` once the name is `impl Display`, and `throughput` applies to the cases registered after it, so each size can declare its own amount:

```rust
let mut g = bench.group("sort_unstable");
for n in [64usize, 1024, 16384] {
    let data = make_data(n);
    g.throughput(Throughput::Elements(n as u64));
    g.bench(format!("n={n}"), move |b| {
        b.iter_with(|| data.clone(), |mut v| { v.sort_unstable(); v })
    });
}
g.finish();
```

```
sort_unstable
  n=64     30.17 ns   p50 31.04 ns   p90 31.78 ns   2.121 Gelem/s   1.00x
  n=1024   345.8 ns   p50 355.3 ns   p90 369.1 ns   2.961 Gelem/s   11.5x  [11.3 .. 11.7]
  n=16384  5.163 us   p50 5.357 us   p90 5.598 us   3.173 Gelem/s    171x  [169 .. 176]
```

The group header carries the declared amount only when every case agrees on it; otherwise the per-case rate column is the whole story.

## Command line

```
<bench binary> [FILTER] [OPTIONS]

  FILTER                substring match on "group/case"

  --quick               ~200ms per benchmark
  --samples N           cap on samples per case (default 50)
  --time MS             per-benchmark budget in ms (default 1000)
  --block N             samples per visit when interleaving (default 1)
  --no-interleave       run each case to completion instead of in rounds
  --save-baseline NAME  write benchit/NAME.tsv beside the built binary
  --baseline NAME       load benchit/NAME.tsv from there, show a delta column
  --format=text|tsv     output format (default text)
  --list                list matching benchmarks without running them
  -h, --help            this message
```

`--samples` is a cap and `--time` is a ceiling, with one exception: a fast case stops at 50 samples long before it reaches its budget, and a slow one stops at its budget — but every case gets at least 10 samples even when that overruns the budget, because fewer than that makes the order statistics meaningless. An operation that takes longer than a tenth of the budget will therefore overrun it. Both limits are measured in wall time, so a benchmark with expensive `iter_with` setup still finishes near the time it was given.

Interleaving has one real cost: with large working sets, each round evicts the previous case's data, so every sample is cold-cache. That is consistent across the group and arguably the more honest number, but for a benchmark deliberately measuring hot-cache behaviour it is wrong. Use `--block N` to take N samples per visit, or `--no-interleave` for one-case-at-a-time ordering.

## Reading the numbers back

`Group::finish` returns the `GroupResult` it just printed, so a benchmark that needs a metric this crate has no opinion about computes it from the samples instead of parsing `--format=tsv` back out of a pipe:

```rust
let result = group.finish();
for case in &result.cases {
    let cpu_seconds = case.stats.min * 1e-9;
    eprintln!("{}: {:.1}x realtime", case.name, SECONDS_PER_BUFFER / cpu_seconds);
}
```

Every case carries its `samples` in round order, its `stats` (`min`, `p50`, `p90`), its `ratio` against the group's reference, and its `baseline` if one was loaded. `rate()` and `delta()` are computed from `min`, matching the printed columns; reach for `stats.p50` instead when the question is whether a budget holds in practice rather than at best. Round order is the part worth knowing: index `i` of one case was measured in the same round as index `i` of every other, so a derived metric can be paired the same way the ratio is. Print to stderr, as above, if the run might be asked for `--format=tsv`.

A group filtered out by the command line, or a run under `--list`, returns a result with no cases rather than nothing at all. Check for that if you gate on the numbers: "nothing was measured" and "nothing measured badly" are the same empty list, so a mistyped filter would otherwise turn a failing gate green.

## Baselines

`--save-baseline main` writes `benchit/main.tsv` into the directory cargo built the bench binary into; `--baseline main` loads it and adds a delta column against the saved minimum. The format is TSV, sorted by group then case, so it diffs cleanly in git and is readable without a tool.

That directory is found from the binary's own path — the nearest ancestor holding a `deps/` directory, which covers bench and test binaries, examples, and a `[[bin]]` run with `cargo run` — so `build.target-dir` in `.cargo/config.toml`, a `--target-dir` flag, and a workspace all resolve correctly. Each build keeps its own baselines: a debug run cannot overwrite what a release run saved, and a `--target` build keeps a separate file. `CARGO_TARGET_DIR` is a fallback for layouts that are not cargo-shaped, and a relative path in the "baseline saved to" line is the sign that no build directory was found.

Saving merges: cases measured by this run replace their existing rows, and rows the run did not measure are carried across untouched. So saving from a filtered run updates just those cases instead of discarding every baseline the filter excluded.

Cross-run comparison is inherently weaker than within-run pairing, because nothing about the two runs is paired. So the delta is reported plainly, without a verdict: a harness that announces "performance has regressed" on a 2% cross-run change is a false-confidence machine.

## Migrating from criterion

| criterion | benchit |
| --- | --- |
| `criterion_group!` + `criterion_main!` | `fn main`, delete both macros |
| `c.benchmark_group(name)` | `bench.group(name)` |
| `group.bench_function(name, ...)` | `group.bench(name, ...)` |
| `BenchmarkId::new(name, param)` | `format!("{name}/{param}")` |
| `group.bench_with_input(id, &input, \|b, input\| ...)` | `group.bench(id, \|b\| ...)`, capture the input |
| `criterion::black_box` | `std::hint::black_box` |
| `group.finish()` | unchanged, or drop it and let scope end |
| `b.iter(...)` | unchanged |
| `group.throughput(Throughput::Bytes(n))` | unchanged |

The positional filter is a substring match on `group/case`, as in criterion, so existing muscle memory and CI invocations transfer unchanged.

Expect one number to move on the way across: any benchmark that allocates inside its timed closure will get faster when converted to `iter_with`, because it stops counting a deallocation it was never meant to measure. That is a correction, not a regression.

One thing does not transfer. Cases are registered, not run: an interleaved schedule cannot exist until the group has seen all of them, so every closure in a group is live at once. Two cases that each capture `&mut scratch` will not compile. Share it through a `RefCell` and borrow outside the `iter` call, or give each case its own.

## Out of scope

HTML reports and plots. Bootstrap resampling, confidence intervals, and p-values. Automatic regression verdicts. Async benchmarks, hardware counters, and multi-threaded scaling.

This is a harness for in-process, CPU-bound microbenchmarks: functions that take between a nanosecond and a few hundred milliseconds and run deterministically. Anything dominated by I/O, network, or another process is out of scope, since the statistics here assume the machine, not the workload, is the source of variance.

## License

MIT OR Apache-2.0, at your option.
