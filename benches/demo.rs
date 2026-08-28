//! A demonstration of the harness against workloads whose relative costs are
//! already known, so the numbers can be sanity-checked by eye.
//!
//! Run with `cargo bench`, or `cargo bench -- --quick lookup`.

use std::collections::{BTreeMap, HashMap};
use std::hint::black_box;

use benchit::{Bench, Throughput};

fn main() {
    let mut bench = Bench::from_args();

    bytes(&mut bench);
    lookup(&mut bench);
    allocation(&mut bench);
    scaling(&mut bench);
}

/// Three ways to add up a megabyte, which should land within a few percent of
/// each other.
fn bytes(bench: &mut Bench) {
    let input = vec![7u8; 1 << 20];

    let mut g = bench.group("sum/1MiB");
    g.throughput(Throughput::Bytes(input.len() as u64));
    g.bench("iter_sum", |b| {
        b.iter(|| black_box(&input).iter().map(|&x| x as u64).sum::<u64>())
    });
    g.bench("fold", |b| {
        b.iter(|| black_box(&input).iter().fold(0u64, |a, &x| a + x as u64))
    });
    g.bench("index_loop", |b| {
        b.iter(|| {
            let v = black_box(&input);
            let mut total = 0u64;
            // Indexing is the thing being measured against the iterator forms.
            #[allow(clippy::needless_range_loop)]
            for i in 0..v.len() {
                total += v[i] as u64;
            }
            total
        })
    });
    g.finish();
}

/// A comparison with an answer everyone already knows, which is what makes it
/// a good check on the harness.
fn lookup(bench: &mut Bench) {
    const N: u64 = 10_000;
    let keys: Vec<u64> = (0..N).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    let hash: HashMap<u64, u64> = keys.iter().map(|&k| (k, k)).collect();
    let tree: BTreeMap<u64, u64> = keys.iter().map(|&k| (k, k)).collect();
    let needle = keys[keys.len() / 2];

    let mut g = bench.group(format!("lookup/{N}_keys"));
    g.throughput(Throughput::Elements(1));
    g.bench("hashmap", |b| {
        b.iter(|| hash.get(black_box(&needle)).copied())
    });
    g.bench("btreemap", |b| {
        b.iter(|| tree.get(black_box(&needle)).copied())
    });
    g.bench("linear_scan", |b| {
        b.iter(|| keys.iter().find(|&&k| k == black_box(needle)).copied())
    });
    g.finish();
}

/// `iter` times the drop of the returned `Vec`; `iter_with` does not. The gap
/// between these two is exactly the deallocation a "how fast is building this"
/// benchmark did not mean to measure.
fn allocation(bench: &mut Bench) {
    let mut g = bench.group("vec/1024");
    g.bench("build_and_drop", |b| b.iter(|| vec![black_box(0u8); 1024]));
    g.bench("fill_only", |b| {
        b.iter_with(
            || vec![0u8; 1024],
            |mut v| {
                v.fill(black_box(1));
                v
            },
        )
    });
    g.finish();
}

/// Parameterized cases: criterion's `BenchmarkId::new("sort", n)` is just
/// `format!("sort/{n}")` once the name is `impl Display`.
///
/// Each case owns its own input, and `iter_with` hands the body a fresh
/// unsorted copy every iteration, so this measures sorting rather than the
/// cost of re-sorting an already sorted vector.
fn scaling(bench: &mut Bench) {
    let mut g = bench.group("sort_unstable");
    for n in [64usize, 1_024, 16_384] {
        let data: Vec<u64> = (0..n as u64)
            .map(|i| i.wrapping_mul(2_654_435_761) ^ (i << 17))
            .collect();
        // Per case, so each size gets its own rate rather than the last
        // size's amount applied to all of them.
        g.throughput(Throughput::Elements(n as u64));
        g.bench(format!("n={n}"), move |b| {
            b.iter_with(
                || data.clone(),
                |mut v| {
                    v.sort_unstable();
                    v
                },
            )
        });
    }

    // Whether the cost really grows as n log n is a question the table cannot
    // answer, so the benchmark answers it from the returned result. On stderr,
    // so `--format=tsv` stays machine-readable.
    let result = g.finish();
    for case in &result.cases {
        let Some(n) = case.throughput.map(|t| t.amount() as f64) else {
            continue;
        };
        eprintln!(
            "  {:<9} {:>6.3} ns/elem   {:>6.4} ns per n log2 n",
            case.name,
            case.stats.min / n,
            case.stats.min / (n * n.log2()),
        );
    }
}
