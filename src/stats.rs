//! Order statistics over collected samples.
//!
//! Everything here is computed from numbers that were actually measured. There
//! is no bootstrap, no confidence interval, and no distributional assumption,
//! which is also why this module needs no random number generator.

/// The quantile of an already-sorted slice, linearly interpolated between the
/// two neighbouring samples.
///
/// # Panics
///
/// Panics if `sorted` is empty.
pub(crate) fn quantile(sorted: &[f64], q: f64) -> f64 {
    assert!(!sorted.is_empty(), "quantile of an empty sample set");
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

/// Sort a *copy* of the samples, so the caller's recording buffer keeps its
/// round order for `--format=tsv` and for pairing.
pub(crate) fn sorted_copy(samples: &[f64]) -> Vec<f64> {
    let mut v = samples.to_vec();
    v.sort_unstable_by(f64::total_cmp);
    v
}

/// The three numbers reported for every case, in nanoseconds per iteration.
///
/// The minimum leads because noise on deterministic CPU-bound code is one-sided;
/// the median and p90 sit beside it because that assumption does not always
/// hold, and the spread between them is what tells the two cases apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Stats {
    pub min: f64,
    pub p50: f64,
    pub p90: f64,
}

impl Stats {
    /// # Panics
    ///
    /// Panics if `samples` is empty.
    pub fn from_sorted(sorted: &[f64]) -> Self {
        Self {
            min: sorted[0],
            p50: quantile(sorted, 0.50),
            p90: quantile(sorted, 0.90),
        }
    }
}

/// A case's cost relative to the group's reference case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Ratio {
    /// The reported ratio: the median of the per-round paired ratios when the
    /// group was interleaved, and `case.min / reference.min` otherwise.
    pub point: f64,
    /// The interquartile range of the per-round paired ratios.
    ///
    /// This is the payoff of interleaving made visible: measured spread in the
    /// quantity being compared, obtained without resampling. It brackets
    /// `point` because both are order statistics of the same paired ratios;
    /// quoting a min-of-A over min-of-B point estimate beside it would put the
    /// two numbers on different footings and routinely place the point outside
    /// its own band.
    pub iqr: Option<(f64, f64)>,
}

/// Pair round `i` of `case` with round `i` of `reference` and take order
/// statistics of the per-round ratios.
///
/// Pairing is what makes drift cancel: both halves of every ratio were measured
/// milliseconds apart, under the same thermal state, clock frequency, and
/// background load. A machine that slows 5% over a run leaves these untouched.
///
/// The interquartile range needs four pairs to mean anything; the median does
/// not, so a very short run still gets a ratio.
pub(crate) fn paired_ratio(case: &[f64], reference: &[f64]) -> Option<Ratio> {
    let n = case.len().min(reference.len());
    let mut ratios: Vec<f64> = (0..n)
        .filter(|&i| reference[i] > 0.0)
        .map(|i| case[i] / reference[i])
        .collect();
    if ratios.is_empty() {
        return None;
    }
    ratios.sort_unstable_by(f64::total_cmp);
    Some(Ratio {
        point: quantile(&ratios, 0.50),
        iqr: (ratios.len() >= 4).then(|| (quantile(&ratios, 0.25), quantile(&ratios, 0.75))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_endpoints_and_interpolation() {
        let s = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile(&s, 0.0), 1.0);
        assert_eq!(quantile(&s, 1.0), 5.0);
        assert_eq!(quantile(&s, 0.5), 3.0);
        // 0.9 * 4 = 3.6 -> between 4.0 and 5.0
        assert!((quantile(&s, 0.9) - 4.6).abs() < 1e-12);
    }

    #[test]
    fn quantile_of_single_sample() {
        assert_eq!(quantile(&[7.0], 0.9), 7.0);
    }

    #[test]
    fn stats_read_off_a_sorted_slice() {
        let sorted = sorted_copy(&[5.0, 1.0, 4.0, 2.0, 3.0]);
        assert_eq!(sorted, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let st = Stats::from_sorted(&sorted);
        assert_eq!(st.min, 1.0);
        assert_eq!(st.p50, 3.0);
    }

    #[test]
    fn sorting_does_not_disturb_round_order() {
        let samples = vec![3.0, 1.0, 2.0];
        let _ = sorted_copy(&samples);
        assert_eq!(samples, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn paired_ratios_cancel_a_drifting_machine() {
        // A machine slowing down by 50% across the run: unpaired, the last
        // rounds of one case would swamp the first rounds of the other.
        let drift = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5];
        let reference: Vec<f64> = drift.iter().map(|d| 100.0 * d).collect();
        let case: Vec<f64> = drift.iter().map(|d| 200.0 * d).collect();
        let r = paired_ratio(&case, &reference).expect("enough pairs");
        let (lo, hi) = r.iqr.expect("enough pairs for an IQR");
        assert!((r.point - 2.0).abs() < 1e-12, "point = {}", r.point);
        assert!((lo - 2.0).abs() < 1e-12, "lo = {lo}");
        assert!((hi - 2.0).abs() < 1e-12, "hi = {hi}");
    }

    #[test]
    fn the_point_always_sits_inside_its_own_band() {
        // The whole reason the point is a paired order statistic: a min-over-min
        // ratio is a different estimator and lands outside the band routinely.
        let reference = [10.0, 10.5, 11.0, 12.0, 30.0, 10.2];
        let case = [20.0, 21.5, 22.0, 25.0, 33.0, 20.4];
        let r = paired_ratio(&case, &reference).expect("pairs");
        let (lo, hi) = r.iqr.expect("iqr");
        assert!(
            lo <= r.point && r.point <= hi,
            "{lo} .. {} .. {hi}",
            r.point
        );
    }

    #[test]
    fn a_short_run_gets_a_median_but_no_band() {
        let r = paired_ratio(&[2.0, 4.0], &[1.0, 2.0]).expect("pairs");
        assert_eq!(r.point, 2.0);
        assert_eq!(r.iqr, None);
    }

    #[test]
    fn no_usable_pairs_means_no_ratio() {
        assert_eq!(paired_ratio(&[1.0], &[0.0]), None);
        assert_eq!(paired_ratio(&[], &[]), None);
    }
}
