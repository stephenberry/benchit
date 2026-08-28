//! Hand-rolled argument parsing.
//!
//! The positional filter is a substring match on `group/case`, which is what
//! people actually use criterion's filter for, and keeping it compatible means
//! existing muscle memory and CI invocations transfer unchanged.

use std::fmt;

/// The per-benchmark time budget, in milliseconds.
const DEFAULT_TIME_MS: u64 = 1000;
/// What `--quick` lowers that budget to.
const QUICK_TIME_MS: u64 = 200;
/// The cap on samples collected per case.
const DEFAULT_SAMPLES: usize = 50;

/// Output format for the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Format {
    /// Aligned text on stdout.
    #[default]
    Text,
    /// One machine-readable row per case.
    Tsv,
}

/// Everything the harness can be told to do.
///
/// Every field has a default that works, and [`Bench::from_args`] fills them
/// from the command line. Construct one directly only when driving the harness
/// from code.
///
/// [`Bench::from_args`]: crate::Bench::from_args
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Substring match on `group/case`. `None` runs everything.
    pub filter: Option<String>,
    /// Upper bound on samples collected per case.
    pub samples: usize,
    /// Per-benchmark time budget in milliseconds.
    pub time_ms: u64,
    /// Samples collected per visit to a case while interleaving.
    ///
    /// The default of 1 gives maximally paired samples. Raising it trades some
    /// of that pairing for warmer caches, which is what you want when the
    /// benchmark is deliberately measuring hot-cache behaviour.
    pub block: usize,
    /// Run the cases of a group in interleaved rounds rather than one at a
    /// time. This is the point of the harness; turn it off only when a
    /// hot-cache number is the one you want.
    pub interleave: bool,
    /// Write results to `benchit/<name>.tsv` in the directory cargo built this
    /// binary into, so a debug and a release run keep separate baselines.
    pub save_baseline: Option<String>,
    /// Load `benchit/<name>.tsv` from that same directory and add a delta
    /// column.
    pub baseline: Option<String>,
    /// Output format.
    pub format: Format,
    /// List matching `group/case` names instead of running anything.
    pub list: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            filter: None,
            samples: DEFAULT_SAMPLES,
            time_ms: DEFAULT_TIME_MS,
            block: 1,
            interleave: true,
            save_baseline: None,
            baseline: None,
            format: Format::Text,
            list: false,
        }
    }
}

impl Config {
    /// Parse the process arguments, skipping `argv[0]`.
    pub fn from_args() -> Result<Self, ArgError> {
        Self::parse(std::env::args().skip(1))
    }

    /// Parse an explicit argument list.
    pub fn parse<I, S>(args: I) -> Result<Self, ArgError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut cfg = Self::default();
        let mut args = args.into_iter().map(Into::into).peekable();

        while let Some(arg) = args.next() {
            let mut value = |flag: &str, inline: Option<String>| -> Result<String, ArgError> {
                match inline.or_else(|| args.next()) {
                    // A flag where a value belongs means the value was
                    // forgotten: `--save-baseline --quick` should say so rather
                    // than write a file named `--quick.tsv`.
                    Some(v) if v.starts_with("--") => Err(ArgError::MissingValue(flag.to_string())),
                    Some(v) => Ok(v),
                    None => Err(ArgError::MissingValue(flag.to_string())),
                }
            };
            // Accept both `--flag value` and `--flag=value`.
            let (flag, inline) = match arg.split_once('=') {
                Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
                _ => (arg.clone(), None),
            };

            if inline.is_some() && !TAKES_A_VALUE.contains(&flag.as_str()) {
                return Err(ArgError::UnexpectedValue(flag));
            }

            match flag.as_str() {
                "--help" | "-h" => return Err(ArgError::HelpRequested),
                "--quick" => cfg.time_ms = QUICK_TIME_MS,
                "--samples" => cfg.samples = parse_num(&flag, &value(&flag, inline)?)?,
                "--time" => cfg.time_ms = parse_num(&flag, &value(&flag, inline)?)?,
                "--block" => cfg.block = parse_num(&flag, &value(&flag, inline)?)?,
                "--no-interleave" => cfg.interleave = false,
                "--save-baseline" => cfg.save_baseline = Some(value(&flag, inline)?),
                "--baseline" => cfg.baseline = Some(value(&flag, inline)?),
                "--format" => {
                    let v = value(&flag, inline)?;
                    cfg.format = match v.as_str() {
                        "text" => Format::Text,
                        "tsv" => Format::Tsv,
                        _ => return Err(ArgError::BadValue(flag, v)),
                    };
                }
                "--list" => cfg.list = true,
                // `cargo bench` passes these through to the harness. Ignoring
                // them is what lets `cargo bench` work with no arguments.
                "--bench" | "--test" | "--nocapture" => {}
                "--color" => {
                    let _ = inline.or_else(|| args.next());
                }
                other if other.starts_with('-') => {
                    return Err(ArgError::UnknownFlag(other.to_string()));
                }
                _ => {
                    if cfg.filter.is_some() {
                        return Err(ArgError::TooManyFilters(arg));
                    }
                    cfg.filter = Some(arg);
                }
            }
        }

        if cfg.samples < 1 {
            return Err(ArgError::BadValue(
                "--samples".into(),
                cfg.samples.to_string(),
            ));
        }
        if cfg.block < 1 {
            return Err(ArgError::BadValue("--block".into(), cfg.block.to_string()));
        }
        if cfg.time_ms < 1 {
            return Err(ArgError::BadValue("--time".into(), cfg.time_ms.to_string()));
        }
        Ok(cfg)
    }

    /// Does this case's `group/case` name pass the filter?
    pub(crate) fn matches(&self, full_name: &str) -> bool {
        match &self.filter {
            Some(f) => full_name.contains(f.as_str()),
            None => true,
        }
    }
}

fn parse_num<T: std::str::FromStr>(flag: &str, raw: &str) -> Result<T, ArgError> {
    raw.parse()
        .map_err(|_| ArgError::BadValue(flag.to_string(), raw.to_string()))
}

/// The flags that consume a following argument.
const TAKES_A_VALUE: [&str; 6] = [
    "--samples",
    "--time",
    "--block",
    "--save-baseline",
    "--baseline",
    "--format",
];

/// What went wrong while parsing arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgError {
    /// `--help` was passed; the caller should print [`USAGE`] and exit
    /// successfully.
    HelpRequested,
    /// A flag that takes a value was given none.
    MissingValue(String),
    /// A flag's value could not be understood.
    BadValue(String, String),
    /// An unrecognised `-`-prefixed argument.
    UnknownFlag(String),
    /// A `--flag=value` where the flag takes no value.
    UnexpectedValue(String),
    /// More than one positional filter.
    TooManyFilters(String),
}

impl fmt::Display for ArgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HelpRequested => write!(f, "help requested"),
            Self::MissingValue(flag) => write!(f, "`{flag}` needs a value"),
            Self::BadValue(flag, v) => write!(f, "`{flag}` does not accept `{v}`"),
            Self::UnknownFlag(flag) => write!(f, "unknown flag `{flag}`"),
            Self::UnexpectedValue(flag) => write!(f, "`{flag}` takes no value"),
            Self::TooManyFilters(v) => write!(f, "only one filter is accepted, got `{v}` as well"),
        }
    }
}

impl std::error::Error for ArgError {}

/// The `--help` text.
pub const USAGE: &str = "\
usage: <bench binary> [FILTER] [OPTIONS]

  FILTER                substring match on \"group/case\"

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
";

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Config {
        Config::parse(args.iter().copied()).expect("parses")
    }

    #[test]
    fn defaults_are_the_documented_ones() {
        let c = Config::default();
        assert_eq!(c.samples, 50);
        assert_eq!(c.time_ms, 1000);
        assert_eq!(c.block, 1);
        assert!(c.interleave);
        assert_eq!(c.format, Format::Text);
    }

    #[test]
    fn no_args_is_the_default() {
        assert_eq!(parse(&[]), Config::default());
    }

    #[test]
    fn flags_take_values_either_way() {
        assert_eq!(parse(&["--samples", "10"]).samples, 10);
        assert_eq!(parse(&["--samples=10"]).samples, 10);
        assert_eq!(parse(&["--format=tsv"]).format, Format::Tsv);
        assert_eq!(parse(&["--format", "tsv"]).format, Format::Tsv);
    }

    #[test]
    fn positional_argument_is_the_filter() {
        assert_eq!(parse(&["decode"]).filter.as_deref(), Some("decode"));
    }

    #[test]
    fn cargo_bench_passes_bench_through() {
        assert_eq!(parse(&["--bench"]), Config::default());
        assert_eq!(
            parse(&["--bench", "decode"]).filter.as_deref(),
            Some("decode")
        );
    }

    #[test]
    fn quick_shortens_the_budget() {
        assert_eq!(parse(&["--quick"]).time_ms, QUICK_TIME_MS);
    }

    #[test]
    fn a_forgotten_value_is_not_swallowed_from_the_next_flag() {
        // Otherwise this writes a baseline named `--quick` and drops `--quick`.
        assert!(matches!(
            Config::parse(["--save-baseline", "--quick"]),
            Err(ArgError::MissingValue(_))
        ));
    }

    #[test]
    fn a_value_on_a_valueless_flag_is_rejected() {
        // Silently dropping it would make `--no-interleave=false` interleave.
        assert!(matches!(
            Config::parse(["--no-interleave=false"]),
            Err(ArgError::UnexpectedValue(_))
        ));
        assert!(matches!(
            Config::parse(["--list=x"]),
            Err(ArgError::UnexpectedValue(_))
        ));
    }

    #[test]
    fn bad_input_is_rejected() {
        assert!(matches!(
            Config::parse(["--samples"]),
            Err(ArgError::MissingValue(_))
        ));
        assert!(matches!(
            Config::parse(["--samples", "lots"]),
            Err(ArgError::BadValue(..))
        ));
        assert!(matches!(
            Config::parse(["--format=html"]),
            Err(ArgError::BadValue(..))
        ));
        assert!(matches!(
            Config::parse(["--plot"]),
            Err(ArgError::UnknownFlag(_))
        ));
        assert!(matches!(
            Config::parse(["a", "b"]),
            Err(ArgError::TooManyFilters(_))
        ));
        assert!(matches!(
            Config::parse(["--samples", "0"]),
            Err(ArgError::BadValue(..))
        ));
        assert!(matches!(
            Config::parse(["--help"]),
            Err(ArgError::HelpRequested)
        ));
    }

    #[test]
    fn filter_is_a_substring_of_group_slash_case() {
        let c = parse(&["decode/mine"]);
        assert!(c.matches("decode/mine"));
        assert!(c.matches("json/decode/mine"));
        assert!(!c.matches("decode/theirs"));
        assert!(Config::default().matches("anything"));
    }
}
