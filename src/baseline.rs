//! Saved baselines, as TSV.
//!
//! TSV parses with `split('\t')` and a `str::parse`, diffs cleanly in git if
//! someone chooses to commit one, and is readable without a tool. It is
//! deliberately not JSON, and deliberately not the format of whatever crate is
//! being benchmarked: storing measurements in the thing being measured turns a
//! regression in that code into corrupted history.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// One saved case.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Row {
    pub group: String,
    pub case: String,
    pub iters: u64,
    pub min_ns: f64,
    pub p50_ns: f64,
    pub p90_ns: f64,
    pub samples: usize,
}

/// Baselines keyed by `"group/case"`, the same string the filter matches, so a
/// lookup can borrow the name the scheduler already built.
pub(crate) type Baseline = HashMap<String, Row>;

const HEADER: &str = "group\tcase\titers\tmin_ns\tp50_ns\tp90_ns\tsamples";

/// `<build-dir>/benchit/<name>.tsv`.
pub(crate) fn path(name: &str) -> io::Result<PathBuf> {
    if name.is_empty() || name.contains(['/', '\\']) || name == ".." {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("`{name}` is not a usable baseline name"),
        ));
    }
    Ok(dir().join("benchit").join(format!("{name}.tsv")))
}

/// Where to keep baselines: beside the build they describe.
///
/// A literal `target` relative to the working directory is only right by
/// coincidence, since it misses `build.target-dir`, misses `--target-dir`, and
/// in a workspace names the member directory rather than the root cargo built
/// into. `CARGO_TARGET_DIR` covers only the first of those, and cargo resolves
/// a relative value against its own invocation directory rather than the bench
/// binary's, so it is a fallback rather than the answer.
fn dir() -> PathBuf {
    build_dir(&std::env::current_exe().unwrap_or_default())
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("target"))
}

/// The directory cargo built `exe` into: the nearest ancestor holding a `deps`
/// directory, which is the one unambiguous marker of a cargo build directory.
///
/// Cargo puts a binary in one of three places, all of them within one level of
/// `<target-dir>/[<triple>/]<profile>/`: bench and test binaries in `deps/`,
/// examples in `examples/`, and a bin target directly in the profile directory
/// itself. Matching on `deps` as the exe's own parent would find only the
/// first, and a `[[bin]]` under `cargo run` would fall through to a relative
/// guess.
///
/// Using the profile directory rather than the target directory above it keeps
/// each build's baselines to itself: a debug binary cannot overwrite what
/// `cargo bench` saved, and a `--target` build keeps its own file. Both would
/// otherwise share one, and a debug timing saved over a release baseline is
/// worse than no baseline at all.
///
/// Anything not cargo-shaped returns `None` rather than a guess.
fn build_dir(exe: &Path) -> Option<&Path> {
    let here = exe.parent()?;
    // An empty path is what a relative `deps/x` leaves behind, and joining onto
    // it would silently mean the working directory.
    let candidates = [Some(here), here.parent()];
    candidates
        .into_iter()
        .flatten()
        .find(|dir| !dir.as_os_str().is_empty() && dir.join("deps").is_dir())
}

/// Write `rows` sorted by group then case, so the file diffs cleanly.
///
/// Rows measured by this run replace same-named rows already in the file, and
/// everything else in it is carried across. Overwriting is not an option: a
/// filtered run measures a handful of cases, and rewriting the file from those
/// alone would silently discard every baseline the filter excluded.
pub(crate) fn save(path: &Path, measured: Vec<Row>) -> io::Result<()> {
    for row in &measured {
        if let Some(bad) = row
            .group
            .find(['\t', '\n', '\r'])
            .map(|_| &row.group)
            .or_else(|| row.case.find(['\t', '\n', '\r']).map(|_| &row.case))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("`{bad}` contains a tab or newline, which a TSV row cannot carry"),
            ));
        }
    }

    // A missing file is the normal first-save case, and an unreadable one is not
    // worth failing over either: the alternative is refusing to save the run
    // that was just measured.
    let mut merged = load(path).unwrap_or_default();
    for row in measured {
        merged.insert(key(&row.group, &row.case), row);
    }
    let mut rows: Vec<Row> = merged.into_values().collect();
    rows.sort_by(|a, b| (&a.group, &a.case).cmp(&(&b.group, &b.case)));
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut out = io::BufWriter::new(fs::File::create(path)?);
    writeln!(
        out,
        "# benchit baseline v1  harness={}  profile={}",
        env!("CARGO_PKG_VERSION"),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    writeln!(out, "{HEADER}")?;
    for r in &rows {
        writeln!(
            out,
            "{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{}",
            r.group, r.case, r.iters, r.min_ns, r.p50_ns, r.p90_ns, r.samples
        )?;
    }
    out.flush()
}

/// Read a baseline file. Unparseable rows are an error rather than a silent
/// skip: a half-loaded baseline would produce a delta column that quietly
/// omits cases.
pub(crate) fn load(path: &Path) -> io::Result<Baseline> {
    let text = fs::read_to_string(path)?;
    let mut out = Baseline::new();
    for (n, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') || line == HEADER {
            continue;
        }
        let row = parse_row(line).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}:{}: cannot parse baseline row", path.display(), n + 1),
            )
        })?;
        out.insert(key(&row.group, &row.case), row);
    }
    Ok(out)
}

/// The lookup key for a row: the same `group/case` string used everywhere else.
pub(crate) fn key(group: &str, case: &str) -> String {
    format!("{group}/{case}")
}

fn parse_row(line: &str) -> Option<Row> {
    let mut f = line.split('\t');
    let row = Row {
        group: f.next()?.to_string(),
        case: f.next()?.to_string(),
        iters: f.next()?.parse().ok()?,
        min_ns: f.next()?.parse().ok()?,
        p50_ns: f.next()?.parse().ok()?,
        p90_ns: f.next()?.parse().ok()?,
        samples: f.next()?.parse().ok()?,
    };
    if f.next().is_some() { None } else { Some(row) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(group: &str, case: &str) -> Row {
        Row {
            group: group.into(),
            case: case.into(),
            iters: 64,
            min_ns: 412300.0,
            p50_ns: 418100.0,
            p90_ns: 431700.0,
            samples: 50,
        }
    }

    #[test]
    fn round_trips_through_a_file() {
        let dir = std::env::temp_dir().join(format!("benchit-test-{}", std::process::id()));
        let file = dir.join("save.tsv");
        let rows = vec![row("decode/1MiB", "theirs"), row("decode/1MiB", "mine")];
        save(&file, rows).expect("saves");

        let loaded = load(&file).expect("loads");
        assert_eq!(loaded.len(), 2);
        let mine = &loaded["decode/1MiB/mine"];
        assert_eq!(mine.iters, 64);
        assert_eq!(mine.min_ns, 412300.0);
        assert_eq!(mine.samples, 50);

        let text = fs::read_to_string(&file).expect("reads");
        assert!(text.starts_with("# benchit baseline v1"));
        assert!(text.contains(HEADER));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_filtered_run_keeps_the_cases_it_did_not_measure() {
        let dir = std::env::temp_dir().join(format!("benchit-merge-{}", std::process::id()));
        let file = dir.join("main.tsv");
        save(&file, vec![row("a", "one"), row("b", "two")]).expect("first save");

        // A second run measuring only `a/one`, faster than before.
        let mut updated = row("a", "one");
        updated.min_ns = 1.0;
        save(&file, vec![updated]).expect("second save");

        let loaded = load(&file).expect("loads");
        assert_eq!(loaded.len(), 2, "the unmeasured case must survive");
        assert_eq!(loaded["a/one"].min_ns, 1.0, "the measured case must update");
        assert_eq!(loaded["b/two"].min_ns, 412300.0);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_name_that_would_corrupt_the_file_is_refused() {
        let dir = std::env::temp_dir().join(format!("benchit-tab-{}", std::process::id()));
        let file = dir.join("main.tsv");
        let mut bad = row("gr\toup", "case");
        bad.group = "gr\toup".into();
        assert!(save(&file, vec![bad]).is_err());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_corrupt_row_is_an_error_not_a_silent_skip() {
        let dir = std::env::temp_dir().join(format!("benchit-corrupt-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("mkdir");
        let file = dir.join("bad.tsv");
        fs::write(
            &file,
            format!("{HEADER}\ndecode\tmine\tsixty-four\t1\t1\t1\t50\n"),
        )
        .expect("write");
        assert!(load(&file).is_err());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn baseline_names_stay_inside_the_target_directory() {
        assert!(path("../../etc/passwd").is_err());
        assert!(path("").is_err());
        let p = path("main").expect("valid name");
        assert!(p.ends_with("benchit/main.tsv"), "{}", p.display());
    }

    #[test]
    fn the_build_directory_is_the_nearest_one_holding_deps() {
        // The three places cargo puts a runnable binary, all within one level
        // of the profile directory.
        let root = std::env::temp_dir().join(format!("benchit-layout-{}", std::process::id()));
        let build = root.join("target.noindex").join("release");
        fs::create_dir_all(build.join("deps")).expect("mkdir");
        fs::create_dir_all(build.join("examples")).expect("mkdir");
        let of = |p: PathBuf| build_dir(&p).map(Path::to_path_buf);

        // A bench or test binary. The profile directory, not the target
        // directory above it: debug and release must not share a baseline.
        assert_eq!(of(build.join("deps").join("demo-1")), Some(build.clone()));
        // A `[[bin]]` under `cargo run`, which has no `deps` parent at all.
        assert_eq!(of(build.join("probe")), Some(build.clone()));
        // An example, one level down beside `deps`.
        assert_eq!(of(build.join("examples").join("demo")), Some(build.clone()));

        // Not a cargo layout: an installed binary, and a bare name.
        assert_eq!(of(root.join("bin").join("demo")), None);
        assert_eq!(of(PathBuf::from("demo")), None);
        // A relative `deps/x` leaves an empty parent, which must not be read as
        // the working directory.
        assert_eq!(of(PathBuf::from("deps").join("demo")), None);

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn this_test_binary_resolves_to_the_directory_cargo_built_it_into() {
        // The derivation above is arithmetic on a string; this is the check
        // that cargo really does lay binaries out that way.
        let exe = std::env::current_exe().expect("a current exe");
        let build = build_dir(&exe).expect("a cargo-shaped layout");
        assert_eq!(Some(build), exe.parent().and_then(Path::parent));
        assert_eq!(
            exe.parent().and_then(Path::file_name),
            Some("deps".as_ref())
        );
        assert!(build.join("deps").is_dir(), "{}", build.display());
    }
}
