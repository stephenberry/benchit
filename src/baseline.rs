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

/// `target/benchit/<name>.tsv`, honouring `CARGO_TARGET_DIR`.
pub(crate) fn path(name: &str) -> io::Result<PathBuf> {
    if name.is_empty() || name.contains(['/', '\\']) || name == ".." {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("`{name}` is not a usable baseline name"),
        ));
    }
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    Ok(target.join("benchit").join(format!("{name}.tsv")))
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
}
