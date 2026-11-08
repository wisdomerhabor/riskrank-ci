"""ApacheJIT ingestion and time-based partitioning (Chapter Three, sections 3.5.1 and 3.8.2).

What this module does, in order:

1. Reads the complete published file (apachejit_total.csv) and the fix links (commit_links_*.csv).
2. Merges the three Hadoop names into one project, leaving 13 projects.
3. Splits every project by date, using each commit's author timestamp in UTC:
       base      first commit  -> start of calibration period   (fits the model)
       calib     final fifth of [first commit, 31 Dec 2015]      (fits the calibration map)
       gap       all of 2016                                     (used for nothing; counted)
       test      1 Jan 2017 -> 31 Dec 2019
4. Relabels: a base or calibration commit counts as bug-inducing only if one of its fixes was
   made before 1 Jan 2017. Test commits keep every fix up to the end of 2019.

Why the timestamp and not the file's `year` column: the two disagree for 46 bug-inducing
commits, and only the timestamp reproduces the count stated in section 3.5
(2,944 of 22,467 training commits with no fix by the end of 2016).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import pandas as pd

UTC = "UTC"
CALIB_END = pd.Timestamp("2016-01-01", tz=UTC)  # calibration ends 31 Dec 2015
TEST_START = pd.Timestamp("2017-01-01", tz=UTC)  # 2016 is the gap year
TEST_END = pd.Timestamp("2020-01-01", tz=UTC)  # test period ends 31 Dec 2019
CALIB_FRACTION = 0.2  # final fifth of the remaining training period

# The 12 change metrics of Kamei et al. (2013) as published in ApacheJIT (Table 3.7).
FEATURES = ["la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp"]

HADOOP_NAMES = {"apache/hadoop", "apache/hadoop-hdfs", "apache/hadoop-mapreduce"}


@dataclass
class ApacheJIT:
    commits: pd.DataFrame  # one row per commit, with `partition` and `label`
    report: pd.DataFrame  # class distribution per project and partition


def load_raw(root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the published commit file and every fix-link file under `root`."""
    total = pd.read_csv(os.path.join(root, "dataset", "apachejit_total.csv"))
    link_files = sorted(glob.glob(os.path.join(root, "data", "commit_links_*.csv")))
    if not link_files:
        raise FileNotFoundError(f"no commit_links_*.csv under {root}/data")
    links = pd.concat([pd.read_csv(f) for f in link_files], ignore_index=True)
    return total, links


VARIANTS = (None, "exclude_relabelled", "window_1y", "from_2010")  # section 3.5.1 sensitivity variants
WINDOW_TEST_END = pd.Timestamp("2019-01-01", tz=UTC)  # 2019 commits cannot be observed for a full year
FROM_2010 = pd.Timestamp("2010-01-01", tz=UTC)


def build(root: str, variant: str | None = None) -> ApacheJIT:
    """The corpus with its primary labels, or one of the three pre-defined sensitivity variants:

    exclude_relabelled  drop every commit whose label differs from the published one (training commits
                        fixed only after 2016, and test commits fixed only after 2019), instead of
                        treating them as clean
    window_1y           a commit is bug-inducing only if its first fix came within one year of it, in
                        every partition; the test period keeps only 2017 and 2018
    from_2010           drop commits made before 2010; partition boundaries stay as in the primary
                        corpus, so only the presence of the older commits changes
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    total, links = load_raw(root)
    df = total.copy()

    df["project"] = df["project"].where(~df["project"].isin(HADOOP_NAMES), "apache/hadoop")
    df["when"] = pd.to_datetime(df["author_date"], unit="s", utc=True)

    # Earliest fix date for each bug-inducing commit, over ALL published links. The links run to
    # June 2021; the label rules below decide which fixes count.
    links = links.assign(fix_when=pd.to_datetime(links["fix_date"], unit="s", utc=True))
    first_fix = links.groupby("bug_hash")["fix_when"].min()
    df["first_fix"] = df["commit_id"].map(first_fix)

    unlinked = df["buggy"] & df["first_fix"].isna()
    if unlinked.any():
        raise ValueError(f"{int(unlinked.sum())} bug-inducing commits have no fix link")

    df["partition"] = _partition(df)
    df["label"] = _label(df)
    df = df[df["when"] < TEST_END]  # nothing after 2019 exists, but make it explicit

    if variant == "exclude_relabelled":
        relabelled = df["buggy"] & ~df["label"] & (df["partition"] != "gap")
        df = df[~relabelled]
    elif variant == "window_1y":
        within_year = df["first_fix"] <= df["when"] + pd.DateOffset(years=1)
        df = df.assign(label=df["buggy"] & within_year & (df["partition"] != "gap"))
        df = df[~((df["partition"] == "test") & (df["when"] >= WINDOW_TEST_END))]
    elif variant == "from_2010":
        df = df[df["when"] >= FROM_2010]

    return ApacheJIT(commits=df.reset_index(drop=True), report=_report(df))


def _partition(df: pd.DataFrame) -> pd.Series:
    """Assign base / calib / gap / test per project, by date."""
    first = df.groupby("project")["when"].transform("min")
    # The fifth is taken "by date" (section 3.8.2), so the boundary is rounded up to the next
    # whole UTC day. Without this, leap days put a 2006 project's boundary at 14:24 on
    # 31 Dec 2013 instead of 1 Jan 2014, contradicting the section's worked example.
    calib_start = (first + (CALIB_END - first) * (1 - CALIB_FRACTION)).dt.ceil("D")

    part = pd.Series("base", index=df.index)
    part[df["when"] >= calib_start] = "calib"
    part[df["when"] >= CALIB_END] = "gap"
    part[df["when"] >= TEST_START] = "test"
    return part


def _label(df: pd.DataFrame) -> pd.Series:
    """Bug-inducing in base/calib only if fixed before 2017; in test only if fixed by end of 2019.

    The second rule turns 165 test commits clean whose only fixes are dated 2020-2021 (measured
    8 Nov 2026), so the test-year shares fall from the published 26.6 / 21.8 / 10.0 percent to
    26.3 / 21.3 / 9.0. Both are reported.
    """
    training_side = df["partition"].isin(["base", "calib"])
    test_side = df["partition"] == "test"
    label = df["buggy"].copy()
    label[training_side] = df.loc[training_side, "buggy"] & (df.loc[training_side, "first_fix"] < TEST_START)
    label[test_side] = df.loc[test_side, "buggy"] & (df.loc[test_side, "first_fix"] < TEST_END)
    label[df["partition"] == "gap"] = False  # never used; set for safety
    return label.astype(bool)


def _report(df: pd.DataFrame) -> pd.DataFrame:
    """Class distribution before and after relabelling, per project and partition."""
    g = df.groupby(["project", "partition"])
    rep = pd.DataFrame(
        {
            "commits": g.size(),
            "buggy_as_published": g["buggy"].sum(),
            "buggy_after_relabel": g["label"].sum(),
        }
    )
    rep["share_after_relabel"] = rep["buggy_after_relabel"] / rep["commits"]
    return rep.reset_index()
