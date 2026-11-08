"""Flask module snapshots: features and R-SZZ labels (section 3.5.1, Table 3.7; config/flask_mining.yaml).

    python -m riskrank.mining.snapshots     -> data/mined/flask/modules.csv

One row per (snapshot, source module). Snapshots fall on quarter boundaries (UTC). The tree at a
snapshot is the last commit on main strictly before it.

  size        Radon `sloc`: non-blank, non-comment source lines
  complexity  sum of McCabe complexity over functions and methods (Radon); class totals are not
              added, because they already contain their methods
  churn       lines added + deleted in the module during the 12 months before the snapshot
  changes     commits that modified it in that period
  prior_fixes commits in that period whose message matches the fix keywords alone (Table 3.7:
              issue labels are not used because they can change after the commit)
  label       True if a bug fix in the 12 months after the snapshot changed lines in the module
              that R-SZZ traces to an inducing commit committed on or before the snapshot

Renames are followed, so a module keeps its history across the 2019 move from flask/ to src/flask/.
Snapshots are dropped when either 12-month period would be truncated: the first snapshot is the
first quarter boundary at least 12 months after Flask's first commit, and the last is the latest
whose observation window closes on or before the history end.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import defaultdict

import pandas as pd
from radon.complexity import cc_visit
from radon.visitors import Function
from radon.raw import analyze

from riskrank.mining.fixes import REPO, load_config, is_source

OUT = "data/mined/flask"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout


# --------------------------------------------------------------- history index
def history(end: str) -> list[dict]:
    """Every non-merge commit with numstat and rename pairs, oldest first."""
    out = git("log", end, "--no-merges", "-M", "--reverse", "--format=\x1e%H%x00%cI%x00%s%x00%b%x00",
              "--numstat")
    rows = []
    for rec in out.split("\x1e")[1:]:
        h, when, subj, body, stats = rec.split("\x00", 4)
        files = []
        for line in stats.strip().splitlines():
            a, d, path = line.split("\t", 2)
            old = new = path
            m = re.match(r"(.*)\{(.*) => (.*)\}(.*)", path)
            if m:
                old = (m[1] + m[2] + m[4]).replace("//", "/")
                new = (m[1] + m[3] + m[4]).replace("//", "/")
            elif " => " in path:
                old, new = path.split(" => ")
            files.append({"old": old, "new": new,
                          "lines": (0 if a == "-" else int(a)) + (0 if d == "-" else int(d))})
        rows.append({"hash": h, "when": pd.Timestamp(when).tz_convert("UTC"), "message": f"{subj}\n{body}",
                     "files": files})
    return rows


def lineage(commits: list[dict]) -> dict[str, str]:
    """Map every path a module ever had to one stable id: its most recent path."""
    parent: dict[str, str] = {}
    for c in commits:
        for f in c["files"]:
            if f["old"] != f["new"]:
                parent[f["old"]] = f["new"]

    def final(p: str) -> str:
        seen = set()
        while p in parent and p not in seen:
            seen.add(p)
            p = parent[p]
        return p

    paths = {f[k] for c in commits for f in c["files"] for k in ("old", "new")}
    return {p: final(p) for p in paths}


# ---------------------------------------------------------------- measurement
def static_metrics(commit: str, path: str) -> tuple[int, int] | None:
    try:
        src = git("show", f"{commit}:{path}")
    except subprocess.CalledProcessError:
        return None
    try:
        sloc = analyze(src).sloc
        cc = sum(b.complexity for b in cc_visit(src) if isinstance(b, Function))
    except SyntaxError:
        return None  # Python 2-only syntax that Radon cannot parse; counted and reported
    return sloc, cc


def quarter_starts(first: pd.Timestamp, last: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(first, last, freq="QS", tz="UTC"))


def build() -> pd.DataFrame:
    cfg = load_config()
    snap_cfg = cfg["module_snapshots"]
    months = snap_cfg["observation_window_months"]
    period = snap_cfg["process_period_months"]
    keywords = re.compile(cfg["bug_fix_rule"]["keywords"], re.IGNORECASE)

    commits = history(cfg["history_end_commit"])
    stable = lineage(commits)
    history_end = commits[-1]["when"]
    first = commits[0]["when"] + pd.DateOffset(months=period)
    last = history_end - pd.DateOffset(months=months)
    snaps = quarter_starts(first.ceil("D"), last)

    # R-SZZ output: blamed modules per fix, with the inducing commit's date.
    fixes = []
    for line in open(os.path.join(OUT, "rszz.jsonl")):
        r = json.loads(line)
        if r["inducing"]:
            fixes.append({"when": pd.Timestamp(r["fix_date"]).tz_convert("UTC"),
                          "inducing_when": pd.Timestamp(r["inducing_date"]).tz_convert("UTC"),
                          "blamed": {stable.get(p, p) for p in r["blamed_files"]}})

    rows, unparsable = [], 0
    for s in snaps:
        tree = git("rev-list", "-1", f"--before={s.isoformat()}", cfg["history_end_commit"]).strip()
        modules = [p for p in git("ls-tree", "-r", "--name-only", tree).splitlines() if is_source(p, cfg)]
        lo = s - pd.DateOffset(months=period)
        churn, changes, prior = defaultdict(int), defaultdict(int), defaultdict(int)
        for c in commits:
            if lo <= c["when"] < s:
                fix = keywords.search(c["message"]) is not None
                for f in c["files"]:
                    mid = stable.get(f["new"], f["new"])
                    churn[mid] += f["lines"]
                    changes[mid] += 1
                    prior[mid] += int(fix)
        hi = s + pd.DateOffset(months=months)
        defective = set()
        for fx in fixes:
            if s < fx["when"] <= hi and fx["inducing_when"] <= s:
                defective |= fx["blamed"]
        for p in modules:
            sm = static_metrics(tree, p)
            if sm is None:
                unparsable += 1
                continue
            mid = stable.get(p, p)
            rows.append({"snapshot": s, "module": p, "module_id": mid, "tree": tree,
                         "size": sm[0], "complexity": sm[1], "churn": churn[mid],
                         "changes": changes[mid], "prior_fixes": prior[mid], "label": mid in defective})
    df = pd.DataFrame(rows)
    df.attrs["unparsable_modules"] = unparsable
    return df


def main() -> None:
    df = build()
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "modules.csv"), index=False)
    g = df.groupby("snapshot")["label"].agg(["size", "sum"])
    print(f"{len(g)} snapshots ({g.index.min().date()} to {g.index.max().date()}), {len(df)} module rows, "
          f"{int(df['label'].sum())} defective; snapshots with no defect: {(g['sum'] == 0).sum()}; "
          f"unparsable skipped: {df.attrs['unparsable_modules']}")


if __name__ == "__main__":
    main()
