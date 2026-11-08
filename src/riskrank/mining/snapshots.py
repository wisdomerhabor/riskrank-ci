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

import pandas as pd
from radon.complexity import cc_visit
from radon.visitors import Function
from radon.raw import analyze

from riskrank.mining.fixes import REPO, load_config, is_source

OUT = "data/mined/flask"


def git(*args: str, repo: str = REPO) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


# --------------------------------------------------------------- history index
def history(end: str, repo: str = REPO) -> list[dict]:
    """Every non-merge commit with numstat and rename pairs, oldest first."""
    out = git("log", end, "--no-merges", "-M", "--reverse", "--format=\x1e%H%x00%cI%x00%s%x00%b%x00",
              "--numstat", repo=repo)
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


def process_window(commits: list[dict], live: list[str], lo, hi, keywords) -> dict[str, dict]:
    """Churn, changes and prior fixes in [lo, hi) for each module live at `hi`, keyed by its path at hi.

    Walks commits newest first. `alias` maps the path a module had at that point in time to its
    path at `hi`; a rename old -> new re-points the alias to `old` for everything earlier. A path
    is only ever followed backwards through renames that happened, in time order, so a later file
    re-created at an old path (Flask's 2023 app.py split) never inherits the old file's history.
    """
    alias = {p: p for p in live}
    out = {p: {"churn": 0, "changes": 0, "prior_fixes": 0} for p in live}
    for c in reversed(commits):
        if c["when"] >= hi:
            continue
        if c["when"] < lo:
            break
        fix = keywords.search(c["message"]) is not None
        for f in c["files"]:
            q = alias.get(f["new"])
            if q is not None and f["lines"] > 0:  # a pure move modifies no content
                out[q]["churn"] += f["lines"]
                out[q]["changes"] += 1
                out[q]["prior_fixes"] += int(fix)
        for f in c["files"]:
            if f["old"] != f["new"] and f["new"] in alias:
                alias[f["old"]] = alias.pop(f["new"])
    return out


def last_change(commits: list[dict], live: list[str], before) -> dict[str, pd.Timestamp | None]:
    """Time of each live module's last content-changing commit before `before`, following renames
    backwards in time as process_window does. Used by the recency baseline."""
    alias = {p: p for p in live}
    out: dict[str, pd.Timestamp | None] = {p: None for p in live}
    for c in reversed(commits):
        if c["when"] >= before:
            continue
        for f in c["files"]:
            q = alias.get(f["new"])
            if q is not None and f["lines"] > 0 and out[q] is None:
                out[q] = c["when"]
        if all(v is not None for v in out.values()):
            break
        for f in c["files"]:
            if f["old"] != f["new"] and f["new"] in alias:
                alias[f["old"]] = alias.pop(f["new"])
    return out

def forward_paths(commits: list[dict], live: list[str], start, fixes_in_window: list[dict]) -> set[str]:
    """Snapshot modules blamed by the given fixes, following renames forward from `start`."""
    cur = {p: p for p in live}  # snapshot path -> its path at the current point in time
    blamed: set[str] = set()
    pending = sorted(fixes_in_window, key=lambda f: f["when"])
    i = 0
    for c in commits:
        if c["when"] <= start:
            continue
        while i < len(pending) and pending[i]["when"] < c["when"]:
            blamed |= {q for q, now in cur.items() if now in pending[i]["blamed"]}
            i += 1
        if i == len(pending):
            break
        renames = {f["old"]: f["new"] for f in c["files"] if f["old"] != f["new"]}
        for q, now in cur.items():
            if now in renames:
                cur[q] = renames[now]
    for f in pending[i:]:
        blamed |= {q for q, now in cur.items() if now in f["blamed"]}
    return blamed


# ---------------------------------------------------------------- measurement
def static_metrics(commit: str, path: str, repo: str = REPO) -> tuple[int, int] | None:
    try:
        src = git("show", f"{commit}:{path}", repo=repo)
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
                          "blamed": set(r["blamed_files"])})

    rows, py2, unparsable = [], [], 0
    for s in snaps:
        tree = git("rev-list", "-1", f"--before={s.isoformat()}", cfg["history_end_commit"]).strip()
        modules = [p for p in git("ls-tree", "-r", "--name-only", tree).splitlines() if is_source(p, cfg)]
        proc = process_window(commits, modules, s - pd.DateOffset(months=period), s, keywords)
        hi = s + pd.DateOffset(months=months)
        window = [fx for fx in fixes if s < fx["when"] <= hi and fx["inducing_when"] <= s]
        defective = forward_paths(commits, modules, s, window)
        for p in modules:
            sm = static_metrics(tree, p)
            row = {"snapshot": s, "module": p, "tree": tree, **proc[p], "label": p in defective,
                   "measured_with": "radon 6 / python 3"}
            if sm is None:
                py2.append(row)
                continue
            rows.append(row | {"size": sm[0], "complexity": sm[1]})
    # Python 2-only source cannot be parsed by Python 3. It is measured in one batch by Radon 4.1.0 on
    # Python 2.7 (tools/radon27), which matched Radon 6 on Python 3 for all 85 files parsable by
    # both (4 snapshots, 2014-2016; checked 8 Nov 2026).
    for row, m in zip(py2, radon27([(r["tree"], r["module"]) for r in py2])):
        if m["sloc"] is None:
            unparsable += 1
            continue
        rows.append(row | {"size": m["sloc"], "complexity": m["complexity"],
                           "measured_with": "radon 4.1.0 / python 2.7"})
    df = pd.DataFrame(rows).sort_values(["snapshot", "module"]).reset_index(drop=True)
    df.attrs["unparsable_modules"] = unparsable
    df.attrs["python2_measured"] = len(py2) - unparsable
    return df


def radon27(items: list[tuple[str, str]]) -> list[dict]:
    if not items:
        return []
    payload = [{"key": f"{t}:{p}", "source": git("show", f"{t}:{p}")} for t, p in items]
    out = subprocess.run(["docker", "run", "--rm", "-i", "radon27:4.1.0"], input=json.dumps(payload),
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main() -> None:
    df = build()
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "modules.csv"), index=False)
    g = df.groupby("snapshot")["label"].agg(["size", "sum"])
    print(f"{len(g)} snapshots ({g.index.min().date()} to {g.index.max().date()}), {len(df)} module rows, "
          f"{int(df['label'].sum())} defective; snapshots with no defect: {(g['sum'] == 0).sum()}; "
          f"measured under Python 2.7: {df.attrs['python2_measured']}; unparsable skipped: {df.attrs['unparsable_modules']}")


if __name__ == "__main__":
    main()
