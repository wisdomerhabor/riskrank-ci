"""Flask commit-level corpus (section 3.5.1, Table 3.7; config/flask_mining.yaml, commit_level).

    python -m riskrank.mining.commits          -> data/mined/flask/commits.csv

One row per non-merge commit in history_end_commit's history that changes at least one source module.

  process   the 12 change metrics of Table 3.7, from the shared just-in-time code (riskrank.jit_features),
            exactly as the CI hook computes them
  static    touched_size / touched_complexity: Radon sloc and summed McCabe complexity of the source modules
            the commit modifies, measured on the code as it stands after the commit (a deleted module adds 0);
            Python 2-only source is measured by Radon 4.1.0 on Python 2.7 (tools/radon27), as for the snapshots
  inducing  True if R-SZZ (rszz.jsonl) names the commit as the inducing commit of at least one fix;
            first_fix is the earliest such fix's date, which the training-side relabelling needs
No author identifier is written: the experience metrics are computed in memory and only their values kept.
"""

from __future__ import annotations

import json
import os
import subprocess

import pandas as pd

from riskrank.jit_features import FEATURES as CHANGE, History, compute
from riskrank.mining.fixes import OUT, REPO, is_source, load_config
from riskrank.mining.snapshots import radon27, static_metrics

STATIC = ["touched_size", "touched_complexity"]


def inducing_fixes(path: str = os.path.join(OUT, "rszz.jsonl")) -> dict[str, pd.Timestamp]:
    """inducing commit -> earliest date of a fix R-SZZ traces to it."""
    first: dict[str, pd.Timestamp] = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("inducing"):
            t = pd.Timestamp(r["fix_date"]).tz_convert("UTC")
            first[r["inducing"]] = min(t, first.get(r["inducing"], t))
    return first


def _exists(repo: str, sha: str, path: str) -> bool:
    return subprocess.run(["git", "-C", repo, "cat-file", "-e", f"{sha}:{path}"], capture_output=True).returncode == 0


def build(repo: str = REPO) -> pd.DataFrame:
    cfg = load_config()
    hist = History(repo, cfg["history_end_commit"])
    fixes = inducing_fixes()
    rows, py2 = [], []
    for c in hist.commits:
        touched = sorted({p for p, _, _ in c.files if is_source(p, cfg)})
        if not touched:
            continue
        row = {"commit_id": c.sha, "when": pd.Timestamp(c.t, unit="s", tz="UTC"), **compute(c, hist),
               "touched_size": 0, "touched_complexity": 0, "modules_touched": len(touched),
               "inducing": c.sha in fixes, "first_fix": fixes.get(c.sha), "measured_with": "radon 6 / python 3"}
        for p in touched:
            if not _exists(repo, c.sha, p):
                continue  # deleted (or renamed away) by this commit: adds 0
            sm = static_metrics(c.sha, p, repo=repo)
            if sm is None:
                py2.append((len(rows), c.sha, p))
                continue
            row["touched_size"] += sm[0]
            row["touched_complexity"] += sm[1]
        rows.append(row)
    unparsable = 0
    for (i, sha, p), m in zip(py2, radon27([(sha, p) for _, sha, p in py2])):
        if m["sloc"] is None:
            unparsable += 1
            continue
        rows[i]["touched_size"] += m["sloc"]
        rows[i]["touched_complexity"] += m["complexity"]
        rows[i]["measured_with"] = "radon 4.1.0 / python 2.7 (some modules)"
    df = pd.DataFrame(rows)
    df.attrs.update(python2_files=len(py2) - unparsable, unmeasured_files=unparsable)
    return df


def main() -> None:
    df = build()
    df.to_csv(os.path.join(OUT, "commits.csv"), index=False)
    meta = {"commits": len(df), "inducing": int(df["inducing"].sum()), "first": str(df["when"].min()),
            "last": str(df["when"].max()), **df.attrs, "features": {"process": CHANGE, "static": STATIC}}
    json.dump(meta, open(os.path.join(OUT, "commits_meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
