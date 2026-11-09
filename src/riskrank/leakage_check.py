"""Leakage check for the mined Flask corpus (Table 1.1, objective 2: "no feature calculated using information
from after its prediction time").

    python -m riskrank.leakage_check                 # 200 sampled commits (seed 0) + every snapshot
    python -m riskrank.leakage_check --all-commits   # every commit row

Each stored feature row is recomputed from ONLY what existed at its prediction time, and compared:

  commit level   the 12 change metrics of each sampled commit, recomputed from a history read from that commit
                 (its ancestors only), which is what the CI hook sees on a push. The corpus computes them from
                 the whole history, filtered by author time; a commit on a parallel branch that was authored
                 earlier but merged later would enter the corpus value and not this one.
  snapshot level churn, changes and prior_fixes of every module in every snapshot, recomputed from the history
                 of the snapshot's tree only; and the tree's commit must be dated before the snapshot date.

Labels are not features and are allowed to look forward; they are not checked here.
Writes runs/leakage-check.json. Exit 1 if any row differs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

import pandas as pd

from riskrank.jit_features import FEATURES as CHANGE, History, compute
from riskrank.mining.fixes import OUT, REPO, load_config
from riskrank.mining.snapshots import git, history, process_window

TOL = 1e-9


def _differs(a: float, b: float) -> bool:
    return not (math.isclose(float(a), float(b), rel_tol=TOL, abs_tol=TOL))


def check_commits(sample: int | None, seed: int = 0) -> dict:
    df = pd.read_csv(os.path.join(OUT, "commits.csv"))
    rows = df if sample is None or sample >= len(df) else df.sample(sample, random_state=seed)
    bad = []
    for _, r in rows.iterrows():
        h = History(REPO, r["commit_id"])  # ancestors of the commit only
        got = compute(h.by_sha[r["commit_id"]], h)
        diff = {k: [r[k], got[k]] for k in CHANGE if _differs(r[k], got[k])}
        if diff:
            bad.append({"commit_id": r["commit_id"], "stored_vs_ancestors_only": diff})
    return {"rows_checked": int(len(rows)), "rows_in_corpus": int(len(df)), "sample_seed": seed,
            "rows_differing": len(bad), "examples": bad[:20]}


def check_snapshots() -> dict:
    cfg = load_config()
    period = cfg["module_snapshots"]["process_period_months"]
    keywords = re.compile(cfg["bug_fix_rule"]["keywords"], re.IGNORECASE)
    df = pd.read_csv(os.path.join(OUT, "modules.csv"))
    bad, late_trees = [], []
    for snap, g in df.groupby("snapshot"):
        s = pd.Timestamp(snap)
        tree = g["tree"].iloc[0]
        tree_when = pd.Timestamp(git("show", "-s", "--format=%cI", tree).strip()).tz_convert("UTC")
        if tree_when >= s:
            late_trees.append({"snapshot": snap, "tree": tree, "tree_committed": tree_when.isoformat()})
        commits = history(tree)  # the tree's history only
        proc = process_window(commits, list(g["module"]), s - pd.DateOffset(months=period), s, keywords)
        for _, r in g.iterrows():
            got = proc[r["module"]]
            diff = {k: [int(r[k]), int(got[k])] for k in ("churn", "changes", "prior_fixes") if int(r[k]) != int(got[k])}
            if diff:
                bad.append({"snapshot": snap, "module": r["module"], "stored_vs_tree_history": diff})
    return {"snapshots": int(df["snapshot"].nunique()), "rows_checked": int(len(df)),
            "trees_not_before_snapshot": late_trees, "rows_differing": len(bad), "examples": bad[:20]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--all-commits", action="store_true")
    ap.add_argument("--out", default=os.path.join("runs", "leakage-check.json"))
    a = ap.parse_args()
    head = git("rev-parse", "HEAD", repo=".").strip() if os.path.isdir(".git") else None
    res = {"started": datetime.now(timezone.utc).isoformat(), "code_commit": head,
           "commit_level": check_commits(None if a.all_commits else a.sample),
           "snapshot_level": check_snapshots()}
    res["finished"] = datetime.now(timezone.utc).isoformat()
    ok = (res["commit_level"]["rows_differing"] == 0 and res["snapshot_level"]["rows_differing"] == 0
          and not res["snapshot_level"]["trees_not_before_snapshot"])
    res["passed"] = ok
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, default=str)
    c, s = res["commit_level"], res["snapshot_level"]
    print(f"commits: {c['rows_differing']} of {c['rows_checked']} differ | snapshots: {s['rows_differing']} of "
          f"{s['rows_checked']} rows differ, {len(s['trees_not_before_snapshot'])} late trees -> "
          f"{'PASS' if ok else 'FAIL'}  ({a.out})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
