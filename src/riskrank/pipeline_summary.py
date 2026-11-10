"""Summarise the counted pushes of a deployment pipeline from their plan files (section 3.10.5).

    python -m riskrank.pipeline_summary PLANS_DIR [--out FILE]

PLANS_DIR holds one folder per run, each with the plan.json the CI hook uploaded (riskrank-plan artifact),
e.g. downloaded with `gh run download <id> -n riskrank-plan -D PLANS_DIR/<id>`.

A green CI run is not evidence that RiskRank worked: the hook falls back to the default test order on any
failure and never fails the build (NFR3). So this counts plan statuses and lists every run that did not
complete. Added 10 Nov 2026, after 34 riskrank-ci pushes were found to have fallen back (scoring image
without pyarrow) while every CI run was green. Exit code 1 when any run is not complete.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np


def summarise(plans: dict[str, dict]) -> dict:
    status = Counter(p.get("status") for p in plans.values())
    done = {k: p for k, p in plans.items() if p.get("status") == "complete"}
    hook = [p["timing_seconds"]["hook_total"] for p in done.values() if p.get("timing_seconds")]
    scoring = [p["timing_seconds"]["module_scoring"] for p in done.values() if p.get("timing_seconds")]
    stats = [(p.get("test_plan") or {}).get("stats") or p.get("stats") or {} for p in done.values()]
    share = [s["in_budget_runtime_share"] for s in stats if s.get("in_budget_runtime_share") is not None]
    count_share = [s["in_budget_fraction"] for s in stats if s.get("in_budget_fraction") is not None]
    since = [s["commits_since_map"] for s in stats if s.get("commits_since_map") is not None]

    def dist(v: list[float]) -> dict | None:
        return None if not v else {"n": len(v), "median": float(np.median(v)), "p95": float(np.quantile(v, 0.95)),
                                   "max": float(max(v)), "min": float(min(v))}

    return {"runs": len(plans), "status": dict(status), "complete": len(done),
            "not_complete": sorted([k, p.get("status"), p.get("reason", "")] for k, p in plans.items()
                                   if p.get("status") != "complete"),
            "escalated": sum(bool(p.get("escalated")) for p in done.values()),
            "hook_seconds": dist(hook), "module_scoring_seconds": dist(scoring),
            "in_budget_runtime_share": dist(share), "in_budget_test_share": dist(count_share),
            "commits_since_coverage_map": dist(since),
            "stale_map_warnings": sum(bool(s.get("stale_map_warning")) for s in stats)}


def load(folder: str) -> dict[str, dict]:
    return {os.path.basename(os.path.dirname(f)): json.load(open(f))
            for f in sorted(glob.glob(os.path.join(folder, "*", "plan.json")))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plans")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = summarise(load(a.plans))
    text = json.dumps(out, indent=1)
    if a.out:
        open(a.out, "w").write(text)
    print(text)
    return 1 if out["not_complete"] else 0


if __name__ == "__main__":
    sys.exit(main())
