"""Fit the cross-project ApacheJIT models (section 3.8.2): hold out one project at a time and train on
the other 12 projects' training periods. Calibration and the MCC threshold are fitted on the source
projects' pooled calibration periods only. Never scores a test partition; evaluate.py does that.

    python -m riskrank.run_cross --seconds 60 --seed 42

Writes, under runs/<run-id>/: manifest.json; cross/<held-out project>/<family>.json and .joblib;
selection.json (mean tuning AUC-PR per family). Cross-project results are reported outside the
significance tests (section 3.10.4).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import numpy as np

from riskrank.apachejit import FEATURES, build
from riskrank.models import FAMILIES, fit_final, tune
from riskrank.run_within import LIBS, RAW, ZIP, _dump, git_commit, sha256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--projects", nargs="*", default=None, help="held-out projects (default: all 13)")
    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("cross-%Y%m%dT%H%M%SZ")
    out = os.path.join("runs", run_id)
    os.makedirs(out)
    commits = build(os.path.join(RAW, "apachejit")).commits
    projects = args.projects or sorted(commits["project"].unique())
    _dump({"run_id": run_id, "kind": "cross-project", "variant": None,
           "dataset": {"source": "Zenodo 5907847", "file": ZIP, "sha256": sha256(ZIP)},
           "code_commit": git_commit(), "python": platform.python_version(),
           "libraries": {lib: version(lib) for lib in LIBS},
           "settings": {"seconds": args.seconds, "seed": args.seed, "families": args.families},
           "projects": projects, "started": datetime.now(timezone.utc).isoformat()}, out, "manifest.json")

    aucpr: dict[str, list[float]] = {f: [] for f in args.families}
    for held in projects:
        src = commits[commits["project"] != held]
        base, calib = src[src["partition"] == "base"], src[src["partition"] == "calib"]
        folder = os.path.join(out, "cross", held.replace("/", "_"))
        os.makedirs(folder)
        for fam in args.families:
            t0 = time.monotonic()
            tuned = tune(base, FEATURES, fam, seconds=args.seconds, seed=args.seed)
            mv = fit_final(base, calib, FEATURES, tuned, seed=args.seed)
            rec = mv.record() | {"held_out": held, "source_projects": sorted(src["project"].unique()),
                                 "rows": {"base": len(base), "calib": len(calib)},
                                 "wall_seconds": round(time.monotonic() - t0, 1)}
            _dump(rec, folder, f"{fam}.json")
            joblib.dump(mv, os.path.join(folder, f"{fam}.joblib"))
            aucpr[fam].append(tuned.cv_aucpr)
            print(f"held out {held:24} {fam:5} cvAP={tuned.cv_aucpr:.3f} cal={mv.calibrator.method:8} "
                  f"{rec['wall_seconds']}s", flush=True)

    means = {f: float(np.mean(v)) for f, v in aucpr.items()}
    _dump({"mean_tuning_aucpr": means, "chosen_family": max(means, key=means.get),
           "families": args.families, "rule": "reported per family; outside the significance tests",
           "finished": datetime.now(timezone.utc).isoformat()}, out, "selection.json")
    print("cross-project selection:", means)


if __name__ == "__main__":
    main()
