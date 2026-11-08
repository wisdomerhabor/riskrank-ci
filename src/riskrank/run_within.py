"""Fit every within-project ApacheJIT model and record it (NFR13). Never scores a test partition.

    python -m riskrank.run_within --seconds 60 --seed 42

Writes, under runs/<run-id>/:
    manifest.json                      dataset checksum, code commit, library versions, settings
    within/<project>/<family>.json     the model version record (models.ModelVersion.record)
    within/<project>/<family>.joblib   the fitted model version, for the frozen evaluation
    selection.json                     mean tuning AUC-PR per family across projects, and the single
                                       family chosen for the baseline comparisons (section 3.10.4),
                                       fixed here, before any test data are scored
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import numpy as np

from riskrank.apachejit import FEATURES, build
from riskrank.models import FAMILIES, fit_final, tune

RAW = "data/raw"
ZIP = os.path.join(RAW, "apachejit.zip")
LIBS = ["pandas", "numpy", "scikit-learn", "lightgbm", "imbalanced-learn", "shap"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True)
    return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--projects", nargs="*", default=None)
    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("within-%Y%m%dT%H%M%SZ")
    out = os.path.join("runs", run_id)
    os.makedirs(out)

    commits = build(os.path.join(RAW, "apachejit")).commits
    projects = args.projects or sorted(commits["project"].unique())

    manifest = {
        "run_id": run_id,
        "kind": "within-project",
        "dataset": {"source": "Zenodo 5907847", "file": ZIP, "sha256": sha256(ZIP)},
        "code_commit": git_commit(),
        "python": platform.python_version(),
        "libraries": {lib: version(lib) for lib in LIBS},
        "settings": {"seconds": args.seconds, "seed": args.seed, "families": args.families},
        "projects": projects,
        "started": datetime.now(timezone.utc).isoformat(),
    }
    _dump(manifest, out, "manifest.json")

    aucpr: dict[str, list[float]] = {f: [] for f in args.families}
    for proj in projects:
        p = commits[commits["project"] == proj]
        base, calib = p[p["partition"] == "base"], p[p["partition"] == "calib"]
        folder = os.path.join(out, "within", proj.replace("/", "_"))
        os.makedirs(folder)
        for fam in args.families:
            t0 = time.monotonic()
            tuned = tune(base, FEATURES, fam, seconds=args.seconds, seed=args.seed)
            mv = fit_final(base, calib, FEATURES, tuned, seed=args.seed)
            rec = mv.record() | {"project": proj, "wall_seconds": round(time.monotonic() - t0, 1)}
            _dump(rec, folder, f"{fam}.json")
            joblib.dump(mv, os.path.join(folder, f"{fam}.joblib"))
            aucpr[fam].append(tuned.cv_aucpr)
            print(f"{proj:24} {fam:5} cvAP={tuned.cv_aucpr:.3f} {tuned.imbalance:6} "
                  f"cal={mv.calibrator.method:8} trials={tuned.trials} {rec['wall_seconds']}s",
                  flush=True)

    means = {f: float(np.mean(v)) for f, v in aucpr.items()}
    chosen = max(means, key=means.get)
    _dump(
        {"mean_tuning_aucpr": means, "chosen_family": chosen,
         "rule": "single family with the highest mean tuning AUC-PR across projects (section 3.10.4)",
         "fixed_before_test_scoring": True,
         "finished": datetime.now(timezone.utc).isoformat()},
        out, "selection.json",
    )
    print("selection:", means, "->", chosen)


def _dump(obj: dict, folder: str, name: str) -> None:
    with open(os.path.join(folder, name), "w") as f:
        json.dump(obj, f, indent=2, default=str)


if __name__ == "__main__":
    main()
