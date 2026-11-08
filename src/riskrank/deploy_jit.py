"""Fit and publish the deployed just-in-time model (FR5, FR16; section 3.10.1; plan A4).

    python -m riskrank.deploy_jit --seconds 120 --seed 42 --version apachejit-jit-v1

One model over all 13 ApacheJIT projects: base partitions pooled for fitting, calibration periods
pooled for the single calibration map. Features: the 12 change metrics minus `asexp`, which the
shared feature code cannot reproduce from ApacheJIT (NFR12 check: rho -0.21); `ndev` is kept with its
measured skew (x0.45) stated. Student's decision, 8 Nov 2026. Training range: ApacheJIT's exclusion
limits, 100 files and 10,000 changed lines; commits beyond either are scored but never escalate.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from riskrank import registry
from riskrank.apachejit import FEATURES as ALL12, build
from riskrank.models import FAMILIES, fit_final, tune

FEATURES = [f for f in ALL12 if f != "asexp"]
TRAINING_RANGE = {"max_files": 100, "max_lines": 10000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--version", default="apachejit-jit-v1")
    a = ap.parse_args()
    c = build("data/raw/apachejit").commits
    base, calib = c[c["partition"] == "base"], c[c["partition"] == "calib"]
    tuned = {f: tune(base, FEATURES, f, seconds=a.seconds, seed=a.seed) for f in FAMILIES}
    best = max(tuned, key=lambda f: tuned[f].cv_aucpr)
    mv = fit_final(base, calib, FEATURES, tuned[best], seed=a.seed)
    meta = registry.publish(
        mv, base, version=a.version, granularity="commit", corpus=sorted(c["project"].unique()),
        training_window={"base": "each project's first commit to its calibration start",
                         "calibration": "final fifth of each project's history to 31 Dec 2015",
                         "gap": "2016", "dropped_features": ["asexp"]},
        training_range=TRAINING_RANGE, root="registry", seed=a.seed)
    out = {"created": datetime.now(timezone.utc).isoformat(), "version": a.version, "chosen": best,
           "features": FEATURES, "tuning_aucpr": {f: t.cv_aucpr for f, t in tuned.items()},
           "trials": {f: t.trials for f, t in tuned.items()},
           "rows": {"base": len(base), "calib": len(calib)},
           "calibration_method": mv.calibrator.method, "bundle_sha256": meta["bundle_sha256"]}
    os.makedirs("runs", exist_ok=True)
    with open(f"runs/deploy_jit_{a.version}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
