"""Fit and publish the deployed Flask module-level model (FR5; plan D5).

    python -m riskrank.deploy_module --seconds 60 --seed 42 --version flask-module-v1

Training data: Flask module snapshots whose 12-month observation window closes on or before the fork
point (config deployment_model.fork_point), so the deployed model never sees a label from the period
it will rank in the replay. The final fifth of those snapshots by date is the calibration partition
(section 3.8.2); the rest is base data. Each family is tuned for the same number of seconds and the
live model is the one with the highest tuning AUC-PR (FR5). Writes runs/<id>/deploy_module.json and
publishes to registry/<version>/.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from riskrank import registry
from riskrank.mining.fixes import load_config
from riskrank.models import FAMILIES, fit_final, tune

FEATURES = ["size", "complexity", "churn", "changes", "prior_fixes"]  # Table 3.7 order


def frames(fork_point: pd.Timestamp, window_months: int):
    d = pd.read_csv("data/mined/flask/modules.csv", parse_dates=["snapshot"])
    d["when"] = d["snapshot"]
    d["id"] = d["snapshot"].dt.strftime("%Y-%m-%d") + ":" + d["module"]
    train = d[d["snapshot"] + pd.DateOffset(months=window_months) <= fork_point]
    snaps = sorted(train["snapshot"].unique())
    cut = snaps[int(len(snaps) * 0.8)]
    return train[train["snapshot"] < cut], train[train["snapshot"] >= cut]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--version", default="flask-module-v1")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES),
                    help="candidate families; flask-module-v3 uses lr rf lgbm, the set v2 was chosen from")
    a = ap.parse_args()
    cfg = load_config()
    fork = pd.Timestamp(cfg["deployment_model"]["fork_point"], tz="UTC")
    base, calib = frames(fork, cfg["module_snapshots"]["observation_window_months"])
    tuned = {f: tune(base, FEATURES, f, seconds=a.seconds, seed=a.seed) for f in a.families}
    best = max(tuned, key=lambda f: tuned[f].cv_aucpr)
    mv = fit_final(base, calib, FEATURES, tuned[best], seed=a.seed)
    meta = registry.publish(
        mv, base, version=a.version, granularity="module", corpus=["pallets/flask"],
        training_window={"base": [str(base["snapshot"].min().date()), str(base["snapshot"].max().date())],
                         "calibration": [str(calib["snapshot"].min().date()), str(calib["snapshot"].max().date())],
                         "fork_point": str(fork.date())},
        training_range=None, root="registry", seed=a.seed)
    out = {"created": datetime.now(timezone.utc).isoformat(), "version": a.version, "chosen": best,
           "tuning_aucpr": {f: t.cv_aucpr for f, t in tuned.items()},
           "trials": {f: t.trials for f, t in tuned.items()},
           "rows": {"base": len(base), "calib": len(calib)},
           "defective": {"base": int(base["label"].sum()), "calib": int(calib["label"].sum())},
           "bundle_sha256": meta["bundle_sha256"]}
    os.makedirs("runs", exist_ok=True)
    with open(f"runs/deploy_module_{a.version}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
