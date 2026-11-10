"""Exploratory static-versus-process comparisons (section 3.8.2), exactly as pre-registered:

    python -m riskrank.static_vs_process flask  --seconds 60 --seed 42   # config/flask_mining.yaml, static_vs_process
    python -m riskrank.static_vs_process groups --run runs/<run of record>  # config/apachejit_d6.yaml, change_metric_groups

flask   Flask commit level: a static-only model against a static-plus-process model, every family, tuned
        for equal seconds on expanding time folds of the base data, calibrated on the final fifth before the
        gap; both scored once on the held-out commits. Plus the transfer check: the deployed
        apachejit-jit-v1, unchanged, scores the same held-out commits.
groups  ApacheJIT: each of four change-metric groups alone, within-project, with the run of record's
        chosen family, scored on each project's test partition.
Both report effect sizes only; no significance tests. Both are time-budgeted, so run them on an idle machine.
Writes runs/static_vs_process-<time>/report.json (flask) or runs/metric_groups-<time>/report.json (groups).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score
import yaml

from riskrank import registry
from riskrank.evaluate import reliability
from riskrank.jit_features import FEATURES as CHANGE
from riskrank.mining.commits import STATIC
from riskrank.models import FAMILIES, fit_final, tune
from riskrank.run_within import LIBS, RAW, git_commit, sha256

FLASK_CFG = "config/flask_mining.yaml"
D6_CFG = "config/apachejit_d6.yaml"
COMMITS = "data/mined/flask/commits.csv"
JIT_VERSION = "apachejit-jit-v1"


def _measures(y: np.ndarray, raw: np.ndarray, p: np.ndarray | None = None, threshold: float | None = None) -> dict:
    y = y.astype(bool)
    out = {"n": int(len(y)), "defects": int(y.sum()), "aucpr_chance": float(y.mean()),
           "aucpr": float(average_precision_score(y, raw)), "roc_auc": float(roc_auc_score(y, raw))}
    if p is not None:
        out["brier_calibrated"] = float(brier_score_loss(y, p))
        if threshold is not None:
            out["mcc"] = float(matthews_corrcoef(y, np.asarray(p) >= threshold))
            out["mcc_threshold"] = float(threshold)
    return out


def _manifest(kind: str, settings: dict, data: dict) -> dict:
    return {"kind": kind, "code_commit": git_commit(), "python": platform.python_version(),
            "libraries": {lib: version(lib) for lib in LIBS}, "settings": settings, "data": data,
            "started": datetime.now(timezone.utc).isoformat()}


# ------------------------------------------------------------------ Flask commit level
def flask_partitions(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = pd.Timestamp(cfg["split_date"])
    gap = split - pd.DateOffset(years=1)
    df = df.sort_values("when", kind="mergesort").reset_index(drop=True)
    pre = df[df["when"] < gap].copy()
    pre["label"] = pre["inducing"] & (pre["first_fix"] < split)
    test = df[df["when"] >= split].copy()
    test["label"] = test["inducing"]
    first = pre["when"].min()
    cal_start = first + (gap - first) * 0.8
    base, calib = pre[pre["when"] < cal_start], pre[pre["when"] >= cal_start]
    return base.reset_index(drop=True), calib.reset_index(drop=True), test.reset_index(drop=True)


def transfer_check(test: pd.DataFrame) -> dict:
    """The deployed commit model, unchanged, on the held-out Flask commits (section 3.10.1): ROC-AUC, AUC-PR,
    MCC at the model's own threshold, the calibrated Brier score and a ten-bin reliability curve."""
    expected = yaml.safe_load(open("riskrank.yml"))["models"]["commit"]  # the checksum the pipeline trusts
    assert expected["version"] == JIT_VERSION
    _meta, mv, _explainer = registry.load(os.path.join("registry", JIT_VERSION), expected["sha256"])
    p, raw = mv.score(test)
    y = test["label"].to_numpy().astype(bool)
    return _measures(y, raw, p, mv.threshold) | {
        "reliability": reliability(np.asarray(p), y.astype(int), 10),
        "model": JIT_VERSION, "features": mv.pipeline.features, "refitted": False}


def run_transfer() -> dict:
    cfg = yaml.safe_load(open(FLASK_CFG))["static_vs_process"]
    df = pd.read_csv(COMMITS, parse_dates=["when", "first_fix"])
    _base, _calib, test = flask_partitions(df, cfg)
    return {"transfer_check": transfer_check(test)}


def run_flask(seconds: float, seed: int, families: list[str]) -> dict:
    cfg = yaml.safe_load(open(FLASK_CFG))["static_vs_process"]
    df = pd.read_csv(COMMITS, parse_dates=["when", "first_fix"])
    base, calib, test = flask_partitions(df, cfg)
    sets = cfg["feature_sets"]
    rep = {"partitions": {k: {"commits": len(v), "inducing": int(v["label"].sum()),
                              "from": str(v["when"].min()), "to": str(v["when"].max())}
                          for k, v in (("base", base), ("calib", calib), ("test", test))},
           "results": {}}
    for fam in families:
        rep["results"][fam] = {}
        for name, feats in sets.items():
            t0 = time.monotonic()
            tuned = tune(base, feats, fam, seconds=seconds, seed=seed)
            mv = fit_final(base, calib, feats, tuned, seed=seed)
            p, raw = mv.score(test)
            rep["results"][fam][name] = _measures(test["label"].to_numpy(), raw, p) | {
                "tuning_aucpr": tuned.cv_aucpr, "imbalance": tuned.imbalance, "trials": tuned.trials,
                "calibration": mv.calibrator.method, "features_kept": mv.pipeline.kept_,
                "wall_seconds": round(time.monotonic() - t0, 1)}
            print(f"flask {fam:5} {name:15} test AP={rep['results'][fam][name]['aucpr']:.3f} "
                  f"(tuning {tuned.cv_aucpr:.3f})", flush=True)
        r = rep["results"][fam]
        r["aucpr_difference"] = r["static_process"]["aucpr"] - r["static"]["aucpr"]
    tuning = {f: rep["results"][f]["static_process"]["tuning_aucpr"] for f in families}
    rep["headline_family"] = max(tuning, key=tuning.get)
    rep["headline_rule"] = "highest tuning AUC-PR on static_process (fixed by the tuning data, not the test data)"

    rep["transfer_check"] = transfer_check(test)
    print("transfer", JIT_VERSION, rep["transfer_check"]["aucpr"], flush=True)
    rep["reporting"] = cfg["reporting"]
    return rep


# ------------------------------------------------------------------ ApacheJIT groups
def run_groups(run_dir: str, seconds: float, seed: int) -> dict:
    from riskrank.apachejit import build

    cfg = yaml.safe_load(open(D6_CFG))["change_metric_groups"]
    fam = json.load(open(os.path.join(run_dir, "selection.json")))["chosen_family"]
    commits = build(os.path.join(RAW, "apachejit")).commits
    rep = {"family": fam, "run_of_record": run_dir, "per_project": {}, "groups": cfg["groups"]}
    for proj in sorted(commits["project"].unique()):
        p = commits[commits["project"] == proj]
        base, calib, test = (p[p["partition"] == k] for k in ("base", "calib", "test"))
        rep["per_project"][proj] = {}
        for g, feats in cfg["groups"].items():
            tuned = tune(base, feats, fam, seconds=seconds, seed=seed)
            mv = fit_final(base, calib, feats, tuned, seed=seed)
            pr, raw = mv.score(test)
            rep["per_project"][proj][g] = _measures(test["label"].to_numpy(), raw, pr) | {
                "tuning_aucpr": tuned.cv_aucpr, "trials": tuned.trials}
            print(f"groups {proj:24} {g:10} AP={rep['per_project'][proj][g]['aucpr']:.3f}", flush=True)
    rep["mean_over_projects"] = {g: {m: float(np.mean([rep["per_project"][q][g][m] for q in rep["per_project"]]))
                                     for m in ("aucpr", "aucpr_chance", "roc_auc")} for g in cfg["groups"]}
    rep["reporting"] = cfg["reporting"]
    return rep


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["flask", "groups", "transfer"])
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--run", help="the run of record (groups only)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = {"flask": "static_vs_process-", "groups": "metric_groups-", "transfer": "transfer_check-"}[a.what]
    out = a.out or os.path.join("runs", prefix + stamp)
    os.makedirs(out, exist_ok=True)
    settings = {"seconds": a.seconds, "seed": a.seed}
    if a.what == "transfer":  # no tuning: the deployed model only, so it needs no idle machine
        man = _manifest("transfer_check", {}, {"commits_csv_sha256": sha256(COMMITS), "config": FLASK_CFG})
        json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
        rep = run_transfer()
    elif a.what == "flask":
        man = _manifest("static_vs_process", settings | {"families": a.families},
                        {"commits_csv_sha256": sha256(COMMITS), "config": FLASK_CFG, "config_sha256": sha256(FLASK_CFG)})
        json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
        rep = run_flask(a.seconds, a.seed, a.families)
    else:
        if not a.run:
            raise SystemExit("--run must name the run of record")
        man = _manifest("change_metric_groups", settings | {"run_of_record": a.run},
                        {"config": D6_CFG, "config_sha256": sha256(D6_CFG)})
        json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
        rep = run_groups(a.run, a.seconds, a.seed)
    rep["finished"] = datetime.now(timezone.utc).isoformat()
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
    print("wrote", out)


if __name__ == "__main__":
    main()
