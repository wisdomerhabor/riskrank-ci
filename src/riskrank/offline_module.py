"""Offline evaluation of the deployed Flask module model on its held-out snapshots (FR13, FR15),
exactly as pre-registered in config/flask_mining.yaml (offline_evaluation).

    python -m riskrank.offline_module --db sqlite:///riskrank.db

Each held-out snapshot is one ranked set. Writes runs/offline_<model>.json and stores the run, its
baselines, the cost-effectiveness curves and the reliability bins through the storage interface.
Results are effect sizes only: the mined corpus is too small for significance tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from riskrank import registry
from riskrank.budget import BUDGET, rank, recall_lines
from riskrank.deploy_module import FEATURES
from riskrank.evaluate import SETTINGS, rank_biserial, recall_block, reliability, uncalibrated_probability
from riskrank.mining.fixes import load_config
from riskrank.mining.snapshots import history, last_change
from riskrank.store import Store

GRID = np.round(np.arange(0, 1.0001, 0.01), 2)


def held_out(cfg: dict) -> pd.DataFrame:
    d = pd.read_csv("data/mined/flask/modules.csv", parse_dates=["snapshot"])
    fork = pd.Timestamp(cfg["deployment_model"]["fork_point"], tz="UTC")
    return d[d["snapshot"] > fork].reset_index(drop=True)


def orderings(f: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"system": rank(f, "default"),
            "largest-first": f.sort_values(["size", "id"], ascending=[False, True], kind="mergesort"),
            "smallest-first": f.sort_values(["size", "id"], ascending=[True, True], kind="mergesort"),
            "recency-based": f.sort_values(["when", "id"], ascending=[False, True], kind="mergesort")}


def curves(frames: list[pd.DataFrame]) -> tuple[dict[str, list], np.ndarray]:
    """Mean offline recall over snapshots at each lines budget; random as the mean of seeded permutations.
    Also returns the random baseline's mean-over-snapshots recall at the budget, one value per permutation."""
    out = {k: np.zeros(len(GRID)) for k in ["system", "random", "largest-first", "smallest-first", "recency-based"]}
    draws_at_budget = np.zeros(SETTINGS["random_draws_lines"])
    for f in frames:
        for name, r in orderings(f).items():
            out[name] += [recall_lines(r, b=x) for x in GRID]
        rng = np.random.default_rng(SETTINGS["random_seed"])
        for i in range(SETTINGS["random_draws_lines"]):
            r = f.iloc[rng.permutation(len(f))]
            out["random"] += np.array([recall_lines(r, b=x) for x in GRID]) / SETTINGS["random_draws_lines"]
            draws_at_budget[i] += recall_lines(r, b=BUDGET)
    n = len(frames)
    return {k: [[float(x), float(y / n)] for x, y in zip(GRID, v)] for k, v in out.items()}, draws_at_budget / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("RISKRANK_DB", "sqlite:///riskrank.db"))
    ap.add_argument("--model", help="evaluate another registry version under the same pre-registered design "
                                    "(default: offline_evaluation.model)")
    a = ap.parse_args()
    cfg = load_config()
    oe = cfg["offline_evaluation"]
    version = a.model or oe["model"]
    meta = json.load(open(f"registry/{version}/model.json"))
    _, mv, _ = registry.load(f"registry/{version}", meta["bundle_sha256"])

    test = held_out(cfg)
    p, raw = mv.score(test)
    test["p"], test["raw"], test["p_uncal"] = p, raw, uncalibrated_probability(mv.model, raw)
    frames, per_snapshot = [], {}
    for snap, g in test.groupby("snapshot"):
        when = last_change(history(g["tree"].iloc[0]), g["module"].tolist(), snap)  # visible at snap only
        f = pd.DataFrame({"id": g["module"], "p": g["p"], "raw": g["raw"], "p_uncal": g["p_uncal"],
                          "size": g["size"], "label": g["label"].astype(bool),
                          "when": g["module"].map(when)}).reset_index(drop=True)
        frames.append(f)
        per_snapshot[str(snap.date())] = recall_block(f)

    y = test["label"].astype(int).to_numpy()
    curve, random_draws = curves(frames)
    keys = ["system_default_lines", "system_default_artefacts", "system_effort_lines", "system_effort_artefacts",
            "random_lines", "random_artefacts", "largest_first_artefacts", "largest_first_lines",
            "smallest_first_lines", "recency_artefacts", "recency_lines"]
    mean = {k: float(np.nanmean([s[k] for s in per_snapshot.values()])) for k in keys}
    effects = {}
    for base, budget in [("random", "artefacts"), ("random", "lines"), ("largest_first", "artefacts"),
                         ("smallest_first", "lines"), ("recency", "artefacts"), ("recency", "lines")]:
        for key in ("default", "effort"):
            diff = np.array([s[f"system_{key}_{budget}"] - s[f"{base}_{budget}"] for s in per_snapshot.values()])
            effects[f"{key}_vs_{base}_{budget}"] = {"mean_difference": float(diff.mean()),
                                                    "rank_biserial": rank_biserial(diff),
                                                    "pairs": len(diff)}
    created = datetime.now(timezone.utc).isoformat()
    out = {"created": created, "model": version, "bundle_sha256": meta["bundle_sha256"],
           "pre_registration": oe, "snapshots": list(per_snapshot), "rows": len(test), "defective": int(y.sum()),
           "predictive": {"roc_auc": float(roc_auc_score(y, raw)), "aucpr": float(average_precision_score(y, raw)),
                          "aucpr_chance": float(y.mean()), "brier": float(brier_score_loss(y, p))},
           "mean_over_snapshots": mean, "per_snapshot": per_snapshot, "effects": effects,
           "random_lines_interval": [float(np.quantile(random_draws, 0.025)), float(np.quantile(random_draws, 0.975))]}
    os.makedirs("runs", exist_ok=True)
    with open(f"runs/offline_{version}.json", "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o))

    store = Store(a.db)
    store.put_model(meta)
    store.put_offline_run(
        project="pallets/flask", corpus_name="Flask module snapshots after the fork point",
        corpus_projects=["pallets/flask"], model=version, created=created, budget_unit="lines",
        budget=BUDGET * 100, ranking_key="default", recall=mean["system_default_lines"],
        baselines=[
            {"baseline": "random", "recall": mean["random_lines"], "permutations": SETTINGS["random_draws_lines"],
             "interval_low": out["random_lines_interval"][0], "interval_high": out["random_lines_interval"][1]},
            {"baseline": "largest-first", "recall": mean["largest_first_lines"]},
            {"baseline": "smallest-first", "recall": mean["smallest_first_lines"]},
            {"baseline": "recency-based", "recall": mean["recency_lines"]}],
        detail={"curves": curve, "reliability": reliability(p, y, SETTINGS["reliability_bins"]),
                "brier": out["predictive"]["brier"], "rows": len(test), "defective": int(y.sum()),
                "snapshots": list(per_snapshot)},
        build_ref=f"offline-{version}-{created[:19]}")
    print(json.dumps({k: out[k] for k in ("snapshots", "rows", "defective")}, indent=1))


if __name__ == "__main__":
    main()
