"""Frozen offline evaluation of a within-project ApacheJIT run (sections 3.10.1, 3.10.2, 3.10.4).

    python -m riskrank.evaluate runs/<run-id>

Every setting is fixed in SETTINGS below and written, with its hash, into the output BEFORE any
test partition is scored. Nothing here refits a model, a feature pipeline or a calibrator.

Per project, on its test partition (2017-2019) as one retrospective batch, and per test year:
  predictive   ROC-AUC and AUC-PR on uncalibrated scores, with chance levels; MCC at the threshold
               fixed on calibration; Brier score of the chosen map, the other map and the
               uncalibrated probability; a 10-bin reliability table.
  recall       defect recall at 20 percent, both budgets x both keys, for the system (calibrated);
               the effort key also from uncalibrated scores (descriptive only); and the baselines
               random, largest-first, smallest-first (lines budget only) and recency.
Across projects, for the family chosen in selection.json:
  the 8-test family: {random, size comparator} x {artefacts, lines} x {default, effort}, where the
  size comparator is largest-first under artefacts and smallest-first under lines. Exact two-sided
  Wilcoxon signed-rank (zeros dropped and counted), Holm-Bonferroni within the family, and the
  matched-pairs rank-biserial correlation (Kerby, 2014).
Every commit-level result is retrospective; Bludau and Pretschner (2026) report that batch
evaluation can overstate recall at a budget by up to 55 percent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score

from riskrank.apachejit import build
from riskrank.budget import BUDGET, Z, random_recall, rank, recall_artefacts, recall_lines
from riskrank.models import LGBMClassifier, LogisticRegression

SETTINGS = {
    "budget": BUDGET,
    "lower_bound_z": Z,
    "size": "churn = la + ld",
    "keys": ["default", "effort"],
    "budgets": ["artefacts", "lines"],
    "tie_break": "uncalibrated score desc, then commit id asc",
    "random_draws_lines": 100,
    "random_seed": 0,
    "baseline_ties": "ascending commit id",
    "reliability_bins": 10,
    "family_tests": 8,
    "wilcoxon": "exact, two-sided, zero differences dropped",
    "correction": "Holm-Bonferroni within the family",
    "effect_size": "matched-pairs rank-biserial (Kerby, 2014)",
    "batch_warning": "retrospective batch; may overstate recall by up to 55% (Bludau & Pretschner, 2026)",
}


# ------------------------------------------------------------------ statistics
def rank_biserial(diff: np.ndarray) -> float:
    d = diff[diff != 0]
    if len(d) == 0:
        return math.nan
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone), same order as the input."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj, running = [0.0] * m, 0.0
    for step, i in enumerate(order):
        running = max(running, min(1.0, (m - step) * pvalues[i]))
        adj[i] = running
    return adj


def paired_test(system: np.ndarray, baseline: np.ndarray) -> dict:
    diff = system - baseline
    nonzero = diff[diff != 0]
    out = {
        "n_projects": int(len(diff)),
        "n_nonzero": int(len(nonzero)),
        "zero_pairs_dropped": int((diff == 0).sum()),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "diffs": [round(float(x), 4) for x in diff],
        "rank_biserial": rank_biserial(diff),
        "min_attainable_p": 2 / 2 ** len(nonzero) if len(nonzero) else math.nan,
    }
    out["p"] = float(wilcoxon(nonzero, method="exact").pvalue) if len(nonzero) else math.nan
    return out


# ----------------------------------------------------------------- per project
def uncalibrated_probability(model, raw: np.ndarray) -> np.ndarray:
    if isinstance(model, (LogisticRegression, LGBMClassifier)):
        return 1 / (1 + np.exp(-raw))
    return raw


def reliability(p: np.ndarray, y: np.ndarray, bins: int) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_p": float(p[m].mean()) if m.any() else None,
                     "observed": float(y[m].mean()) if m.any() else None})
    return rows


def recall_block(frame: pd.DataFrame) -> dict:
    """System and baseline recall on one ranked set (a project's batch or one test year)."""
    out = {}
    for key in SETTINGS["keys"]:
        r = rank(frame, key)
        out[f"system_{key}_artefacts"] = recall_artefacts(r)
        out[f"system_{key}_lines"] = recall_lines(r)
    unc = frame.assign(p=frame["p_uncal"])
    r = rank(unc, "effort")
    out["uncalibrated_effort_artefacts"] = recall_artefacts(r)
    out["uncalibrated_effort_lines"] = recall_lines(r)

    out["random_artefacts"] = random_recall(frame, "artefacts")
    rng = np.random.default_rng(SETTINGS["random_seed"])
    draws = [recall_lines(frame.iloc[rng.permutation(len(frame))]) for _ in range(SETTINGS["random_draws_lines"])]
    out["random_lines"] = float(np.mean(draws))
    out["random_lines_interval"] = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]

    largest = frame.sort_values(["size", "id"], ascending=[False, True], kind="mergesort")
    smallest = frame.sort_values(["size", "id"], ascending=[True, True], kind="mergesort")
    recent = frame.sort_values(["when", "id"], ascending=[False, True], kind="mergesort")
    out["largest_first_artefacts"] = recall_artefacts(largest)
    out["largest_first_lines"] = recall_lines(largest)
    out["smallest_first_lines"] = recall_lines(smallest)
    out["recency_artefacts"] = recall_artefacts(recent)
    out["recency_lines"] = recall_lines(recent)
    out["realised_share_artefacts"] = math.ceil(BUDGET * len(frame)) / len(frame)
    out["zero_size"] = int((frame["size"] <= 0).sum())
    out["n"], out["defects"] = int(len(frame)), int(frame["label"].sum())
    return out


def evaluate_project(mv, test: pd.DataFrame) -> dict:
    p, raw = mv.score(test)
    y = test["label"].to_numpy()
    other = "platt" if mv.calibrator.method == "isotonic" else "isotonic"
    p_unc = uncalibrated_probability(mv.model, raw)
    pred = {
        "roc_auc": float(roc_auc_score(y, raw)),
        "roc_auc_chance": 0.5,
        "aucpr": float(average_precision_score(y, raw)),
        "aucpr_chance": float(y.mean()),
        "mcc": float(matthews_corrcoef(y, p >= mv.threshold)),
        "mcc_threshold": mv.threshold,
        "brier_calibrated": float(brier_score_loss(y, p)),
        "brier_other_map": float(brier_score_loss(y, mv.calibrator(raw, other))),
        "other_map": other,
        "brier_uncalibrated": float(brier_score_loss(y, np.clip(p_unc, 0, 1))),
        "reliability": reliability(p, y, SETTINGS["reliability_bins"]),
    }
    frame = pd.DataFrame({"id": test["commit_id"].to_numpy(), "p": p, "raw": raw, "p_uncal": p_unc,
                          "size": (test["la"] + test["ld"]).to_numpy(), "label": y,
                          "when": test["when"].to_numpy()})
    batch = recall_block(frame)
    years = {}
    for yr, g in frame.groupby(pd.to_datetime(frame["when"]).dt.year):
        if g["label"].any():
            years[int(yr)] = recall_block(g.reset_index(drop=True))
    return {"predictive": pred, "recall_batch": batch, "recall_by_year": years}


# ------------------------------------------------------------------ the family
FAMILY = [
    ("random", "artefacts", "default"), ("random", "artefacts", "effort"),
    ("random", "lines", "default"), ("random", "lines", "effort"),
    ("largest_first", "artefacts", "default"), ("largest_first", "artefacts", "effort"),
    ("smallest_first", "lines", "default"), ("smallest_first", "lines", "effort"),
]


def family_tests(per_project: dict[str, dict]) -> list[dict]:
    rows = []
    for baseline, budget, key in FAMILY:
        sys_v = np.array([r["recall_batch"][f"system_{key}_{budget}"] for r in per_project.values()])
        base_v = np.array([r["recall_batch"][f"{baseline}_{budget}"] for r in per_project.values()])
        ok = ~(np.isnan(sys_v) | np.isnan(base_v))
        rows.append({"baseline": baseline, "budget": budget, "key": key,
                     "excluded_undefined": int((~ok).sum())} | paired_test(sys_v[ok], base_v[ok]))
    adj = holm([r["p"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = a
        r["significant_holm_0.05"] = bool(a < 0.05)
    return rows


def main(run_dir: str) -> None:
    sel = json.load(open(os.path.join(run_dir, "selection.json")))
    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    frozen = json.dumps(SETTINGS, sort_keys=True)
    kind, variant = manifest.get("kind", "within-project"), manifest.get("variant")
    out = {"settings": SETTINGS, "settings_sha256": hashlib.sha256(frozen.encode()).hexdigest(),
           "kind": kind, "variant": variant, "chosen_family": sel["chosen_family"], "per_family": {}}
    commits = build("data/raw/apachejit", variant).commits
    within = os.path.join(run_dir, "cross" if kind == "cross-project" else "within")
    for fam in sel.get("families", sel["mean_tuning_aucpr"]):
        per_project = {}
        for folder in sorted(os.listdir(within)):
            mv = joblib.load(os.path.join(within, folder, f"{fam}.joblib"))
            proj = folder.replace("_", "/", 1)
            test = commits[(commits["project"] == proj) & (commits["partition"] == "test")]
            per_project[proj] = evaluate_project(mv, test)
        out["per_family"][fam] = {"per_project": per_project}
    chosen = out["per_family"][sel["chosen_family"]]
    # Significance tests only for the primary within-project run (section 3.10.4); cross-project results
    # and the sensitivity variants are reported descriptively.
    if kind == "within-project" and not variant:
        chosen["family_tests"] = family_tests(chosen["per_project"])
    with open(os.path.join(run_dir, "evaluation.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o))
    print("wrote", os.path.join(run_dir, "evaluation.json"))


if __name__ == "__main__":
    main(sys.argv[1])
