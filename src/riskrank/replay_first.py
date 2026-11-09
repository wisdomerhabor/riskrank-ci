"""Replay harness for the first bounded check (section 3.10.3): runtime-set defect recall on Flask.
Evaluation only; never part of the deployed pipeline. Rules: config/flask_mining.yaml, first_check.

    python -m riskrank.replay_first --seconds 60 --seed 42

Steps, in the pre-registered order:
1. the separation date: the earliest month's first commit that leaves the training-snapshot minimum
   (recorded, with every candidate date, before any recall is calculated);
2. the validation model, trained only on those snapshots, chosen and calibrated as the deployed model is;
3. each anchor (an R-SZZ inducing commit on or after the date) is replayed as a single-commit push:
   touched modules and their features come from the CI hook's own code, scores from the model's own
   scoring code, the ranking and the 20 percent budgets from riskrank.budget, all unmodified;
4. recall of the R-SZZ-blamed modules inside the budget, against each anchor's own chance level and
   the largest-first, smallest-first and recency baselines.
Writes runs/first_check.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from riskrank.budget import mark, rank
from riskrank.deploy_module import FEATURES
from riskrank.hook import module_features, touched_modules
from riskrank.mining.fixes import REPO, load_config
from riskrank.mining.snapshots import history, last_change
from riskrank.models import FAMILIES, fit_final, tune

B = 0.2


# ------------------------------------------------------------------ 1. separation date
def candidate_dates(times: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """The first commit (by committer time, merges included) of each calendar month."""
    firsts: dict[tuple[int, int], pd.Timestamp] = {}
    for t in times:
        key = (t.year, t.month)
        if key not in firsts or t < firsts[key]:
            firsts[key] = t
    return sorted(firsts.values())


def training_snapshots(snapshots: list[pd.Timestamp], date: pd.Timestamp, months: int) -> list[pd.Timestamp]:
    """Snapshots whose observation window ends strictly before `date`."""
    return [s for s in snapshots if s + pd.DateOffset(months=months) < date]


def separation_date(candidates, snapshots, months: int, minimum: int) -> tuple[pd.Timestamp, list[dict]]:
    table = [{"date": d, "training_snapshots": len(training_snapshots(snapshots, d, months))} for d in candidates]
    chosen = next(r["date"] for r in table if r["training_snapshots"] >= minimum)
    return chosen, table


# ------------------------------------------------------------------ 3. labels
def back_to_anchor(commits: list[dict], paths: set[str], anchor_when, fix_when) -> dict[str, str]:
    """Map each path as it was at the fix to its name at the anchor, following renames made in between."""
    alias = {p: p for p in paths}  # name at the fix -> name now (walking back in time)
    for c in reversed(commits):
        if c["when"] >= fix_when:
            continue
        if c["when"] <= anchor_when:
            break
        for f in c["files"]:
            if f["old"] != f["new"]:
                for k, v in alias.items():
                    if v == f["new"]:
                        alias[k] = f["old"]
    return alias


def anchors(rszz: list[dict], date: pd.Timestamp) -> dict[str, dict]:
    """Inducing commits on or after the date, with every fix that R-SZZ traces back to them."""
    out: dict[str, dict] = {}
    for r in rszz:
        if not r["inducing"]:
            continue
        when = pd.Timestamp(r["inducing_date"]).tz_convert("UTC")
        if when < date:
            continue
        a = out.setdefault(r["inducing"], {"when": when, "fixes": []})
        a["fixes"].append({"fix": r["fix_commit_hash"], "when": pd.Timestamp(r["fix_date"]).tz_convert("UTC"),
                           "blamed": r["blamed_files"]})
    return out


# ------------------------------------------------------------------ 4. recall and chance
def recall(ordered: pd.DataFrame, unit: str, blamed: set[str]) -> tuple[float, float]:
    """(defect recall inside the budget, realised share) for one ordering of the ranked set."""
    m = mark(ordered, unit, B, "default")
    found = len(set(ordered.loc[m, "id"]) & blamed)
    total = ordered["size"].sum()
    share = float(m.mean()) if unit == "artefacts" else (float(ordered.loc[m, "size"].sum() / total) if total else 0.0)
    return found / len(blamed), share


def lines_chance(frame: pd.DataFrame, blamed: set[str], seed: int = 0, draws: int = 100) -> dict:
    n = len(frame)
    if n <= 8:
        orders = [list(p) for p in itertools.permutations(range(n))]
        method = "every order"
    else:
        rng = np.random.default_rng(seed)
        orders = [list(rng.permutation(n)) for _ in range(draws)]
        method = f"{draws} seeded orders"
    vals = np.array([recall(frame.iloc[o].reset_index(drop=True), "lines", blamed)[0] for o in orders])
    return {"mean": float(vals.mean()), "interval": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
            "method": method}


def evaluate_anchor(frame: pd.DataFrame, blamed: set[str], last: dict[str, pd.Timestamp | None]) -> dict:
    """frame: id, p, raw, size for the ranked set; blamed: blamed modules inside it."""
    system = rank(frame, "default")
    largest = frame.sort_values(["size", "id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    smallest = frame.sort_values(["size", "id"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    rec = frame.assign(_t=frame["id"].map(lambda m: last.get(m)))
    rec = rec.assign(_has=rec["_t"].notna(), _ts=rec["_t"].map(lambda t: t.value if t is not None else 0))
    recency = (rec.sort_values(["_has", "_ts", "id"], ascending=[False, False, True], kind="mergesort")
               .drop(columns=["_t", "_has", "_ts"]).reset_index(drop=True))
    out = {"modules": len(frame), "blamed": len(blamed)}
    for unit in ("artefacts", "lines"):
        r, share = recall(system, unit, blamed)
        out[f"system_{unit}"], out[f"realised_share_{unit}"] = r, share
        out[f"recency_{unit}"] = recall(recency, unit, blamed)[0]
    out["largest_first_artefacts"] = recall(largest, "artefacts", blamed)[0]
    out["smallest_first_lines"] = recall(smallest, "lines", blamed)[0]
    out["chance_artefacts"] = out["realised_share_artefacts"]
    ch = lines_chance(frame, blamed)
    out["chance_lines"], out["chance_lines_interval"], out["chance_lines_method"] = ch["mean"], ch["interval"], ch["method"]
    return out


# ------------------------------------------------------------------ the run
def git(*args: str) -> str:
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = load_config()
    fc = cfg["first_check"]
    months = cfg["module_snapshots"]["observation_window_months"]
    keywords = cfg["bug_fix_rule"]["keywords"]
    commits = history(cfg["history_end_commit"])
    mods = pd.read_csv("data/mined/flask/modules.csv", parse_dates=["snapshot"])
    snapshots = sorted(mods["snapshot"].unique())

    # 1. the separation date, recorded before anything is measured
    times = [pd.Timestamp(t).tz_convert("UTC") for t in git("log", "--format=%cI", cfg["history_end_commit"]).split()]
    date, table = separation_date(candidate_dates(times), snapshots, months, fc["training_snapshot_minimum"])
    train_snaps = training_snapshots(snapshots, date, months)
    record = {"created": datetime.now(timezone.utc).isoformat(), "pre_registration": fc,
              "separation_date": str(date), "training_snapshots": [str(s.date()) for s in train_snaps]}
    os.makedirs("runs", exist_ok=True)
    with open("runs/first_check_date.json", "w") as f:
        json.dump(record, f, indent=1)
    print("separation date", date, "| training snapshots", len(train_snaps), flush=True)

    # 2. the validation model
    train = mods[mods["snapshot"].isin(train_snaps)].copy()
    train["when"] = train["snapshot"]
    train["id"] = train["snapshot"].dt.strftime("%Y-%m-%d") + ":" + train["module"]
    cut = train_snaps[int(len(train_snaps) * 0.8)]
    base, calib = train[train["snapshot"] < cut], train[train["snapshot"] >= cut]
    tuned = {f: tune(base, FEATURES, f, seconds=a.seconds, seed=a.seed) for f in FAMILIES}
    best = max(tuned, key=lambda f: tuned[f].cv_aucpr)
    mv = fit_final(base, calib, FEATURES, tuned[best], seed=a.seed)
    model = {"family": best, "tuning_aucpr": {f: t.cv_aucpr for f, t in tuned.items()},
             "rows": {"base": len(base), "calib": len(calib)},
             "defective": {"base": int(base["label"].sum()), "calib": int(calib["label"].sum())},
             "record": mv.record()}
    print("validation model", best, model["rows"], flush=True)

    # 3-4. replay every anchor
    rszz = [json.loads(line) for line in open("data/mined/flask/rszz.jsonl")]
    found = anchors(rszz, date)
    anchors_on_or_after = {str(r["date"]): len(anchors(rszz, r["date"])) for r in table if r["date"] >= date}
    results, excluded = [], {"merge_commit": 0, "no_scoreable_module": 0, "blamed_outside_ranked_set": 0,
                             "every_module_blamed": 0}
    files_removed = 0
    for sha, info in sorted(found.items(), key=lambda kv: kv[1]["when"]):
        parents = git("rev-list", "--parents", "-n", "1", sha).split()[1:]
        if len(parents) != 1:
            excluded["merge_commit"] += 1
            continue
        touched = touched_modules(REPO, parents[0], sha, cfg["source_modules"], cfg["exclude_paths"])
        feats = module_features(REPO, sha, touched, cfg["module_snapshots"]["process_period_months"], keywords)
        feats = feats.dropna(subset=["size"])
        if feats.empty:
            excluded["no_scoreable_module"] += 1
            continue
        blamed_all = set()
        for fx in info["fixes"]:
            back = back_to_anchor(commits, set(fx["blamed"]), info["when"], fx["when"])
            blamed_all |= {back[p] for p in fx["blamed"]}
        ranked_ids = set(feats["id"])
        blamed = blamed_all & ranked_ids
        files_removed += len(blamed_all - ranked_ids)
        if not blamed:
            excluded["blamed_outside_ranked_set"] += 1
            continue
        if blamed == ranked_ids:
            excluded["every_module_blamed"] += 1
            continue
        p, raw = mv.score(feats)
        frame = pd.DataFrame({"id": feats["id"].to_numpy(), "p": p, "raw": raw, "size": feats["size"].to_numpy()})
        last = last_change(history(sha), frame["id"].tolist(), info["when"])  # the anchor's own history
        row = {"anchor": sha, "when": str(info["when"]), "fixes": len(info["fixes"]),
               "blamed_removed_untouched": len(blamed_all - ranked_ids)} | evaluate_anchor(frame, blamed, last)
        results.append(row)
        print(f"{sha[:8]} modules={row['modules']} blamed={row['blamed']} "
              f"sys_lines={row['system_lines']:.2f} chance={row['chance_lines']:.2f}", flush=True)

    df = pd.DataFrame(results)
    summary = {"eligible_anchors": len(df), "meets_minimum": len(df) >= fc["minimum_anchors"]}
    if len(df):
        for unit in ("artefacts", "lines"):
            diff = df[f"system_{unit}"] - df[f"chance_{unit}"]
            summary[f"difference_from_chance_{unit}"] = {"mean": float(diff.mean()), "sd": float(diff.std(ddof=1)) if len(df) > 1 else None,
                                                         "min": float(diff.min()), "max": float(diff.max())}
        for col in ["system_artefacts", "system_lines", "chance_artefacts", "chance_lines", "largest_first_artefacts",
                    "smallest_first_lines", "recency_artefacts", "recency_lines"]:
            summary[f"mean_{col}"] = float(df[col].mean())
        summary["blamed_per_anchor"] = df["blamed"].value_counts().sort_index().to_dict()
    out = record | {"candidates": [{"date": str(r["date"]), "training_snapshots": r["training_snapshots"]} for r in table],
                    "inducing_commits_on_or_after_candidate": anchors_on_or_after, "validation_model": model,
                    "anchors_found": len(found), "excluded": excluded, "blamed_files_removed_untouched": files_removed,
                    "summary": summary, "per_anchor": results}
    with open("runs/first_check.json", "w") as f:
        json.dump(out, f, indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
