"""Module-level benchmark models on NASA MDP (D'') and PROMISE (section 3.5.1, Table 3.6), under the
protocol pre-registered in config/benchmarks.yaml (student decision 8 Nov 2026: 20 trials, 10 x 10).

    python -m riskrank.run_benchmarks all --jobs 6            # tune -> select -> score -> report
    python -m riskrank.run_benchmarks tune --out runs/bench-X # resumable; then select, score, report
    python -m riskrank.run_benchmarks all --check --jobs 6    # pipeline check: tiny settings, labelled as such

Phases, in order; each refuses to start until the one before it has finished:
  tune    every (corpus, mode, project, round, family): split, tune (a fixed number of trials per
          imbalance method, stratified 4-fold on the base data), and store ONLY the tuned
          hyperparameters and tuning scores -> tuned.jsonl. No test data is scored.
  select  per corpus, the family with the highest mean tuning AUC-PR across projects -> selection.json
          (section 3.10.4: selected before the test data are scored)
  score   refit each final model from its stored hyperparameters (same seed and one thread, so the
          same model), calibrate, fix the MCC threshold, and only then score its test data ->
          scores.jsonl and predictions.csv
  report  per-project means, the 8-test family for the chosen family of each corpus, and the cross-
          project tables (descriptive) -> report.json

Offline comparators only; never deployed (Table 3.6).
"""

from __future__ import annotations

import os

# One thread per numerical library in every process, set before NumPy loads: eight workers each
# starting their own BLAS/OpenMP pool oversubscribes the CPU, and threaded reductions can change the
# last digits of a result, which would break exact reproducibility from the seed.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "NUMEXPR_NUM_THREADS")
for _v in THREAD_VARS:
    os.environ[_v] = "1"

import argparse
import csv
import glob
import heapq
import itertools
import json
import math
import platform
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

from riskrank import benchmarks, models
from riskrank.evaluate import family_tests, recall_block, reliability, uncalibrated_probability
from riskrank.models import FAMILIES, N_FOLDS, Tuned, fit_final, tune
from riskrank.run_within import LIBS, git_commit, sha256

CONFIG = "config/benchmarks.yaml"
META = ["label", "size", "id"]
CALIB_SHARE = 0.2
FULL = {"trials": 20, "repeats": 10, "folds": 10, "seed": 42}
CHECK = {"trials": 1, "repeats": 1, "folds": 3, "seed": 42}
CHECK_PROJECTS = {"nasa": ["MC2", "KC3"], "promise": ["ivy", "log4j"]}
HEAVY = {"rf": 3, "lgbm": 2, "mlp": 2, "lr": 1}  # schedule big tasks first

DATA: dict = {}


# ------------------------------------------------------------------ data
def load_all() -> dict:
    return {"nasa": benchmarks.nasa(), "promise": benchmarks.promise()}


def _init(loader=None) -> None:
    """Worker set-up: one thread per model, and the corpora loaded once per process."""
    warnings.filterwarnings("ignore")
    threadpool_limits(1)
    models.N_JOBS = 1
    DATA.clear()
    DATA.update((loader or load_all)())


def nasa_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META]  # in the order supplied (the feature table)


def shared_nasa(data: dict) -> list[str]:
    common = set.intersection(*(set(nasa_features(d)) for d in data.values()))
    first = data[sorted(data)[0]]
    return [c for c in nasa_features(first) if c in common]


def _calib_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A stratified fifth for calibration and the MCC threshold; the rest is the base-model data."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=CALIB_SHARE, random_state=seed)
    b, c = next(sss.split(np.zeros(len(frame)), frame["label"].astype(int)))
    return frame.iloc[b].reset_index(drop=True), frame.iloc[c].reset_index(drop=True)


def split(t: dict, s: dict):
    """(base, calib, [(test name, test frame)], features) for one task; deterministic from the seed."""
    seed = s["seed"]
    if t["corpus"] == "nasa":
        data = {k: v for k, v in DATA["nasa"].items() if k in s["projects"]["nasa"]}
        if t["mode"] == "within":
            df = data[t["project"]]
            outer = RepeatedStratifiedKFold(n_splits=s["folds"], n_repeats=s["repeats"], random_state=seed)
            i = int(t["unit"][1:])
            tr, te = next(itertools.islice(outer.split(np.zeros(len(df)), df["label"].astype(int)), i, None))
            base, calib = _calib_split(df.iloc[tr], seed)
            return base, calib, [(t["unit"], df.iloc[te].reset_index(drop=True))], nasa_features(df)
        feats = shared_nasa(data)
        src = pd.concat([d[feats + META] for n, d in sorted(data.items()) if n != t["project"]], ignore_index=True)
        base, calib = _calib_split(src, seed)
        return base, calib, [(t["project"], data[t["project"]][feats + META])], feats
    data = {k: v for k, v in DATA["promise"].items() if k in s["projects"]["promise"]}
    feats = list(benchmarks.CK)
    if t["mode"] == "within":
        releases = dict(data[t["project"]])
        a, b = t["unit"].split(">")
        base, calib = _calib_split(releases[a], seed)
        return base, calib, [(b, releases[b])], feats
    src = pd.concat([x for sys_, rs in sorted(data.items()) if sys_ != t["project"] for _, x in rs], ignore_index=True)
    base, calib = _calib_split(src, seed)
    return base, calib, list(data[t["project"]]), feats


# ------------------------------------------------------------------ tasks
def key(t: dict) -> str:
    return "|".join((t["corpus"], t["mode"], t["project"], t["unit"], t["family"]))


def units(corpus: str, mode: str, project: str, s: dict) -> list[str]:
    if mode == "cross":
        return ["holdout"]
    if corpus == "nasa":
        return [f"r{i:03d}" for i in range(s["repeats"] * s["folds"])]
    names = [r for r, _ in DATA["promise"][project]]
    return [f"{a}>{b}" for a, b in zip(names, names[1:])]


def all_tasks(s: dict) -> list[dict]:
    out = []
    for corpus in s["corpora"]:
        for mode in s["modes"]:
            for project in s["projects"][corpus]:
                for u in units(corpus, mode, project, s):
                    for fam in s["families"]:
                        out.append({"corpus": corpus, "mode": mode, "project": project, "unit": u, "family": fam})
    return out


def _size_hint(t: dict, s: dict) -> int:
    if t["corpus"] == "nasa":
        n = len(DATA["nasa"][t["project"]]) if t["mode"] == "within" else sum(
            len(d) for k, d in DATA["nasa"].items() if k in s["projects"]["nasa"] and k != t["project"])
    elif t["mode"] == "within":
        n = len(dict(DATA["promise"][t["project"]])[t["unit"].split(">")[0]])
    else:
        n = sum(len(x) for k, rs in DATA["promise"].items() if k in s["projects"]["promise"] and k != t["project"]
                for _, x in rs)
    return n * HEAVY[t["family"]]


# ------------------------------------------------------------------ workers
def tune_task(t: dict, s: dict) -> dict:
    t0 = time.monotonic()
    base, calib, _tests, feats = split(t, s)  # the test frames are not touched in this phase
    out = {"key": key(t), **t, "n_base": len(base), "n_calib": len(calib), "defects_base": int(base["label"].sum())}
    try:
        tuned = tune(base, feats, t["family"], seed=s["seed"], trials=s["trials"], folds="stratified")
    except ValueError as e:
        return out | {"error": str(e)}
    return out | {"imbalance": tuned.imbalance, "params": tuned.params, "cv_aucpr": tuned.cv_aucpr,
                  "cv_recall_lines": tuned.cv_recall_lines, "trials": tuned.trials,
                  "weight_vs_smote": tuned.weight_vs_smote, "wall_seconds": round(time.monotonic() - t0, 2)}


def metrics(mv, test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    p, raw = mv.score(test)
    y = test["label"].to_numpy().astype(bool)
    other = "platt" if mv.calibrator.method == "isotonic" else "isotonic"
    p_unc = np.clip(uncalibrated_probability(mv.model, raw), 0, 1)
    both = 0 < y.sum() < len(y)
    pred = {
        "n": int(len(y)), "defects": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, raw)) if both else math.nan,
        "aucpr": float(average_precision_score(y, raw)) if y.any() else math.nan,
        "aucpr_chance": float(y.mean()),
        "mcc": float(matthews_corrcoef(y, p >= mv.threshold)),
        "brier_calibrated": float(brier_score_loss(y, p)),
        "brier_other_map": float(brier_score_loss(y, mv.calibrator(raw, other))),
        "brier_uncalibrated": float(brier_score_loss(y, p_unc)),
    }
    frame = pd.DataFrame({"id": test["id"].to_numpy(), "p": p, "raw": raw, "p_uncal": p_unc,
                          "size": test["size"].to_numpy(), "label": y})
    rec = recall_block(frame)
    rec.pop("random_lines_interval", None)
    return pred | {"recall": {k: v for k, v in rec.items() if k not in ("n", "defects")}}, frame


def score_task(rec: dict, s: dict) -> tuple[list[dict], list[list]]:
    t = {k: rec[k] for k in ("corpus", "mode", "project", "unit", "family")}
    base, calib, tests, feats = split(t, s)
    params = {k: tuple(v) if isinstance(v, list) else v for k, v in rec["params"].items()}
    tuned = Tuned(family=rec["family"], imbalance=rec["imbalance"], params=params, cv_aucpr=rec["cv_aucpr"],
                  cv_recall_lines=rec["cv_recall_lines"], trials=rec["trials"],
                  weight_vs_smote=rec["weight_vs_smote"], budget={"trials": s["trials"]}, folds="stratified")
    mv = fit_final(base, calib, feats, tuned, seed=s["seed"])
    rows, preds = [], []
    for name, test in tests:
        m, frame = metrics(mv, test)
        rows.append({"key": rec["key"], **t, "test": name, "imbalance": rec["imbalance"],
                     "calibration": mv.calibrator.method, "platt_slope": mv.calibrator.platt_slope,
                     "mcc_threshold": mv.threshold, "calib_mcc": mv.calib_mcc} | m)
        preds += [[rec["key"], name, i, int(lab), round(float(pp), 6), round(float(rr), 6)]
                  for i, lab, pp, rr in zip(frame["id"], frame["label"], frame["p"], frame["raw"])]
    return rows, preds


def run_pool(fn, items: list, s: dict, jobs: int, sink) -> None:
    if jobs <= 1:
        if not DATA:
            _init()
        for it in items:
            sink(fn(it, s))
        return
    with ProcessPoolExecutor(jobs, initializer=_init) as ex:
        futures = [ex.submit(fn, it, s) for it in items]
        for f in as_completed(futures):
            sink(f.result())


# ------------------------------------------------------------------ sharding (GitHub Actions)
# The full protocol is too large for one workstation (largest training sets ~250,000 rows), so it can run
# as N shards on separate machines. Tuning is split into pieces of (task, trials): a task too costly for one
# shard is split into single trials, which models.combine merges back into exactly what tune() returns.
# Shard assignment is deterministic (longest piece first onto the least-loaded shard), so every shard
# computes the same plan from the manifest alone.
SPLIT_SHARE = 0.25  # a task costing more than this share of one shard's work is split into single trials


def _cost(t: dict, s: dict) -> float:
    n = _size_hint(t, s)
    return n * math.log2(n + 2)


def _assign(items: list, cost, shards: int) -> list[list]:
    loads = [(0.0, i) for i in range(shards)]
    out: list[list] = [[] for _ in range(shards)]
    for it in sorted(items, key=lambda x: (-cost(x), json.dumps(x, sort_keys=True, default=str))):
        load, i = heapq.heappop(loads)
        out[i].append(it)
        heapq.heappush(loads, (load + cost(it), i))
    return out


def pieces(s: dict, shards: int) -> list[list[dict]]:
    pairs = [(imb, i) for imb in models.IMBALANCES for i in range(s["trials"])]
    tasks = all_tasks(s)
    total = sum(_cost(t, s) for t in tasks) * len(pairs)
    cap = SPLIT_SHARE * total / shards
    items = []
    for t in tasks:
        c = _cost(t, s)
        groups = [[p] for p in pairs] if c * len(pairs) > cap else [pairs]
        items += [{"key": key(t), "task": t, "pairs": g, "cost": c * len(g)} for g in groups]
    return _assign(items, lambda x: x["cost"], shards)


def score_shards(recs: list[dict], s: dict, shards: int) -> list[list[dict]]:
    return _assign(recs, lambda r: _cost(r, s), shards)


def piece_task(pc: dict, s: dict) -> dict:
    t0 = time.monotonic()
    t = pc["task"]
    base, calib, _tests, feats = split(t, s)  # the test frames are not touched in this phase
    pairs = [tuple(p) for p in pc["pairs"]]
    try:
        recs = models.run_trials(base, feats, t["family"], s["seed"], s["trials"], "stratified", pairs)
    except ValueError as e:  # no usable folds: every trial fails, so the merged task records the error
        recs = [{"imbalance": imb, "trial": i, "error": str(e)} for imb, i in pairs]
    return {"key": pc["key"], "records": recs, "n_base": len(base), "n_calib": len(calib),
            "defects_base": int(base["label"].sum()), "wall_seconds": round(time.monotonic() - t0, 2)}


# ------------------------------------------------------------------ files
def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut short when a run was stopped: that task is simply redone on resume
    return rows


def _append(path: str, rows: list[dict]) -> None:
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _dump(obj, out: str, name: str) -> None:
    with open(os.path.join(out, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=str)


def _progress(label: str, total: int):
    state = {"n": 0, "t0": time.monotonic()}

    def tick(msg: str) -> None:
        state["n"] += 1
        el = time.monotonic() - state["t0"]
        eta = el / state["n"] * (total - state["n"])
        print(f"[{label} {state['n']}/{total} {el / 3600:.2f}h eta {eta / 3600:.2f}h] {msg}", flush=True)
    return tick


# ------------------------------------------------------------------ phases
def phase_tune(out: str, s: dict, jobs: int) -> None:
    path = os.path.join(out, "tuned.jsonl")
    done = {r["key"] for r in _read_jsonl(path)}
    todo = [t for t in all_tasks(s) if key(t) not in done]
    todo.sort(key=lambda t: -_size_hint(t, s))
    tick = _progress("tune", len(todo))

    def sink(r):
        _append(path, [r])
        tick(f"{r['key']} " + (f"ERROR {r['error']}" if "error" in r else f"cvAP={r['cv_aucpr']:.3f} {r['wall_seconds']}s"))
    run_pool(tune_task, todo, s, jobs, sink)


def _shard(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"--shard {spec}: need 0 <= i < N")
    return i, n


def phase_tune_units(out: str, s: dict, jobs: int, shard: tuple[int, int]) -> None:
    i, n = shard
    path = os.path.join(out, f"trials-{i:03d}.jsonl")
    done = {(r["key"], x["imbalance"], x["trial"]) for f in sorted(glob.glob(os.path.join(out, "trials-*.jsonl")))
            for r in _read_jsonl(f) for x in r["records"]}
    todo = []
    for pc in pieces(s, n)[i]:
        left = [p for p in pc["pairs"] if (pc["key"], *p) not in done]
        if left:
            todo.append(pc | {"pairs": left})
    tick = _progress(f"tune shard {i}/{n}", len(todo))

    def sink(r):
        _append(path, [r])
        errs = sum("error" in x for x in r["records"])
        tick(f"{r['key']} {len(r['records'])} trials{f' ({errs} failed)' if errs else ''} {r['wall_seconds']}s")
    run_pool(piece_task, todo, s, jobs, sink)


def phase_tune_merge(out: str, s: dict) -> None:
    got: dict = defaultdict(lambda: {"records": {}, "wall_seconds": 0.0})
    for f in sorted(glob.glob(os.path.join(out, "trials-*.jsonl"))):
        for r in _read_jsonl(f):
            g = got[r["key"]]
            g.update({k: r[k] for k in ("n_base", "n_calib", "defects_base")})
            g["wall_seconds"] += r["wall_seconds"]
            for x in r["records"]:
                g["records"].setdefault((x["imbalance"], x["trial"]), x)
    need = 2 * s["trials"]
    tasks = all_tasks(s)
    short = [key(t) for t in tasks if len(got[key(t)]["records"]) < need]
    if short:
        raise SystemExit(f"tuning unfinished: {len(short)} tasks have missing trials; run the tune-units shards again")
    rows = []
    for t in tasks:
        g = got[key(t)]
        rec = {"key": key(t), **t, "n_base": g["n_base"], "n_calib": g["n_calib"], "defects_base": g["defects_base"]}
        try:
            tuned = models.combine(list(g["records"].values()), t["family"], s["seed"], s["trials"], "stratified")
        except ValueError as e:
            rows.append(rec | {"error": str(e)})
            continue
        rows.append(rec | {"imbalance": tuned.imbalance, "params": tuned.params, "cv_aucpr": tuned.cv_aucpr,
                           "cv_recall_lines": tuned.cv_recall_lines, "trials": tuned.trials,
                           "weight_vs_smote": tuned.weight_vs_smote, "wall_seconds": round(g["wall_seconds"], 2)})
    path = os.path.join(out, "tuned.jsonl")
    if os.path.exists(path):
        os.remove(path)
    _append(path, rows)
    print(f"merged {len(rows)} tasks ({sum('error' in r for r in rows)} tuning errors) -> {path}", flush=True)


def phase_score_merge(out: str, inputs: list[str]) -> None:
    dirs = sorted(d for pat in inputs for d in glob.glob(pat))
    rows = [r for d in dirs for r in _read_jsonl(os.path.join(d, "scores.jsonl"))]
    seen, uniq = set(), []
    for r in rows:
        k = (r["key"], r["test"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    path, ppath = os.path.join(out, "scores.jsonl"), os.path.join(out, "predictions.csv")
    for f in (path, ppath):
        if os.path.exists(f):
            os.remove(f)
    _append(path, uniq)
    frames = [pd.read_csv(os.path.join(d, "predictions.csv")) for d in dirs
              if os.path.exists(os.path.join(d, "predictions.csv"))]
    pd.concat(frames).drop_duplicates(["key", "test", "id"]).to_csv(ppath, index=False)
    print(f"merged {len(uniq)} score rows from {len(dirs)} shard folders -> {path}", flush=True)


def _expected(s: dict) -> set[str]:
    return {key(t) for t in all_tasks(s)}


def phase_select(out: str, s: dict) -> dict:
    recs = sorted(_read_jsonl(os.path.join(out, "tuned.jsonl")), key=lambda r: r["key"])  # order-independent means
    missing = _expected(s) - {r["key"] for r in recs}
    if missing:
        raise SystemExit(f"tuning unfinished: {len(missing)} tasks missing; run the tune phase again")
    sel = {"rule": "per corpus, the family with the highest mean tuning AUC-PR across its projects "
                   "(mean over a project's rounds or pairs first); selected before any test data is scored",
           "tuning_errors": [r["key"] for r in recs if "error" in r]}
    for corpus in s["corpora"]:
        for mode in s["modes"]:
            per: dict = defaultdict(lambda: defaultdict(list))
            for r in recs:
                if r["corpus"] == corpus and r["mode"] == mode and "error" not in r:
                    per[r["family"]][r["project"]].append(r["cv_aucpr"])
            proj = {f: {p: float(np.mean(v)) for p, v in sorted(ps.items())} for f, ps in per.items()}
            means = {f: float(np.mean(list(v.values()))) for f, v in proj.items()}
            entry = {"mean_tuning_aucpr": means, "per_project": proj}
            if mode == "within":
                entry["chosen_family"] = max(means, key=means.get)
            sel[f"{corpus}_{mode}"] = entry
    sel["written"] = datetime.now(timezone.utc).isoformat()
    _dump(sel, out, "selection.json")
    for k, v in sel.items():
        if isinstance(v, dict) and "mean_tuning_aucpr" in v:
            print(k, {f: round(x, 3) for f, x in v["mean_tuning_aucpr"].items()}, v.get("chosen_family", ""))
    return sel


def phase_score(out: str, s: dict, jobs: int, shard: tuple[int, int] | None = None) -> None:
    if not os.path.exists(os.path.join(out, "selection.json")):
        raise SystemExit("no selection.json: the select phase must run before any test data is scored")
    tuned = [r for r in _read_jsonl(os.path.join(out, "tuned.jsonl")) if "error" not in r]
    path, ppath = os.path.join(out, "scores.jsonl"), os.path.join(out, "predictions.csv")
    done = {r["key"] for r in _read_jsonl(path)}
    if shard:
        tuned = score_shards(tuned, s, shard[1])[shard[0]]
    todo = [r for r in tuned if r["key"] not in done]
    todo.sort(key=lambda r: -_size_hint(r, s))
    if not os.path.exists(ppath):
        with open(ppath, "w", newline="") as fh:
            csv.writer(fh).writerow(["key", "test", "id", "label", "p", "raw"])
    tick = _progress("score", len(todo))

    def sink(res):
        rows, preds = res
        with open(ppath, "a", newline="") as fh:  # predictions first: a resumed run may repeat them, never lose them
            csv.writer(fh).writerows(preds)
        _append(path, rows)
        tick(f"{rows[0]['key']} AP={np.nanmean([r['aucpr'] for r in rows]):.3f}")
    run_pool(score_task, todo, s, jobs, sink)


def _mean_block(rows: list[dict]) -> dict:
    pred_keys = ["roc_auc", "aucpr", "aucpr_chance", "mcc", "brier_calibrated", "brier_other_map", "brier_uncalibrated"]
    rec_keys = sorted({k for r in rows for k in r["recall"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN column stays NaN
        return {
            "predictive": {k: float(np.nanmean([r[k] for r in rows])) for k in pred_keys},
            "recall_batch": {k: float(np.nanmean([r["recall"].get(k, math.nan) for r in rows])) for k in rec_keys},
            "observations": len(rows),
            "calibration_methods": dict(sorted(pd.Series([r["calibration"] for r in rows]).value_counts().items())),
        }


def phase_report(out: str, s: dict) -> dict:
    sel = json.load(open(os.path.join(out, "selection.json")))
    tuned_ok = {r["key"] for r in _read_jsonl(os.path.join(out, "tuned.jsonl")) if "error" not in r}
    rows = sorted(_read_jsonl(os.path.join(out, "scores.jsonl")), key=lambda r: (r["key"], r["test"]))
    missing = tuned_ok - {r["key"] for r in rows}
    if missing:
        raise SystemExit(f"scoring unfinished: {len(missing)} tasks missing; run the score phase again")
    preds = (pd.read_csv(os.path.join(out, "predictions.csv")).drop_duplicates(["key", "test", "id"])
             .sort_values(["key", "test", "id"], kind="mergesort"))
    rep = {"protocol": CONFIG, "pipeline_check": s["pipeline_check"], "tuning_errors": sel["tuning_errors"]}
    for corpus in s["corpora"]:
        for mode in s["modes"]:
            grp: dict = defaultdict(lambda: defaultdict(list))
            for r in rows:
                if r["corpus"] == corpus and r["mode"] == mode:
                    grp[r["project"]][r["family"]].append(r)
            per = {p: {f: _mean_block(v) for f, v in sorted(fs.items())} for p, fs in sorted(grp.items())}
            fams = sorted({f for fs in per.values() for f in fs})
            entry = {"per_project": per, "corpus_means": {
                f: {m: float(np.nanmean([per[p][f]["predictive"][m] for p in per if f in per[p]]))
                    for m in ("roc_auc", "aucpr", "mcc", "brier_calibrated")}
                | {m: float(np.nanmean([per[p][f]["recall_batch"][m] for p in per if f in per[p]]))
                   for m in ("system_default_lines", "system_default_artefacts", "random_lines", "smallest_first_lines")}
                for f in fams}}
            if mode == "within":
                fam = sel[f"{corpus}_within"]["chosen_family"]
                entry["family"] = fam
                entry["family_tests"] = family_tests({p: per[p][fam] for p in per if fam in per[p]})
                keys = {r["key"] for r in rows if r["corpus"] == corpus and r["mode"] == mode and r["family"] == fam}
                pf = preds[preds["key"].isin(keys)]
                entry["reliability_pooled"] = reliability(pf["p"].to_numpy(), pf["label"].to_numpy(), 10)
                entry["pooling_note"] = "pooled over every round's test predictions; descriptive only"
            else:
                entry["note"] = "cross-project: descriptive, outside the significance tests (section 3.10.4)"
            rep[f"{corpus}_{mode}"] = entry
    rep["finished"] = datetime.now(timezone.utc).isoformat()
    _dump(rep, out, "report.json")
    for k, v in rep.items():
        if isinstance(v, dict) and "family_tests" in v:
            print(k, v["family"], [(t["baseline"], t["budget"], t["key"], round(t["mean_diff"], 3),
                                    round(t["p_holm"], 4)) for t in v["family_tests"]])
    return rep


# ------------------------------------------------------------------ entry point
def _settings(check: bool, corpora: list[str], modes: list[str], families: list[str]) -> dict:
    base = CHECK if check else FULL
    projects = {"nasa": sorted(DATA["nasa"]), "promise": sorted(DATA["promise"])}
    if check:
        projects = {c: [p for p in CHECK_PROJECTS[c] if p in projects[c]] or projects[c][:2] for c in projects}
    return base | {"pipeline_check": check, "families": families, "corpora": corpora, "modes": modes,
                   "projects": {c: projects[c] for c in corpora}, "calibration_share": CALIB_SHARE,
                   "inner_folds": N_FOLDS, "threads_per_model": 1}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["init", "tune", "tune-units", "tune-merge", "select", "score", "score-merge",
                                      "report", "all"])
    ap.add_argument("--shard", help="i/N: run shard i of N (tune-units, score)")
    ap.add_argument("--inputs", nargs="+", default=[], help="score-merge: shard folders (glob patterns)")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--check", action="store_true", help="pipeline check: 1 trial, 1 x 3 folds, two projects per corpus")
    ap.add_argument("--corpora", nargs="+", default=["nasa", "promise"])
    ap.add_argument("--modes", nargs="+", default=["within", "cross"])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    a = ap.parse_args(argv)

    if not DATA:
        _init()
    out = a.out
    manifest = os.path.join(out, "manifest.json") if out else None
    if manifest and os.path.exists(manifest):
        s = json.load(open(manifest))["settings"]  # a resumed run keeps its own settings
    elif a.phase in ("init", "tune", "all"):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = out or os.path.join("runs", ("bench-check-" if a.check else "bench-") + stamp)
        os.makedirs(out, exist_ok=True)
        s = _settings(a.check, a.corpora, a.modes, a.families)
        bm = os.path.join(benchmarks.RAW, "benchmarks_manifest.json")
        _dump({"run_id": os.path.basename(out), "kind": "benchmarks", "pipeline_check": a.check,
               "protocol": CONFIG, "protocol_sha256": sha256(CONFIG) if os.path.exists(CONFIG) else None,
               "datasets": {"nasa_md5": benchmarks.NASA_MD5, "nasa_version": benchmarks.NASA_VERSION,
                            "benchmarks_manifest_sha256": sha256(bm) if os.path.exists(bm) else None},
               "code_commit": git_commit(), "python": platform.python_version(),
               "libraries": {lib: version(lib) for lib in LIBS}, "settings": s,
               "tasks": len(all_tasks(s)), "started": datetime.now(timezone.utc).isoformat()}, out, "manifest.json")
    else:
        raise SystemExit("--out must name an existing benchmark run for this phase")
    print("run:", out, flush=True)

    shard = _shard(a.shard)
    if a.phase == "tune-units":
        if not shard:
            raise SystemExit("tune-units needs --shard i/N")
        phase_tune_units(out, s, a.jobs, shard)
    if a.phase == "tune-merge":
        phase_tune_merge(out, s)
    if a.phase == "score-merge":
        phase_score_merge(out, a.inputs)
    if a.phase in ("tune", "all"):
        phase_tune(out, s, a.jobs)
    if a.phase in ("select", "all"):
        phase_select(out, s)
    if a.phase in ("score", "all"):
        phase_score(out, s, a.jobs, shard)
    if a.phase in ("report", "all"):
        phase_report(out, s)


if __name__ == "__main__":
    main()
