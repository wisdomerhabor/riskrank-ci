"""Model development, imbalance handling and calibration (section 3.8.2).

Candidate families: logistic regression (baseline), random forest and LightGBM (the tree family), and
a compact neural model (scikit-learn multi-layer perceptron, one or two small hidden layers with early
stopping), which section 3.8.2 evaluates offline in every experiment.

For one corpus slice (for example one ApacheJIT project):

1. tune()      For each family and each imbalance method (class weighting, SMOTE), random search
               over hyperparameters for the same fixed number of seconds. Every trial is scored
               on time-ordered folds of the BASE data only: each fold trains on earlier commits and
               validates on later ones, and a fresh FeaturePipeline is fitted inside every fold.
               Hyperparameters are chosen on mean AUC-PR. SMOTE is kept only if it raises mean
               defect recall at the 20 percent lines budget (default key) over class weighting.
2. fit_final() Fit the FeaturePipeline and the chosen model on all base data. Fit BOTH a Platt map
               and an isotonic map on the calibration partition, applied to the model's raw output.
               Use isotonic when the calibration partition holds at least 1,000 artefacts, Platt
               otherwise, and isotonic if Platt's slope ties or reverses the score order. Fix the
               MCC threshold on the calibration partition.

Nothing here reads a test partition.

Implementation choices not fixed by Chapter Three (recorded with every model version):
    folds = 4 expanding windows; search = seeded random search; seconds per (family, imbalance)
    = `seconds`; trials completed is recorded because a time budget makes the count machine-
    dependent, so the tuned hyperparameters (not the search) are what make a run reproducible.

Benchmark corpora (NASA MDP, PROMISE; config/benchmarks.yaml): no dates exist, so tuning folds are
stratified 4-fold (`folds="stratified"`), and every candidate gets the same NUMBER of trials
(`trials=`) instead of the same seconds, so the search itself is reproducible from the seed. Those
runs set N_JOBS = 1 so a model's result never depends on how many processes run beside it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

from riskrank.budget import rank, recall_lines
from riskrank.features import FeaturePipeline

FAMILIES = ("lr", "rf", "lgbm", "mlp")
SCALED = {"lr", "mlp"}  # tree ensembles use unscaled features (section 3.5.2)
N_FOLDS = 4
ISOTONIC_MIN = 1000
N_JOBS = -1  # threads per random forest / LightGBM model; benchmark runs set 1


# ---------------------------------------------------------------- search spaces
def sample_params(family: str, rng: np.random.Generator) -> dict:
    if family == "lr":
        return {"C": float(10 ** rng.uniform(-3, 2))}
    if family == "rf":
        return {
            "n_estimators": int(rng.choice([100, 200, 400])),
            "max_depth": [None, 6, 10, 16][int(rng.integers(4))],
            "min_samples_leaf": int(rng.choice([1, 2, 5, 10, 20])),
            "max_features": ["sqrt", 0.5, 0.8][int(rng.integers(3))],
        }
    if family == "lgbm":
        return {
            "n_estimators": int(rng.choice([100, 200, 400, 800])),
            "learning_rate": float(10 ** rng.uniform(-2.3, -0.7)),
            "num_leaves": int(rng.choice([7, 15, 31, 63])),
            "min_child_samples": int(rng.choice([5, 10, 20, 50])),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "subsample_freq": 1,
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            "reg_lambda": float(10 ** rng.uniform(-3, 1)),
        }
    if family == "mlp":
        return {
            "hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)][int(rng.integers(5))],
            "alpha": float(10 ** rng.uniform(-5, -1)),
            "learning_rate_init": float(10 ** rng.uniform(-4, -2)),
            "batch_size": int(rng.choice([64, 128, 256])),
        }
    raise ValueError(family)


def make(family: str, params: dict, imbalance: str, seed: int):
    weighted = imbalance == "weight"
    if family == "lr":
        return LogisticRegression(
            **params, class_weight="balanced" if weighted else None, max_iter=5000
        )
    if family == "rf":
        return RandomForestClassifier(
            **params,
            class_weight="balanced_subsample" if weighted else None,
            n_jobs=N_JOBS,
            random_state=seed,
        )
    if family == "lgbm":
        return LGBMClassifier(
            **params, is_unbalance=weighted, random_state=seed, n_jobs=N_JOBS, verbose=-1
        )
    if family == "mlp":  # class weighting is applied as balanced sample weights in _fit_one
        return MLPClassifier(**params, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                             max_iter=200, random_state=seed)
    raise ValueError(family)


def raw_score(model, x: pd.DataFrame) -> np.ndarray:
    """The base model's uncalibrated output, on the scale SHAP will explain (section 3.8.3)."""
    if isinstance(model, LogisticRegression):
        return model.decision_function(x)  # log-odds
    if isinstance(model, LGBMClassifier):
        return model.predict(x, raw_score=True)  # log-odds
    return model.predict_proba(x)[:, 1]  # random forest and neural model: probability


# ------------------------------------------------------------------ the folds
def time_folds(base: pd.DataFrame, n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over base data sorted by time: train on chunks 0..i, validate on i+1."""
    order = np.argsort(base["when"].to_numpy(), kind="mergesort")
    chunks = np.array_split(order, n_folds + 1)
    folds = []
    for i in range(n_folds):
        tr, va = np.concatenate(chunks[: i + 1]), chunks[i + 1]
        y_tr, y_va = base["label"].to_numpy()[tr], base["label"].to_numpy()[va]
        if y_tr.min() != y_tr.max() and y_va.any():  # both classes to train, a defect to find
            folds.append((tr, va))
    return folds


def stratified_folds(base: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffled stratified folds, for corpora with no dates: every fold keeps the defective share."""
    y = base["label"].to_numpy().astype(int)
    if min(y.sum(), len(y) - y.sum()) < n_folds:
        raise ValueError(f"fewer than {n_folds} examples of a class for stratified tuning folds")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def _fit_one(family, params, imbalance, seed, tr_frame, features):
    pipe = FeaturePipeline(features).fit(tr_frame)
    x, _ = pipe.transform(tr_frame, scale=family in SCALED)
    y = tr_frame["label"].to_numpy()
    if imbalance == "smote":
        minority = int(min(y.sum(), len(y) - y.sum()))
        if minority < 2:
            raise ValueError("too few minority examples for SMOTE")
        x, y = SMOTE(k_neighbors=min(5, minority - 1), random_state=seed).fit_resample(x, y)
    model = make(family, params, imbalance, seed)
    if family == "mlp" and imbalance == "weight":
        model.fit(x, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(x, y)
    return pipe, model


def effort_size(frame: pd.DataFrame) -> np.ndarray:
    """Size for the lines budget: churn (la + ld) for commits, lines of code for modules."""
    if "la" in frame and "ld" in frame:
        return (frame["la"] + frame["ld"]).to_numpy()
    return frame["size"].to_numpy()


def ident(frame: pd.DataFrame) -> np.ndarray:
    return (frame["commit_id"] if "commit_id" in frame else frame["id"]).to_numpy()


def cv_score(family, params, imbalance, seed, base, features, folds) -> tuple[float, float]:
    """Mean AUC-PR and mean recall at the 20% lines budget over the time folds."""
    aps, recalls = [], []
    for tr, va in folds:
        pipe, model = _fit_one(family, params, imbalance, seed, base.iloc[tr], features)
        va_frame = base.iloc[va]
        x_va, _ = pipe.transform(va_frame, scale=family in SCALED)
        s = raw_score(model, x_va)
        aps.append(average_precision_score(va_frame["label"], s))
        ranked = rank(
            pd.DataFrame(
                {"id": ident(va_frame), "p": s, "raw": s,
                 "size": effort_size(va_frame),
                 "label": va_frame["label"].to_numpy()}
            ),
            "default",
        )
        recalls.append(recall_lines(ranked))
    return float(np.mean(aps)), float(np.nanmean(recalls))


# ------------------------------------------------------------------- tuning
@dataclass
class Tuned:
    family: str
    imbalance: str
    params: dict
    cv_aucpr: float
    cv_recall_lines: float
    trials: dict = field(default_factory=dict)  # trials completed per imbalance method
    weight_vs_smote: dict = field(default_factory=dict)
    budget: dict = field(default_factory=dict)  # {"seconds": s} or {"trials": n} per imbalance method
    folds: str = "time"  # "time" (expanding windows) or "stratified"


def tune(base, features, family, seconds: float | None = None, seed: int = 42, *,
         trials: int | None = None, folds: str = "time") -> Tuned:
    """Random search per imbalance method, under EITHER a seconds budget or a fixed trial count."""
    if (seconds is None) == (trials is None):
        raise ValueError("give exactly one of seconds= or trials=")
    if trials is not None and trials < 1:
        raise ValueError("trials must be at least 1")
    kind = folds
    if kind == "time":
        folds = time_folds(base)
    elif kind == "stratified":
        folds = stratified_folds(base, seed=seed)
    else:
        raise ValueError(f"unknown fold kind {kind!r}")
    if not folds:
        raise ValueError("no usable time folds in the base data")
    best: dict[str, tuple] = {}
    done: dict[str, int] = {}
    for imbalance in ("weight", "smote"):
        rng = np.random.default_rng(seed)
        deadline, n, top = time.monotonic() + (seconds or 0), 0, None
        while (n < trials) if trials is not None else (n == 0 or time.monotonic() < deadline):
            params = sample_params(family, rng)
            try:
                ap, rec = cv_score(family, params, imbalance, seed, base, features, folds)
            except ValueError:
                break  # SMOTE impossible on this slice; class weighting stands
            n += 1
            if top is None or ap > top[1]:
                top = (params, ap, rec)
        done[imbalance] = n
        if top is not None:
            best[imbalance] = top
    w = best["weight"]
    s = best.get("smote")
    use_smote = s is not None and s[2] > w[2]
    chosen, pick = ("smote", s) if use_smote else ("weight", w)
    return Tuned(
        family=family,
        imbalance=chosen,
        params=pick[0],
        cv_aucpr=pick[1],
        cv_recall_lines=pick[2],
        trials=done,
        weight_vs_smote={
            k: {"cv_aucpr": round(v[1], 4), "cv_recall_lines": round(v[2], 4)} for k, v in best.items()
        },
        budget={"trials": trials} if trials is not None else {"seconds": seconds},
        folds=kind,
    )


# ---------------------------------------------------------------- calibration
@dataclass
class Calibrator:
    method: str  # "platt" | "isotonic"
    platt_slope: float
    platt: LogisticRegression
    isotonic: IsotonicRegression

    def __call__(self, raw: np.ndarray, method: str | None = None) -> np.ndarray:
        m = method or self.method
        if m == "platt":
            return self.platt.predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]
        return self.isotonic.predict(np.asarray(raw))


def fit_calibrator(raw: np.ndarray, y: np.ndarray) -> Calibrator:
    platt = LogisticRegression(C=1e6, max_iter=5000).fit(np.asarray(raw).reshape(-1, 1), y)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
    slope = float(platt.coef_[0, 0])
    method = "isotonic" if len(y) >= ISOTONIC_MIN or slope <= 0 else "platt"
    return Calibrator(method=method, platt_slope=slope, platt=platt, isotonic=iso)


def mcc_threshold(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """The calibrated-probability threshold that maximises MCC on the calibration partition."""
    candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    scores = [matthews_corrcoef(y, p >= t) for t in candidates]
    i = int(np.argmax(scores))
    return float(candidates[i]), float(scores[i])


# ---------------------------------------------------------------- final model
@dataclass
class ModelVersion:
    tuned: Tuned
    pipeline: FeaturePipeline
    model: object
    calibrator: Calibrator
    threshold: float
    calib_mcc: float
    seed: int
    n_base: int
    n_calib: int

    def score(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (calibrated probability, raw score) for each row."""
        x, _ = self.pipeline.transform(frame, scale=self.tuned.family in SCALED)
        raw = raw_score(self.model, x)
        return self.calibrator(raw), raw

    def record(self) -> dict:
        return {
            "family": self.tuned.family,
            "imbalance": self.tuned.imbalance,
            "params": self.tuned.params,
            "tuning": {
                "cv_aucpr": self.tuned.cv_aucpr,
                "cv_recall_lines": self.tuned.cv_recall_lines,
                "trials": self.tuned.trials,
                "weight_vs_smote": self.tuned.weight_vs_smote,
                "folds": N_FOLDS,
                "fold_kind": self.tuned.folds,
                "budget_per_imbalance": self.tuned.budget,
                "search": "seeded random search, equal "
                          + ("trials" if "trials" in self.tuned.budget else "seconds")
                          + " per family and imbalance method",
            },
            "calibration": {
                "method": self.calibrator.method,
                "platt_slope": self.calibrator.platt_slope,
                "n_calib": self.n_calib,
                "rule": f"isotonic if n_calib >= {ISOTONIC_MIN} or Platt slope <= 0",
            },
            "mcc_threshold": self.threshold,
            "calib_mcc": self.calib_mcc,
            "seed": self.seed,
            "n_base": self.n_base,
            "schema": self.pipeline.schema,
        }


def fit_final(base, calib, features, tuned: Tuned, seed: int) -> ModelVersion:
    pipe, model = _fit_one(tuned.family, tuned.params, tuned.imbalance, seed, base, features)
    x_cal, _ = pipe.transform(calib, scale=tuned.family in SCALED)
    raw = raw_score(model, x_cal)
    y = calib["label"].to_numpy()
    cal = fit_calibrator(raw, y)
    thr, mcc = mcc_threshold(cal(raw), y)
    return ModelVersion(tuned, pipe, model, cal, thr, mcc, seed, len(base), len(calib))
