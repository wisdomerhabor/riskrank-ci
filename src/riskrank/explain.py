"""Reason codes (section 3.8.3, FR7).

SHAP explains each base model's UNCALIBRATED output on its own scale: log-odds for logistic
regression and LightGBM, probability for the random forest. Calibration then maps that output to the
probability the dashboard shows, so the contributions add up to the raw output, not to the
calibrated probability, and are never presented as parts of it.

Background data is a k-means summary of the model's base data: weighted cluster centres, with any
cluster of fewer than five members merged into its nearest cluster, so no individual training record
is stored with a model version. Tree SHAP (interventional) is used for tree ensembles; for logistic
regression the exact linear form coef * (x - weighted mean) is used, which is what Linear SHAP gives
under an independent masker. The neural model uses Kernel SHAP over the expanded background with a
fixed sample count and seed: its contributions are approximate, but still add up to the raw output
(Kernel SHAP's efficiency constraint) and repeat exactly for the same input.

Two checks from section 3.8.3 are enforced on every call:
  1. additivity: base value + sum of contributions == raw output (to 1e-6), else ExplanationError;
  2. determinism: the same input and model version give identical reason codes (no randomness here).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from riskrank.models import SCALED, raw_score

K = 10
MIN_MEMBERS = 5
REPLICATES = 100  # weighted centres are expanded to ~100 rows for Tree SHAP, which takes no weights
TOLERANCE = 1e-6
KERNEL_SEED = 0
KERNEL_SAMPLES = 512


class ExplanationError(RuntimeError):
    """A missing or non-additive explanation; NFR3 treats it as a scoring failure."""


@dataclass
class Background:
    centres: pd.DataFrame
    weights: np.ndarray
    k_requested: int
    merged: int

    def expanded(self) -> pd.DataFrame:
        reps = np.maximum(1, np.round(self.weights * REPLICATES)).astype(int)
        return self.centres.loc[self.centres.index.repeat(reps)].reset_index(drop=True)

    def mean(self) -> pd.Series:
        return (self.centres.mul(self.weights, axis=0)).sum()


def make_background(x: pd.DataFrame, seed: int, k: int = K) -> Background:
    """Weighted k-means summary of model-ready base features."""
    k = min(k, len(x))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x.to_numpy())
    labels = km.labels_.copy()
    merged = 0
    while True:
        counts = np.bincount(labels, minlength=labels.max() + 1)
        live = [c for c in range(len(counts)) if counts[c] > 0]
        small = [c for c in live if counts[c] < MIN_MEMBERS]
        if not small or len(live) == 1:
            break
        c = small[0]
        centres = {j: x.to_numpy()[labels == j].mean(axis=0) for j in live}
        others = [j for j in live if j != c]
        nearest = min(others, key=lambda j: np.linalg.norm(centres[j] - centres[c]))
        labels[labels == c] = nearest
        merged += 1
    live = sorted(set(labels))
    centres = pd.DataFrame([x.to_numpy()[labels == j].mean(axis=0) for j in live], columns=x.columns)
    weights = np.array([(labels == j).sum() for j in live], dtype=float)
    return Background(centres, weights / weights.sum(), k, merged)


class Explainer:
    def __init__(self, mv, background: Background):
        self.mv = mv
        self.background = background
        self.family = mv.tuned.family
        model = mv.model
        self._kernel = None
        if isinstance(model, LogisticRegression):
            self._mean = background.mean()
            self._coef = pd.Series(model.coef_[0], index=background.centres.columns)
            self._base = float(model.intercept_[0] + (self._coef * self._mean).sum())
            self._tree = None
        elif isinstance(model, MLPClassifier):
            cols = list(background.centres.columns)
            self._tree = None
            self._kernel = shap.KernelExplainer(
                lambda a: raw_score(model, pd.DataFrame(a, columns=cols)), background.expanded().to_numpy())
        else:
            out = "raw"
            self._tree = shap.TreeExplainer(model, data=background.expanded(), model_output=out,
                                            feature_perturbation="interventional")

    def contributions(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(base values, contributions) on the model's raw scale, for model-ready rows `x`."""
        if self._kernel is not None:
            np.random.seed(KERNEL_SEED)  # Kernel SHAP samples coalitions with numpy's global generator
            contrib = np.asarray(self._kernel.shap_values(x.to_numpy(), nsamples=KERNEL_SAMPLES, silent=True))
            base = np.full(len(x), float(self._kernel.expected_value))
        elif self._tree is None:
            contrib = ((x - self._mean) * self._coef).to_numpy()
            base = np.full(len(x), self._base)
        else:
            sv = self._tree(x)
            values, base = sv.values, np.asarray(sv.base_values)
            if values.ndim == 3:  # sklearn classifier: one slice per class -> take the defect class
                values, base = values[:, :, 1], base[:, 1]
            contrib, base = values, np.broadcast_to(base, (len(x),)).astype(float)
        raw = raw_score(self.mv.model, x)
        gap = np.abs(base + contrib.sum(axis=1) - raw)
        if not np.all(gap <= TOLERANCE * np.maximum(1.0, np.abs(raw))):
            raise ExplanationError(f"additivity failed: max gap {gap.max():.3g}")
        return base, contrib

    def reason_codes(self, frame: pd.DataFrame) -> list[list[dict]]:
        """Ranked reason codes per row: every kept feature, largest |contribution| first."""
        x, _ = self.mv.pipeline.transform(frame, scale=self.family in SCALED)
        base, contrib = self.contributions(x)
        names = list(x.columns)
        raw_values = frame[names].to_numpy(dtype=float)
        out = []
        for i in range(len(x)):
            order = np.argsort(-np.abs(contrib[i]), kind="mergesort")
            out.append([{"feature": names[j], "value": float(raw_values[i, j]),
                         "contribution": round(float(contrib[i, j]), 6),
                         "direction": "raised" if contrib[i, j] > 0 else "lowered" if contrib[i, j] < 0 else "none"}
                        for j in order])
        return out

    def record(self) -> dict:
        attribution = ("Kernel SHAP (approximate)" if self._kernel is not None
                       else "linear (exact)" if self._tree is None else "Tree SHAP, interventional")
        return {"attribution": attribution,
                "kernel": {"seed": KERNEL_SEED, "samples": KERNEL_SAMPLES} if self._kernel is not None else None,
                "scale": "log-odds" if self.family in ("lr", "lgbm") else "probability",
                "background": {"k_requested": self.background.k_requested,
                               "clusters": len(self.background.centres),
                               "merged_small_clusters": self.background.merged,
                               "min_members": MIN_MEMBERS, "tree_replicates": REPLICATES},
                "shap_version": shap.__version__}
