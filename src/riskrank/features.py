"""Feature preparation (Chapter Three, section 3.5.2).

A FeaturePipeline is fitted on one model's base-model data only, then applied unchanged to that
model's calibration and test partitions. Inside hyperparameter tuning, a fresh pipeline is fitted on
each tuning fold's training data. Nothing here ever looks at a calibration or test partition while
fitting.

Steps, in order:
1. Impute: a value that cannot be calculated (NaN) is replaced with that feature's base-data median;
   the replacement is counted so the corpus report can state it.
2. Correlation filter: within each feature family (static or process), where two features have a
   Spearman correlation of at least 0.8 in absolute value, the one listed later in the feature table
   is removed. Kept features are fixed at fit time.
3. Transform: log(1 + x) on every kept feature (Kamei et al., 2013). Extreme values are not clipped.
4. Scale (optional): standardise with the base-data mean and standard deviation. Used for logistic
   regression and the neural model; tree ensembles use the unscaled step-3 output.

The fitted pipeline exposes `schema`, which is stored with every model version. The scoring service
rejects any feature set whose names or order differ from it.

Interpretation to confirm: section 3.5.2 says "a Spearman correlation of at least 0.8". This module
reads that as |rho| >= 0.8, so a strong negative correlation also removes the later feature. On the
ApacheJIT base data no pair reaches -0.8, so the choice changes nothing there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLD = 0.8
STATIC = {"size", "complexity", "touched_size", "touched_complexity"}  # static code metrics (Table 3.7)


class SchemaMismatch(ValueError):
    """Raised when a feature set does not match the fitted schema."""


class FeaturePipeline:
    def __init__(self, features: list[str], families: dict[str, str] | None = None):
        """`features` in feature-table order; `families` maps feature -> 'static' | 'process'.

        Without `families`, size and complexity measures are static and everything else is a process
        metric. All 12 ApacheJIT change metrics are therefore one (process) family.
        """
        self.features = list(features)
        self.families = families or {f: "static" if f in STATIC else "process" for f in features}
        missing = set(self.features) - set(self.families)
        if missing:
            raise ValueError(f"no family given for {sorted(missing)}")

    # ------------------------------------------------------------------ fitting
    def fit(self, base: pd.DataFrame) -> "FeaturePipeline":
        x = self._select(base)
        self.medians_ = x.median()
        x = x.fillna(self.medians_)
        self.kept_, self.dropped_ = self._correlation_filter(x)
        logged = np.log1p(x[self.kept_])
        self.mean_ = logged.mean()
        self.std_ = logged.std(ddof=0).replace(0, 1.0)  # a constant column is left unscaled
        return self

    def _correlation_filter(self, x: pd.DataFrame) -> tuple[list[str], list[tuple[str, str, float]]]:
        rho = x.corr(method="spearman")
        kept: list[str] = []
        dropped: list[tuple[str, str, float]] = []
        for f in self.features:  # feature-table order: earlier features win
            clash = [
                k
                for k in kept
                if self.families[k] == self.families[f] and abs(rho.loc[k, f]) >= THRESHOLD
            ]
            if clash:
                dropped.append((f, clash[0], round(float(rho.loc[clash[0], f]), 3)))
            else:
                kept.append(f)
        return kept, dropped

    # ---------------------------------------------------------------- applying
    def transform(self, frame: pd.DataFrame, scale: bool) -> tuple[pd.DataFrame, int]:
        """Return the model-ready features and the number of imputed values."""
        x = self._select(frame)
        imputed = int(x[self.kept_].isna().sum().sum())
        x = np.log1p(x[self.kept_].fillna(self.medians_[self.kept_]))
        if scale:
            x = (x - self.mean_) / self.std_
        return x, imputed

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        absent = [f for f in self.features if f not in frame.columns]
        if absent:
            raise SchemaMismatch(f"missing features: {absent}")
        x = frame[self.features].astype(float)
        if (x < 0).any().any():
            raise ValueError("negative feature value: log(1 + x) is undefined below -1")
        return x

    # ------------------------------------------------------------------ schema
    @property
    def schema(self) -> dict:
        return {
            "features_in": self.features,
            "features_kept": self.kept_,
            "dropped": [{"feature": f, "because_of": k, "rho": r} for f, k, r in self.dropped_],
            "transform": "log1p",
            "impute": "base-data median",
            "medians": self.medians_[self.kept_].round(6).to_dict(),
            "scale_mean": self.mean_.round(6).to_dict(),
            "scale_std": self.std_.round(6).to_dict(),
            "correlation_threshold": THRESHOLD,
        }

    def check(self, names: list[str]) -> None:
        """Reject a feature set whose names or order differ from the fitted input schema."""
        if list(names) != self.features:
            raise SchemaMismatch(f"expected {self.features}, got {list(names)}")
