"""Ranking keys, testing budgets and offline defect recall (sections 3.8.1 and 3.10.2, FR9).

Ranking keys
    default         calibrated risk p, highest first
    effort          p' / s, highest first, where p' = p(1 - z) + z with z = 0.05 and s is size
                    (lines of code for modules, churn = lines added + deleted for commits).
                    Zero-size artefacts cannot be ranked by this key; they go last and are never
                    marked.
Ties under either key: uncalibrated score (highest first), then ascending artefact identifier.

Budgets (b = 0.2 by default)
    artefacts       mark the top ceil(b * n) artefacts, at least one when n >= 1.
    lines           mark from the top until cumulative size reaches b * total size.
                    Runtime and the bounded checks keep the boundary artefact whole.
                    Offline recall counts only the fraction of a defective boundary artefact's
                    lines that falls inside the limit (the 1,000-line example in section 3.10.2).
                    Zero-size artefacts are never marked under this budget.

Recall = defective artefacts found / all defective artefacts in the ranked set, including defective
zero-size artefacts (which count as not found whenever they cannot be marked).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

Z = 0.05
BUDGET = 0.2


def rank(
    frame: pd.DataFrame,
    key: str,
    p: str = "p",
    raw: str = "raw",
    size: str = "size",
    ident: str = "id",
    z: float = Z,
) -> pd.DataFrame:
    """Return `frame` sorted by the ranking key, with the tie-break of FR9 applied."""
    f = frame.copy()
    if key == "default":
        f["_key"] = f[p]
        f["_rankable"] = True
    elif key == "effort":
        f["_rankable"] = f[size] > 0
        lower_bounded = f[p] * (1 - z) + z
        f["_key"] = np.where(f["_rankable"], lower_bounded / f[size].where(f[size] > 0, 1), -np.inf)
    else:
        raise ValueError(f"unknown ranking key {key!r}")
    f = f.sort_values(
        ["_rankable", "_key", raw, ident], ascending=[False, False, False, True], kind="mergesort"
    )
    return f.drop(columns=["_key"]).reset_index(drop=True)


def recall_artefacts(ranked: pd.DataFrame, label: str = "label", b: float = BUDGET) -> float:
    """Defect recall when the top ceil(b * n) artefacts are marked."""
    n, defects = len(ranked), int(ranked[label].sum())
    if defects == 0:
        return math.nan
    k = max(1, math.ceil(b * n)) if n else 0
    marked = ranked.head(k)
    if "_rankable" in marked:
        marked = marked[marked["_rankable"]]
    return float(marked[label].sum()) / defects


def recall_lines(
    ranked: pd.DataFrame, label: str = "label", size: str = "size", b: float = BUDGET
) -> float:
    """Offline defect recall at a lines budget, counting only the in-limit part of the boundary."""
    defects = int(ranked[label].sum())
    if defects == 0:
        return math.nan
    s = ranked[size].to_numpy(dtype=float)
    y = ranked[label].to_numpy(dtype=bool)
    limit = b * s.sum()
    found, used = 0.0, 0.0
    for size_i, defective in zip(s, y):
        if size_i <= 0:
            continue  # zero-size artefacts are never marked under a lines budget
        if used >= limit:
            break
        share = min(1.0, (limit - used) / size_i)
        if defective:
            found += share
        used += size_i
    return found / defects


def marked_lines_runtime(ranked: pd.DataFrame, size: str = "size", b: float = BUDGET) -> pd.Series:
    """Runtime marking under a lines budget: the boundary artefact is kept whole."""
    s = ranked[size].to_numpy(dtype=float)
    limit = b * s.sum()
    marked, used = [], 0.0
    for size_i in s:
        take = size_i > 0 and used < limit
        marked.append(bool(take))
        if take:
            used += size_i
    return pd.Series(marked, index=ranked.index)


def mark(ranked: pd.DataFrame, unit: str, b: float, key: str, size: str = "size") -> pd.Series:
    """Runtime budget marking of a ranked frame (FR9), shared by the CI hook and the dashboard preview.

    artefacts: the top ceil(b * n), at least one; lines: from the top, boundary artefact kept whole.
    Zero-size artefacts are never marked under the lines budget or the effort key.
    """
    if not len(ranked):
        return pd.Series([], dtype=bool, index=ranked.index)
    if unit == "artefacts":
        k = max(1, math.ceil(b * len(ranked)))
        marked = pd.Series([i < k for i in range(len(ranked))], index=ranked.index)
    elif unit == "lines":
        marked = marked_lines_runtime(ranked, size=size, b=b)
    else:
        raise ValueError(f"unknown budget unit {unit!r}")
    if key == "effort" or unit == "lines":
        marked &= ranked[size] > 0
    return marked


def realised_share(ranked: pd.DataFrame, marked: pd.Series, unit: str, size: str = "size") -> float | None:
    """Share of the ranked set actually marked: of artefacts, or of their lines."""
    if not len(ranked):
        return None
    if unit == "artefacts":
        return float(marked.mean())
    total = ranked[size].sum()
    return float(ranked.loc[marked, size].sum() / total) if total else None


def random_recall(
    frame: pd.DataFrame,
    budget: str,
    label: str = "label",
    size: str = "size",
    b: float = BUDGET,
    draws: int = 100,
    seed: int = 0,
) -> float:
    """Random-ordering baseline: exact k/n for artefacts, mean of `draws` orders for lines."""
    n = len(frame)
    if int(frame[label].sum()) == 0:
        return math.nan
    if budget == "artefacts":
        return max(1, math.ceil(b * n)) / n
    rng = np.random.default_rng(seed)
    vals = [recall_lines(frame.iloc[rng.permutation(n)], label, size, b) for _ in range(draws)]
    return float(np.mean(vals))
