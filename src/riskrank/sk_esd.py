"""Scott-Knott ESD test for ranking model families (section 3.10.4; Tantithamthavorn et al., 2017).

    python -m riskrank.sk_esd runs/within-20261108T224535Z/evaluation.json [--measure aucpr] [--out FILE]

Input: one score per family per project (e.g. held-out AUC-PR across the 13 ApacheJIT projects).
Step 1, Scott-Knott (Scott & Knott, 1974): sort the families by mean score, find the split of the ordered
means that maximises the between-group sum of squares B0, and test it with the likelihood-ratio statistic
    lambda = pi / (2 (pi - 2)) * B0 / sigma0^2,   sigma0^2 = (sum (m_t - m)^2 + v * s2) / (k + v)
against a chi-square with k / (pi - 2) degrees of freedom (alpha 0.05), where m_t are the k family means,
s2 = MSE / r is the variance of a mean (r projects per family) and v the residual degrees of freedom of the
families-by-projects ANOVA. Significant splits are split again, recursively.
Step 2, effect-size correction (the "ESD" part): adjacent groups whose difference has a negligible
Cohen's d (|d| < 0.2, pooled standard deviation) are merged.
Rank 1 is the best group. The output is descriptive about families; it is never a claim about ordering.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy import stats

ALPHA = 0.05
NEGLIGIBLE_D = 0.2


def _residual(scores: dict[str, list[float]]) -> tuple[float, int]:
    """MSE and residual df of the two-way (family x project) ANOVA without interaction."""
    names = list(scores)
    y = np.array([scores[n] for n in names], dtype=float)  # k x r
    k, r = y.shape
    resid = y - y.mean(axis=1, keepdims=True) - y.mean(axis=0, keepdims=True) + y.mean()
    v = (k - 1) * (r - 1)
    return (float((resid ** 2).sum() / v) if v else 0.0), v


def _split(means: list[float], s2: float, v: int) -> int | None:
    """Index of the best significant split of the ordered means, or None."""
    k = len(means)
    if k < 2:
        return None
    m = float(np.mean(means))
    best, b0 = None, -1.0
    for i in range(1, k):
        a, b = means[:i], means[i:]
        bss = len(a) * (np.mean(a) - m) ** 2 + len(b) * (np.mean(b) - m) ** 2
        if bss > b0:
            best, b0 = i, bss
    sigma0 = (sum((x - m) ** 2 for x in means) + v * s2) / (k + v)
    if sigma0 <= 0:
        return best if b0 > 0 else None
    lam = math.pi / (2 * (math.pi - 2)) * b0 / sigma0
    p = float(stats.chi2.sf(lam, k / (math.pi - 2)))
    return best if p < ALPHA else None


def cohen_d(a: list[float], b: list[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)) if na + nb > 2 else 0.0
    return float((a.mean() - b.mean()) / sp) if sp > 0 else (math.inf if a.mean() != b.mean() else 0.0)


def sk_esd(scores: dict[str, list[float]]) -> dict:
    """scores: family -> per-project scores (same project order for every family). Higher is better."""
    if len({len(v) for v in scores.values()}) != 1:
        raise ValueError("every family needs one score per project")
    order = sorted(scores, key=lambda n: -float(np.mean(scores[n])))
    mse, v = _residual(scores)
    r = len(next(iter(scores.values())))
    s2 = mse / r

    def recurse(names: list[str]) -> list[list[str]]:
        i = _split([float(np.mean(scores[n])) for n in names], s2, v)
        return [names] if i is None else recurse(names[:i]) + recurse(names[i:])

    groups = recurse(order)
    merged = [groups[0]]
    for g in groups[1:]:
        prev = merged[-1]
        d = cohen_d([x for n in prev for x in scores[n]], [x for n in g for x in scores[n]])
        if abs(d) < NEGLIGIBLE_D:
            merged[-1] = prev + g
        else:
            merged.append(g)
    return {"method": "Scott-Knott ESD (alpha 0.05; adjacent groups merged when |Cohen's d| < 0.2)",
            "projects": r, "families": len(order),
            "means": {n: float(np.mean(scores[n])) for n in order},
            "scott_knott_groups": groups,
            "ranks": {n: i + 1 for i, g in enumerate(merged) for n in g}}


def benchmark_scores(rows: list[dict], corpus: str, measure: str = "aucpr", mode: str = "within") -> dict[str, list[float]]:
    """Per family, one within-project score per project: the mean over that project's test units
    (NASA: the 100 cross-validation rounds; PROMISE: its release pairs). Projects are the unit (section 3.10.4)."""
    acc: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        if r["corpus"] == corpus and r["mode"] == mode and r.get(measure) is not None:
            acc.setdefault(r["family"], {}).setdefault(r["project"], []).append(float(r[measure]))
    projects = sorted(set.intersection(*(set(p) for p in acc.values())))
    return {f: [float(np.mean(acc[f][p])) for p in projects] for f in sorted(acc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("evaluation", help="a within-project evaluation.json, or a benchmark scores.jsonl with --corpus")
    ap.add_argument("--corpus", choices=["nasa", "promise"])
    ap.add_argument("--measure", default="aucpr")
    ap.add_argument("--mode", choices=["within", "cross"], default="within",
                    help="benchmarks only; cross-project results are descriptive (section 3.10.4)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.corpus:
        scores = benchmark_scores([json.loads(l) for l in open(a.evaluation)], a.corpus, a.measure, a.mode)
    else:
        fams = json.load(open(a.evaluation))["per_family"]
        projects = sorted(next(iter(fams.values()))["per_project"])
        scores = {f: [float(d["per_project"][p]["predictive"][a.measure]) for p in projects] for f, d in fams.items()}
    out = {"evaluation": a.evaluation, "corpus": a.corpus, "mode": a.mode, "measure": a.measure} | sk_esd(scores)
    text = json.dumps(out, indent=1)
    if a.out:
        open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
