"""NFR12 check: recompute ApacheJIT's change metrics on a sample of commits and report the differences.

    python -m riskrank.nfr12 data/repos/apache/zookeeper apache/zookeeper --n 60 --seed 0

Prints, per metric, the share of commits reproduced exactly (relative tolerance 1e-3), the Spearman
correlation, and the median ratio published / recomputed. This is the measurement NFR12 requires; the
remaining differences are reported, not hidden.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from riskrank.jit_features import FEATURES, History

TOTAL = "data/raw/apachejit/dataset/apachejit_total.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("project")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    published = pd.read_csv(TOTAL)
    sample = published[published["project"] == a.project].sample(a.n, random_state=a.seed)
    h = History(a.repo)
    sample = sample[sample["commit_id"].isin(h.by_sha)]
    mine = pd.DataFrame([h.features(s) for s in sample["commit_id"]])
    pub = sample[FEATURES].reset_index(drop=True)
    print(f"{a.project}: {len(sample)} commits recomputed\n")
    print(f"{'metric':7} {'exact':>6} {'rho':>6} {'median ratio':>13}")
    for f in FEATURES:
        exact = np.isclose(pub[f], mine[f], rtol=1e-3, atol=1e-3).mean()
        rho = pub[f].corr(mine[f], method="spearman")
        m = mine[f] > 0
        ratio = float((pub.loc[m, f] / mine.loc[m, f]).median()) if m.any() else float("nan")
        print(f"{f:7} {exact:6.0%} {rho:6.2f} {ratio:13.3f}")


if __name__ == "__main__":
    main()
