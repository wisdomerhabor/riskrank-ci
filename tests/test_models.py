import numpy as np
import pandas as pd
import pytest

from riskrank.models import fit_calibrator, fit_final, time_folds, tune

FEATS = ["la", "ld", "nf"]


def synthetic_commits(n=600, seed=0):
    """Unit-test fixture only. Never used as evidence (section 3.5.1)."""
    rng = np.random.default_rng(seed)
    la = rng.integers(0, 500, n)
    ld = rng.integers(0, 200, n)
    nf = rng.integers(1, 20, n)
    logit = -2 + 0.004 * la + 0.1 * nf
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "commit_id": [f"c{i:04d}" for i in range(n)],
            "when": pd.date_range("2010-01-01", periods=n, freq="D", tz="UTC"),
            "la": la, "ld": ld, "nf": nf, "label": y,
        }
    )


def test_time_folds_train_only_on_earlier_commits():
    base = synthetic_commits()
    for tr, va in time_folds(base):
        assert base["when"].iloc[tr].max() < base["when"].iloc[va].min()


@pytest.mark.parametrize("family", ["lr", "rf", "lgbm"])
def test_tune_and_fit_final_never_touch_test(family):
    d = synthetic_commits(800)
    base, calib = d.iloc[:500], d.iloc[500:650]
    test = d.iloc[650:].assign(la=10**9)  # absurd values: if fitting read these, stats would show it
    tuned = tune(base, FEATS, family, seconds=0.5, seed=1)
    mv = fit_final(base, calib, FEATS, tuned, seed=1)
    rec = mv.record()
    assert rec["schema"]["scale_mean"]["la"] < 10
    assert tuned.trials["weight"] >= 1
    p, raw = mv.score(test)
    assert ((p >= 0) & (p <= 1)).all()


def test_calibration_rule_by_partition_size():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=1500)
    y = rng.random(1500) < 1 / (1 + np.exp(-raw))
    assert fit_calibrator(raw, y).method == "isotonic"
    assert fit_calibrator(raw[:500], y[:500]).method == "platt"


def test_reversed_platt_slope_falls_back_to_isotonic():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=300)
    y = rng.random(300) < 1 / (1 + np.exp(raw))  # higher raw -> LOWER risk
    cal = fit_calibrator(raw, y)
    assert cal.platt_slope < 0 and cal.method == "isotonic"


def test_platt_preserves_order():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=400)
    y = rng.random(400) < 1 / (1 + np.exp(-raw))
    cal = fit_calibrator(raw, y)
    p = cal(np.sort(raw), "platt")
    assert np.all(np.diff(p) > 0)
