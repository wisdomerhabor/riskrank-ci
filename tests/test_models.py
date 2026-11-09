import numpy as np
import pandas as pd
import pytest

from riskrank import models
from riskrank.models import fit_calibrator, fit_final, stratified_folds, time_folds, tune

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


@pytest.mark.parametrize("family", ["lr", "rf", "lgbm", "mlp"])
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


def test_stratified_folds_keep_the_defective_share():
    base = synthetic_commits(400)
    share = base["label"].mean()
    folds = stratified_folds(base, seed=3)
    assert len(folds) == 4
    for tr, va in folds:
        assert abs(base["label"].iloc[va].mean() - share) < 0.02
        assert not set(tr) & set(va)


@pytest.mark.parametrize("family", ["lr", "rf"])
def test_trial_budget_is_exact_and_reproducible(family, monkeypatch):
    monkeypatch.setattr(models, "N_JOBS", 1)
    base = synthetic_commits(300)
    a = tune(base, FEATS, family, seed=7, trials=3, folds="stratified")
    b = tune(base, FEATS, family, seed=7, trials=3, folds="stratified")
    assert a.trials == {"weight": 3, "smote": 3}
    assert (a.params, a.imbalance, a.cv_aucpr) == (b.params, b.imbalance, b.cv_aucpr)
    assert a.budget == {"trials": 3} and a.folds == "stratified"


def test_tune_needs_exactly_one_budget():
    base = synthetic_commits(200)
    with pytest.raises(ValueError):
        tune(base, FEATS, "lr", seed=1)
    with pytest.raises(ValueError):
        tune(base, FEATS, "lr", 1.0, 1, trials=2)


def test_time_folds_skip_a_fold_with_one_defect_to_train_on():
    # window_1y, apache/groovy (9 Nov 2026): a training chunk with ONE defect broke the neural model's
    # stratified early-stopping split, and tune() hid the error as "SMOTE impossible".
    base = synthetic_commits(500).assign(label=False)
    base.loc[[10, 300, 350, 420, 480], "label"] = True  # chunk 0 holds one defect, later chunks more
    for tr, _ in time_folds(base):
        assert base["label"].iloc[tr].sum() >= 2
    tuned = tune(base, FEATS, "mlp", trials=1, seed=1)
    assert tuned.trials["weight"] == 1
