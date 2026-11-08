import numpy as np
import pandas as pd

from riskrank.static_vs_process import flask_partitions


def test_flask_partitions_follow_the_preregistered_rule():
    """Unit-test fixture only. Never used as evidence (section 3.5.1)."""
    when = pd.date_range("2010-01-01", "2020-01-01", periods=400, tz="UTC")
    rng = np.random.default_rng(0)
    inducing = rng.random(400) < 0.2
    first_fix = pd.Series(when + pd.Timedelta(days=500)).where(inducing)
    df = pd.DataFrame({"when": when, "inducing": inducing, "first_fix": first_fix})
    split = pd.Timestamp("2018-06-01", tz="UTC")
    base, calib, test = flask_partitions(df, {"split_date": "2018-06-01T00:00:00Z"})
    gap = split - pd.DateOffset(years=1)
    assert base["when"].max() < calib["when"].min() and calib["when"].max() < gap
    assert test["when"].min() >= split
    # training labels need a fix before the split; test labels keep every fix
    late = pd.concat([base, calib])
    assert not (late["label"] & (late["first_fix"] >= split)).any()
    assert (test["label"] == test["inducing"]).all()
    span = gap - base["when"].min()
    assert abs((calib["when"].min() - base["when"].min()) / span - 0.8) < 0.02
