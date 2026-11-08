import numpy as np
import pandas as pd
import pytest

from riskrank.features import FeaturePipeline, SchemaMismatch


def frame(**cols):
    return pd.DataFrame(cols)


def test_later_feature_of_a_correlated_pair_is_dropped():
    rng = np.random.default_rng(0)
    a = rng.integers(1, 100, 500)
    base = frame(a=a, b=a * 2 + 1, c=rng.integers(1, 100, 500))
    p = FeaturePipeline(["a", "b", "c"]).fit(base)
    assert p.kept_ == ["a", "c"]
    assert p.dropped_[0][:2] == ("b", "a")


def test_filter_is_within_family_only():
    a = np.arange(1, 501)
    base = frame(a=a, b=a + 3)
    p = FeaturePipeline(["a", "b"], {"a": "static", "b": "process"}).fit(base)
    assert p.kept_ == ["a", "b"]


def test_strong_negative_correlation_also_drops():
    a = np.arange(1, 501)
    base = frame(a=a, b=1000 - a)
    assert FeaturePipeline(["a", "b"]).fit(base).kept_ == ["a"]


def test_imputation_uses_base_median_and_is_counted():
    base = frame(a=[1.0, 2.0, 3.0, 4.0, 100.0], c=[5.0, 1.0, 4.0, 2.0, 3.0])
    p = FeaturePipeline(["a", "c"]).fit(base)
    x, n = p.transform(frame(a=[np.nan], c=[1.0]), scale=False)
    assert n == 1
    assert x["a"].iloc[0] == pytest.approx(np.log1p(3.0))


def test_scaled_base_has_zero_mean_unit_sd():
    rng = np.random.default_rng(1)
    base = frame(a=rng.integers(0, 1000, 300), c=rng.integers(0, 50, 300))
    p = FeaturePipeline(["a", "c"]).fit(base)
    x, _ = p.transform(base, scale=True)
    assert np.allclose(x.mean(), 0, atol=1e-9) and np.allclose(x.std(ddof=0), 1)


def test_fit_never_reads_other_partitions():
    base = frame(a=[1.0, 2.0, 3.0, 4.0], c=[4.0, 1.0, 3.0, 2.0])
    p = FeaturePipeline(["a", "c"]).fit(base)
    before = p.schema
    p.transform(frame(a=[1e6, 1e7], c=[0.0, 0.0]), scale=True)
    assert p.schema == before


def test_schema_rejects_wrong_names_or_order():
    p = FeaturePipeline(["a", "c"]).fit(frame(a=[1.0, 2.0, 3.0], c=[3.0, 1.0, 2.0]))
    with pytest.raises(SchemaMismatch):
        p.check(["c", "a"])
    with pytest.raises(SchemaMismatch):
        p.transform(frame(a=[1.0]), scale=False)


def test_negative_values_rejected():
    with pytest.raises(ValueError):
        FeaturePipeline(["a"]).fit(frame(a=[-5.0, 1.0]))
