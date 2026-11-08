import pandas as pd
import pytest

from riskrank import apachejit as aj


def fake(monkeypatch):
    t = lambda d: pd.Timestamp(d, tz="UTC").timestamp()
    total = pd.DataFrame({
        "commit_id": ["a", "b", "c", "d", "e", "f"],
        "project": ["apache/x"] * 6,
        "author_date": [t("2008-06-01"), t("2012-01-01"), t("2015-06-01"), t("2017-03-01"), t("2018-03-01"), t("2019-03-01")],
        "buggy": [True, True, True, True, True, False],
    })
    links = pd.DataFrame({"bug_hash": ["a", "b", "c", "d", "e"],
                          "fix_date": [t("2009-01-01"), t("2017-06-01"), t("2015-09-01"), t("2019-06-01"), t("2020-06-01")]})
    monkeypatch.setattr(aj, "load_raw", lambda root: (total, links))


def labels(variant):
    c = aj.build("x", variant).commits
    return dict(zip(c["commit_id"], c["label"]))


def test_variants(monkeypatch):
    fake(monkeypatch)
    # primary: b fixed after 2016 -> clean; e fixed after 2019 -> clean
    assert labels(None) == {"a": True, "b": False, "c": True, "d": True, "e": False, "f": False}
    # exclude_relabelled drops b and e instead of calling them clean
    assert labels("exclude_relabelled") == {"a": True, "c": True, "d": True, "f": False}
    # window_1y: a (7 months) and c (3 months) inside; b (5.5 years) out; d (27 months) out; 2019 dropped
    assert labels("window_1y") == {"a": True, "b": False, "c": True, "d": False, "e": False}
    # from_2010 drops a only
    assert set(labels("from_2010")) == {"b", "c", "d", "e", "f"}
    with pytest.raises(ValueError):
        aj.build("x", "nonsense")
