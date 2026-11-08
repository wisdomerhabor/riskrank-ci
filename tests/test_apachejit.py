"""Checks for the ApacheJIT partitioning. They run against the real published file."""

import os

import pandas as pd
import pytest

from riskrank.apachejit import CALIB_END, TEST_START, build

ROOT = os.environ.get("APACHEJIT_ROOT", "data/raw/apachejit")
pytestmark = pytest.mark.skipif(not os.path.exists(ROOT), reason="ApacheJIT not downloaded")


@pytest.fixture(scope="module")
def jit():
    return build(ROOT)


def test_thirteen_projects(jit):
    assert jit.commits["project"].nunique() == 13


def test_reproduces_section_3_5_count(jit):
    # Section 3.5: 2,944 of the 22,467 training commits labelled bug-inducing have no fix by
    # the end of 2016. "Training" here is everything before 1 Jan 2017 (base, calib and gap).
    before_2017 = jit.commits[jit.commits["when"] < TEST_START]
    published_buggy = before_2017[before_2017["buggy"]]
    assert len(published_buggy) == 22467
    assert (published_buggy["first_fix"] >= TEST_START).sum() == 2944


def test_gap_year_is_exactly_2016(jit):
    gap = jit.commits[jit.commits["partition"] == "gap"]
    assert gap["when"].min() >= CALIB_END and gap["when"].max() < TEST_START
    assert not jit.commits.loc[jit.commits["partition"] != "gap", "when"].between(
        CALIB_END, TEST_START, inclusive="left"
    ).any()


def test_chronology_within_each_project(jit):
    order = {"base": 0, "calib": 1, "gap": 2, "test": 3}
    for _, p in jit.commits.groupby("project"):
        p = p.sort_values("when")
        assert p["partition"].map(order).is_monotonic_increasing


def test_no_training_label_uses_a_test_period_fix(jit):
    train = jit.commits[jit.commits["partition"].isin(["base", "calib"])]
    assert not (train["label"] & (train["first_fix"] >= TEST_START)).any()


def test_test_labels_ignore_fixes_after_2019(jit):
    test = jit.commits[jit.commits["partition"] == "test"]
    late = test["buggy"] & (test["first_fix"] >= pd.Timestamp("2020-01-01", tz="UTC"))
    assert late.sum() == 165
    assert not test.loc[late, "label"].any()


def test_worked_example_from_section_3_8_2():
    # A project starting Jan 2006: base 2006-2013, calibration 2014-2015.
    from riskrank.apachejit import _partition

    frame = pd.DataFrame(
        {
            "project": ["p"] * 4,
            "when": pd.to_datetime(
                ["2006-01-01 00:00", "2013-12-31 23:59", "2014-01-01 00:00", "2016-06-01 00:00"],
                utc=True,
            ),
        }
    )
    assert list(_partition(frame)) == ["base", "base", "calib", "gap"]
