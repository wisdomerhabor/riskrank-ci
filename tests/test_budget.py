import pandas as pd
import pytest

from riskrank.budget import (
    marked_lines_runtime,
    random_recall,
    rank,
    recall_artefacts,
    recall_lines,
)


def artefacts(rows):
    return pd.DataFrame(rows, columns=["id", "p", "raw", "size", "label"])


def test_section_3_10_2_boundary_example():
    # 1,000-line defective module ranked above two 50-line modules; 20% of 1,100 = 220 lines.
    f = artefacts([("a", 0.9, 0.9, 1000, True), ("b", 0.5, 0.5, 50, False), ("c", 0.4, 0.4, 50, False)])
    r = rank(f, "default")
    assert recall_lines(r) == pytest.approx(0.22)  # offline: only 220 of its 1,000 lines count
    marked = marked_lines_runtime(r)
    assert marked.tolist() == [True, False, False]  # runtime keeps it whole
    assert r.loc[marked, "size"].sum() / r["size"].sum() == pytest.approx(1000 / 1100)  # 91%


def test_section_3_8_1_artefact_rounding():
    # 20 percent of 12 modules is 2.4, which rounds up to 3.
    f = artefacts([(f"m{i:02d}", 1 - i / 100, 1 - i / 100, 10, i < 3) for i in range(12)])
    assert recall_artefacts(rank(f, "default")) == 1.0
    f2 = artefacts([(f"m{i:02d}", 1 - i / 100, 1 - i / 100, 10, i == 3) for i in range(12)])
    assert recall_artefacts(rank(f2, "default")) == 0.0


def test_at_least_one_artefact_marked():
    f = artefacts([("a", 0.9, 0.9, 10, True), ("b", 0.1, 0.1, 10, False)])
    assert recall_artefacts(rank(f, "default"), b=0.01) == 1.0


def test_effort_key_uses_lower_bound():
    # p' = p(1 - 0.05) + 0.05. A zero-risk 5-line file outranks a 1,000-line file at p = 0.6:
    # 0.05 / 5 = 0.01 > (0.6 * 0.95 + 0.05) / 1000 = 0.00062.
    f = artefacts([("big", 0.6, 0.6, 1000, True), ("tiny", 0.0, 0.0, 5, False)])
    assert rank(f, "effort")["id"].tolist() == ["tiny", "big"]


def test_tie_break_uncalibrated_then_identifier():
    f = artefacts([("b", 0.5, 0.30, 10, False), ("a", 0.5, 0.30, 10, False), ("c", 0.5, 0.40, 10, True)])
    assert rank(f, "default")["id"].tolist() == ["c", "a", "b"]


def test_zero_size_never_marked_under_lines_but_counted():
    f = artefacts([("z", 0.9, 0.9, 0, True), ("a", 0.5, 0.5, 100, True), ("b", 0.1, 0.1, 400, False)])
    r = rank(f, "default")
    # Limit 100 lines: "a" is marked in full, "z" can never be marked -> recall 1 of 2.
    assert recall_lines(r) == pytest.approx(0.5)
    e = rank(f, "effort")
    assert e["id"].iloc[-1] == "z"  # zero-size goes last under the effort key


def test_random_baseline_artefacts_is_exactly_k_over_n():
    f = artefacts([(f"m{i}", 0.5, 0.5, 10, i % 4 == 0) for i in range(10)])
    assert random_recall(f, "artefacts") == pytest.approx(2 / 10)


def test_random_baseline_lines_close_to_budget_for_equal_sizes():
    f = artefacts([(f"m{i:03d}", 0.5, 0.5, 10, i % 5 == 0) for i in range(200)])
    assert random_recall(f, "lines", draws=100) == pytest.approx(0.2, abs=0.03)


def test_no_defects_gives_undefined_recall():
    f = artefacts([("a", 0.5, 0.5, 10, False)])
    assert pd.isna(recall_artefacts(rank(f, "default")))


def test_5000_artefacts_ranked_and_marked_within_target_on_a_workstation():
    """Table 3.10 ranking time: up to 5,000 artefacts within 60 seconds offline (fixture, not evidence)."""
    import time

    import numpy as np

    from riskrank.budget import mark

    rng = np.random.default_rng(0)
    n = 5000
    frame = pd.DataFrame({"id": [f"m{i:05d}" for i in range(n)], "p": rng.random(n), "raw": rng.normal(size=n),
                          "size": rng.integers(0, 2000, n), "label": rng.random(n) < 0.1})
    t0 = time.perf_counter()
    for key in ("default", "effort"):
        r = rank(frame, key)
        for unit in ("artefacts", "lines"):
            mark(r, unit, 0.2, key)
            (recall_artefacts if unit == "artefacts" else recall_lines)(r)
    assert time.perf_counter() - t0 < 60
