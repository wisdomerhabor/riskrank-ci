"""Scott-Knott ESD rules on hand-made scores. Unit-test fixtures only; never evidence."""
from riskrank.sk_esd import cohen_d, sk_esd


def test_clearly_separated_families_get_two_ranks():
    s = {"a": [0.80, 0.81, 0.79, 0.80, 0.82, 0.80], "b": [0.79, 0.80, 0.81, 0.80, 0.79, 0.81],
         "c": [0.50, 0.51, 0.49, 0.50, 0.52, 0.50]}
    out = sk_esd(s)
    assert out["ranks"]["a"] == out["ranks"]["b"] == 1 and out["ranks"]["c"] == 2


def test_identical_families_share_one_rank():
    s = {"a": [0.6, 0.7, 0.5, 0.6], "b": [0.6, 0.7, 0.5, 0.6]}
    assert set(sk_esd(s)["ranks"].values()) == {1}


def test_cohen_d_of_two_known_samples():
    # means 2 and 4, both sample sd 1 -> d = -2
    assert abs(cohen_d([1, 2, 3], [3, 4, 5]) + 2.0) < 1e-12


def test_benchmark_scores_average_each_projects_units_per_family():
    from riskrank.sk_esd import benchmark_scores
    rows = [{"corpus": "nasa", "mode": "within", "family": f, "project": p, "aucpr": v}
            for f, p, v in [("lr", "A", 0.2), ("lr", "A", 0.4), ("lr", "B", 0.5),
                            ("rf", "A", 0.6), ("rf", "B", 0.7), ("rf", "B", 0.9)]]
    rows.append({"corpus": "nasa", "mode": "cross", "family": "lr", "project": "A", "aucpr": 1.0})  # ignored
    assert benchmark_scores(rows, "nasa") == {"lr": [0.30000000000000004, 0.5], "rf": [0.6, 0.8]}
