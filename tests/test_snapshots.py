"""Rename handling for module snapshots, modelled on Flask's real 2019 move and 2023 app.py split."""

import re

import pandas as pd

from riskrank.mining.snapshots import forward_paths, process_window

KW = re.compile(r"\bfix\b", re.IGNORECASE)


def c(when, files, message="change"):
    return {"hash": when, "when": pd.Timestamp(when, tz="UTC"), "message": message,
            "files": [{"old": o, "new": n, "lines": l} for o, n, l in files]}


# 2019: flask/app.py -> src/flask/app.py. 2023: src/flask/app.py -> src/flask/sansio/app.py,
# then a NEW src/flask/app.py is created.
HISTORY = [
    c("2019-01-10", [("flask/app.py", "flask/app.py", 5)], "fix a bug"),
    c("2019-06-01", [("flask/app.py", "src/flask/app.py", 0)]),
    c("2023-08-19 10:00", [("src/flask/app.py", "src/flask/sansio/app.py", 0)]),
    c("2023-08-19 11:00", [("src/flask/app.py", "src/flask/app.py", 900)]),
    c("2023-09-01", [("src/flask/app.py", "src/flask/app.py", 7), ("src/flask/sansio/app.py", "src/flask/sansio/app.py", 3)]),
]


def test_backward_walk_follows_the_2019_move():
    out = process_window(HISTORY, ["src/flask/app.py"], pd.Timestamp("2019-01-01", tz="UTC"),
                         pd.Timestamp("2020-01-01", tz="UTC"), KW)
    assert out["src/flask/app.py"] == {"churn": 5, "changes": 1, "prior_fixes": 1}


def test_recreated_app_py_does_not_inherit_the_old_history():
    out = process_window(HISTORY, ["src/flask/app.py", "src/flask/sansio/app.py"],
                         pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"), KW)
    assert out["src/flask/app.py"]["churn"] == 900 + 7          # only the new file's own changes
    assert out["src/flask/sansio/app.py"]["churn"] == 5 + 3     # the original lineage, via both renames
    assert out["src/flask/sansio/app.py"]["prior_fixes"] == 1


def test_forward_walk_labels_the_right_module():
    snap = pd.Timestamp("2023-01-01", tz="UTC")
    live = ["src/flask/app.py"]  # before the split there is one app.py
    fix_on_sansio = [{"when": pd.Timestamp("2023-10-01", tz="UTC"), "blamed": {"src/flask/sansio/app.py"}}]
    fix_on_new_app = [{"when": pd.Timestamp("2023-10-01", tz="UTC"), "blamed": {"src/flask/app.py"}}]
    assert forward_paths(HISTORY, live, snap, fix_on_sansio) == {"src/flask/app.py"}
    assert forward_paths(HISTORY, live, snap, fix_on_new_app) == set()  # a different file now


def test_last_change_follows_renames_and_ignores_pure_moves():
    import pandas as pd
    from riskrank.mining.snapshots import last_change
    t = lambda d: pd.Timestamp(d, tz="UTC")
    commits = [
        {"when": t("2025-01-01"), "files": [{"old": "a.py", "new": "a.py", "lines": 5}]},
        {"when": t("2025-02-01"), "files": [{"old": "a.py", "new": "b.py", "lines": 0}]},   # pure move
        {"when": t("2025-03-01"), "files": [{"old": "c.py", "new": "c.py", "lines": 2}]},
        {"when": t("2025-05-01"), "files": [{"old": "b.py", "new": "b.py", "lines": 9}]},   # after the snapshot
    ]
    got = last_change(commits, ["b.py", "c.py"], t("2025-04-01"))
    assert got == {"b.py": t("2025-01-01"), "c.py": t("2025-03-01")}
