import pandas as pd

from riskrank.replay_first import back_to_anchor, boundary_tie, candidate_dates, evaluate_anchor, separation_date

t = lambda d: pd.Timestamp(d, tz="UTC")


def test_separation_date_is_the_earliest_month_start_meeting_the_minimum():
    times = [t("2017-03-15"), t("2017-03-02"), t("2017-04-03"), t("2017-05-01 09:00")]
    cands = candidate_dates(times)
    assert cands == [t("2017-03-02"), t("2017-04-03"), t("2017-05-01 09:00")]
    snaps = [t("2016-01-01"), t("2016-04-01")]            # windows end 2017-01-01 and 2017-04-01
    date, table = separation_date(cands, snaps, 12, 2)
    assert date == t("2017-04-03")                       # strictly after the second window's end
    assert [r["training_snapshots"] for r in table] == [1, 2, 2]


def test_blamed_paths_are_followed_back_through_renames_after_the_anchor():
    commits = [
        {"when": t("2019-01-01"), "files": []},                                          # anchor
        {"when": t("2020-01-01"), "files": [{"old": "flask/app.py", "new": "src/flask/app.py", "lines": 0}]},
        {"when": t("2021-01-01"), "files": []},                                          # fix
    ]
    back = back_to_anchor(commits, {"src/flask/app.py", "src/flask/cli.py"}, t("2019-01-01"), t("2021-01-01"))
    assert back == {"src/flask/app.py": "flask/app.py", "src/flask/cli.py": "src/flask/cli.py"}


def test_one_anchor_worked_by_hand():
    # three touched modules; b is blamed. Risk order a, b, c. Sizes 10, 30, 60 (100 lines).
    f = pd.DataFrame({"id": ["a", "b", "c"], "p": [0.9, 0.5, 0.1], "raw": [0.0, 0.0, 0.0], "size": [10, 30, 60]})
    last = {"a": t("2020-01-01"), "b": t("2020-06-01"), "c": None}
    r = evaluate_anchor(f, {"b"}, last)
    # artefacts: ceil(0.2 * 3) = 1 -> only a -> recall 0; chance = realised share 1/3
    assert r["system_artefacts"] == 0.0 and abs(r["chance_artefacts"] - 1 / 3) < 1e-12
    # lines: limit 20; a (10) taken, b crosses the limit and is kept whole -> recall 1, share 0.4
    assert r["system_lines"] == 1.0 and abs(r["realised_share_lines"] - 0.4) < 1e-12
    # recency: b (newest), a, c (never changed, last) -> artefacts picks b
    assert r["recency_artefacts"] == 1.0
    # largest-first artefacts picks c; smallest-first lines takes a then b
    assert r["largest_first_artefacts"] == 0.0 and r["smallest_first_lines"] == 1.0
    # off-budget size baselines: largest-first lines takes c whole (60 > limit 20) -> 0;
    # smallest-first artefacts picks a -> 0
    assert r["largest_first_lines"] == 0.0 and r["smallest_first_artefacts"] == 0.0
    # realised line shares per ordering: largest-first marks c only (0.6); smallest-first a+b (0.4);
    # recency b only (0.3)
    assert abs(r["largest_first_share_lines"] - 0.6) < 1e-12 and abs(r["smallest_first_share_lines"] - 0.4) < 1e-12
    assert abs(r["recency_share_lines"] - 0.3) < 1e-12
    # lines chance over all 6 orders (limit 20): b is marked in abc, bac and bca; in acb and cab/cba
    # c is taken first or second and the limit is reached before b -> 3/6
    assert r["chance_lines"] == 0.5 and r["chance_lines_method"] == "every order"


def test_boundary_tie_is_found_only_when_equal_scores_straddle_the_budget():
    from riskrank.budget import rank
    # five modules, artefact budget ceil(0.2 * 5) = 1: top module a; b has the same score as a
    tied = pd.DataFrame({"id": list("abcde"), "p": [0.9, 0.9, 0.5, 0.3, 0.1], "raw": [0.0] * 5, "size": [10] * 5})
    clear = tied.assign(p=[0.9, 0.8, 0.5, 0.3, 0.1])
    assert boundary_tie(rank(tied, "default"), "artefacts") is True
    assert boundary_tie(rank(clear, "default"), "artefacts") is False
