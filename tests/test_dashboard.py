import json

import pytest
from streamlit.testing.v1 import AppTest

from riskrank import auth
from riskrank.store import Store
from tests.test_store import PLAN

APP = "../src/riskrank/dashboard.py"


@pytest.fixture
def env(tmp_path, monkeypatch):
    rec = tmp_path / "auth.json"
    rec.write_text(json.dumps(auth.make_record("qa-lead", "correct horse battery")))
    db = f"sqlite:///{tmp_path}/d.db"
    s = Store(db)
    s.put_model({"version": "m1", "granularity": "module", "algorithm": "lr", "corpus": ["pallets/flask"],
                 "explanation": {"attribution": "linear (exact)"}})
    s.put_runtime_run("o/r", "101", "2026-11-08T10:00:00Z", PLAN)
    monkeypatch.setenv("RISKRANK_AUTH", str(rec))
    monkeypatch.setenv("RISKRANK_DB", db)


def text(at):
    return " ".join(m.value for m in at.markdown) + " ".join(str(c.value) for c in at.caption)


def plain(at):
    """The page text with the labelled-rows HTML flattened: tags become single spaces."""
    import re
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text(at)))


def sign_in(at, user, pw):
    at.text_input[0].input(user)
    at.text_input[1].input(pw)
    at.button[0].click()
    return at.run()


def test_nothing_is_shown_before_login(env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "src/x.py" not in text(at) and not at.dataframe
    at = sign_in(at, "qa-lead", "wrong password")
    assert at.error and not at.dataframe


def test_regions_after_login(env):
    at = sign_in(AppTest.from_file(APP, default_timeout=30).run(), "qa-lead", "correct horse battery")
    assert not at.exception, at.exception
    t = text(at)
    p = plain(at)
    assert "Signed in qa-lead" in p and "Escalation Yes :" in p and 'rr-pill rr-warn">Escalated' in t  # labelled rows
    warnings = " ".join(w.value for w in at.warning)
    assert "Scores transferred:" in warnings                                  # a banner, not body text (HD-3)
    assert "threshold 0.50" in p and "Decided by the pipeline, using the threshold in riskrank.yml" in p           # unrounded value vs threshold (HD-1/2)
    assert "Why this risk score?" in t and "Selected src/x.py" in p       # reasons beside the ranking
    assert "In-budget tests 1 of 3 (33.3% of the suite)" in p   # budget's own unit first (T-3)
    assert "Map commit cm1 Map age" in p and "Not in the map 1 ranked module" in p
    assert at.selectbox(key=at.selectbox[-1].key).label.startswith("Explain module")  # keyboard path (R-3)
    infos = " ".join(i.value for i in at.info)
    assert "No offline evaluation is stored" in infos and "not on `o/r`'s builds" in infos  # corpus banner (O-1)


def test_header_region_low_severity_fixes(env):
    """Walkthrough header findings HD-4 to HD-9 (severity 1 and 2)."""
    at = sign_in(AppTest.from_file(APP, default_timeout=30).run(), "qa-lead", "correct horse battery")
    assert not at.exception, at.exception
    t = text(at)
    assert at.title and all(x.value == "RiskRank: Regression-Test-Prioritisation Dashboard" for x in at.title)  # HD-6, HD-8
    assert "highest commit-level risk" in plain(at) and "Module-level model" in plain(at)                                  # HD-4
    assert "Project: `o/r`" not in t and "Build run 101" not in t                                           # HD-7
    build = next(b for b in at.selectbox if b.label == "Build")
    assert build.value.endswith("|  escalated")                                                               # HD-5
    assert at.query_params["project"] == "o/r" and str(at.query_params["build"]).isdigit()                  # HD-9


def test_build_choice_is_restored_from_the_address(env):
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["project"] = "o/r"
    at.query_params["build"] = "999"            # unknown id falls back to the newest build, not an error
    at = sign_in(at.run(), "qa-lead", "correct horse battery")
    assert not at.exception, at.exception
    assert next(b for b in at.selectbox if b.label == "Build").value.startswith("2026-11-08")


def test_settings_bar_low_severity_fixes(env):
    """Walkthrough settings-bar findings S-1 to S-9 (severity 1 and 2)."""
    at = sign_in(AppTest.from_file(APP, default_timeout=30).run(), "qa-lead", "correct horse battery")
    assert not at.exception, at.exception
    t = text(at)
    assert "match `riskrank.yml` as recorded for this build" in t                                      # S-1
    labels = {n.label: n for n in at.number_input}
    budget = next(n for lbl, n in labels.items() if lbl.startswith("Budget (%"))
    assert budget.label in ("Budget (% of changed modules)", "Budget (% of changed lines)")            # S-7
    assert all(n.help for n in at.number_input) and all(b.help for b in at.selectbox[2:5])             # S-2, S-4
    assert "Valid range 1–100" in t                                                                    # S-3
    assert "Flagged 1 of 3 modules above the 0.50 threshold" in plain(at)                                                     # S-5
    labels["Risk threshold"].set_value(0.05)
    at = at.run()
    t = text(at)
    assert "Preview" in " ".join(i.value for i in at.info)
    assert "no change to the in-budget selection" in t                                                 # S-9


def test_ranked_plan_low_severity_fixes(env):
    """Walkthrough ranked-plan findings R-5 to R-13 (severity 1 and 2); R-11 was fixed in 3b2babb."""
    at = sign_in(AppTest.from_file(APP, default_timeout=30).run(), "qa-lead", "correct horse battery")
    assert not at.exception, at.exception
    table = at.dataframe[0].value
    assert set(table["Budget"]) <= {"In budget", "Outside budget"} and set(table["Flagged"]) <= {"Flagged", "Below threshold"}
    assert set(table["Tests mapped"]) <= {"Mapped", "Unmapped"}                                       # feedback 9 Nov  # R-5, R-7
    assert list(table.columns[:4]) == ["#", "Risk score", "Budget", "Flagged"]                              # R-8
    t = text(at)
    assert "No row selected: the explanation shows rank 1 by default." in t                              # R-6
    assert "Touched 3 modules" in plain(at)
    assert "ranks 1–1" not in t                                                                          # R-12
    assert any(b.label == "Reset to rank order" for b in at.button)                                      # R-10
    next(b for b in at.button if b.label == "Reset to rank order").click()
    assert not at.run().exception


def test_explanation_panel_low_severity_fixes(env):
    """Walkthrough explanation-panel findings E-2 to E-9 (severity 1 and 2)."""
    from riskrank.dashboard import contribution_domain, reading
    assert reading("prior_fixes", 0.0, "lowered") == "no prior fixes (0) lowers risk"                    # E-5
    assert reading("size", 120.0, "raised") == "lines of code = 120 raises risk"
    assert contribution_domain([[{"contribution": 0.5}], [{"contribution": -1.0}]]) == [-1.075, 0.575]
    assert contribution_domain([[{"contribution": 2.0}]])[0] == 0.0  # no empty negative half     # E-3
    at = sign_in(AppTest.from_file(APP, default_timeout=30).run(), "qa-lead", "correct horse battery")
    assert not at.exception, at.exception
    t = text(at)
    assert "(shown by default; select a row to change)" in t                                          # E-6
    assert "dark blue raises risk, tan lowers it" in t                                                # E-7
    assert "= uncalibrated score 0.900" in t and "calibration maps it to the risk 0.70" in t          # E-4
    assert "Attribution: linear (exact): for a linear model" in t                                     # E-9


def test_test_plan_low_severity_fixes(env):
    """Walkthrough ordered-test-plan findings T-4 to T-10 (severity 1 and 2)."""
    at = sign_in(AppTest.from_file(APP, default_timeout=30).run(), "qa-lead", "correct horse battery")
    assert not at.exception, at.exception
    t = text(at)
    assert "**Ranked tests (2)**" in t and any(e.label.startswith("Unranked tests (1)") for e in at.expander)  # T-4
    tables = [d.value for d in at.dataframe]
    ranked = next(d for d in tables if "Module that sets its rank" in d.columns)
    assert "None" not in ranked.astype(str).to_numpy()                                                  # T-5
    assert list(ranked.columns[:2]) == ["#", "Budget"]                                               # T-6
    assert "Unranked tests 1 of the 3" in plain(at) and "excluded for zero size" not in t                  # T-4, T-10


def test_commit_view_low_severity_fixes(env):
    """Walkthrough commit-view findings C-4 to C-9 (severity 1 and 2); C-7 is covered by E-2's short names,
    C-8 was fixed in 3b2babb."""
    at = sign_in(AppTest.from_file(APP, default_timeout=30).run(), "qa-lead", "correct horse battery")
    at.selectbox(key="granularity").select("commits")
    at = at.run()
    assert not at.exception, at.exception
    t = text(at)
    assert "Risk 0.6000" in plain(at)                                                                      # C-5
    assert "Showing the top" in t and "five largest per commit" in t                                   # C-9
    table = next(d.value for d in at.dataframe if "Commit" in d.value.columns)
    assert set(table["Flagged"]) <= {"Flagged", "Below threshold"}
    assert table["Scored in build at"].str.match(r"^\d{4}-\d\d-\d\d \d\d:\d\d$").all()                 # C-6
    next(n for n in at.number_input if n.label == "Risk threshold").set_value(0.2)
    at = at.run()
    assert "the flagged column uses threshold 0.20" in " ".join(i.value for i in at.info)              # C-4


def test_offline_panels_low_severity_fixes(env, monkeypatch):
    """Walkthrough offline-panel findings O-4 to O-10 (severity 1 and 2); O-3 was fixed in 3b2babb."""
    import os
    s = Store(os.environ["RISKRANK_DB"])
    s.put_offline_run(project="pallets/flask", corpus_name="mined Flask", corpus_projects=["pallets/flask"],
                      model="m1", created="2026-11-08T09:00:00Z", budget_unit="lines", budget=20.0,
                      ranking_key="default", recall=0.4,
                      baselines=[{"baseline": "random", "recall": 0.2}, {"baseline": "largest-first", "recall": 0.3}],
                      detail={"curves": {"system": [[0, 0], [0.2, 0.4], [1, 1]], "random": [[0, 0], [1, 1]]},
                              "reliability": [{"mean_p": 0.1, "observed": 0.12, "n": 40},
                                              {"mean_p": 0.3, "observed": 0.25, "n": 3}],
                              "brier": 0.057, "rows": 100, "defective": 7},
                      build_ref="offline-m1")
    at = sign_in(AppTest.from_file(APP, default_timeout=30).run(), "qa-lead", "correct horse battery")
    assert not at.exception, at.exception
    t = text(at)
    assert "a constant forecast of the defect rate, 7.0%, scores 0.065" in t                           # O-6
    assert "numbers = modules in each bin" in t
    assert "Click a legend entry to isolate its curve" in t                                            # O-4
    assert "the selected build is the larger dark point" in t                                          # O-7


def test_layout_low_severity_fixes(env):
    """Walkthrough layout findings L-1 to L-5: status tiles above three labelled tabs of bordered cards."""
    at = sign_in(AppTest.from_file(APP, default_timeout=30).run(), "qa-lead", "correct horse battery")
    assert not at.exception, at.exception
    assert [m.label for m in at.metric] == ["Build escalated", "Pipeline status", "Coverage map age", "Pushed commit"]
    assert at.metric[0].value == "yes" and at.metric[2].value == "4 commits ago"                              # L-4
    assert [x.label for x in at.tabs] == ["This build: ranked changes", "Test plan the pipeline ran",
                                          "Offline model evaluation"]                                        # L-2, L-3


def test_coverage_map_age_carries_a_date(env):
    """Feedback 9 Nov: "28 commits ago" should include the date and time (shown in WAT)."""
    from riskrank.dashboard import local_time, map_age
    assert local_time("2026-11-08T20:29:24Z") == "8 Nov 2026, 21:29 WAT"
    assert map_age({"commits_since_map": 28, "coverage_map_committed_at": "2026-11-08T17:15:06+01:00"}) \
        == "28 commits ago, committed 8 Nov 2026, 17:15 WAT"
    assert map_age({"commits_since_map": 1}) == "1 commit ago" and map_age({}) == "age not recorded"
    import os
    s = Store(os.environ["RISKRANK_DB"])
    assert s.backfill_map_dates("o/r", {"cm1": "2026-11-08T09:00:00Z"}) == 1
    assert s.backfill_map_dates("o/r", {"cm1": "2026-01-01T00:00:00Z"}) == 0     # never overwrites
    at = sign_in(AppTest.from_file(APP, default_timeout=30).run(), "qa-lead", "correct horse battery")
    assert not at.exception, at.exception
    assert at.metric[2].value == "4 commits ago, committed 8 Nov 2026, 10:00 WAT"
    assert "Map age 4 commits ago, committed 8 Nov 2026, 10:00 WAT" in plain(at)


def test_dropdown_still_chooses_the_explained_module(env):
    """The keyboard path (R-3) keeps working now that a ticked row also drives the explanation."""
    at = sign_in(AppTest.from_file(APP, default_timeout=30).run(), "qa-lead", "correct horse battery")
    pick = next(b for b in at.selectbox if b.label.startswith("Explain module"))
    pick.select(pick.options[1])
    at = at.run()
    assert not at.exception, at.exception
    t = text(at)
    assert "Selected src/y.py" in plain(at) and "No row selected" not in t
