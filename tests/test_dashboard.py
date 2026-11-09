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
    assert "signed in: **qa-lead**" in t and "Build escalated: **yes**" in t
    warnings = " ".join(w.value for w in at.warning)
    assert "Scores transferred:" in warnings                                  # a banner, not body text (HD-3)
    assert "threshold 0.50" in t and "decided by the pipeline" in t           # unrounded value vs threshold (HD-1/2)
    assert "Why this risk score?" in t and "Selected: `src/x.py`" in t       # reasons beside the ranking
    assert "tests that exercise them: **1** of 3 (33.3% of the suite)" in t   # budget's own unit first (T-3)
    assert "commit `cm1`" in t and "**1** ranked module(s) not in the map" in t
    assert at.selectbox(key=at.selectbox[-1].key).label.startswith("Explain module")  # keyboard path (R-3)
    infos = " ".join(i.value for i in at.info)
    assert "No offline evaluation is stored" in infos and "not on `o/r`'s builds" in infos  # corpus banner (O-1)


def test_header_region_low_severity_fixes(env):
    """Walkthrough header findings HD-4 to HD-9 (severity 1 and 2)."""
    at = sign_in(AppTest.from_file(APP, default_timeout=30).run(), "qa-lead", "correct horse battery")
    assert not at.exception, at.exception
    t = text(at)
    assert at.title and all(x.value == "RiskRank: Regression-Test-Prioritisation Dashboard" for x in at.title)  # HD-6, HD-8
    assert "highest commit-level risk" in t and "Module-level model:" in t                                  # HD-4
    assert "Project: `o/r`" not in t and "Build run 101" not in t                                           # HD-7
    build = next(b for b in at.selectbox if b.label == "Build")
    assert build.value.endswith("· escalated")                                                               # HD-5
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
    assert "modules above the 0.50 threshold" in t                                                     # S-5
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
    assert set(table["in budget"]) <= {"✓ in budget", "–"} and set(table["flagged"]) <= {"⚑ above threshold", "–"}  # R-5, R-7
    assert list(table.columns[:4]) == ["#", "risk", "in budget", "flagged"]                              # R-8
    t = text(at)
    assert "No row selected: the explanation shows rank 1 by default." in t                              # R-6
    assert "modules touched" in t or "module touched" in t
    assert "ranks 1–1" not in t                                                                          # R-12
    assert any(b.label == "Reset to rank order" for b in at.button)                                      # R-10
    next(b for b in at.button if b.label == "Reset to rank order").click()
    assert not at.run().exception
