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
