import copy

import pandas as pd
import pytest

from riskrank.budget import mark, rank, realised_share
from riskrank.plan_format import PLAN_FORMAT, problems
from riskrank.settings import validate
from riskrank.store import Store

RC = [{"feature": "churn", "value": 120.0, "contribution": 0.8, "direction": "raised"},
      {"feature": "size", "value": 40.0, "contribution": -0.2, "direction": "lowered"}]

PLAN = {
    "format": PLAN_FORMAT, "status": "complete",
    "settings": {"budget": 20, "budget_unit": "lines", "ranking_key": "default", "risk_threshold": 0.5,
                 "models": {"module": {"version": "m1", "sha256": "a" * 64},
                            "commit": {"version": "c1", "sha256": "b" * 64}}},
    "push": {"before": "b0", "after": "a1", "commits": 2},
    "module_model": "m1", "module_transferred": True, "modules_touched": 3, "modules_unparsable": [],
    "ranking": [
        {"rank": 1, "module": "src/x.py", "p": 0.7, "raw": 0.9, "size": 100, "in_budget": True, "flagged": True, "reason_codes": RC},
        {"rank": 2, "module": "src/y.py", "p": 0.3, "raw": 0.2, "size": 300, "in_budget": False, "flagged": False, "reason_codes": RC},
        {"rank": 3, "module": "src/z.py", "p": 0.1, "raw": 0.1, "size": 0, "in_budget": False, "flagged": False, "reason_codes": []},
    ],
    "realised_share": 0.25,
    "test_plan": {"tests": [
        {"test": "t::a", "position": 1, "in_budget": True, "deciding_module": "src/x.py", "group": "ranked"},
        {"test": "t::b", "position": 2, "in_budget": False, "deciding_module": "src/y.py", "group": "ranked"},
        {"test": "t::c", "position": 3, "in_budget": False, "deciding_module": None, "group": "no_ranked_module"}],
        "stats": {"tests_total": 3, "in_budget_count": 1, "coverage_map_commit": "cm1", "commits_since_map": 4,
                  "ranked_modules_missing_from_map": ["src/y.py"], "stale_map_warning": False}},
    "commit_level": {"status": "assessed", "model_version": "c1", "transferred": True,
                     "commits": [{"id": "a1", "p": 0.6, "in_training_range": True, "above_threshold": True, "reason_codes": RC}]},
    "escalated": True, "timing_seconds": {"hook_total": 1.0},
}


@pytest.fixture
def store(tmp_path):
    s = Store(f"sqlite:///{tmp_path}/t.db")
    s.put_model({"version": "m1", "granularity": "module", "algorithm": "lr", "corpus": ["pallets/flask"]})
    s.put_model({"version": "c1", "granularity": "commit", "algorithm": "lgbm", "corpus": ["apachejit"]})
    return s


def test_valid_plan_has_no_problems():
    assert problems(PLAN) == []


@pytest.mark.parametrize("breaks", [
    lambda p: p.update(format=0), lambda p: p.update(status="weird"), lambda p: p.pop("ranking"),
    lambda p: p["ranking"][0].pop("raw"), lambda p: p["test_plan"]["tests"][0].update(position="1"),
    lambda p: p["settings"].pop("budget"), lambda p: p["push"].pop("after"),
])
def test_malformed_plans_are_rejected(breaks):
    p = copy.deepcopy(PLAN)
    breaks(p)
    assert problems(p)


def test_default_order_plan_is_accepted():
    p = {"format": PLAN_FORMAT, "status": "default order", "reason": "Coverage map unavailable.", "settings": {},
         "tests": [{"test": "t::a", "position": 1, "in_budget": False}]}
    assert problems(p) == []
    assert problems("not a plan")


def test_runtime_run_round_trips(store):
    rid = store.put_runtime_run("o/r", "101", "2026-11-08T10:00:00Z", PLAN, durations={"t::a": 0.5})
    run = store.run(rid)
    assert run["escalated"] is True and run["coverage_map_commit"] == "cm1" and run["zero_size_excluded"] == 1
    assert run["transferred"] is True and run["status"] == "complete"
    r = store.ranking(rid)
    assert r["module"].tolist() == ["src/x.py", "src/y.py", "src/z.py"]
    assert r.loc[0, "reason_codes"][0]["feature"] == "churn" and r.loc[0, "raw"] == 0.9
    tp = store.test_plan(rid)
    assert tp["test"].tolist() == ["t::a", "t::b", "t::c"] and tp.loc[0, "duration"] == 0.5
    assert store.commit_scores("o/r")["commit"].tolist() == ["a1"]
    assert store.has_run("o/r", "101") and not store.has_run("o/r", "102")
    trend = store.risk_trend("o/r")
    assert trend["mean_in_budget_risk"].tolist() == [0.7]


def test_not_configured_escalation_is_stored_as_unknown(store):
    p = copy.deepcopy(PLAN)
    p["settings"]["models"].pop("commit")
    p["commit_level"] = {"status": "not configured", "commits": []}
    p["escalated"] = False
    run = store.run(store.put_runtime_run("o/r", "7", "2026-11-08T10:00:00Z", p))
    assert run["jit_status"] == "not configured" and run["escalated"] is None


def test_latest_offline_evaluation_for_a_model(store):
    for day, rec in [("2026-11-01", 0.3), ("2026-11-05", 0.4)]:
        store.put_offline_run(project="all", corpus_name="ApacheJIT", corpus_projects=["a", "b"], model="c1",
                              created=day, budget_unit="lines", budget=20, ranking_key="default", recall=rec,
                              baselines=[{"baseline": "random", "recall": 0.2, "permutations": 100,
                                          "interval_low": 0.18, "interval_high": 0.22}],
                              detail={}, build_ref=f"eval-{day}")
    latest = store.latest_offline("c1")
    assert latest["recall"] == 0.4 and latest["corpus"] == "ApacheJIT" and latest["jit_model"] == "c1"
    assert latest["baselines"][0]["baseline"] == "random"
    assert store.latest_offline("m1") is None


def test_dump_and_load_moves_every_row_to_another_database(store, tmp_path):
    store.put_runtime_run("o/r", "101", "2026-11-08T10:00:00Z", PLAN)
    store.log_ingest("o/r", "101", "stored")
    other = Store(f"sqlite:///{tmp_path}/other.db")
    other.load(store.dump())
    assert {k: len(v) for k, v in other.dump().items()} == {k: len(v) for k, v in store.dump().items()}
    assert other.ranking(1)["module"].tolist() == store.ranking(1)["module"].tolist()


def test_shared_marking_matches_the_runtime_rules():
    f = pd.DataFrame({"id": ["a", "b", "c", "d"], "p": [0.9, 0.8, 0.5, 0.1], "raw": [0, 0, 0, 0],
                      "size": [10, 50, 0, 40]})
    r = rank(f, "default")
    m = mark(r, "lines", 0.2, "default")          # limit 20 of 100: a (10) taken, b crosses and is kept whole
    assert m.tolist() == [True, True, False, False]
    assert realised_share(r, m, "lines") == 0.6
    m = mark(r, "artefacts", 0.2, "default")      # ceil(0.8) = 1
    assert m.tolist() == [True, False, False, False]
    assert realised_share(r, m, "artefacts") == 0.25


def test_commit_model_is_optional_in_settings():
    s = {"budget": 20, "budget_unit": "lines", "ranking_key": "default", "risk_threshold": 0.5,
         "models": {"module": {"version": "m1", "sha256": "a" * 64}}, "scoring_timeout_seconds": 60,
         "coverage_refresh_interval": "weekly"}
    assert validate(s) == []
