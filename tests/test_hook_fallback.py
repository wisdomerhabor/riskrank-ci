"""Forced failures and escalation (Table 3.10 integration reliability; section 3.10.5 unit-test list).

Every forced failure must leave the build passing with the default test order and a warning, and
escalation must work independently of the testing budget. The scoring service is replaced by a stub of
the hook's own HTTP calls; the repository is a two-commit fixture. Unit-test fixtures only: never evidence.
"""
import argparse
import io
import json
import subprocess
import urllib.error

import pytest
import yaml

from riskrank import hook

TESTS = ["t_a", "t_b", "t_none", "t_new"]
HEX = "a" * 64


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def push(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "pkg").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "src" / "pkg" / "a.py").write_text("def a():\n    return 1\n")
    (repo / "src" / "pkg" / "b.py").write_text("".join(f"def b{i}():\n    return {i}\n" for i in range(50)))
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "first")
    before = _git(repo, "rev-parse", "HEAD")
    (repo / "src" / "pkg" / "a.py").write_text("def a():\n    return 2\n")
    (repo / "src" / "pkg" / "b.py").write_text("".join(f"def b{i}():\n    return {i + 1}\n" for i in range(50)))
    _git(repo, "commit", "-q", "-am", "second")
    after = _git(repo, "rev-parse", "HEAD")
    cov = tmp_path / "coverage-map.json"
    cov.write_text(json.dumps({"meta": {"commit": before},
                               "tests": {"t_a": ["src/pkg/a.py"], "t_b": ["src/pkg/b.py"], "t_none": []}}))
    tests = tmp_path / "tests.jsonl"
    tests.write_text("".join(json.dumps(t) + "\n" for t in TESTS))
    return {"repo": repo, "before": before, "after": after, "cov": cov, "tests": tests, "tmp": tmp_path}


def _settings(tmp, **over):
    s = {"repository": "o/r", "budget": 20, "budget_unit": "lines", "ranking_key": "default", "risk_threshold": 0.5,
         "models": {"module": {"version": "m1", "sha256": HEX}, "commit": {"version": "c1", "sha256": HEX}},
         "scoring_timeout_seconds": 5, "coverage_refresh_interval": "weekly", "source_roots": ["src/"],
         "exclude_paths": ["tests/"], "deploying_team": ["t@t"]} | over
    path = tmp / f"settings-{len(list(tmp.glob('settings-*')))}.yml"
    path.write_text(yaml.safe_dump(s))
    return str(path)


def _args(p, settings, *, coverage=True, service="http://scoring.invalid:8000"):
    return argparse.Namespace(repo=str(p["repo"]), before=p["before"], after=p["after"], settings=settings,
                              service=service, coverage=str(p["cov"]) if coverage else str(p["tmp"] / "none.json"),
                              tests=str(p["tests"]), out=str(p["tmp"] / "plan"))


def _stub_service(monkeypatch, module_p, commit_p, health_fails=False):
    def post(url, body, timeout):
        if body["granularity"] == "module":
            return {"model_version": "m1", "transferred": False, "scores": [
                {"id": x["id"], "p": module_p[x["id"]], "raw": module_p[x["id"]], "in_training_range": True,
                 "reason_codes": []} for x in body["artefacts"]]}
        return {"model_version": "c1", "transferred": False, "scores": [
            {"id": x["id"], "p": commit_p, "raw": commit_p, "in_training_range": True, "reason_codes": []}
            for x in body["artefacts"]]}

    def urlopen(url, timeout=None):
        if health_fails:
            raise urllib.error.URLError("service down")
        return io.BytesIO(json.dumps({"models": {"commit": {"features": list(hook.JIT_FEATURES)}}}).encode())

    monkeypatch.setattr(hook, "post", post)
    monkeypatch.setattr(hook.urllib.request, "urlopen", urlopen)


def _order(p):
    return (p["tmp"] / "plan" / "order.txt").read_text().split()


MODULE_P = {"src/pkg/a.py": 0.9, "src/pkg/b.py": 0.1}


# ------------------------------------------------------------------ forced failures -> default order
def _assert_default(plan, p, reason, capsys):
    assert plan["status"] == "default order" and reason in plan["reason"]
    assert _order(p) == TESTS  # the default order, every test kept
    assert "::warning" in capsys.readouterr().out


def test_invalid_settings_fall_back(push, capsys):
    plan = hook.run(_args(push, _settings(push["tmp"], budget=0)))
    _assert_default(plan, push, "Invalid settings", capsys)


def test_unreachable_service_falls_back(push, capsys):
    plan = hook.run(_args(push, _settings(push["tmp"], scoring_timeout_seconds=2), service="http://127.0.0.1:9"))
    _assert_default(plan, push, "Module scoring failed", capsys)


def test_refused_model_falls_back(push, monkeypatch, capsys):
    def post(url, body, timeout):  # the service answers 503 when a bundle fails its checksum
        raise urllib.error.HTTPError(url, 503, "model failed to load", {}, None)
    monkeypatch.setattr(hook, "post", post)
    plan = hook.run(_args(push, _settings(push["tmp"])))
    _assert_default(plan, push, "Module scoring failed", capsys)


def test_missing_coverage_map_falls_back(push, monkeypatch, capsys):
    _stub_service(monkeypatch, MODULE_P, 0.1)
    plan = hook.run(_args(push, _settings(push["tmp"]), coverage=False))
    _assert_default(plan, push, "Coverage map unavailable", capsys)


def test_unexpected_error_never_fails_the_build(push, monkeypatch, capsys):
    def boom(a):
        raise RuntimeError("anything")
    monkeypatch.setattr(hook, "run", boom)
    monkeypatch.setattr("sys.argv", ["hook", "--before", push["before"], "--after", push["after"],
                                     "--tests", str(push["tests"]), "--out", str(push["tmp"] / "plan")])
    hook.main()  # must not raise
    plan = json.loads((push["tmp"] / "plan" / "plan.json").read_text())
    _assert_default(plan, push, "Unexpected error", capsys)


def test_commit_scoring_failure_keeps_the_module_plan(push, monkeypatch):
    _stub_service(monkeypatch, MODULE_P, 0.9, health_fails=True)
    plan = hook.run(_args(push, _settings(push["tmp"])))
    assert plan["status"] == "partial" and plan["commit_level"]["status"] == "not assessed"
    assert [r["module"] for r in plan["ranking"]] == ["src/pkg/a.py", "src/pkg/b.py"]
    assert plan["escalated"] is False


# ------------------------------------------------------------------ escalation is independent of the budget
def test_escalation_never_changes_the_order_or_depends_on_the_budget(push, monkeypatch):
    runs = {}
    for commit_p in (0.9, 0.1):
        for budget in (20, 100):
            _stub_service(monkeypatch, MODULE_P, commit_p)
            plan = hook.run(_args(push, _settings(push["tmp"], budget=budget)))
            runs[(commit_p, budget)] = (plan["escalated"], _order(push), plan["status"])
    assert {k: v[0] for k, v in runs.items()} == {(0.9, 20): True, (0.9, 100): True, (0.1, 20): False, (0.1, 100): False}
    orders = {tuple(v[1]) for v in runs.values()}
    assert orders == {("t_a", "t_b", "t_new", "t_none")}  # a.py ranks first; untouched tests trail by id
    assert all(v[2] == "complete" for v in runs.values())


def test_no_commit_model_means_escalation_not_configured(push, monkeypatch):
    _stub_service(monkeypatch, MODULE_P, 0.9)
    s = _settings(push["tmp"], models={"module": {"version": "m1", "sha256": HEX}})
    plan = hook.run(_args(push, s))
    assert plan["commit_level"]["status"] == "not configured" and plan["escalated"] is False


def test_plan_records_when_the_coverage_map_was_made(push, monkeypatch):
    """The dashboard shows the map's age with a date (feedback 9 Nov), so the plan carries it."""
    _stub_service(monkeypatch, MODULE_P, 0.1)
    meta = json.loads(push["cov"].read_text())
    meta["meta"]["collected_at"] = "2026-11-08T20:00:00+00:00"
    push["cov"].write_text(json.dumps(meta))
    plan = hook.run(_args(push, _settings(push["tmp"])))
    stats = plan["test_plan"]["stats"]
    expected = _git(push["repo"], "show", "-s", "--format=%cI", push["before"])
    assert stats["commits_since_map"] == 1 and stats["coverage_map_committed_at"] == expected
    assert stats["coverage_map_collected_at"] == "2026-11-08T20:00:00+00:00"
