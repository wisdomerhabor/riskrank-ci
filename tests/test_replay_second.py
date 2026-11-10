"""The second check's pure rules (section 3.10.3). Unit-test fixtures only; never evidence."""
import pytest

from riskrank.replay_second import (added_tests, apfd, apfdc, backport_source, must_precede, ordering_result,
                                    python_for, run_status, strip_dev_tag)


def test_section_3_10_3_worked_examples():
    assert apfd(10, 200) == pytest.approx(0.9525)
    order = ["a", "b", "f"]  # 400 s suite, 30 s before the failing test, failing test 4 s
    assert apfdc(order, {"a": 10, "b": 20, "f": 4, "z": 366}, "f") == pytest.approx(
        (34 - 30 - 2) / 34)  # only tests in the order count towards T
    assert apfdc(["a", "f", "z"], {"a": 30, "f": 4, "z": 366}, "f") == pytest.approx(0.92)


def test_python_for_the_anchor_date():
    assert python_for("2017-04-03T10:00:00+00:00") == "3.6"
    assert python_for("2020-10-05T00:00:00+00:00") == "3.9"
    assert python_for("2020-10-04T23:00:00+00:00") == "3.8"
    assert python_for("2026-01-01T00:00:00Z") == "3.13"


BEFORE = "def test_old():\n    assert 1\n\n\nclass TestX:\n    def test_a(self):\n        assert 1\n"
AFTER = BEFORE + "\n\n@pytest.mark.parametrize('v', [1, 2])\ndef test_new(v):\n    assert v\n"


def test_exactly_the_new_test_is_found_and_parametrised_counts_once():
    assert added_tests({"tests/t.py": BEFORE}, {"tests/t.py": AFTER}) == [("tests/t.py", None, "test_new")]
    assert added_tests({"tests/n.py": None}, {"tests/n.py": "def test_a():\n    pass\ndef test_b():\n    pass\n"}) == [
        ("tests/n.py", None, "test_a"), ("tests/n.py", None, "test_b")]


def test_backport_appends_the_function_with_its_decorator():
    out = backport_source(BEFORE, AFTER, None, "test_new")
    assert "@pytest.mark.parametrize" in out and out.startswith(BEFORE.rstrip("\n"))
    method_fix = BEFORE + "        assert 1\n\n    def test_b(self):\n        assert 2\n"
    out = backport_source(BEFORE, method_fix, "TestX", "test_b")
    assert "    def test_b(self):" in out
    assert backport_source("x = 1\n", method_fix, "TestX", "test_b") is None  # the class is absent at the anchor
    assert backport_source(None, AFTER, None, "test_new") == AFTER          # new file copied whole


def test_run_status_separates_assertions_from_errors():
    ok = {"t.py::f": {"outcome": "passed", "exc": None, "call": 0.1}}
    fail = {"t.py::f": {"outcome": "failed", "exc": "AssertionError", "call": 0.1}}
    err = {"t.py::f": {"outcome": "failed", "exc": "NameError", "call": 0.1}}
    setup = {"t.py::f": {"outcome": "error", "exc": "ValueError", "call": 0.0}}
    raises = {"t.py::f[1]": {"outcome": "failed", "exc": "Failed", "call": 0.1}}
    assert run_status(ok, ["t.py::f"]) == "pass"
    assert run_status(fail, ["t.py::f"]) == "fail"
    assert run_status(err, ["t.py::f"]) == "error" and run_status(setup, ["t.py::f"]) == "error"
    assert run_status(raises, ["t.py::f"]) == "fail"
    assert run_status({}, ["t.py::f"]) == "error"


def test_ordering_tie_group_pessimistic_bound_and_must_precede():
    ranking = ["m1", "m2", "m3"]
    coverage = {"t_bp": {"m1", "m3"}, "t_same": {"m1", "m3"}, "t_super": {"m1", "m2", "m3"}, "t_m2": {"m2"}}
    current = sorted(coverage) + ["t_new"]
    d = {t: 1.0 for t in current}
    r = ordering_result(ranking, coverage, current, d, ["t_bp"])
    # keys: t_super (0,1,2), t_bp (0,2), t_same (0,2), t_m2 (1,), then t_new; ties by id: t_bp before t_same
    assert r["tf"] == 2 and r["n"] == 5 and r["tie_group"] == 2 and r["tie_decided"]
    assert r["apfd"] == pytest.approx(1 - 2 / 5 + 1 / 10)
    # pessimistic: t_bp treated as covering only m3 (rank 2) -> after t_super, t_same and t_m2
    assert r["pessimistic"]["tf"] == 4
    assert must_precede(coverage, {"m1", "m2", "m3"}, "t_bp") == 1


def test_tox_branch_archives_are_found_for_pinning(tmp_path):
    # Flask's 2020-21 tox.ini tested against its sibling libraries' branch archives; the harness pins each to its
    # last commit at the anchor's date instead of dropping them (which left Werkzeug 1.x under Flask 2.0 code).
    tox = tmp_path / "tox.ini"
    tox.write_text("[testenv]\ndeps =\n    -r requirements/tests.txt\n\n"
                   "    https://github.com/pallets/werkzeug/archive/master.tar.gz\n"
                   "    https://github.com/pallets/click/archive/refs/heads/main.zip\n"
                   "    examples/tutorial[test]\n")
    from riskrank.replay_second import tox_archives
    assert tox_archives(str(tox)) == [("pallets/werkzeug", "master"), ("pallets/click", "main")]


def test_combine_prefers_a_real_outcome_over_a_build_failure(tmp_path, monkeypatch):
    # The amendment's containers screen candidates that an earlier run recorded as build failures.
    import json
    from riskrank import replay_second as rs
    base, older = tmp_path / "base", tmp_path / "older"
    for d, rec in ((base, {"fix": "f1", "anchor_date": "2018-01-01T00:00:00Z", "excluded": "build_failure"}),
                   (older, {"fix": "f1", "anchor_date": "2018-01-01T00:00:00Z", "excluded": "backport_passes_all_three"})):
        (d / "second_check").mkdir(parents=True)
        (d / "second_check" / "screen.jsonl").write_text(json.dumps(rec) + "\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rs, "load_config", lambda: {"second_check": {}})
    monkeypatch.setattr(rs, "OUT", str(tmp_path / "out" / "second_check"))
    rs.phase_combine([str(base), str(older), str(tmp_path / "missing")])
    out = json.load(open(tmp_path / "out" / "second_check.json"))
    assert out["excluded"]["build_failure"] == 0 and out["excluded"]["backport_passes_all_three"] == 1
    assert out["reproduced"] == 0 and out["reporting"].startswith("fewer than 12")


def test_only_the_egg_info_dev_tag_is_removed_from_a_sibling_setup_cfg():
    cfg = "[egg_info]\ntag_build = 'dev'\ntag_date = 1\n\n[aliases]\nrelease = egg_info -RDb ''\n\n[x]\ntag_build = keep\n"
    out, dropped = strip_dev_tag(cfg)
    assert dropped and out == "[egg_info]\n\n[aliases]\nrelease = egg_info -RDb ''\n\n[x]\ntag_build = keep\n"
    assert strip_dev_tag("[metadata]\nlicense_file = LICENSE\n") == ("[metadata]\nlicense_file = LICENSE\n", False)


def test_a_teardown_error_does_not_hide_a_call_phase_assertion_failure(tmp_path):
    import json
    import os
    import subprocess
    import sys
    (tmp_path / "test_x.py").write_text(
        "import pytest\n\n@pytest.fixture\ndef leaky():\n    yield\n    assert False, 'teardown'\n\n"
        "def test_fails_then_teardown_errors(leaky):\n    assert 1 == 2\n\n"
        "def test_passes_then_teardown_errors(leaky):\n    pass\n")
    tools = os.path.join(os.path.dirname(__file__), "..", "tools", "second_check")
    res = tmp_path / "r.json"
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "results_plugin", "-p", "no:cacheprovider", "test_x.py"],
                   cwd=tmp_path, env=os.environ | {"PYTHONPATH": tools, "RESULTS": str(res)}, capture_output=True)
    got = json.loads(res.read_text())
    assert got["test_x.py::test_fails_then_teardown_errors"]["outcome"] == "failed"
    assert got["test_x.py::test_fails_then_teardown_errors"]["exc"] == "AssertionError"
    assert got["test_x.py::test_passes_then_teardown_errors"]["outcome"] == "error"
    assert run_status(got, ["test_x.py::test_fails_then_teardown_errors"]) == "fail"


def test_pessimistic_bound_is_aggregated_for_every_module_ordering():
    from riskrank.replay_second import summarise

    def res(apfd, papfd):
        o = {"tf": 1, "apfd": apfd, "apfdc": apfd, "pessimistic": {"tf": 9, "apfd": papfd, "apfdc": papfd}}
        return {"system": o, "largest_first": o, "recency": o, "random": {"tf": 5, "apfd": 0.5, "apfdc": 0.5}}

    results = [res(0.9, 0.005)] * 11 + [res(0.6, 0.4)]
    out = summarise([], results, {}, {"second_check": {}})
    agg = out["aggregate_pessimistic"]
    assert set(agg) == {"system", "largest_first", "recency"}
    assert agg["system"]["below_0.01"] == 11
    assert abs(agg["recency"]["apfd"] - (11 * 0.005 + 0.4) / 12) < 1e-12
    assert summarise([], results[:11], {}, {"second_check": {}})["aggregate_pessimistic"] is None


def test_system_positions_are_paired_against_each_baseline():
    from riskrank.replay_second import summarise

    def res(sys_tf, big_tf, rec_tf):
        o = lambda tf: {"tf": tf, "apfd": 0.5, "apfdc": 0.5, "pessimistic": {"tf": tf, "apfd": 0.5, "apfdc": 0.5}}
        return {"system": o(sys_tf), "largest_first": o(big_tf), "recency": o(rec_tf),
                "random": {"tf": 5, "apfd": 0.5, "apfdc": 0.5}}

    out = summarise([], [res(1, 5, 1), res(9, 3, 2), res(4, 4, 8)], {}, {"second_check": {}})
    assert out["system_vs_baselines"]["largest_first"] == {"earlier": 1, "later": 1, "same": 1}
    assert out["system_vs_baselines"]["recency"] == {"earlier": 1, "later": 1, "same": 1}
