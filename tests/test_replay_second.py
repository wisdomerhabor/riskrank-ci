"""The second check's pure rules (section 3.10.3). Unit-test fixtures only; never evidence."""
import pytest

from riskrank.replay_second import (added_tests, apfd, apfdc, backport_source, must_precede, ordering_result,
                                    python_for, run_status)


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
