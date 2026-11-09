"""pytest plugin for the second bounded check (runs inside the anchor's own environment; Python 3.6+).

RESULTS=<file>    write {node id: {"outcome", "exc", "call"}} as JSON at the end of the session:
                  outcome = passed | failed (call phase) | error (setup or teardown) | skipped
                  exc     = the exception type name of a failure or error, else null
                  call    = call-phase duration in seconds
COVERAGE_CONTEXTS=1   switch coverage.py's dynamic context to the node id at each test's setup, so a later
                      `coverage run -m pytest` records which test executed each line
"""
import json
import os

import pytest

_r = {}


def _cov():
    try:
        import coverage
        return coverage.Coverage.current()
    except Exception:
        return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    rec = _r.setdefault(rep.nodeid, {"outcome": "passed", "exc": None, "call": 0.0})
    exc = call.excinfo.typename if call.excinfo is not None else None
    if rep.when == "call":
        rec["call"] = round(rep.duration, 6)
        if rep.skipped:
            rec["outcome"] = "skipped"
        elif rep.failed:
            rec["outcome"], rec["exc"] = "failed", exc
    elif rep.failed:
        rec["outcome"], rec["exc"] = "error", exc
    elif rep.skipped and rep.when == "setup":
        rec["outcome"] = "skipped"


def pytest_runtest_setup(item):
    if os.environ.get("COVERAGE_CONTEXTS"):
        cov = _cov()
        if cov is not None:
            cov.switch_context(item.nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    yield  # teardown still belongs to this test's context; only then leave it
    if os.environ.get("COVERAGE_CONTEXTS"):
        cov = _cov()
        if cov is not None:
            cov.switch_context("")


def pytest_sessionfinish(session):
    path = os.environ.get("RESULTS")
    if path:
        with open(path, "w") as f:
            json.dump(_r, f)
