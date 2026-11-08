"""pytest plugin: record each test's call-phase duration by node id into $RISKRANK_DURATIONS (JSON)."""
import json
import os

_d = {}


def pytest_runtest_logreport(report):
    if report.when == "call":
        _d[report.nodeid] = round(report.duration, 6)


def pytest_sessionfinish(session):
    path = os.environ.get("RISKRANK_DURATIONS")
    if path:
        with open(path, "w") as f:
            json.dump(_d, f)
