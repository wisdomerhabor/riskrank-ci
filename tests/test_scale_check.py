"""scale_check runs both timed jobs and reports them against their targets. Fixture only; never evidence."""
from riskrank.scale_check import TARGETS, run


def test_both_jobs_are_timed_against_their_targets():
    out = run(repeats=1)
    assert set(out["results"]) == set(TARGETS)
    for k, r in out["results"].items():
        assert r["target_seconds"] == TARGETS[k] and r["max_seconds"] >= 0
    assert "processor" in out["machine"] and "memory_mb" in out["machine"]
