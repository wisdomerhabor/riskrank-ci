import pytest

from riskrank.settings import validate

GOOD = {
    "budget": 20, "budget_unit": "lines", "ranking_key": "default", "risk_threshold": 0.5,
    "models": {"module": {"version": "m1", "sha256": "a" * 64}, "commit": {"version": "c1", "sha256": "b" * 64}},
    "scoring_timeout_seconds": 60, "coverage_refresh_interval": "weekly",
}


def test_good_settings_pass():
    assert validate(GOOD, registry_versions={"m1", "c1"}, coverage_cron="0 3 * * 1") == []


@pytest.mark.parametrize("change", [
    {"budget": 0}, {"budget": 101}, {"budget": True}, {"budget_unit": "minutes"}, {"ranking_key": "size"},
    {"risk_threshold": 1.5}, {"scoring_timeout_seconds": 0}, {"coverage_refresh_interval": "hourly"},
    {"stop_at_first_failure": "yes"},
])
def test_each_rule_rejects(change):
    assert validate(GOOD | change)


def test_budget_of_exactly_100_is_allowed():
    assert validate(GOOD | {"budget": 100}) == []


def test_unpublished_model_and_bad_checksum_rejected():
    bad = GOOD | {"models": {"module": {"version": "m9", "sha256": "a" * 64}, "commit": {"version": "c1", "sha256": "xyz"}}}
    problems = validate(bad, registry_versions={"m1", "c1"})
    assert len(problems) == 2


def test_interval_must_match_the_coverage_workflow():
    assert validate(GOOD, coverage_cron="0 3 * * *")  # weekly configured, daily scheduled
