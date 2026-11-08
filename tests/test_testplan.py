from riskrank.testplan import build_plan

RANKED = [("app.py", True), ("ctx.py", True), ("json.py", False)]  # best first; json out of budget
COVERAGE = {
    "t_app_only": {"app.py"},
    "t_app_ctx": {"app.py", "ctx.py"},
    "t_app_json": {"app.py", "json.py"},
    "t_ctx": {"ctx.py"},
    "t_json": {"json.py"},
    "t_cli": {"cli.py"},           # exercises no touched module
    "t_empty_mod": {"empty.py"},   # exercises only a zero-size touched module
}
CURRENT = sorted(COVERAGE) + ["t_new"]  # t_new was added after the map was collected


def plan(**kw):
    return build_plan(RANKED, {"empty.py"}, COVERAGE, CURRENT, **kw)


def test_fr18_order_and_tie_rules():
    order = [i.test for i in plan().items]
    # app.py ties: t_app_ctx (0,1) before t_app_json (0,2) before t_app_only (0) -- a key that
    # runs out sorts after tied keys that continue; then ctx.py, then json.py
    assert order[:5] == ["t_app_ctx", "t_app_json", "t_app_only", "t_ctx", "t_json"]
    assert order[5:] == ["t_cli", "t_empty_mod", "t_new"]  # trailing block, ascending id


def test_in_budget_follows_the_deciding_module():
    marks = {i.test: i.in_budget for i in plan().items}
    assert marks["t_app_json"] is True   # decided by app.py, even though json.py is out of budget
    assert marks["t_json"] is False
    assert sum(marks.values()) == 4


def test_three_trailing_groups_counted_separately_and_no_test_lost():
    p = plan()
    assert p.stats["trailing"] == {"zero_size_only": 1, "no_coverage_record": 1, "no_ranked_module": 1}
    assert p.stats["tests_total"] == len(CURRENT)


def test_runtime_share_and_missing_durations():
    p = plan(durations={"t_app_ctx": 3.0, "t_ctx": 1.0, "t_json": 6.0})
    assert p.stats["in_budget_runtime_share"] == 4.0 / 10.0
    assert p.stats["tests_without_duration"] == len(CURRENT) - 3


def test_stale_map_and_missing_ranked_module():
    p = build_plan([("app.py", True), ("new_mod.py", True)], set(), {"t": {"app.py"}}, ["t"],
                   map_age_intervals=2.5)
    assert p.stats["stale_map_warning"] is True
    assert p.stats["ranked_modules_missing_from_map"] == ["new_mod.py"]


def test_5000_tests_within_nfr18_on_a_workstation():
    import time
    ranked = [(f"m{i:03d}.py", i < 20) for i in range(100)]
    cov = {f"t{i:05d}": {f"m{(i * 7) % 130:03d}.py", f"m{(i * 13) % 130:03d}.py"} for i in range(5000)}
    t0 = time.perf_counter()
    build_plan(ranked, set(), cov, sorted(cov))
    assert time.perf_counter() - t0 < 30  # NFR18 target; the measured runner timing is reported separately
