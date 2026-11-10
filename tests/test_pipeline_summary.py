"""pipeline_summary on hand-made plan files. Unit-test fixtures only; never evidence."""
import json

from riskrank.pipeline_summary import main, summarise


def plan(status, hook=0.5, share=0.9, since=3, reason=""):
    p = {"status": status, "reason": reason}
    if status == "complete":
        p |= {"timing_seconds": {"hook_total": hook, "module_scoring": 0.1, "test_ordering": 0.0},
              "escalated": False,
              "test_plan": {"stats": {"in_budget_runtime_share": share, "in_budget_fraction": 0.5,
                                      "commits_since_map": since, "stale_map_warning": False}}}
    return p


def test_fallbacks_are_counted_and_listed_even_when_ci_was_green():
    out = summarise({"1": plan("complete", hook=0.4), "2": plan("complete", hook=0.6),
                     "3": plan("default order", reason="Module scoring failed (HTTPError: HTTP Error 500)")})
    assert out["complete"] == 2 and out["status"] == {"complete": 2, "default order": 1}
    assert out["not_complete"] == [["3", "default order", "Module scoring failed (HTTPError: HTTP Error 500)"]]
    assert out["hook_seconds"]["median"] == 0.5 and out["hook_seconds"]["max"] == 0.6


def test_exit_code_is_one_when_any_run_did_not_complete(tmp_path):
    for k, p in {"a": plan("complete"), "b": plan("partial")}.items():
        (tmp_path / k).mkdir()
        (tmp_path / k / "plan.json").write_text(json.dumps(p))
    assert main([str(tmp_path)]) == 1
    (tmp_path / "b" / "plan.json").write_text(json.dumps(plan("complete")))
    assert main([str(tmp_path)]) == 0


def test_runs_are_grouped_by_settings_and_can_be_filtered(tmp_path):
    a, b = plan("complete"), plan("complete")
    b["settings"] = {"models": {"module": {"version": "v3"}}}
    for k, p in {"1": a, "2": b, "3": plan("partial")}.items():
        (tmp_path / k).mkdir()
        (tmp_path / k / "plan.json").write_text(json.dumps(p))
    out = summarise({"1": a, "2": b})
    assert sorted(sum(v.values()) for v in out["by_settings"].values()) == [1, 1]
    keep = tmp_path / "counted.txt"
    keep.write_text("1\n2\n")
    assert main([str(tmp_path), "--runs", str(keep)]) == 0  # run 3 is not counted, so it is ignored
