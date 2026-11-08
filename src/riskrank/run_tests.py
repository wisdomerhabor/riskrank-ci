"""Run the repository's tests in plan order (FR18): in-budget tests first, then every remaining test,
recording each test's duration. Optional settings, both off by default and unused in evaluation:
stop_at_first_failure (-x) and run_only_in_budget. A plan in default order runs the plain suite.

    python -m riskrank.run_tests riskrank-plan riskrank.yml
"""
import json
import os
import subprocess
import sys

import yaml


def main() -> int:
    plan_dir, settings = sys.argv[1], sys.argv[2]
    s = yaml.safe_load(open(settings)) if os.path.exists(settings) else {}
    plan = json.load(open(os.path.join(plan_dir, "plan.json")))
    env = os.environ | {"RISKRANK_DURATIONS": os.path.join(plan_dir, "durations.json")}
    args = [sys.executable, "-m", "pytest", "-q", "-p", "riskrank.durations_plugin",
            "-p", "riskrank.order_plugin", "-p", "no:randomly"]
    if s.get("stop_at_first_failure") is True:
        args.append("-x")
    if plan.get("status") == "complete":
        env["RISKRANK_ORDER"] = _jsonl(plan_dir, "order", [t["test"] for t in plan["test_plan"]["tests"]])
        if s.get("run_only_in_budget") is True:
            env["RISKRANK_ONLY"] = _jsonl(plan_dir, "only",
                                          [t["test"] for t in plan["test_plan"]["tests"] if t["in_budget"]])
    return subprocess.call(args, env=env)


def _jsonl(plan_dir: str, name: str, ids: list[str]) -> str:
    path = os.path.join(plan_dir, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(t) + "\n" for t in ids)
    return path


if __name__ == "__main__":
    sys.exit(main())
