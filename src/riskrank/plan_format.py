"""The required plan-file format (NFR7). The plan-ingest component stores a plan only if this returns
no problems; anything else is rejected and the rejection recorded.

A plan is either complete (a ranking and an ordered test plan) or in default order (the hook fell back,
NFR3), which carries a reason and the tests in their usual order.
"""

from __future__ import annotations

PLAN_FORMAT = 1  # the hook writes it; the plan-ingest component rejects any other version

_NUM = (int, float)


def problems(plan: object) -> list[str]:
    if not isinstance(plan, dict):
        return ["plan is not a JSON object"]
    out: list[str] = []
    if plan.get("format") != PLAN_FORMAT:
        out.append(f"format is {plan.get('format')!r}, expected {PLAN_FORMAT}")
    status = plan.get("status")
    if status == "default order":
        if not isinstance(plan.get("reason"), str):
            out.append("default-order plan has no reason")
        if not _list_of_dicts(plan.get("tests"), {"test": str, "position": int}):
            out.append("default-order plan has no valid test list")
        return out
    if status not in ("complete", "partial"):  # partial: module plan kept, commit level not assessed
        return out + [f"unknown status {status!r}"]
    for key, kind in [("settings", dict), ("push", dict), ("ranking", list), ("test_plan", dict),
                      ("commit_level", dict), ("escalated", bool), ("modules_touched", int)]:
        if not isinstance(plan.get(key), kind):
            out.append(f"{key} missing or not {kind.__name__}")
    if out:
        return out
    if not all(isinstance(plan["push"].get(k), str) for k in ("before", "after")):
        out.append("push.before / push.after missing")
    row = {"rank": int, "module": str, "p": _NUM, "raw": _NUM, "size": _NUM, "in_budget": bool,
           "flagged": bool, "reason_codes": list}
    if not _list_of_dicts(plan["ranking"], row):
        out.append("ranking rows do not match the required fields")
    tp = plan["test_plan"]
    if not _list_of_dicts(tp.get("tests"), {"test": str, "position": int, "in_budget": bool, "group": str}):
        out.append("test_plan.tests do not match the required fields")
    if not isinstance(tp.get("stats"), dict) or "tests_total" not in tp["stats"]:
        out.append("test_plan.stats missing")
    s = plan["settings"]
    for k in ("budget", "budget_unit", "ranking_key", "risk_threshold"):
        if k not in s:
            out.append(f"settings.{k} missing")
    return out


def _list_of_dicts(v: object, fields: dict) -> bool:
    if not isinstance(v, list):
        return False
    for item in v:
        if not isinstance(item, dict):
            return False
        for k, t in fields.items():
            if not isinstance(item.get(k), t) or (t is int and isinstance(item.get(k), bool)):
                return False
    return True
