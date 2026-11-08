"""pytest plugin for the ordered test plan (FR18). Avoids passing test ids on the command line,
because `pytest --collect-only -q` escapes non-ASCII characters in ids and an escaped id does not
select its test.

  RISKRANK_COLLECT=<file>   write every collected test id, exactly as pytest holds it (JSON lines)
  RISKRANK_ORDER=<file>     reorder collected tests to the plan's order (JSON lines of ids); any test
                            the plan does not know is appended in ascending id order, never dropped
  RISKRANK_ONLY=<file>      with RISKRANK_ORDER: deselect every test not listed (run_only_in_budget)
"""
import json
import os


def _ids(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pytest_collection_modifyitems(session, config, items):
    collect = os.environ.get("RISKRANK_COLLECT")
    if collect:
        with open(collect, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it.nodeid) + "\n")
    order = os.environ.get("RISKRANK_ORDER")
    if order:
        pos = {t: i for i, t in enumerate(_ids(order))}
        known = sorted((it for it in items if it.nodeid in pos), key=lambda it: pos[it.nodeid])
        rest = sorted((it for it in items if it.nodeid not in pos), key=lambda it: it.nodeid)
        only = os.environ.get("RISKRANK_ONLY")
        if only:
            keep = set(_ids(only))
            deselected = [it for it in known + rest if it.nodeid not in keep]
            known = [it for it in known if it.nodeid in keep]
            rest = []
            config.hook.pytest_deselected(items=deselected)
        items[:] = known + rest
