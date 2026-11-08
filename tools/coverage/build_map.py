"""Build the per-test coverage map (FR17) from a coverage.py data file with per-test contexts.

Run with the deployment repository's own Python (it needs only `coverage`):
    python build_map.py --data .coverage --repo . --durations durations.json --out coverage-map.json

A test exercises a module when any of its contexts (setup, run or teardown) executed a line in it.
Output: {"meta": {commit, collected_at, durations, tests, modules, build_seconds}, "tests": {id: [modules]}}
"""

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone

from coverage import CoverageData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".coverage")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--durations", default=None)
    ap.add_argument("--out", default="coverage-map.json")
    a = ap.parse_args()
    t0 = time.time()
    root = os.path.realpath(a.repo)
    data = CoverageData(basename=a.data)
    data.read()
    tests = defaultdict(set)
    for path in data.measured_files():
        rel = os.path.relpath(os.path.realpath(path), root)
        if rel.startswith(".."):  # e.g. an editable install resolving outside the repository
            continue
        for contexts in (data.contexts_by_lineno(path) or {}).values():
            for ctx in contexts:
                if ctx:
                    tests[ctx.split("|")[0]].add(rel)
    commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    durations = json.load(open(a.durations)) if a.durations and os.path.exists(a.durations) else {}
    out = {"meta": {"commit": commit, "collected_at": datetime.now(timezone.utc).isoformat(),
                    "durations": durations, "tests": len(tests),
                    "modules": len(set().union(*tests.values())) if tests else 0,
                    "build_seconds": round(time.time() - t0, 2)},
           "tests": {t: sorted(m) for t, m in sorted(tests.items())}}
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"{len(tests)} tests, {out['meta']['modules']} modules, built in {out['meta']['build_seconds']}s "
          f"-> {a.out} ({os.path.getsize(a.out)} bytes)")


if __name__ == "__main__":
    main()
