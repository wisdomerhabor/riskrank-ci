"""Per-test coverage map from a .coverage file with node-id contexts (Python 3.6+, inside the anchor's env).

    python map_from_coverage.py <.coverage> <repo root> <out.json> <root>[,<root>...] <exclude>[,<exclude>...]

Writes {node id: [module paths relative to the repo]} for .py files under a root and outside every exclude.
"""
import json
import os
import sys
from collections import defaultdict

from coverage import CoverageData

data_file, root, out, roots, excludes = sys.argv[1:6]
roots, excludes = roots.split(","), excludes.split(",")
root = os.path.realpath(root)
data = CoverageData(basename=data_file)
data.read()
tests = defaultdict(set)
for path in data.measured_files():
    rel = os.path.relpath(os.path.realpath(path), root)
    if rel.startswith("..") or not rel.endswith(".py"):
        continue
    if not any(rel.startswith(r) for r in roots) or any(rel.startswith(e) for e in excludes):
        continue
    for contexts in (data.contexts_by_lineno(path) or {}).values():
        for ctx in contexts:
            if ctx:
                tests[ctx].add(rel)
with open(out, "w") as f:
    json.dump({t: sorted(m) for t, m in tests.items()}, f)
