"""Scale check for the two 5,000-item time targets, run on the CI runner itself (NFR2, NFR18).

    python -m riskrank.scale_check [--repeats 5] [--out scale-check.json]

The unit tests time the same work on the workstation; the targets name the runner, so this runs the same
two jobs there and records the runner's class, processor and memory beside the times (NFR2).
- ranking: 5,000 artefacts ranked under both keys and marked under both budgets (target 60 s);
- ordering: a 5,000-test plan built from a 100-module ranking and a per-test coverage map (target 30 s).
The inputs are generated, with a fixed seed: this measures time, never prediction quality.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd

from riskrank.budget import mark, rank, recall_artefacts, recall_lines
from riskrank.testplan import build_plan

TARGETS = {"ranking_5000_artefacts": 60.0, "ordering_5000_tests": 30.0}


def ranking_seconds(n: int = 5000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"id": [f"m{i:05d}" for i in range(n)], "p": rng.random(n), "raw": rng.normal(size=n),
                          "size": rng.integers(0, 2000, n), "label": rng.random(n) < 0.1})
    t0 = time.perf_counter()
    for key in ("default", "effort"):
        r = rank(frame, key)
        for unit in ("artefacts", "lines"):
            mark(r, unit, 0.2, key)
            (recall_artefacts if unit == "artefacts" else recall_lines)(r)
    return time.perf_counter() - t0


def ordering_seconds(n: int = 5000) -> float:
    ranked = [(f"m{i:03d}.py", i < 20) for i in range(100)]
    cov = {f"t{i:05d}": {f"m{(i * 7) % 130:03d}.py", f"m{(i * 13) % 130:03d}.py"} for i in range(n)}
    t0 = time.perf_counter()
    build_plan(ranked, set(), cov, sorted(cov))
    return time.perf_counter() - t0


def machine() -> dict:
    mem = None
    try:
        with open("/proc/meminfo") as f:
            mem = next((int(line.split()[1]) // 1024 for line in f if line.startswith("MemTotal")), None)
    except OSError:
        pass
    cpu = platform.processor() or None
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), cpu)
    except OSError:
        pass
    return {"runner_name": os.environ.get("RUNNER_NAME"), "runner_os": os.environ.get("RUNNER_OS"),
            "runner_arch": os.environ.get("RUNNER_ARCH"), "image": os.environ.get("ImageOS"),
            "processor": cpu, "cpus": os.cpu_count(), "memory_mb": mem, "python": platform.python_version()}


def run(repeats: int = 5) -> dict:
    times = {"ranking_5000_artefacts": [ranking_seconds() for _ in range(repeats)],
             "ordering_5000_tests": [ordering_seconds() for _ in range(repeats)]}
    return {"machine": machine(), "repeats": repeats,
            "results": {k: {"max_seconds": max(v), "median_seconds": float(np.median(v)), "target_seconds": TARGETS[k],
                            "within_target": max(v) < TARGETS[k]} for k, v in times.items()}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="scale-check.json")
    a = ap.parse_args(argv)
    out = run(a.repeats)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
