"""Drive the Flask fork's batch replay from the workstation (section 3.7.8): one workflow_dispatch per
batch, in upstream order, waiting for the merge and for the scoring run each push triggers.

    python tools/replay_batches.py --log runs/replay-rehearsal.jsonl            # every remaining batch
    python tools/replay_batches.py --log runs/replay-counted.jsonl --stop-after 1

A batch is everything upstream up to one first-parent commit of pallets/flask main that changes the
source package, after the fork commit (34 batches). Batches already in the fork's main are skipped.
Stops at the first failed replay or scoring run, so it can be fixed before going on. Uses `gh`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone

FORK = "wisdomerhabor/flask"
UPSTREAM_CLONE = "data/repos/pallets/flask"
FORK_COMMIT = "b78b5a21"


def sh(*args: str, cwd: str | None = None, tries: int = 4) -> str:
    """Run a command; GitHub API calls through `gh` can fail transiently, so retry with a pause."""
    for attempt in range(1, tries + 1):
        p = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
        if p.returncode == 0:
            return p.stdout.strip()
        print(f"attempt {attempt} failed: {' '.join(args[:3])}: {p.stderr.strip()[:300]}", flush=True)
        time.sleep(20 * attempt)
    raise subprocess.CalledProcessError(p.returncode, args, p.stdout, p.stderr)


def batches() -> list[str]:
    sh("git", "-C", UPSTREAM_CLONE, "fetch", "-q", "origin", "main")
    out = sh("git", "-C", UPSTREAM_CLONE, "log", "--first-parent", "--reverse", "--format=%H",
             f"{FORK_COMMIT}..origin/main", "--", "src/flask")
    return out.split()


def fork_main() -> str:
    return sh("gh", "api", f"repos/{FORK}/commits/main", "-q", ".sha")


def in_fork(sha: str) -> bool:
    st = sh("gh", "api", f"repos/{FORK}/compare/{sha}...main", "-q", ".status")
    return st in ("ahead", "identical")


def wait_run(workflow: str, *, after: str | None = None, head_sha: str | None = None, timeout: int = 3600) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        runs = json.loads(sh("gh", "run", "list", "-R", FORK, "-w", workflow, "-L", "10", "--json",
                             "databaseId,status,conclusion,headSha,createdAt"))
        for r in runs:
            if (after is None or r["createdAt"] > after) and (head_sha is None or r["headSha"] == head_sha):
                if r["status"] == "completed":
                    return r
                break
        time.sleep(15)
    raise TimeoutError(f"{workflow} run did not finish")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--stop-after", type=int, default=0, help="number of batches to replay; 0 = all remaining")
    a = ap.parse_args()
    todo = batches()
    print(f"{len(todo)} batches in upstream history", flush=True)
    done = 0
    for i, sha in enumerate(todo, 1):
        if in_fork(sha):
            continue
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        sh("gh", "workflow", "run", "riskrank-replay.yml", "-R", FORK, "--ref", "main", "-f", f"upstream_commit={sha}")
        time.sleep(5)
        rep = wait_run("riskrank-replay", after=started)
        rec = {"batch": i, "upstream": sha, "replay_run": rep["databaseId"], "replay": rep["conclusion"]}
        if rep["conclusion"] == "success":
            head = fork_main()
            score = wait_run("riskrank", head_sha=head)
            rec |= {"fork_head": head, "scoring_run": score["databaseId"], "scoring": score["conclusion"]}
        rec["finished"] = datetime.now(timezone.utc).isoformat()
        with open(a.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if rec.get("scoring") != "success":
            raise SystemExit(f"batch {i} failed; stopping")
        done += 1
        if a.stop_after and done >= a.stop_after:
            break


if __name__ == "__main__":
    main()
