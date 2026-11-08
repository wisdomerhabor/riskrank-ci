"""Install a project's test environment as it stood on the date of the commit under test.

    python -m riskrank.test_env riskrank.yml

Used for the Flask fork: it replays upstream history from 2025, and today's packages (pytest,
Werkzeug) break tests written then. Two optional settings, both off by default:

    test_requirements: [requirements/tests.txt]   # the project's own pinned test requirements
    test_env_date: commit                          # resolve every package as of the tested commit's date

The date is the committer date of the upstream commit being tested: HEAD^2 when HEAD is a sync merge
(the replay), otherwise HEAD. Resolution uses uv's --exclude-newer, so nothing published after that
date is installed. pytest-cov is resolved the same way, so it matches that era's pytest. Prints the
date and the command for the build log. Without the settings it installs the project and pytest-cov
normally.
"""

from __future__ import annotations

import subprocess
import sys

import yaml


def tested_commit_date() -> str:
    parents = subprocess.run(["git", "rev-list", "--parents", "-n", "1", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.split()
    rev = "HEAD^2" if len(parents) > 2 else "HEAD"
    return subprocess.run(["git", "log", "-1", "--format=%cI", rev],
                          capture_output=True, text=True, check=True).stdout.strip()


def command(s: dict) -> list[str]:
    reqs = [a for r in s.get("test_requirements", []) for a in ("-r", r)]
    if s.get("test_env_date") == "commit":
        date = tested_commit_date()
        return ["uv", "pip", "install", "--system", "--exclude-newer", date, "-e", ".", *reqs, "pytest-cov"]
    return [sys.executable, "-m", "pip", "install", "-e", ".", *reqs, "pytest", "pytest-cov"]


def main() -> int:
    s = yaml.safe_load(open(sys.argv[1])) or {}
    cmd = command(s)
    if cmd[0] == "uv":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    print("test environment:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
