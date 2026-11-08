"""Install a project's test environment as it stood on the date of the commit under test.

    python -m riskrank.test_env riskrank.yml

Used for the Flask fork: it replays upstream history from 2025, and today's packages (pytest,
Werkzeug) break tests written then. Two optional settings, both off by default:

    test_requirements: [requirements/tests.txt]   # the project's own pinned test requirements, used if present
    test_dependency_group: tests                   # a PEP 735 group in pyproject.toml, used if defined
    test_env_date: commit                          # resolve every package as of the tested commit's date

Both test-dependency settings are read at the tested commit, so one settings file covers a project that
moves from requirement files to dependency groups during the replay (Flask did, in May 2025).

The date is the committer date of the newest upstream commit in HEAD's history: the newest non-merge
commit that changes anything other than RiskRank's own files (riskrank.yml and the RiskRank workflows,
.github/workflows/riskrank*). That is the synced upstream commit after a replay merge, and the fork-point commit before
any replay, even when HEAD is one of our own settings commits. Resolution uses uv's --exclude-newer, so nothing published after that
date is installed. pytest-cov is resolved the same way, so it matches that era's pytest. Prints the
date and the command for the build log. Without the settings it installs the project and pytest-cov
normally.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib

import yaml


OURS = [":(exclude)riskrank.yml", ":(exclude,glob).github/workflows/riskrank*"]


def tested_commit_date() -> str:
    out = subprocess.run(["git", "log", "--no-merges", "--format=%ct %cI", "HEAD", "--", ".",
                          *OURS],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    # newest by numeric timestamp: ISO strings with different UTC offsets do not sort as times
    return max((int(t), iso) for t, iso in (line.split(" ", 1) for line in out))[1]


def has_group(name: str, pyproject: str = "pyproject.toml") -> bool:
    try:
        with open(pyproject, "rb") as f:
            return name in tomllib.load(f).get("dependency-groups", {})
    except (OSError, tomllib.TOMLDecodeError):
        return False


def command(s: dict) -> list[str]:
    reqs = [a for r in s.get("test_requirements", []) if os.path.exists(r) for a in ("-r", r)]
    group = s.get("test_dependency_group")
    if group and has_group(group):
        reqs += ["--group", group]
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
