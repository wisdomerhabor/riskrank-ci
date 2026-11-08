"""Find Flask's bug-fixing commits under the pre-registered rule (config/flask_mining.yaml).

    python -m riskrank.mining.fixes            -> data/mined/flask/bugfixes.json (PySZZ input)

A non-merge commit is a bug fix if it touches a source module and either references an issue
labelled `bug`, or its message contains a fix keyword as a whole word while its subject matches no
exclusion. Where the fix references at least one real issue (not a pull request) the earliest
issue creation date is passed to PySZZ for its issue-date filter.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

import yaml

CONFIG = "config/flask_mining.yaml"
REPO = "data/repos/pallets/flask"
OUT = "data/mined/flask"
SEP = "\x1e"  # record separator in git log output


def load_config(path: str = CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def is_source(path: str, cfg: dict) -> bool:
    """A .py file at any depth under a source root, outside every excluded path."""
    if not path.endswith(".py") or any(path.startswith(p) for p in cfg["exclude_paths"]):
        return False
    return any(path.startswith(root) for root in cfg["source_modules"])


def commits(repo: str, end: str) -> list[dict]:
    """Every non-merge commit up to `end`, with message and the files it changed."""
    fmt = f"{SEP}%H%x00%aI%x00%s%x00%b%x00"
    out = subprocess.run(
        ["git", "-C", repo, "log", end, "--no-merges", f"--format={fmt}", "--name-only"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for rec in out.split(SEP)[1:]:
        h, when, subject, body, files = rec.split("\x00", 4)
        rows.append({"hash": h, "when": when, "subject": subject, "body": body,
                     "files": [f for f in files.strip().splitlines() if f]})
    return rows


def classify(c: dict, cfg: dict, issues: dict) -> dict | None:
    """Return the PySZZ input record if `c` is a bug fix under the rule, else None."""
    rule = cfg["bug_fix_rule"]
    flags = re.IGNORECASE if rule["case_insensitive"] else 0
    src = [f for f in c["files"] if is_source(f, cfg)]
    if not src:
        return None
    message = f"{c['subject']}\n{c['body']}"
    refs = [issues.get(n) for n in re.findall(rule["issue_reference"], message)]
    refs = [r for r in refs if r]
    bug_issue = any(rule["issue_label"] in r["labels"] for r in refs)
    keyword = re.search(rule["keywords"], message, flags) is not None
    excluded = re.search(rule["exclude_subject"], c["subject"], flags) is not None
    if not (bug_issue or (keyword and not excluded)):
        return None
    record = {"repo_name": cfg["repository"], "fix_commit_hash": c["hash"],
              "fix_date": c["when"], "source_files": src,
              "reason": "bug-labelled issue" if bug_issue else "fix keyword"}
    real_issues = [r["created_at"] for r in refs if not r["is_pr"]]
    if real_issues:
        record["earliest_issue_date"] = min(real_issues)
    return record


def main() -> None:
    cfg = load_config()
    with open(cfg["bug_fix_rule"]["issue_cache"]) as f:
        issues = {str(k): v for k, v in json.load(f).items()}
    all_commits = commits(REPO, cfg["history_end_commit"])
    fixes = [r for c in all_commits if (r := classify(c, cfg, issues))]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "bugfixes.json"), "w") as f:
        json.dump(fixes, f, indent=1)
    by_reason = {k: sum(r["reason"] == k for r in fixes) for k in ("bug-labelled issue", "fix keyword")}
    print(f"{len(all_commits)} non-merge commits; {len(fixes)} bug fixes {by_reason}; "
          f"{sum('earliest_issue_date' in r for r in fixes)} with an issue date")


if __name__ == "__main__":
    main()
