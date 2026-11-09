"""Features must not see a commit authored before the prediction point on a branch merged after it."""
import os
import re
import subprocess

import pandas as pd

from riskrank.jit_features import History, compute
from riskrank.mining.snapshots import history, process_window


def _commit(repo, msg, when, author, files):
    for name, text in files.items():
        (repo / name).write_text(text)
    env = os.environ | {"GIT_AUTHOR_DATE": f"{when} +0000", "GIT_COMMITTER_DATE": f"{when} +0000"}
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "-c", f"user.email={author}", "-c", "user.name=x", "commit", "-q",
                    "-m", msg], check=True, env=env)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def _repo(tmp_path):
    """main: c1 (Jan) -> c2 (Mar) -> merge (May); side branch from c1: s1 (Feb), merged in May."""
    repo = tmp_path / "r"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)  # noqa: E731
    c1 = _commit(repo, "c1", "2020-01-01T00:00:00", "alice@x", {"a.py": "1\n"})
    git("checkout", "-q", "-b", "side")
    s1 = _commit(repo, "fix s1", "2020-02-01T00:00:00", "bob@x", {"a.py": "1\n2\n"})
    git("checkout", "-q", "main")
    c2 = _commit(repo, "c2", "2020-03-01T00:00:00", "alice@x", {"b.py": "x\n", "a.py": "1\n3\n"})
    env = os.environ | {"GIT_AUTHOR_DATE": "2020-05-01T00:00:00 +0000", "GIT_COMMITTER_DATE": "2020-05-01T00:00:00 +0000"}
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=alice@x", "-c", "user.name=x", "merge", "-q",
                    "--no-edit", "-X", "theirs", "side"], check=True, env=env, capture_output=True)
    return str(repo), c1, s1, c2


def test_commit_features_exclude_a_later_merged_branch(tmp_path):
    repo, _, s1, c2 = _repo(tmp_path)
    leaky = History(repo, "HEAD")  # the whole history filtered by author time only
    assert compute(leaky.by_sha[c2], leaky)["nuc"] == 1.0  # (c1 + s1) / 2 files: s1 leaks in
    safe = History(repo, "HEAD", ancestors_only=True)
    hook = History(repo, c2)  # what a CI hook reading history from c2 sees
    got, want = compute(safe.by_sha[c2], safe), compute(hook.by_sha[c2], hook)
    assert got == want and got["nuc"] == 0.5  # c1 only
    assert s1 in safe.by_sha  # s1 still has its own row, computed from its own ancestors


def test_snapshot_process_metrics_read_only_the_trees_history(tmp_path):
    repo, _, _, c2 = _repo(tmp_path)
    s = pd.Timestamp("2020-04-01", tz="UTC")  # after s1 was authored and committed, before it was merged
    kw = re.compile(r"\bfix\b", re.IGNORECASE)
    lo = s - pd.DateOffset(months=12)
    full = process_window(history("HEAD", repo=repo), ["a.py"], lo, s, kw)["a.py"]
    tree = process_window(history(c2, repo=repo), ["a.py"], lo, s, kw)["a.py"]
    assert full["changes"] == 3 and full["prior_fixes"] == 1  # the leak the corpus had
    assert tree["changes"] == 2 and tree["prior_fixes"] == 0
