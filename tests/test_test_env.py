import subprocess

from riskrank.test_env import command, tested_commit_date


def _repo(tmp_path):
    r = tmp_path / "r"
    run = lambda *a, env=None: subprocess.run(["git", "-C", str(r), *a], check=True, capture_output=True, env=env)
    subprocess.run(["git", "init", "-q", "-b", "main", str(r)], check=True)
    ident = ["-c", "user.email=t@t", "-c", "user.name=t"]
    (r / "a").write_text("1")
    run("add", ".")
    run(*ident, "commit", "-q", "-m", "base", env={"GIT_COMMITTER_DATE": "2025-03-30T10:00:00+00:00", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})
    run("checkout", "-q", "-b", "up")
    (r / "b").write_text("2")
    run("add", ".")
    run(*ident, "commit", "-q", "-m", "upstream", env={"GIT_COMMITTER_DATE": "2025-04-02T09:00:00+00:00", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})
    run("checkout", "-q", "main")
    (r / "c").write_text("3")
    run("add", ".")
    run(*ident, "commit", "-q", "-m", "ours", env={"GIT_COMMITTER_DATE": "2026-11-08T10:00:00+00:00", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})
    run(*ident, "merge", "-q", "--no-ff", "up", "-m", "sync")
    return r


def test_sync_merge_uses_the_upstream_commit_date(tmp_path, monkeypatch):
    monkeypatch.chdir(_repo(tmp_path))
    assert tested_commit_date().startswith("2025-04-02")


def test_command_shapes(tmp_path, monkeypatch):
    monkeypatch.chdir(_repo(tmp_path))
    dated = command({"test_requirements": ["requirements/tests.txt"], "test_env_date": "commit"})
    assert dated[:5] == ["uv", "pip", "install", "--system", "--exclude-newer"] and "requirements/tests.txt" in dated
    assert "--exclude-newer" not in command({})
