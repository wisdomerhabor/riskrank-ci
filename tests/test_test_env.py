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
    (r / "riskrank.yml").write_text("budget: 20\n")  # our commits only ever touch RiskRank files
    run("add", ".")
    run(*ident, "commit", "-q", "-m", "ours", env={"GIT_COMMITTER_DATE": "2026-11-08T10:00:00+00:00", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})
    run(*ident, "merge", "-q", "--no-ff", "up", "-m", "sync")
    return r


def test_sync_merge_uses_the_upstream_commit_date(tmp_path, monkeypatch):
    monkeypatch.chdir(_repo(tmp_path))
    assert tested_commit_date().startswith("2025-04-02")


def test_command_shapes(tmp_path, monkeypatch):
    r = _repo(tmp_path)
    (r / "requirements").mkdir()
    (r / "requirements" / "tests.txt").write_text("pytest\n")
    monkeypatch.chdir(r)
    dated = command({"test_requirements": ["requirements/tests.txt"], "test_env_date": "commit"})
    assert dated[:5] == ["uv", "pip", "install", "--system", "--exclude-newer"] and "requirements/tests.txt" in dated
    assert "--exclude-newer" not in command({})


def test_our_own_settings_commit_on_top_is_ignored(tmp_path, monkeypatch):
    r = _repo(tmp_path)
    (r / "riskrank.yml").write_text("budget: 30\n")
    subprocess.run(["git", "-C", str(r), "add", "."], check=True)
    subprocess.run(["git", "-C", str(r), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "settings"],
                   check=True, env={"GIT_COMMITTER_DATE": "2026-11-09T10:00:00+00:00", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})
    monkeypatch.chdir(r)
    # both of our commits touch only riskrank.yml, so the upstream commit still sets the date
    assert tested_commit_date().startswith("2025-04-02")


def test_newest_is_by_time_not_by_string(tmp_path, monkeypatch):
    r = tmp_path / "z"
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    for i, d in enumerate(["2025-03-30T13:00:00-07:00", "2025-03-30T21:30:00+01:00"]):  # 20:00Z then 20:30Z
        (r / f"f{i}").write_text("x")
        subprocess.run(["git", "-C", str(r), "add", "."], check=True)
        subprocess.run(["git", "-C", str(r), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", str(i)],
                       check=True, env={"GIT_COMMITTER_DATE": d, "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})
    monkeypatch.chdir(r)
    assert tested_commit_date() == "2025-03-30T21:30:00+01:00"  # a string max would pick the -07:00 one


def test_replay_workflow_commit_is_ignored_too(tmp_path, monkeypatch):
    r = _repo(tmp_path)
    (r / ".github" / "workflows").mkdir(parents=True)
    (r / ".github" / "workflows" / "riskrank-replay.yml").write_text("name: x\n")
    subprocess.run(["git", "-C", str(r), "add", "."], check=True)
    subprocess.run(["git", "-C", str(r), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "replay"],
                   check=True, env={"GIT_COMMITTER_DATE": "2026-11-09T10:00:00+00:00", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})
    monkeypatch.chdir(r)
    assert tested_commit_date().startswith("2025-04-02")


def test_requirement_files_or_dependency_group_whichever_the_commit_has(tmp_path, monkeypatch):
    monkeypatch.chdir(_repo(tmp_path))
    s = {"test_requirements": ["requirements/tests.txt"], "test_dependency_group": "tests"}
    cmd = command(s)
    assert "-r" not in cmd and "--group" not in cmd           # neither exists here
    (tmp_path / "r" / "pyproject.toml").write_text('[dependency-groups]\ntests = ["pytest"]\n')
    assert command(s)[-4:-2] == ["--group", "tests"]
    (tmp_path / "r" / "requirements").mkdir()
    (tmp_path / "r" / "requirements" / "tests.txt").write_text("pytest\n")
    assert "requirements/tests.txt" in command(s)
