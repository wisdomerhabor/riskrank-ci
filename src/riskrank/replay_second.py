"""Second bounded check (section 3.10.3): early fault detection with an executed, back-ported test, on Flask.
Evaluation only; never part of the deployed pipeline. Rules: config/flask_mining.yaml, second_check
(pre-registered 9 Nov 2026, before any historical build was attempted).

    python -m riskrank.replay_second screen            # eligibility of every candidate; resumable
    python -m riskrank.replay_second rank              # coverage, rankings and APFD/APFDc for reproduced defects

screen  For each candidate fix (on or after the separation date): anchor = its parent; build the anchor's dated
        environment; run the full suite there (must pass; gives every duration); find the single added test; run it
        at the fix (must pass); back-port it to the anchor and run it three times (must fail by assertion every time).
        The first failing step is the exclusion category, in the pre-registered order. -> runs/second_check/screen.jsonl
rank    For each reproduced defect: per-test coverage at the anchor with the back-ported test included; rank every
        source module with the first check's validation model (rebuilt exactly, not re-tuned), largest-first and
        recency; turn each ranking into a test order with the system's own build_plan; record the failing test's
        position, APFD, APFDc, its tie group, the pessimistic bound and the tests that precede it under any ranking.
        -> runs/second_check.json
Needs uv on PATH (it provides the historical Pythons). Worktrees and environments live under data/work/.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import json
import os
import re
import shutil
import statistics
import subprocess
import textwrap
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from riskrank.budget import rank
from riskrank.deploy_module import FEATURES
from riskrank.hook import module_features
from riskrank.mining.fixes import REPO, is_source, load_config
from riskrank.mining.snapshots import history, last_change
from riskrank.models import Tuned, fit_final
from riskrank.testplan import build_plan

OUT = "runs/second_check"
WORK = os.path.abspath("data/work/second_check")
TOOLS = os.path.abspath("tools/second_check")
PYTHONS = [("3.6", "2016-12-23"), ("3.7", "2018-06-27"), ("3.8", "2019-10-14"), ("3.9", "2020-10-05"),
           ("3.10", "2021-10-04"), ("3.11", "2022-10-24"), ("3.12", "2023-10-02"), ("3.13", "2024-10-07")]
EXCLUSIONS = ["build_failure", "suite_fails_at_parent", "not_exactly_one_test_added", "added_test_fails_at_fix",
              "backport_error_or_not_applicable", "backport_passes_all_three", "flaky"]
ASSERTION = {"AssertionError", "Failed"}
TIMEOUT = 1800
BUILD_TOOLS_DATE = "2026-11-09"
MIN_BUILD_SETUPTOOLS_DATE = "2019-02-28"  # uv needs setuptools >= 40.8 (5 Feb 2019) to build a setup.py project  # the project's editable build backend only (setuptools, wheel); see build_env

PROJECT = "flask"  # the distribution re-installed editable (Flask has kept this name throughout)


def git(*args: str, repo: str = REPO) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


def sh(cmd: list[str], cwd: str | None = None, env: dict | None = None, timeout: int = TIMEOUT):
    try:
        return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(cmd, 124, e.stdout or "", f"timeout after {timeout}s")


# ------------------------------------------------------------------ pure rules (unit-tested)
def python_for(date: str) -> str:
    """The newest CPython minor release published on or before the anchor's committer date."""
    day = pd.Timestamp(date).tz_convert("UTC").strftime("%Y-%m-%d")
    out = None
    for v, released in PYTHONS:
        if released <= day:
            out = v
    if out is None:
        raise ValueError(f"no listed Python before {day}")
    return out


def test_names(src: str) -> dict[tuple[str | None, str], ast.AST]:
    """pytest's default collection: functions starting 'test', at top level or in classes starting 'Test'."""
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            out[(None, node.name)] = node
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith("test"):
                    out[(node.name, m.name)] = m
    return out


def added_tests(before: dict[str, str | None], after: dict[str, str]) -> list[tuple[str, str | None, str]]:
    """Tests present after and absent before, by (file, class, name); `before[file]` is None for a new file."""
    out = []
    for f, src in sorted(after.items()):
        old = set(test_names(before[f])) if before.get(f) else set()
        out += [(f, c, n) for c, n in sorted(set(test_names(src)) - old, key=lambda k: (k[0] or "", k[1]))]
    return out


def backport_source(anchor_src: str | None, fix_src: str, cls: str | None, name: str) -> str | None:
    """The anchor file with the added test inserted, or None when it cannot be applied."""
    if anchor_src is None:
        return fix_src  # the test file is new in the fix: copied whole
    node = test_names(fix_src)[(cls, name)]
    lines = fix_src.splitlines(keepends=True)
    start = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
    seg = textwrap.dedent("".join(lines[start: node.end_lineno]))
    if cls is None:
        new = anchor_src.rstrip("\n") + "\n\n\n" + seg
    else:
        tree = ast.parse(anchor_src)
        klass = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls), None)
        if klass is None:
            return None
        al = anchor_src.splitlines(keepends=True)
        first = al[klass.body[0].lineno - 1]
        indent = first[: len(first) - len(first.lstrip())]
        new = "".join(al[: klass.end_lineno]).rstrip("\n") + "\n\n" + textwrap.indent(seg, indent) + "".join(al[klass.end_lineno:])
    try:
        ast.parse(new)
    except SyntaxError:
        return None
    return new


def run_status(results: dict, nodes: list[str]) -> str:
    """One run of the back-ported test: 'error', 'fail' (an assertion failure) or 'pass'."""
    mine = {n: r for n, r in results.items() if n in nodes or any(n.startswith(x + "[") for x in nodes)}
    if not mine:
        return "error"  # not collected
    if any(r["outcome"] == "error" or (r["outcome"] == "failed" and r["exc"] not in ASSERTION) for r in mine.values()):
        return "error"
    return "fail" if any(r["outcome"] == "failed" for r in mine.values()) else "pass"


def apfd(tf: int, n: int) -> float:
    return 1 - tf / n + 1 / (2 * n)


def apfdc(order: list[str], durations: dict[str, float], failing: str) -> float:
    total = sum(durations.get(t, 0.0) for t in order)
    i = order.index(failing)
    before = sum(durations.get(t, 0.0) for t in order[:i])
    return (total - before - durations.get(failing, 0.0) / 2) / total if total else float("nan")


def ordering_result(ranking: list[str], coverage: dict[str, set[str]], current: list[str],
                    durations: dict[str, float], failing_nodes: list[str]) -> dict:
    """Position, APFD and APFDc of the earliest failing case under one module ranking, plus its tie group and the
    pessimistic bound (the failing case treated as covering only its lowest-ranked module)."""
    def order_for(cov):
        plan = build_plan([(m, True) for m in ranking], set(), cov, current)
        return [i.test for i in plan.items]

    order = order_for(coverage)
    pos = {t: i + 1 for i, t in enumerate(order)}
    first = min(failing_nodes, key=lambda t: pos[t])
    rk = {m: i for i, m in enumerate(ranking)}
    key = lambda t: tuple(sorted(rk[m] for m in coverage.get(t, ()) if m in rk))  # noqa: E731
    mine = key(first)
    tie = [t for t in order if t in coverage and key(t) == mine] if mine else [t for t in order if not key(t)]
    pess_cov = dict(coverage)
    for t in failing_nodes:
        ranked = [m for m in coverage.get(t, ()) if m in rk]
        if ranked:
            pess_cov[t] = {max(ranked, key=lambda m: rk[m])}
    pess = order_for(pess_cov)
    ppos = {t: i + 1 for i, t in enumerate(pess)}
    pfirst = min(failing_nodes, key=lambda t: ppos[t])
    n = len(order)
    return {"tf": pos[first], "n": n, "failing_case": first, "apfd": apfd(pos[first], n),
            "apfdc": apfdc(order, durations, first), "tie_group": len(tie), "tie_decided": len(tie) > 1,
            "pessimistic": {"tf": ppos[pfirst], "apfd": apfd(ppos[pfirst], n), "apfdc": apfdc(pess, durations, pfirst)}}


def must_precede(coverage: dict[str, set[str]], scoreable: set[str], node: str) -> int:
    """Tests that exercise every ranked module the failing case exercises and at least one more: always ahead."""
    mine = coverage.get(node, set()) & scoreable
    return sum(1 for t, ms in coverage.items() if t != node and (ms & scoreable) > mine) if mine else 0


# ------------------------------------------------------------------ environments and runs
def tox_deps(path: str) -> list[str]:
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(path)
    raw = cp.get("testenv", "deps", fallback="")
    return [x.strip() for x in raw.splitlines() if x.strip() and ":" not in x and not x.strip().startswith(("http", "-"))]


ARCHIVE = re.compile(r"github\.com/([\w.-]+/[\w.-]+)/archive/(?:refs/heads/)?([\w.-]+)\.(?:tar\.gz|zip)")
_SHAS: dict = {}


def tox_archives(path: str) -> list[tuple[str, str]]:
    """(owner/repo, branch) for every branch archive tox.ini's [testenv] installs (Pallets projects in development)."""
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(path)
    return ARCHIVE.findall(cp.get("testenv", "deps", fallback=""))


def sha_at(repo: str, branch: str, date: str) -> str:
    """The last commit on the branch at or before the date (a renamed branch is followed via the default one)."""
    date = pd.Timestamp(date).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")  # a "+hh:mm" offset breaks the query
    k = (repo, branch, date)
    if k not in _SHAS:
        import urllib.request
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        for b in (branch, "main", "master"):
            req = urllib.request.Request(f"https://api.github.com/repos/{repo}/commits?sha={b}&until={date}&per_page=1",
                                         headers={"Accept": "application/vnd.github+json", "User-Agent": "riskrank"}
                                         | ({"Authorization": f"Bearer {token}"} if token else {}))
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    got = json.load(r)
            except Exception:  # a renamed or missing branch: try the next name
                continue
            if got:
                _SHAS[k] = got[0]["sha"]
                break
        else:
            raise RuntimeError(f"no commit of {repo}@{branch} before {date}")
    return _SHAS[k]


def has_group(pyproject: str, name: str) -> bool:
    import tomllib
    try:
        return name in tomllib.load(open(pyproject, "rb")).get("dependency-groups", {})
    except (OSError, tomllib.TOMLDecodeError):
        return False


def sibling_source(repo: str, sha: str) -> str:
    """What uv installs for a pinned sibling archive: the git URL, unless that commit's setup.cfg stamps a dev tag on
    the version ([egg_info] tag_build/tag_date). Those tags made versions such as "1.0-dev-20261109" (itsdangerous,
    2017-2018: tag_build = 'dev' in quotes, tag_date = the BUILD's date, not the anchor's). pip of the time accepted
    them as legacy versions; uv rejects them. Then the commit is checked out locally with only those two options removed
    from setup.cfg, so the installed code is identical and only the version label loses its tag (fourth amendment,
    9 Nov 2026)."""
    url = f"git+https://github.com/{repo}@{sha}"
    d = os.path.join(WORK, "siblings", f"{repo.replace('/', '-')}-{sha[:10]}")
    if not os.path.isdir(d):
        tmp = d + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        for cmd in (["git", "init", "-q", tmp], ["git", "-C", tmp, "fetch", "-q", "--depth", "1",
                    f"https://github.com/{repo}", sha], ["git", "-C", tmp, "checkout", "-q", "FETCH_HEAD"]):
            r = sh(cmd)
            if r.returncode:
                shutil.rmtree(tmp, ignore_errors=True)
                return url  # cannot inspect it: uv installs it as before, and any version error is a build failure
        os.replace(tmp, d)
    cfg = os.path.join(d, "setup.cfg")
    sh(["git", "-C", d, "checkout", "-q", "--", "."])  # back to the commit's own files before deciding (reruns)
    if not os.path.exists(cfg):
        return url
    text, dropped = strip_dev_tag(open(cfg).read())
    if not dropped:
        return url
    open(cfg, "w").write(text)
    return d


def strip_dev_tag(cfg: str) -> tuple[str, bool]:
    """setup.cfg text without [egg_info] tag_build and tag_date; whether anything was removed."""
    out, section, dropped = [], "", False
    for ln in cfg.splitlines(keepends=True):
        s = ln.strip()
        if s.startswith("[") and s.endswith("]"):
            section = s[1:-1].strip()
        elif section == "egg_info" and re.match(r"(tag_build|tag_date)\s*[=:]", s):
            dropped = True
            continue
        out.append(ln)
    return "".join(out), dropped


def develop_link(wt: str, vpy: str) -> tuple[bool, str]:
    """The editable install as `setup.py develop` made it before PEP 660: the project's installed package directory
    is removed (its metadata stays, so entry points and version lookups still resolve) and a .pth file puts the
    working tree's source directory on sys.path, so the tests import and measure the checked-out code. Used on
    Python 3.6 only, where setuptools 64+ (the first with build_editable) cannot run (third amendment, 9 Nov 2026)."""
    site = sh([vpy, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"]).stdout.strip()
    src = os.path.join(wt, "src") if os.path.isdir(os.path.join(wt, "src", PROJECT)) else wt
    if not os.path.isdir(os.path.join(src, PROJECT)) or not os.path.isdir(site):
        return False, f"no {PROJECT} package under {src} or no site-packages ({site!r})"
    installed = os.path.join(site, PROJECT)
    if os.path.isdir(installed):
        shutil.rmtree(installed)
    with open(os.path.join(site, f"__editable__.{PROJECT}.pth"), "w") as fh:
        fh.write(os.path.realpath(src) + "\n")
    r = sh([vpy, "-c", f"import {PROJECT}, os; print(os.path.realpath({PROJECT}.__file__))"], cwd="/")
    where = r.stdout.strip()
    if r.returncode or not where.startswith(os.path.realpath(src)):
        return False, f"{PROJECT} imports from {where or r.stderr.strip()[-300:]}, not {src}"
    return True, ""


def build_env(wt: str, date: str, venv: str) -> tuple[bool, str]:
    py = python_for(date)
    r = sh(["uv", "venv", "-q", "--python", py, venv])
    if r.returncode:
        return False, f"python {py} unavailable: {r.stderr.strip()[-300:]}"
    vpy = os.path.join(venv, "bin", "python")
    # uv byte-compiles through py_compile.PycInvalidationMode, which Python 3.6 lacks: there the environment is
    # compiled with the standard library's compileall once everything is installed (the same effect, as pip did).
    compile_flag = [] if py == "3.6" else ["--compile-bytecode"]
    if os.path.exists(os.path.join(wt, "requirements", "tests.txt")):
        deps, source = ["-r", "requirements/tests.txt"], "requirements/tests.txt"
    elif has_group(os.path.join(wt, "pyproject.toml"), "tests"):
        deps, source = ["--group", "tests"], "dependency group tests"
    else:
        deps, source = tox_deps(os.path.join(wt, "tox.ini")), "tox.ini [testenv] deps"
    # Everything in the environment is dated, the project included (installed normally, which old setuptools can
    # build). The project is then re-installed editable with no dependencies: only its isolated build backend may
    # be newer (BUILD_TOOLS_DATE), because setuptools before 64 (Aug 2022) cannot build an editable install as uv
    # requires it (PEP 660). Amended 9 Nov 2026 before any defect was reproduced; see config second_check.
    # --compile-bytecode as pip does: old packages compiled at import under the project's "warnings are errors"
    # pytest setting would otherwise fail on deprecation warnings (blinker 1.4's escape sequences, 2021 anchors).
    # Branch archives that tox.ini installed (the project's sibling libraries in development) are pinned to their
    # last commit at the anchor's date; setuptools is installed dated, as every virtualenv of the time had it.
    shas = [(r, sha_at(r, br, date)) for r, br in tox_archives(os.path.join(wt, "tox.ini"))]
    pinned = [sibling_source(r, s) for r, s in shas]
    if pinned:
        source += " + " + ", ".join(f"{r.split('/')[-1]}@{s}" + ("" if p.startswith("git+") else " (dev tag removed)")
                                    for (r, s), p in zip(shas, pinned))
    # uv builds a project that has no pyproject.toml with setuptools 40.8 or later (5 Feb 2019), so for an older
    # anchor the build backend is dated 28 Feb 2019 instead: still old enough for 2017-era setup.py files (MarkupSafe
    # 1.0 imports setuptools.Feature, removed in Jan 2020). For an anchor after that date this is the anchor's own
    # date, i.e. no change. setuptools in the environment itself is then re-installed at the anchor's date, so
    # nothing the tests can import is newer (second amendment, 9 Nov 2026).
    floor = max(pd.Timestamp(date).tz_convert("UTC"), pd.Timestamp(MIN_BUILD_SETUPTOOLS_DATE, tz="UTC")).isoformat()
    build = [f"--exclude-newer-package={p}={floor}" for p in ("setuptools", "wheel")]
    r = sh(["uv", "pip", "install", "-q", *compile_flag, "--python", vpy, "--exclude-newer", date, *build, ".",
            *deps, *pinned], cwd=wt)
    if r.returncode:
        return False, f"{source}: {r.stderr.strip()[-400:]}"
    r = sh(["uv", "pip", "install", "-q", *compile_flag, "--python", vpy, "--exclude-newer", date,
            "--reinstall-package", "setuptools", "setuptools"], cwd=wt)
    if r.returncode:
        return False, f"dated setuptools: {r.stderr.strip()[-400:]}"
    if py == "3.6":
        ok, msg = develop_link(wt, vpy)  # no setuptools that runs on 3.6 can build a PEP 660 editable install
        if not ok:
            return False, f"editable install: {msg}"
    else:
        newer = [f"--exclude-newer-package={p}={BUILD_TOOLS_DATE}" for p in ("setuptools", "wheel")]  # editable step
        r = sh(["uv", "pip", "install", "-q", *compile_flag, "--python", vpy, "--exclude-newer", date, *newer,
                "--no-deps", "--reinstall-package", PROJECT, "-e", "."], cwd=wt)
        if r.returncode:
            return False, f"editable install: {r.stderr.strip()[-400:]}"
    if not compile_flag:
        site = sh([vpy, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"]).stdout.strip()
        sh([vpy, "-m", "compileall", "-q", site])
    # measurement tool, undated (pre-registered): the newest coverage.py supporting this Python, upgrading any older
    # copy the dated test requirements brought in (pytest-cov pulls coverage 4.x into 2017-2019 environments)
    r = sh(["uv", "pip", "install", "-q", "--python", vpy, "--upgrade-package", "coverage", "coverage"], cwd=wt)
    if r.returncode:
        return False, f"coverage: {r.stderr.strip()[-300:]}"
    return True, f"python {py}; {source}"


def pytest(venv: str, wt: str, args: list[str], results: str, coverage_file: str | None = None) -> tuple[int, dict]:
    env = os.environ | {"PYTHONPATH": TOOLS, "RESULTS": results, "PYTHONDONTWRITEBYTECODE": "1"}
    vpy = os.path.join(venv, "bin", "python")
    base = ["-m", "pytest", "-p", "results_plugin", "-q", "-p", "no:cacheprovider", *args]
    if coverage_file:
        env |= {"COVERAGE_CONTEXTS": "1", "COVERAGE_FILE": coverage_file}
        cmd = [vpy, "-m", "coverage", "run", f"--include={wt}/*", *base]
    else:
        cmd = [vpy, *base]
    if os.path.exists(results):
        os.remove(results)
    r = sh(cmd, cwd=wt, env=env)
    res = json.load(open(results)) if os.path.exists(results) else {}
    return r.returncode, res


def _nodes(item: tuple[str, str | None, str]) -> list[str]:
    f, c, n = item
    return [f"{f}::{c}::{n}" if c else f"{f}::{n}"]


def _files_at(commit: str, files: list[str]) -> dict[str, str | None]:
    out = {}
    for f in files:
        try:
            out[f] = git("show", f"{commit}:{f}")
        except subprocess.CalledProcessError:
            out[f] = None
    return out


def screen_one(fx: dict) -> dict:
    fix = fx["fix_commit_hash"]
    anchor = git("rev-list", "--parents", "-n", "1", fix).split()[1]
    date = git("show", "-s", "--format=%cI", anchor).strip()
    rec = {"fix": fix, "anchor": anchor, "anchor_date": date, "python": python_for(date)}
    wt, venv = os.path.join(WORK, f"wt-{fix[:10]}"), os.path.join(WORK, f"venv-{fix[:10]}")
    tmp = os.path.join(WORK, f"res-{fix[:10]}")
    for d in (wt, venv, tmp):
        if os.path.exists(d):
            shutil.rmtree(d) if d != wt else sh(["git", "-C", REPO, "worktree", "remove", "--force", wt])
    os.makedirs(tmp)
    sh(["git", "-C", REPO, "worktree", "add", "-q", "--detach", wt, anchor])
    keep = False
    try:
        ok, why = build_env(wt, date, venv)
        rec["environment"] = why
        if not ok:
            return rec | {"excluded": "build_failure"}
        rc, full = pytest(venv, wt, ["tests"], os.path.join(tmp, "full.json"))
        bad = sorted(n for n, r in full.items() if r["outcome"] in ("failed", "error"))
        if rc != 0 or bad or not full:
            return rec | {"excluded": "suite_fails_at_parent", "returncode": rc, "failing": bad[:20], "collected": len(full)}
        durations = {n: r["call"] for n, r in full.items()}
        changed = [f for f in git("diff", "--name-only", "--diff-filter=AM", anchor, fix, "--", "tests/").split()
                   if f.endswith(".py")]
        try:
            added = added_tests(_files_at(anchor, changed), {f: s for f, s in _files_at(fix, changed).items() if s})
        except SyntaxError:
            added = None
        rec["tests_added"] = None if added is None else [list(a) for a in added]
        if not added or len(added) != 1:
            return rec | {"excluded": "not_exactly_one_test_added"}
        item, nodes = added[0], _nodes(added[0])
        sh(["git", "-C", wt, "checkout", "-q", fix, "--", "."])
        _, at_fix = pytest(venv, wt, nodes, os.path.join(tmp, "fix.json"))
        sh(["git", "-C", wt, "checkout", "-q", "--force", anchor])
        sh(["git", "-C", wt, "clean", "-fdq", "-e", "*.egg-info"])  # keep the editable install's metadata
        if run_status(at_fix, nodes) != "pass":
            return rec | {"excluded": "added_test_fails_at_fix"}
        f, cls, name = item
        new = backport_source(_files_at(anchor, [f])[f], _files_at(fix, [f])[f], cls, name)
        if new is None:
            return rec | {"excluded": "backport_error_or_not_applicable", "detail": "cannot be applied"}
        path = os.path.join(wt, f)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(new)
        runs = [pytest(venv, wt, nodes, os.path.join(tmp, f"bp{i}.json"))[1] for i in range(3)]
        status = [run_status(r, nodes) for r in runs]
        rec["backport_runs"] = status
        if "error" in status:
            return rec | {"excluded": "backport_error_or_not_applicable", "detail": "error"}
        if all(s == "pass" for s in status):
            return rec | {"excluded": "backport_passes_all_three"}
        cases = sorted({n for r in runs for n in r})
        per_case = {n: [r.get(n, {}).get("outcome") for r in runs] for n in cases}
        if any(len(set(v)) > 1 for v in per_case.values()):
            return rec | {"excluded": "flaky", "per_case": per_case}
        failing = [n for n, v in per_case.items() if v == ["failed"] * 3]
        keep = True
        return rec | {"reproduced": True, "test": list(item), "failing_cases": failing,
                      "case_durations": {n: statistics.median(r[n]["call"] for r in runs) for n in cases},
                      "durations": durations}
    finally:
        if not keep:
            sh(["git", "-C", REPO, "worktree", "remove", "--force", wt])
            shutil.rmtree(venv, ignore_errors=True)


# ------------------------------------------------------------------ the validation model
def validation_model():
    fc = json.load(open("runs/first_check.json"))
    rec = fc["validation_model"]["record"]
    mods = pd.read_csv("data/mined/flask/modules.csv", parse_dates=["snapshot"])
    days = set(fc["training_snapshots"])
    train = mods[mods["snapshot"].dt.strftime("%Y-%m-%d").isin(days)].copy()
    snaps = sorted(train["snapshot"].unique())
    train["when"] = train["snapshot"]
    train["id"] = train["snapshot"].dt.strftime("%Y-%m-%d") + ":" + train["module"]
    cut = snaps[int(len(snaps) * 0.8)]
    base, calib = train[train["snapshot"] < cut], train[train["snapshot"] >= cut]
    t = rec["tuning"]
    tuned = Tuned(family=rec["family"], imbalance=rec["imbalance"], params=rec["params"], cv_aucpr=t["cv_aucpr"],
                  cv_recall_lines=t["cv_recall_lines"], trials=t["trials"], weight_vs_smote=t["weight_vs_smote"],
                  budget={"seconds": 60}, folds="time")
    mv = fit_final(base, calib, FEATURES, tuned, seed=rec["seed"])
    check = {"mcc_threshold": [mv.threshold, rec["mcc_threshold"]],
             "platt_slope": [mv.calibrator.platt_slope, rec["calibration"]["platt_slope"]]}
    if any(abs(a - b) > 1e-9 for a, b in check.values()):
        raise SystemExit(f"the rebuilt validation model differs from the recorded one: {check}")
    return mv, check


# ------------------------------------------------------------------ phases
def phase_screen(limit: int | None, python: str | None = None) -> None:
    cfg = load_config()
    sep = pd.Timestamp(json.load(open("runs/first_check_date.json"))["separation_date"])
    fixes = [f for f in json.load(open("data/mined/flask/bugfixes.json")) if pd.Timestamp(f["fix_date"]) >= sep]
    fixes.sort(key=lambda f: pd.Timestamp(f["fix_date"]))
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(OUT, "screen.jsonl")
    done = {json.loads(line)["fix"] for line in open(path)} if os.path.exists(path) else set()
    todo = [f for f in fixes if f["fix_commit_hash"] not in done]
    if python:  # one interpreter per container (the 9 Nov amendment): only the candidates that need it
        todo = [f for f in todo if python_for(git("show", "-s", "--format=%cI", f["fix_commit_hash"] + "^").strip()) == python]
    todo = todo[: limit or None]
    print(f"{len(fixes)} candidates on or after {sep.date()}; {len(done)} screened; {len(todo)} to go "
          f"(config {cfg['repository']})", flush=True)
    for fx in todo:
        r = screen_one(fx)
        with open(path, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(fx["fix_commit_hash"][:8], r["anchor_date"][:10], r["python"], r.get("excluded", "REPRODUCED"), flush=True)


def phase_rank() -> None:
    cfg = load_config()
    screened = [json.loads(line) for line in open(os.path.join(OUT, "screen.jsonl"))]
    mv, model_check = validation_model()
    keywords = cfg["bug_fix_rule"]["keywords"]
    months = cfg["module_snapshots"]["process_period_months"]
    results = []
    for rec in [r for r in screened if r.get("reproduced")]:
        fix, anchor = rec["fix"], rec["anchor"]
        wt, venv = os.path.join(WORK, f"wt-{fix[:10]}"), os.path.join(WORK, f"venv-{fix[:10]}")
        tmp = os.path.join(WORK, f"res-{fix[:10]}")
        covfile, mapfile = os.path.join(tmp, ".coverage"), os.path.join(tmp, "map.json")
        rc, _ = pytest(venv, wt, ["tests"], os.path.join(tmp, "cov.json"), coverage_file=covfile)
        m = sh([os.path.join(venv, "bin", "python"), os.path.join(TOOLS, "map_from_coverage.py"), covfile, wt, mapfile,
                ",".join(cfg["source_modules"]), ",".join(cfg["exclude_paths"])])
        if m.returncode or not os.path.exists(mapfile):
            raise RuntimeError(f"{fix[:8]}: coverage map not built (pytest exit {rc}, coverage file "
                               f"{'present' if os.path.exists(covfile) else 'missing'}): {(m.stderr or '')[-600:]}")
        coverage = {t: set(m) for t, m in json.load(open(mapfile)).items()}
        modules = [p for p in git("ls-tree", "-r", "--name-only", anchor).splitlines() if is_source(p, cfg)]
        feats = module_features(REPO, anchor, modules, months, keywords).dropna(subset=["size"])
        p, raw = mv.score(feats)
        frame = pd.DataFrame({"id": feats["id"].to_numpy(), "p": p, "raw": raw, "size": feats["size"].to_numpy()})
        when = pd.Timestamp(rec["anchor_date"]).tz_convert("UTC") + pd.Timedelta(seconds=1)  # up to and including
        last = last_change(history(anchor), frame["id"].tolist(), when)  # the anchor's own history
        rankings = {
            "system": rank(frame, "default")["id"].tolist(),
            "largest_first": frame.sort_values(["size", "id"], ascending=[False, True], kind="mergesort")["id"].tolist(),
            "recency": sorted(frame["id"], key=lambda m: (last.get(m) is None,
                                                          -(last[m].value if last.get(m) is not None else 0), m)),
        }
        durations = rec["durations"] | rec["case_durations"]
        current = sorted(set(rec["durations"]) | set(rec["case_durations"]))
        scoreable = set(frame["id"])
        row = {"fix": fix, "anchor": anchor, "anchor_date": rec["anchor_date"], "test": rec["test"],
               "failing_cases": rec["failing_cases"], "modules_ranked": len(frame), "tests": len(current),
               "coverage_records": len(coverage)}
        for name, order in rankings.items():
            row[name] = ordering_result(order, coverage, current, durations, rec["failing_cases"])
        first = row["system"]["failing_case"]
        row["ranked_modules_covered_by_failing_case"] = len(coverage.get(first, set()) & scoreable)
        row["tests_preceding_under_any_ranking"] = must_precede(coverage, scoreable, first)
        row["random"] = {"tf": (len(current) + 1) / 2, "apfd": 0.5, "apfdc": 0.5}
        results.append(row)
        print(fix[:8], {k: round(row[k]["apfd"], 3) for k in ("system", "largest_first", "recency")}, flush=True)

    out = summarise(screened, results, model_check, cfg)
    with open(OUT + ".json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("candidates", "reproduced", "excluded")}, indent=1))


def summarise(screened: list[dict], results: list[dict], model_check: dict, cfg: dict) -> dict:
    counts = {c: sum(1 for r in screened if r.get("excluded") == c) for c in EXCLUSIONS}
    by_year = {}
    for r in screened:
        y = r["anchor_date"][:4]
        by_year.setdefault(y, {"candidates": 0, "build_failure": 0, "reproduced": 0})
        by_year[y]["candidates"] += 1
        by_year[y]["build_failure"] += r.get("excluded") == "build_failure"
        by_year[y]["reproduced"] += bool(r.get("reproduced"))
    out = {"created": datetime.now(timezone.utc).isoformat(), "pre_registration": cfg["second_check"],
           "validation_model_check": model_check, "candidates": len(screened), "reproduced": len(results),
           "excluded": counts, "by_anchor_year": by_year, "per_defect": results,
           "aggregate": None if len(results) < 12 else {
               k: {m: float(np.mean([r[k][m] for r in results])) for m in ("tf", "apfd", "apfdc")}
               for k in ("system", "largest_first", "recency", "random")},
           "reporting": ("fewer than 12 reproduced defects: each reported individually, no mean, no comparative conclusion"
                         if len(results) < 12 else "12 or more reproduced defects: means over defects reported")}
    return out


def phase_combine(dirs: list[str]) -> None:
    """Merge screens and rankings run in different environments (the 9 Nov amendment's containers): for a fix
    screened more than once, the record that is not a build failure wins; the summary is recomputed."""
    cfg = load_config()
    screened: dict[str, dict] = {}
    results: dict[str, dict] = {}
    checks = []
    for d in dirs:
        sf = os.path.join(d, "second_check", "screen.jsonl")
        if not os.path.exists(sf):  # a container that produced nothing contributes nothing
            print(f"no screen in {d}", flush=True)
            continue
        for line in open(sf):
            r = json.loads(line)
            if r["fix"] not in screened or screened[r["fix"]].get("excluded") == "build_failure":
                screened[r["fix"]] = r
        f = os.path.join(d, "second_check.json")
        if os.path.exists(f):
            prev = json.load(open(f))
            checks.append(prev["validation_model_check"])
            for row in prev["per_defect"]:
                results[row["fix"]] = row
    kept = [results[f] for f in screened if f in results and screened[f].get("reproduced")]
    out = summarise(list(screened.values()), sorted(kept, key=lambda r: r["anchor_date"]), checks[0] if checks else {}, cfg)
    out["validation_model_checks"] = checks
    out["combined_from"] = dirs
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "screen.jsonl"), "w") as fh:
        for r in sorted(screened.values(), key=lambda r: r["anchor_date"]):
            fh.write(json.dumps(r) + "\n")
    with open(OUT + ".json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("candidates", "reproduced", "excluded", "reporting")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["screen", "rank", "combine"])
    ap.add_argument("--limit", type=int, default=None, help="screen at most N more candidates")
    ap.add_argument("--python", help="screen only the candidates that need this Python version (one per container)")
    ap.add_argument("--inputs", nargs="+", default=[], help="combine: folders each holding second_check/ and second_check.json")
    a = ap.parse_args()
    if a.phase == "screen":
        phase_screen(a.limit, a.python)
    elif a.phase == "rank":
        phase_rank()
    else:
        phase_combine(a.inputs)


if __name__ == "__main__":
    main()
